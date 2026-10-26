// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "types/array_count.h"
#include "utf8/ascii.h"
#include "utf8/case_mapping.inc"
#include "utf8/input.h"
#include "utf8/transform.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t case_delta(const uint32_t* table, const uint16_t* pages, size_t page_count, uint32_t scalar) {
    if (scalar < 0x800) {
        return table[scalar];
    }
    return (scalar >> 8U) < page_count ? table[pages[scalar >> 8U] + (scalar & 255U)] : 0;
}

uint32_t dora_utf8_case_map_scalar(uint32_t scalar, enum dora_utf8_case mapping) {
    const uint32_t delta =
        mapping == DORA_UTF8_UPPER
            ? case_delta(UTF8_CASE_UPPER, UTF8_CASE_UPPER_PAGES, ARRAY_COUNT(UTF8_CASE_UPPER_PAGES), scalar)
            : case_delta(UTF8_CASE_LOWER, UTF8_CASE_LOWER_PAGES, ARRAY_COUNT(UTF8_CASE_LOWER_PAGES), scalar);
    if (delta == 0) {
        return scalar;
    }
    uint32_t packed = scalar;
    if (scalar >= 0x10000) {
        packed = (0xf0U | (scalar >> 18U)) | ((0x80U | ((scalar >> 12U) & 0x3fU)) << 8U) |
                 ((0x80U | ((scalar >> 6U) & 0x3fU)) << 16U) | ((0x80U | (scalar & 0x3fU)) << 24U);
    } else if (scalar >= 0x800) {
        packed = (0xe0U | (scalar >> 12U)) | ((0x80U | ((scalar >> 6U) & 0x3fU)) << 8U) |
                 ((0x80U | (scalar & 0x3fU)) << 16U);
    } else if (scalar >= 0x80) {
        packed = (0xc0U | (scalar >> 6U)) | ((0x80U | (scalar & 0x3fU)) << 8U);
    }
    packed ^= delta;
    const uint32_t lead = packed & 0xffU;
    if (lead >= 0xf0) {
        return ((lead & 7U) << 18U) | (((packed >> 8U) & 0x3fU) << 12U) | (((packed >> 16U) & 0x3fU) << 6U) |
               ((packed >> 24U) & 0x3fU);
    }
    if (lead >= 0xe0) {
        return ((lead & 0x0fU) << 12U) | (((packed >> 8U) & 0x3fU) << 6U) | ((packed >> 16U) & 0x3fU);
    }
    if (lead >= 0xc0) {
        return ((lead & 0x1fU) << 6U) | ((packed >> 8U) & 0x3fU);
    }
    return lead;
}

int dora_utf8_case_map(struct dora_utf8_input input,
                       enum dora_utf8_case mapping,
                       enum dora_utf8_emit_mode mode,
                       struct dora_mutable_bytes output,
                       size_t* out_byte_count) {
    if (!utf8_input_is_valid(input) || out_byte_count == nullptr || mapping > DORA_UTF8_UPPER ||
        mode > DORA_UTF8_EMIT || (mode == DORA_UTF8_EMIT && !utf8_output_is_valid(output))) {
        return EINVAL;
    }
    const struct dora_bytes source = input.bytes;
    const bool write = mode == DORA_UTF8_EMIT;
    if (input.proof == DORA_UTF8_VALIDATED_ASCII || utf8_span_is_ascii(source.data, source.byte_count)) {
        if (write && source.byte_count > output.byte_count) {
            return ENOBUFS;
        }
        if (write) {
            const uint8_t lower = mapping == DORA_UTF8_UPPER ? 'a' : 'A';
            const uint8_t upper = mapping == DORA_UTF8_UPPER ? 'z' : 'Z';
            for (size_t i = 0; i < source.byte_count; ++i) {
                const uint8_t byte = source.data[i];
                output.data[i] = byte >= lower && byte <= upper ? (uint8_t)(byte ^ 0x20U) : byte;
            }
        }
        *out_byte_count = source.byte_count;
        return DORA_OK;
    }

    const uint32_t* const table = mapping == DORA_UTF8_UPPER ? UTF8_CASE_UPPER : UTF8_CASE_LOWER;
    const uint16_t* const pages = mapping == DORA_UTF8_UPPER ? UTF8_CASE_UPPER_PAGES : UTF8_CASE_LOWER_PAGES;
    const size_t page_count =
        mapping == DORA_UTF8_UPPER ? ARRAY_COUNT(UTF8_CASE_UPPER_PAGES) : ARRAY_COUNT(UTF8_CASE_LOWER_PAGES);
    size_t input_offset = 0, output_offset = 0;
    while (input_offset < source.byte_count) {
        // Partial chunks must not alter malformed-input or capacity failure boundaries.
        uint8_t chunk[256 * 4 + 3];
        size_t chunk_bytes = 0;
        for (size_t count = 0; count < 256 && input_offset < source.byte_count; ++count) {
            const uint8_t* const p = source.data + input_offset;
            const size_t remaining = source.byte_count - input_offset;
            uint32_t scalar = p[0], packed = p[0];
            size_t width = 1;
            if (scalar >= 0x80) {
                if (scalar < 0xc2 || scalar > 0xf4 || remaining < 2 || (p[1] & 0xc0U) != 0x80) {
                    return EILSEQ;
                }
                packed |= (uint32_t)p[1] << 8U;
                if (scalar < 0xe0) {
                    scalar = ((scalar & 0x1fU) << 6U) | (p[1] & 0x3fU);
                    width = 2;
                } else {
                    if (remaining < 3 || (p[2] & 0xc0U) != 0x80) {
                        return EILSEQ;
                    }
                    packed |= (uint32_t)p[2] << 16U;
                    if (scalar < 0xf0) {
                        scalar = ((scalar & 0x0fU) << 12U) | ((p[1] & 0x3fU) << 6U) | (p[2] & 0x3fU);
                        if (scalar < 0x800 || (scalar >= 0xd800 && scalar <= 0xdfff)) {
                            return EILSEQ;
                        }
                        width = 3;
                    } else {
                        if (remaining < 4 || (p[3] & 0xc0U) != 0x80) {
                            return EILSEQ;
                        }
                        scalar =
                            ((scalar & 7U) << 18U) | ((p[1] & 0x3fU) << 12U) | ((p[2] & 0x3fU) << 6U) | (p[3] & 0x3fU);
                        if (scalar < 0x10000 || scalar > 0x10ffff) {
                            return EILSEQ;
                        }
                        packed |= (uint32_t)p[3] << 24U;
                        width = 4;
                    }
                }
            }
            input_offset += width;
            packed ^= case_delta(table, pages, page_count, scalar);
            const uint8_t lead = (uint8_t)packed;
            const size_t mapped_width = 1U + (size_t)(lead >= 0xc0) + (size_t)(lead >= 0xe0) + (size_t)(lead >= 0xf0);
            if (write) {
                // Fixed stores require three spare bytes.
                chunk[chunk_bytes] = lead;
                chunk[chunk_bytes + 1] = (uint8_t)(packed >> 8U);
                chunk[chunk_bytes + 2] = (uint8_t)(packed >> 16U);
                chunk[chunk_bytes + 3] = (uint8_t)(packed >> 24U);
            }
            chunk_bytes += mapped_width;
        }
        size_t next;
        if (ckd_add(&next, output_offset, chunk_bytes)) {
            return EOVERFLOW;
        }
        if (write) {
            if (next > output.byte_count || output.data == nullptr) {
                return ENOBUFS;
            }
            memcpy(output.data + output_offset, chunk, chunk_bytes);
        }
        output_offset = next;
    }
    *out_byte_count = output_offset;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(ARRAY_COUNT(UTF8_CASE_LOWER) >= 0x800 && ARRAY_COUNT(UTF8_CASE_UPPER) >= 0x800);
