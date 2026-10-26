// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 validation and traversal.

#ifndef DORA_UTF8_UTF8_H
#define DORA_UTF8_UTF8_H

#include "dora/vector/types.h"
#include "types/types_internal.h"
#include "utf8/input.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline bool utf8_byte_is_continuation(uint8_t byte) {
    return (byte & 0b1100'0000U) == 0b1000'0000U;
}

//------------------------------------------------------------------------------
//
// Validation
//
//------------------------------------------------------------------------------

int dora_utf8_validate_span(struct dora_bytes bytes, bool* out_all_ascii);

enum dora_utf8_prefix_stop : uint8_t {
    DORA_UTF8_PREFIX_END,
    DORA_UTF8_PREFIX_LIMIT,
    DORA_UTF8_PREFIX_INVALID
};

enum dora_utf8_bound : uint8_t {
    DORA_UTF8_BOUND_LOWER,
    DORA_UTF8_BOUND_UPPER
};

// Progress counts describe the complete prefix on success, ENOBUFS or EILSEQ.
int dora_utf8_decode(struct dora_bytes bytes,
                     uint32_t* out_scalars,
                     size_t scalar_capacity,
                     size_t* out_input_byte_count,
                     size_t* out_scalar_count);
int dora_utf8_encode(const uint32_t* scalars,
                     size_t scalar_count,
                     struct dora_mutable_bytes output,
                     size_t* out_input_scalar_count,
                     size_t* out_byte_count);
// Preserves outputs on error.
int dora_utf8_decode_utf16_scalar(const uint16_t* units,
                                  size_t utf16_unit_count,
                                  uint32_t* out_scalar,
                                  size_t* out_utf16_unit_count);
// Source/output may overlap; a nonempty output is NUL-terminated on success.
int dora_utf8_copy_cstr_prefix(const char* source,
                               struct dora_mutable_bytes output,
                               size_t* out_byte_count,
                               enum dora_utf8_prefix_stop* out_stop);
int dora_utf8_bom_prefix(struct dora_bytes bytes, bool input_complete, size_t* out_byte_count);
// Bound results and destination remain unchanged on error; unavailable bounds write no bytes.
int dora_utf8_measure_bound(struct dora_utf8_input input,
                            enum dora_utf8_bound bound,
                            size_t budget,
                            size_t* out_byte_count,
                            bool* out_available,
                            bool* out_exact);
int dora_utf8_make_bound(struct dora_utf8_input input,
                         enum dora_utf8_bound bound,
                         struct dora_mutable_bytes output,
                         size_t* out_byte_count,
                         bool* out_available,
                         bool* out_exact);

enum dora_utf8_write_policy : uint8_t {
    DORA_UTF8_ATOMIC,
    DORA_UTF8_UNCOMMITTED
};

enum dora_utf8_matcher_kind : uint8_t {
    DORA_UTF8_MATCHER_EMPTY,
    DORA_UTF8_MATCHER_FULL,
    DORA_UTF8_MATCHER_SMALL,
    DORA_UTF8_MATCHER_JSON,
    DORA_UTF8_MATCHER_BITS
};

struct dora_utf8_byte_matcher {
    uint64_t words[4];
    uint8_t small_bytes[5];
    uint8_t small_byte_count;
    enum dora_utf8_matcher_kind kind;
};

int dora_utf8_count(struct dora_utf8_input input, size_t* out_scalar_count);
int dora_utf8_locate(struct dora_utf8_input input,
                     size_t start_scalar_index,
                     size_t scalar_count,
                     struct dora_range* out_byte_range);
int dora_utf8_locate_tail(uint64_t scalar_distance,
                          struct dora_utf8_input input,
                          size_t scalar_count,
                          struct dora_range* out_byte_range);
int dora_utf8_find(struct dora_utf8_input input,
                   struct dora_utf8_input needle,
                   size_t start_byte_offset,
                   size_t* out_byte_offset,
                   bool* out_found);
int dora_utf8_byte_matcher_init(const uint64_t set[4], struct dora_utf8_byte_matcher* out_matcher);
int dora_utf8_find_byte_set_general(struct dora_bytes bytes,
                                    const struct dora_utf8_byte_matcher* matcher,
                                    size_t start_byte_offset,
                                    size_t* out_byte_offset);

static inline int utf8_find_byte_set(struct dora_bytes bytes,
                                     const struct dora_utf8_byte_matcher* matcher,
                                     size_t start_byte_offset,
                                     size_t* out_byte_offset) {
    if (matcher == nullptr || matcher->kind != DORA_UTF8_MATCHER_JSON) {
        return dora_utf8_find_byte_set_general(bytes, matcher, start_byte_offset, out_byte_offset);
    }
    if (!bytes_are_well_formed(bytes) || out_byte_offset == nullptr || start_byte_offset > bytes.byte_count) {
        return EINVAL;
    }
    size_t position = start_byte_offset;
    while (position < bytes.byte_count) {
        const uint8_t byte = bytes.data[position];
        if (byte < 32 || byte == '"' || byte == '\\') {
            break;
        }
        ++position;
    }
    *out_byte_offset = position;
    return 0;
}

static inline bool utf8_output_is_valid(struct dora_mutable_bytes output) {
    return output.byte_count == 0 || output.data != nullptr;
}

static inline bool utf8_input_is_valid(struct dora_utf8_input input) {
    return bytes_are_well_formed(input.bytes) && input.proof <= DORA_UTF8_VALIDATED_ASCII;
}

static inline int utf8_check_input(struct dora_utf8_input input) {
    return input.proof == DORA_UTF8_CHECKED ? dora_utf8_validate_span(input.bytes, nullptr) : 0;
}

// Admitted disjoint spans; input validity remains the caller's proof.
static inline void utf8_copy_validated_bytes(struct dora_bytes input, uint8_t* output) {
    if (input.byte_count != 0) {
        memcpy(output, input.data, input.byte_count);
    }
}

static inline int utf8_copy(struct dora_utf8_input input,
                            enum dora_utf8_write_policy policy,
                            struct dora_mutable_bytes output,
                            size_t* out_written) {
    if (!utf8_input_is_valid(input) || !utf8_output_is_valid(output) || out_written == nullptr ||
        (policy != DORA_UTF8_ATOMIC && policy != DORA_UTF8_UNCOMMITTED)) {
        return EINVAL;
    }
    if (policy == DORA_UTF8_UNCOMMITTED && input.bytes.byte_count > output.byte_count) {
        return ENOBUFS;
    }
    const int status = utf8_check_input(input);
    if (status != 0) {
        return status;
    }
    if (input.bytes.byte_count > output.byte_count) {
        return ENOBUFS;
    }
    utf8_copy_validated_bytes(input.bytes, output.data);
    *out_written = input.bytes.byte_count;
    return 0;
}

int dora_utf8_move(struct dora_utf8_input input, struct dora_mutable_bytes output, size_t* out_written);

enum dora_utf8_emit_mode : uint8_t {
    DORA_UTF8_MEASURE,
    DORA_UTF8_EMIT
};

struct dora_utf8_trim_set {
    const uint32_t* scalars;
    size_t scalar_count;
};

int dora_utf8_trim_range(struct dora_utf8_input input,
                         const struct dora_utf8_trim_set* character_set,
                         struct dora_range* out_range);
int dora_utf8_first_scalar(struct dora_utf8_input input, int32_t* out_scalar);

// Input is valid and stable; EOF is not an admissible starting offset.
static inline int utf8_advance(struct dora_bytes validated, size_t byte_offset, size_t* out_byte_offset) {
    if (out_byte_offset == nullptr || !bytes_are_well_formed(validated) || byte_offset >= validated.byte_count ||
        utf8_byte_is_continuation(validated.data[byte_offset])) {
        return EINVAL;
    }
    size_t next = byte_offset + 1;
    while (next < validated.byte_count && utf8_byte_is_continuation(validated.data[next])) {
        ++next;
    }
    *out_byte_offset = next;
    return 0;
}

// Requires well-formed, validated UTF-8 spans.
static inline bool utf8_has_affix_validated(struct dora_bytes input, struct dora_bytes affix, bool suffix) {
    return affix.byte_count <= input.byte_count &&
           (affix.byte_count == 0 ||
            memcmp(input.data + (suffix ? input.byte_count - affix.byte_count : 0), affix.data, affix.byte_count) == 0);
}

static inline int
utf8_has_affix(struct dora_utf8_input input, struct dora_utf8_input affix, bool suffix, bool* out_match) {
    if (!utf8_input_is_valid(input) || !utf8_input_is_valid(affix) || out_match == nullptr) {
        return EINVAL;
    }
    int status = utf8_check_input(input);
    if (status == 0) {
        status = utf8_check_input(affix);
    }
    if (status != 0) {
        return status;
    }
    *out_match = utf8_has_affix_validated(input.bytes, affix.bytes, suffix);
    return 0;
}

// ASCII folding leaves non-ASCII bytes unchanged; no validity proof is established.
static inline bool utf8_bytes_are_equal(struct dora_bytes left, struct dora_bytes right, bool ascii_case_insensitive) {
    if (left.byte_count != right.byte_count) {
        return false;
    }
    if (!ascii_case_insensitive) {
        return bytes_are_equal(left, right);
    }
    for (size_t i = 0; i < left.byte_count; ++i) {
        uint8_t a = left.data[i], b = right.data[i];
        if (a >= 'A' && a <= 'Z') {
            a += 'a' - 'A';
        }
        if (b >= 'A' && b <= 'Z') {
            b += 'a' - 'A';
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

// Allows overlap; establishes no UTF-8 validity proof.
static inline int utf8_move_bytes(struct dora_bytes input, struct dora_mutable_bytes output, size_t* out_written) {
    if (!bytes_are_well_formed(input) || !utf8_output_is_valid(output) || out_written == nullptr) {
        return EINVAL;
    }
    if (input.byte_count > output.byte_count) {
        return ENOBUFS;
    }
    if (input.byte_count != 0) {
        memmove(output.data, input.data, input.byte_count);
    }
    *out_written = input.byte_count;
    return 0;
}

static inline bool utf8_scalar_is_ascii(uint32_t scalar) {
    return scalar < 0b1000'0000;
}

static inline int utf8_scalar_width(uint32_t scalar, size_t* out_width) {
    if (scalar > 0x10'FFFF || (scalar >= 0xd800 && scalar <= 0xdfff)) {
        return EILSEQ;
    }
    *out_width = scalar < 0b1000'0000 ? 1 : scalar < 0x800 ? 2 : scalar < 0x1'0000 ? 3 : 4;
    return 0;
}

static inline int utf8_scalar_bytes(uint32_t scalar, uint8_t bytes[static 4], size_t* out_size) {
    if (scalar < 0b1000'0000) {
        bytes[0] = (uint8_t)scalar;
        *out_size = 1;
    } else if (scalar < 0x800) {
        bytes[0] = (uint8_t)(0b1100'0000U | (scalar >> 6U));
        bytes[1] = (uint8_t)(0b1000'0000U | (scalar & 0b0011'1111U));
        *out_size = 2;
    } else if (scalar < 0x1'0000) {
        if (scalar >= 0xD800 && scalar <= 0xDFFF) {
            return EILSEQ;
        }
        bytes[0] = (uint8_t)(0b1110'0000U | (scalar >> 12U));
        bytes[1] = (uint8_t)(0b1000'0000U | ((scalar >> 6U) & 0b0011'1111U));
        bytes[2] = (uint8_t)(0b1000'0000U | (scalar & 0b0011'1111U));
        *out_size = 3;
    } else if (scalar <= 0x10'FFFF) {
        bytes[0] = (uint8_t)(0b1111'0000U | (scalar >> 18U));
        bytes[1] = (uint8_t)(0b1000'0000U | ((scalar >> 12U) & 0b0011'1111U));
        bytes[2] = (uint8_t)(0b1000'0000U | ((scalar >> 6U) & 0b0011'1111U));
        bytes[3] = (uint8_t)(0b1000'0000U | (scalar & 0b0011'1111U));
        *out_size = 4;
    } else {
        return EILSEQ;
    }
    return 0;
}

static inline int utf8_encode_scalar(uint32_t scalar, struct dora_mutable_bytes output, size_t* out_byte_count) {
    if (!utf8_output_is_valid(output) || out_byte_count == nullptr) {
        return EINVAL;
    }
    if (scalar > 0x10'FFFF || (scalar >= 0xd800 && scalar <= 0xdfff)) {
        return EILSEQ;
    }
    const size_t size = scalar < 0b1000'0000 ? 1 : scalar < 0x800 ? 2 : scalar < 0x1'0000 ? 3 : 4;
    if (size > output.byte_count) {
        return ENOBUFS;
    }
    return utf8_scalar_bytes(scalar, output.data, out_byte_count);
}

// Requires admitted spans and start_byte_offset <= input.byte_count.
size_t dora_utf8_find_valid(struct dora_bytes input, struct dora_bytes needle, size_t start_byte_offset);

#endif
