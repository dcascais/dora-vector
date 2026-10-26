// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// JSON validation and token parsing.

#ifndef DORA_JSON_JSON_H
#define DORA_JSON_JSON_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

static inline bool json_byte_is_whitespace(uint8_t byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}

// Non-finite numbers require scalar record members.
int dora_json_validate_record(struct dora_pool pool, struct dora_bytes bytes, enum dora_utf8_proof proof);
int dora_json_read_hex_unit(struct dora_bytes bytes, size_t* in_out_byte_offset, uint16_t* out_utf16_unit);

static inline int
json_unicode_escape(const uint8_t* bytes, size_t byte_count, size_t* in_out_byte_offset, uint32_t* out_scalar) {
    uint16_t units[2];
    int status = dora_json_read_hex_unit((struct dora_bytes){bytes, byte_count}, in_out_byte_offset, &units[0]);
    if (status != 0) {
        return EILSEQ;
    }
    size_t consumed;
    status = dora_utf8_decode_utf16_scalar(units, 1, out_scalar, &consumed);
    if (status == EAGAIN) {
        if (byte_count - *in_out_byte_offset < 6 || bytes[*in_out_byte_offset] != '\\' ||
            bytes[*in_out_byte_offset + 1] != 'u') {
            return EILSEQ;
        }
        *in_out_byte_offset += 2;
        status = dora_json_read_hex_unit((struct dora_bytes){bytes, byte_count}, in_out_byte_offset, &units[1]);
        if (status == 0) {
            status = dora_utf8_decode_utf16_scalar(units, 2, out_scalar, &consumed);
        }
    }
    if (status != 0) {
        return EILSEQ;
    }
    return 0;
}

// Returns SIZE_MAX for an unterminated token.
size_t dora_json_find_string_end(struct dora_bytes bytes, size_t begin_byte_offset);

// Retains frames across calls; reports invalid content separately from allocation failure.
int dora_json_validate_frames(struct dora_pool pool,
                              struct dora_bytes bytes,
                              enum dora_utf8_proof proof,
                              struct dora_allocation* frames,
                              bool* out_invalid_content);
int dora_json_validate(struct dora_pool pool, struct dora_bytes bytes, enum dora_utf8_proof proof);

struct dora_json_span {
    uint32_t begin_byte_offset;
    uint32_t end_byte_offset;
};

uint32_t dora_json_skip_whitespace(const uint8_t* bytes, uint32_t byte_offset, uint32_t end_byte_offset);
int dora_json_object_members(const uint8_t* bytes,
                             uint32_t byte_count,
                             const char* const* names,
                             uint32_t name_count,
                             struct dora_json_span* values,
                             uint32_t* out_present);
int dora_json_parse_unsigned(const uint8_t* bytes, struct dora_json_span value, uint64_t* out_value);
uint32_t dora_json_array_next(const uint8_t* bytes,
                              struct dora_json_span array,
                              uint32_t byte_offset,
                              struct dora_json_span* out_value);

int dora_json_prepare_string_events(struct dora_utf8_byte_matcher* out_matcher);

#endif
