// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "json/json.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

int dora_json_prepare_string_events(struct dora_utf8_byte_matcher* out_matcher) {
    uint64_t set[4];
    bitmap_fill_bytes((uint8_t*)set, sizeof set, false);
    dora_bitmap_set_range(set, 0, 32, true);
    bitmap_set(set, '\"');
    bitmap_set(set, '\\');
    return dora_utf8_byte_matcher_init(set, out_matcher);
}

static uint32_t hex_digit(uint8_t byte) {
    static constexpr uint8_t digits[256] = {['0'] = 1,  ['1'] = 2,  ['2'] = 3,  ['3'] = 4,  ['4'] = 5,  ['5'] = 6,
                                            ['6'] = 7,  ['7'] = 8,  ['8'] = 9,  ['9'] = 10, ['a'] = 11, ['b'] = 12,
                                            ['c'] = 13, ['d'] = 14, ['e'] = 15, ['f'] = 16, ['A'] = 11, ['B'] = 12,
                                            ['C'] = 13, ['D'] = 14, ['E'] = 15, ['F'] = 16};
    return digits[byte] != 0 ? (uint32_t)digits[byte] - 1U : UINT32_MAX;
}

int dora_json_read_hex_unit(struct dora_bytes bytes, size_t* in_out_byte_offset, uint16_t* out_utf16_unit) {
    if (*in_out_byte_offset > bytes.byte_count || bytes.byte_count - *in_out_byte_offset < 4) {
        return EILSEQ;
    }
    uint32_t value = 0;
    for (uint32_t index = 0; index < 4; ++index) {
        const uint8_t byte = bytes.data[(*in_out_byte_offset)++];
        const uint32_t digit = hex_digit(byte);
        if (digit == UINT32_MAX) {
            return EILSEQ;
        }
        value = (value << 4U) | digit;
    }
    *out_utf16_unit = (uint16_t)value;
    return 0;
}

enum json_frame : uint8_t {
    JSON_ARRAY_FIRST,
    JSON_ARRAY_VALUE,
    JSON_ARRAY_NEXT,
    JSON_OBJECT_FIRST,
    JSON_OBJECT_KEY,
    JSON_OBJECT_COLON,
    JSON_OBJECT_VALUE,
    JSON_OBJECT_NEXT,
    JSON_RECORD_FIRST,
    JSON_RECORD_KEY,
    JSON_RECORD_COLON,
    JSON_RECORD_VALUE,
    JSON_RECORD_NEXT,
    JSON_DOCUMENT_VALUE,
    JSON_DOCUMENT_RECORD_VALUE,
    JSON_DOCUMENT_END
};

enum json_token : uint8_t {
    JSON_TOKEN_NONE,
    JSON_TOKEN_STRING,
    JSON_TOKEN_ESCAPE,
    JSON_TOKEN_UNICODE,
    JSON_TOKEN_MINUS,
    JSON_TOKEN_ZERO,
    JSON_TOKEN_INTEGER,
    JSON_TOKEN_DOT,
    JSON_TOKEN_FRACTION,
    JSON_TOKEN_EXPONENT,
    JSON_TOKEN_SIGN,
    JSON_TOKEN_EXPONENT_DIGITS,
    JSON_TOKEN_LITERAL
};

struct json_validation {
    struct dora_bytes bytes;
    uint8_t* frames;
    size_t frame_capacity;
    size_t frame_count;
    size_t byte_offset;
    enum json_token token;
    uint8_t literal_index;
    uint8_t token_byte_offset;
    uint8_t utf16_unit_count;
    uint16_t utf16_units[2];
    enum json_frame root_frame;
};

size_t dora_json_find_string_end(struct dora_bytes bytes, size_t begin_byte_offset) {
    size_t position = begin_byte_offset + 1;
    while (position < bytes.byte_count) {
        const uint8_t byte = bytes.data[position++];
        if (byte == '"') {
            return position;
        }
        if (byte == '\\') {
            if (position == bytes.byte_count) {
                return SIZE_MAX;
            }
            ++position;
        }
    }
    return SIZE_MAX;
}

static uint32_t json_value_end(const uint8_t* bytes, uint32_t position, uint32_t end) {
    if (bytes[position] == '"') {
        const size_t string_end = dora_json_find_string_end((struct dora_bytes){bytes, end}, position);
        if (string_end == SIZE_MAX) {
            return end;
        }
        return (uint32_t)string_end;
    }
    if (bytes[position] == '[' || bytes[position] == '{') {
        uint32_t depth = 1;
        ++position;
        while (position < end && depth != 0) {
            if (bytes[position] == '"') {
                const size_t string_end = dora_json_find_string_end((struct dora_bytes){bytes, end}, position);
                if (string_end == SIZE_MAX) {
                    return end;
                }
                position = (uint32_t)string_end;
                continue;
            }
            if (bytes[position] == '[' || bytes[position] == '{') {
                ++depth;
            } else if (bytes[position] == ']' || bytes[position] == '}') {
                --depth;
            }
            ++position;
        }
        return position;
    }
    while (position < end && bytes[position] != ',' && bytes[position] != ']' && bytes[position] != '}' &&
           !json_byte_is_whitespace(bytes[position])) {
        ++position;
    }
    return position;
}

static bool json_key_equals(const uint8_t* bytes, struct dora_json_span key, const char* expected) {
    size_t expected_index = 0;
    size_t position = key.begin_byte_offset + 1;
    while (position + 1 < key.end_byte_offset) {
        unsigned value = bytes[position++];
        if (value == '\\') {
            value = bytes[position++];
            if (value == 'u') {
                uint32_t scalar;
                const int status = json_unicode_escape(bytes, key.end_byte_offset, &position, &scalar);
                if (status != 0) {
                    return false;
                }
                value = scalar;
            } else if (value == 'b' || value == 'f' || value == 'n' || value == 'r' || value == 't') {
                return false;
            }
        }
        if (expected[expected_index] == '\0' || value != (unsigned char)expected[expected_index]) {
            return false;
        }
        ++expected_index;
    }
    return expected[expected_index] == '\0';
}

static bool json_byte_is_digit(uint8_t byte) {
    return byte >= '0' && byte <= '9';
}

static int json_consume_byte(struct json_validation* state, uint8_t byte, bool* out_consumed) {
    *out_consumed = true;
    switch (state->token) {
    case JSON_TOKEN_STRING:
        if (state->utf16_unit_count != 0 && byte != '\\') {
            return EINVAL;
        }
        if (byte < 0x20) {
            return EINVAL;
        }
        if (byte == '"') {
            state->token = JSON_TOKEN_NONE;
        } else if (byte == '\\') {
            state->token = JSON_TOKEN_ESCAPE;
        }
        return 0;
    case JSON_TOKEN_ESCAPE:
        if (state->utf16_unit_count != 0 && byte != 'u') {
            return EINVAL;
        }
        if (byte == 'u') {
            state->utf16_units[state->utf16_unit_count] = 0;
            state->token = JSON_TOKEN_UNICODE;
            state->token_byte_offset = 0;
            return 0;
        }
        if (byte != '"' && byte != '\\' && byte != '/' && byte != 'b' && byte != 'f' && byte != 'n' && byte != 'r' &&
            byte != 't') {
            return EINVAL;
        }
        state->token = JSON_TOKEN_STRING;
        return 0;
    case JSON_TOKEN_UNICODE: {
        const uint32_t digit = hex_digit(byte);
        if (digit == UINT32_MAX) {
            return EINVAL;
        }
        state->utf16_units[state->utf16_unit_count] =
            (uint16_t)(state->utf16_units[state->utf16_unit_count] * 16U + digit);
        if (++state->token_byte_offset == 4) {
            ++state->utf16_unit_count;
            uint32_t scalar;
            size_t consumed;
            const int status =
                dora_utf8_decode_utf16_scalar(state->utf16_units, state->utf16_unit_count, &scalar, &consumed);
            if (status != 0 && status != EAGAIN) {
                return EINVAL;
            }
            if (status == 0) {
                state->utf16_unit_count = 0;
            }
            state->token = JSON_TOKEN_STRING;
        }
        return 0;
    }
    case JSON_TOKEN_LITERAL: {
        static const char* const literals[] = {"true", "false", "null", "infinity", "nan"};
        const char* const literal = literals[state->literal_index];
        const uint8_t fold = state->literal_index >= 3 ? 0x20 : 0;
        if (state->literal_index == 3 && state->token_byte_offset == 3 && (byte | fold) != 'i') {
            break;
        }
        if ((byte | fold) != (uint8_t)literal[state->token_byte_offset]) {
            return EINVAL;
        }
        if (literal[++state->token_byte_offset] == '\0') {
            state->token = JSON_TOKEN_NONE;
        }
        return 0;
    }
    case JSON_TOKEN_MINUS:
        if (state->frame_count != 0 && state->frames[state->frame_count - 1] == JSON_RECORD_NEXT &&
            (byte == 'i' || byte == 'I' || byte == 'n' || byte == 'N')) {
            goto special_literal;
        }
        if (!json_byte_is_digit(byte)) {
            return EINVAL;
        }
        state->token = byte == '0' ? JSON_TOKEN_ZERO : JSON_TOKEN_INTEGER;
        return 0;
    case JSON_TOKEN_ZERO:
    case JSON_TOKEN_INTEGER:
    case JSON_TOKEN_DOT:
    case JSON_TOKEN_FRACTION:
    case JSON_TOKEN_EXPONENT:
    case JSON_TOKEN_SIGN:
    case JSON_TOKEN_EXPONENT_DIGITS: {
        static constexpr uint8_t transitions[7][5] = {
            {JSON_TOKEN_NONE, JSON_TOKEN_NONE, JSON_TOKEN_DOT, JSON_TOKEN_EXPONENT, JSON_TOKEN_NONE},
            {JSON_TOKEN_NONE, JSON_TOKEN_INTEGER, JSON_TOKEN_DOT, JSON_TOKEN_EXPONENT, JSON_TOKEN_NONE},
            {UINT8_MAX, JSON_TOKEN_FRACTION, UINT8_MAX, UINT8_MAX, UINT8_MAX},
            {JSON_TOKEN_NONE, JSON_TOKEN_FRACTION, JSON_TOKEN_NONE, JSON_TOKEN_EXPONENT, JSON_TOKEN_NONE},
            {UINT8_MAX, JSON_TOKEN_EXPONENT_DIGITS, UINT8_MAX, UINT8_MAX, JSON_TOKEN_SIGN},
            {UINT8_MAX, JSON_TOKEN_EXPONENT_DIGITS, UINT8_MAX, UINT8_MAX, UINT8_MAX},
            {JSON_TOKEN_NONE, JSON_TOKEN_EXPONENT_DIGITS, JSON_TOKEN_NONE, JSON_TOKEN_NONE, JSON_TOKEN_NONE}
        };
        static constexpr uint8_t classes[256] = {
            ['0'] = 1,
            ['1'] = 1,
            ['2'] = 1,
            ['3'] = 1,
            ['4'] = 1,
            ['5'] = 1,
            ['6'] = 1,
            ['7'] = 1,
            ['8'] = 1,
            ['9'] = 1,
            ['.'] = 2,
            ['e'] = 3,
            ['E'] = 3,
            ['+'] = 4,
            ['-'] = 4
        };
        const uint8_t next = transitions[state->token - JSON_TOKEN_ZERO][classes[byte]];
        if (next == UINT8_MAX) {
            return EINVAL;
        }
        state->token = next;
        *out_consumed = next != JSON_TOKEN_NONE;
        return 0;
    }
    case JSON_TOKEN_NONE:
        goto grammar;
    default:
        return EINVAL;
    }
    state->token = JSON_TOKEN_NONE;
    *out_consumed = false;
    return 0;

grammar:
    if (json_byte_is_whitespace(byte)) {
        return 0;
    }
    static constexpr uint8_t grammar[][3] = {
        {']', 0, JSON_ARRAY_NEXT},
        {0, 0, JSON_ARRAY_NEXT},
        {']', ',', JSON_ARRAY_VALUE},
        {'}', '"', JSON_OBJECT_COLON},
        {0, '"', JSON_OBJECT_COLON},
        {0, ':', JSON_OBJECT_VALUE},
        {0, 0, JSON_OBJECT_NEXT},
        {'}', ',', JSON_OBJECT_KEY},
        {'}', '"', JSON_RECORD_COLON},
        {0, '"', JSON_RECORD_COLON},
        {0, ':', JSON_RECORD_VALUE},
        {0, 0, JSON_RECORD_NEXT},
        {'}', ',', JSON_RECORD_KEY}
    };
    uint8_t delimiter = 0;
    const uint8_t frame = state->frame_count == 0 ? state->root_frame : state->frames[state->frame_count - 1];
    if (state->frame_count != 0) {
        if (grammar[frame][0] != 0 && byte == grammar[frame][0]) {
            --state->frame_count;
            return 0;
        }
        delimiter = grammar[frame][1];
        if (delimiter != 0 && byte != delimiter) {
            return EINVAL;
        }
    } else if (state->root_frame == JSON_DOCUMENT_END) {
        return EINVAL;
    }
    if (delimiter == 0 && (byte == '[' || byte == '{') && state->frame_count == state->frame_capacity) {
        return ENOBUFS;
    }
    if (state->frame_count == 0) {
        state->root_frame = JSON_DOCUMENT_END;
    } else {
        state->frames[state->frame_count - 1] = grammar[frame][2];
    }
    if (delimiter != 0) {
        if (delimiter == '"') {
            state->token = JSON_TOKEN_STRING;
        }
        return 0;
    }
    const bool record_value = frame == JSON_RECORD_VALUE;
    switch (byte) {
    case '[':
    case '{':
        state->frames[state->frame_count++] = byte == '['                           ? JSON_ARRAY_FIRST
                                              : frame == JSON_DOCUMENT_RECORD_VALUE ? JSON_RECORD_FIRST
                                                                                    : JSON_OBJECT_FIRST;
        return 0;
    case '"':
        state->token = JSON_TOKEN_STRING;
        return 0;
    case 't':
    case 'f':
    case 'n':
        state->token = JSON_TOKEN_LITERAL;
        state->literal_index = byte == 't'   ? 0
                               : byte == 'f' ? 1
                               : record_value && state->byte_offset + 1 < state->bytes.byte_count &&
                                       (state->bytes.data[state->byte_offset + 1] | 0x20U) == 'a'
                                   ? 4
                                   : 2;
        state->token_byte_offset = 1;
        return 0;
    case 'i':
    case 'I':
    case 'N':
        if (!record_value) {
            return EINVAL;
        }
        goto special_literal;
    case '-':
        state->token = JSON_TOKEN_MINUS;
        return 0;
    case '0':
        state->token = JSON_TOKEN_ZERO;
        return 0;
    default:
        if (byte < '1' || byte > '9') {
            return EINVAL;
        }
        state->token = JSON_TOKEN_INTEGER;
        return 0;
    }
special_literal:
    state->token = JSON_TOKEN_LITERAL;
    state->literal_index = (byte | 0x20U) == 'i' ? 3 : 4;
    state->token_byte_offset = 1;
    return 0;
}

static int json_finish(const struct json_validation* state) {
    if (state->frame_count != 0 || state->root_frame != JSON_DOCUMENT_END) {
        return EINVAL;
    }
    switch (state->token) {
    case JSON_TOKEN_NONE:
    case JSON_TOKEN_ZERO:
    case JSON_TOKEN_INTEGER:
    case JSON_TOKEN_FRACTION:
    case JSON_TOKEN_EXPONENT_DIGITS:
        return 0;
    case JSON_TOKEN_LITERAL:
        return state->literal_index == 3 && state->token_byte_offset == 3 ? 0 : EINVAL;
    default:
        return EINVAL;
    }
}

static int json_validation_init(struct json_validation* state,
                                struct dora_bytes bytes,
                                enum dora_utf8_proof proof,
                                uint8_t* frames,
                                size_t frame_capacity) {
    if (state == nullptr || (bytes.byte_count != 0 && bytes.data == nullptr) ||
        (frame_capacity != 0 && frames == nullptr)) {
        return EINVAL;
    }
    const int status = proof == DORA_UTF8_CHECKED ? dora_utf8_validate_span(bytes, nullptr) : 0;
    if (status != 0) {
        return status;
    }
    *state = (struct json_validation){
        .bytes = bytes,
        .frames = frames,
        .frame_capacity = frame_capacity,
        .root_frame = JSON_DOCUMENT_VALUE
    };
    return 0;
}

// ENOBUFS leaves the opening byte and its parent frame unconsumed.
static int json_validation_step(struct json_validation* state) {
    if (state == nullptr) {
        return EINVAL;
    }
    while (state->byte_offset < state->bytes.byte_count) {
        bool consumed = false;
        const int status = json_consume_byte(state, state->bytes.data[state->byte_offset], &consumed);
        if (status != 0) {
            return status;
        }
        state->byte_offset += consumed ? 1U : 0U;
    }
    return json_finish(state);
}

static int validate_frames(struct dora_pool pool,
                           struct dora_bytes bytes,
                           enum dora_utf8_proof proof,
                           struct dora_allocation* frames,
                           bool* out_invalid_content,
                           enum json_frame root) {
    uint8_t root_frame = 0;
    struct json_validation validation;
    const bool inline_frames = frames->address == nullptr;
    *out_invalid_content = false;
    int status = json_validation_init(&validation,
                                      bytes,
                                      proof,
                                      inline_frames ? &root_frame : frames->address,
                                      inline_frames ? 1 : frames->capacity_bytes);
    if (status != DORA_OK) {
        *out_invalid_content = true;
        return status;
    }
    validation.root_frame = root;
    for (;;) {
        status = json_validation_step(&validation);
        if (status != ENOBUFS) {
            *out_invalid_content = status != DORA_OK;
            return status;
        }
        size_t required = 0;
        if (ckd_add(&required, validation.frame_count, (size_t)1)) {
            return EOVERFLOW;
        }
        if (required > bytes.byte_count) {
            return EOVERFLOW;
        }
        const bool copy_root = frames->address == nullptr;
        status = dora_allocation_reserve(pool, required, copy_root ? 0 : validation.frame_count, frames);
        if (status != DORA_OK) {
            return status;
        }
        if (copy_root) {
            ((uint8_t*)frames->address)[0] = root_frame;
        }
        validation.frames = frames->address;
        validation.frame_capacity =
            frames->capacity_bytes < bytes.byte_count ? frames->capacity_bytes : bytes.byte_count;
    }
}

int dora_json_validate_frames(struct dora_pool pool,
                              struct dora_bytes bytes,
                              enum dora_utf8_proof proof,
                              struct dora_allocation* frames,
                              bool* out_invalid_content) {
    return validate_frames(pool, bytes, proof, frames, out_invalid_content, JSON_DOCUMENT_VALUE);
}

static int validate(struct dora_pool pool, struct dora_bytes bytes, enum dora_utf8_proof proof, enum json_frame root) {
    struct dora_allocation frames = {};
    bool invalid_content = false;
    const int status = validate_frames(pool, bytes, proof, &frames, &invalid_content, root);
    dora_allocation_release(&frames);
    return status;
}

int dora_json_validate(struct dora_pool pool, struct dora_bytes bytes, enum dora_utf8_proof proof) {
    return validate(pool, bytes, proof, JSON_DOCUMENT_VALUE);
}

int dora_json_validate_record(struct dora_pool pool, struct dora_bytes bytes, enum dora_utf8_proof proof) {
    return validate(pool, bytes, proof, JSON_DOCUMENT_RECORD_VALUE);
}

uint32_t dora_json_skip_whitespace(const uint8_t* bytes, uint32_t byte_offset, uint32_t end_byte_offset) {
    while (byte_offset < end_byte_offset && json_byte_is_whitespace(bytes[byte_offset])) {
        ++byte_offset;
    }
    return byte_offset;
}

int dora_json_object_members(const uint8_t* bytes,
                             uint32_t byte_count,
                             const char* const* names,
                             uint32_t name_count,
                             struct dora_json_span* values,
                             uint32_t* out_present) {
    if (name_count > 32) {
        return EINVAL;
    }
    uint32_t position = dora_json_skip_whitespace(bytes, 0, byte_count);
    if (position == byte_count || bytes[position] != '{') {
        return EINVAL;
    }
    position = dora_json_skip_whitespace(bytes, position + 1, byte_count);
    uint32_t present = 0;
    while (bytes[position] != '}') {
        const size_t key_end = dora_json_find_string_end((struct dora_bytes){bytes, byte_count}, position);
        if (key_end == SIZE_MAX) {
            return EINVAL;
        }
        const struct dora_json_span key = {.begin_byte_offset = position, .end_byte_offset = (uint32_t)key_end};
        position = dora_json_skip_whitespace(bytes, key.end_byte_offset, byte_count);
        position = dora_json_skip_whitespace(bytes, position + 1, byte_count);
        const struct dora_json_span value = {
            .begin_byte_offset = position,
            .end_byte_offset = json_value_end(bytes, position, byte_count)
        };
        for (uint32_t index = 0; index < name_count; ++index) {
            if (names[index] != nullptr && json_key_equals(bytes, key, names[index])) {
                const uint32_t bit = 1U << index;
                if ((present & bit) != 0) {
                    return EINVAL;
                }
                present |= bit;
                values[index] = value;
                break;
            }
        }
        position = dora_json_skip_whitespace(bytes, value.end_byte_offset, byte_count);
        if (bytes[position] == ',') {
            position = dora_json_skip_whitespace(bytes, position + 1, byte_count);
        }
    }
    *out_present = present;
    return 0;
}

int dora_json_parse_unsigned(const uint8_t* bytes, struct dora_json_span value, uint64_t* out_value) {
    uint32_t position = value.begin_byte_offset;
    const bool negative = bytes[position] == '-';
    if (negative) {
        ++position;
    }
    if (position == value.end_byte_offset) {
        return EINVAL;
    }
    uint64_t result = 0;
    while (position < value.end_byte_offset) {
        if (bytes[position] < '0' || bytes[position] > '9') {
            return EINVAL;
        }
        const uint64_t digit = bytes[position] - (uint8_t)'0';
        if (ckd_mul(&result, result, 10) || ckd_add(&result, result, digit)) {
            return EOVERFLOW;
        }
        ++position;
    }
    if (negative && result != 0) {
        return EINVAL;
    }
    *out_value = result;
    return 0;
}

uint32_t dora_json_array_next(const uint8_t* bytes,
                              struct dora_json_span array,
                              uint32_t byte_offset,
                              struct dora_json_span* out_value) {
    byte_offset = dora_json_skip_whitespace(bytes, byte_offset, array.end_byte_offset);
    if (bytes[byte_offset] == ']') {
        return array.end_byte_offset;
    }
    *out_value = (struct dora_json_span){
        .begin_byte_offset = byte_offset,
        .end_byte_offset = json_value_end(bytes, byte_offset, array.end_byte_offset)
    };
    byte_offset = dora_json_skip_whitespace(bytes, out_value->end_byte_offset, array.end_byte_offset);
    return bytes[byte_offset] == ',' ? byte_offset + 1 : byte_offset;
}
