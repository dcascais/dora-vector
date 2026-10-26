// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "builder/builder_internal.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/json.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "identity/identity.h"
#include "io/inferred_columns.h"
#include "io/json/json_inference.h"
#include "io/json/json_internal.h"
#include "io/schema_inference_internal.h"
#include "io/text.h"
#include "io/text_reader.h"
#include "json/json.h"
#include "memory/memory_internal.h"
#include "utf8/input.h"
#include "utf8/utf8.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

enum json_array_phase : uint8_t {
    JSON_ARRAY_EXPECT_OPEN,
    JSON_ARRAY_EXPECT_FIRST_OR_END,
    // Rejects trailing commas.
    JSON_ARRAY_EXPECT_OBJECT,
    JSON_ARRAY_EXPECT_COMMA_OR_END,
    JSON_ARRAY_AFTER_END
};

struct json_parser {
    struct text_input text;
    struct dora_json_options options;
    struct dora_utf8_byte_matcher string_events;
    size_t depth;
    enum json_array_phase phase;
    bool is_in_string;
    bool is_escaped;
    enum dora_utf8_proof record_proof;
    bool is_decoded_text_ascii;
};

struct json_reader {
    struct dora_vector_source source;
    struct dora_allocation allocation;
    struct dora_allocation fields;
    struct dora_schema schema;
    struct dora_builder builder;
    struct dora_allocation scratch;
    struct json_parser parser;
    struct text_columns inferred;
    uint32_t column_count;
    bool is_inferred;
    struct dora_vector block;
};

struct json_field {
    const char* name;
    size_t name_byte_count;
    enum dora_physical_type physical_type;
    bool was_seen;
    bool is_json_extension;
};

struct json_schema_inference {
    struct text_schema_inference_state common;
    struct json_parser parser;
    struct text_columns columns;
    struct dora_allocation scratch;
};

static size_t json_skip_space(struct dora_bytes bytes, size_t position) {
    while (position < bytes.byte_count && json_byte_is_whitespace(bytes.data[position])) {
        ++position;
    }
    return position;
}

static int json_string(uint8_t* bytes,
                       size_t byte_count,
                       size_t* position,
                       const struct dora_utf8_byte_matcher* events,
                       struct dora_bytes* out_text,
                       bool* ascii) {
    if (*position >= byte_count || bytes[*position] != '"') {
        return EILSEQ;
    }
    uint8_t* const output = bytes + ++*position;
    size_t length = 0;
    bool displaced = false;
    while (*position < byte_count) {
        size_t event;
        int status = utf8_find_byte_set((struct dora_bytes){bytes, byte_count}, events, *position, &event);
        if (status != DORA_OK) {
            return status;
        }
        const size_t run = event - *position;
        if (displaced && run != 0) {
            size_t written;
            status = dora_utf8_move((struct dora_utf8_input){{bytes + *position, run}, DORA_UTF8_VALIDATED},
                                    (struct dora_mutable_bytes){output + length, run},
                                    &written);
            if (status != DORA_OK) {
                return status;
            }
        }
        length += run;
        *position = event;
        if (event == byte_count) {
            break;
        }
        const uint8_t byte = bytes[(*position)++];
        if (byte == '"') {
            *out_text = (struct dora_bytes){.data = output, .byte_count = length};
            return DORA_OK;
        }
        if (byte < 0x20) {
            return EILSEQ;
        }
        displaced = true;
        if (*position == byte_count) {
            return EILSEQ;
        }
        const uint8_t escape = bytes[(*position)++];
        uint32_t codepoint;
        switch (escape) {
        case '"':
        case '\\':
        case '/':
            codepoint = escape;
            break;
        case 'b':
            codepoint = '\b';
            break;
        case 'f':
            codepoint = '\f';
            break;
        case 'n':
            codepoint = '\n';
            break;
        case 'r':
            codepoint = '\r';
            break;
        case 't':
            codepoint = '\t';
            break;
        case 'u': {
            const int status = json_unicode_escape(bytes, byte_count, position, &codepoint);
            if (status != DORA_OK) {
                return EILSEQ;
            }
            break;
        }
        default:
            return EILSEQ;
        }
        *ascii = *ascii && utf8_scalar_is_ascii(codepoint);
        size_t written;
        status = utf8_encode_scalar(
            codepoint,
            (struct dora_mutable_bytes){output + length, byte_count - (size_t)(output - bytes) - length},
            &written);
        if (status != DORA_OK) {
            return status;
        }
        length += written;
    }
    return EILSEQ;
}

static int json_value_span(struct dora_bytes bytes, size_t begin, size_t* out_end) {
    if (begin >= bytes.byte_count) {
        return EILSEQ;
    }
    if (bytes.data[begin] == '"') {
        const size_t end = dora_json_find_string_end(bytes, begin);
        if (end == SIZE_MAX) {
            return EILSEQ;
        }
        *out_end = end;
        return DORA_OK;
    }
    size_t position = begin;
    if (bytes.data[begin] == '[' || bytes.data[begin] == '{') {
        size_t depth = 1;
        ++position;
        while (position < bytes.byte_count) {
            const uint8_t byte = bytes.data[position];
            if (byte == '"') {
                position = dora_json_find_string_end(bytes, position);
                if (position == SIZE_MAX) {
                    return EILSEQ;
                }
                continue;
            }
            ++position;
            if (byte == '[' || byte == '{') {
                if (ckd_add(&depth, depth, (size_t)1)) {
                    return EOVERFLOW;
                }
            } else if (byte == ']' || byte == '}') {
                --depth;
                if (depth == 0) {
                    *out_end = position;
                    return DORA_OK;
                }
            }
        }
        return EILSEQ;
    }
    while (position < bytes.byte_count && !json_byte_is_whitespace(bytes.data[position]) &&
           bytes.data[position] != ',' && bytes.data[position] != ']' && bytes.data[position] != '}') {
        ++position;
    }
    if (position == begin) {
        return EILSEQ;
    }
    *out_end = position;
    return DORA_OK;
}

static int json_decode_token(struct json_parser* parser,
                             struct dora_allocation* scratch,
                             struct dora_bytes token,
                             struct dora_bytes* out_text) {
    if (token.byte_count == 0) {
        return EILSEQ;
    }
    int status = dora_memory_payload_reserve(parser->text.pool, token.byte_count, 0, scratch);
    if (status != DORA_OK) {
        return status;
    }
    memcpy(scratch->address, token.data, token.byte_count);
    size_t position = 0;
    parser->is_decoded_text_ascii = parser->record_proof == DORA_UTF8_VALIDATED_ASCII;
    status = json_string(scratch->address,
                         token.byte_count,
                         &position,
                         &parser->string_events,
                         out_text,
                         &parser->is_decoded_text_ascii);
    return status != DORA_OK ? status : position != token.byte_count ? EILSEQ : DORA_OK;
}

static int json_member_next(struct dora_bytes bytes,
                            size_t* in_out_position,
                            struct dora_bytes* out_key_token,
                            struct text_infer_value* out_value) {
    size_t position = json_skip_space(bytes, *in_out_position);
    if (position >= bytes.byte_count) {
        return EILSEQ;
    }
    if (bytes.data[position] == '}') {
        return ENODATA;
    }
    if (bytes.data[position] != '"') {
        return EILSEQ;
    }
    const size_t key_end = dora_json_find_string_end(bytes, position);
    if (key_end == SIZE_MAX) {
        return EILSEQ;
    }
    const struct dora_bytes key = {bytes.data + position, key_end - position};
    position = json_skip_space(bytes, key_end);
    if (position == bytes.byte_count || bytes.data[position] != ':') {
        return EILSEQ;
    }
    position = json_skip_space(bytes, position + 1);
    size_t end = 0;
    const int status = json_value_span(bytes, position, &end);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_bytes raw = {bytes.data + position, end - position};
    const uint8_t first = raw.data[0];
    const bool is_null = first == 'n' && raw.byte_count == 4;
    enum text_value_kind kind = TEXT_VALUE_NUMBER;
    if (is_null) {
        kind = TEXT_VALUE_NULL;
    } else if (first == 't' || first == 'f') {
        kind = TEXT_VALUE_BOOL;
    } else if (first == '"') {
        kind = TEXT_VALUE_STRING;
    } else if (first == '[' || first == '{') {
        kind = TEXT_VALUE_OPAQUE;
    }
    position = json_skip_space(bytes, end);
    if (position < bytes.byte_count && bytes.data[position] == ',') {
        ++position;
    }
    *in_out_position = position;
    *out_key_token = key;
    *out_value = (struct text_infer_value){.bytes = raw, .raw = raw, .kind = kind, .is_null = is_null};
    return DORA_OK;
}

static int
json_record_begin(struct json_parser* parser, size_t end, struct dora_bytes* out_bytes, size_t* out_position) {
    const struct dora_bytes bytes = {
        (const uint8_t*)parser->text.buffer.address + parser->text.cursor_byte_offset,
        end - parser->text.cursor_byte_offset
    };
    bool ascii = false;
    int status =
        parser->options.content_policy == DORA_FAILURE_POLICY_TRUST ? DORA_OK : dora_utf8_validate_span(bytes, &ascii);
    parser->record_proof = parser->options.content_policy == DORA_FAILURE_POLICY_TRUST ? DORA_UTF8_PROMISED
                           : ascii                                                     ? DORA_UTF8_VALIDATED_ASCII
                                                                                       : DORA_UTF8_VALIDATED;
    if (status == DORA_OK) {
        status = dora_json_validate_record(parser->text.pool, bytes, parser->record_proof);
        if (status == EINVAL) {
            status = EILSEQ;
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    const size_t begin = json_skip_space(bytes, 0);
    if (begin == bytes.byte_count || bytes.data[begin] != '{') {
        return EILSEQ;
    }
    *out_bytes = bytes;
    *out_position = begin + 1;
    return DORA_OK;
}

static int json_parse_object(struct json_reader* state, size_t end, struct dora_builder* builder) {
    struct dora_bytes bytes = {};
    size_t position = 0;
    int status = json_record_begin(&state->parser, end, &bytes, &position);
    if (status != DORA_OK) {
        return status;
    }
    struct json_field* const fields = state->fields.address;
    for (uint32_t index = 0; index < state->column_count; ++index) {
        fields[index].was_seen = false;
    }
    while (true) {
        struct dora_bytes key_token = {};
        struct text_infer_value value = {};
        status = json_member_next(bytes, &position, &key_token, &value);
        if (status == ENODATA) {
            break;
        }
        if (status != DORA_OK) {
            return status;
        }
        struct dora_bytes key = {};
        status = json_decode_token(&state->parser, &state->scratch, key_token, &key);
        if (status != DORA_OK) {
            return status;
        }
        if (key.byte_count != 0 && memchr(key.data, 0, key.byte_count) != nullptr) {
            return ENOTSUP;
        }
        uint32_t column = 0;
        for (; column < state->column_count; ++column) {
            if (fields[column].name_byte_count == key.byte_count &&
                (key.byte_count == 0 || memcmp(fields[column].name, key.data, key.byte_count) == 0)) {
                break;
            }
        }
        if (column == state->column_count || fields[column].was_seen) {
            return EILSEQ;
        }
        fields[column].was_seen = true;
        value.proof = state->parser.record_proof;
        value.raw_proof = state->parser.record_proof;
        if (value.kind == TEXT_VALUE_STRING && !fields[column].is_json_extension) {
            status = json_decode_token(&state->parser, &state->scratch, value.raw, &value.bytes);
            value.proof = value.proof == DORA_UTF8_PROMISED     ? DORA_UTF8_PROMISED
                          : state->parser.is_decoded_text_ascii ? DORA_UTF8_VALIDATED_ASCII
                                                                : DORA_UTF8_VALIDATED;
            if (status != DORA_OK) {
                return status;
            }
        }
        if (!state->is_inferred && !fields[column].is_json_extension && value.kind == TEXT_VALUE_OPAQUE) {
            return ENOTSUP;
        }
        struct dora_builder child = {.handle = builder->handle, .node_index = column + 1};
        child.handle.is_owner = false;
        const bool numeric = value.kind == TEXT_VALUE_NUMBER;
        const bool negative = numeric && value.bytes.data[0] == '-';
        const uint8_t first = numeric ? value.bytes.data[negative ? 1 : 0] | 0x20U : 0;
        if (first == 'i' || first == 'n') {
            if (fields[column].is_json_extension) {
                return EILSEQ;
            }
            if (fields[column].physical_type == DORA_PHYSICAL_TYPE_NONE) {
                status = state->parser.options.content_policy == DORA_FAILURE_POLICY_NULL
                             ? dora_builder_append_null(&child)
                             : EINVAL;
            } else {
                static constexpr union decoded_row_value numbers[3][4] = {
                    {{.f16.bits = 0x7c00}, {.f16.bits = 0xfc00}, {.f16.bits = 0x7e00}, {.f16.bits = 0xfe00}},
                    {{.f32 = INFINITY}, {.f32 = -INFINITY}, {.f32 = NAN}, {.f32 = -NAN}},
                    {{.f64 = INFINITY}, {.f64 = -INFINITY}, {.f64 = NAN}, {.f64 = -NAN}}
                };
                const size_t special = (first == 'n' ? 2U : 0U) + (negative ? 1U : 0U);
                status = dora_builder_append_scalar_error(
                    &child,
                    fields[column].physical_type,
                    &numbers[fields[column].physical_type - DORA_PHYSICAL_TYPE_F16][special],
                    nullptr);
            }
        } else if (state->is_inferred) {
            const struct text_infer_column* const inferred = state->inferred.column_storage.address;
            status =
                dora_text_inferred_append(&child, inferred[column].kind, value, state->parser.options.content_policy);
        } else {
            const struct dora_bytes text = fields[column].is_json_extension ? value.raw : value.bytes;
            status = value.is_null ? dora_builder_append_null(&child)
                                   : dora_builder_append_reader_text(&child,
                                                                     (const char*)text.data,
                                                                     text.byte_count,
                                                                     fields[column].is_json_extension ? value.raw_proof
                                                                                                      : value.proof);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    for (uint32_t index = 0; index < state->column_count; ++index) {
        if (!fields[index].was_seen) {
            struct dora_builder child = {.handle = builder->handle, .node_index = index + 1};
            child.handle.is_owner = false;
            status = dora_builder_append_null(&child);
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    return dora_builder_end_row(builder);
}

static int json_find_record(size_t* out_end, struct json_parser* state, size_t* out_next) {
    const struct dora_bytes bytes = {.data = state->text.buffer.address, .byte_count = state->text.used_bytes};
    *out_end = bytes.byte_count;
    if (state->options.format == DORA_JSON_FORMAT_NEWLINE_DELIMITED) {
        for (size_t position = state->text.scan_byte_offset; position < bytes.byte_count; ++position) {
            if (bytes.data[position] == '\n' || bytes.data[position] == '\r') {
                size_t next = position + 1;
                if (bytes.data[position] == '\r') {
                    if (next == bytes.byte_count && !state->text.is_end) {
                        *out_end = position;
                        state->text.scan_byte_offset = position;
                        return EAGAIN;
                    }
                    next += next < bytes.byte_count && bytes.data[next] == '\n' ? 1U : 0U;
                }
                *out_end = position;
                *out_next = next;
                return DORA_OK;
            }
        }
        if (!state->text.is_end) {
            state->text.scan_byte_offset = bytes.byte_count;
            return EAGAIN;
        }
        if (state->text.cursor_byte_offset == bytes.byte_count) {
            return ENODATA;
        }
        *out_next = bytes.byte_count;
        return DORA_OK;
    }

    while (true) {
        if (state->depth > 0) {
            size_t depth = state->depth;
            for (size_t position = state->text.scan_byte_offset; position < bytes.byte_count; ++position) {
                const uint8_t byte = bytes.data[position];
                if (state->is_in_string) {
                    if (state->is_escaped) {
                        state->is_escaped = false;
                    } else if (byte == '\\') {
                        state->is_escaped = true;
                    } else if (byte == '"') {
                        state->is_in_string = false;
                    }
                } else if (byte == '"') {
                    state->is_in_string = true;
                } else if (byte == '{' || byte == '[') {
                    if (ckd_add(&depth, depth, (size_t)1)) {
                        return EOVERFLOW;
                    }
                } else if (byte == '}' || byte == ']') {
                    if (--depth == 0) {
                        state->depth = 0;
                        *out_end = position + 1;
                        *out_next = position + 1;
                        return DORA_OK;
                    }
                }
            }
            state->depth = depth;
            if (!state->text.is_end) {
                state->text.scan_byte_offset = bytes.byte_count;
                return EAGAIN;
            }
            return EILSEQ;
        }

        const size_t position = json_skip_space(bytes, state->text.scan_byte_offset);
        if (position == bytes.byte_count) {
            state->text.cursor_byte_offset = position;
            state->text.scan_byte_offset = position;
            return !state->text.is_end ? EAGAIN : state->phase == JSON_ARRAY_AFTER_END ? ENODATA : EILSEQ;
        }
        const uint8_t byte = bytes.data[position];
        if (state->phase == JSON_ARRAY_EXPECT_OPEN) {
            if (byte != '[') {
                return EILSEQ;
            }
            state->phase = JSON_ARRAY_EXPECT_FIRST_OR_END;
        } else if (state->phase == JSON_ARRAY_AFTER_END) {
            return EILSEQ;
        } else if (byte == ']') {
            if (state->phase == JSON_ARRAY_EXPECT_OBJECT) {
                return EILSEQ;
            }
            state->phase = JSON_ARRAY_AFTER_END;
        } else if (state->phase == JSON_ARRAY_EXPECT_COMMA_OR_END) {
            if (byte != ',') {
                return EILSEQ;
            }
            state->phase = JSON_ARRAY_EXPECT_OBJECT;
        } else {
            if (byte != '{') {
                return EILSEQ;
            }
            state->text.cursor_byte_offset = position;
            state->text.scan_byte_offset = position + 1;
            state->depth = 1;
            continue;
        }
        state->text.cursor_byte_offset = position + 1;
        state->text.scan_byte_offset = position + 1;
    }
}

static void json_record_advance(struct json_parser* state, size_t next) {
    state->text.cursor_byte_offset = next;
    state->text.scan_byte_offset = next;
    if (state->options.format == DORA_JSON_FORMAT_ARRAY) {
        state->phase = JSON_ARRAY_EXPECT_COMMA_OR_END;
    }
}

static int json_next_object(struct json_parser* state, struct text_record_result* out_record, bool buffered) {
    out_record->is_blocked = false;
    out_record->is_end_of_stream = false;
    while (true) {
        int status = json_find_record(&out_record->end_byte_offset, state, &out_record->next_byte_offset);
        if (status == ENODATA) {
            out_record->is_end_of_stream = true;
            return DORA_OK;
        }
        if (status == EAGAIN && buffered) {
            out_record->is_blocked = true;
            return DORA_OK;
        }
        if (out_record->end_byte_offset - state->text.cursor_byte_offset > state->options.max_record_bytes) {
            return ENOBUFS;
        }
        if (status == DORA_OK) {
            const struct dora_bytes bytes = {state->text.buffer.address, out_record->end_byte_offset};
            if (json_skip_space(bytes, state->text.cursor_byte_offset) != out_record->end_byte_offset) {
                return DORA_OK;
            }
            json_record_advance(state, out_record->next_byte_offset);
            continue;
        }
        if (status != EAGAIN) {
            return status;
        }
        status = dora_text_input_refill(&state->text, &out_record->is_blocked);
        if (status != DORA_OK || out_record->is_blocked) {
            return status;
        }
    }
}

static void json_reader_clear(struct json_reader* state) {
    dora_vector_release(&state->block);
    dora_builder_release(&state->builder);
    dora_text_input_release(&state->parser.text);
    dora_text_columns_release(&state->inferred);
    dora_allocation_release(&state->scratch);
    dora_allocation_release(&state->fields);
}

static int json_reader_next(struct dora_vector_source* source,
                            struct dora_pool pool,
                            struct dora_vector* out_vector,
                            struct dora_stream_result* out_result,
                            struct dora_error* error) {
    struct json_reader* const state = (struct json_reader*)source;
    const uint64_t capacity = state->parser.options.row_capacity;
    int status = DORA_OK;
    if (state->block.handle.slot_index != 0) {
        goto publish;
    }
    const uint64_t block_capacity = capacity < TEXT_DEFAULT_ROW_CAPACITY ? TEXT_DEFAULT_ROW_CAPACITY : capacity;
    uint64_t rows = 0;
    struct text_record_result record;
    status = json_next_object(&state->parser, &record, false);
    if (status == DORA_OK && !record.is_blocked && !record.is_end_of_stream) {
        status = dora_text_reader_prepare(&state->builder,
                                          pool,
                                          &state->schema,
                                          capacity,
                                          state->parser.options.content_policy);
        while (status == DORA_OK) {
            status = json_parse_object(state, record.end_byte_offset, &state->builder);
            if (status != DORA_OK) {
                break;
            }
            ++rows;
            json_record_advance(&state->parser, record.next_byte_offset);
            if (rows == block_capacity) {
                break;
            }
            status = json_next_object(&state->parser, &record, capacity < TEXT_DEFAULT_ROW_CAPACITY);
            if (record.is_blocked || record.is_end_of_stream) {
                break;
            }
        }
    }
    if (status == DORA_OK && rows == 0) {
        *out_result = (struct dora_stream_result){
            .status = record.is_blocked ? DORA_STREAM_STATUS_BLOCKED : DORA_STREAM_STATUS_END
        };
        if (!record.is_blocked) {
            json_reader_clear(state);
        }
        return DORA_OK;
    }
publish:
    if (status == DORA_OK) {
        status = dora_text_reader_publish(&state->builder, &state->block, capacity, out_vector, out_result);
    }
    if (status != DORA_OK) {
        json_reader_clear(state);
        text_error_set(error, "Invalid JSON record or field value");
    }
    return status;
}

static void json_reader_release(struct dora_vector_source* source) {
    struct json_reader* const state = (struct json_reader*)source;
    json_reader_clear(state);
    dora_schema_release(&state->schema);
    struct dora_allocation allocation = state->allocation;
    dora_allocation_release(&allocation);
}

static int json_reader_init_core(struct dora_reader* reader,
                                 struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 struct dora_byte_source* input,
                                 const struct dora_json_options* options,
                                 struct text_input* adopted,
                                 struct text_columns* inferred,
                                 struct dora_error* out_error) {
    struct dora_json_options copy = {};
    int status = dora_json_options_copy(options, &copy);
    size_t window_bytes = 0;
    if (status == DORA_OK && ckd_mul(&window_bytes, copy.max_record_bytes, (size_t)2)) {
        status = EOVERFLOW;
    }
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(schema, &field);
    }
    if (status == DORA_OK && (field.type.id != DORA_TYPE_STRUCT || field.layout != DORA_LAYOUT_STRUCT ||
                              input == nullptr || input->read == nullptr)) {
        status = EINVAL;
    }
    const uint32_t columns = field.child_count;
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, columns, sizeof(struct json_field))) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, sizeof(struct json_reader), &allocation);
    }
    if (status != DORA_OK) {
        text_error_set(out_error, "Invalid JSON options or schema");
        return status;
    }
    struct json_reader* const state = allocation.address;
    *state = (struct json_reader){
        .source = {.next = json_reader_next, .release = json_reader_release},
        .allocation = allocation,
        .parser =
            {.text = {.pool = pool, .input = input, .window_bytes = window_bytes},
             .options = copy,
             .phase = JSON_ARRAY_EXPECT_OPEN},
        .column_count = columns,
    };
    status = dora_json_prepare_string_events(&state->parser.string_events);
    // The reader's schema claim prevents source-pool reset.
    struct dora_handle owner = {};
    if (status == DORA_OK) {
        status = dora_identity_claim_acquire(pool, &schema->handle, IDENTITY_KIND_SCHEMA, &owner);
    }
    if (status == DORA_OK) {
        state->schema = (struct dora_schema){.handle = owner, .node_index = schema->node_index};
        if (bytes != 0) {
            status = dora_memory_payload_allocate(pool, bytes, &state->fields);
        }
    }
    struct json_field* const fields = state->fields.address;
    for (uint32_t index = 0; status == DORA_OK && index < columns; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_child(schema, index, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == DORA_OK && (field.child_count != 0 || !text_field_is_supported(&field, true))) {
            status = ENOTSUP;
        }
        if (status != DORA_OK) {
            break;
        }
        const struct dora_bytes original = inferred != nullptr
                                               ? dora_text_column_name(inferred, index)
                                               : (struct dora_bytes){
                                                     (const uint8_t*)(field.name != nullptr ? field.name : ""),
                                                     field.name != nullptr ? strlen(field.name) : 0
                                                 };
        fields[index] = (struct json_field){
            .name = (const char*)original.data,
            .name_byte_count = original.byte_count,
            .physical_type = field.type.id >= DORA_TYPE_FLOAT16 && field.type.id <= DORA_TYPE_FLOAT64
                                 ? DORA_PHYSICAL_TYPE_F16 + field.type.id - DORA_TYPE_FLOAT16
                                 : DORA_PHYSICAL_TYPE_NONE,
            .is_json_extension = field.type.extension_name.byte_count != 0
        };
        status = dora_utf8_validate_span(
            (struct dora_bytes){
                .data = (const uint8_t*)fields[index].name,
                .byte_count = fields[index].name_byte_count
            },
            nullptr);
        for (uint32_t previous = 0; status == DORA_OK && previous < index; ++previous) {
            if (strcmp(fields[index].name, fields[previous].name) == 0) {
                status = EINVAL;
            }
        }
    }
    if (status != DORA_OK) {
        json_reader_release(&state->source);
        text_error_set(out_error, "Unsupported JSON field or duplicate field name");
        return status;
    }
    if (adopted != nullptr) {
        status = dora_text_input_adopt(adopted, &state->parser.text);
        if (status != DORA_OK) {
            json_reader_release(&state->source);
            return status;
        }
        state->inferred = *inferred;
        *inferred = (struct text_columns){};
        state->is_inferred = true;
    }
    return dora_reader_init(reader, pool, &state->source);
}

int dora_json_reader_init(struct dora_reader* reader,
                          struct dora_pool pool,
                          const struct dora_schema* schema,
                          struct dora_byte_source* input,
                          const struct dora_json_options* options,
                          struct dora_error* out_error) {
    return json_reader_init_core(reader, pool, schema, input, options, nullptr, nullptr, out_error);
}

void dora_json_schema_inference_cleanup(struct text_schema_inference_state* inference_state) {
    struct json_schema_inference* const state = (struct json_schema_inference*)inference_state;
    dora_text_input_release(&state->parser.text);
    dora_allocation_release(&state->scratch);
    dora_text_columns_release(&state->columns);
}

static int json_infer_record(struct json_schema_inference* state, size_t end) {
    struct dora_bytes bytes = {};
    size_t position = 0;
    int status = json_record_begin(&state->parser, end, &bytes, &position);
    if (status != DORA_OK) {
        return status;
    }
    dora_text_columns_unsee(&state->columns);
    while (true) {
        struct dora_bytes key_token = {};
        struct text_infer_value value = {};
        status = json_member_next(bytes, &position, &key_token, &value);
        if (status == ENODATA) {
            return DORA_OK;
        }
        if (status != DORA_OK) {
            return status;
        }
        struct dora_bytes key = {};
        status = json_decode_token(&state->parser, &state->scratch, key_token, &key);
        if (status != DORA_OK) {
            return status;
        }
        if (key.byte_count != 0 && memchr(key.data, 0, key.byte_count) != nullptr) {
            return ENOTSUP;
        }
        uint32_t column = dora_text_column_find(&state->columns, key);
        if (column == UINT32_MAX) {
            column = state->columns.column_count;
            status = dora_text_columns_add(&state->columns, state->common.allocation.pool, key);
            if (status != DORA_OK) {
                return status;
            }
        }
        struct text_infer_column* const fields = state->columns.column_storage.address;
        if (fields[column].was_seen) {
            return EILSEQ;
        }
        fields[column].was_seen = true;
        if (value.kind == TEXT_VALUE_STRING) {
            status = json_decode_token(&state->parser, &state->scratch, value.raw, &value.bytes);
            if (status != DORA_OK) {
                return status;
            }
        }
        dora_text_infer_json(&value, &fields[column]);
    }
}

int dora_json_schema_inference_scan(struct text_schema_inference_state* inference_state,
                                    struct text_scan_result* out_result) {
    struct json_schema_inference* const state = (struct json_schema_inference*)inference_state;
    while (true) {
        if (inference_state->sample_row_limit != UINT64_MAX &&
            inference_state->row_count == inference_state->sample_row_limit) {
            out_result->is_ready = true;
            return DORA_OK;
        }
        struct text_record_result record = {};
        int status = json_next_object(&state->parser, &record, false);
        out_result->is_blocked = record.is_blocked;
        if (status != DORA_OK || record.is_blocked) {
            return status;
        }
        if (record.is_end_of_stream) {
            out_result->is_ready = true;
            return DORA_OK;
        }
        status = json_infer_record(state, record.end_byte_offset);
        if (status != DORA_OK) {
            return status;
        }
        if (ckd_add(&inference_state->row_count, inference_state->row_count, UINT64_C(1))) {
            return EOVERFLOW;
        }
        json_record_advance(&state->parser, record.next_byte_offset);
    }
}

int dora_json_schema_inference_handoff(struct text_schema_inference_state* inference_state,
                                       struct dora_schema* out_schema,
                                       struct dora_reader* out_reader) {
    struct json_schema_inference* const state = (struct json_schema_inference*)inference_state;
    const struct dora_pool pool = inference_state->allocation.pool;
    int status = dora_text_columns_schema(&state->columns, pool, true, false, out_schema);
    if (status == DORA_OK) {
        status = json_reader_init_core(out_reader,
                                       pool,
                                       out_schema,
                                       state->parser.text.input,
                                       &state->parser.options,
                                       &state->parser.text,
                                       &state->columns,
                                       nullptr);
    }
    return status;
}

int dora_json_schema_inference_init(struct dora_schema_inference* inference,
                                    struct dora_pool pool,
                                    struct dora_byte_source* input,
                                    const struct dora_json_options* options,
                                    const struct dora_schema_inference_options* inference_options,
                                    struct dora_error* out_error) {
    struct dora_json_options copy = {};
    int status = dora_json_options_copy(options, &copy);
    size_t window_bytes = 0;
    if (status == DORA_OK && ckd_mul(&window_bytes, copy.max_record_bytes, (size_t)2)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK &&
        (!dora_text_schema_inference_is_empty(inference) || input == nullptr || input->read == nullptr)) {
        status = EINVAL;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, sizeof(struct json_schema_inference), &allocation);
    }
    if (status != DORA_OK) {
        text_error_set(out_error, "Invalid JSON discovery options");
        return status;
    }
    struct json_schema_inference* const state = allocation.address;
    *state = (struct json_schema_inference){
        .common = {.allocation = allocation, .format = TEXT_FORMAT_JSON},
        .parser =
            {.text = {.pool = pool, .window_bytes = window_bytes}, .options = copy, .phase = JSON_ARRAY_EXPECT_OPEN}
    };
    status = dora_json_prepare_string_events(&state->parser.string_events);
    if (status == DORA_OK) {
        status = dora_text_schema_inference_publish(inference,
                                                    &state->common,
                                                    &state->parser.text,
                                                    input,
                                                    inference_options);
    } else {
        dora_allocation_release(&allocation);
    }
    if (status != DORA_OK) {
        text_error_set(out_error, "JSON discovery initialization failed");
    }
    return status;
}
