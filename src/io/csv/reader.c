// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "builder/builder_internal.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "identity/identity.h"
#include "io/csv/csv_inference.h"
#include "io/csv/csv_internal.h"
#include "io/inferred_columns.h"
#include "io/schema_inference_internal.h"
#include "io/text.h"
#include "io/text_reader.h"
#include "memory/memory_internal.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

struct csv_parser {
    struct text_input text;
    struct dora_csv_options options;
    struct dora_utf8_byte_matcher field_events;
    struct dora_utf8_byte_matcher quoted_events;
    struct dora_utf8_byte_matcher delimiter_event;
    bool has_seen_header;
    bool has_seen_bom;
    bool is_in_quotes;
    bool is_field_start;
};

struct csv_reader {
    struct dora_vector_source source;
    struct dora_allocation allocation;
    struct dora_schema schema;
    struct dora_builder builder;
    struct csv_parser parser;
    struct text_columns inferred;
    uint32_t column_count;
    bool is_inferred;
    struct dora_vector block;
    uint8_t null_bytes[];
};

struct csv_schema_inference {
    struct text_schema_inference_state common;
    struct csv_parser parser;
    struct text_columns columns;
    struct dora_allocation decode;
    bool has_seen_width;
    uint8_t null_bytes[];
};

static int csv_input_window(struct dora_csv_options* options) {
    static constexpr size_t default_record_bytes = 2000000;
    if (options->max_record_bytes == 0) {
        options->max_record_bytes =
            options->input_window_bytes != 0 ? options->input_window_bytes : default_record_bytes;
    }
    if (options->input_window_bytes == 0) {
        const size_t record_bytes =
            options->max_record_bytes > default_record_bytes ? options->max_record_bytes : default_record_bytes;
        if (ckd_mul(&options->input_window_bytes, record_bytes, (size_t)16)) {
            return EOVERFLOW;
        }
    }
    return options->input_window_bytes < options->max_record_bytes ? EINVAL : DORA_OK;
}

static bool csv_is_outer_space(uint8_t byte, const struct dora_csv_options* options) {
    return byte == ' ' && byte != options->delimiter && byte != options->quote && byte != options->escape;
}

static int csv_find_record(size_t* out_end, struct csv_parser* state, size_t* out_next) {
    const struct dora_bytes input = {state->text.buffer.address, state->text.used_bytes};
    const uint8_t* bytes = input.data;
    *out_end = state->text.cursor_byte_offset;
    if (!state->has_seen_bom) {
        size_t bom = 0;
        const int status = dora_utf8_bom_prefix(input, state->text.is_end, &bom);
        if (status != DORA_OK) {
            return status;
        }
        state->text.cursor_byte_offset = bom;
        state->text.scan_byte_offset = bom;
        state->has_seen_bom = true;
    }
    if (state->text.is_end && state->text.cursor_byte_offset == input.byte_count) {
        return ENODATA;
    }
    bool quoted = state->is_in_quotes;
    bool is_field_start = state->is_field_start;
    size_t index = state->text.scan_byte_offset;
    *out_end = input.byte_count;
    while (index < input.byte_count) {
        size_t event;
        const int status = dora_utf8_find_byte_set_general(input,
                                                           quoted ? &state->quoted_events : &state->field_events,
                                                           index,
                                                           &event);
        if (status != 0) {
            return status;
        }
        if (!quoted && event != index) {
            is_field_start = false;
        }
        index = event;
        if (index == input.byte_count) {
            break;
        }
        const uint8_t byte = bytes[index];
        if (quoted) {
            if (byte == state->options.escape) {
                if (index + 1 == input.byte_count && !state->text.is_end) {
                    break;
                }
                if (index + 1 < input.byte_count && (byte != state->options.quote || bytes[index + 1] == byte)) {
                    index += 2;
                    continue;
                }
            }
            if (byte == state->options.quote) {
                quoted = false;
            }
            ++index;
            continue;
        }
        if (byte == '\r' || byte == '\n') {
            size_t next = index + 1;
            if (byte == '\r') {
                if (next == input.byte_count && !state->text.is_end) {
                    *out_end = index;
                    break;
                }
                next += next < input.byte_count && bytes[next] == '\n' ? 1U : 0U;
            }
            *out_end = index;
            *out_next = next;
            return 0;
        }
        if (byte == state->options.delimiter) {
            is_field_start = true;
        } else if (is_field_start && byte == state->options.quote) {
            quoted = true;
            is_field_start = false;
        } else if (byte != ' ' || byte == state->options.escape) {
            is_field_start = false;
        }
        ++index;
    }
    if (!state->text.is_end) {
        state->text.scan_byte_offset = index;
        state->is_in_quotes = quoted;
        state->is_field_start = is_field_start;
        return EAGAIN;
    }
    if (quoted) {
        return EILSEQ;
    }
    *out_next = input.byte_count;
    return 0;
}

static int csv_field_next(struct csv_parser* state,
                          size_t end,
                          size_t* in_out_position,
                          struct text_infer_value* out_value,
                          bool* out_more,
                          struct dora_allocation* scratch) {
    uint8_t* const bytes = state->text.buffer.address;
    const struct dora_bytes input = {bytes, end};
    size_t position = *in_out_position;
    const size_t raw_begin = position;
    while (position < end && csv_is_outer_space(bytes[position], &state->options)) {
        ++position;
    }
    const bool quoted = position < end && bytes[position] == state->options.quote;
    size_t begin = raw_begin;
    size_t length = 0;
    bool decoded = false;
    if (quoted) {
        begin = ++position;
        size_t output = begin;
        bool displaced = false;
        bool closed = false;
        while (position < end) {
            size_t event = 0;
            const int status = dora_utf8_find_byte_set_general(input, &state->quoted_events, position, &event);
            if (status != DORA_OK) {
                return status;
            }
            const size_t run = event - position;
            if (displaced && run != 0) {
                size_t written = 0;
                const int moved =
                    utf8_move_bytes((struct dora_bytes){bytes + position, run},
                                    (struct dora_mutable_bytes){
                                        decoded ? (uint8_t*)scratch->address + output - begin : bytes + output,
                                        run
                                    },
                                    &written);
                if (moved != 0) {
                    return moved;
                }
            }
            output += run;
            position = event;
            if (position == end) {
                break;
            }
            const uint8_t byte = bytes[position];
            if (byte == state->options.escape && state->options.escape != 0 && position + 1 < end &&
                (byte != state->options.quote || bytes[position + 1] == byte)) {
                if (!displaced && scratch != nullptr) {
                    const int reserved = dora_memory_payload_reserve(state->text.pool, end - begin, 0, scratch);
                    if (reserved != DORA_OK) {
                        return reserved;
                    }
                    memcpy(scratch->address, bytes + begin, output - begin);
                    decoded = true;
                }
                displaced = true;
                if (decoded) {
                    ((uint8_t*)scratch->address)[output++ - begin] = bytes[position + 1];
                } else {
                    bytes[output++] = bytes[position + 1];
                }
                position += 2;
            } else if (byte == state->options.quote) {
                ++position;
                closed = true;
                break;
            } else {
                return EILSEQ;
            }
        }
        if (!closed) {
            return EILSEQ;
        }
        length = output - begin;
        while (position < end && csv_is_outer_space(bytes[position], &state->options)) {
            ++position;
        }
    } else {
        const int status = dora_utf8_find_byte_set_general(input, &state->delimiter_event, raw_begin, &position);
        if (status != DORA_OK) {
            return status;
        }
        length = position - begin;
    }
    if (position < end && bytes[position] != state->options.delimiter) {
        return EILSEQ;
    }
    const uint8_t* const value = decoded ? scratch->address : bytes + begin;
    *out_value = (struct text_infer_value){
        .bytes = {value, length},
        .kind = TEXT_VALUE_CSV,
        .is_null = !quoted && length == state->options.null_token.byte_count &&
                   (length == 0 || memcmp(value, state->options.null_token.data, length) == 0)
    };
    *out_more = position < end;
    *in_out_position = position < end ? position + 1 : position;
    return DORA_OK;
}

static int csv_parse_record(struct csv_reader* state, size_t end, struct dora_builder* builder) {
    size_t position = state->parser.text.cursor_byte_offset;
    if (state->column_count == 0) {
        return EILSEQ;
    }
    for (uint32_t column = 0; column < state->column_count; ++column) {
        struct text_infer_value value = {};
        bool more = false;
        int status = csv_field_next(&state->parser, end, &position, &value, &more, nullptr);
        if (status != DORA_OK) {
            return status;
        }
        const bool last = column + 1 == state->column_count;
        if (more == last) {
            return EILSEQ;
        }
        if (builder == nullptr && value.bytes.byte_count != 0 &&
            memchr(value.bytes.data, 0, value.bytes.byte_count) != nullptr) {
            return ENOTSUP;
        }
        if (builder != nullptr) {
            struct dora_builder child = {.handle = builder->handle, .node_index = column + 1};
            child.handle.is_owner = false;
            if (state->is_inferred) {
                const struct text_infer_column* const fields = state->inferred.column_storage.address;
                status =
                    dora_text_inferred_append(&child, fields[column].kind, value, state->parser.options.content_policy);
            } else {
                status = value.is_null ? dora_builder_append_null(&child)
                                       : dora_builder_append_reader_text(&child,
                                                                         (const char*)value.bytes.data,
                                                                         value.bytes.byte_count,
                                                                         DORA_UTF8_CHECKED);
            }
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    return builder != nullptr ? dora_builder_end_row(builder) : DORA_OK;
}

static int csv_prepare_reader_events(struct csv_parser* state) {
    uint64_t set[4];
    bitmap_fill_bytes((uint8_t*)set, sizeof set, false);
    bitmap_set(set, state->options.delimiter);
    int status = dora_utf8_byte_matcher_init(set, &state->delimiter_event);
    bitmap_set(set, state->options.quote);
    bitmap_set(set, '\r');
    bitmap_set(set, '\n');
    bitmap_set(set, ' ');
    if (status == 0) {
        status = dora_utf8_byte_matcher_init(set, &state->field_events);
    }
    bitmap_fill_bytes((uint8_t*)set, sizeof set, false);
    bitmap_set(set, state->options.quote);
    if (state->options.escape != 0) {
        bitmap_set(set, state->options.escape);
    }
    if (status == 0) {
        status = dora_utf8_byte_matcher_init(set, &state->quoted_events);
    }
    return status;
}

static int csv_next_record(struct csv_parser* state, struct text_record_result* out_record, bool buffered) {
    out_record->is_blocked = false;
    out_record->is_end_of_stream = false;
    while (true) {
        int status = csv_find_record(&out_record->end_byte_offset, state, &out_record->next_byte_offset);
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
        if (status == EAGAIN) {
            status = dora_text_input_refill(&state->text, &out_record->is_blocked);
            if (status != DORA_OK || out_record->is_blocked) {
                return status;
            }
            continue;
        }
        return status;
    }
}

static void csv_record_advance(struct csv_parser* state, size_t next) {
    state->text.cursor_byte_offset = next;
    state->text.scan_byte_offset = next;
    state->is_in_quotes = false;
    state->is_field_start = true;
}

static void csv_reader_clear(struct csv_reader* state) {
    dora_vector_release(&state->block);
    dora_builder_release(&state->builder);
    dora_text_input_release(&state->parser.text);
    dora_text_columns_release(&state->inferred);
}

static int csv_reader_next(struct dora_vector_source* source,
                           struct dora_pool pool,
                           struct dora_vector* out_vector,
                           struct dora_stream_result* out_result,
                           struct dora_error* error) {
    struct csv_reader* const state = (struct csv_reader*)source;
    const uint64_t capacity = state->parser.options.row_capacity;
    int status = DORA_OK;
    if (state->block.handle.slot_index != 0) {
        goto publish;
    }
    const uint64_t block_capacity = capacity < TEXT_DEFAULT_ROW_CAPACITY ? TEXT_DEFAULT_ROW_CAPACITY : capacity;
    uint64_t rows = 0;
    struct text_record_result record;
    status = csv_next_record(&state->parser, &record, false);
    if (status == DORA_OK && !record.is_blocked && !record.is_end_of_stream) {
        status = dora_text_reader_prepare(&state->builder,
                                          pool,
                                          &state->schema,
                                          capacity,
                                          state->parser.options.content_policy);
        while (status == DORA_OK) {
            const bool header = state->parser.options.has_header && !state->parser.has_seen_header;
            status = csv_parse_record(state, record.end_byte_offset, header ? nullptr : &state->builder);
            if (status != DORA_OK) {
                break;
            }
            state->parser.has_seen_header = true;
            rows += !header;
            csv_record_advance(&state->parser, record.next_byte_offset);
            if (rows == block_capacity) {
                break;
            }
            status = csv_next_record(&state->parser, &record, rows != 0 && capacity < TEXT_DEFAULT_ROW_CAPACITY);
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
            csv_reader_clear(state);
        }
        return DORA_OK;
    }
publish:
    if (status == DORA_OK) {
        status = dora_text_reader_publish(&state->builder, &state->block, capacity, out_vector, out_result);
    }
    if (status != DORA_OK) {
        csv_reader_clear(state);
        text_error_set(error, "Invalid CSV record or field value");
    }
    return status;
}

static void csv_reader_release(struct dora_vector_source* source) {
    struct csv_reader* const state = (struct csv_reader*)source;
    csv_reader_clear(state);
    dora_schema_release(&state->schema);
    struct dora_allocation allocation = state->allocation;
    dora_allocation_release(&allocation);
}

static int csv_reader_init_core(struct dora_reader* reader,
                                struct dora_pool pool,
                                const struct dora_schema* schema,
                                struct dora_byte_source* input,
                                const struct dora_csv_options* options,
                                struct text_input* adopted,
                                struct text_columns* inferred,
                                struct dora_error* out_error) {
    struct dora_csv_options copy = {};
    int status = dora_csv_options_copy(options, &copy);
    if (status == DORA_OK) {
        status = csv_input_window(&copy);
    }
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(schema, &field);
    }
    if (status == DORA_OK &&
        (field.type.id != DORA_TYPE_STRUCT || field.layout != DORA_LAYOUT_STRUCT ||
         (field.child_count == 0 && adopted == nullptr) || input == nullptr || input->read == nullptr)) {
        status = EINVAL;
    }
    const uint32_t columns = field.child_count;
    for (uint32_t index = 0; status == DORA_OK && index < columns; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_child(schema, index, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == DORA_OK && (field.child_count != 0 || !text_field_is_supported(&field, false))) {
            status = ENOTSUP;
        }
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, sizeof(struct csv_reader), copy.null_token.byte_count)) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &allocation);
    }
    if (status != DORA_OK) {
        text_error_set(out_error, "Invalid CSV options or schema");
        return status;
    }
    struct csv_reader* const state = allocation.address;
    *state = (struct csv_reader){
        .source = {.next = csv_reader_next, .release = csv_reader_release},
        .allocation = allocation,
        .parser =
            {.text = {.pool = pool, .input = input, .window_bytes = copy.input_window_bytes},
             .options = copy,
             .is_field_start = true},
        .column_count = columns,
    };
    if (copy.null_token.byte_count != 0) {
        memcpy(state->null_bytes, copy.null_token.data, copy.null_token.byte_count);
    }
    state->parser.options.null_token.data = state->null_bytes;
    status = csv_prepare_reader_events(&state->parser);
    // The reader's schema claim prevents source-pool reset.
    struct dora_handle owner = {};
    if (status == DORA_OK) {
        status = dora_identity_claim_acquire(pool, &schema->handle, IDENTITY_KIND_SCHEMA, &owner);
    }
    if (status == DORA_OK) {
        state->schema = (struct dora_schema){.handle = owner, .node_index = schema->node_index};
    }
    if (status != DORA_OK) {
        csv_reader_release(&state->source);
        return status;
    }
    if (adopted != nullptr) {
        status = dora_text_input_adopt(adopted, &state->parser.text);
        if (status != DORA_OK) {
            csv_reader_release(&state->source);
            return status;
        }
        state->inferred = *inferred;
        *inferred = (struct text_columns){};
        state->is_inferred = true;
    }
    return dora_reader_init(reader, pool, &state->source);
}

int dora_csv_reader_init(struct dora_reader* reader,
                         struct dora_pool pool,
                         const struct dora_schema* schema,
                         struct dora_byte_source* input,
                         const struct dora_csv_options* options,
                         struct dora_error* out_error) {
    return csv_reader_init_core(reader, pool, schema, input, options, nullptr, nullptr, out_error);
}

void dora_csv_schema_inference_cleanup(struct text_schema_inference_state* inference_state) {
    struct csv_schema_inference* const state = (struct csv_schema_inference*)inference_state;
    dora_text_input_release(&state->parser.text);
    dora_text_columns_release(&state->columns);
    dora_allocation_release(&state->decode);
}

static int csv_infer_record(struct csv_schema_inference* state, size_t end, bool header) {
    size_t position = state->parser.text.cursor_byte_offset;
    uint32_t column = 0;
    bool more = true;
    while (more) {
        struct text_infer_value value = {};
        int status = csv_field_next(&state->parser, end, &position, &value, &more, &state->decode);
        if (status != DORA_OK) {
            return status;
        }
        if (state->has_seen_width && column >= state->columns.column_count) {
            return EILSEQ;
        }
        if (header) {
            status = dora_utf8_validate_span(value.bytes, nullptr);
            if (status != DORA_OK) {
                return status;
            }
            if (value.bytes.byte_count != 0 && memchr(value.bytes.data, 0, value.bytes.byte_count) != nullptr) {
                return ENOTSUP;
            }
        }
        if (!state->has_seen_width) {
            const struct dora_bytes name = header && !value.is_null ? value.bytes : (struct dora_bytes){};
            status = dora_text_columns_add(&state->columns, state->common.allocation.pool, name);
            if (status != DORA_OK) {
                return status;
            }
        }
        if (!header && !value.is_null) {
            struct text_infer_column* const fields = state->columns.column_storage.address;
            status = dora_text_infer_csv(value.bytes, &fields[column]);
            if (status == EILSEQ && state->parser.options.content_policy == DORA_FAILURE_POLICY_NULL) {
                status = DORA_OK;
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        if (column == UINT32_MAX) {
            return EOVERFLOW;
        }
        ++column;
    }
    if (column != state->columns.column_count) {
        return EILSEQ;
    }
    state->has_seen_width = true;
    return DORA_OK;
}

int dora_csv_schema_inference_scan(struct text_schema_inference_state* inference_state,
                                   struct text_scan_result* out_result) {
    struct csv_schema_inference* const state = (struct csv_schema_inference*)inference_state;
    while (true) {
        if (inference_state->sample_row_limit != UINT64_MAX &&
            inference_state->row_count == inference_state->sample_row_limit) {
            out_result->is_ready = true;
            return DORA_OK;
        }
        struct text_record_result record = {};
        int status = csv_next_record(&state->parser, &record, false);
        out_result->is_blocked = record.is_blocked;
        if (status != DORA_OK || out_result->is_blocked) {
            return status;
        }
        if (record.is_end_of_stream) {
            out_result->is_ready = true;
            return DORA_OK;
        }
        const bool header = state->parser.options.has_header && !state->parser.has_seen_header;
        status = csv_infer_record(state, record.end_byte_offset, header);
        if (status != DORA_OK) {
            return status;
        }
        state->parser.has_seen_header = true;
        if (!header && ckd_add(&inference_state->row_count, inference_state->row_count, UINT64_C(1))) {
            return EOVERFLOW;
        }
        csv_record_advance(&state->parser, record.next_byte_offset);
    }
}

int dora_csv_schema_inference_handoff(struct text_schema_inference_state* inference_state,
                                      struct dora_schema* out_schema,
                                      struct dora_reader* out_reader) {
    struct csv_schema_inference* const state = (struct csv_schema_inference*)inference_state;
    const struct dora_pool pool = inference_state->allocation.pool;
    int status = dora_text_columns_schema(&state->columns, pool, false, state->parser.options.has_header, out_schema);
    if (status == DORA_OK) {
        status = csv_reader_init_core(out_reader,
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

int dora_csv_schema_inference_init(struct dora_schema_inference* inference,
                                   struct dora_pool pool,
                                   struct dora_byte_source* input,
                                   const struct dora_csv_options* options,
                                   const struct dora_schema_inference_options* inference_options,
                                   struct dora_error* out_error) {
    struct dora_csv_options copy = {};
    int status = dora_csv_options_copy(options, &copy);
    if (status == DORA_OK) {
        status = csv_input_window(&copy);
    }
    if (status == DORA_OK &&
        (!dora_text_schema_inference_is_empty(inference) || input == nullptr || input->read == nullptr)) {
        status = EINVAL;
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, sizeof(struct csv_schema_inference), copy.null_token.byte_count)) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &allocation);
    }
    if (status != DORA_OK) {
        text_error_set(out_error, "Invalid CSV discovery options");
        return status;
    }
    struct csv_schema_inference* const state = allocation.address;
    *state = (struct csv_schema_inference){
        .common = {.allocation = allocation, .format = TEXT_FORMAT_CSV},
        .parser =
            {.text = {.pool = pool, .window_bytes = copy.input_window_bytes}, .options = copy, .is_field_start = true}
    };
    if (copy.null_token.byte_count != 0) {
        memcpy(state->null_bytes, copy.null_token.data, copy.null_token.byte_count);
        state->parser.options.null_token.data = state->null_bytes;
    }
    status = csv_prepare_reader_events(&state->parser);
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
        text_error_set(out_error, "CSV discovery initialization failed");
    }
    return status;
}
