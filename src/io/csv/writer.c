// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "io/csv/csv_internal.h"
#include "io/text.h"
#include "io/text_writer.h"
#include "memory/memory_internal.h"
#include "types/text.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

struct csv_writer {
    struct dora_vector_sink sink;
    struct dora_allocation allocation;
    struct text_writer_output text;
    struct dora_csv_options options;
    struct dora_utf8_byte_matcher quote_events;
    struct dora_utf8_byte_matcher escape_events;
    bool has_started_output;
    uint8_t null_bytes[];
};

static int csv_write_field(struct csv_writer* state, struct dora_bytes value, bool is_null) {
    if (is_null) {
        return dora_text_writer_write(&state->text,
                                      state->options.null_token.data,
                                      state->options.null_token.byte_count);
    }
    bool quote = value.byte_count == state->options.null_token.byte_count &&
                 (value.byte_count == 0 || memcmp(value.data, state->options.null_token.data, value.byte_count) == 0);
    size_t event = 0;
    while (event < value.byte_count) {
        const int status = dora_utf8_find_byte_set_general(value, &state->quote_events, event, &event);
        if (status != DORA_OK) {
            return status;
        }
        if (event == value.byte_count) {
            break;
        }
        if (value.data[event] == state->options.quote && state->options.escape == 0) {
            return EILSEQ;
        }
        quote = true;
        ++event;
    }
    if (!quote) {
        return dora_text_writer_write(&state->text, value.data, value.byte_count);
    }
    int status = dora_text_writer_write(&state->text, &state->options.quote, 1);
    size_t begin = 0;
    event = 0;
    while (status == DORA_OK && event < value.byte_count) {
        status = dora_utf8_find_byte_set_general(value, &state->escape_events, event, &event);
        if (status != DORA_OK || event == value.byte_count) {
            break;
        }
        status = dora_text_writer_write(&state->text, value.data + begin, event - begin);
        if (status == DORA_OK) {
            status = dora_text_writer_write(&state->text, &state->options.escape, 1);
        }
        begin = event++;
    }
    if (status == DORA_OK && value.byte_count != 0) {
        status = dora_text_writer_write(&state->text, value.data + begin, value.byte_count - begin);
    }
    return status == DORA_OK ? dora_text_writer_write(&state->text, &state->options.quote, 1) : status;
}

static int csv_writer_push(struct dora_vector_sink* sink,
                           const struct dora_vector* input,
                           struct dora_stream_result* out_result,
                           struct dora_error* out_error) {
    struct csv_writer* const state = (struct csv_writer*)sink;
    struct text_writer_input prepared = {};
    int status = dora_text_writer_prepare_input(state->allocation.pool, input, &prepared);
    struct text_writer_column* const columns = prepared.columns;
    const uint32_t count = prepared.column_count;
    char inline_text[TYPE_FLOAT_TEXT_CAPACITY];
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        status = dora_text_writer_bind_input_column(&prepared, index);
    }
    if (status == DORA_OK && !state->has_started_output && (prepared.view.length != 0 || state->options.has_header)) {
        status = dora_text_writer_prepare_output(&state->text, state->allocation.pool);
        if (status == DORA_OK && state->options.has_header) {
            for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
                if (index != 0) {
                    status = dora_text_writer_write(&state->text, &state->options.delimiter, 1);
                }
                if (status == DORA_OK) {
                    const char* const name = columns[index].field.name;
                    status = csv_write_field(state,
                                             (struct dora_bytes){
                                                 .data = (const uint8_t*)name,
                                                 .byte_count = name != nullptr ? strlen(name) : 0
                                             },
                                             false);
                }
            }
            if (status == DORA_OK) {
                status = dora_text_writer_write(&state->text, "\n", 1);
            }
        }
        state->has_started_output = status == DORA_OK;
    }
    for (uint64_t row = 0; status == DORA_OK && row < prepared.view.length; ++row) {
        uint64_t root_row = row;
        bool root_null = false;
        if (prepared.bound.nodes != nullptr) {
            const struct dora_vector_view* view = nullptr;
            status = text_writer_resolve(&prepared.bound, 0, &root_row, &view, &root_null);
        }
        for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
            uint64_t value_row = root_row;
            const struct dora_vector_view* view = &columns[index].view;
            bool is_null = root_null;
            if (!is_null) {
                if (prepared.bound.nodes != nullptr) {
                    status = text_writer_resolve(&prepared.bound,
                                                 columns[index].ownership.root_node_index,
                                                 &value_row,
                                                 &view,
                                                 &is_null);
                } else if (text_writer_is_encoded(view)) {
                    status = text_writer_resolve(&columns[index].ownership.bound, 0, &value_row, &view, &is_null);
                } else {
                    is_null = dora_vector_view_is_null(view, value_row);
                }
            }
            struct dora_bytes value = {};
            if (status == DORA_OK && !is_null) {
                status = text_writer_format(&columns[index].field.type,
                                            view,
                                            value_row,
                                            &prepared.scope,
                                            &prepared.format_buffer,
                                            inline_text,
                                            sizeof(inline_text),
                                            &value);
            }
            if (status == DORA_OK && index != 0) {
                status = dora_text_writer_write(&state->text, &state->options.delimiter, 1);
            }
            if (status == DORA_OK) {
                status = csv_write_field(state, value, is_null);
            }
        }
        if (status == DORA_OK) {
            status = dora_text_writer_write(&state->text, "\n", 1);
        }
    }
    dora_text_writer_release_input(&prepared);
    if (status == DORA_OK) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY, .is_input_consumed = true};
    } else {
        dora_allocation_release(&state->text.buffer);
        state->text.used_bytes = 0;
        text_error_set(out_error, "CSV field formatting or write failed");
    }
    return status;
}

static int
csv_writer_finish(struct dora_vector_sink* sink, struct dora_stream_result* out_result, struct dora_error* error) {
    struct csv_writer* const state = (struct csv_writer*)sink;
    const int drained = dora_text_writer_drain(&state->text);
    if (drained != DORA_OK) {
        text_error_set(error, "Text output drain failed");
        dora_allocation_release(&state->text.buffer);
        return drained;
    }
    if (state->text.sink->flush != nullptr) {
        const int status = state->text.sink->flush(state->text.sink);
        if (status != DORA_OK) {
            text_error_set(error, "CSV flush failed");
            dora_allocation_release(&state->text.buffer);
            state->text.used_bytes = 0;
            return status;
        }
    }
    dora_allocation_release(&state->text.buffer);
    *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
    return DORA_OK;
}

static void csv_writer_release(struct dora_vector_sink* sink) {
    struct csv_writer* const state = (struct csv_writer*)sink;
    if (state->text.sink->release != nullptr) {
        state->text.sink->release(state->text.sink);
    }
    dora_allocation_release(&state->text.buffer);
    struct dora_allocation allocation = state->allocation;
    dora_allocation_release(&allocation);
}

int dora_csv_writer_init(struct dora_writer* writer,
                         struct dora_pool pool,
                         struct dora_byte_sink* output,
                         const struct dora_csv_options* options,
                         struct dora_error* out_error) {
    struct dora_csv_options copy = {};
    int status = dora_csv_options_copy(options, &copy);
    if (status == DORA_OK && (output == nullptr || output->write == nullptr)) {
        status = EINVAL;
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, sizeof(struct csv_writer), copy.null_token.byte_count)) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &allocation);
    }
    if (status != DORA_OK) {
        if (output != nullptr && output->release != nullptr) {
            output->release(output);
        }
        text_error_set(out_error, "Invalid CSV options or sink");
        return status;
    }
    struct csv_writer* const state = allocation.address;
    *state = (struct csv_writer){
        .sink = {.push = csv_writer_push, .finish = csv_writer_finish, .release = csv_writer_release},
        .allocation = allocation,
        .text.sink = output,
        .options = copy
    };
    if (copy.null_token.byte_count != 0) {
        memcpy(state->null_bytes, copy.null_token.data, copy.null_token.byte_count);
    }
    state->options.null_token.data = state->null_bytes;
    uint64_t set[4];
    bitmap_fill_bytes((uint8_t*)set, sizeof set, false);
    bitmap_set(set, copy.delimiter);
    bitmap_set(set, copy.quote);
    bitmap_set(set, '\r');
    bitmap_set(set, '\n');
    status = dora_utf8_byte_matcher_init(set, &state->quote_events);
    bitmap_fill_bytes((uint8_t*)set, sizeof set, false);
    bitmap_set(set, copy.quote);
    if (copy.escape != 0) {
        bitmap_set(set, copy.escape);
    }
    if (status == DORA_OK) {
        status = dora_utf8_byte_matcher_init(set, &state->escape_events);
    }
    if (status != DORA_OK) {
        csv_writer_release(&state->sink);
        return status;
    }
    return dora_writer_init(writer, pool, &state->sink);
}
