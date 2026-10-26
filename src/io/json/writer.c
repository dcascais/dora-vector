// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/json.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "io/json/json_internal.h"
#include "io/text.h"
#include "io/text_writer.h"
#include "json/json.h"
#include "memory/memory_internal.h"
#include "types/text.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

struct json_writer {
    struct dora_vector_sink sink;
    struct dora_allocation allocation;
    struct text_writer_output text;
    struct dora_json_options options;
    struct dora_utf8_byte_matcher string_events;
    bool has_rows;
};

static int json_write_string(struct json_writer* state, struct dora_bytes value, enum dora_utf8_proof proof) {
    int status = proof < DORA_UTF8_VALIDATED ? dora_utf8_validate_span(value, nullptr) : DORA_OK;
    if (status == DORA_OK) {
        status = dora_text_writer_write(&state->text, "\"", 1);
    }
    size_t begin = 0;
    for (size_t index = 0; status == DORA_OK && index < value.byte_count; ++index) {
        size_t event;
        status = utf8_find_byte_set(value, &state->string_events, index, &event);
        if (status != DORA_OK || event == value.byte_count) {
            break;
        }
        index = event;
        const uint8_t byte = value.data[index];
        status = dora_text_writer_write(&state->text, value.data + begin, index - begin);
        static constexpr char escapes[93][7] = {
            [0] = "\\u0000",  [1] = "\\u0001",  [2] = "\\u0002",  [3] = "\\u0003",  [4] = "\\u0004",  [5] = "\\u0005",
            [6] = "\\u0006",  [7] = "\\u0007",  [8] = "\\b",      [9] = "\\t",      [10] = "\\n",     [11] = "\\u000B",
            [12] = "\\f",     [13] = "\\r",     [14] = "\\u000E", [15] = "\\u000F", [16] = "\\u0010", [17] = "\\u0011",
            [18] = "\\u0012", [19] = "\\u0013", [20] = "\\u0014", [21] = "\\u0015", [22] = "\\u0016", [23] = "\\u0017",
            [24] = "\\u0018", [25] = "\\u0019", [26] = "\\u001A", [27] = "\\u001B", [28] = "\\u001C", [29] = "\\u001D",
            [30] = "\\u001E", [31] = "\\u001F", ['"'] = "\\\"",   ['\\'] = "\\\\"
        };
        const char* const escaped = escapes[byte];
        const size_t length = escaped[1] == 'u' ? 6 : 2;
        if (status == DORA_OK) {
            status = dora_text_writer_write(&state->text, escaped, length);
        }
        begin = index + 1;
    }
    if (status == DORA_OK && value.byte_count != 0) {
        status = dora_text_writer_write(&state->text, value.data + begin, value.byte_count - begin);
    }
    return status == DORA_OK ? dora_text_writer_write(&state->text, "\"", 1) : status;
}

static int json_writer_push(struct dora_vector_sink* sink,
                            const struct dora_vector* input,
                            struct dora_stream_result* out_result,
                            struct dora_error* error) {
    struct json_writer* const state = (struct json_writer*)sink;
    struct text_writer_input prepared = {};
    int status = dora_text_writer_prepare_input(state->allocation.pool, input, &prepared);
    struct text_writer_column* const columns = prepared.columns;
    const uint32_t count = prepared.column_count;
    char inline_text[TYPE_FLOAT_TEXT_CAPACITY];
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        status = dora_text_writer_bind_input_column(&prepared, index);
        if (status != DORA_OK) {
            break;
        }
        const char* const name = columns[index].field.name != nullptr ? columns[index].field.name : "";
        status = dora_utf8_validate_span((struct dora_bytes){.data = (const uint8_t*)name, .byte_count = strlen(name)},
                                         nullptr);
        for (uint32_t previous = 0; status == DORA_OK && previous < index; ++previous) {
            const char* const other = columns[previous].field.name != nullptr ? columns[previous].field.name : "";
            if (strcmp(name, other) == 0) {
                status = EINVAL;
            }
        }
    }
    if (status == DORA_OK && !state->has_rows && prepared.view.length != 0) {
        status = dora_text_writer_prepare_output(&state->text, state->allocation.pool);
        if (status == DORA_OK && state->options.format == DORA_JSON_FORMAT_ARRAY) {
            status = dora_text_writer_write(&state->text, "[\n\t", 3);
        }
    }
    for (uint64_t row = 0; status == DORA_OK && row < prepared.view.length; ++row) {
        uint64_t root_row = row;
        bool root_null = false;
        if (prepared.bound.nodes != nullptr) {
            const struct dora_vector_view* view = nullptr;
            status = text_writer_resolve(&prepared.bound, 0, &root_row, &view, &root_null);
        }
        if (status == DORA_OK && state->has_rows && state->options.format == DORA_JSON_FORMAT_ARRAY) {
            status = dora_text_writer_write(&state->text, ",\n\t", 3);
        }
        if (status == DORA_OK) {
            status = dora_text_writer_write(&state->text, "{", 1);
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
                status = dora_text_writer_write(&state->text, ",", 1);
            }
            if (status == DORA_OK) {
                const char* const name = columns[index].field.name;
                status = json_write_string(
                    state,
                    (struct dora_bytes){.data = (const uint8_t*)name, .byte_count = name != nullptr ? strlen(name) : 0},
                    DORA_UTF8_VALIDATED);
            }
            if (status == DORA_OK) {
                status = dora_text_writer_write(&state->text, ":", 1);
            }
            if (status != DORA_OK) {
                break;
            }
            const enum dora_type type = columns[index].field.type.id;
            const bool floating = type >= DORA_TYPE_FLOAT16 && type <= DORA_TYPE_FLOAT64;
            if (floating && value.byte_count != 0) {
                if (value.data[0] == 'n') {
                    value = (struct dora_bytes)DORA_BYTES_LITERAL("NaN");
                } else if (value.data[0] == 'i') {
                    value = (struct dora_bytes)DORA_BYTES_LITERAL("Infinity");
                } else if (value.byte_count > 1 && value.data[1] == 'i') {
                    value = (struct dora_bytes)DORA_BYTES_LITERAL("-Infinity");
                }
            }
            const bool string = type == DORA_TYPE_UTF8 || type >= DORA_TYPE_DATE32;
            status = is_null ? dora_text_writer_write(&state->text, "null", 4)
                     : string
                         ? json_write_string(state,
                                             value,
                                             type == DORA_TYPE_UTF8 ? columns[index].proof : DORA_UTF8_VALIDATED_ASCII)
                         : dora_text_writer_write(&state->text, value.data, value.byte_count);
        }
        if (status == DORA_OK) {
            const struct dora_bytes ending = state->options.format == DORA_JSON_FORMAT_NEWLINE_DELIMITED
                                                 ? (struct dora_bytes)DORA_BYTES_LITERAL("}\n")
                                                 : (struct dora_bytes)DORA_BYTES_LITERAL("}");
            status = dora_text_writer_write(&state->text, ending.data, ending.byte_count);
        }
        state->has_rows = true;
    }
    dora_text_writer_release_input(&prepared);
    if (status == DORA_OK) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY, .is_input_consumed = true};
    } else {
        dora_allocation_release(&state->text.buffer);
        state->text.used_bytes = 0;
        text_error_set(error, "JSON field formatting or write failed");
    }
    return status;
}

static int
json_writer_finish(struct dora_vector_sink* sink, struct dora_stream_result* out_result, struct dora_error* error) {
    struct json_writer* const state = (struct json_writer*)sink;
    if (state->options.format == DORA_JSON_FORMAT_ARRAY) {
        const int status = state->has_rows ? dora_text_writer_write(&state->text, "\n]\n", 3)
                                           : dora_text_writer_write(&state->text, "[\n\t\n]\n", 6);
        if (status != DORA_OK) {
            text_error_set(error, "JSON final write or flush failed");
            dora_allocation_release(&state->text.buffer);
            state->text.used_bytes = 0;
            return status;
        }
    }
    const int drained = dora_text_writer_drain(&state->text);
    if (drained != DORA_OK) {
        text_error_set(error, "Text output drain failed");
        dora_allocation_release(&state->text.buffer);
        return drained;
    }
    if (state->text.sink->flush != nullptr) {
        const int status = state->text.sink->flush(state->text.sink);
        if (status != DORA_OK) {
            text_error_set(error, "JSON final write or flush failed");
            dora_allocation_release(&state->text.buffer);
            state->text.used_bytes = 0;
            return status;
        }
    }
    dora_allocation_release(&state->text.buffer);
    *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
    return DORA_OK;
}

static void json_writer_release(struct dora_vector_sink* sink) {
    struct json_writer* const state = (struct json_writer*)sink;
    if (state->text.sink->release != nullptr) {
        state->text.sink->release(state->text.sink);
    }
    dora_allocation_release(&state->text.buffer);
    struct dora_allocation allocation = state->allocation;
    dora_allocation_release(&allocation);
}

int dora_json_writer_init(struct dora_writer* writer,
                          struct dora_pool pool,
                          struct dora_byte_sink* output,
                          const struct dora_json_options* options,
                          struct dora_error* out_error) {
    struct dora_json_options copy = {};
    int status = dora_json_options_copy(options, &copy);
    if (status == DORA_OK && (output == nullptr || output->write == nullptr)) {
        status = EINVAL;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, sizeof(struct json_writer), &allocation);
    }
    if (status != DORA_OK) {
        if (output != nullptr && output->release != nullptr) {
            output->release(output);
        }
        text_error_set(out_error, "Invalid JSON options or sink");
        return status;
    }
    struct json_writer* const state = allocation.address;
    *state = (struct json_writer){
        .sink = {.push = json_writer_push, .finish = json_writer_finish, .release = json_writer_release},
        .allocation = allocation,
        .text.sink = output,
        .options = copy
    };
    status = dora_json_prepare_string_events(&state->string_events);
    if (status != DORA_OK) {
        json_writer_release(&state->sink);
        return status;
    }
    return dora_writer_init(writer, pool, &state->sink);
}
