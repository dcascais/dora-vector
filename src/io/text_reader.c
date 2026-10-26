// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "io/text_reader.h"

#include "builder/builder_internal.h"
#include "builder/state.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "io/text.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

static int input_segment_create(struct text_input* state, struct text_input_segment** out_segment) {
    struct dora_allocation allocation = {};
    const int status = dora_memory_payload_allocate(state->pool, sizeof(struct text_input_segment), &allocation);
    if (status == DORA_OK) {
        *out_segment = allocation.address;
        **out_segment = (struct text_input_segment){.allocation = allocation};
    }
    return status;
}

static void input_segment_append(struct text_input* state, struct text_input_segment* segment) {
    if (state->tail == nullptr) {
        state->head = segment;
    } else {
        state->tail->next = segment;
    }
    state->tail = segment;
}

static void input_segment_next(struct text_input* state) {
    struct text_input_segment* const segment = state->head;
    state->head = segment->next;
    if (state->head == nullptr) {
        state->tail = nullptr;
    }
    dora_allocation_release(&state->buffer);
    state->buffer = segment->buffer;
    state->used_bytes = segment->valid_bytes;
    state->cursor_byte_offset = 0;
    state->scan_byte_offset = 0;
    state->is_end = segment->is_end;
    struct dora_allocation allocation = segment->allocation;
    dora_allocation_release(&allocation);
}

void dora_text_input_release(struct text_input* state) {
    dora_allocation_release(&state->buffer);
    while (state->head != nullptr) {
        input_segment_next(state);
    }
    dora_allocation_release(&state->buffer);
    state->used_bytes = 0;
    state->cursor_byte_offset = 0;
    state->scan_byte_offset = 0;
}

int dora_text_input_adopt(struct text_input* source, struct text_input* destination) {
    if (source->head != nullptr) {
        struct text_input_segment* final = nullptr;
        const int status = input_segment_create(source, &final);
        if (status != DORA_OK) {
            return status;
        }
        final->buffer = source->buffer;
        final->valid_bytes = source->used_bytes;
        final->is_end = source->is_end;
        input_segment_append(source, final);
        source->buffer = (struct dora_allocation){};
        input_segment_next(source);
    }
    source->cursor_byte_offset = 0;
    source->scan_byte_offset = 0;
    source->should_retain = false;
    source->replay_byte_limit = 0;
    *destination = *source;
    *source = (struct text_input){};
    return DORA_OK;
}

static size_t input_preference(struct text_input* state, size_t required) {
    size_t preferred = state->window_bytes;
    if (state->input->get_byte_count != nullptr) {
        uint64_t count = 0;
        const int status = state->input->get_byte_count(state->input, &count);
        if (status == DORA_OK && count >= state->byte_position) {
            const uint64_t remaining = count - state->byte_position;
            if (required < preferred && remaining < preferred - required) {
                preferred = (size_t)remaining + required;
            }
        }
    }
    return preferred < required ? required : preferred;
}

int dora_text_input_refill(struct text_input* state, bool* out_blocked) {
    if (!state->should_retain && state->cursor_byte_offset == state->used_bytes && state->head != nullptr) {
        input_segment_next(state);
        *out_blocked = false;
        return DORA_OK;
    }
    if (state->should_retain && state->replay_byte_limit != 0 && state->byte_position >= state->replay_byte_limit) {
        return ENOBUFS;
    }
    if (state->cursor_byte_offset != 0) {
        const size_t prefix = state->cursor_byte_offset;
        const size_t tail = state->used_bytes - prefix;
        if (state->should_retain) {
            struct text_input_segment* segment = nullptr;
            int status = input_segment_create(state, &segment);
            struct dora_allocation next = {};
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_capacity(state->pool,
                                                               tail + 1,
                                                               input_preference(state, tail + 1),
                                                               &next);
            }
            if (status != DORA_OK) {
                if (segment != nullptr) {
                    struct dora_allocation allocation = segment->allocation;
                    dora_allocation_release(&allocation);
                }
                return status;
            }
            if (tail != 0) {
                memcpy(next.address, (uint8_t*)state->buffer.address + prefix, tail);
            }
            segment->buffer = state->buffer;
            segment->valid_bytes = prefix;
            input_segment_append(state, segment);
            state->buffer = next;
        } else if (tail != 0) {
            memmove(state->buffer.address, (uint8_t*)state->buffer.address + prefix, tail);
        }
        state->scan_byte_offset -= prefix;
        state->used_bytes = tail;
        state->cursor_byte_offset = 0;
    }
    size_t normal = 0;
    const int rounded = dora_memory_preferred_capacity(state->pool, 0, state->window_bytes, &normal);
    if (rounded == EOVERFLOW) {
        normal = SIZE_MAX;
    } else if (rounded != DORA_OK) {
        return rounded;
    }
    if (!state->should_retain && state->head == nullptr && state->buffer.capacity_bytes > normal &&
        state->used_bytes < normal) {
        struct dora_allocation normal_buffer = {};
        const int status = dora_memory_payload_allocate_capacity(
            state->pool,
            state->used_bytes + 1,
            state->window_bytes > state->used_bytes ? state->window_bytes : state->used_bytes + 1,
            &normal_buffer);
        if (status == DORA_OK) {
            if (state->used_bytes != 0) {
                memcpy(normal_buffer.address, state->buffer.address, state->used_bytes);
            }
            dora_allocation_release(&state->buffer);
            state->buffer = normal_buffer;
        } else if (status != ENOMEM) {
            return status;
        }
    }
    if (state->used_bytes == state->buffer.capacity_bytes) {
        size_t required;
        if (ckd_add(&required, state->used_bytes, (size_t)1)) {
            return EOVERFLOW;
        }
        const int status = state->buffer.address == nullptr
                               ? dora_memory_payload_allocate_capacity(state->pool,
                                                                       required,
                                                                       input_preference(state, required),
                                                                       &state->buffer)
                               : dora_memory_payload_reserve(state->pool, required, state->used_bytes, &state->buffer);
        if (status != DORA_OK) {
            return status;
        }
    }
    size_t available = state->buffer.capacity_bytes - state->used_bytes;
    if (state->should_retain && state->replay_byte_limit != 0) {
        const size_t remaining = state->replay_byte_limit - (size_t)state->byte_position;
        available = available < remaining ? available : remaining;
    }
    if (available > UINT64_MAX - state->byte_position) {
        return EOVERFLOW;
    }
    size_t count = 0;
    bool end = false;
    int status =
        state->input->read(state->input, (uint8_t*)state->buffer.address + state->used_bytes, available, &count, &end);
    if (status == DORA_OK && count > available) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        state->used_bytes += count;
        state->byte_position += count;
        state->is_end = end;
        *out_blocked = count == 0 && !end;
    }
    return status;
}

int dora_text_reader_prepare(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema* schema,
                             uint64_t row_capacity,
                             enum dora_failure_policy policy) {
    uint64_t previous_rows = 0;
    if (builder->handle.slot_index != 0) {
        if (row_capacity >= TEXT_DEFAULT_ROW_CAPACITY) {
            return DORA_OK;
        }
        const int status = dora_builder_clear(builder);
        if (status != EBUSY) {
            return status;
        }
        struct builder_state* previous = nullptr;
        const int resolved = dora_builder_resolve_state(builder, &previous);
        if (resolved != DORA_OK) {
            return resolved;
        }
        previous_rows = previous->nodes[0].rows.row_count;
        dora_builder_release(builder);
    }
    const struct dora_builder_options options = {
        .row_capacity = row_capacity < TEXT_DEFAULT_ROW_CAPACITY ? TEXT_DEFAULT_ROW_CAPACITY : row_capacity,
        .content_policy = policy
    };
    int status = dora_builder_init_schema(builder, pool, schema, &options);
    struct dora_schema_field field = {};
    if (status == DORA_OK && previous_rows != 0) {
        status = dora_schema_get_field(schema, &field);
        for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
            struct dora_builder child = {};
            status = dora_builder_get_child(builder, index, &child);
            if (status == DORA_OK) {
                status = dora_builder_reserve(&child, previous_rows, 0);
                if (status == ENOMEM) {
                    status = DORA_OK;
                }
            }
        }
    }
    return status;
}

int dora_text_reader_publish(struct dora_builder* builder,
                             struct dora_vector* block,
                             uint64_t row_capacity,
                             struct dora_vector* out_vector,
                             struct dora_stream_result* out_result) {
    if (row_capacity >= TEXT_DEFAULT_ROW_CAPACITY) {
        const int status = dora_builder_finish_batch(builder, out_vector);
        if (status == DORA_OK) {
            *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY};
        }
        return status;
    }
    int status = block->handle.slot_index == 0 ? dora_builder_share(builder, block) : DORA_OK;
    if (status != DORA_OK) {
        return status;
    }
    const uint64_t count = block->rows.length < row_capacity ? block->rows.length : row_capacity;
    struct dora_vector slice = {};
    status = dora_vector_slice(block, (struct dora_range){.length = count}, &slice);
    if (status == DORA_OK) {
        status = dora_vector_retain(&slice, out_vector);
    }
    if (status == DORA_OK) {
        block->rows.offset += count;
        block->rows.length -= count;
        if (block->rows.length == 0) {
            dora_vector_release(block);
        }
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY};
    }
    return status;
}
