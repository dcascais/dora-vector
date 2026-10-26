// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/io/byte.h"

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t BYTE_SINK_BUFFER_BYTES = 512;

//------------------------------------------------------------------------------
//
// Memory Callbacks
//
//------------------------------------------------------------------------------

static int byte_memory_read_at(struct dora_byte_source* source,
                               uint64_t byte_offset,
                               uint8_t* out_bytes,
                               size_t capacity_bytes,
                               size_t* out_byte_count) {
    if (source == nullptr || out_byte_count == nullptr || (capacity_bytes != 0 && out_bytes == nullptr)) {
        return EINVAL;
    }
    const struct dora_byte_source_memory* const state = (const struct dora_byte_source_memory*)source;
    if ((state->bytes.byte_count != 0 && state->bytes.data == nullptr) || byte_offset > state->bytes.byte_count) {
        return EINVAL;
    }
    const size_t remaining = state->bytes.byte_count - (size_t)byte_offset;
    const size_t count = capacity_bytes < remaining ? capacity_bytes : remaining;
    if (count != 0) {
        memmove(out_bytes, state->bytes.data + (size_t)byte_offset, count);
    }
    *out_byte_count = count;
    return DORA_OK;
}

static int byte_memory_read(struct dora_byte_source* source,
                            uint8_t* out_bytes,
                            size_t capacity_bytes,
                            size_t* out_byte_count,
                            bool* out_is_end) {
    if (source == nullptr || out_is_end == nullptr) {
        return EINVAL;
    }
    struct dora_byte_source_memory* const state = (struct dora_byte_source_memory*)source;
    size_t count = 0;
    if (out_byte_count == nullptr) {
        return EINVAL;
    }
    const int status = byte_memory_read_at(source, state->byte_position, out_bytes, capacity_bytes, &count);
    if (status != DORA_OK) {
        return status;
    }
    state->byte_position += count;
    *out_byte_count = count;
    *out_is_end = state->byte_position == state->bytes.byte_count;
    return DORA_OK;
}

static int byte_memory_get_byte_count(struct dora_byte_source* source, uint64_t* out_byte_count) {
    if (source == nullptr || out_byte_count == nullptr) {
        return EINVAL;
    }
    *out_byte_count = ((const struct dora_byte_source_memory*)source)->bytes.byte_count;
    return DORA_OK;
}

static void byte_memory_source_release(struct dora_byte_source* source) {
    if (source != nullptr) {
        *(struct dora_byte_source_memory*)source = (struct dora_byte_source_memory){};
    }
}

static int byte_memory_write(struct dora_byte_sink* sink, struct dora_bytes bytes) {
    if (sink == nullptr || (bytes.byte_count != 0 && bytes.data == nullptr)) {
        return EINVAL;
    }
    struct dora_byte_sink_memory* const state = (struct dora_byte_sink_memory*)sink;
    if (state->buffer.address == nullptr || state->buffer.pool.generation == 0 ||
        state->byte_count > state->buffer.capacity_bytes) {
        return EINVAL;
    }
    if (bytes.byte_count == 0) {
        return DORA_OK;
    }
    size_t required;
    if (ckd_add(&required, state->byte_count, bytes.byte_count)) {
        return EOVERFLOW;
    }
    const uintptr_t source_address = (uintptr_t)bytes.data;
    const uintptr_t buffer_address = (uintptr_t)state->buffer.address;
    const bool is_buffered =
        source_address >= buffer_address && source_address - buffer_address < state->buffer.capacity_bytes;
    const size_t source_offset = is_buffered ? (size_t)(source_address - buffer_address) : 0;
    if (is_buffered && (source_offset > state->byte_count || bytes.byte_count > state->byte_count - source_offset)) {
        return EINVAL;
    }
    const int status = dora_allocation_reserve(state->buffer.pool, required, state->byte_count, &state->buffer);
    if (status != DORA_OK) {
        return status;
    }
    const uint8_t* const source_bytes =
        is_buffered ? (const uint8_t*)state->buffer.address + source_offset : bytes.data;
    memmove((uint8_t*)state->buffer.address + state->byte_count, source_bytes, bytes.byte_count);
    state->byte_count = required;
    return DORA_OK;
}

static int byte_memory_flush(struct dora_byte_sink* sink) {
    return sink == nullptr ? EINVAL : DORA_OK;
}

static void byte_memory_sink_release(struct dora_byte_sink* sink) {
    if (sink != nullptr) {
        struct dora_byte_sink_memory* const state = (struct dora_byte_sink_memory*)sink;
        dora_allocation_release(&state->buffer);
        *state = (struct dora_byte_sink_memory){};
    }
}

//------------------------------------------------------------------------------
//
// Window State
//
//------------------------------------------------------------------------------

static bool byte_window_is_empty(const struct dora_byte_window* window) {
    return window->source == nullptr && window->buffer.address == nullptr && window->buffer.capacity_bytes == 0 &&
           window->buffer.pool.index == 0 && window->buffer.pool.generation == 0 && window->begin_byte_offset == 0 &&
           window->end_byte_offset == 0 && !window->is_end;
}

static bool byte_window_is_valid(const struct dora_byte_window* window) {
    return window->source != nullptr && window->source->read != nullptr && window->buffer.address != nullptr &&
           window->buffer.capacity_bytes != 0 && window->buffer.pool.generation != 0 &&
           window->begin_byte_offset <= window->end_byte_offset &&
           window->end_byte_offset <= window->buffer.capacity_bytes;
}

//------------------------------------------------------------------------------
//
// Memory Transport
//
//------------------------------------------------------------------------------

int dora_byte_source_memory_init(struct dora_byte_source_memory* source, struct dora_bytes bytes) {
    if (source == nullptr || source->source.read != nullptr || source->source.read_at != nullptr ||
        source->source.get_byte_count != nullptr || source->source.release != nullptr ||
        source->bytes.data != nullptr || source->bytes.byte_count != 0 || source->byte_position != 0 ||
        (bytes.byte_count != 0 && bytes.data == nullptr)) {
        return EINVAL;
    }
    *source = (struct dora_byte_source_memory){
        .source =
            {.read = byte_memory_read,
             .read_at = byte_memory_read_at,
             .get_byte_count = byte_memory_get_byte_count,
             .release = byte_memory_source_release},
        .bytes = bytes
    };
    return DORA_OK;
}

int dora_byte_sink_memory_init(struct dora_byte_sink_memory* sink, struct dora_pool pool) {
    if (sink == nullptr || sink->sink.write != nullptr || sink->sink.flush != nullptr ||
        sink->sink.release != nullptr || sink->buffer.address != nullptr || sink->buffer.capacity_bytes != 0 ||
        sink->buffer.pool.index != 0 || sink->buffer.pool.generation != 0 || sink->byte_count != 0) {
        return EINVAL;
    }
    struct dora_allocation buffer = {};
    const int status = dora_pool_allocate(pool, BYTE_SINK_BUFFER_BYTES, &buffer);
    if (status != DORA_OK) {
        return status;
    }
    *sink = (struct dora_byte_sink_memory){
        .sink = {.write = byte_memory_write, .flush = byte_memory_flush, .release = byte_memory_sink_release},
        .buffer = buffer
    };
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Window Lifecycle
//
//------------------------------------------------------------------------------

int dora_byte_window_init(struct dora_byte_window* window,
                          struct dora_pool pool,
                          struct dora_byte_source* source,
                          size_t capacity_bytes) {
    if (window == nullptr || !byte_window_is_empty(window) || source == nullptr || source->read == nullptr ||
        capacity_bytes == 0) {
        return EINVAL;
    }
    struct dora_allocation buffer = {};
    const int status = dora_pool_allocate(pool, capacity_bytes, &buffer);
    if (status != DORA_OK) {
        return status;
    }
    *window = (struct dora_byte_window){.source = source, .buffer = buffer};
    return DORA_OK;
}

int dora_byte_window_move(struct dora_byte_window* in_out_source, struct dora_byte_window* out_destination) {
    if (in_out_source == nullptr || out_destination == nullptr || !byte_window_is_empty(out_destination) ||
        (!byte_window_is_empty(in_out_source) && !byte_window_is_valid(in_out_source))) {
        return EINVAL;
    }
    const uintptr_t source_address = (uintptr_t)in_out_source;
    const uintptr_t destination_address = (uintptr_t)out_destination;
    const uintptr_t distance = source_address <= destination_address ? destination_address - source_address
                                                                     : source_address - destination_address;
    if (distance < sizeof(*in_out_source)) {
        return EINVAL;
    }
    const int status = dora_allocation_move(&in_out_source->buffer, &out_destination->buffer);
    if (status != DORA_OK) {
        return status;
    }
    out_destination->source = in_out_source->source;
    out_destination->begin_byte_offset = in_out_source->begin_byte_offset;
    out_destination->end_byte_offset = in_out_source->end_byte_offset;
    out_destination->is_end = in_out_source->is_end;
    *in_out_source = (struct dora_byte_window){};
    return DORA_OK;
}

void dora_byte_window_release(struct dora_byte_window* window) {
    if (window != nullptr) {
        dora_allocation_release(&window->buffer);
        *window = (struct dora_byte_window){};
    }
}

//------------------------------------------------------------------------------
//
// Window Access
//
//------------------------------------------------------------------------------

int dora_byte_window_peek(const struct dora_byte_window* window, struct dora_bytes* out_bytes) {
    if (window == nullptr || out_bytes == nullptr || !byte_window_is_valid(window)) {
        return EINVAL;
    }
    const size_t byte_count = window->end_byte_offset - window->begin_byte_offset;
    const uint8_t* const bytes =
        byte_count == 0 ? nullptr : (const uint8_t*)window->buffer.address + window->begin_byte_offset;
    *out_bytes = (struct dora_bytes){.data = bytes, .byte_count = byte_count};
    return DORA_OK;
}

int dora_byte_window_consume(struct dora_byte_window* window, size_t byte_count) {
    if (window == nullptr || !byte_window_is_valid(window) ||
        byte_count > window->end_byte_offset - window->begin_byte_offset) {
        return EINVAL;
    }
    window->begin_byte_offset += byte_count;
    if (window->begin_byte_offset == window->end_byte_offset) {
        window->begin_byte_offset = 0;
        window->end_byte_offset = 0;
    }
    return DORA_OK;
}

int dora_byte_window_refill(struct dora_byte_window* window, bool* out_is_end) {
    if (window == nullptr || out_is_end == nullptr || !byte_window_is_valid(window)) {
        return EINVAL;
    }
    if (window->is_end || window->end_byte_offset - window->begin_byte_offset == window->buffer.capacity_bytes) {
        *out_is_end = window->is_end;
        return DORA_OK;
    }
    const size_t remaining = window->end_byte_offset - window->begin_byte_offset;
    if (window->begin_byte_offset != 0 && remaining != 0) {
        memmove(window->buffer.address, (const uint8_t*)window->buffer.address + window->begin_byte_offset, remaining);
    }
    window->begin_byte_offset = 0;
    window->end_byte_offset = remaining;
    const size_t available = window->buffer.capacity_bytes - remaining;
    size_t byte_count = 0;
    bool is_end = false;
    const int status = window->source->read(window->source,
                                            (uint8_t*)window->buffer.address + remaining,
                                            available,
                                            &byte_count,
                                            &is_end);
    if (status != DORA_OK) {
        return status;
    }
    if (byte_count > available) {
        return EINVAL;
    }
    window->end_byte_offset += byte_count;
    window->is_end = is_end;
    *out_is_end = is_end;
    return DORA_OK;
}
