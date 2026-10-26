// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Byte sources, sinks and windows.

#ifndef DORA_VECTOR_IO_BYTE_H
#define DORA_VECTOR_IO_BYTE_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Capabilities
//
//------------------------------------------------------------------------------

//--------------------------------------
// Source
//--------------------------------------

/// Pool reset and release return EBUSY during active reader callbacks.
struct dora_byte_source {
    /// Reads at the current position and reports end-of-input.
    int (*read)(struct dora_byte_source* source,
                uint8_t* out_bytes,
                size_t capacity_bytes,
                size_t* out_byte_count,
                bool* out_is_end);
    /// Reads at byte_offset without changing the current position; optional.
    int (*read_at)(struct dora_byte_source* source,
                   uint64_t byte_offset,
                   uint8_t* out_bytes,
                   size_t capacity_bytes,
                   size_t* out_byte_count);
    /// Returns the source length in bytes; optional.
    int (*get_byte_count)(struct dora_byte_source* source, uint64_t* out_byte_count);
    /// Frees source state.
    void (*release)(struct dora_byte_source* source);
};

//--------------------------------------
// Sink
//--------------------------------------

/// Pool reset and release return EBUSY during active writer callbacks.
struct dora_byte_sink {
    /// Writes borrowed bytes.
    int (*write)(struct dora_byte_sink* sink, struct dora_bytes bytes);
    /// Flushes pending bytes.
    int (*flush)(struct dora_byte_sink* sink);
    /// Frees sink state.
    void (*release)(struct dora_byte_sink* sink);
};

//------------------------------------------------------------------------------
//
// Memory Implementations
//
//------------------------------------------------------------------------------

//--------------------------------------
// Source
//--------------------------------------

struct dora_byte_source_memory {
    /// Source callbacks.
    struct dora_byte_source source;
    /// Borrowed input bytes.
    struct dora_bytes bytes;
    /// Current byte position.
    size_t byte_position;
};

int dora_byte_source_memory_init(struct dora_byte_source_memory* source, struct dora_bytes bytes);

//--------------------------------------
// Sink
//--------------------------------------

struct dora_byte_sink_memory {
    /// Sink callbacks.
    struct dora_byte_sink sink;
    /// Owned output storage.
    struct dora_allocation buffer;
    /// Initialized output bytes.
    size_t byte_count;
};

int dora_byte_sink_memory_init(struct dora_byte_sink_memory* sink, struct dora_pool pool);

//------------------------------------------------------------------------------
//
// Byte Window
//
//------------------------------------------------------------------------------

struct dora_byte_window {
    /// Borrowed byte source.
    struct dora_byte_source* source;
    /// Owned window storage.
    struct dora_allocation buffer;
    /// First unread buffer byte.
    size_t begin_byte_offset;
    /// One past the last buffered byte.
    size_t end_byte_offset;
    /// Whether the source reached end-of-input.
    bool is_end;
};

int dora_byte_window_init(struct dora_byte_window* window,
                          struct dora_pool pool,
                          struct dora_byte_source* source,
                          size_t capacity_bytes);

int dora_byte_window_peek(const struct dora_byte_window* window, struct dora_bytes* out_bytes);

int dora_byte_window_consume(struct dora_byte_window* window, size_t byte_count);

int dora_byte_window_refill(struct dora_byte_window* window, bool* out_is_end);

int dora_byte_window_move(struct dora_byte_window* in_out_source, struct dora_byte_window* out_destination);

void dora_byte_window_release(struct dora_byte_window* window);

#endif
