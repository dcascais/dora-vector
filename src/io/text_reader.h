// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Buffered text input and batch publication.

#ifndef DORA_IO_TEXT_READER_H
#define DORA_IO_TEXT_READER_H

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/memory.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

struct text_record_result {
    size_t end_byte_offset;
    size_t next_byte_offset;
    bool is_blocked;
    bool is_end_of_stream;
};

struct text_input_segment {
    struct dora_allocation allocation;
    struct dora_allocation buffer;
    struct text_input_segment* next;
    size_t valid_bytes;
    bool is_end;
};

struct text_input {
    struct dora_pool pool;
    struct dora_allocation buffer;
    struct dora_byte_source* input;
    struct text_input_segment* head;
    struct text_input_segment* tail;
    uint64_t byte_position;
    size_t replay_byte_limit;
    size_t window_bytes;
    size_t used_bytes;
    size_t cursor_byte_offset;
    size_t scan_byte_offset;
    bool is_end;
    bool should_retain;
};

void dora_text_input_release(struct text_input* state);
int dora_text_input_adopt(struct text_input* source, struct text_input* destination);

int dora_text_reader_prepare(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema* schema,
                             uint64_t row_capacity,
                             enum dora_failure_policy policy);

int dora_text_reader_publish(struct dora_builder* builder,
                             struct dora_vector* block,
                             uint64_t row_capacity,
                             struct dora_vector* out_vector,
                             struct dora_stream_result* out_result);

int dora_text_input_refill(struct text_input* state, bool* out_blocked);

#endif
