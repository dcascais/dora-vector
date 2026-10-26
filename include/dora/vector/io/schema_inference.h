// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Incremental schema inference.

#ifndef DORA_VECTOR_IO_SCHEMA_INFERENCE_H
#define DORA_VECTOR_IO_SCHEMA_INFERENCE_H

#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"

#include <stddef.h>
#include <stdint.h>

struct dora_schema_inference_options {
    /// Sample records; zero selects 20,480, UINT64_MAX samples through EOF.
    uint64_t sample_row_limit;
    /// Maximum retained input bytes; zero adds no limit beyond the pool.
    size_t max_replay_bytes;
};

/// Publishes schema and reader once on READY; borrows input until reader release or failure cleanup.
int dora_schema_inference_next(struct dora_schema_inference* inference,
                               struct dora_schema* out_schema,
                               struct dora_reader* out_reader,
                               struct dora_stream_result* out_result,
                               struct dora_error* out_error);

int dora_schema_inference_move(struct dora_schema_inference* in_out_source,
                               struct dora_schema_inference* out_destination);

/// Frees inference state without releasing borrowed input.
void dora_schema_inference_release(struct dora_schema_inference* inference);

bool dora_schema_inference_is_valid(const struct dora_schema_inference* inference);

#endif
