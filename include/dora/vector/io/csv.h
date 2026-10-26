// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// CSV readers, writers and inference.

#ifndef DORA_VECTOR_IO_CSV_H
#define DORA_VECTOR_IO_CSV_H

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Format Options
//
//------------------------------------------------------------------------------

struct dora_csv_options {
    /// Field delimiter byte.
    uint8_t delimiter;
    /// Quote byte.
    uint8_t quote;
    /// Quote escape byte.
    uint8_t escape;
    /// Whether the first record contains field names.
    bool has_header;
    /// Bytes denoting a null field.
    struct dora_bytes null_token;
    /// Maximum batch rows; zero selects 2,048.
    uint64_t row_capacity;
    /// ERROR, NULL or TRUST; TRUST skips content checks, not parsing.
    enum dora_failure_policy content_policy;
    /// Maximum encoded record bytes; zero selects input_window_bytes or 2,000,000.
    size_t max_record_bytes;
    /// Preferred input window bytes; zero selects 16 * max(max_record_bytes, 2,000,000).
    size_t input_window_bytes;
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

/// Copies options; borrows exclusive input without reading.
int dora_csv_schema_inference_init(struct dora_schema_inference* inference,
                                   struct dora_pool pool,
                                   struct dora_byte_source* input,
                                   const struct dora_csv_options* options,
                                   const struct dora_schema_inference_options* inference_options,
                                   struct dora_error* out_error);

int dora_csv_reader_init(struct dora_reader* reader,
                         struct dora_pool pool,
                         const struct dora_schema* schema,
                         struct dora_byte_source* input,
                         const struct dora_csv_options* options,
                         struct dora_error* out_error);

int dora_csv_writer_init(struct dora_writer* writer,
                         struct dora_pool pool,
                         struct dora_byte_sink* output,
                         const struct dora_csv_options* options,
                         struct dora_error* out_error);

#endif
