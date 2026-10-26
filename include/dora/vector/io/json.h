// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// JSON readers, writers and inference.

#ifndef DORA_VECTOR_IO_JSON_H
#define DORA_VECTOR_IO_JSON_H

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

enum dora_json_format : uint8_t {
    DORA_JSON_FORMAT_NEWLINE_DELIMITED,
    DORA_JSON_FORMAT_ARRAY
};

struct dora_json_options {
    /// Record framing.
    enum dora_json_format format;
    /// Maximum batch rows; zero selects 2,048.
    uint64_t row_capacity;
    /// ERROR, NULL or TRUST; TRUST skips content checks, not parsing.
    enum dora_failure_policy content_policy;
    /// Maximum encoded record bytes; zero selects 16 MiB.
    size_t max_record_bytes;
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

/// Copies options; borrows exclusive input without reading.
int dora_json_schema_inference_init(struct dora_schema_inference* inference,
                                    struct dora_pool pool,
                                    struct dora_byte_source* input,
                                    const struct dora_json_options* options,
                                    const struct dora_schema_inference_options* inference_options,
                                    struct dora_error* out_error);

int dora_json_reader_init(struct dora_reader* reader,
                          struct dora_pool pool,
                          const struct dora_schema* schema,
                          struct dora_byte_source* input,
                          const struct dora_json_options* options,
                          struct dora_error* out_error);

int dora_json_writer_init(struct dora_writer* writer,
                          struct dora_pool pool,
                          struct dora_byte_sink* output,
                          const struct dora_json_options* options,
                          struct dora_error* out_error);

#endif
