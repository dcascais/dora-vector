// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Statistics and segment pruning.

#ifndef DORA_VECTOR_STATISTICS_H
#define DORA_VECTOR_STATISTICS_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Statistics and Pruning
//
//------------------------------------------------------------------------------

//--------------------------------------
// Requests
//--------------------------------------

enum dora_statistics_kind : uint16_t {
    DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH = 1,
    DORA_STATISTICS_KIND_DISTINCT_COUNT = 2,
    DORA_STATISTICS_KIND_MAX_BYTE_WIDTH = 4,
    DORA_STATISTICS_KIND_MAX = 8,
    DORA_STATISTICS_KIND_MIN = 16,
    DORA_STATISTICS_KIND_NULL_COUNT = 32,
    DORA_STATISTICS_KIND_ROW_COUNT = 64,
    DORA_STATISTICS_KIND_MIN_BYTE_WIDTH = 128,
    DORA_STATISTICS_KIND_NAN_COUNT = 256
};

struct dora_statistics_request {
    /// Target schema node.
    uint32_t schema_node_index;
    /// dora_statistics_kind bitmask.
    uint16_t kinds;
    /// Maximum encoded bound bytes.
    uint32_t max_bound_bytes;
};

//--------------------------------------
// Pruning Results
//--------------------------------------

enum dora_pruning_result : uint8_t {
    /// Statistics prove neither verdict.
    DORA_PRUNING_RESULT_MAY_MATCH,
    /// No target value matches.
    DORA_PRUNING_RESULT_CANNOT_MATCH,
    /// Every nonnull target value matches; ancestor and row nulls remain excluded.
    DORA_PRUNING_RESULT_ALL_MATCH
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

/// Allocates in the target pool.
int dora_statistics_schema_init(struct dora_schema* schema,
                                const struct dora_schema* target,
                                const struct dora_statistics_request* requests,
                                uint32_t request_count);

/// Allocates in the source pool.
int dora_statistics_compute(const struct dora_vector* source,
                            const struct dora_statistics_request* requests,
                            uint32_t request_count,
                            struct dora_vector* out_result);

/// Allocates in the left pool; right may belong to another pool.
int dora_statistics_merge(const struct dora_vector* left,
                          const struct dora_vector* right,
                          struct dora_vector* out_result);

//------------------------------------------------------------------------------
//
// Pruning
//
//------------------------------------------------------------------------------

/// Publishes one verdict per segment; failure preserves out_results; charges scratch to the operand pool.
int dora_statistics_prune(const struct dora_vector* segments,
                          uint32_t segment_count,
                          uint32_t schema_node_index,
                          enum dora_compare_op op,
                          const struct dora_vector* operand,
                          enum dora_pruning_result* out_results);

/// Prunes by absolute value; failure preserves out_results; charges scratch to the operand pool.
int dora_statistics_prune_abs(const struct dora_vector* segments,
                              uint32_t segment_count,
                              uint32_t schema_node_index,
                              enum dora_compare_op op,
                              const struct dora_vector* operand,
                              enum dora_pruning_result* out_results);

#endif
