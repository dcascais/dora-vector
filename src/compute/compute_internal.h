// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Computation in explicit pools.

#ifndef DORA_COMPUTE_COMPUTE_INTERNAL_H
#define DORA_COMPUTE_COMPUTE_INTERNAL_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Element-Wise
//
//------------------------------------------------------------------------------

int dora_compute_cast_in_pool(struct dora_pool pool,
                              const struct dora_vector* source,
                              const struct dora_logical_type* target,
                              enum dora_rounding_mode rounding_mode,
                              enum dora_failure_policy policy,
                              struct dora_vector* out_result);
int dora_compute_hash_in_pool(struct dora_pool pool,
                              const struct dora_vector* source,
                              uint64_t seed,
                              struct dora_vector* out_hashes);
int dora_compute_bytes_slice_in_pool(struct dora_pool pool,
                                     const struct dora_vector* source,
                                     int64_t byte_offset,
                                     uint64_t byte_count,
                                     bool is_text,
                                     struct dora_vector* out_result);

#endif
