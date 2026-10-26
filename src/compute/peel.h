// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Dictionary and run-end computation reuse.

#ifndef DORA_COMPUTE_PEEL_H
#define DORA_COMPUTE_PEEL_H

#include "dora/vector/handle.h"
#include "vector/image.h"

//------------------------------------------------------------------------------
//
// Binding and Publication
//
//------------------------------------------------------------------------------

// Unused encoded values must not introduce failures.
// All logical types; layouts: Dictionary, RunEnd
int dora_compute_peel_values(const struct dora_vector* source,
                             const struct vector_bound_input* input,
                             bool error_can_fail,
                             struct dora_vector* out_values,
                             bool* out_applied);

int dora_compute_peel_publish(struct dora_pool pool,
                              const struct dora_vector* source,
                              const struct vector_bound_input* input,
                              const struct dora_vector* transformed,
                              struct dora_vector* out_result);

#endif
