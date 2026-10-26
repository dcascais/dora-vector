// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Exact statistics bounds.

#ifndef DORA_STATISTICS_BOUNDS_H
#define DORA_STATISTICS_BOUNDS_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "vector/vector_internal.h"

struct statistics_bounds {
    union decoded_row_value minimum;
    union decoded_row_value maximum;
};

bool dora_statistics_try_get_exact_bounds(const struct dora_vector* vector,
                                          const struct dora_logical_type* type,
                                          struct statistics_bounds* out_bounds);

#endif
