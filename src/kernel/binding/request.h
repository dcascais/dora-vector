// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Kernel binding requests.

#ifndef DORA_KERNEL_BINDING_REQUEST_H
#define DORA_KERNEL_BINDING_REQUEST_H

#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Requests
//
//------------------------------------------------------------------------------

// Borrows types for the binding lifetime.
struct kernel_bind_request {
    const struct dora_logical_type* input_type;
    const struct dora_logical_type* right_type;
    const struct dora_logical_type* upper_type;
    const struct dora_logical_type* result_type;
    enum dora_failure_policy policy;
    enum dora_rounding_mode rounding;
    enum dora_compare_op comparison;
    enum dora_null_order null_order;
    uint8_t range_bounds;
    uint64_t seed;
};

#endif
