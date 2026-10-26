// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Reduction kernel bindings.

#ifndef DORA_KERNEL_BINDING_REDUCE_H
#define DORA_KERNEL_BINDING_REDUCE_H

#include "dora/vector/types.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "numeric/decimal.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Requests
//
//------------------------------------------------------------------------------

enum kernel_reduction : uint8_t {
    KERNEL_REDUCTION_COUNT_VALID,
    KERNEL_REDUCTION_SUM,
    KERNEL_REDUCTION_PRODUCT,
    KERNEL_REDUCTION_PRODUCT_DECIMAL,
    KERNEL_REDUCTION_MIN,
    KERNEL_REDUCTION_MAX,
    KERNEL_REDUCTION_ANY,
    KERNEL_REDUCTION_ALL,
    KERNEL_REDUCTION_NAN_COUNT,
    KERNEL_REDUCTION_BYTE_WIDTH,
    KERNEL_REDUCTION_LOWER_BOUND,
    KERNEL_REDUCTION_UPPER_BOUND
};

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

union kernel_reduction_value {
    struct dora_i256 integer;
    double floating;
    uint64_t count;
    bool boolean;
    uint8_t extremum[32];
    struct dora_bytes bytes;
};

struct kernel_reduction_state {
    union kernel_reduction_value value;
    uint64_t input_count;
    uint64_t minimum_width;
    uint64_t maximum_width;
    struct dora_u128 total_width;
    bool has_value;
};

struct kernel_reduction_binding {
    int (*update)(const struct kernel_packet* source,
                  const struct kernel_reduction_binding* binding,
                  struct kernel_reduction_state* in_out_state);
    int (*finish)(const struct kernel_reduction_binding* binding,
                  const struct kernel_reduction_state* state,
                  struct kernel_result* out_result);
    struct kernel_bind_request request;
    struct dora_logical_type result_type;
    enum dora_physical_type input_physical;
    enum dora_physical_type result_physical;
    struct decimal_prepared input_decimal;
    struct decimal_prepared result_decimal;
    enum kernel_reduction operation;
    uint32_t input_byte_width;
    uint32_t result_byte_width;
};

//------------------------------------------------------------------------------
//
// Bind Functions
//
//------------------------------------------------------------------------------

int dora_kernel_reduction_bind(enum kernel_reduction operation,
                               const struct kernel_bind_request* request,
                               struct kernel_reduction_binding* out_binding);

void dora_kernel_reduction_state_init(const struct kernel_reduction_binding* binding,
                                      struct kernel_reduction_state* out_state);

#endif
