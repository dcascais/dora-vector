// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Elementwise kernel bindings.

#ifndef DORA_KERNEL_BINDING_ELEMENTWISE_H
#define DORA_KERNEL_BINDING_ELEMENTWISE_H

#include "dora/vector/types.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "numeric/decimal.h"
#include "types/text.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Requests
//
//------------------------------------------------------------------------------

enum kernel_operation : uint8_t {
    KERNEL_OPERATION_ADD,
    KERNEL_OPERATION_SUBTRACT,
    KERNEL_OPERATION_MULTIPLY,
    KERNEL_OPERATION_DIVIDE,
    KERNEL_OPERATION_REMAINDER,
    KERNEL_OPERATION_NEGATE,
    KERNEL_OPERATION_ABS,
    KERNEL_OPERATION_COUNT
};

enum kernel_boolean_operation : uint8_t {
    KERNEL_BOOLEAN_AND,
    KERNEL_BOOLEAN_OR,
    KERNEL_BOOLEAN_XOR,
    KERNEL_BOOLEAN_NOT
};

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

struct kernel_decimal_rescale {
    struct dora_i128 factor;
    struct dora_i128 limit;
};

struct kernel_cast_facts {
    struct decimal_scale_pair scale;
    struct kernel_decimal_rescale rescale;
    typeof(dora_type_parse_select(nullptr, DORA_PHYSICAL_TYPE_I32)) parse;
    int64_t ticks_per_day;
    size_t maximum_text_byte_count;
};

struct kernel_unary_binding {
    int (*fill)(const struct kernel_packet* source,
                const struct kernel_unary_binding* binding,
                struct kernel_result* out_result);
    struct kernel_bind_request request;
    struct dora_logical_type result_type;
    enum dora_physical_type input_physical;
    enum dora_physical_type result_physical;
    struct decimal_prepared input_decimal;
    struct decimal_prepared result_decimal;
    struct kernel_cast_facts cast;
    uint32_t input_byte_width;
    uint32_t result_byte_width;
    bool is_result_all_valid;
    bool can_fail;
};

struct kernel_binary_binding {
    int (*fill)(const struct kernel_packet* left,
                const struct kernel_packet* right,
                const struct kernel_binary_binding* binding,
                struct kernel_result* out_result);
    struct kernel_bind_request request;
    struct dora_logical_type result_type;
    enum dora_physical_type input_physical;
    enum dora_physical_type right_physical;
    enum dora_physical_type result_physical;
    struct decimal_prepared left_decimal;
    struct decimal_prepared right_decimal;
    struct decimal_prepared result_decimal;
    int64_t ticks_per_day;
    uint32_t left_byte_width;
    uint32_t right_byte_width;
    uint32_t result_byte_width;
    bool is_result_all_valid;
    bool can_fail;
};

struct kernel_ternary_binding {
    int (*fill)(const struct kernel_packet* left,
                const struct kernel_packet* right,
                const struct kernel_packet* addend,
                const struct kernel_ternary_binding* binding,
                void* product_scratch,
                struct kernel_result* out_result);
    struct kernel_binary_binding product;
    struct kernel_binary_binding sum;
    uint32_t scratch_bytes;
};

struct kernel_range_binding {
    int (*fill)(const struct kernel_packet* source,
                const struct kernel_packet* lower,
                const struct kernel_packet* upper,
                const struct kernel_range_binding* binding,
                struct kernel_result* out_result);
    struct kernel_binary_binding lower;
    struct kernel_binary_binding upper;
};

//------------------------------------------------------------------------------
//
// Bind Functions
//
//------------------------------------------------------------------------------

int dora_kernel_arithmetic_bind(enum kernel_operation operation,
                                const struct kernel_bind_request* request,
                                struct kernel_binary_binding* out_binding);

int dora_kernel_unary_arithmetic_bind(enum kernel_operation operation,
                                      const struct kernel_bind_request* request,
                                      struct kernel_unary_binding* out_binding);

int dora_kernel_multiply_add_bind(const struct kernel_bind_request* request,
                                  struct kernel_ternary_binding* out_binding);

int dora_kernel_boolean_bind(enum kernel_boolean_operation operation, struct kernel_binary_binding* out_binding);

int dora_kernel_compare_bind(const struct kernel_bind_request* request, struct kernel_binary_binding* out_binding);

int dora_kernel_compare_distinct_bind(const struct kernel_bind_request* request,
                                      struct kernel_binary_binding* out_binding);

int dora_kernel_order_bind(const struct kernel_bind_request* request, struct kernel_binary_binding* out_binding);

int dora_kernel_range_bind(const struct kernel_bind_request* request, struct kernel_range_binding* out_binding);

int dora_kernel_temporal_bind(enum kernel_operation operation,
                              const struct dora_logical_type* left_type,
                              const struct dora_logical_type* right_type,
                              enum dora_failure_policy policy,
                              struct kernel_binary_binding* out_binding);

int dora_kernel_hash_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding);

int dora_kernel_hash_combine_bind(const struct kernel_bind_request* request, struct kernel_binary_binding* out_binding);

int dora_kernel_cast_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding);

// Writes Int64 byte lengths.
// Utf8, Binary, FixedBinary
int dora_kernel_cast_measure_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding);

#endif
