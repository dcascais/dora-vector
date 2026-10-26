// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/arithmetic.h"

#include "compute/output.h"
#include "compute/peel.h"
#include "compute/proof.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "numeric/failure.h"
#include "statistics/bounds.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Route
//
//------------------------------------------------------------------------------

union arithmetic_binding {
    struct kernel_binary_binding binary;
    struct kernel_unary_binding unary;
};

static int arithmetic_failure(uint8_t failure_bits) {
    if ((failure_bits & NUMERIC_FAILURE_OVERFLOW) != 0) {
        return EOVERFLOW;
    }
    if ((failure_bits & NUMERIC_FAILURE_DIVIDE_BY_ZERO) != 0) {
        return EDOM;
    }
    if ((failure_bits & NUMERIC_FAILURE_RANGE) != 0) {
        return ERANGE;
    }
    return (failure_bits & NUMERIC_FAILURE_INVALID_VALUE) != 0 ? EINVAL : DORA_OK;
}

static int arithmetic_bind(const struct vector_bound_input* inputs,
                           enum kernel_operation operation,
                           enum dora_failure_policy policy,
                           union arithmetic_binding* out_binding) {
    const bool unary = operation >= KERNEL_OPERATION_NEGATE;
    const struct dora_logical_type* const left_type = &inputs[0].nodes[0].field.type;
    const struct dora_logical_type* const right_type = unary ? nullptr : &inputs[1].nodes[0].field.type;
    const bool numeric = left_type->id >= DORA_TYPE_INT8 && left_type->id <= DORA_TYPE_DECIMAL;
    if (!unary && (!numeric || right_type->id < DORA_TYPE_INT8 || right_type->id > DORA_TYPE_DECIMAL)) {
        return dora_kernel_temporal_bind(operation, left_type, right_type, policy, &out_binding->binary);
    }
    if (!numeric || (!unary && !dora_type_is_equal(left_type, right_type, 0))) {
        return EINVAL;
    }
    const struct kernel_bind_request request = {
        .input_type = left_type,
        .right_type = right_type,
        .policy = policy,
    };
    return unary ? dora_kernel_unary_arithmetic_bind(operation, &request, &out_binding->unary)
                 : dora_kernel_arithmetic_bind(operation, &request, &out_binding->binary);
}

//--------------------------------------
// Statistics Proof
//--------------------------------------

#define PROOF_CARRIERS(X)                                                                                              \
    X(i8, int8_t, INT8)                                                                                                \
    X(i16, int16_t, INT16)                                                                                             \
    X(i32, int32_t, INT32)                                                                                             \
    X(i64, int64_t, INT64)                                                                                             \
    X(i128, struct dora_i128, INT128)                                                                                  \
    X(u8, uint8_t, UINT8)                                                                                              \
    X(u16, uint16_t, UINT16)                                                                                           \
    X(u32, uint32_t, UINT32)                                                                                           \
    X(u64, uint64_t, UINT64)                                                                                           \
    X(u128, struct dora_u128, UINT128)

// Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128
#define DEFINE_PROOF_CORNERS(S, T, ID)                                                                                 \
    static bool proof_corners_are_safe_##S(enum kernel_operation operation, const T* left, const T* right) {           \
        T value = {};                                                                                                  \
        switch (operation) {                                                                                           \
        case KERNEL_OPERATION_ADD:                                                                                     \
            return !numeric_add_checked_##S(left[0], right[0], &value) &&                                              \
                   !numeric_add_checked_##S(left[1], right[1], &value);                                                \
        case KERNEL_OPERATION_SUBTRACT:                                                                                \
            return !numeric_subtract_checked_##S(left[0], right[1], &value) &&                                         \
                   !numeric_subtract_checked_##S(left[1], right[0], &value);                                           \
        case KERNEL_OPERATION_MULTIPLY:                                                                                \
            return !numeric_multiply_checked_##S(left[0], right[0], &value) &&                                         \
                   !numeric_multiply_checked_##S(left[0], right[1], &value) &&                                         \
                   !numeric_multiply_checked_##S(left[1], right[0], &value) &&                                         \
                   !numeric_multiply_checked_##S(left[1], right[1], &value);                                           \
        case KERNEL_OPERATION_NEGATE:                                                                                  \
            return !numeric_negate_checked_##S(left[0], &value) && !numeric_negate_checked_##S(left[1], &value);       \
        case KERNEL_OPERATION_ABS:                                                                                     \
            return !numeric_abs_checked_##S(left[0], &value) && !numeric_abs_checked_##S(left[1], &value);             \
        default:                                                                                                       \
            return false;                                                                                              \
        }                                                                                                              \
    }
PROOF_CARRIERS(DEFINE_PROOF_CORNERS)
#undef DEFINE_PROOF_CORNERS

bool dora_compute_arithmetic_is_proven(enum kernel_operation operation,
                                       const struct dora_logical_type* type,
                                       const struct dora_vector* left,
                                       const struct dora_vector* right) {
    const bool unary = operation >= KERNEL_OPERATION_NEGATE;
    const bool carrier = (type->id >= DORA_TYPE_INT8 && type->id <= DORA_TYPE_INT128) ||
                         (type->id >= DORA_TYPE_UINT8 && type->id <= DORA_TYPE_UINT128);
    struct statistics_bounds bounds[2] = {};
    if (!carrier || operation == KERNEL_OPERATION_DIVIDE || operation == KERNEL_OPERATION_REMAINDER ||
        !dora_statistics_try_get_exact_bounds(left, type, &bounds[0]) ||
        !dora_statistics_try_get_exact_bounds(unary ? left : right, type, &bounds[1])) {
        return false;
    }
    switch (type->id) {
#define PROOF_CASE(S, T, ID)                                                                                           \
    case DORA_TYPE_##ID:                                                                                               \
        return proof_corners_are_safe_##S(operation,                                                                   \
                                          (const T[2]){bounds[0].minimum.S, bounds[0].maximum.S},                      \
                                          (const T[2]){bounds[1].minimum.S, bounds[1].maximum.S});
        PROOF_CARRIERS(PROOF_CASE)
#undef PROOF_CASE
    default:
        return false;
    }
}

#undef PROOF_CARRIERS

static int compute_arithmetic(struct dora_pool pool,
                              const struct dora_vector* left,
                              const struct dora_vector* right,
                              enum kernel_operation operation,
                              enum dora_failure_policy policy,
                              struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result) || policy > DORA_FAILURE_POLICY_WRAP) {
        return EINVAL;
    }
    const bool adds_nulls = policy == DORA_FAILURE_POLICY_NULL;
    const bool unary = operation >= KERNEL_OPERATION_NEGATE;
    const uint32_t input_count = unary ? 1 : 2;
    struct vector_bound_node leaves[2];
    struct vector_bound_input inputs[2] = {};
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, left, &leaves[0], &inputs[0]);
    }
    if (status == DORA_OK && !unary) {
        status = dora_vector_bind_input(&scope, right, &leaves[1], &inputs[1]);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        status = dora_vector_input_admit(&inputs[index], 0, true);
        if (status == DORA_OK && inputs[index].length != inputs[0].length) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK && policy != DORA_FAILURE_POLICY_WRAP) {
        // Temporal arithmetic must retain checked failure policy.
        const struct dora_logical_type operand = inputs[0].nodes[0].field.type;
        const bool same_type = unary || dora_type_is_equal(&operand, &inputs[1].nodes[0].field.type, 0);
        if (same_type && dora_compute_arithmetic_is_proven(operation, &operand, left, unary ? nullptr : right)) {
            policy = DORA_FAILURE_POLICY_WRAP;
        }
    }
    union arithmetic_binding binding = {};
    if (status == DORA_OK) {
        status = arithmetic_bind(inputs, operation, policy, &binding);
    }
    bool peeled = false;
    if (status == DORA_OK && unary) {
        struct dora_vector values = {};
        struct dora_vector transformed = {};
        status = dora_compute_peel_values(left,
                                          &inputs[0],
                                          policy == DORA_FAILURE_POLICY_ERROR && binding.unary.can_fail,
                                          &values,
                                          &peeled);
        if (status == DORA_OK && peeled) {
            status = compute_arithmetic(pool, &values, nullptr, operation, policy, &transformed);
        }
        if (status == DORA_OK && peeled) {
            status = dora_compute_peel_publish(pool, left, &inputs[0], &transformed, &published);
        }
        dora_vector_release(&transformed);
    }
    if (peeled || status != DORA_OK) {
        dora_scratch_scope_end(&scope);
        return vector_finish_publication(&published, out_result, status);
    }
    struct vector_operand operands[2] = {};
    bool null_constant = false;
    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        status =
            dora_vector_operand_bind(&scope, index, &inputs[index], VECTOR_OPERAND_REQUIRE_PACKET, &operands[index]);
        null_constant = null_constant || operands[index].kind == VECTOR_OPERAND_CONSTANT_NULL;
    }
    const bool constant = null_constant || (inputs[0].is_constant && (unary || inputs[1].is_constant));
    const bool all_valid = !null_constant && policy != DORA_FAILURE_POLICY_NULL &&
                           dora_vector_operands_are_all_valid(operands, input_count);
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool,
                                          unary ? &binding.unary.result_type : &binding.binary.result_type,
                                          inputs[0].length,
                                          constant,
                                          all_valid,
                                          &output);
    }
    if (status == DORA_OK && !adds_nulls) {
        status = dora_compute_output_preserve_operands_validity(&output,
                                                                (const struct dora_vector* const[]){left, right},
                                                                operands,
                                                                input_count);
    }
    if (null_constant && status == DORA_OK) {
        status = dora_compute_output_null(&output);
    }
    const uint64_t stored = output.stored;
    for (uint64_t first = 0; status == DORA_OK && !null_constant && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_packet packets[2];
        for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
            status = dora_vector_operand_prepare(&operands[index], rows, nullptr, &packets[index]);
        }
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        if (status == DORA_OK) {
            status = unary ? binding.unary.fill(&packets[0], &binding.unary, &result)
                           : binding.binary.fill(&packets[0], &packets[1], &binding.binary, &result);
        }
        if (status == DORA_OK) {
            status = arithmetic_failure(result.failure_bits);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &result);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

static int compute_multiply_add(struct dora_pool pool,
                                const struct dora_vector* const* sources,
                                enum dora_failure_policy policy,
                                struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result) || policy > DORA_FAILURE_POLICY_WRAP) {
        return EINVAL;
    }
    struct vector_bound_node leaves[3];
    struct vector_bound_input inputs[3] = {};
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    for (uint32_t index = 0; status == DORA_OK && index < 3; ++index) {
        status = dora_vector_bind_input(&scope, sources[index], &leaves[index], &inputs[index]);
        if (status == DORA_OK) {
            status = dora_vector_input_admit(&inputs[index], 0, true);
        }
        if (status == DORA_OK &&
            (inputs[index].length != inputs[0].length ||
             !dora_type_is_equal(&inputs[index].nodes[0].field.type, &inputs[0].nodes[0].field.type, 0))) {
            status = EINVAL;
        }
    }
    struct dora_logical_type type = {};
    if (status == DORA_OK) {
        type = inputs[0].nodes[0].field.type;
        if (type.id < DORA_TYPE_INT8 || type.id > DORA_TYPE_DECIMAL ||
            (policy == DORA_FAILURE_POLICY_WRAP && type.id >= DORA_TYPE_FLOAT16)) {
            status = EINVAL;
        }
    }
    struct kernel_ternary_binding binding = {};
    if (status == DORA_OK) {
        const struct kernel_bind_request request = {
            .input_type = &type,
            .right_type = &type,
            .policy = policy,
        };
        status = dora_kernel_multiply_add_bind(&request, &binding);
    }
    struct vector_operand operands[3] = {};
    bool null_constant = false;
    for (uint32_t index = 0; status == DORA_OK && index < 3; ++index) {
        status =
            dora_vector_operand_bind(&scope, index, &inputs[index], VECTOR_OPERAND_REQUIRE_PACKET, &operands[index]);
        null_constant = null_constant || operands[index].kind == VECTOR_OPERAND_CONSTANT_NULL;
    }
    const bool constant = null_constant || (inputs[0].is_constant && inputs[1].is_constant && inputs[2].is_constant);
    const bool all_valid =
        !null_constant && policy != DORA_FAILURE_POLICY_NULL && dora_vector_operands_are_all_valid(operands, 3);
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, inputs[0].length, constant, all_valid, &output);
    }
    if (status == DORA_OK && policy != DORA_FAILURE_POLICY_NULL) {
        status = dora_compute_output_preserve_operands_validity(&output, sources, operands, 3);
    }
    if (null_constant && status == DORA_OK) {
        status = dora_compute_output_null(&output);
    }
    void* scratch = nullptr;
    if (status == DORA_OK && !null_constant) {
        status = dora_scratch_slot_require(&scope,
                                           vector_operand_slot(0, VECTOR_OPERAND_SLOT_CACHE),
                                           &scratch,
                                           binding.scratch_bytes,
                                           64);
    }
    const uint64_t stored = output.stored;
    for (uint64_t first = 0; status == DORA_OK && !null_constant && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_packet packets[3];
        for (uint32_t index = 0; status == DORA_OK && index < 3; ++index) {
            status = dora_vector_operand_prepare(&operands[index], rows, nullptr, &packets[index]);
        }
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        if (status == DORA_OK) {
            status = binding.fill(&packets[0], &packets[1], &packets[2], &binding, scratch, &result);
        }
        if (status == DORA_OK) {
            status = arithmetic_failure(result.failure_bits);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &result);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Element-Wise
//
//------------------------------------------------------------------------------

//--------------------------------------
// Arithmetic
//--------------------------------------

int dora_compute_add(const struct dora_vector* left,
                     const struct dora_vector* right,
                     enum dora_failure_policy policy,
                     struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, left, right, KERNEL_OPERATION_ADD, policy, out_result);
}

int dora_compute_subtract(const struct dora_vector* left,
                          const struct dora_vector* right,
                          enum dora_failure_policy policy,
                          struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, left, right, KERNEL_OPERATION_SUBTRACT, policy, out_result);
}

int dora_compute_multiply(const struct dora_vector* left,
                          const struct dora_vector* right,
                          enum dora_failure_policy policy,
                          struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, left, right, KERNEL_OPERATION_MULTIPLY, policy, out_result);
}

int dora_compute_divide(const struct dora_vector* left,
                        const struct dora_vector* right,
                        enum dora_failure_policy policy,
                        struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, left, right, KERNEL_OPERATION_DIVIDE, policy, out_result);
}

int dora_compute_remainder(const struct dora_vector* left,
                           const struct dora_vector* right,
                           enum dora_failure_policy policy,
                           struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, left, right, KERNEL_OPERATION_REMAINDER, policy, out_result);
}

int dora_compute_multiply_add(const struct dora_vector* left,
                              const struct dora_vector* right,
                              const struct dora_vector* addend,
                              enum dora_failure_policy policy,
                              struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right, addend}, 3, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct dora_vector* const sources[3] = {left, right, addend};
    return compute_multiply_add(pool, sources, policy, out_result);
}

int dora_compute_abs(const struct dora_vector* source,
                     enum dora_failure_policy policy,
                     struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, source, nullptr, KERNEL_OPERATION_ABS, policy, out_result);
}

int dora_compute_negate(const struct dora_vector* source,
                        enum dora_failure_policy policy,
                        struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_arithmetic(pool, source, nullptr, KERNEL_OPERATION_NEGATE, policy, out_result);
}
