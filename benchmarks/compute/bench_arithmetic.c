// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "types/array_count.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_ARRAY,
    VARIANT_ARRAY_SCALAR,
    VARIANT_BOUNDS_PROVEN,
    VARIANT_BOUNDS_UNPROVEN
};

struct inputs {
    struct dora_vector left;
    struct dora_vector right;
};

enum operation : uint8_t {
    OPERATION_ADD,
    OPERATION_MULTIPLY,
    OPERATION_DIVIDE
};

static constexpr _BitInt(128) PRODUCT_OUTLIER = INT64_C(1) << 32;

static int prepare_bounds(struct bench* bench,
                          const struct bench_args* args,
                          enum operation operation,
                          struct inputs* out_inputs) {
    struct bench_range range = {};
    int status = bench_integer_range(args->type.id, &range);
    const _BitInt(128) value = operation == OPERATION_MULTIPLY ? PRODUCT_OUTLIER : (range.maximum + 1) / 2;
    const struct bench_outlier left = {.from_row = 0, .value = value};
    const struct bench_outlier right = {.from_row = 1, .value = value};
    const bool has_outliers = args->variant == VARIANT_BOUNDS_UNPROVEN;
    if (status == 0) {
        status = bench_with_bounds(bench, args, 1, has_outliers ? &left : nullptr, &out_inputs->left);
    }
    if (status == 0) {
        status = bench_with_bounds(bench, args, 2, has_outliers ? &right : nullptr, &out_inputs->right);
    }
    return status;
}

static bool has_bounds(const struct bench_args* args) {
    return args->variant == VARIANT_BOUNDS_PROVEN || args->variant == VARIANT_BOUNDS_UNPROVEN;
}

static struct inputs prepare(struct bench* bench, const struct bench_args* args, enum operation operation) {
    struct inputs inputs = {};
    if (has_bounds(args)) {
        bench_check(bench, prepare_bounds(bench, args, operation, &inputs));
        return inputs;
    }
    bench_check(bench, bench_random(bench, args, 1, &inputs.left));
    bench_check(bench,
                args->variant == VARIANT_ARRAY_SCALAR ? bench_constant(bench, args, 2, &inputs.right)
                                                      : bench_random(bench, args, 2, &inputs.right));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

#define BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER(value_type, type_tag)                                                 \
    static int expected_##type_tag(value_type lhs, value_type rhs, enum operation operation, value_type* out_value) {  \
        if (operation == OPERATION_ADD) {                                                                              \
            return ckd_add(out_value, lhs, rhs) ? EOVERFLOW : 0;                                                       \
        }                                                                                                              \
        if (operation == OPERATION_MULTIPLY) {                                                                         \
            return ckd_mul(out_value, lhs, rhs) ? EOVERFLOW : 0;                                                       \
        }                                                                                                              \
        if (rhs == 0) {                                                                                                \
            return EDOM;                                                                                               \
        }                                                                                                              \
        *out_value = lhs / rhs;                                                                                        \
        return 0;                                                                                                      \
    }

BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER(int32_t, i32)
BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER(int64_t, i64)
BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER(uint32_t, u32)
BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER(uint64_t, u64)

#undef BENCH_ARITHMETIC_DEFINE_EXPECTED_INTEGER

#define BENCH_ARITHMETIC_DEFINE_EXPECTED_FLOAT(value_type, type_tag)                                                   \
    static int expected_##type_tag(value_type lhs, value_type rhs, enum operation operation, value_type* out_value) {  \
        *out_value = operation == OPERATION_MULTIPLY ? lhs * rhs                                                       \
                     : operation == OPERATION_DIVIDE ? lhs / rhs                                                       \
                                                     : lhs + rhs;                                                      \
        return isfinite(*out_value) ? 0 : EDOM;                                                                        \
    }

BENCH_ARITHMETIC_DEFINE_EXPECTED_FLOAT(float, f32)
BENCH_ARITHMETIC_DEFINE_EXPECTED_FLOAT(double, f64)

#undef BENCH_ARITHMETIC_DEFINE_EXPECTED_FLOAT

#define BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(value_type, type_tag)                                                    \
    static int check_##type_tag(const struct dora_vector* left,                                                        \
                                const struct dora_vector* right,                                                       \
                                uint64_t rows,                                                                         \
                                const struct dora_vector* result,                                                      \
                                enum operation operation) {                                                            \
        for (uint64_t row = 0; row < rows; ++row) {                                                                    \
            value_type lhs = {};                                                                                       \
            value_type rhs = {};                                                                                       \
            value_type actual = {};                                                                                    \
            bool lhs_null;                                                                                             \
            bool rhs_null;                                                                                             \
            bool result_null;                                                                                          \
            int status = dora_vector_read_##type_tag(left, row, &lhs, &lhs_null);                                      \
            if (status == 0) {                                                                                         \
                status = dora_vector_read_##type_tag(right, row, &rhs, &rhs_null);                                     \
            }                                                                                                          \
            if (status == 0) {                                                                                         \
                status = dora_vector_read_##type_tag(result, row, &actual, &result_null);                              \
            }                                                                                                          \
            if (status != 0) {                                                                                         \
                return status;                                                                                         \
            }                                                                                                          \
            if (result_null != (lhs_null || rhs_null)) {                                                               \
                return EDOM;                                                                                           \
            }                                                                                                          \
            if (!result_null) {                                                                                        \
                value_type expected = {};                                                                              \
                status = expected_##type_tag(lhs, rhs, operation, &expected);                                          \
                if (status != 0 || actual != expected) {                                                               \
                    return status != 0 ? status : EDOM;                                                                \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return 0;                                                                                                      \
    }

BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(int32_t, i32)
BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(int64_t, i64)
BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(uint32_t, u32)
BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(uint64_t, u64)
BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(float, f32)
BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC(double, f64)

#undef BENCH_ARITHMETIC_DEFINE_CHECK_NUMERIC

static int check_result(const struct bench_args* args,
                        const struct inputs* inputs,
                        const struct dora_vector* result,
                        enum operation operation) {
    uint64_t rows;
    struct dora_logical_type type;
    int status = dora_vector_get_length(result, &rows);
    if (status == 0) {
        status = dora_vector_get_type(result, &type);
    }
    if (status != 0) {
        return status;
    }
    if (rows != args->rows || type.id != args->type.id) {
        return EDOM;
    }
    switch (type.id) {
    case DORA_TYPE_INT32:
        return check_i32(&inputs->left, &inputs->right, rows, result, operation);
    case DORA_TYPE_INT64:
        return check_i64(&inputs->left, &inputs->right, rows, result, operation);
    case DORA_TYPE_UINT32:
        return check_u32(&inputs->left, &inputs->right, rows, result, operation);
    case DORA_TYPE_UINT64:
        return check_u64(&inputs->left, &inputs->right, rows, result, operation);
    case DORA_TYPE_FLOAT32:
        return check_f32(&inputs->left, &inputs->right, rows, result, operation);
    case DORA_TYPE_FLOAT64:
        return check_f64(&inputs->left, &inputs->right, rows, result, operation);
    default:
        return ENOTSUP;
    }
}

static int check_proof(const struct bench_args* args, const struct inputs* inputs, enum operation operation) {
    struct bench_range left = {};
    struct bench_range right = {};
    struct bench_range range = {};
    int status = bench_exact_bounds(&inputs->left, &left);
    if (status == 0) {
        status = bench_exact_bounds(&inputs->right, &right);
    }
    if (status == 0) {
        status = bench_integer_range(args->type.id, &range);
    }
    if (status != 0) {
        return status;
    }
    const _BitInt(128) left_bounds[] = {left.minimum, left.maximum};
    const _BitInt(128) right_bounds[] = {right.minimum, right.maximum};
    bool proven = true;
    for (size_t left_side = 0; left_side < ARRAY_COUNT(left_bounds); ++left_side) {
        for (size_t right_side = 0; right_side < ARRAY_COUNT(right_bounds); ++right_side) {
            const _BitInt(128) lhs = left_bounds[left_side];
            const _BitInt(128) rhs = right_bounds[right_side];
            const _BitInt(128) corner = operation == OPERATION_MULTIPLY ? lhs * rhs : lhs + rhs;
            proven = proven && corner >= range.minimum && corner <= range.maximum;
        }
    }
    const bool expects_proof = args->variant == VARIANT_BOUNDS_PROVEN;
    return proven == expects_proof ? 0 : EDOM;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t left_bytes = 0;
    uint64_t right_bytes = 0;
    bench_check(bench, bench_input_bytes(&inputs->left, &left_bytes));
    bench_check(bench, bench_input_bytes(&inputs->right, &right_bytes));
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, left_bytes + right_bytes);
    dora_vector_release(&inputs->right);
    dora_vector_release(&inputs->left);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

#define BENCH_ARITHMETIC_DEFINE_CASE(name, operation)                                                                  \
    static int compute_##name(const struct inputs* inputs, struct dora_vector* out_slot) {                             \
        return dora_compute_##name(&inputs->left, &inputs->right, DORA_FAILURE_POLICY_ERROR, out_slot);                \
    }                                                                                                                  \
    BENCH_DEFINE_OPERATION(name##_operation, struct dora_vector, compute_##name, bench_release_vectors)                \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args, operation);                                                        \
        if (bench->status == 0) {                                                                                      \
            struct dora_vector result = {};                                                                            \
            bench_check(bench, compute_##name(&inputs, &result));                                                      \
            bench_check(bench, check_result(args, &inputs, &result, operation));                                       \
            bench_check(bench, has_bounds(args) ? check_proof(args, &inputs, operation) : 0);                          \
            dora_vector_release(&result);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

BENCH_ARITHMETIC_DEFINE_CASE(add, OPERATION_ADD)
BENCH_ARITHMETIC_DEFINE_CASE(multiply, OPERATION_MULTIPLY)
BENCH_ARITHMETIC_DEFINE_CASE(divide, OPERATION_DIVIDE)

#undef BENCH_ARITHMETIC_DEFINE_CASE

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"add/int32/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"add/int32/nulls:0/bounds:proven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .variant = VARIANT_BOUNDS_PROVEN}},
        {"add/int32/nulls:0/bounds:unproven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"add/int64/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"add/int64/nulls:0/bounds:proven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_PROVEN}},
        {"add/int64/nulls:0/bounds:unproven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"add/uint32/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT32}},
        {"add/uint32/nulls:0/bounds:proven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT32, .variant = VARIANT_BOUNDS_PROVEN}},
        {"add/uint32/nulls:0/bounds:unproven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT32, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"add/uint64/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT64}},
        {"add/uint64/nulls:0/bounds:proven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT64, .variant = VARIANT_BOUNDS_PROVEN}},
        {"add/uint64/nulls:0/bounds:unproven",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT64, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"add/float32/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT32}},
        {"add/float64/nulls:0", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"add/int64/nulls:0.5", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5}},
        {"add/float64/nulls:0.5", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64, .null_fraction = 0.5}},
        {"add/int64/scalar", add, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_ARRAY_SCALAR}},
        {"add/float64/scalar",
         add,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64, .variant = VARIANT_ARRAY_SCALAR}},
        {"multiply/int64", multiply, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"multiply/int64/bounds:proven",
         multiply,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_PROVEN}},
        {"multiply/int64/bounds:unproven",
         multiply,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"divide/float64", divide, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
