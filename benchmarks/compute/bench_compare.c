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
    VARIANT_UTF8_VIEW,
    VARIANT_LARGE_UTF8,
    VARIANT_UNARY
};

static constexpr uint64_t LEFT_SEED = 1;
static constexpr uint64_t RIGHT_SEED = 2;

struct inputs {
    struct dora_vector left;
    struct dora_vector right;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args, uint64_t right_seed) {
    struct inputs inputs = {};
    if (args->variant == VARIANT_UTF8_VIEW || args->variant == VARIANT_LARGE_UTF8) {
        const enum dora_layout layout =
            args->variant == VARIANT_UTF8_VIEW ? DORA_LAYOUT_BINARY_VIEW : DORA_LAYOUT_LARGE_BINARY;
        bench_check(bench, bench_utf8(bench, args, LEFT_SEED, layout, &inputs.left));
        bench_check(bench, bench_utf8(bench, args, right_seed, layout, &inputs.right));
        return inputs;
    }
    bench_check(bench, bench_random(bench, args, LEFT_SEED, &inputs.left));
    if (args->variant == VARIANT_ARRAY_SCALAR) {
        bench_check(bench, bench_constant(bench, args, right_seed, &inputs.right));
    } else if (args->variant == VARIANT_ARRAY) {
        bench_check(bench, bench_random(bench, args, right_seed, &inputs.right));
    }
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_result(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    uint64_t length = 0;
    struct dora_logical_type type = {};
    int status = dora_vector_get_length(result, &length);
    if (status == 0) {
        status = dora_vector_get_type(&inputs->left, &type);
    }
    if (status != 0 || length != args->rows) {
        return status != 0 ? status : EDOM;
    }
    const bool is_unary = !dora_vector_is_valid(&inputs->right);
    for (uint64_t row = 0; row < length; ++row) {
        bool actual = false;
        bool actual_null = false;
        bool left_null = false;
        bool right_null = false;
        bool expected = false;
        status = dora_vector_read_bool(result, row, &actual, &actual_null);
        if (status == 0 && type.id == DORA_TYPE_BOOLEAN) {
            bool left = false;
            bool right = false;
            status = dora_vector_read_bool(&inputs->left, row, &left, &left_null);
            if (status == 0 && is_unary) {
                expected = !left;
            } else if (status == 0) {
                status = dora_vector_read_bool(&inputs->right, row, &right, &right_null);
                expected = left && right;
                if ((!left_null && !left) || (!right_null && !right)) {
                    left_null = false;
                    right_null = false;
                }
            }
        } else if (status == 0 && (type.id == DORA_TYPE_UTF8 || type.id == DORA_TYPE_FIXED_BINARY)) {
            struct dora_bytes left = {};
            struct dora_bytes right = {};
            status = dora_vector_read_bytes(&inputs->left, row, &left, &left_null);
            if (status == 0) {
                status = dora_vector_read_bytes(&inputs->right, row, &right, &right_null);
            }
            expected = bench_bytes_equal(left, right);
        } else if (status == 0) {
            double left = 0;
            double right = 0;
            status = dora_vector_is_null(&inputs->left, row, &left_null);
            if (status == 0 && !left_null) {
                status = bench_read_number(&inputs->left, row, &left);
            }
            if (status == 0) {
                status = dora_vector_is_null(&inputs->right, row, &right_null);
            }
            if (status == 0 && !right_null) {
                status = bench_read_number(&inputs->right, row, &right);
            }
            expected = left < right;
        }
        if (status != 0) {
            return status;
        }
        const bool expected_null = left_null || right_null;
        if (actual_null != expected_null || (!expected_null && actual != expected)) {
            return EDOM;
        }
    }
    return 0;
}

static int check_mirror(struct bench* bench,
                        const struct bench_args* args,
                        int (*compute)(const struct inputs*, struct dora_vector*)) {
    struct inputs mirror = prepare(bench, args, LEFT_SEED);
    struct dora_vector result = {};
    int status = bench->status;
    if (status == 0) {
        status = compute(&mirror, &result);
    }
    if (status == 0) {
        status = check_result(args, &mirror, &result);
    }
    dora_vector_release(&result);
    dora_vector_release(&mirror.right);
    dora_vector_release(&mirror.left);
    return status;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t left_bytes = 0;
    uint64_t right_bytes = 0;
    bench_check(bench, bench_input_bytes(&inputs->left, &left_bytes));
    if (dora_vector_is_valid(&inputs->right)) {
        bench_check(bench, bench_input_bytes(&inputs->right, &right_bytes));
    }
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

#define BENCH_COMPARE_DEFINE_CASE(name)                                                                                \
    BENCH_DEFINE_OPERATION(name##_operation, struct dora_vector, compute_##name, bench_release_vectors)                \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args, RIGHT_SEED);                                                       \
        if (bench->status == 0) {                                                                                      \
            struct dora_vector result = {};                                                                            \
            bench_check(bench, compute_##name(&inputs, &result));                                                      \
            bench_check(bench, check_result(args, &inputs, &result));                                                  \
            bench_check(bench, check_mirror(bench, args, compute_##name));                                             \
            dora_vector_release(&result);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

//--------------------------------------
// Comparison
//--------------------------------------

static int compute_lt(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_compare(&inputs->left, DORA_COMPARE_OP_LT, &inputs->right, out_slot);
}

BENCH_COMPARE_DEFINE_CASE(lt)

static int compute_eq(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_compare(&inputs->left, DORA_COMPARE_OP_EQ, &inputs->right, out_slot);
}

BENCH_COMPARE_DEFINE_CASE(eq)

//--------------------------------------
// Boolean Logic
//--------------------------------------

static int compute_and_bool(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_and(&inputs->left, &inputs->right, out_slot);
}

BENCH_COMPARE_DEFINE_CASE(and_bool)

static int compute_not_bool(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_not(&inputs->left, out_slot);
}

BENCH_COMPARE_DEFINE_CASE(not_bool)

#undef BENCH_COMPARE_DEFINE_CASE

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"lt/int32", lt, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"lt/int64", lt, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"lt/float64", lt, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"lt/timestamp",
         lt,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_TIMESTAMP,
          .type.parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr}}},
        {"lt/int64/nulls:0.5", lt, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5}},
        {"lt/int64/scalar", lt, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_ARRAY_SCALAR}},
        {"eq/utf8", eq, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"eq/utf8_view", eq, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"eq/large_utf8", eq, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}},
        {"eq/fixed_binary<16>",
         eq,
         {.rows = BENCH_ROWS, .type = {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16}}},
        {"and/boolean/nulls:0", and_bool, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5}},
        {"and/boolean/nulls:0.5",
         and_bool,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .null_fraction = 0.5, .selectivity = 0.5}},
        {"not/boolean",
         not_bool,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5, .variant = VARIANT_UNARY}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
