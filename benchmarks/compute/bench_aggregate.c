// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
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
    VARIANT_PLAIN,
    VARIANT_BOUNDS_PROVEN,
    VARIANT_BOUNDS_UNPROVEN
};

struct inputs {
    struct dora_vector source;
};

static constexpr struct bench_outlier SUM_OUTLIER = {.value = INT64_C(1) << 47};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {};
    const struct bench_outlier* outlier = args->variant == VARIANT_BOUNDS_UNPROVEN ? &SUM_OUTLIER : nullptr;
    bench_check(bench,
                args->variant == VARIANT_PLAIN ? bench_random(bench, args, 1, &inputs.source)
                                               : bench_with_bounds(bench, args, 1, outlier, &inputs.source));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

enum operation : uint8_t {
    OPERATION_SUM,
    OPERATION_MIN,
    OPERATION_MAX,
    OPERATION_COUNT
};

static int check_result(const struct bench_args* args,
                        const struct inputs* inputs,
                        const struct dora_vector* result,
                        enum operation operation) {
    struct dora_vector_view view = {};
    uint64_t length = 0;
    int status = dora_vector_get_view(&inputs->source, &view);
    if (status == 0) {
        status = dora_vector_get_length(result, &length);
    }
    if (status != 0 || length != 1) {
        return status != 0 ? status : EDOM;
    }
    double expected = 0;
    uint64_t count = 0;
    for (uint64_t row = 0; row < args->rows; ++row) {
        if (dora_vector_view_is_null(&view, row)) {
            continue;
        }
        const double value = args->type.id == DORA_TYPE_FLOAT64 ? dora_vector_view_get_values_f64(&view)[row]
                             : args->type.id == DORA_TYPE_INT32 ? dora_vector_view_get_values_i32(&view)[row]
                                                                : (double)dora_vector_view_get_values_i64(&view)[row];
        if (operation == OPERATION_SUM) {
            expected += value;
        } else if (operation == OPERATION_MIN) {
            expected = count == 0 || value < expected ? value : expected;
        } else if (operation == OPERATION_MAX) {
            expected = count == 0 || value > expected ? value : expected;
        }
        ++count;
    }
    bool is_null = false;
    double actual = 0;
    if (operation == OPERATION_COUNT) {
        int64_t number = 0;
        status = dora_vector_read_i64(result, 0, &number, &is_null);
        actual = (double)number;
        expected = (double)count;
    } else if (args->type.id == DORA_TYPE_FLOAT64) {
        status = dora_vector_read_f64(result, 0, &actual, &is_null);
    } else if (operation == OPERATION_SUM) {
        struct dora_i128 number = {};
        status = dora_vector_read_i128(result, 0, &number, &is_null);
        if (status != 0) {
            return status;
        }
        if (number.limbs[1] != 0) {
            return EDOM;
        }
        actual = (double)number.limbs[0];
    } else {
        int64_t number = 0;
        status = dora_vector_read_i64(result, 0, &number, &is_null);
        actual = (double)number;
    }
    return status != 0 ? status : is_null || actual != expected ? EDOM : 0;
}

static int check_proof(const struct bench_args* args, const struct inputs* inputs) {
    struct bench_range bounds = {};
    const int status = bench_exact_bounds(&inputs->source, &bounds);
    const _BitInt(128) rows = args->rows;
    const bool proven = bounds.minimum * rows > INT64_MIN && bounds.maximum * rows < INT64_MAX;
    const bool expects_proof = args->variant == VARIANT_BOUNDS_PROVEN;
    return status != 0 ? status : proven == expects_proof ? 0 : EDOM;
}

static void
finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs, enum operation operation) {
    uint64_t bytes = 0;
    if (operation != OPERATION_COUNT) {
        bench_check(bench, bench_input_bytes(&inputs->source, &bytes));
    }
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, bytes);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

#define BENCH_AGGREGATE_DEFINE_CASE(name, operation)                                                                   \
    static int compute_##name(const struct inputs* inputs, struct dora_vector* out_slot) {                             \
        return dora_compute_##name(&inputs->source, out_slot);                                                         \
    }                                                                                                                  \
    BENCH_DEFINE_OPERATION(name##_operation, struct dora_vector, compute_##name, bench_release_vectors)                \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct dora_vector result = {};                                                                            \
            bench_check(bench, compute_##name(&inputs, &result));                                                      \
            bench_check(bench, check_result(args, &inputs, &result, operation));                                       \
            bench_check(bench, args->variant != VARIANT_PLAIN ? check_proof(args, &inputs) : 0);                       \
            dora_vector_release(&result);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs, operation);                                                                       \
    }

BENCH_AGGREGATE_DEFINE_CASE(sum, OPERATION_SUM)
BENCH_AGGREGATE_DEFINE_CASE(min, OPERATION_MIN)
BENCH_AGGREGATE_DEFINE_CASE(max, OPERATION_MAX)
BENCH_AGGREGATE_DEFINE_CASE(count, OPERATION_COUNT)

#undef BENCH_AGGREGATE_DEFINE_CASE

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"sum/int32/nulls:0", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"sum/int32/nulls:0/bounds:proven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .variant = VARIANT_BOUNDS_PROVEN}},
        {"sum/int32/nulls:0.5", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5}},
        {"sum/int32/nulls:0.5/bounds:proven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5, .variant = VARIANT_BOUNDS_PROVEN}},
        {"sum/int64/nulls:0", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"sum/int64/nulls:0/bounds:proven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_PROVEN}},
        {"sum/int64/nulls:0/bounds:unproven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"sum/int64/nulls:0.5", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5}},
        {"sum/int64/nulls:0.5/bounds:proven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5, .variant = VARIANT_BOUNDS_PROVEN}},
        {"sum/int64/nulls:0.5/bounds:unproven",
         sum,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5, .variant = VARIANT_BOUNDS_UNPROVEN}},
        {"sum/float64/nulls:0", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"sum/float64/nulls:0.5", sum, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64, .null_fraction = 0.5}},
        {"min/int64", min, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"max/int64", max, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"count/int64/nulls:0.5", count, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
