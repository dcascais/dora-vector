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
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_PLAIN,
    VARIANT_BOUNDS_PROVEN
};

struct inputs {
    struct dora_vector source;
    struct dora_logical_type target;
    enum dora_rounding_mode rounding;
};

static struct bench_args parsed_args(const struct bench_args* args) {
    return (struct bench_args){.rows = args->rows, .type = args->target, .null_fraction = args->null_fraction};
}

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {
        .target = args->target,
        .rounding = args->target.id == DORA_TYPE_FLOAT64 ? DORA_ROUNDING_MODE_HALF_EVEN : DORA_ROUNDING_MODE_TOWARD_ZERO
    };
    const struct bench_args value_args = parsed_args(args);
    bench_check(bench,
                args->type.id == DORA_TYPE_UTF8          ? bench_formatted(bench, &value_args, 1, &inputs.source)
                : args->variant == VARIANT_BOUNDS_PROVEN ? bench_with_bounds(bench, args, 1, nullptr, &inputs.source)
                                                         : bench_random(bench, args, 1, &inputs.source));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static constexpr int SHORTEST_DIGITS_LIMIT = 17;
static constexpr size_t SCIENTIFIC_TEXT_BYTES = 32;

static int shortest_digits(double value, int* out_digits) {
    for (int digits = 1; digits <= SHORTEST_DIGITS_LIMIT; ++digits) {
        char text[SCIENTIFIC_TEXT_BYTES];
        const int length = snprintf(text, sizeof(text), "%.*e", digits - 1, value);
        const int status = bench_format_status(length, sizeof(text));
        if (status != 0) {
            return status;
        }
        if (strtod(text, nullptr) == value) {
            *out_digits = digits;
            return 0;
        }
    }
    return EDOM;
}

static bool is_shortest_fixed(struct dora_bytes text, double value) {
    size_t first = text.byte_count;
    size_t last = 0;
    for (size_t index = 0; index < text.byte_count; ++index) {
        const uint8_t byte = text.data[index];
        if ((byte < '0' || byte > '9') && byte != '.' && byte != '-') {
            return false;
        }
        if (byte >= '1' && byte <= '9') {
            first = first < index ? first : index;
            last = index;
        }
    }
    int expected = 0;
    if (first == text.byte_count || shortest_digits(value, &expected) != 0) {
        return false;
    }
    int digits = 0;
    for (size_t index = first; index <= last; ++index) {
        digits += text.data[index] != '.';
    }
    return digits == expected;
}

static int
check_text(const struct bench_args* args, const struct inputs* inputs, uint64_t row, struct dora_bytes actual) {
    char text[BENCH_FORMATTED_VALUE_BYTES] = {};
    size_t size = 0;
    bool is_null = false;
    int status = ENOTSUP;
    if (args->type.id == DORA_TYPE_FLOAT64) {
        double expected = 0;
        status =
            actual.byte_count < sizeof(text) ? dora_vector_read_f64(&inputs->source, row, &expected, &is_null) : EDOM;
        if (status != 0 || is_null || actual.byte_count == 0) {
            return status != 0 ? status : EDOM;
        }
        memcpy(text, actual.data, actual.byte_count);
        char* end = nullptr;
        const double parsed = strtod(text, &end);
        return end == text + actual.byte_count && parsed == expected && is_shortest_fixed(actual, expected) ? 0 : EDOM;
    }
    if (args->type.id == DORA_TYPE_DATE32) {
        int32_t days = 0;
        status = dora_vector_read_i32(&inputs->source, row, &days, &is_null);
        if (status == 0) {
            status = bench_temporal_text(&args->type, days, text, &size);
        }
    } else if (args->type.id == DORA_TYPE_TIMESTAMP) {
        int64_t ticks = 0;
        status = dora_vector_read_i64(&inputs->source, row, &ticks, &is_null);
        if (status == 0) {
            status = bench_temporal_text(&args->type, ticks, text, &size);
        }
    } else if (args->type.id == DORA_TYPE_INT64) {
        int64_t value = 0;
        status = dora_vector_read_i64(&inputs->source, row, &value, &is_null);
        if (status == 0) {
            const int length = snprintf(text, sizeof(text), "%" PRId64, value);
            status = bench_format_status(length, sizeof(text));
            size = status == 0 ? (size_t)length : 0;
        }
    }
    const struct dora_bytes expected = {.data = (const uint8_t*)text, .byte_count = size};
    return status != 0 ? status : !is_null && bench_bytes_equal(expected, actual) ? 0 : EDOM;
}

static int check_result(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    uint64_t length = 0;
    struct dora_logical_type actual_type = {};
    int status = dora_vector_get_length(result, &length);
    if (status == 0) {
        status = dora_vector_get_type(result, &actual_type);
    }
    if (status != 0 || length != args->rows || actual_type.id != args->target.id) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t row = 0; row < args->rows; ++row) {
        if (args->target.id == DORA_TYPE_UTF8) {
            struct dora_bytes actual = {};
            bool actual_null = false;
            status = dora_vector_read_bytes(result, row, &actual, &actual_null);
            if (status == 0) {
                status = actual_null ? EDOM : check_text(args, inputs, row, actual);
            }
            if (status != 0) {
                return status;
            }
        } else {
            double expected = 0;
            double actual = 0;
            status = bench_read_number(&inputs->source, row, &expected);
            if (status == 0) {
                status = bench_read_number(result, row, &actual);
            }
            if (args->target.id == DORA_TYPE_INT64 || args->target.id == DORA_TYPE_INT32) {
                expected = trunc(expected);
            }
            if (status != 0 || actual != expected) {
                return status != 0 ? status : EDOM;
            }
        }
    }
    return 0;
}

static int check_proof(const struct bench_args* args, const struct inputs* inputs) {
    struct bench_range bounds = {};
    struct bench_range target = {};
    int status = bench_exact_bounds(&inputs->source, &bounds);
    if (status == 0) {
        status = bench_integer_range(args->target.id, &target);
    }
    return status != 0 ? status : bounds.minimum >= target.minimum && bounds.maximum <= target.maximum ? 0 : EDOM;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t bytes = 0;
    bench_check(bench, bench_input_bytes(&inputs->source, &bytes));
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, bytes);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

static int compute_cast(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_cast(&inputs->source, &inputs->target, inputs->rounding, DORA_FAILURE_POLICY_ERROR, out_slot);
}

BENCH_DEFINE_OPERATION(cast_operation, struct dora_vector, compute_cast, bench_release_vectors)

//--------------------------------------
// Conversions
//--------------------------------------

static void cast(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct dora_vector result = {};
        bench_check(bench, compute_cast(&inputs, &result));
        bench_check(bench, check_result(args, &inputs, &result));
        bench_check(bench, args->variant == VARIANT_BOUNDS_PROVEN ? check_proof(args, &inputs) : 0);
        dora_vector_release(&result);
    }
    bench_measure(bench, &cast_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Parsing
//--------------------------------------

static void parse(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        const struct bench_args value_args = parsed_args(args);
        struct dora_vector expected = {};
        struct dora_vector result = {};
        bench_check(bench, bench_random(bench, &value_args, 1, &expected));
        bench_check(bench, compute_cast(&inputs, &result));
        bench_check(bench, bench_vectors_equal(&expected, &result));
        dora_vector_release(&result);
        dora_vector_release(&expected);
    }
    bench_measure(bench, &cast_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"cast/int32/to:int64", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .target.id = DORA_TYPE_INT64}},
        {"cast/int32/to:int64/bounds:proven",
         cast,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_INT32,
          .variant = VARIANT_BOUNDS_PROVEN,
          .target.id = DORA_TYPE_INT64}},
        {"cast/int64/to:int32", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .target.id = DORA_TYPE_INT32}},
        {"cast/int64/to:int32/bounds:proven",
         cast,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_INT64,
          .variant = VARIANT_BOUNDS_PROVEN,
          .target.id = DORA_TYPE_INT32}},
        {"cast/uint32/to:int32", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UINT32, .target.id = DORA_TYPE_INT32}},
        {"cast/uint32/to:int32/bounds:proven",
         cast,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_UINT32,
          .variant = VARIANT_BOUNDS_PROVEN,
          .target.id = DORA_TYPE_INT32}},
        {"cast/int64/to:float64",
         cast,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .target.id = DORA_TYPE_FLOAT64}},
        {"cast/float64/to:int64",
         cast,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64, .target.id = DORA_TYPE_INT64}},
        {"cast/int64/to:utf8", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .target.id = DORA_TYPE_UTF8}},
        {"cast/float64/to:utf8", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64, .target.id = DORA_TYPE_UTF8}},
        {"cast/date32/to:utf8", cast, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_DATE32, .target.id = DORA_TYPE_UTF8}},
        {"cast/timestamp/to:utf8",
         cast,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_TIMESTAMP,
          .type.parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr},
          .target.id = DORA_TYPE_UTF8}},
        {"cast/decimal128/to:float64",
         cast,
         {.rows = BENCH_ROWS,
          .type = {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 38, .scale = 8, .bit_width = 128}},
          .target.id = DORA_TYPE_FLOAT64}},
        {"cast/utf8/to:int64", parse, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .target.id = DORA_TYPE_INT64}},
        {"cast/utf8/to:float64",
         parse,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .target.id = DORA_TYPE_FLOAT64}},
        {"cast/utf8/to:timestamp",
         parse,
         {.rows = BENCH_ROWS,
          .type.id = DORA_TYPE_UTF8,
          .target.id = DORA_TYPE_TIMESTAMP,
          .target.parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr}}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
