// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "types/array_count.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_FLAT,
    VARIANT_UTF8_VIEW,
    VARIANT_LARGE_UTF8,
    VARIANT_DICTIONARY,
    VARIANT_DECODED,
    VARIANT_SLICE,
    VARIANT_PAIR
};

struct inputs {
    struct dora_vector source;
    struct dora_vector other;
    struct dora_vector_view view;
    uint64_t rows;
    struct dora_range range;

    union {
        const int32_t* i32;
        const int64_t* i64;
        const double* f64;
    } scan_values;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {.rows = args->rows};
    if (args->variant == VARIANT_DECODED) {
        bench_check(bench, bench_decoded_dictionary(bench, args, 1, &inputs.source));
    } else if (args->variant == VARIANT_DICTIONARY) {
        bench_check(bench, bench_dictionary(bench, args, 1, &inputs.source));
    } else if (args->variant == VARIANT_UTF8_VIEW || args->variant == VARIANT_LARGE_UTF8) {
        const enum dora_layout layout =
            args->variant == VARIANT_UTF8_VIEW ? DORA_LAYOUT_BINARY_VIEW : DORA_LAYOUT_LARGE_BINARY;
        bench_check(bench, bench_utf8(bench, args, 1, layout, &inputs.source));
    } else {
        bench_check(bench, bench_random(bench, args, 1, &inputs.source));
    }
    bench_check(bench, dora_vector_get_view(&inputs.source, &inputs.view));
    if (args->variant == VARIANT_PAIR) {
        bench_check(bench, bench_random(bench, args, 1, &inputs.other));
    }
    if (args->variant == VARIANT_SLICE) {
        inputs.range = (struct dora_range){.offset = args->rows / 4, .length = args->rows / 2};
    }
    return inputs;
}

static int prepare_scan_values(struct inputs* inputs) {
    switch (inputs->view.physical_type) {
    case DORA_PHYSICAL_TYPE_I32:
        inputs->scan_values.i32 = dora_vector_view_get_values_i32(&inputs->view);
        return inputs->scan_values.i32 == nullptr ? EDOM : 0;
    case DORA_PHYSICAL_TYPE_I64:
        inputs->scan_values.i64 = dora_vector_view_get_values_i64(&inputs->view);
        return inputs->scan_values.i64 == nullptr ? EDOM : 0;
    case DORA_PHYSICAL_TYPE_F64:
        inputs->scan_values.f64 = dora_vector_view_get_values_f64(&inputs->view);
        return inputs->scan_values.f64 == nullptr ? EDOM : 0;
    default:
        return 0;
    }
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_encoded(const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector_view view = {};
    struct dora_vector domain = {};
    uint64_t domain_rows = 0;
    int status = bench_vectors_equal(&inputs->source, result);
    if (status == 0) {
        status = dora_vector_get_view(result, &view);
    }
    if (status == 0) {
        status = dora_vector_get_child(result, 0, &domain);
    }
    if (status == 0) {
        status = dora_vector_get_length(&domain, &domain_rows);
    }
    dora_vector_release(&domain);
    return status != 0                                                                            ? status
           : view.layout == DORA_LAYOUT_DICTIONARY && domain_rows == BENCH_DICTIONARY_CARDINALITY ? 0
                                                                                                  : EDOM;
}

static int check_vector(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    if (args->variant == VARIANT_DECODED) {
        return check_encoded(inputs, result);
    }
    if (args->variant == VARIANT_DICTIONARY) {
        int status = bench_vectors_equal(&inputs->source, result);
        struct dora_vector_view view = {};
        if (status == 0) {
            status = dora_vector_get_view(result, &view);
        }
        return status != 0 ? status : view.layout == DORA_LAYOUT_BINARY && view.selection.indices == nullptr ? 0 : EDOM;
    }
    uint64_t length = 0;
    int status = dora_vector_get_length(result, &length);
    if (status != 0 || length != inputs->range.length) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t row = 0; row < length; ++row) {
        int64_t actual = 0;
        int64_t expected = 0;
        bool actual_null = false;
        bool expected_null = false;
        status = dora_vector_read_i64(result, row, &actual, &actual_null);
        if (status == 0) {
            status = dora_vector_read_i64(&inputs->source, row + inputs->range.offset, &expected, &expected_null);
        }
        if (status != 0 || actual != expected || actual_null || expected_null) {
            return status != 0 ? status : EDOM;
        }
    }
    return 0;
}

static int rows_equal(const struct dora_vector* left,
                      uint64_t shift,
                      const struct dora_vector* right,
                      uint64_t length,
                      bool* out_equal) {
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(left, &type);
    bool equal = true;
    for (uint64_t row = 0; status == 0 && equal && row < length; ++row) {
        bool left_null = false;
        bool right_null = false;
        if (type.id == DORA_TYPE_UTF8) {
            struct dora_bytes left_value = {};
            struct dora_bytes right_value = {};
            status = dora_vector_read_bytes(left, row + shift, &left_value, &left_null);
            if (status == 0) {
                status = dora_vector_read_bytes(right, row, &right_value, &right_null);
            }
            equal = bench_bytes_equal(left_value, right_value);
        } else {
            int32_t left_value = 0;
            int32_t right_value = 0;
            status = dora_vector_read_i32(left, row + shift, &left_value, &left_null);
            if (status == 0) {
                status = dora_vector_read_i32(right, row, &right_value, &right_null);
            }
            equal = left_value == right_value;
        }
        equal = left_null == right_null && (left_null || equal);
    }
    if (status == 0) {
        *out_equal = equal;
    }
    return status;
}

static int check_equality(const struct bench_args* args, const struct inputs* inputs, bool actual) {
    const uint64_t length = args->rows - 1;
    uint64_t source_rows = 0;
    uint64_t other_rows = 0;
    bool expected = false;
    bool shifted_expected = true;
    int status = dora_vector_get_length(&inputs->source, &source_rows);
    if (status == 0) {
        status = dora_vector_get_length(&inputs->other, &other_rows);
    }
    if (status == 0) {
        status = rows_equal(&inputs->source, 0, &inputs->other, args->rows, &expected);
    }
    if (status == 0) {
        status = rows_equal(&inputs->source, 1, &inputs->other, length, &shifted_expected);
    }
    if (status != 0 || source_rows != args->rows || other_rows != args->rows || !expected || shifted_expected) {
        return status != 0 ? status : EDOM;
    }
    struct dora_vector shifted = {};
    struct dora_vector prefix = {};
    bool shifted_equal = true;
    status = dora_vector_slice(&inputs->source, (struct dora_range){.offset = 1, .length = length}, &shifted);
    if (status == 0) {
        status = dora_vector_slice(&inputs->other, (struct dora_range){.length = length}, &prefix);
    }
    if (status == 0) {
        status = dora_vector_is_equal(&shifted, &prefix, &shifted_equal);
    }
    dora_vector_release(&prefix);
    dora_vector_release(&shifted);
    return status != 0 ? status : actual == expected && shifted_equal == shifted_expected ? 0 : EDOM;
}

static uint64_t byte_sum(struct dora_bytes value) {
    uint64_t total = 0;
    for (size_t index = 0; index < value.byte_count; ++index) {
        total += value.data[index];
    }
    return total;
}

static int check_fixed_binary(const struct inputs* inputs, uint64_t* out_total) {
    struct dora_vector_view view;
    int status = dora_vector_get_view(&inputs->source, &view);
    uint64_t total = 0;
    for (uint64_t row = 0; status == 0 && row < inputs->rows; ++row) {
        struct dora_bytes value = {};
        bool is_null = false;
        status = dora_vector_read_bytes(&inputs->source, row, &value, &is_null);
        if (status == 0 && (is_null || value.byte_count != view.byte_width ||
                            !bench_bytes_equal(value, dora_vector_view_get_bytes(&view, row)))) {
            status = EDOM;
        }
        total += byte_sum(value);
    }
    if (status == 0) {
        *out_total = total;
    }
    return status;
}

static bool reads_string_lengths(const struct bench_args* args) {
    return args->type.id == DORA_TYPE_UTF8 &&
           (args->variant == VARIANT_FLAT || args->variant == VARIANT_UTF8_VIEW || args->variant == VARIANT_LARGE_UTF8);
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t bytes = 0;
    uint64_t other_bytes = 0;
    if (args->variant != VARIANT_SLICE && !reads_string_lengths(args)) {
        bench_check(bench, bench_input_bytes(&inputs->source, &bytes));
    }
    if (args->variant == VARIANT_PAIR) {
        bench_check(bench, bench_input_bytes(&inputs->other, &other_bytes));
    }
    bench_set_items(bench, args->variant == VARIANT_SLICE ? 1 : args->rows);
    bench_set_bytes(bench, bytes + other_bytes);
    dora_vector_release(&inputs->other);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Scans
//--------------------------------------

static constexpr uint64_t BYTE_LANES = UINT64_C(0x00ff'00ff'00ff'00ff);
static constexpr uint64_t LANE_SUM = UINT64_C(0x0001'0001'0001'0001);
static constexpr uint32_t LANE_SUM_SHIFT = 48;

static uint64_t word_byte_sum(struct dora_bytes value) {
    const size_t word_bytes = value.byte_count - value.byte_count % sizeof(uint64_t);
    uint64_t total = 0;
    for (size_t index = 0; index < word_bytes; index += sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, value.data + index, sizeof(word));
        word = (word & BYTE_LANES) + ((word >> 8U) & BYTE_LANES);
        total += (word * LANE_SUM) >> LANE_SUM_SHIFT;
    }
    for (size_t index = word_bytes; index < value.byte_count; ++index) {
        total += value.data[index];
    }
    return total;
}

#define BENCH_READ_DEFINE_SCAN(value_type, type_tag)                                                                   \
    static int compute_scan_##type_tag(const struct inputs* inputs, uint64_t* out_slot) {                              \
        const value_type* values = inputs->scan_values.type_tag;                                                       \
        const uint64_t rows = inputs->rows;                                                                            \
        uint64_t total = 0;                                                                                            \
        for (uint64_t row = 0; row < rows; ++row) {                                                                    \
            total += (uint64_t)values[row];                                                                            \
        }                                                                                                              \
        *out_slot = total;                                                                                             \
        return 0;                                                                                                      \
    }

BENCH_READ_DEFINE_SCAN(int32_t, i32)
BENCH_READ_DEFINE_SCAN(int64_t, i64)

#undef BENCH_READ_DEFINE_SCAN

static constexpr uint64_t VALID_ROW_BIAS = 1;

static int compute_scan_i32_nulls(const struct inputs* inputs, uint64_t* out_slot) {
    struct dora_vector_view view = inputs->view;
    view.selection.indices = nullptr;
    const int32_t* values = inputs->scan_values.i32;
    struct dora_row_cursor cursor = {};
    uint64_t row = 0;
    uint64_t total = 0;
    while (dora_vector_view_next_valid(&view, &cursor, &row)) {
        total += (uint64_t)values[row] + VALID_ROW_BIAS;
    }
    *out_slot = total;
    return 0;
}

static int compute_scan_f64(const struct inputs* inputs, uint64_t* out_slot) {
    const double* values = inputs->scan_values.f64;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        total += (uint64_t)(values[row] < BENCH_FLOAT_MIDPOINT);
    }
    *out_slot = total;
    return 0;
}

static int compute_scan_boolean(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector_view view = inputs->view;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        total += (uint64_t)dora_vector_view_get_bool(&view, row);
    }
    *out_slot = total;
    return 0;
}

static int compute_scan_utf8(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector_view view = inputs->view;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        total += dora_vector_view_get_bytes(&view, row).byte_count;
    }
    *out_slot = total;
    return 0;
}

static int compute_scan_fixed_binary(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector_view view = inputs->view;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        total += word_byte_sum(dora_vector_view_get_bytes(&view, row));
    }
    *out_slot = total;
    return 0;
}

//--------------------------------------
// Reads
//--------------------------------------

#define BENCH_READ_DEFINE_READ(case_tag, value_type, type_tag, row_bias)                                               \
    static int compute_read_##case_tag(const struct inputs* inputs, uint64_t* out_slot) {                              \
        static constexpr uint64_t row_bias_value = (row_bias);                                                         \
        const struct dora_vector* source = &inputs->source;                                                            \
        const uint64_t rows = inputs->rows;                                                                            \
        uint64_t total = 0;                                                                                            \
        for (uint64_t row = 0; row < rows; ++row) {                                                                    \
            value_type value = {};                                                                                     \
            bool is_null = false;                                                                                      \
            const int status = dora_vector_read_##type_tag(source, row, &value, &is_null);                             \
            if (status != 0) {                                                                                         \
                return status;                                                                                         \
            }                                                                                                          \
            total += is_null ? 0 : (uint64_t)value + row_bias_value;                                                   \
        }                                                                                                              \
        *out_slot = total;                                                                                             \
        return 0;                                                                                                      \
    }

BENCH_READ_DEFINE_READ(i32, int32_t, i32, 0)
BENCH_READ_DEFINE_READ(i32_nulls, int32_t, i32, VALID_ROW_BIAS)
BENCH_READ_DEFINE_READ(i64, int64_t, i64, 0)
BENCH_READ_DEFINE_READ(boolean, bool, bool, 0)

#undef BENCH_READ_DEFINE_READ

static int compute_read_f64(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector* source = &inputs->source;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        double value = 0;
        bool is_null = false;
        const int status = dora_vector_read_f64(source, row, &value, &is_null);
        if (status != 0) {
            return status;
        }
        total += (uint64_t)(!is_null && value < BENCH_FLOAT_MIDPOINT);
    }
    *out_slot = total;
    return 0;
}

static int compute_read_utf8(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector* source = &inputs->source;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        struct dora_bytes value = {};
        bool is_null = false;
        const int status = dora_vector_read_bytes(source, row, &value, &is_null);
        if (status != 0) {
            return status;
        }
        total += is_null ? 0 : value.byte_count;
    }
    *out_slot = total;
    return 0;
}

static int compute_read_fixed_binary(const struct inputs* inputs, uint64_t* out_slot) {
    const struct dora_vector* source = &inputs->source;
    const uint64_t rows = inputs->rows;
    uint64_t total = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        struct dora_bytes value = {};
        bool is_null = false;
        const int status = dora_vector_read_bytes(source, row, &value, &is_null);
        if (status != 0) {
            return status;
        }
        total += is_null ? 0 : word_byte_sum(value);
    }
    *out_slot = total;
    return 0;
}

//--------------------------------------
// Scan and Read Cases
//--------------------------------------

#define BENCH_READ_DEFINE_TOTAL_CASE(name, oracle_function)                                                            \
    BENCH_DEFINE_OPERATION(name##_operation, uint64_t, compute_##name, nullptr)                                        \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            bench_check(bench, prepare_scan_values(&inputs));                                                          \
        }                                                                                                              \
        if (bench->status == 0) {                                                                                      \
            uint64_t total = 0;                                                                                        \
            uint64_t expected = 0;                                                                                     \
            bench_check(bench, compute_##name(&inputs, &total));                                                       \
            bench_check(bench, oracle_function(&inputs, &expected));                                                   \
            bench_check(bench, total == expected ? 0 : EDOM);                                                          \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

BENCH_READ_DEFINE_TOTAL_CASE(scan_i32, compute_read_i32)
BENCH_READ_DEFINE_TOTAL_CASE(scan_i32_nulls, compute_read_i32_nulls)
BENCH_READ_DEFINE_TOTAL_CASE(scan_i64, compute_read_i64)
BENCH_READ_DEFINE_TOTAL_CASE(scan_f64, compute_read_f64)
BENCH_READ_DEFINE_TOTAL_CASE(scan_boolean, compute_read_boolean)
BENCH_READ_DEFINE_TOTAL_CASE(scan_utf8, compute_read_utf8)
BENCH_READ_DEFINE_TOTAL_CASE(scan_fixed_binary, check_fixed_binary)
BENCH_READ_DEFINE_TOTAL_CASE(read_i32, compute_scan_i32)
BENCH_READ_DEFINE_TOTAL_CASE(read_i32_nulls, compute_scan_i32_nulls)
BENCH_READ_DEFINE_TOTAL_CASE(read_i64, compute_scan_i64)
BENCH_READ_DEFINE_TOTAL_CASE(read_f64, compute_scan_f64)
BENCH_READ_DEFINE_TOTAL_CASE(read_boolean, compute_scan_boolean)
BENCH_READ_DEFINE_TOTAL_CASE(read_utf8, compute_scan_utf8)
BENCH_READ_DEFINE_TOTAL_CASE(read_fixed_binary, check_fixed_binary)

#undef BENCH_READ_DEFINE_TOTAL_CASE

//--------------------------------------
// Derived Vectors
//--------------------------------------

#define BENCH_READ_DEFINE_VECTOR_CASE(name)                                                                            \
    BENCH_DEFINE_OPERATION(name##_operation, struct dora_vector, compute_##name, bench_release_vectors)                \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct dora_vector result = {};                                                                            \
            bench_check(bench, compute_##name(&inputs, &result));                                                      \
            bench_check(bench, check_vector(args, &inputs, &result));                                                  \
            dora_vector_release(&result);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

static int compute_decode(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_vector_decode(&inputs->source, out_slot);
}

BENCH_READ_DEFINE_VECTOR_CASE(decode)

static constexpr enum dora_layout ENCODED_LAYOUTS[] = {DORA_LAYOUT_DICTIONARY};

static int compute_encode(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_vector_recode(&inputs->source, ENCODED_LAYOUTS, ARRAY_COUNT(ENCODED_LAYOUTS), out_slot);
}

BENCH_READ_DEFINE_VECTOR_CASE(encode)

static int compute_slice(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_vector_slice(&inputs->source, inputs->range, out_slot);
}

BENCH_READ_DEFINE_VECTOR_CASE(slice)

#undef BENCH_READ_DEFINE_VECTOR_CASE

//--------------------------------------
// Equality
//--------------------------------------

static int compute_is_equal(const struct inputs* inputs, bool* out_slot) {
    return dora_vector_is_equal(&inputs->source, &inputs->other, out_slot);
}

BENCH_DEFINE_OPERATION(is_equal_operation, bool, compute_is_equal, nullptr)

static void is_equal(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        bool equal = false;
        bench_check(bench, compute_is_equal(&inputs, &equal));
        bench_check(bench, check_equality(args, &inputs, equal));
    }
    bench_measure(bench, &is_equal_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"scan/int32", scan_i32, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"read/int32", read_i32, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"scan/int32/nulls:0.5",
         scan_i32_nulls,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5}},
        {"read/int32/nulls:0.5",
         read_i32_nulls,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5}},
        {"scan/int64", scan_i64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"read/int64", read_i64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"scan/float64", scan_f64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"read/float64", read_f64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"scan/boolean", scan_boolean, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5}},
        {"read/boolean", read_boolean, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5}},
        {"scan/utf8", scan_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"read/utf8", read_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"scan/utf8_view", scan_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"read/utf8_view", read_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"scan/large_utf8", scan_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}},
        {"read/large_utf8", read_utf8, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}},
        {"scan/fixed_binary<16>",
         scan_fixed_binary,
         {.rows = BENCH_ROWS, .type = {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16}}},
        {"read/fixed_binary<16>",
         read_fixed_binary,
         {.rows = BENCH_ROWS, .type = {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16}}},
        {"decode/dictionary<utf8>",
         decode,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_DICTIONARY}},
        {"encode/dictionary<utf8>",
         encode,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_DECODED}},
        {"slice/int64", slice, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_SLICE}},
        {"is_equal/int32", is_equal, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .variant = VARIANT_PAIR}},
        {"is_equal/utf8", is_equal, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_PAIR}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
