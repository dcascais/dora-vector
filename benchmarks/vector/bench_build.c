// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "types/array_count.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_FLAT,
    VARIANT_TRUST,
    VARIANT_UTF8_VIEW,
    VARIANT_LARGE_UTF8,
    VARIANT_DICTIONARY,
    VARIANT_LIST_LENGTHS
};

struct inputs {
    struct dora_pool pool;
    uint64_t rows;
    const struct dora_builder_options* options;
    struct dora_vector source;
    struct dora_vector domain;
    struct dora_schema schema;
    const void* values;
    uint32_t* lengths;
    struct dora_bytes* bytes;
    bool* booleans;
    bool* validity;
};

static int prepare_validity(const struct dora_vector_view* view, uint64_t rows, struct inputs* inputs) {
    inputs->validity = calloc(rows, sizeof(*inputs->validity));
    if (inputs->validity == nullptr) {
        return ENOMEM;
    }
    dora_vector_view_copy_validity(view, (struct dora_range){.length = rows}, inputs->validity);
    return 0;
}

static int prepare_bytes(const struct dora_vector_view* view, uint64_t rows, struct inputs* inputs) {
    inputs->bytes = calloc(rows, sizeof(*inputs->bytes));
    if (inputs->bytes == nullptr) {
        return ENOMEM;
    }
    for (uint64_t row = 0; row < rows; ++row) {
        inputs->bytes[row] = dora_vector_view_get_bytes(view, row);
    }
    return 0;
}

static int prepare_lengths(const struct dora_vector_view* view, uint64_t rows, struct inputs* inputs) {
    inputs->lengths = calloc(rows, sizeof(*inputs->lengths));
    if (inputs->lengths == nullptr) {
        return ENOMEM;
    }
    for (uint64_t row = 0; row < rows; ++row) {
        inputs->lengths[row] = (uint32_t)dora_vector_view_get_elements(view, row).length;
    }
    return 0;
}

static int prepare_booleans(const struct dora_vector_view* view, uint64_t rows, struct inputs* inputs) {
    inputs->booleans = calloc(rows, sizeof(*inputs->booleans));
    if (inputs->booleans == nullptr) {
        return ENOMEM;
    }
    for (uint64_t row = 0; row < rows; ++row) {
        inputs->booleans[row] = dora_vector_view_get_bool(view, row);
    }
    return 0;
}

static constexpr struct dora_builder_options TRUST_OPTIONS = {.content_policy = DORA_FAILURE_POLICY_TRUST};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs =
        {.pool = bench->pool, .rows = args->rows, .options = args->variant == VARIANT_TRUST ? &TRUST_OPTIONS : nullptr};
    struct dora_vector first = {};
    struct dora_vector second = {};
    struct dora_vector_view view = {};
    struct dora_vector_view first_view = {};
    struct dora_vector_view second_view = {};
    const bool is_nested = args->type.id == DORA_TYPE_LIST || args->type.id == DORA_TYPE_STRUCT;
    int status;
    if (args->variant == VARIANT_DICTIONARY) {
        status = bench_dictionary(bench, args, 1, &inputs.source);
    } else if (args->variant == VARIANT_UTF8_VIEW || args->variant == VARIANT_LARGE_UTF8) {
        const enum dora_layout layout =
            args->variant == VARIANT_UTF8_VIEW ? DORA_LAYOUT_BINARY_VIEW : DORA_LAYOUT_LARGE_BINARY;
        status = bench_utf8(bench, args, 1, layout, &inputs.source);
    } else {
        status = bench_random(bench, args, 1, &inputs.source);
    }
    if (status == 0) {
        status = dora_vector_retain_schema(&inputs.source, &inputs.schema);
    }
    if (status == 0) {
        status = dora_vector_get_view(&inputs.source, &view);
    }
    if (status == 0 && args->variant == VARIANT_DICTIONARY) {
        status = dora_vector_get_child(&inputs.source, 0, &inputs.domain);
    }
    if (status == 0 && is_nested) {
        status = dora_vector_get_child(&inputs.source, 0, &first);
    }
    if (status == 0 && is_nested) {
        status = dora_vector_get_view(&first, &first_view);
    }
    if (status == 0 && args->type.id == DORA_TYPE_STRUCT) {
        status = dora_vector_get_child(&inputs.source, 1, &second);
    }
    if (status == 0 && args->type.id == DORA_TYPE_STRUCT) {
        status = dora_vector_get_view(&second, &second_view);
    }
    if (status == 0) {
        inputs.values = bench_view_values(is_nested ? &first_view : &view, nullptr);
    }
    if (status == 0 && args->variant == VARIANT_LIST_LENGTHS && inputs.values != nullptr) {
        inputs.values = (const int32_t*)inputs.values + dora_vector_view_get_elements(&view, 0).offset;
        status = prepare_lengths(&view, args->rows, &inputs);
    }
    if (status == 0 && !is_nested && view.validity.data != nullptr) {
        status = prepare_validity(&view, args->rows, &inputs);
    }
    if (status == 0 && args->type.id == DORA_TYPE_BOOLEAN) {
        status = prepare_booleans(&view, args->rows, &inputs);
    }
    if (status == 0 && ((args->type.id == DORA_TYPE_UTF8 && args->variant != VARIANT_DICTIONARY) ||
                        args->type.id == DORA_TYPE_FIXED_BINARY)) {
        status = prepare_bytes(&view, args->rows, &inputs);
    }
    if (status == 0 && args->type.id == DORA_TYPE_STRUCT) {
        status = prepare_bytes(&second_view, args->rows, &inputs);
    }
    if (status == 0 && inputs.values == nullptr && inputs.booleans == nullptr && inputs.bytes == nullptr) {
        status = EDOM;
    }
    dora_vector_release(&second);
    dora_vector_release(&first);
    bench_check(bench, status);
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static uint64_t null_rows(const struct inputs* inputs) {
    uint64_t count = 0;
    for (uint64_t row = 0; inputs->validity != nullptr && row < inputs->rows; ++row) {
        count += inputs->validity[row] ? 0 : 1;
    }
    return count;
}

static int check_result(const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector_view expected = {};
    struct dora_vector_view actual = {};
    uint64_t length = 0;
    uint64_t null_count = 0;
    int status = bench_vectors_equal(&inputs->source, result);
    if (status == 0) {
        status = dora_vector_get_view(&inputs->source, &expected);
    }
    if (status == 0) {
        status = dora_vector_get_view(result, &actual);
    }
    if (status == 0) {
        status = dora_vector_get_length(result, &length);
    }
    if (status == 0) {
        status = dora_vector_get_null_count(result, &null_count);
    }
    if (status == 0 &&
        (actual.layout != expected.layout || length != inputs->rows || null_count != null_rows(inputs))) {
        status = EDOM;
    }
    return status;
}

static int value_bytes(const struct bench_args* args, const struct dora_vector* source, uint64_t* out_bytes) {
    struct dora_vector child = {};
    struct dora_vector_view view = {};
    const bool is_nested = args->type.id == DORA_TYPE_LIST || args->type.id == DORA_TYPE_STRUCT;
    int status = is_nested ? dora_vector_get_child(source, 0, &child) : 0;
    if (status == 0) {
        status = dora_vector_get_view(is_nested ? &child : source, &view);
    }
    if (status == 0) {
        *out_bytes = view.values.byte_count;
    }
    dora_vector_release(&child);
    return status;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t bytes = 0;
    if (args->type.id == DORA_TYPE_BOOLEAN) {
        bytes = args->rows * sizeof(*inputs->booleans);
    } else if (args->type.id != DORA_TYPE_UTF8 || args->variant == VARIANT_DICTIONARY) {
        bench_check(bench, value_bytes(args, &inputs->source, &bytes));
    }
    if (inputs->bytes != nullptr && args->type.id != DORA_TYPE_FIXED_BINARY) {
        for (uint64_t row = 0; row < args->rows; ++row) {
            bytes += inputs->bytes[row].byte_count;
        }
    }
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, bytes);
    free(inputs->validity);
    free(inputs->lengths);
    free(inputs->booleans);
    free(inputs->bytes);
    dora_schema_release(&inputs->schema);
    dora_vector_release(&inputs->domain);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

struct build_slot {
    struct dora_builder builder;
    struct dora_vector result;
};

static void release_builds(void* slots, size_t count) {
    struct build_slot* builds = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_vector_release(&builds[slot].result);
        dora_builder_release(&builds[slot].builder);
    }
}

#define BENCH_BUILD_DEFINE_CASE(name)                                                                                  \
    BENCH_DEFINE_OPERATION(name##_operation, struct build_slot, compute_##name, release_builds)                        \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct build_slot slot = {};                                                                               \
            bench_check(bench, compute_##name(&inputs, &slot));                                                        \
            bench_check(bench, check_result(&inputs, &slot.result));                                                   \
            release_builds(&slot, 1);                                                                                  \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

//--------------------------------------
// Per-Value Appends
//--------------------------------------

#define BENCH_BUILD_DEFINE_APPEND(value_type, type_tag, values_member)                                                 \
    static int compute_append_##type_tag(const struct inputs* inputs, struct build_slot* out_slot) {                   \
        const value_type* values = inputs->values_member;                                                              \
        const uint64_t rows = inputs->rows;                                                                            \
        struct dora_builder* builder = &out_slot->builder;                                                             \
        int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);                \
        for (uint64_t row = 0; status == 0 && row < rows; ++row) {                                                     \
            status = dora_builder_append_##type_tag(builder, values[row]);                                             \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            status = dora_builder_finish(builder, &out_slot->result);                                                  \
        }                                                                                                              \
        return status;                                                                                                 \
    }

#define BENCH_BUILD_DEFINE_APPEND_NULLS(value_type, type_tag, values_member)                                           \
    static int compute_append_##type_tag##_nulls(const struct inputs* inputs, struct build_slot* out_slot) {           \
        const value_type* values = inputs->values_member;                                                              \
        const bool* validity = inputs->validity;                                                                       \
        const uint64_t rows = inputs->rows;                                                                            \
        struct dora_builder* builder = &out_slot->builder;                                                             \
        int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);                \
        for (uint64_t row = 0; status == 0 && row < rows; ++row) {                                                     \
            status = validity[row] ? dora_builder_append_##type_tag(builder, values[row])                              \
                                   : dora_builder_append_null(builder);                                                \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            status = dora_builder_finish(builder, &out_slot->result);                                                  \
        }                                                                                                              \
        return status;                                                                                                 \
    }

BENCH_BUILD_DEFINE_APPEND(int32_t, i32, values)
BENCH_BUILD_DEFINE_APPEND(int64_t, i64, values)
BENCH_BUILD_DEFINE_APPEND(double, f64, values)
BENCH_BUILD_DEFINE_APPEND(bool, bool, booleans)
BENCH_BUILD_DEFINE_APPEND(struct dora_bytes, bytes, bytes)
BENCH_BUILD_DEFINE_APPEND_NULLS(int32_t, i32, values)
BENCH_BUILD_DEFINE_APPEND_NULLS(struct dora_bytes, bytes, bytes)

#undef BENCH_BUILD_DEFINE_APPEND_NULLS
#undef BENCH_BUILD_DEFINE_APPEND

BENCH_BUILD_DEFINE_CASE(append_i32)
BENCH_BUILD_DEFINE_CASE(append_i64)
BENCH_BUILD_DEFINE_CASE(append_f64)
BENCH_BUILD_DEFINE_CASE(append_bool)
BENCH_BUILD_DEFINE_CASE(append_bytes)
BENCH_BUILD_DEFINE_CASE(append_i32_nulls)
BENCH_BUILD_DEFINE_CASE(append_bytes_nulls)

static int compute_append_dictionary(const struct inputs* inputs, struct build_slot* out_slot) {
    const uint32_t* codes = inputs->values;
    const uint64_t rows = inputs->rows;
    struct dora_builder* builder = &out_slot->builder;
    int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);
    if (status == 0) {
        status = dora_builder_set_dictionary(builder, &inputs->domain);
    }
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        status = dora_builder_append_u32(builder, codes[row]);
    }
    if (status == 0) {
        status = dora_builder_finish(builder, &out_slot->result);
    }
    return status;
}

BENCH_BUILD_DEFINE_CASE(append_dictionary)

static int compute_append_list(const struct inputs* inputs, struct build_slot* out_slot) {
    const uint32_t* lengths = inputs->lengths;
    const int32_t* values = inputs->values;
    const uint64_t rows = inputs->rows;
    struct dora_builder* builder = &out_slot->builder;
    struct dora_builder child = {};
    int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);
    if (status == 0) {
        status = dora_builder_get_child(builder, 0, &child);
    }
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        status = dora_builder_append_values_i32(&child, values, nullptr, lengths[row]);
        if (status == 0) {
            status = dora_builder_append_list(builder, lengths[row]);
        }
        values += lengths[row];
    }
    if (status == 0) {
        status = dora_builder_finish(builder, &out_slot->result);
    }
    return status;
}

BENCH_BUILD_DEFINE_CASE(append_list)

static int compute_append_struct(const struct inputs* inputs, struct build_slot* out_slot) {
    const int64_t* values = inputs->values;
    const struct dora_bytes* strings = inputs->bytes;
    const uint64_t rows = inputs->rows;
    struct dora_builder* builder = &out_slot->builder;
    struct dora_builder number = {};
    struct dora_builder text = {};
    int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);
    if (status == 0) {
        status = dora_builder_get_child(builder, 0, &number);
    }
    if (status == 0) {
        status = dora_builder_get_child(builder, 1, &text);
    }
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        status = dora_builder_append_i64(&number, values[row]);
        if (status == 0) {
            status = dora_builder_append_bytes(&text, strings[row]);
        }
        if (status == 0) {
            status = dora_builder_end_row(builder);
        }
    }
    if (status == 0) {
        status = dora_builder_finish(builder, &out_slot->result);
    }
    return status;
}

BENCH_BUILD_DEFINE_CASE(append_struct)

//--------------------------------------
// Bulk Appends
//--------------------------------------

#define BENCH_BUILD_DEFINE_APPEND_VALUES(type_tag, values_member)                                                      \
    static int compute_append_values_##type_tag(const struct inputs* inputs, struct build_slot* out_slot) {            \
        struct dora_builder* builder = &out_slot->builder;                                                             \
        int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);                \
        if (status == 0) {                                                                                             \
            status =                                                                                                   \
                dora_builder_append_values_##type_tag(builder, inputs->values_member, inputs->validity, inputs->rows); \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            status = dora_builder_finish(builder, &out_slot->result);                                                  \
        }                                                                                                              \
        return status;                                                                                                 \
    }

BENCH_BUILD_DEFINE_APPEND_VALUES(i32, values)
BENCH_BUILD_DEFINE_APPEND_VALUES(i64, values)
BENCH_BUILD_DEFINE_APPEND_VALUES(f64, values)
BENCH_BUILD_DEFINE_APPEND_VALUES(bool, booleans)
BENCH_BUILD_DEFINE_APPEND_VALUES(bytes, bytes)

#undef BENCH_BUILD_DEFINE_APPEND_VALUES

BENCH_BUILD_DEFINE_CASE(append_values_i32)
BENCH_BUILD_DEFINE_CASE(append_values_i64)
BENCH_BUILD_DEFINE_CASE(append_values_f64)
BENCH_BUILD_DEFINE_CASE(append_values_bool)
BENCH_BUILD_DEFINE_CASE(append_values_bytes)

static int compute_append_values_dictionary(const struct inputs* inputs, struct build_slot* out_slot) {
    struct dora_builder* builder = &out_slot->builder;
    int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);
    if (status == 0) {
        status = dora_builder_set_dictionary(builder, &inputs->domain);
    }
    if (status == 0) {
        status = dora_builder_append_values_u32(builder, inputs->values, nullptr, inputs->rows);
    }
    if (status == 0) {
        status = dora_builder_finish(builder, &out_slot->result);
    }
    return status;
}

BENCH_BUILD_DEFINE_CASE(append_values_dictionary)

static int compute_append_vector(const struct inputs* inputs, struct build_slot* out_slot) {
    struct dora_builder* builder = &out_slot->builder;
    int status = dora_builder_init_schema(builder, inputs->pool, &inputs->schema, inputs->options);
    if (status == 0) {
        status = dora_builder_append_vector(builder, &inputs->source);
    }
    if (status == 0) {
        status = dora_builder_finish(builder, &out_slot->result);
    }
    return status;
}

BENCH_BUILD_DEFINE_CASE(append_vector)

#undef BENCH_BUILD_DEFINE_CASE

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"append/int32", append_i32, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"append_values/int32", append_values_i32, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32}},
        {"append/int32/nulls:0.5",
         append_i32_nulls,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5}},
        {"append_values/int32/nulls:0.5",
         append_values_i32,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 0.5}},
        {"append/int32/nulls:1",
         append_i32_nulls,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 1}},
        {"append_values/int32/nulls:1",
         append_values_i32,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT32, .null_fraction = 1}},
        {"append/int64", append_i64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"append_values/int64", append_values_i64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"append/float64", append_f64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"append_values/float64", append_values_f64, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_FLOAT64}},
        {"append/boolean", append_bool, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5}},
        {"append_values/boolean",
         append_values_bool,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_BOOLEAN, .selectivity = 0.5}},
        {"append/utf8", append_bytes, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"append_values/utf8", append_values_bytes, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"append/utf8/nulls:0.5",
         append_bytes_nulls,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .null_fraction = 0.5}},
        {"append_values/utf8/nulls:0.5",
         append_values_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .null_fraction = 0.5}},
        {"append/utf8/policy:trust",
         append_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_TRUST}},
        {"append_values/utf8/policy:trust",
         append_values_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_TRUST}},
        {"append/utf8_view",
         append_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"append_values/utf8_view",
         append_values_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"append/large_utf8",
         append_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}},
        {"append_values/large_utf8",
         append_values_bytes,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}},
        {"append/fixed_binary<16>",
         append_bytes,
         {.rows = BENCH_ROWS, .type = {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16}}},
        {"append_values/fixed_binary<16>",
         append_values_bytes,
         {.rows = BENCH_ROWS, .type = {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16}}},
        {"append/dictionary<utf8>",
         append_dictionary,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_DICTIONARY}},
        {"append_values/dictionary<utf8>",
         append_values_dictionary,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_DICTIONARY}},
        {"append/list<int32>",
         append_list,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_LIST, .variant = VARIANT_LIST_LENGTHS}},
        {"append_vector/list<int32>", append_vector, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_LIST}},
        {"append/struct<int64,utf8>", append_struct, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_STRUCT}},
        {"append_vector/struct<int64,utf8>", append_vector, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_STRUCT}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
