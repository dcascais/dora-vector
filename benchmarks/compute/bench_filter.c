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
#include <stdlib.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_FLAT,
    VARIANT_UTF8_VIEW,
    VARIANT_LARGE_UTF8
};

enum selection : uint8_t {
    SELECTION_MASK,
    SELECTION_INDICES
};

struct inputs {
    struct dora_vector source;
    struct dora_vector selection;
    uint64_t result_bytes;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args, enum selection selection) {
    struct inputs inputs = {};
    const struct bench_args source_args = {.rows = args->rows, .type = args->type};
    const struct bench_args mask_args = {
        .rows = args->rows,
        .type.id = DORA_TYPE_BOOLEAN,
        .null_fraction = args->null_fraction,
        .selectivity = args->selectivity
    };
    if (args->variant == VARIANT_FLAT) {
        bench_check(bench, bench_random(bench, &source_args, 1, &inputs.source));
    } else {
        const enum dora_layout layout =
            args->variant == VARIANT_UTF8_VIEW ? DORA_LAYOUT_BINARY_VIEW : DORA_LAYOUT_LARGE_BINARY;
        bench_check(bench, bench_utf8(bench, &source_args, 1, layout, &inputs.source));
    }
    bench_check(bench,
                selection == SELECTION_INDICES ? bench_take_indices(bench, args, 2, &inputs.selection)
                                               : bench_random(bench, &mask_args, 2, &inputs.selection));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_sample(const struct bench_args* args,
                        const struct dora_vector* source,
                        const struct dora_vector* result,
                        uint64_t source_row,
                        uint64_t row) {
    bool expected_null = false;
    bool actual_null = false;
    int status;
    bool equal;
    if (args->type.id == DORA_TYPE_UTF8) {
        struct dora_bytes expected = {};
        struct dora_bytes actual = {};
        status = dora_vector_read_bytes(source, source_row, &expected, &expected_null);
        if (status == 0) {
            status = dora_vector_read_bytes(result, row, &actual, &actual_null);
        }
        equal = bench_bytes_equal(expected, actual);
    } else if (args->type.id == DORA_TYPE_FLOAT64) {
        double expected = 0;
        double actual = 0;
        status = dora_vector_read_f64(source, source_row, &expected, &expected_null);
        if (status == 0) {
            status = dora_vector_read_f64(result, row, &actual, &actual_null);
        }
        equal = actual == expected;
    } else {
        int64_t expected = 0;
        int64_t actual = 0;
        status = dora_vector_read_i64(source, source_row, &expected, &expected_null);
        if (status == 0) {
            status = dora_vector_read_i64(result, row, &actual, &actual_null);
        }
        equal = actual == expected;
    }
    return status != 0 ? status : actual_null != expected_null || (!actual_null && !equal) ? EDOM : 0;
}

static int check_rows(const struct bench_args* args,
                      const struct inputs* inputs,
                      const struct dora_vector* result,
                      const uint32_t* expected,
                      uint64_t count) {
    uint64_t length = 0;
    int status = dora_vector_get_length(result, &length);
    if (status != 0 || length != count) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t row = 0; status == 0 && row < count; ++row) {
        status = check_sample(args, &inputs->source, result, expected[row], row);
    }
    return status;
}

static int check_filter(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector_view mask = {};
    int status = args->rows <= UINT32_MAX ? dora_vector_get_view(&inputs->selection, &mask) : EOVERFLOW;
    uint32_t* expected = status == 0 ? calloc(args->rows, sizeof(*expected)) : nullptr;
    if (status == 0 && expected == nullptr) {
        status = ENOMEM;
    }
    uint64_t count = 0;
    for (uint64_t row = 0; status == 0 && row < mask.length; ++row) {
        if (!dora_vector_view_is_null(&mask, row) && dora_vector_view_get_bool(&mask, row)) {
            expected[count++] = (uint32_t)row;
        }
    }
    if (status == 0) {
        status = check_rows(args, inputs, result, expected, count);
    }
    free(expected);
    return status;
}

static int check_take(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector_view indices = {};
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(&inputs->selection, &indices);
    const uint32_t* expected = status == 0 ? dora_vector_view_get_values_u32(&indices) : nullptr;
    if (status == 0 && expected == nullptr) {
        status = EDOM;
    }
    if (status == 0) {
        status = check_rows(args, inputs, result, expected, args->rows);
    }
    if (status == 0) {
        status = dora_vector_get_view(result, &view);
    }
    return status != 0 ? status : view.selection.indices == nullptr && view.layout != DORA_LAYOUT_DICTIONARY ? 0 : EDOM;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t selection_bytes = 0;
    bench_check(bench, bench_input_bytes(&inputs->selection, &selection_bytes));
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, selection_bytes + inputs->result_bytes);
    dora_vector_release(&inputs->selection);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

struct selection_slot {
    struct dora_vector selection;
    struct dora_vector result;
};

static void release_selections(void* slots, size_t count) {
    struct selection_slot* selections = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_vector_release(&selections[slot].result);
        dora_vector_release(&selections[slot].selection);
    }
}

//--------------------------------------
// Filter
//--------------------------------------

static int compute_filter(const struct inputs* inputs, struct selection_slot* out_slot) {
    int status = dora_compute_select_mask(&inputs->selection, &out_slot->selection);
    if (status == 0) {
        status = dora_vector_init_retain_selection(&out_slot->result, &inputs->source, &out_slot->selection);
    }
    return status;
}

BENCH_DEFINE_OPERATION(filter_operation, struct selection_slot, compute_filter, release_selections)

static void filter(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args, SELECTION_MASK);
    if (bench->status == 0) {
        struct selection_slot slot = {};
        bench_check(bench, compute_filter(&inputs, &slot));
        bench_check(bench, check_filter(args, &inputs, &slot.result));
        release_selections(&slot, 1);
    }
    bench_measure(bench, &filter_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Take
//--------------------------------------

static int compute_take(const struct inputs* inputs, struct selection_slot* out_slot) {
    int status = dora_vector_init_retain_selection(&out_slot->selection, &inputs->source, &inputs->selection);
    if (status == 0) {
        status = dora_vector_decode(&out_slot->selection, &out_slot->result);
    }
    return status;
}

BENCH_DEFINE_OPERATION(take_operation, struct selection_slot, compute_take, release_selections)

static void take(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args, SELECTION_INDICES);
    if (bench->status == 0) {
        struct selection_slot slot = {};
        bench_check(bench, compute_take(&inputs, &slot));
        bench_check(bench, check_take(args, &inputs, &slot.result));
        bench_check(bench, bench_input_bytes(&slot.result, &inputs.result_bytes));
        release_selections(&slot, 1);
    }
    bench_measure(bench, &take_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"filter/int64/selectivity:0.01",
         filter,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .selectivity = 0.01}},
        {"filter/int64/selectivity:0.5", filter, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .selectivity = 0.5}},
        {"filter/int64/selectivity:0.99",
         filter,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .selectivity = 0.99}},
        {"filter/int64/mask_nulls:0.5/selectivity:0.5",
         filter,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .null_fraction = 0.5, .selectivity = 0.5}},
        {"take/int64", take, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"take/utf8", take, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"take/utf8_view", take, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"take/large_utf8", take, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_LARGE_UTF8}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
