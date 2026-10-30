// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
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

struct inputs {
    struct dora_vector source;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {};
    bench_check(bench, bench_unchecked_utf8(bench, args, 1, &inputs.source));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_length(const struct bench_args* args, const struct inputs* inputs) {
    uint64_t length = 0;
    const int status = dora_vector_get_length(&inputs->source, &length);
    return status != 0 ? status : length == args->rows ? 0 : EDOM;
}

static int check_rejects_invalid(struct bench* bench, const struct bench_args* args) {
    struct dora_vector invalid = {};
    struct dora_error error = {};
    int status = bench_invalid_utf8(bench, args, 1, &invalid);
    if (status == 0) {
        status = dora_vector_validate(&invalid, &error);
        if (status == EILSEQ) {
            status = error.message[0] != 0 ? 0 : EDOM;
        } else if (status == 0) {
            status = EDOM;
        }
    }
    dora_vector_release(&invalid);
    return status;
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

static int compute_validate(const struct inputs* inputs) {
    return dora_vector_validate(&inputs->source, nullptr);
}

BENCH_DEFINE_STATUS_OPERATION(validate_operation, compute_validate)

static void validate(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        bench_check(bench, compute_validate(&inputs));
        bench_check(bench, check_length(args, &inputs));
        bench_check(bench, check_rejects_invalid(bench, args));
    }
    bench_measure(bench, &validate_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"validate/utf8/ascii",
         validate,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_ASCII}},
        {"validate/utf8/unicode",
         validate,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_UNICODE}},
        {"validate/list<utf8>", validate, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_LIST}},
        {"validate/struct<int64,utf8>", validate, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_STRUCT}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
