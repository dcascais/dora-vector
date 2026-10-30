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
    VARIANT_DICTIONARY,
    VARIANT_COMBINE
};

static constexpr uint64_t HASH_SEED = 42;

struct inputs {
    struct dora_vector source;
    struct dora_vector other;
    struct dora_vector hashes;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {};
    if (args->type.id == DORA_TYPE_UTF8 && args->variant == VARIANT_FLAT) {
        bench_check(bench, bench_decoded_dictionary(bench, args, 1, &inputs.source));
    } else if (args->variant == VARIANT_DICTIONARY) {
        bench_check(bench, bench_dictionary(bench, args, 1, &inputs.source));
    } else {
        bench_check(bench, bench_random(bench, args, 1, &inputs.source));
    }
    if (args->variant == VARIANT_COMBINE) {
        const struct bench_args text_args = {.rows = args->rows, .type.id = DORA_TYPE_UTF8};
        bench_check(bench, bench_decoded_dictionary(bench, &text_args, 1, &inputs.other));
        if (bench->status == 0) {
            bench_check(bench, dora_compute_hash(&inputs.source, HASH_SEED, &inputs.hashes));
        }
    }
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_result(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector_view view = {};
    uint64_t null_count = 0;
    int status = dora_vector_get_view(result, &view);
    if (status == 0) {
        status = dora_vector_get_null_count(result, &null_count);
    }
    if (status != 0 || view.length != args->rows || view.type != DORA_TYPE_UINT64 || null_count != 0) {
        return status != 0 ? status : EDOM;
    }
    const bool is_combined = dora_vector_is_valid(&inputs->other);
    for (uint64_t row = 0; row < args->rows; ++row) {
        struct dora_vector slice = {};
        struct dora_vector other_slice = {};
        struct dora_vector hashes = {};
        struct dora_vector expected = {};
        const struct dora_range range = {.offset = row, .length = 1};
        status = dora_vector_slice(&inputs->source, range, &slice);
        if (status == 0) {
            status = dora_compute_hash(&slice, HASH_SEED, &hashes);
        }
        if (status == 0 && is_combined) {
            status = dora_vector_slice(&inputs->other, range, &other_slice);
        }
        if (status == 0 && is_combined) {
            status = dora_compute_hash_combine(&hashes, &other_slice, &expected);
        }
        uint64_t actual_value = 0;
        uint64_t expected_value = 0;
        bool is_null = false;
        if (status == 0) {
            status = dora_vector_read_u64(result, row, &actual_value, &is_null);
        }
        if (status == 0 && !is_null) {
            status = dora_vector_read_u64(is_combined ? &expected : &hashes, 0, &expected_value, &is_null);
        }
        dora_vector_release(&expected);
        dora_vector_release(&hashes);
        dora_vector_release(&other_slice);
        dora_vector_release(&slice);
        if (status != 0 || is_null || actual_value != expected_value) {
            return status != 0 ? status : EDOM;
        }
    }
    return 0;
}

static int seeded_hashes(const struct inputs* inputs, uint64_t seed, struct dora_vector* out_hashes) {
    if (!dora_vector_is_valid(&inputs->other)) {
        return dora_compute_hash(&inputs->source, seed, out_hashes);
    }
    struct dora_vector hashes = {};
    int status = dora_compute_hash(&inputs->source, seed, &hashes);
    if (status == 0) {
        status = dora_compute_hash_combine(&hashes, &inputs->other, out_hashes);
    }
    dora_vector_release(&hashes);
    return status;
}

static int check_seed(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    struct dora_vector reseeded = {};
    int status = seeded_hashes(inputs, HASH_SEED + 1, &reseeded);
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        uint64_t original = 0;
        uint64_t changed = 0;
        bool is_null = false;
        bool is_changed_null = false;
        status = dora_vector_read_u64(result, row, &original, &is_null);
        if (status == 0) {
            status = dora_vector_read_u64(&reseeded, row, &changed, &is_changed_null);
        }
        if (status == 0 && (is_null || is_changed_null || changed == original)) {
            status = EDOM;
        }
    }
    dora_vector_release(&reseeded);
    return status;
}

struct keyed_row {
    uint64_t hash;
    uint64_t row;
};

static int compare_keyed_rows(const void* left, const void* right) {
    return (((const struct keyed_row*)left)->hash > ((const struct keyed_row*)right)->hash) -
           (((const struct keyed_row*)left)->hash < ((const struct keyed_row*)right)->hash);
}

static int rows_equal(const struct dora_vector* vector, const struct keyed_row* pair, bool* out_equal) {
    struct dora_vector left_slice = {};
    struct dora_vector right_slice = {};
    int status = dora_vector_slice(vector, (struct dora_range){.offset = pair[0].row, .length = 1}, &left_slice);
    if (status == 0) {
        status = dora_vector_slice(vector, (struct dora_range){.offset = pair[1].row, .length = 1}, &right_slice);
    }
    if (status == 0) {
        status = dora_vector_is_equal(&left_slice, &right_slice, out_equal);
    }
    dora_vector_release(&right_slice);
    dora_vector_release(&left_slice);
    return status;
}

static int
check_collisions(const struct bench_args* args, const struct inputs* inputs, const struct dora_vector* result) {
    struct keyed_row* keyed = calloc(args->rows, sizeof(*keyed));
    if (keyed == nullptr) {
        return ENOMEM;
    }
    int status = 0;
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        bool is_null = false;
        keyed[row].row = row;
        status = dora_vector_read_u64(result, row, &keyed[row].hash, &is_null);
        status = status == 0 && is_null ? EDOM : status;
    }
    if (status == 0) {
        qsort(keyed, args->rows, sizeof(*keyed), compare_keyed_rows);
    }
    const bool is_combined = dora_vector_is_valid(&inputs->other);
    for (uint64_t index = 1; status == 0 && index < args->rows; ++index) {
        if (keyed[index].hash != keyed[index - 1].hash) {
            continue;
        }
        bool equal = false;
        status = rows_equal(&inputs->source, &keyed[index - 1], &equal);
        if (status == 0 && equal && is_combined) {
            status = rows_equal(&inputs->other, &keyed[index - 1], &equal);
        }
        status = status == 0 && !equal ? EDOM : status;
    }
    free(keyed);
    return status;
}

static int check_pair(struct bench* bench, const struct bench_args* args, const struct dora_vector* result) {
    if (args->type.id != DORA_TYPE_UTF8) {
        return 0;
    }
    struct dora_vector counterpart = {};
    struct dora_vector hashes = {};
    int status = args->variant == VARIANT_DICTIONARY ? bench_decoded_dictionary(bench, args, 1, &counterpart)
                                                     : bench_dictionary(bench, args, 1, &counterpart);
    if (status == 0) {
        status = dora_compute_hash(&counterpart, HASH_SEED, &hashes);
    }
    if (status == 0) {
        status = bench_vectors_equal(&hashes, result);
    }
    dora_vector_release(&hashes);
    dora_vector_release(&counterpart);
    return status;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    const bool is_combined = dora_vector_is_valid(&inputs->other);
    uint64_t bytes = 0;
    uint64_t other_bytes = 0;
    bench_check(bench, bench_input_bytes(is_combined ? &inputs->hashes : &inputs->source, &bytes));
    if (is_combined) {
        bench_check(bench, bench_input_bytes(&inputs->other, &other_bytes));
    }
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, bytes + other_bytes);
    dora_vector_release(&inputs->hashes);
    dora_vector_release(&inputs->other);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Hashes
//--------------------------------------

static int compute_hash(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_hash(&inputs->source, HASH_SEED, out_slot);
}

BENCH_DEFINE_OPERATION(hash_operation, struct dora_vector, compute_hash, bench_release_vectors)

static void hash(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct dora_vector result = {};
        bench_check(bench, compute_hash(&inputs, &result));
        bench_check(bench, check_result(args, &inputs, &result));
        bench_check(bench, check_seed(args, &inputs, &result));
        bench_check(bench, check_collisions(args, &inputs, &result));
        bench_check(bench, check_pair(bench, args, &result));
        dora_vector_release(&result);
    }
    bench_measure(bench, &hash_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Combined Hashes
//--------------------------------------

static int compute_hash_combine(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_hash_combine(&inputs->hashes, &inputs->other, out_slot);
}

BENCH_DEFINE_OPERATION(hash_combine_operation, struct dora_vector, compute_hash_combine, bench_release_vectors)

static void hash_combine(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct dora_vector result = {};
        bench_check(bench, compute_hash_combine(&inputs, &result));
        bench_check(bench, check_result(args, &inputs, &result));
        bench_check(bench, check_seed(args, &inputs, &result));
        bench_check(bench, check_collisions(args, &inputs, &result));
        dora_vector_release(&result);
    }
    bench_measure(bench, &hash_combine_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"hash/int64", hash, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"hash/utf8", hash, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"hash/dictionary<utf8>", hash, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_DICTIONARY}},
        {"hash_combine/int64,utf8",
         hash_combine,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_COMBINE}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
