// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "types/array_count.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_NATIVE,
    VARIANT_CSV_PIPELINE
};

static constexpr int64_t FILTER_THRESHOLD = BENCH_INTEGER_MIDPOINT;

struct inputs {
    struct dora_pool pool;
    struct dora_vector batches[BENCH_BATCH_COUNT];
    struct dora_allocation text;
    size_t byte_count;
    struct dora_schema schema;
    struct dora_vector bound;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {.pool = bench->pool};
    if (args->variant == VARIANT_NATIVE) {
        bench_check(bench, bench_batches(bench, ARRAY_COUNT(inputs.batches), inputs.batches));
        return inputs;
    }
    const struct bench_text_args text_args =
        {.rows = args->rows, .format = BENCH_TEXT_FORMAT_CSV, .columns = BENCH_TEXT_COLUMNS_NUMERIC};
    bench_check(bench, bench_text(bench, &text_args, &inputs.text, &inputs.byte_count));
    bench_check(bench, bench_text_schema(bench, BENCH_TEXT_COLUMNS_NUMERIC, &inputs.schema));
    struct dora_vector threshold = {};
    bench_check(bench,
                dora_vector_init_copy_i64(&threshold,
                                          inputs.pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_INT64},
                                          &FILTER_THRESHOLD,
                                          nullptr,
                                          1));
    bench_check(bench, args->rows % BENCH_BATCH_ROWS == 0 ? 0 : EINVAL);
    bench_check(bench, dora_vector_init_retain_constant(&inputs.bound, &threshold, BENCH_BATCH_ROWS));
    dora_vector_release(&threshold);
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_sum(const struct inputs* inputs, int64_t actual) {
    const char* text = inputs->text.address;
    if (text == nullptr) {
        return EINVAL;
    }
    const char* end = text + inputs->byte_count;
    const char* row = text + bench_line_end(text, inputs->byte_count, 1);
    int64_t expected = 0;
    while (row != nullptr && row < end) {
        const int64_t id = strtoll(row, nullptr, 10);
        expected += id < FILTER_THRESHOLD ? id : 0;
        row = memchr(row, '\n', (size_t)(end - row));
        row = row != nullptr ? row + 1 : nullptr;
    }
    return actual == expected ? 0 : EDOM;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    bench_set_items(bench, args->variant == VARIANT_NATIVE ? ARRAY_COUNT(inputs->batches) : args->rows);
    bench_set_bytes(bench, inputs->byte_count);
    bench_release_vectors(inputs->batches, ARRAY_COUNT(inputs->batches));
    dora_vector_release(&inputs->bound);
    dora_schema_release(&inputs->schema);
    dora_allocation_release(&inputs->text);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Readers
//--------------------------------------

struct drain_slot {
    struct bench_batch_source source;
    struct dora_reader reader;
    struct dora_vector batches[BENCH_BATCH_COUNT];
    size_t count;
};

static void release_drains(void* slots, size_t count) {
    struct drain_slot* drains = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        bench_release_vectors(drains[slot].batches, ARRAY_COUNT(drains[slot].batches));
        dora_reader_release(&drains[slot].reader);
    }
}

static int compute_drain(const struct inputs* inputs, struct drain_slot* out_slot) {
    bench_batch_source_init(&out_slot->source, inputs->batches, ARRAY_COUNT(inputs->batches));
    int status = dora_reader_init(&out_slot->reader, inputs->pool, &out_slot->source.source);
    if (status == 0) {
        status = bench_collect(&out_slot->reader, ARRAY_COUNT(out_slot->batches), out_slot->batches, &out_slot->count);
    }
    return status;
}

BENCH_DEFINE_OPERATION(drain_operation, struct drain_slot, compute_drain, release_drains)

static void drain(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct drain_slot slot = {};
        bench_check(bench, compute_drain(&inputs, &slot));
        bench_check(bench, bench_batches_equal(inputs.batches, ARRAY_COUNT(inputs.batches), slot.batches, slot.count));
        release_drains(&slot, 1);
    }
    bench_measure(bench, &drain_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Pipelines
//--------------------------------------

struct filter_sum {
    struct dora_vector_transform transform;
    struct dora_vector threshold;
    int64_t sum;
};

static int select_sum(struct dora_vector_transform* transform,
                      struct dora_pool,
                      const struct dora_vector* input,
                      bool is_end,
                      struct dora_vector*,
                      struct dora_stream_result* out_result,
                      struct dora_error*) {
    struct filter_sum* state = (struct filter_sum*)transform;
    if (input == nullptr) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
        return is_end ? 0 : EINVAL;
    }
    struct dora_vector ids = {};
    struct dora_vector selection = {};
    struct dora_vector selected = {};
    struct dora_vector sum = {};
    int status = dora_vector_get_child(input, 0, &ids);
    if (status == 0) {
        status = dora_compute_select(&ids, DORA_COMPARE_OP_LT, &state->threshold, &selection);
    }
    if (status == 0) {
        status = dora_vector_init_retain_selection(&selected, &ids, &selection);
    }
    if (status == 0) {
        status = dora_compute_sum(&selected, &sum);
    }
    struct dora_i128 value = {};
    bool is_null = false;
    if (status == 0) {
        status = dora_vector_read_i128(&sum, 0, &value, &is_null);
    }
    if (status == 0 && !is_null) {
        const int64_t low = (int64_t)value.limbs[0];
        if (value.limbs[1] != (low < 0 ? UINT64_MAX : 0) || ckd_add(&state->sum, state->sum, low)) {
            status = EOVERFLOW;
        }
    }
    if (status == 0) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY, .is_input_consumed = true};
    }
    dora_vector_release(&sum);
    dora_vector_release(&selected);
    dora_vector_release(&selection);
    dora_vector_release(&ids);
    return status;
}

static void release_transform(struct dora_vector_transform* transform) {
    *transform = (struct dora_vector_transform){};
}

struct pipeline_slot {
    struct dora_byte_source_memory source;
    struct dora_reader reader;
    struct dora_pipeline pipeline;
    struct filter_sum state;
};

static void release_pipelines(void* slots, size_t count) {
    struct pipeline_slot* pipelines = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_pipeline_release(&pipelines[slot].pipeline);
        dora_reader_release(&pipelines[slot].reader);
        bench_release_byte_source(&pipelines[slot].source.source);
    }
}

static int compute_pipeline(const struct inputs* inputs, struct pipeline_slot* out_slot) {
    out_slot->state = (struct filter_sum){
        .transform = {.apply = select_sum, .release = release_transform},
        .threshold = dora_vector_borrow(&inputs->bound)
    };
    struct dora_vector_transform* const transforms[] = {&out_slot->state.transform};
    int status = dora_byte_source_memory_init(
        &out_slot->source,
        (struct dora_bytes){.data = inputs->text.address, .byte_count = inputs->byte_count});
    if (status == 0) {
        status = dora_csv_reader_init(&out_slot->reader,
                                      inputs->pool,
                                      &inputs->schema,
                                      &out_slot->source.source,
                                      &BENCH_CSV_OPTIONS,
                                      nullptr);
    }
    if (status == 0) {
        status = dora_pipeline_init(&out_slot->pipeline,
                                    inputs->pool,
                                    &out_slot->reader,
                                    transforms,
                                    ARRAY_COUNT(transforms),
                                    nullptr);
    }
    while (status == 0) {
        struct dora_pipeline_observation observation = {};
        status = dora_pipeline_step(&out_slot->pipeline, &observation, nullptr);
        if (status != 0 || observation.progress == DORA_PIPELINE_PROGRESS_FINISHED) {
            break;
        }
        if (observation.progress != DORA_PIPELINE_PROGRESS_ADVANCED) {
            status = EDOM;
        }
    }
    return status;
}

BENCH_DEFINE_OPERATION(pipeline_operation, struct pipeline_slot, compute_pipeline, release_pipelines)

static void pipeline(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct pipeline_slot slot = {};
        bench_check(bench, compute_pipeline(&inputs, &slot));
        bench_check(bench, check_sum(&inputs, slot.state.sum));
        release_pipelines(&slot, 1);
    }
    bench_measure(bench, &pipeline_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"drain/native", drain, {.rows = BENCH_ROWS}},
        {"pipeline/csv_filter_sum", pipeline, {.rows = BENCH_ROWS, .variant = VARIANT_CSV_PIPELINE}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
