// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/schema.h"
#include "dora/vector/stream.h"
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
    VARIANT_ARRAY,
    VARIANT_ROUND_TRIP,
    VARIANT_STREAM
};

static void prepare_mixed(struct bench* bench, uint64_t rows, struct dora_vector* out_vector) {
    const struct dora_schema_field fields[] = {
        {.name = "f0", .type = {.id = DORA_TYPE_UTF8}},
        {.name = "f1",
         .type =
             {.id = DORA_TYPE_TIMESTAMP,
              .parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = "UTC"}}},
        {.name = "f2", .type = {.id = DORA_TYPE_INT64}},
        {.name = "f3", .type = {.id = DORA_TYPE_INT16}},
        {.name = "f4", .type = {.id = DORA_TYPE_INT16}},
        {.name = "f5", .type = {.id = DORA_TYPE_FLOAT32}},
        {.name = "f6", .type = {.id = DORA_TYPE_FLOAT32}},
        {.name = "f7", .type = {.id = DORA_TYPE_FLOAT32}},
        {.name = "f8",
         .type = {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 19, .scale = 10, .bit_width = 128}}}
    };
    struct dora_vector children[ARRAY_COUNT(fields)] = {};
    const char* names[ARRAY_COUNT(fields)] = {};
    for (size_t index = 0; index < ARRAY_COUNT(fields); ++index) {
        names[index] = fields[index].name;
        const struct bench_args args = {.rows = rows, .type = fields[index].type};
        bench_check(bench, bench_random(bench, &args, index + 1, &children[index]));
    }
    bench_check(bench,
                dora_vector_init_retain_struct(out_vector, children, names, ARRAY_COUNT(children), nullptr, rows));
    for (size_t index = 0; index < ARRAY_COUNT(children); ++index) {
        dora_vector_release(&children[index]);
    }
}

struct inputs {
    struct dora_pool pool;
    struct dora_schema schema;
    struct dora_vector source;
    struct dora_vector batches[BENCH_BATCH_COUNT];
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {.pool = bench->pool};
    if (args->variant == VARIANT_STREAM) {
        bench_check(bench, bench_batches(bench, ARRAY_COUNT(inputs.batches), inputs.batches));
        bench_check(bench, dora_vector_retain_schema(&inputs.batches[0], &inputs.schema));
        return inputs;
    }
    if (args->type.id == DORA_TYPE_STRUCT) {
        prepare_mixed(bench, args->rows, &inputs.source);
    } else {
        bench_check(bench, bench_random(bench, args, 1, &inputs.source));
    }
    bench_check(bench, dora_vector_retain_schema(&inputs.source, &inputs.schema));
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_buffers(const struct dora_vector_view* view, const struct ArrowArray* array) {
    const uint8_t* validity = array->buffers[0];
    for (uint64_t row = 0; row < view->length; ++row) {
        const bool is_valid = validity == nullptr || dora_bitmap_test(validity, (uint64_t)array->offset + row);
        if (is_valid == dora_vector_view_is_null(view, row)) {
            return EDOM;
        }
    }
    if (view->layout == DORA_LAYOUT_FIXED) {
        size_t width = 0;
        const void* expected = bench_view_values(view, &width);
        const uint8_t* actual = array->buffers[1];
        return expected != nullptr && actual != nullptr &&
                       memcmp(actual + (uint64_t)array->offset * width, expected, view->length * width) == 0
                   ? 0
                   : EDOM;
    }
    if (view->layout == DORA_LAYOUT_BINARY) {
        const int32_t* offsets = array->buffers[1];
        const uint8_t* data = array->buffers[2];
        if (offsets == nullptr || data == nullptr) {
            return EDOM;
        }
        for (uint64_t row = 0; row < view->length; ++row) {
            const uint64_t slot = (uint64_t)array->offset + row;
            const struct dora_bytes expected = dora_vector_view_get_bytes(view, row);
            const struct dora_bytes actual = {
                .data = data + offsets[slot],
                .byte_count = (size_t)(offsets[slot + 1] - offsets[slot])
            };
            if (!dora_vector_view_is_null(view, row) && !bench_bytes_equal(expected, actual)) {
                return EDOM;
            }
        }
        return 0;
    }
    return view->layout == DORA_LAYOUT_STRUCT ? 0 : ENOTSUP;
}

static int check_array(const struct dora_vector* source, const struct ArrowArray* array) {
    struct dora_vector_view view;
    uint64_t null_count = 0;
    uint32_t child_count = 0;
    int status = dora_vector_get_view(source, &view);
    if (status == 0) {
        status = dora_vector_get_null_count(source, &null_count);
    }
    if (status == 0) {
        status = dora_vector_get_child_count(source, &child_count);
    }
    if (status != 0) {
        return status;
    }
    const int64_t buffer_count = view.layout == DORA_LAYOUT_STRUCT ? 1 : view.layout == DORA_LAYOUT_BINARY ? 3 : 2;
    if (array->length != (int64_t)view.length ||
        (array->null_count != -1 && array->null_count != (int64_t)null_count) ||
        array->offset != (int64_t)view.offset || array->n_buffers != buffer_count ||
        array->n_children != (int64_t)child_count || array->dictionary != nullptr) {
        return EDOM;
    }
    status = check_buffers(&view, array);
    for (uint32_t index = 0; status == 0 && index < child_count; ++index) {
        struct dora_vector child = {};
        status = dora_vector_get_child(source, index, &child);
        if (status == 0) {
            status = check_array(&child, array->children[index]);
        }
        dora_vector_release(&child);
    }
    return status;
}

static void release_array(struct ArrowArray* array) {
    if (array->release != nullptr) {
        array->release(array);
    }
}

static int export_stream(const struct inputs* inputs,
                         struct bench_batch_source* out_source,
                         struct dora_reader* out_reader,
                         struct ArrowArrayStream* out_stream) {
    bench_batch_source_init(out_source, inputs->batches, ARRAY_COUNT(inputs->batches));
    int status = dora_reader_init(out_reader, inputs->pool, &out_source->source);
    if (status == 0) {
        status = dora_arrow_export_stream(&inputs->schema, out_reader, out_stream);
    }
    return status;
}

static int check_stream(const struct inputs* inputs) {
    struct bench_batch_source source = {};
    struct dora_reader reader = {};
    struct ArrowArrayStream stream = {};
    size_t count = 0;
    int status = export_stream(inputs, &source, &reader, &stream);
    while (status == 0) {
        struct ArrowArray array = {};
        status = stream.get_next(&stream, &array);
        if (status != 0 || array.release == nullptr) {
            break;
        }
        status = count < ARRAY_COUNT(inputs->batches) ? check_array(&inputs->batches[count], &array) : EOVERFLOW;
        release_array(&array);
        ++count;
    }
    if (stream.release != nullptr) {
        stream.release(&stream);
    }
    dora_reader_release(&reader);
    return status != 0 ? status : count == ARRAY_COUNT(inputs->batches) ? 0 : EDOM;
}

static constexpr struct dora_arrow_limits IMPORT_LIMITS = {
    .max_depth = 64,
    .max_node_count = 1'000'000,
    .max_edge_count = 1'000'000,
    .max_coordinate_work_count = 100'000'000,
    .max_output_bytes = SIZE_MAX
};

static int check_exported(const struct inputs* inputs, struct ArrowArray* array) {
    struct dora_vector imported = {};
    int status = dora_arrow_import_move_array(inputs->pool, &inputs->schema, array, &IMPORT_LIMITS, &imported, nullptr);
    if (status == 0 && array->release != nullptr) {
        status = EDOM;
    }
    if (status == 0) {
        status = bench_vectors_equal(&inputs->source, &imported);
    }
    dora_vector_release(&imported);
    return status;
}

static int
check_imported(const struct inputs* inputs, const struct ArrowArray* array, const struct dora_vector* result) {
    return array->release == nullptr ? bench_vectors_equal(&inputs->source, result) : EDOM;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t items = args->variant == VARIANT_STREAM ? ARRAY_COUNT(inputs->batches) : 1;
    uint64_t bytes = 0;
    if (args->variant == VARIANT_ROUND_TRIP && args->type.id != DORA_TYPE_INT64) {
        items = args->rows;
        bench_check(bench, bench_input_bytes(&inputs->source, &bytes));
    }
    bench_set_items(bench, items);
    bench_set_bytes(bench, bytes);
    bench_release_vectors(inputs->batches, ARRAY_COUNT(inputs->batches));
    dora_vector_release(&inputs->source);
    dora_schema_release(&inputs->schema);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Arrays
//--------------------------------------

static void release_arrays(void* slots, size_t count) {
    struct ArrowArray* arrays = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        release_array(&arrays[slot]);
    }
}

static int compute_export_array(const struct inputs* inputs, struct ArrowArray* out_slot) {
    return dora_arrow_export_array(&inputs->source, out_slot);
}

BENCH_DEFINE_OPERATION(export_array_operation, struct ArrowArray, compute_export_array, release_arrays)

static void export_array(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct ArrowArray array = {};
        bench_check(bench, compute_export_array(&inputs, &array));
        bench_check(bench, check_array(&inputs.source, &array));
        bench_check(bench, check_exported(&inputs, &array));
        release_array(&array);
    }
    bench_measure(bench, &export_array_operation, &inputs);
    finish(bench, args, &inputs);
}

struct round_trip_slot {
    struct ArrowArray array;
    struct dora_vector result;
};

static void release_round_trips(void* slots, size_t count) {
    struct round_trip_slot* round_trips = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_vector_release(&round_trips[slot].result);
        release_array(&round_trips[slot].array);
    }
}

static int compute_export_import_array(const struct inputs* inputs, struct round_trip_slot* out_slot) {
    int status = compute_export_array(inputs, &out_slot->array);
    if (status == 0) {
        status = dora_arrow_import_move_array(inputs->pool,
                                              &inputs->schema,
                                              &out_slot->array,
                                              &IMPORT_LIMITS,
                                              &out_slot->result,
                                              nullptr);
    }
    return status;
}

BENCH_DEFINE_OPERATION(export_import_array_operation,
                       struct round_trip_slot,
                       compute_export_import_array,
                       release_round_trips)

static void export_import_array(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct round_trip_slot slot = {};
        struct ArrowArray exported = {};
        bench_check(bench, compute_export_import_array(&inputs, &slot));
        bench_check(bench, check_imported(&inputs, &slot.array, &slot.result));
        bench_check(bench, compute_export_array(&inputs, &exported));
        bench_check(bench, check_array(&inputs.source, &exported));
        release_array(&exported);
        release_round_trips(&slot, 1);
    }
    bench_measure(bench, &export_import_array_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Streams
//--------------------------------------

struct stream_slot {
    struct bench_batch_source source;
    struct dora_reader reader;
    struct ArrowArrayStream stream;
    struct dora_reader imported;
    struct dora_vector batches[BENCH_BATCH_COUNT];
    size_t count;
};

static void release_streams(void* slots, size_t count) {
    struct stream_slot* streams = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        bench_release_vectors(streams[slot].batches, ARRAY_COUNT(streams[slot].batches));
        dora_reader_release(&streams[slot].imported);
        if (streams[slot].stream.release != nullptr) {
            streams[slot].stream.release(&streams[slot].stream);
        }
        dora_reader_release(&streams[slot].reader);
    }
}

static int compute_export_import_stream(const struct inputs* inputs, struct stream_slot* out_slot) {
    int status = export_stream(inputs, &out_slot->source, &out_slot->reader, &out_slot->stream);
    if (status == 0) {
        status =
            dora_arrow_import_stream(inputs->pool, &out_slot->stream, &IMPORT_LIMITS, &out_slot->imported, nullptr);
    }
    if (status == 0) {
        status =
            bench_collect(&out_slot->imported, ARRAY_COUNT(out_slot->batches), out_slot->batches, &out_slot->count);
    }
    return status;
}

BENCH_DEFINE_OPERATION(export_import_stream_operation,
                       struct stream_slot,
                       compute_export_import_stream,
                       release_streams)

static void export_import_stream(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct stream_slot slot = {};
        bench_check(bench, compute_export_import_stream(&inputs, &slot));
        bench_check(bench, bench_batches_equal(inputs.batches, ARRAY_COUNT(inputs.batches), slot.batches, slot.count));
        bench_check(bench, check_stream(&inputs));
        release_streams(&slot, 1);
    }
    bench_measure(bench, &export_import_stream_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"export/int64", export_array, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64}},
        {"export/utf8", export_array, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"export/mixed", export_array, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_STRUCT}},
        {"export_import/int64",
         export_import_array,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_INT64, .variant = VARIANT_ROUND_TRIP}},
        {"export_import/utf8",
         export_import_array,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_ROUND_TRIP}},
        {"export_import/mixed",
         export_import_array,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_STRUCT, .variant = VARIANT_ROUND_TRIP}},
        {"stream/export_import", export_import_stream, {.rows = BENCH_ROWS, .variant = VARIANT_STREAM}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
