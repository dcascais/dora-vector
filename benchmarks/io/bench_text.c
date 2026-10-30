// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/io/json.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "types/array_count.h"

#include <errno.h>
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
    VARIANT_READ,
    VARIANT_READ_NUMERIC,
    VARIANT_WRITE,
    VARIANT_INFER
};

struct inputs {
    struct dora_pool pool;
    enum bench_text_format format;
    struct dora_allocation text;
    size_t byte_count;
    size_t sample_byte_count;
    struct dora_schema schema;
    struct dora_vector batches[BENCH_BATCH_COUNT];
};

static struct dora_bytes text_bytes(const struct inputs* inputs) {
    return (struct dora_bytes){.data = inputs->text.address, .byte_count = inputs->byte_count};
}

static constexpr struct dora_json_options JSON_OPTIONS = {
    .format = DORA_JSON_FORMAT_NEWLINE_DELIMITED,
    .row_capacity = BENCH_BATCH_ROWS
};

static int init_reader(struct dora_reader* reader,
                       struct dora_byte_source_memory* source,
                       const struct inputs* inputs,
                       struct dora_bytes bytes) {
    int status = dora_byte_source_memory_init(source, bytes);
    if (status == 0) {
        switch (inputs->format) {
        case BENCH_TEXT_FORMAT_CSV:
            status = dora_csv_reader_init(reader,
                                          inputs->pool,
                                          &inputs->schema,
                                          &source->source,
                                          &BENCH_CSV_OPTIONS,
                                          nullptr);
            break;
        case BENCH_TEXT_FORMAT_NDJSON:
            status =
                dora_json_reader_init(reader, inputs->pool, &inputs->schema, &source->source, &JSON_OPTIONS, nullptr);
            break;
        default:
            status = EINVAL;
            break;
        }
    }
    return status;
}

static void load_batches(struct bench* bench, struct inputs* inputs) {
    struct dora_byte_source_memory source = {};
    struct dora_reader reader = {};
    size_t count = 0;
    int status = init_reader(&reader, &source, inputs, text_bytes(inputs));
    if (status == 0) {
        status = bench_collect(&reader, ARRAY_COUNT(inputs->batches), inputs->batches, &count);
    }
    bench_check(bench, status != 0 ? status : count == ARRAY_COUNT(inputs->batches) ? 0 : EDOM);
    for (size_t index = 0; index < count; ++index) {
        uint64_t rows = 0;
        bench_check(bench, dora_vector_get_length(&inputs->batches[index], &rows));
        bench_check(bench, rows == BENCH_BATCH_ROWS ? 0 : EDOM);
    }
    dora_reader_release(&reader);
    bench_release_byte_source(&source.source);
}

static constexpr uint64_t INFERENCE_SAMPLE_ROWS = 20'480;

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = {.pool = bench->pool, .format = args->format};
    const enum bench_text_columns columns =
        args->variant == VARIANT_READ_NUMERIC ? BENCH_TEXT_COLUMNS_NUMERIC : BENCH_TEXT_COLUMNS_MIXED;
    const struct bench_text_args text_args = {.rows = args->rows, .format = args->format, .columns = columns};
    bench_check(bench, bench_text(bench, &text_args, &inputs.text, &inputs.byte_count));
    if (args->variant != VARIANT_INFER) {
        bench_check(bench, bench_text_schema(bench, columns, &inputs.schema));
    }
    if (args->variant == VARIANT_WRITE) {
        load_batches(bench, &inputs);
    } else if (args->variant == VARIANT_INFER && inputs.text.address != nullptr) {
        const uint64_t header_lines = args->format == BENCH_TEXT_FORMAT_CSV ? 1 : 0;
        inputs.sample_byte_count =
            bench_line_end(inputs.text.address, inputs.byte_count, header_lines + INFERENCE_SAMPLE_ROWS);
    }
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static constexpr int64_t NANOSECONDS_PER_MICROSECOND = 1000;

static int check_column(const struct dora_vector* column, const struct bench_text_row* expected, uint64_t rows) {
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(column, &type);
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        bool is_null = false;
        bool equal = false;
        switch (type.id) {
        case DORA_TYPE_INT64: {
            int64_t value = 0;
            status = dora_vector_read_i64(column, row, &value, &is_null);
            equal = value == expected[row].id;
            break;
        }
        case DORA_TYPE_FLOAT64: {
            double value = 0;
            status = dora_vector_read_f64(column, row, &value, &is_null);
            equal = value == expected[row].value;
            break;
        }
        case DORA_TYPE_UTF8: {
            struct dora_bytes value = {};
            const struct dora_bytes text = {
                .data = (const uint8_t*)expected[row].text,
                .byte_count = strlen(expected[row].text)
            };
            status = dora_vector_read_bytes(column, row, &value, &is_null);
            equal = bench_bytes_equal(value, text);
            break;
        }
        case DORA_TYPE_BOOLEAN: {
            bool value = false;
            status = dora_vector_read_bool(column, row, &value, &is_null);
            equal = value == expected[row].flag;
            break;
        }
        case DORA_TYPE_TIMESTAMP: {
            const int64_t ticks_per_microsecond =
                type.parameters.timestamp.unit == DORA_TIME_UNIT_NANOSECOND ? NANOSECONDS_PER_MICROSECOND : 1;
            int64_t value = 0;
            status = dora_vector_read_i64(column, row, &value, &is_null);
            equal = value == expected[row].ts_microseconds * ticks_per_microsecond;
            break;
        }
        default:
            status = ENOTSUP;
            break;
        }
        if (status == 0 && (is_null || !equal)) {
            status = EDOM;
        }
    }
    return status;
}

static int check_batches(const struct bench_args* args,
                         const struct dora_schema* schema,
                         const struct dora_vector* batches,
                         size_t count) {
    uint32_t columns = 0;
    int status = dora_schema_get_child_count(schema, &columns);
    if (status != 0) {
        return status;
    }
    struct bench_text_row* expected = calloc(args->rows, sizeof(*expected));
    if (expected == nullptr) {
        return ENOMEM;
    }
    bench_text_rows(args->rows, expected);
    uint64_t first = 0;
    for (size_t index = 0; status == 0 && index < count; ++index) {
        uint32_t column_count = 0;
        uint64_t length = 0;
        status = dora_vector_get_child_count(&batches[index], &column_count);
        if (status == 0 && column_count != columns) {
            status = EDOM;
        }
        if (status == 0) {
            status = dora_vector_get_length(&batches[index], &length);
        }
        if (status == 0 && length > args->rows - first) {
            status = EDOM;
        }
        for (uint32_t column = 0; status == 0 && column < column_count; ++column) {
            struct dora_vector child = {};
            status = dora_vector_get_child(&batches[index], column, &child);
            if (status == 0) {
                status = check_column(&child, expected + first, length);
            }
            dora_vector_release(&child);
        }
        first += length;
    }
    free(expected);
    return status != 0 ? status : first == args->rows ? 0 : EDOM;
}

static int check_written(const struct inputs* inputs, const struct dora_byte_sink_memory* sink) {
    if (sink->buffer.address == nullptr || sink->byte_count == 0) {
        return EDOM;
    }
    struct dora_byte_source_memory source = {};
    struct dora_reader reader = {};
    struct dora_vector batches[BENCH_BATCH_COUNT] = {};
    size_t count = 0;
    int status = init_reader(&reader,
                             &source,
                             inputs,
                             (struct dora_bytes){.data = sink->buffer.address, .byte_count = sink->byte_count});
    if (status == 0) {
        status = bench_collect(&reader, ARRAY_COUNT(batches), batches, &count);
    }
    if (status == 0) {
        status = bench_batches_equal(inputs->batches, ARRAY_COUNT(inputs->batches), batches, count);
    }
    bench_release_vectors(batches, ARRAY_COUNT(batches));
    dora_reader_release(&reader);
    bench_release_byte_source(&source.source);
    return status;
}

static int check_inferred(const struct bench_args* args, const struct dora_schema* schema, struct dora_reader* reader) {
    const enum dora_type types[] =
        {DORA_TYPE_INT64, DORA_TYPE_FLOAT64, DORA_TYPE_UTF8, DORA_TYPE_BOOLEAN, DORA_TYPE_TIMESTAMP};

    uint32_t count = 0;
    int status = dora_schema_get_child_count(schema, &count);
    if (status != 0 || count != ARRAY_COUNT(types)) {
        return status != 0 ? status : EDOM;
    }
    for (uint32_t index = 0; index < ARRAY_COUNT(types); ++index) {
        struct dora_schema child = {};
        struct dora_schema_field field = {};
        status = dora_schema_get_child(schema, index, &child);
        if (status == 0) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == 0 &&
            (field.type.id != types[index] || (field.type.id == DORA_TYPE_TIMESTAMP &&
                                               field.type.parameters.timestamp.unit != DORA_TIME_UNIT_NANOSECOND))) {
            status = EDOM;
        }
        dora_schema_release(&child);
        if (status != 0) {
            return status;
        }
    }
    struct dora_vector batches[BENCH_BATCH_COUNT] = {};
    size_t batch_count = 0;
    status = bench_collect(reader, ARRAY_COUNT(batches), batches, &batch_count);
    if (status == 0) {
        status = check_batches(args, schema, batches, batch_count);
    }
    bench_release_vectors(batches, ARRAY_COUNT(batches));
    return status;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t rows = args->rows;
    uint64_t bytes = inputs->byte_count;
    if (args->variant == VARIANT_INFER) {
        rows = rows < INFERENCE_SAMPLE_ROWS ? rows : INFERENCE_SAMPLE_ROWS;
        bytes = inputs->sample_byte_count;
    } else if (args->variant == VARIANT_WRITE) {
        bytes = 0;
        for (size_t index = 0; index < ARRAY_COUNT(inputs->batches); ++index) {
            uint64_t batch_bytes = 0;
            bench_check(bench, bench_input_bytes(&inputs->batches[index], &batch_bytes));
            bytes += batch_bytes;
        }
    }
    bench_set_items(bench, rows);
    bench_set_bytes(bench, bytes);
    bench_release_vectors(inputs->batches, ARRAY_COUNT(inputs->batches));
    dora_schema_release(&inputs->schema);
    dora_allocation_release(&inputs->text);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Reading
//--------------------------------------

struct read_slot {
    struct dora_byte_source_memory source;
    struct dora_reader reader;
    struct dora_vector batches[BENCH_BATCH_COUNT];
    size_t count;
};

static void release_reads(void* slots, size_t count) {
    struct read_slot* reads = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        bench_release_vectors(reads[slot].batches, ARRAY_COUNT(reads[slot].batches));
        dora_reader_release(&reads[slot].reader);
        bench_release_byte_source(&reads[slot].source.source);
    }
}

#define BENCH_TEXT_DEFINE_READ(format_tag, init_function, options_address)                                             \
    static int compute_read_##format_tag(const struct inputs* inputs, struct read_slot* out_slot) {                    \
        int status = dora_byte_source_memory_init(&out_slot->source, text_bytes(inputs));                              \
        if (status == 0) {                                                                                             \
            status = init_function(&out_slot->reader,                                                                  \
                                   inputs->pool,                                                                       \
                                   &inputs->schema,                                                                    \
                                   &out_slot->source.source,                                                           \
                                   options_address,                                                                    \
                                   nullptr);                                                                           \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            status =                                                                                                   \
                bench_collect(&out_slot->reader, ARRAY_COUNT(out_slot->batches), out_slot->batches, &out_slot->count); \
        }                                                                                                              \
        return status;                                                                                                 \
    }                                                                                                                  \
    BENCH_DEFINE_OPERATION(read_##format_tag##_operation, struct read_slot, compute_read_##format_tag, release_reads)  \
    static void read_##format_tag(struct bench* bench, const struct bench_args* args) {                                \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct read_slot slot = {};                                                                                \
            bench_check(bench, compute_read_##format_tag(&inputs, &slot));                                             \
            bench_check(bench, check_batches(args, &inputs.schema, slot.batches, slot.count));                         \
            release_reads(&slot, 1);                                                                                   \
        }                                                                                                              \
        bench_measure(bench, &read_##format_tag##_operation, &inputs);                                                 \
        finish(bench, args, &inputs);                                                                                  \
    }

BENCH_TEXT_DEFINE_READ(csv, dora_csv_reader_init, &BENCH_CSV_OPTIONS)
BENCH_TEXT_DEFINE_READ(ndjson, dora_json_reader_init, &JSON_OPTIONS)

#undef BENCH_TEXT_DEFINE_READ

//--------------------------------------
// Writing
//--------------------------------------

struct write_slot {
    struct dora_byte_sink_memory sink;
    struct dora_writer writer;
};

static void release_writes(void* slots, size_t count) {
    struct write_slot* writes = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_writer_release(&writes[slot].writer);
    }
}

#define BENCH_TEXT_DEFINE_WRITE(format_tag, init_function, options_address)                                            \
    static int compute_write_##format_tag(const struct inputs* inputs, struct write_slot* out_slot) {                  \
        int status = dora_byte_sink_memory_init(&out_slot->sink, inputs->pool);                                        \
        if (status == 0) {                                                                                             \
            status = init_function(&out_slot->writer, inputs->pool, &out_slot->sink.sink, options_address, nullptr);   \
        }                                                                                                              \
        struct dora_stream_result flow = {};                                                                           \
        for (size_t index = 0; status == 0 && index < ARRAY_COUNT(inputs->batches); ++index) {                         \
            status = dora_writer_push(&out_slot->writer, &inputs->batches[index], &flow, nullptr);                     \
            if (status == 0 && (flow.status != DORA_STREAM_STATUS_READY || !flow.is_input_consumed)) {                 \
                status = EDOM;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            status = dora_writer_finish(&out_slot->writer, &flow, nullptr);                                            \
        }                                                                                                              \
        if (status == 0 && flow.status != DORA_STREAM_STATUS_END) {                                                    \
            status = EDOM;                                                                                             \
        }                                                                                                              \
        return status;                                                                                                 \
    }                                                                                                                  \
    BENCH_DEFINE_OPERATION(write_##format_tag##_operation,                                                             \
                           struct write_slot,                                                                          \
                           compute_write_##format_tag,                                                                 \
                           release_writes)                                                                             \
    static void write_##format_tag(struct bench* bench, const struct bench_args* args) {                               \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct write_slot slot = {};                                                                               \
            bench_check(bench, compute_write_##format_tag(&inputs, &slot));                                            \
            bench_check(bench, check_written(&inputs, &slot.sink));                                                    \
            release_writes(&slot, 1);                                                                                  \
        }                                                                                                              \
        bench_measure(bench, &write_##format_tag##_operation, &inputs);                                                \
        finish(bench, args, &inputs);                                                                                  \
    }

BENCH_TEXT_DEFINE_WRITE(csv, dora_csv_writer_init, &BENCH_CSV_OPTIONS)
BENCH_TEXT_DEFINE_WRITE(ndjson, dora_json_writer_init, &JSON_OPTIONS)

#undef BENCH_TEXT_DEFINE_WRITE

//--------------------------------------
// Inference
//--------------------------------------

struct infer_slot {
    struct dora_byte_source_memory source;
    struct dora_schema_inference inference;
    struct dora_schema schema;
    struct dora_reader reader;
};

static void release_inferences(void* slots, size_t count) {
    struct infer_slot* inferences = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_schema_inference_release(&inferences[slot].inference);
        dora_reader_release(&inferences[slot].reader);
        dora_schema_release(&inferences[slot].schema);
        bench_release_byte_source(&inferences[slot].source.source);
    }
}

static constexpr struct dora_schema_inference_options INFERENCE_OPTIONS = {.sample_row_limit = INFERENCE_SAMPLE_ROWS};

#define BENCH_TEXT_DEFINE_INFER(format_tag, init_function, options_address)                                            \
    static int compute_infer_##format_tag(const struct inputs* inputs, struct infer_slot* out_slot) {                  \
        int status = dora_byte_source_memory_init(&out_slot->source, text_bytes(inputs));                              \
        if (status == 0) {                                                                                             \
            status = init_function(&out_slot->inference,                                                               \
                                   inputs->pool,                                                                       \
                                   &out_slot->source.source,                                                           \
                                   options_address,                                                                    \
                                   &INFERENCE_OPTIONS,                                                                 \
                                   nullptr);                                                                           \
        }                                                                                                              \
        struct dora_stream_result flow = {};                                                                           \
        while (status == 0) {                                                                                          \
            status = dora_schema_inference_next(&out_slot->inference,                                                  \
                                                &out_slot->schema,                                                     \
                                                &out_slot->reader,                                                     \
                                                &flow,                                                                 \
                                                nullptr);                                                              \
            if (status != 0 || flow.status == DORA_STREAM_STATUS_READY) {                                              \
                break;                                                                                                 \
            }                                                                                                          \
            if (flow.status != DORA_STREAM_STATUS_BLOCKED) {                                                           \
                status = EDOM;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        return status;                                                                                                 \
    }                                                                                                                  \
    BENCH_DEFINE_OPERATION(infer_##format_tag##_operation,                                                             \
                           struct infer_slot,                                                                          \
                           compute_infer_##format_tag,                                                                 \
                           release_inferences)                                                                         \
    static void infer_##format_tag(struct bench* bench, const struct bench_args* args) {                               \
        struct inputs inputs = prepare(bench, args);                                                                   \
        if (bench->status == 0) {                                                                                      \
            struct infer_slot slot = {};                                                                               \
            bench_check(bench, compute_infer_##format_tag(&inputs, &slot));                                            \
            bench_check(bench, check_inferred(args, &slot.schema, &slot.reader));                                      \
            release_inferences(&slot, 1);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &infer_##format_tag##_operation, &inputs);                                                \
        finish(bench, args, &inputs);                                                                                  \
    }

BENCH_TEXT_DEFINE_INFER(csv, dora_csv_schema_inference_init, &BENCH_CSV_OPTIONS)
BENCH_TEXT_DEFINE_INFER(ndjson, dora_json_schema_inference_init, &JSON_OPTIONS)

#undef BENCH_TEXT_DEFINE_INFER

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"csv/read/mixed", read_csv, {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_CSV}},
        {"csv/read/numeric",
         read_csv,
         {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_CSV, .variant = VARIANT_READ_NUMERIC}},
        {"csv/write/mixed", write_csv, {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_CSV, .variant = VARIANT_WRITE}},
        {"csv/infer/mixed", infer_csv, {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_CSV, .variant = VARIANT_INFER}},
        {"ndjson/read/mixed", read_ndjson, {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_NDJSON}},
        {"ndjson/write/mixed",
         write_ndjson,
         {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_NDJSON, .variant = VARIANT_WRITE}},
        {"ndjson/infer/mixed",
         infer_ndjson,
         {.rows = BENCH_ROWS, .format = BENCH_TEXT_FORMAT_NDJSON, .variant = VARIANT_INFER}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
