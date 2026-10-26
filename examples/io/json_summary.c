// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/file.h"
#include "dora/vector/io/json.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdckdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        if (fprintf(stderr, "usage: %s <cars.json>\n", argv[0]) < 0) {
            return EXIT_FAILURE;
        }
        return EXIT_FAILURE;
    }
    const struct dora_schema_field fields[] = {
        {.name = "Name", .type = {.id = DORA_TYPE_UTF8}},
        {.name = "Miles_per_Gallon", .type = {.id = DORA_TYPE_FLOAT64}},
        {.name = "Cylinders", .type = {.id = DORA_TYPE_INT32}},
        {.name = "Displacement", .type = {.id = DORA_TYPE_FLOAT64}},
        {.name = "Horsepower", .type = {.id = DORA_TYPE_FLOAT64}},
        {.name = "Weight_in_lbs", .type = {.id = DORA_TYPE_INT64}},
        {.name = "Acceleration", .type = {.id = DORA_TYPE_FLOAT64}},
        {.name = "Year", .type = {.id = DORA_TYPE_UTF8}},
        {.name = "Origin", .type = {.id = DORA_TYPE_UTF8}}
    };
    const struct dora_schema_input input = {.fields = fields, .field_count = 9};
    struct dora_pool pool = {};
    struct dora_schema schema = {};
    struct dora_byte_source_file source = {};
    struct dora_reader reader = {};
    struct dora_vector batch = {};
    struct dora_error error = {};
    FILE* file = nullptr;
    uint64_t rows = 0;
    int64_t observed = 0;
    double total = 0;
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_schema_init_copy(&schema, pool, &input);
    }
    if (status == 0) {
        file = fopen(argv[1], "rb");
        if (file == nullptr) {
            status = errno;
        }
    }
    if (status == 0) {
        status = dora_byte_source_file_init(&source, file);
    }
    if (status == 0) {
        status = dora_json_reader_init(
            &reader,
            pool,
            &schema,
            &source.source,
            &(const struct dora_json_options){.format = DORA_JSON_FORMAT_ARRAY, .row_capacity = 128},
            &error);
    }
    while (status == 0) {
        struct dora_stream_result flow = {};
        status = dora_reader_next(&reader, &batch, &flow, &error);
        if (status != 0 || flow.status == DORA_STREAM_STATUS_END) {
            break;
        }
        if (flow.status != DORA_STREAM_STATUS_READY) {
            status = EAGAIN;
            break;
        }
        struct dora_vector column = {};
        struct dora_vector sum = {};
        struct dora_vector count = {};
        uint64_t batch_rows = 0;
        int64_t batch_count = 0;
        double batch_sum = 0;
        bool is_null = false;
        status = dora_vector_get_length(&batch, &batch_rows);
        if (status == 0) {
            status = dora_vector_get_child(&batch, 1, &column);
        }
        if (status == 0) {
            status = dora_compute_count(&column, &count);
        }
        if (status == 0) {
            status = dora_vector_read_i64(&count, 0, &batch_count, &is_null);
        }
        if (status == 0 && batch_count > 0) {
            status = dora_compute_sum(&column, &sum);
            if (status == 0) {
                status = dora_vector_read_f64(&sum, 0, &batch_sum, &is_null);
            }
            if (status == 0 && is_null) {
                status = EINVAL;
            }
        }
        if (status == 0 && (ckd_add(&rows, rows, batch_rows) || ckd_add(&observed, observed, batch_count))) {
            status = EOVERFLOW;
        }
        if (status == 0) {
            total += batch_sum;
        }
        dora_vector_release(&count);
        dora_vector_release(&sum);
        dora_vector_release(&batch);
    }
    if (status == 0 &&
        (observed == 0
             ? printf("cars=%" PRIu64 " observed_mpg=0 mean_mpg=null\n", rows)
             : printf("cars=%" PRIu64 " observed_mpg=%" PRId64 " mean_mpg=%.2f\n", rows, observed, total / observed)) <
            0) {
        status = EIO;
    }
    dora_vector_release(&batch);
    dora_reader_release(&reader);
    if (source.source.release != nullptr) {
        source.source.release(&source.source);
    }
    // file remains caller-owned.
    if (file != nullptr && fclose(file) != 0 && status == 0) {
        status = EIO;
    }
    dora_schema_release(&schema);
    if (status != 0 && error.message[0] != '\0') {
        if (fprintf(stderr, "%s: %s\n", argv[1], error.message) < 0) {
            status = EIO;
        }
    }

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "json_summary failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
