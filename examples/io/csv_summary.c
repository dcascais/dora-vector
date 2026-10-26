// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/io/file.h"
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
        if (fprintf(stderr, "usage: %s <flights.csv>\n", argv[0]) < 0) {
            return EXIT_FAILURE;
        }
        return EXIT_FAILURE;
    }
    const struct dora_schema_field fields[] = {
        {.name = "origin", .type = {.id = DORA_TYPE_UTF8}},
        {.name = "destination", .type = {.id = DORA_TYPE_UTF8}},
        {.name = "count", .type = {.id = DORA_TYPE_INT64}}
    };
    const struct dora_schema_input input = {.fields = fields, .field_count = 3};
    struct dora_pool pool = {};
    struct dora_schema schema = {};
    struct dora_byte_source_file source = {};
    struct dora_reader reader = {};
    struct dora_vector batch = {};
    struct dora_error error = {};
    FILE* file = nullptr;
    uint64_t rows = 0;
    int64_t observed = 0;
    int64_t total = 0;
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
        status = dora_csv_reader_init(&reader,
                                      pool,
                                      &schema,
                                      &source.source,
                                      &(const struct dora_csv_options){.has_header = true, .row_capacity = 128},
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
        struct dora_vector sum_i64 = {};
        struct dora_vector count = {};
        uint64_t batch_rows = 0;
        int64_t batch_count = 0;
        int64_t batch_sum = 0;
        bool is_null = false;
        status = dora_vector_get_length(&batch, &batch_rows);
        if (status == 0) {
            status = dora_vector_get_child(&batch, 2, &column);
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
                // Int64 sums return Int128.
                status = dora_compute_cast(&sum,
                                           &(const struct dora_logical_type){.id = DORA_TYPE_INT64},
                                           DORA_ROUNDING_MODE_EXACT,
                                           DORA_FAILURE_POLICY_ERROR,
                                           &sum_i64);
            }
            if (status == 0) {
                status = dora_vector_read_i64(&sum_i64, 0, &batch_sum, &is_null);
            }
            if (status == 0 && is_null) {
                status = EINVAL;
            }
        }
        if (status == 0 && (ckd_add(&rows, rows, batch_rows) || ckd_add(&observed, observed, batch_count))) {
            status = EOVERFLOW;
        }
        if (status == 0) {
            if (ckd_add(&total, total, batch_sum)) {
                status = EOVERFLOW;
            }
        }
        dora_vector_release(&count);
        dora_vector_release(&sum_i64);
        dora_vector_release(&sum);
        dora_vector_release(&batch);
    }
    if (status == 0 &&
        printf("routes=%" PRIu64 " observed=%" PRId64 " flights=%" PRId64 "\n", rows, observed, total) < 0) {
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
        if (fprintf(stderr, "csv_summary failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
