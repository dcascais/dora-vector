// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_schema_field field = {.name = "reading", .type = {.id = DORA_TYPE_INT32}};
    const struct dora_schema_input input = {.fields = &field, .field_count = 1};
    const struct dora_arrow_limits limits = {
        .max_depth = 8,
        .max_node_count = 16,
        .max_edge_count = 16,
        .max_coordinate_work_count = 1024,
        .max_output_bytes = 65536
    };
    struct dora_pool pool = {};
    struct dora_schema schema = {};
    struct dora_byte_source_memory bytes = {};
    struct dora_reader producer = {};
    struct dora_reader consumer = {};
    struct ArrowArrayStream stream = {};
    struct dora_vector batch = {};
    struct dora_error error = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_schema_init_copy(&schema, pool, &input);
    }
    if (status == 0) {
        status = dora_byte_source_memory_init(&bytes, (struct dora_bytes)DORA_BYTES_LITERAL("reading\n10\n20\n30\n"));
    }
    if (status == 0) {
        status = dora_csv_reader_init(&producer,
                                      pool,
                                      &schema,
                                      &bytes.source,
                                      &(const struct dora_csv_options){.has_header = true, .row_capacity = 2},
                                      &error);
    }
    if (status == 0) {
        // Success moves producer into stream.
        status = dora_arrow_export_stream(&schema, &producer, &stream);
    }
    if (status == 0) {
        // Success moves stream into consumer.
        status = dora_arrow_import_stream(pool, &stream, &limits, &consumer, &error);
    }
    while (status == 0) {
        struct dora_stream_result flow = {};
        status = dora_reader_next(&consumer, &batch, &flow, &error);
        if (status != 0 || flow.status == DORA_STREAM_STATUS_END) {
            break;
        }
        if (flow.status != DORA_STREAM_STATUS_READY) {
            status = EAGAIN;
            break;
        }
        uint64_t rows = 0;
        struct dora_vector column = {};
        status = dora_vector_get_length(&batch, &rows);
        if (status == 0) {
            status = dora_vector_get_child(&batch, 0, &column);
        }
        for (uint64_t row = 0; status == 0 && row < rows; ++row) {
            int32_t value = 0;
            bool is_null = false;
            status = dora_vector_read_i32(&column, row, &value, &is_null);
            if (status == 0 && is_null) {
                status = EINVAL;
            }
            if (status == 0 && printf("received=%" PRId32 "\n", value) < 0) {
                status = EIO;
            }
        }
        dora_vector_release(&batch);
    }
    dora_vector_release(&batch);
    dora_reader_release(&consumer);
    if (stream.release != nullptr) {
        stream.release(&stream);
    }
    dora_reader_release(&producer);
    if (bytes.source.release != nullptr) {
        bytes.source.release(&bytes.source);
    }
    dora_schema_release(&schema);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "arrow_stream failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
