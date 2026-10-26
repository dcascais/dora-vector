// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type timestamp = {
        .id = DORA_TYPE_TIMESTAMP,
        .parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = "UTC"}
    };
    const int64_t times[] = {1718000000000000, 0, 1718000060000000};
    const bool valid[] = {true, false, true};
    const struct dora_arrow_limits limits = {
        .max_depth = 8,
        .max_node_count = 16,
        .max_edge_count = 16,
        .max_coordinate_work_count = 1024,
        .max_output_bytes = 65536
    };
    struct dora_pool pool = {};
    struct dora_vector events = {};
    struct dora_vector imported = {};
    struct dora_schema native_schema = {};
    struct dora_schema imported_schema = {};
    struct ArrowSchema schema = {};
    struct ArrowArray array = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i64(&events, pool, &timestamp, times, valid, 3);
    }
    if (status == 0) {
        status = dora_vector_retain_schema(&events, &native_schema);
    }
    if (status == 0) {
        status = dora_arrow_export_schema(pool, &native_schema, &schema);
    }
    if (status == 0) {
        status = dora_arrow_export_array(&events, &array);
    }
    // array retains storage.
    dora_vector_release(&events);
    dora_schema_release(&native_schema);
    if (status == 0 && printf("Arrow format=%s rows=%" PRId64 "\n", schema.format, array.length) < 0) {
        status = EIO;
    }
    if (status == 0) {
        status = dora_arrow_import_schema(pool, &schema, &limits, &imported_schema, nullptr);
    }
    if (status == 0) {
        // Success clears array.release.
        status = dora_arrow_import_move_array(pool, &imported_schema, &array, &limits, &imported, nullptr);
    }
    for (uint64_t row = 0; status == 0 && row < 3; ++row) {
        int64_t value = 0;
        bool is_null = false;
        status = dora_vector_read_i64(&imported, row, &value, &is_null);
        if (status == 0) {
            const int printed = is_null ? printf("null\n") : printf("%" PRId64 " us\n", value);
            if (printed < 0) {
                status = EIO;
            }
        }
    }
    dora_vector_release(&imported);
    dora_schema_release(&imported_schema);
    if (array.release != nullptr) {
        array.release(&array);
    }
    if (schema.release != nullptr) {
        schema.release(&schema);
    }

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "arrow_array failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
