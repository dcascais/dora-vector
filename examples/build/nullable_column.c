// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type type = {.id = DORA_TYPE_INT32};
    const int32_t first[] = {18, 0, 21};
    const bool valid[] = {true, false, true};
    const int32_t second[] = {24, 26};
    struct dora_pool pool = {};
    struct dora_builder builder = {};
    struct dora_vector readings = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_builder_init_type(&builder, pool, &type, nullptr);
    }
    if (status == 0) {
        status = dora_builder_append_values_i32(&builder, first, valid, 3);
    }
    if (status == 0) {
        status = dora_builder_append_values_i32(&builder, second, nullptr, 2);
    }
    if (status == 0) {
        status = dora_builder_finish(&builder, &readings);
    }
    dora_builder_release(&builder);

    uint64_t rows = 0;
    if (status == 0) {
        status = dora_vector_get_length(&readings, &rows);
    }
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        int32_t value = 0;
        bool is_null = false;
        status = dora_vector_read_i32(&readings, row, &value, &is_null);
        if (status == 0) {
            const int printed = is_null ? printf("reading[%" PRIu64 "]=null\n", row)
                                        : printf("reading[%" PRIu64 "]=%" PRId32 "\n", row, value);
            if (printed < 0) {
                status = EIO;
            }
        }
    }
    dora_vector_release(&readings);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "nullable_column failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
