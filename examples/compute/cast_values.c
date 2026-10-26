// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const struct dora_logical_type i8 = {.id = DORA_TYPE_INT8};
    const int32_t values[] = {42, 300, -7, 0};
    const bool valid[] = {true, true, true, false};
    struct dora_pool pool = {};
    struct dora_vector source = {};
    struct dora_vector narrowed = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&source, pool, &i32, values, valid, 4);
    }
    if (status == 0) {
        status = dora_compute_cast(&source, &i8, DORA_ROUNDING_MODE_EXACT, DORA_FAILURE_POLICY_NULL, &narrowed);
    }
    for (uint64_t row = 0; status == 0 && row < 4; ++row) {
        int8_t value = 0;
        bool is_null = false;
        status = dora_vector_read_i8(&narrowed, row, &value, &is_null);
        if (status == 0) {
            const int printed = is_null ? printf("null\n") : printf("%d\n", value);
            if (printed < 0) {
                status = EIO;
            }
        }
    }
    dora_vector_release(&narrowed);
    dora_vector_release(&source);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "cast_values failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
