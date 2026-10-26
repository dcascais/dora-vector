// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

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
    const int32_t values[] = {100, 200, 300, 400};
    struct dora_pool pool = {};
    struct dora_vector source = {};
    struct dora_vector slice = {};
    struct dora_vector retained = {};
    struct dora_vector copied = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&source, pool, &i32, values, nullptr, 4);
    }
    if (status == 0) {
        // slice borrows source storage.
        status = dora_vector_slice(&source, (struct dora_range){.offset = 1, .length = 2}, &slice);
    }
    if (status == 0) {
        status = dora_vector_retain(&slice, &retained);
    }
    if (status == 0) {
        status = dora_vector_copy(&slice, pool, &copied);
    }
    dora_vector_release(&source);
    for (uint64_t row = 0; status == 0 && row < 2; ++row) {
        int32_t shared_value = 0;
        int32_t copied_value = 0;
        bool shared_null = false;
        bool copied_null = false;
        status = dora_vector_read_i32(&retained, row, &shared_value, &shared_null);
        if (status == 0) {
            status = dora_vector_read_i32(&copied, row, &copied_value, &copied_null);
        }
        if (status == 0 && (shared_null || copied_null)) {
            status = EINVAL;
        }
        if (status == 0 && printf("retained=%" PRId32 " copied=%" PRId32 "\n", shared_value, copied_value) < 0) {
            status = EIO;
        }
    }
    dora_vector_release(&copied);
    dora_vector_release(&retained);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "retain_slice failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
