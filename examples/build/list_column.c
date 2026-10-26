// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const int32_t item_values[] = {7, 8, 9};
    const uint64_t lengths[] = {2, 0, 0, 1};
    const bool valid[] = {true, true, false, true};
    struct dora_pool pool = {};
    struct dora_vector items = {};
    struct dora_vector lists = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&items, pool, &i32, item_values, nullptr, 3);
    }
    if (status == 0) {
        status = dora_vector_init_retain_list(&lists, DORA_LAYOUT_LIST, &items, lengths, valid, 4);
    }
    struct dora_vector elements = {};
    struct dora_vector_view view = {};
    if (status == 0) {
        status = dora_vector_get_child(&lists, 0, &elements);
    }
    if (status == 0) {
        status = dora_vector_get_view(&lists, &view);
    }
    for (uint64_t row = 0; status == 0 && row < 4; ++row) {
        if (dora_vector_view_is_null(&view, row)) {
            if (printf("null\n") < 0) {
                status = EIO;
            }
            continue;
        }
        const struct dora_range range = dora_vector_view_get_elements(&view, row);
        if (printf("[") < 0) {
            status = EIO;
        }
        for (uint64_t item = 0; status == 0 && item < range.length; ++item) {
            int32_t value = 0;
            bool is_null = false;
            status = dora_vector_read_i32(&elements, range.offset + item, &value, &is_null);
            if (status == 0 && is_null) {
                status = EINVAL;
            }
            if (status == 0 && printf("%s%" PRId32, item == 0 ? "" : ", ", value) < 0) {
                status = EIO;
            }
        }
        if (status == 0 && printf("]\n") < 0) {
            status = EIO;
        }
    }
    dora_vector_release(&lists);
    dora_vector_release(&items);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "list_column failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
