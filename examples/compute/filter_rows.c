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
    const int32_t order_ids[] = {101, 102, 103, 104};
    const int32_t amounts[] = {25, 80, 0, 120};
    const bool valid[] = {true, true, false, true};
    const int32_t minimum = 50;
    const char* const names[] = {"order_id", "amount"};
    struct dora_pool pool = {};
    struct dora_vector columns[2] = {};
    struct dora_vector orders = {};
    struct dora_vector scalar = {};
    struct dora_vector threshold = {};
    struct dora_vector selection = {};
    struct dora_vector filtered = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&columns[0], pool, &i32, order_ids, nullptr, 4);
    }
    if (status == 0) {
        status = dora_vector_init_copy_i32(&columns[1], pool, &i32, amounts, valid, 4);
    }
    if (status == 0) {
        status = dora_vector_init_retain_struct(&orders, columns, names, 2, nullptr, 4);
    }
    if (status == 0) {
        status = dora_vector_init_copy_i32(&scalar, pool, &i32, &minimum, nullptr, 1);
    }
    if (status == 0) {
        status = dora_vector_init_retain_constant(&threshold, &scalar, 4);
    }
    if (status == 0) {
        status = dora_compute_select(&columns[1], DORA_COMPARE_OP_GE, &threshold, &selection);
    }
    if (status == 0) {
        status = dora_vector_init_retain_selection(&filtered, &orders, &selection);
    }
    struct dora_vector ids = {};
    if (status == 0) {
        status = dora_vector_get_child(&filtered, 0, &ids);
    }
    uint64_t rows = 0;
    if (status == 0) {
        status = dora_vector_get_length(&ids, &rows);
    }
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        int32_t id = 0;
        bool is_null = false;
        status = dora_vector_read_i32(&ids, row, &id, &is_null);
        if (status == 0 && is_null) {
            status = EINVAL;
        }
        if (status == 0 && printf("order_id=%" PRId32 "\n", id) < 0) {
            status = EIO;
        }
    }
    dora_vector_release(&filtered);
    dora_vector_release(&selection);
    dora_vector_release(&threshold);
    dora_vector_release(&scalar);
    dora_vector_release(&orders);
    dora_vector_release(&columns[1]);
    dora_vector_release(&columns[0]);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "filter_rows failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
