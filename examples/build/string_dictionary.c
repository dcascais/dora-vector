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
    const struct dora_logical_type utf8 = {.id = DORA_TYPE_UTF8};
    const struct dora_logical_type i8 = {.id = DORA_TYPE_INT8};
    const struct dora_bytes labels[] = {DORA_BYTES_LITERAL("paid"), DORA_BYTES_LITERAL("pending")};
    const int8_t codes[] = {0, 1, 0, 0};
    const bool valid[] = {true, true, true, false};
    struct dora_pool pool = {};
    struct dora_vector dictionary_values = {};
    struct dora_vector indices = {};
    struct dora_vector states = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_bytes(&dictionary_values,
                                             pool,
                                             &utf8,
                                             DORA_LAYOUT_BINARY,
                                             labels,
                                             nullptr,
                                             2,
                                             DORA_FAILURE_POLICY_ERROR);
    }
    if (status == 0) {
        status = dora_vector_init_copy_i8(&indices, pool, &i8, codes, valid, 4);
    }
    if (status == 0) {
        status = dora_vector_init_retain_dictionary(&states, &indices, &dictionary_values);
    }
    dora_vector_release(&indices);
    dora_vector_release(&dictionary_values);
    for (uint64_t row = 0; status == 0 && row < 4; ++row) {
        struct dora_bytes label = {};
        bool is_null = false;
        status = dora_vector_read_bytes(&states, row, &label, &is_null);
        if (status == 0) {
            const struct dora_bytes text = is_null ? (struct dora_bytes)DORA_BYTES_LITERAL("null") : label;
            if ((text.byte_count != 0 && fwrite(text.data, 1, text.byte_count, stdout) != text.byte_count) ||
                printf("\n") < 0) {
                status = EIO;
            }
        }
    }
    dora_vector_release(&states);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "string_dictionary failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
