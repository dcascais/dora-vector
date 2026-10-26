// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_schema_field fields[] = {
        {.name = "id", .type = {.id = DORA_TYPE_INT32}, .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE},
        {.name = "city", .type = {.id = DORA_TYPE_UTF8}}
    };
    const struct dora_schema_input schema = {.fields = fields, .field_count = 2};
    const int32_t ids[] = {101, 102, 103};
    const struct dora_bytes cities[] = {DORA_BYTES_LITERAL("Santiago"), DORA_BYTES_LITERAL("Lisbon"), {}};
    struct dora_pool pool = {};
    struct dora_builder builder = {};
    struct dora_builder id_writer = {};
    struct dora_builder city_writer = {};
    struct dora_vector batch = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_builder_init_fields(&builder, pool, &schema, nullptr);
    }
    if (status == 0) {
        status = dora_builder_get_child(&builder, 0, &id_writer);
    }
    if (status == 0) {
        status = dora_builder_get_child(&builder, 1, &city_writer);
    }
    for (uint64_t row = 0; status == 0 && row < 3; ++row) {
        status = dora_builder_append_i32(&id_writer, ids[row]);
        if (status == 0) {
            status = cities[row].data == nullptr ? dora_builder_append_null(&city_writer)
                                                 : dora_builder_append_bytes(&city_writer, cities[row]);
        }
        if (status == 0) {
            status = dora_builder_end_row(&builder);
        }
    }
    if (status == 0) {
        status = dora_builder_finish(&builder, &batch);
    }
    dora_builder_release(&builder);

    // Children borrow batch storage.
    struct dora_vector id_column = {};
    struct dora_vector city_column = {};
    if (status == 0) {
        status = dora_vector_get_child(&batch, 0, &id_column);
    }
    if (status == 0) {
        status = dora_vector_get_child(&batch, 1, &city_column);
    }
    for (uint64_t row = 0; status == 0 && row < 3; ++row) {
        int32_t id = 0;
        struct dora_bytes city = {};
        bool id_is_null = false;
        bool city_is_null = false;
        status = dora_vector_read_i32(&id_column, row, &id, &id_is_null);
        if (status == 0) {
            status = dora_vector_read_bytes(&city_column, row, &city, &city_is_null);
        }
        if (status == 0 && id_is_null) {
            status = EINVAL;
        }
        if (status == 0 && printf("%" PRId32 ",", id) < 0) {
            status = EIO;
        }
        if (status == 0) {
            const struct dora_bytes text = city_is_null ? (struct dora_bytes)DORA_BYTES_LITERAL("null") : city;
            if ((text.byte_count != 0 && fwrite(text.data, 1, text.byte_count, stdout) != text.byte_count) ||
                printf("\n") < 0) {
                status = EIO;
            }
        }
    }
    dora_vector_release(&batch);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "record_batch failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
