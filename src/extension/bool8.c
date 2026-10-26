// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/builder.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>

static constexpr uint64_t BOOL8_CAST_CHUNK_WORDS = 32;
static constexpr uint64_t BOOL8_CAST_CHUNK_ROWS = BOOL8_CAST_CHUNK_WORDS * DORA_BITMAP_WORD_BIT_COUNT;

static int bool8_bind(const struct dora_extension_definition*,
                      struct dora_pool,
                      const struct dora_schema* storage,
                      struct dora_bytes metadata,
                      struct dora_allocation*,
                      struct dora_error*) {
    struct dora_schema_field field = {};
    const int status = dora_schema_get_field(storage, &field);
    return status != DORA_OK ? status : field.type.id == DORA_TYPE_INT8 && metadata.byte_count == 0 ? DORA_OK : EINVAL;
}

static int bool8_cast(const struct dora_extension_definition*,
                      struct dora_pool pool,
                      const struct dora_allocation*,
                      const struct dora_vector* source,
                      const struct dora_logical_type* target,
                      enum dora_rounding_mode,
                      enum dora_failure_policy,
                      struct dora_vector* out_result) {
    if (target == nullptr || target->id != DORA_TYPE_BOOLEAN) {
        return ENOTSUP;
    }
    struct dora_vector storage = {};
    struct dora_vector decoded = {};
    struct dora_vector_view view = {};
    int status = dora_vector_get_storage(source, &storage);
    if (status == DORA_OK) {
        status = dora_vector_get_view(&storage, &view);
    }
    if (status == DORA_OK && (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END)) {
        status = dora_vector_decode_error(&storage, pool, &decoded, nullptr);
        if (status == DORA_OK) {
            status = dora_vector_get_view(&decoded, &view);
        }
    }
    struct dora_builder builder = {};
    const struct dora_logical_type type = {.id = DORA_TYPE_BOOLEAN};
    const struct dora_builder_options options = {.row_capacity = view.length};
    if (status == DORA_OK) {
        status = dora_builder_init_type(&builder, pool, &type, &options);
    }
    for (uint64_t first = 0; status == DORA_OK && first < view.length;) {
        uint64_t values[BOOL8_CAST_CHUNK_WORDS] = {};
        uint64_t validity[BOOL8_CAST_CHUNK_WORDS];
        bitmap_fill_bytes((uint8_t*)validity, sizeof validity, false);
        const uint64_t count =
            view.length - first < BOOL8_CAST_CHUNK_ROWS ? view.length - first : BOOL8_CAST_CHUNK_ROWS;
        for (uint64_t row = 0; row < count; ++row) {
            if (dora_vector_view_is_null(&view, first + row)) {
                continue;
            }
            const uint64_t physical = dora_vector_view_get_physical_row(&view, first + row);
            const int8_t value = ((const int8_t*)view.values.data)[physical];
            bitmap_set(validity, row);
            bitmap_assign(values, row, value != 0);
        }
        status =
            dora_builder_append_packed_bool(&builder, (const uint8_t*)values, 0, (const uint8_t*)validity, 0, count);
        first += count;
    }
    if (status == DORA_OK) {
        status = dora_builder_finish(&builder, out_result);
    }
    dora_builder_release(&builder);
    dora_vector_release(&decoded);
    return status;
}

const struct dora_extension_definition dora_extension_bool8 = {
    .name = {.data = (const uint8_t*)"arrow.bool8", .byte_count = sizeof("arrow.bool8") - 1},
    .bind = bool8_bind,
    .cast = bool8_cast
};
