// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "json/json.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "utf8/input.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>

static int json_bind(const struct dora_extension_definition*,
                     struct dora_pool pool,
                     const struct dora_schema* storage,
                     struct dora_bytes metadata,
                     struct dora_allocation*,
                     struct dora_error*) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(storage, &field);
    if (status != DORA_OK || field.type.id != DORA_TYPE_UTF8 || metadata.byte_count > UINT32_MAX ||
        (metadata.byte_count != 0 && metadata.data == nullptr)) {
        return status != DORA_OK ? status : EINVAL;
    }
    if (metadata.byte_count == 0) {
        return DORA_OK;
    }
    status = dora_json_validate(pool, metadata, DORA_UTF8_CHECKED);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t first = dora_json_skip_whitespace(metadata.data, 0, (uint32_t)metadata.byte_count);
    return first < metadata.byte_count && metadata.data[first] == '{' ? DORA_OK : EINVAL;
}

static int json_validate(const struct dora_extension_definition*,
                         struct dora_pool pool,
                         const struct dora_allocation*,
                         const struct dora_vector* storage,
                         uint64_t* in_out_validity,
                         struct dora_error*) {
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(storage, &view);
    const uint64_t row_count = view.length;
    if (status != DORA_OK || (row_count != 0 && in_out_validity == nullptr)) {
        return status != DORA_OK ? status : EINVAL;
    }
    struct dora_allocation frames = {};
    bool invalid_content = false;
    for (uint64_t row = 0; row < row_count && status == DORA_OK; ++row) {
        if (!dora_bitmap_test((const uint8_t*)in_out_validity, row)) {
            continue;
        }
        if (dora_vector_view_is_null(&view, row)) {
            continue;
        }
        const struct dora_bytes bytes = dora_vector_view_get_bytes(&view, row);
        bool invalid = false;
        status = dora_json_validate_frames(pool, bytes, DORA_UTF8_CHECKED, &frames, &invalid);
        if (invalid) {
            bitmap_clear(in_out_validity, row);
            invalid_content = true;
            status = DORA_OK;
        }
    }
    dora_allocation_release(&frames);
    return status != DORA_OK ? status : invalid_content ? EILSEQ : DORA_OK;
}

static int json_cast(const struct dora_extension_definition*,
                     struct dora_pool pool,
                     const struct dora_allocation*,
                     const struct dora_vector* source,
                     const struct dora_logical_type* target,
                     enum dora_rounding_mode,
                     enum dora_failure_policy,
                     struct dora_vector* out_result) {
    if (target == nullptr || target->id != DORA_TYPE_UTF8) {
        return ENOTSUP;
    }
    struct dora_vector storage = {};
    int status = dora_vector_get_storage(source, &storage);
    if (status == DORA_OK && target->extension_name.byte_count != 0) {
        status = dora_vector_init_retain_extension_error(out_result,
                                                         pool,
                                                         target->extension_name,
                                                         target->extension_metadata,
                                                         &storage,
                                                         nullptr,
                                                         nullptr);
    } else if (status == DORA_OK) {
        status = dora_vector_copy(&storage, pool, out_result);
    }
    return status;
}

const struct dora_extension_definition dora_extension_json = {
    .name = {.data = (const uint8_t*)"arrow.json", .byte_count = sizeof("arrow.json") - 1},
    .flags = DORA_EXTENSION_FLAG_STORAGE_EQUALITY,
    .bind = json_bind,
    .validate = json_validate,
    .cast = json_cast
};
