// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "extension/storage.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

static int timestamp_offset_bind(const struct dora_extension_definition*,
                                 struct dora_pool,
                                 const struct dora_schema* storage,
                                 struct dora_bytes metadata,
                                 struct dora_allocation*,
                                 struct dora_error*) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(storage, &field);
    if (status != DORA_OK || field.type.id != DORA_TYPE_STRUCT || field.child_count != 2 || metadata.byte_count != 0) {
        return status != DORA_OK ? status : EINVAL;
    }
    struct dora_schema timestamp = {};
    struct dora_schema offset = {};
    status = dora_schema_get_child(storage, 0, &timestamp);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&timestamp, &field);
    }
    if (status != DORA_OK || field.name == nullptr || strcmp(field.name, "timestamp") != 0 ||
        (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0 || field.layout != DORA_LAYOUT_FIXED ||
        field.type.id != DORA_TYPE_TIMESTAMP || field.type.parameters.timestamp.timezone == nullptr ||
        strcmp(field.type.parameters.timestamp.timezone, "UTC") != 0) {
        return status != DORA_OK ? status : EINVAL;
    }
    status = dora_schema_get_child(storage, 1, &offset);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&offset, &field);
    }
    if (status != DORA_OK || field.name == nullptr || strcmp(field.name, "offset_minutes") != 0 ||
        (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
        return status != DORA_OK ? status : EINVAL;
    }
    status = extension_resolve_storage_field(&offset, &field);
    if (status != DORA_OK) {
        return status;
    }
    return field.type.id == DORA_TYPE_INT16 ? DORA_OK : EINVAL;
}

const struct dora_extension_definition dora_extension_timestamp_with_offset = {
    .name =
        {.data = (const uint8_t*)"arrow.timestamp_with_offset",
         .byte_count = sizeof("arrow.timestamp_with_offset") - 1},
    .bind = timestamp_offset_bind
};
