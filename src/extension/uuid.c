// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stdint.h>

static int uuid_bind(const struct dora_extension_definition*,
                     struct dora_pool,
                     const struct dora_schema* storage,
                     struct dora_bytes metadata,
                     struct dora_allocation*,
                     struct dora_error*) {
    struct dora_schema_field field = {};
    const int status = dora_schema_get_field(storage, &field);
    return status != DORA_OK ? status
           : field.type.id == DORA_TYPE_FIXED_BINARY && field.type.parameters.byte_width == 16 &&
                   metadata.byte_count == 0
               ? DORA_OK
               : EINVAL;
}

const struct dora_extension_definition dora_extension_uuid = {
    .name = {.data = (const uint8_t*)"arrow.uuid", .byte_count = sizeof("arrow.uuid") - 1},
    .flags = DORA_EXTENSION_FLAG_STORAGE_EQUALITY | DORA_EXTENSION_FLAG_STORAGE_ORDER,
    .bind = uuid_bind
};
