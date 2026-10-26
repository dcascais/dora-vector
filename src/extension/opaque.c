// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "json/json.h"
#include "utf8/input.h"

#include <errno.h>
#include <stdint.h>

static int opaque_bind(const struct dora_extension_definition*,
                       struct dora_pool pool,
                       const struct dora_schema*,
                       struct dora_bytes metadata,
                       struct dora_allocation*,
                       struct dora_error*) {
    if (metadata.byte_count == 0 || metadata.byte_count > UINT32_MAX || metadata.data == nullptr) {
        return EINVAL;
    }
    int status = dora_json_validate(pool, metadata, DORA_UTF8_CHECKED);
    if (status != DORA_OK) {
        return status;
    }
    const char* const names[] = {"type_name", "vendor_name"};
    struct dora_json_span values[2] = {};
    uint32_t present = 0;
    status = dora_json_object_members(metadata.data, (uint32_t)metadata.byte_count, names, 2, values, &present);
    if (status != DORA_OK) {
        return status;
    }
    return present == 3 && metadata.data[values[0].begin_byte_offset] == '"' &&
                   metadata.data[values[1].begin_byte_offset] == '"'
               ? DORA_OK
               : EINVAL;
}

const struct dora_extension_definition dora_extension_opaque = {
    .name = {.data = (const uint8_t*)"arrow.opaque", .byte_count = sizeof("arrow.opaque") - 1},
    .bind = opaque_bind
};
