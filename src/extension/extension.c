// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/extension.h"

#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "extension/extension_internal.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "types/array_count.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stddef.h>

//------------------------------------------------------------------------------
//
// Registration Storage
//
//------------------------------------------------------------------------------

static const struct dora_extension_definition* const linked_extensions[] = {
    &dora_extension_bool8,
    &dora_extension_uuid,
    &dora_extension_json,
    &dora_extension_opaque,
    &dora_extension_fixed_shape_tensor,
    &dora_extension_variable_shape_tensor,
    &dora_extension_parquet_variant,
    &dora_extension_timestamp_with_offset
};

static const size_t linked_extension_count = ARRAY_COUNT(linked_extensions);

static bool name_is_valid(struct dora_bytes name) {
    return name.data != nullptr && name.byte_count != 0;
}

static const struct dora_extension_definition* registered_definition(const struct pool_row* row,
                                                                     struct dora_bytes name) {
    for (const struct extension_registration* node = row->registrations; node != nullptr; node = node->next) {
        if (bytes_are_equal(node->definition->name, name)) {
            return node->definition;
        }
    }
    return nullptr;
}

static const struct dora_extension_definition* linked_definition(struct dora_bytes name) {
    for (size_t index = 0; index < linked_extension_count; ++index) {
        const struct dora_extension_definition* const definition = linked_extensions[index];
        if (bytes_are_equal(definition->name, name)) {
            return definition;
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------
//
// Definition Lookup
//
//------------------------------------------------------------------------------

int dora_extension_find(struct dora_pool pool,
                        struct dora_bytes name,
                        const struct dora_extension_definition** out_definition) {
    const struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_definition == nullptr) {
        return EINVAL;
    }
    if (!name_is_valid(name)) {
        *out_definition = nullptr;
        return DORA_OK;
    }
    const struct dora_extension_definition* const definition = registered_definition(row, name);
    *out_definition = definition != nullptr ? definition : linked_definition(name);
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Registration Lifecycle
//
//------------------------------------------------------------------------------

void dora_extension_registrations_close(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return;
    }
    struct extension_registration* node = row->registrations;
    row->registrations = nullptr;
    while (node != nullptr) {
        struct extension_registration* const next = node->next;
        dora_memory_overhead_release(pool, node, sizeof(*node));
        node = next;
    }
}

//------------------------------------------------------------------------------
//
// Registration
//
//------------------------------------------------------------------------------

static int registration_admission(const struct pool_row* row, struct dora_bytes name) {
    if (registered_definition(row, name) != nullptr || linked_definition(name) != nullptr) {
        return EINVAL;
    }
    return row->directory.live_slot_count != 0 ? EBUSY : DORA_OK;
}

int dora_extension_register(const struct dora_extension_definition* definition, struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (definition == nullptr || !name_is_valid(definition->name) || row == nullptr) {
        return EINVAL;
    }
    int status = registration_admission(row, definition->name);
    if (status != DORA_OK) {
        return status;
    }
    void* address = nullptr;
    status = dora_memory_overhead_allocate(pool, sizeof(struct extension_registration), &address);
    if (status != DORA_OK) {
        return status;
    }
    // Allocator reentry may register or publish before linking.
    status = registration_admission(row, definition->name);
    if (status != DORA_OK) {
        dora_memory_overhead_release(pool, address, sizeof(struct extension_registration));
        return status;
    }
    struct extension_registration* const node = address;
    *node = (struct extension_registration){.definition = definition, .next = row->registrations};
    row->registrations = node;
    return DORA_OK;
}
