// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"
#include "schema/schema_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int dora_schema_walk(struct dora_pool pool,
                     const struct dora_schema* schema,
                     struct dora_allocation* out_nodes,
                     uint32_t* out_count) {
    if (!dora_schema_is_valid(schema) || out_nodes == nullptr || out_nodes->address != nullptr ||
        out_nodes->capacity_bytes != 0 || out_count == nullptr) {
        return EINVAL;
    }
    struct dora_allocation storage = {};
    struct dora_schema next = dora_schema_borrow(schema);
    uint32_t count = 0;
    uint32_t parent = UINT32_MAX;
    uint32_t child_index = 0;
    int status;
    for (;;) {
        size_t bytes = 0;
        if (count == UINT32_MAX || ckd_mul(&bytes, (size_t)count + 1, sizeof(struct schema_walk_node))) {
            status = EOVERFLOW;
            break;
        }
        status = dora_memory_payload_reserve(pool, bytes, (size_t)count * sizeof(struct schema_walk_node), &storage);
        if (status != DORA_OK) {
            break;
        }
        struct schema_walk_node* const nodes = storage.address;
        nodes[count] =
            (struct schema_walk_node){.schema = next, .parent_node_index = parent, .child_index = child_index};
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&next, &field);
        uint32_t child_count = field.child_count;
        if (status != DORA_OK) {
            break;
        }
        uint32_t cursor = count++;
        next = (struct dora_schema){};
        if (child_count != 0) {
            parent = cursor;
            child_index = 0;
            status = dora_schema_get_physical_child(&nodes[cursor].schema, 0, &next);
        } else {
            for (;;) {
                parent = nodes[cursor].parent_node_index;
                if (parent == UINT32_MAX) {
                    *out_nodes = storage;
                    *out_count = count;
                    return DORA_OK;
                }
                child_index = nodes[cursor].child_index + 1;
                status = dora_schema_get_field(&nodes[parent].schema, &field);
                child_count = field.child_count;
                if (status != DORA_OK) {
                    break;
                }
                if (child_index < child_count) {
                    status = dora_schema_get_physical_child(&nodes[parent].schema, child_index, &next);
                    break;
                }
                cursor = parent;
            }
        }
        if (status != DORA_OK) {
            break;
        }
    }
    dora_allocation_release(&storage);
    return status;
}
