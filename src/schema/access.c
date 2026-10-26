// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "identity/identity.h"
#include "schema/schema_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Node Inspection
//
//------------------------------------------------------------------------------

static struct dora_schema
child_borrow(const struct dora_schema* parent, const struct schema_image* image, uint32_t child_index) {
    const struct dora_schema* const child =
        &image->children[image->nodes[parent->node_index].first_child_index + child_index];
    struct dora_schema borrow = {
        .handle = child->handle.slot_index == 0 ? parent->handle : child->handle,
        .node_index = child->node_index
    };
    borrow.handle.is_owner = false;
    return borrow;
}

int dora_schema_resolve_image(const struct dora_schema* schema, const struct schema_image** out_image) {
    const struct identity_slot* slot = nullptr;
    if (schema == nullptr || out_image == nullptr ||
        dora_identity_resolve(&schema->handle, IDENTITY_KIND_SCHEMA, &slot) != DORA_OK) {
        return EINVAL;
    }
    const struct schema_image* const image = slot->claims->payload;
    if (schema->node_index >= image->node_count) {
        return EINVAL;
    }
    *out_image = image;
    return DORA_OK;
}

int dora_schema_get_pool(const struct dora_schema* schema, struct dora_pool* out_pool) {
    const struct schema_image* image = nullptr;
    if (out_pool == nullptr || dora_schema_resolve_image(schema, &image) != DORA_OK) {
        return EINVAL;
    }
    *out_pool = image->storage.pool;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Navigation
//
//------------------------------------------------------------------------------

int dora_schema_get_field(const struct dora_schema* schema, struct dora_schema_field* out_field) {
    const struct schema_image* image = nullptr;
    if (out_field == nullptr || dora_schema_resolve_image(schema, &image) != DORA_OK) {
        return EINVAL;
    }
    *out_field = image->nodes[schema->node_index].field;
    return DORA_OK;
}

int dora_schema_get_physical_child(const struct dora_schema* schema,
                                   uint32_t child_index,
                                   struct dora_schema* out_child) {
    const struct schema_image* image = nullptr;
    if (out_child == nullptr || !schema_is_empty(out_child) || dora_schema_resolve_image(schema, &image) != DORA_OK ||
        child_index >= image->nodes[schema->node_index].field.child_count) {
        return EINVAL;
    }
    *out_child = child_borrow(schema, image, child_index);
    return DORA_OK;
}

static int
public_parent(const struct dora_schema* schema, struct dora_schema* out_parent, struct dora_schema_field* out_field) {
    int status = dora_schema_get_field(schema, out_field);
    if (status != DORA_OK) {
        return status;
    }
    *out_parent = dora_schema_borrow(schema);
    if (out_field->type.id == DORA_TYPE_MAP && out_field->layout != DORA_LAYOUT_DICTIONARY &&
        out_field->layout != DORA_LAYOUT_RUN_END) {
        *out_parent = (struct dora_schema){};
        status = dora_schema_get_physical_child(schema, 0, out_parent);
        if (status == DORA_OK) {
            status = dora_schema_get_field(out_parent, out_field);
        }
    }
    return status;
}

int dora_schema_get_child_count(const struct dora_schema* schema, uint32_t* out_child_count) {
    if (out_child_count == nullptr) {
        return EINVAL;
    }
    struct dora_schema parent = {};
    struct dora_schema_field field = {};
    const int status = public_parent(schema, &parent, &field);
    if (status == DORA_OK) {
        *out_child_count = field.child_count;
    }
    return status;
}

int dora_schema_get_child(const struct dora_schema* schema, uint32_t child_index, struct dora_schema* out_child) {
    if (out_child == nullptr || !schema_is_empty(out_child)) {
        return EINVAL;
    }
    struct dora_schema parent = {};
    struct dora_schema_field field = {};
    const int status = public_parent(schema, &parent, &field);
    return status == DORA_OK ? dora_schema_get_physical_child(&parent, child_index, out_child) : status;
}

int dora_schema_find_child(const struct dora_schema* schema, const char* name, struct dora_schema* out_child) {
    if (name == nullptr || out_child == nullptr || !schema_is_empty(out_child)) {
        return EINVAL;
    }
    uint32_t child_count = 0;
    int status = dora_schema_get_child_count(schema, &child_count);
    for (uint32_t child_index = 0; status == DORA_OK && child_index < child_count; ++child_index) {
        struct dora_schema child = {};
        struct dora_schema_field field = {};
        status = dora_schema_get_child(schema, child_index, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == DORA_OK && strcmp(field.name, name) == 0) {
            *out_child = child;
            return DORA_OK;
        }
    }
    return status == DORA_OK ? ENOENT : status;
}
