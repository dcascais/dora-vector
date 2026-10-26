// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Schema images and traversal.

#ifndef DORA_SCHEMA_SCHEMA_INTERNAL_H
#define DORA_SCHEMA_SCHEMA_INTERNAL_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "identity/identity.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Field Topology
//
//------------------------------------------------------------------------------

struct schema_node {
    struct dora_schema_field field;
    uint32_t first_child_index;
    uint32_t subtree_end;
};

struct schema_walk_node {
    struct dora_schema schema;
    uint32_t parent_node_index;
    uint32_t child_index;
};

int dora_schema_walk(struct dora_pool pool,
                     const struct dora_schema* schema,
                     struct dora_allocation* out_nodes,
                     uint32_t* out_count);

//------------------------------------------------------------------------------
//
// Immutable Image
//
//------------------------------------------------------------------------------

struct schema_image {
    struct dora_allocation storage;
    struct identity_claim_state claims;
    struct dora_schema* children;
    struct schema_image* next_retired;
    struct schema_image* next_cached;
    struct dora_schema cached_owner;
    uint32_t node_count;
    uint32_t child_count;
    struct schema_node nodes[];
};

int dora_schema_node_claim(struct dora_pool pool,
                           const struct dora_schema_field* field,
                           const struct dora_schema* children,
                           struct dora_schema* out_schema);

size_t dora_schema_cache_bytes(struct dora_pool pool);
uint32_t dora_schema_cache_count(struct dora_pool pool);
uint32_t dora_schema_cache_slot_count(struct dora_pool pool);
void dora_schema_cache_drain(struct dora_pool pool);
bool dora_schema_cache_evict(struct dora_pool pool);

//------------------------------------------------------------------------------
//
// Node Inspection
//
//------------------------------------------------------------------------------

static inline bool schema_is_empty(const struct dora_schema* schema) {
    return schema != nullptr && handle_is_empty(schema->handle) && schema->node_index == 0;
}

int dora_schema_resolve_image(const struct dora_schema* schema, const struct schema_image** out_image);

int dora_schema_get_physical_child(const struct dora_schema* schema,
                                   uint32_t child_index,
                                   struct dora_schema* out_child);

int dora_schema_field_matches(const struct dora_schema* schema,
                              const struct dora_schema_field* field,
                              bool* out_matches);

//------------------------------------------------------------------------------
//
// Node Composition
//
//------------------------------------------------------------------------------

int dora_schema_init_retain_node(struct dora_schema* schema,
                                 struct dora_pool pool,
                                 const struct dora_schema_field* field,
                                 const struct dora_schema* children);

int dora_schema_init_retain_field(struct dora_schema* schema,
                                  struct dora_pool pool,
                                  const struct dora_schema* source,
                                  const struct dora_schema_field* field);

int dora_schema_init_retain_struct_in_pool(struct dora_schema* schema,
                                           struct dora_pool pool,
                                           const struct dora_schema* children,
                                           uint32_t child_count);

int dora_schema_declarations_equal(const struct dora_schema* left, const struct dora_schema* right, bool* out_equal);

#endif
