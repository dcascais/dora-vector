// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Pool registry rows.

#ifndef DORA_MEMORY_REGISTRY_H
#define DORA_MEMORY_REGISTRY_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "extension/extension_internal.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Pool Rows
//
//------------------------------------------------------------------------------

struct pool_row {
    // nullptr while closed.
    struct dora_allocator* allocator;
    struct memory_budget budget;
    // Entered operations and callbacks in flight.
    uint32_t work_depth;
    uint16_t generation;
    bool is_draining;
    struct identity_directory directory;
    struct extension_registration* registrations;
    struct scratch_workspace* idle_workspace;
    struct schema_image* cached_schemas;
};

extern struct pool_row dora_pool_rows[DORA_POOL_CAPACITY];

//--------------------------------------
// Lookup
//--------------------------------------

// Ignores generation and liveness.
static inline struct pool_row* pool_row_at(uint16_t pool_index) {
    return pool_index < DORA_POOL_CAPACITY ? &dora_pool_rows[pool_index] : nullptr;
}

static inline struct pool_row* pool_row_of(struct dora_pool pool) {
    struct pool_row* const row = pool_row_at(pool.index);
    return row != nullptr && row->allocator != nullptr && row->generation == pool.generation ? row : nullptr;
}

//------------------------------------------------------------------------------
//
// Pool Registry
//
//------------------------------------------------------------------------------

int dora_pool_registry_open(struct dora_pool* out_pool,
                            struct dora_allocator* allocator,
                            size_t limit_bytes,
                            size_t alignment_bytes);

void dora_pool_registry_close(struct dora_pool* in_out_pool);

#endif
