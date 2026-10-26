// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scoped scratch storage.

#ifndef DORA_MEMORY_SCRATCH_H
#define DORA_MEMORY_SCRATCH_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"

#include <stddef.h>

//------------------------------------------------------------------------------
//
// Scratch Allocations
//
//------------------------------------------------------------------------------

struct scratch_slot {
    struct dora_allocation allocation;
    struct scratch_slot* next;
    void* address;
    size_t capacity_bytes;
    size_t extent_bytes;
    size_t slot_index;
};

struct scratch_workspace {
    struct dora_allocation allocation;
    struct scratch_slot* slots;
    struct scratch_slot* arena;
};

struct scratch_scope {
    struct dora_pool pool;
    struct scratch_workspace* workspace;
    struct scratch_slot* arena_cursor;
};

int dora_scratch_scope_begin(struct dora_pool pool, struct scratch_scope* scope);
// Each slot may be acquired once per scope.
int dora_scratch_slot_require(struct scratch_scope* scope,
                              size_t slot_index,
                              void** out_address,
                              size_t byte_count,
                              size_t alignment_bytes);
int dora_scratch_allocate(struct scratch_scope* scope, size_t byte_count, void** out_address);
int dora_scratch_allocate_growing(struct scratch_scope* scope,
                                  size_t byte_count,
                                  size_t current_capacity_bytes,
                                  void** out_address,
                                  size_t* out_capacity_bytes);
void dora_scratch_scope_end(struct scratch_scope* scope);
size_t dora_scratch_idle_bytes(struct dora_pool pool);
void dora_scratch_idle_drain(struct dora_pool pool);

#endif
