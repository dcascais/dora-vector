// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Pool allocation and accounting.

#ifndef DORA_VECTOR_MEMORY_H
#define DORA_VECTOR_MEMORY_H

#include "dora/vector/handle.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Allocation State
//
//------------------------------------------------------------------------------

//--------------------------------------
// Allocator
//--------------------------------------

/// Callbacks run synchronously; must preserve active arguments; pool reset and release return EBUSY.
struct dora_allocator {
    /// Allocates capacity_bytes with alignment_bytes.
    int (*allocate)(struct dora_allocator* allocator,
                    size_t capacity_bytes,
                    size_t alignment_bytes,
                    void** out_address);
    /// Frees a matching allocation.
    void (*deallocate)(struct dora_allocator* allocator, void* address, size_t capacity_bytes, size_t alignment_bytes);
};

//--------------------------------------
// Allocation
//--------------------------------------

struct dora_allocation {
    /// Allocation start.
    void* address;
    /// Allocated bytes.
    size_t capacity_bytes;
    /// Charged pool.
    struct dora_pool pool;
    /// Whether the allocation remains caller-owned.
    bool is_raw;
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

/// Charges payload, overhead and retained capacity to limit_bytes; zero admits no bytes.
int dora_pool_init(struct dora_pool* pool, size_t limit_bytes, size_t alignment_bytes);

/// Borrows allocator without host fallback; zero limit_bytes admits no bytes.
int dora_pool_init_allocator(struct dora_pool* pool,
                             struct dora_allocator* allocator,
                             size_t limit_bytes,
                             size_t alignment_bytes);

/// Borrows region; alignment_bytes sets its granule size; more than 65,536 granules returns ERANGE.
int dora_pool_init_fixed_region(struct dora_pool* pool, void* region, size_t region_byte_count, size_t alignment_bytes);

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

/// Returns EBUSY while objects, raw allocations, claims or pool work remain; success invalidates handles.
int dora_pool_release(struct dora_pool* pool);

/// Frees pool-owned objects; raw allocations, foreign claims or active pool work return EBUSY without changes.
int dora_pool_reset(struct dora_pool pool);

bool dora_pool_is_valid(struct dora_pool pool);

/// Reserves total identity capacity; failure may retain acquired segments without changing objects.
int dora_pool_reserve_slots(struct dora_pool pool, uint32_t slot_count);

//------------------------------------------------------------------------------
//
// Allocation
//
//------------------------------------------------------------------------------

int dora_pool_allocate(struct dora_pool pool, size_t required_byte_count, struct dora_allocation* out_allocation);

/// Preserves initialized_prefix_byte_count bytes; failure preserves the allocation; capacity stays pool-charged.
int dora_allocation_reserve(struct dora_pool pool,
                            size_t required_byte_count,
                            size_t initialized_prefix_byte_count,
                            struct dora_allocation* in_out_allocation);

int dora_allocation_move(struct dora_allocation* in_out_source, struct dora_allocation* out_destination);

void dora_allocation_release(struct dora_allocation* allocation);

//------------------------------------------------------------------------------
//
// Accounting
//
//------------------------------------------------------------------------------

/// Includes retained internal capacity until eviction, reset or release.
int dora_pool_get_used_bytes(struct dora_pool pool, size_t* out_used_bytes);

int dora_pool_get_overhead_bytes(struct dora_pool pool, size_t* out_overhead_bytes);

#endif
