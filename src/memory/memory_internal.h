// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Pool allocation and callback brackets.

#ifndef DORA_MEMORY_MEMORY_INTERNAL_H
#define DORA_MEMORY_MEMORY_INTERNAL_H

#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Configuration
//
//------------------------------------------------------------------------------

#ifndef DORA_POOL_FIXED_REGION_MAX_BLOCKS
#define DORA_POOL_FIXED_REGION_MAX_BLOCKS 65'536U
#endif

#ifndef DORA_POOL_CAPACITY
#define DORA_POOL_CAPACITY 128U
#endif

//------------------------------------------------------------------------------
//
// Allocation State
//
//------------------------------------------------------------------------------

//--------------------------------------
// Fixed Region Allocator
//--------------------------------------

struct fixed_region_allocator {
    struct dora_allocator base;
    size_t block_count;
    size_t payload_byte_offset;
    size_t search_block_index;
    uint64_t bitmap[];
};

//--------------------------------------
// Backends
//--------------------------------------

static inline bool memory_alignment_is_valid(size_t alignment_bytes) {
    return alignment_bytes >= 8 && dora_bitmap_word_count_ones(alignment_bytes) == 1;
}

struct dora_allocator* dora_memory_host_allocator();

int dora_memory_fixed_region_init(void* region,
                                  size_t region_byte_count,
                                  size_t alignment_bytes,
                                  struct dora_allocator** out_allocator,
                                  size_t* out_limit_bytes);

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

static inline bool pool_is_same(struct dora_pool left, struct dora_pool right) {
    return left.index == right.index && left.generation == right.generation;
}

//--------------------------------------
// Callback Brackets
//--------------------------------------

// Prevents pool reset and release until leave; ignores invalid pools.
void dora_pool_callback_enter(struct dora_pool pool);
void dora_pool_callback_leave(struct dora_pool pool);

//------------------------------------------------------------------------------
//
// Allocation
//
//------------------------------------------------------------------------------

//--------------------------------------
// Budget
//--------------------------------------

struct memory_budget {
    size_t limit_bytes;
    size_t alignment_bytes;
    size_t used_bytes;
    size_t overhead_bytes;
    size_t raw_allocation_count;
};

//--------------------------------------
// Overhead Allocations
//--------------------------------------

// Requires the allocation geometry's byte_count; rounds to pool alignment.
int dora_memory_overhead_allocate(struct dora_pool pool, size_t byte_count, void** out_address);

void dora_memory_overhead_release(struct dora_pool pool, void* address, size_t byte_count);

//--------------------------------------
// Payload Allocations
//--------------------------------------

int dora_memory_payload_allocate(struct dora_pool pool,
                                 size_t required_byte_count,
                                 struct dora_allocation* out_allocation);

// Shares one idle-cache eviction across fallback attempts.
int dora_memory_payload_allocate_attempt(struct dora_pool pool,
                                         size_t required_byte_count,
                                         bool* in_out_has_attempted_recovery,
                                         struct dora_allocation* out_allocation);

// Requires empty output and preference >= requirement; retries the exact requirement on ENOMEM or overflow.
int dora_memory_payload_allocate_capacity(struct dora_pool pool,
                                          size_t required_byte_count,
                                          size_t preferred_byte_count,
                                          struct dora_allocation* out_allocation);

// Growth and exact fallback share one idle-cache eviction.
int dora_memory_payload_allocate_preferred(struct dora_pool pool,
                                           size_t current_capacity_bytes,
                                           size_t required_byte_count,
                                           struct dora_allocation* out_allocation);

int dora_memory_payload_reserve(struct dora_pool pool,
                                size_t required_byte_count,
                                size_t initialized_prefix_byte_count,
                                struct dora_allocation* in_out_allocation);

int dora_memory_preferred_capacity(struct dora_pool pool,
                                   size_t current_capacity_bytes,
                                   size_t required_byte_count,
                                   size_t* out_capacity_bytes);

//--------------------------------------
// Allocation ownership
//--------------------------------------

void dora_allocation_internalize(struct dora_allocation* allocation);

// Requires an empty destination; transfers ownership as an internal allocation.
int dora_allocation_adopt(struct dora_allocation* in_out_source, struct dora_allocation* out_destination);

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(alignof(struct dora_allocation) <= 8);
static_assert(DORA_POOL_CAPACITY <= (uint32_t)UINT16_MAX + 1);

#endif
