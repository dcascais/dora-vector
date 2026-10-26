// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/vector.h"
#include "extension/extension_internal.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

static bool pool_is_empty(const struct dora_pool* pool) {
    return pool->index == 0 && pool->generation == 0;
}

int dora_pool_init_allocator(struct dora_pool* pool,
                             struct dora_allocator* allocator,
                             size_t limit_bytes,
                             size_t alignment_bytes) {
    if (pool == nullptr || !pool_is_empty(pool)) {
        return EINVAL;
    }
    return dora_pool_registry_open(pool, allocator, limit_bytes, alignment_bytes);
}

int dora_pool_init(struct dora_pool* pool, size_t limit_bytes, size_t alignment_bytes) {
    return dora_pool_init_allocator(pool, dora_memory_host_allocator(), limit_bytes, alignment_bytes);
}

int dora_pool_init_fixed_region(struct dora_pool* pool,
                                void* region,
                                size_t region_byte_count,
                                size_t alignment_bytes) {
    if (pool == nullptr || !pool_is_empty(pool) || region == nullptr || !memory_alignment_is_valid(alignment_bytes)) {
        return EINVAL;
    }
    const uintptr_t pool_address = (uintptr_t)pool;
    const uintptr_t region_address = (uintptr_t)region;
    uintptr_t pool_end;
    uintptr_t region_end;
    if (ckd_add(&pool_end, pool_address, sizeof(*pool)) || ckd_add(&region_end, region_address, region_byte_count)) {
        return ENOMEM;
    }
    if (region_byte_count != 0 && region_address < pool_end && pool_address < region_end) {
        return EINVAL;
    }
    struct dora_allocator* allocator = nullptr;
    size_t limit_bytes = 0;
    const int status =
        dora_memory_fixed_region_init(region, region_byte_count, alignment_bytes, &allocator, &limit_bytes);
    if (status != DORA_OK) {
        return status;
    }
    return dora_pool_registry_open(pool, allocator, limit_bytes, alignment_bytes);
}

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

//--------------------------------------
// Composition
//--------------------------------------

static void owner_release(struct dora_handle owner, enum identity_kind kind) {
    switch (kind) {
    case IDENTITY_KIND_SCHEMA: {
        struct dora_schema schema = {.handle = owner};
        dora_schema_release(&schema);
        return;
    }
    case IDENTITY_KIND_VECTOR: {
        struct dora_vector vector = {.handle = owner};
        dora_vector_release(&vector);
        return;
    }
    case IDENTITY_KIND_BUILDER: {
        struct dora_builder builder = {.handle = owner};
        dora_builder_release(&builder);
        return;
    }
    case IDENTITY_KIND_READER: {
        struct dora_reader reader = {.handle = owner};
        dora_reader_release(&reader);
        return;
    }
    case IDENTITY_KIND_WRITER: {
        struct dora_writer writer = {.handle = owner};
        dora_writer_release(&writer);
        return;
    }
    case IDENTITY_KIND_SCHEMA_INFERENCE: {
        struct dora_schema_inference inference = {.handle = owner};
        dora_schema_inference_release(&inference);
        return;
    }
    case IDENTITY_KIND_PIPELINE: {
        struct dora_pipeline pipeline = {.handle = owner};
        dora_pipeline_release(&pipeline);
        return;
    }
    }
}

int dora_pool_release(struct dora_pool* pool) {
    if (pool == nullptr) {
        return EINVAL;
    }
    struct pool_row* const row = pool_row_of(*pool);
    if (row == nullptr) {
        return EINVAL;
    }
    if (row->work_depth != 0 || row->is_draining) {
        return EBUSY;
    }
    if (row->directory.live_slot_count != dora_schema_cache_slot_count(*pool) ||
        row->budget.used_bytes != dora_scratch_idle_bytes(*pool) + dora_schema_cache_bytes(*pool)) {
        return EBUSY;
    }
    row->is_draining = true;
    dora_scratch_idle_drain(*pool);
    dora_schema_cache_drain(*pool);
    dora_extension_registrations_close(*pool);
    dora_identity_directory_close(*pool);
    dora_pool_registry_close(pool);
    return DORA_OK;
}

int dora_pool_reset(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return EINVAL;
    }
    if (row->work_depth != 0 || row->is_draining) {
        return EBUSY;
    }
    if (row->budget.raw_allocation_count != 0 || row->directory.cross_pool_claim_count != 0) {
        return EBUSY;
    }
    row->is_draining = true;
    for (uint32_t index = 1; index < row->directory.slot_limit; ++index) {
        const struct identity_slot* const slot = dora_identity_slot_at(&row->directory, index);
        if (slot->claims != nullptr && slot->holder_pool_index == 0) {
            owner_release(
                (struct dora_handle){
                    .slot_index = index,
                    .generation = slot->generation,
                    .pool_index = pool.index,
                    .is_owner = true
                },
                slot->kind);
        }
    }
    dora_scratch_idle_drain(pool);
    dora_schema_cache_drain(pool);
    dora_identity_directory_reset(pool);
    row->is_draining = false;
    return DORA_OK;
}

bool dora_pool_is_valid(struct dora_pool pool) {
    return pool_row_of(pool) != nullptr;
}

//--------------------------------------
// Reservation
//--------------------------------------

int dora_pool_reserve_slots(struct dora_pool pool, uint32_t slot_count) {
    return dora_identity_directory_reserve(pool, slot_count);
}

//------------------------------------------------------------------------------
//
// Accounting
//
//------------------------------------------------------------------------------

int dora_pool_get_used_bytes(struct dora_pool pool, size_t* out_used_bytes) {
    const struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_used_bytes == nullptr) {
        return EINVAL;
    }
    *out_used_bytes = row->budget.used_bytes;
    return DORA_OK;
}

int dora_pool_get_overhead_bytes(struct dora_pool pool, size_t* out_overhead_bytes) {
    const struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_overhead_bytes == nullptr) {
        return EINVAL;
    }
    *out_overhead_bytes = row->budget.overhead_bytes;
    return DORA_OK;
}
