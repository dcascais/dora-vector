// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Admission and Geometry
//
//------------------------------------------------------------------------------

static bool ranges_overlap(const void* first, size_t first_byte_count, const void* second, size_t second_byte_count) {
    const uintptr_t first_address = (uintptr_t)first;
    const uintptr_t second_address = (uintptr_t)second;
    if (first_byte_count == 0 || second_byte_count == 0) {
        return false;
    }
    if (first_address <= second_address) {
        return second_address - first_address < first_byte_count;
    }
    return first_address - second_address < second_byte_count;
}

static int rounded_capacity(size_t byte_count, size_t alignment_bytes, size_t* out_capacity_bytes) {
    size_t rounded;
    if (ckd_add(&rounded, byte_count, alignment_bytes - 1)) {
        return EOVERFLOW;
    }
    *out_capacity_bytes = rounded & ~(alignment_bytes - 1);
    return DORA_OK;
}

static bool allocation_is_empty(const struct dora_allocation* allocation) {
    return allocation->address == nullptr && allocation->capacity_bytes == 0 && allocation->pool.index == 0 &&
           allocation->pool.generation == 0 && !allocation->is_raw;
}

static bool allocation_is_owner(const struct dora_allocation* allocation, const struct pool_row* row) {
    return allocation->address != nullptr && allocation->capacity_bytes != 0 &&
           allocation->capacity_bytes % row->budget.alignment_bytes == 0;
}

static bool budget_admits(const struct memory_budget* budget, size_t byte_count) {
    return byte_count <= budget->limit_bytes - budget->used_bytes - budget->overhead_bytes;
}

static int preferred_capacity(const struct pool_row* row,
                              size_t current_capacity_bytes,
                              size_t required_byte_count,
                              size_t* out_capacity_bytes) {
    const size_t alignment_bytes = row->budget.alignment_bytes;
    const size_t growth_bytes =
        row->allocator == dora_memory_host_allocator() ? current_capacity_bytes : current_capacity_bytes / 2;
    size_t preferred = 0;
    if (ckd_add(&preferred, current_capacity_bytes, growth_bytes) || preferred < required_byte_count ||
        rounded_capacity(preferred, alignment_bytes, &preferred) != DORA_OK) {
        return rounded_capacity(required_byte_count, alignment_bytes, out_capacity_bytes);
    }
    *out_capacity_bytes = preferred;
    return DORA_OK;
}

int dora_memory_preferred_capacity(struct dora_pool pool,
                                   size_t current_capacity_bytes,
                                   size_t required_byte_count,
                                   size_t* out_capacity_bytes) {
    const struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_capacity_bytes == nullptr) {
        return EINVAL;
    }
    return preferred_capacity(row, current_capacity_bytes, required_byte_count, out_capacity_bytes);
}

//------------------------------------------------------------------------------
//
// Capacity Acquisition
//
//------------------------------------------------------------------------------

// Allocator reentry requires reset and release to stay blocked.
static int backend_obtain(struct pool_row* row, size_t capacity_bytes, void** out_address) {
    if (row->is_draining) {
        return EBUSY;
    }
    struct dora_allocator* const allocator = row->allocator;
    const size_t alignment_bytes = row->budget.alignment_bytes;
    void* address = nullptr;
    ++row->work_depth;
    int status = allocator->allocate(allocator, capacity_bytes, alignment_bytes, &address);
    if (status != DORA_OK && address != nullptr) {
        allocator->deallocate(allocator, address, capacity_bytes, alignment_bytes);
        status = EINVAL;
    } else if (status == DORA_OK && (address == nullptr || (uintptr_t)address % alignment_bytes != 0)) {
        if (address != nullptr) {
            allocator->deallocate(allocator, address, capacity_bytes, alignment_bytes);
        }
        status = EINVAL;
    }
    --row->work_depth;
    if (status == DORA_OK) {
        *out_address = address;
    }
    return status;
}

static void backend_retire(struct pool_row* row, void* address, size_t capacity_bytes) {
    ++row->work_depth;
    row->allocator->deallocate(row->allocator, address, capacity_bytes, row->budget.alignment_bytes);
    --row->work_depth;
}

static int acquire_capacity(struct pool_row* row,
                            size_t capacity_bytes,
                            size_t* in_out_charged_bytes,
                            void** out_address,
                            bool* in_out_has_attempted_recovery) {
    const struct dora_pool pool = {.index = (uint16_t)(row - dora_pool_rows), .generation = row->generation};
    for (;;) {
        int status = ENOMEM;
        if (row->is_draining) {
            return EBUSY;
        }
        if (budget_admits(&row->budget, capacity_bytes)) {
            *in_out_charged_bytes += capacity_bytes;
            status = backend_obtain(row, capacity_bytes, out_address);
            if (status == DORA_OK) {
                return status;
            }
            *in_out_charged_bytes -= capacity_bytes;
        }
        if (status != ENOMEM || *in_out_has_attempted_recovery) {
            return status;
        }
        *in_out_has_attempted_recovery = true;
        bool evicted = row->idle_workspace != nullptr;
        dora_scratch_idle_drain(pool);
        if (pool_row_of(pool) != row || row->is_draining) {
            return EBUSY;
        }
        evicted |= dora_schema_cache_evict(pool);
        if (pool_row_of(pool) != row || row->is_draining) {
            return EBUSY;
        }
        if (!evicted) {
            return ENOMEM;
        }
    }
}

static int allocate_rounded(struct pool_row* row,
                            struct dora_pool pool,
                            size_t capacity,
                            bool* in_out_has_attempted_recovery,
                            struct dora_allocation* out_allocation) {
    void* address = nullptr;
    const int status =
        acquire_capacity(row, capacity, &row->budget.used_bytes, &address, in_out_has_attempted_recovery);
    if (status == DORA_OK) {
        if (!allocation_is_empty(out_allocation)) {
            backend_retire(row, address, capacity);
            row->budget.used_bytes -= capacity;
            return EINVAL;
        }
        *out_allocation = (struct dora_allocation){.address = address, .capacity_bytes = capacity, .pool = pool};
    }
    return status;
}

int dora_memory_payload_allocate_attempt(struct dora_pool pool,
                                         size_t required_byte_count,
                                         bool* in_out_has_attempted_recovery,
                                         struct dora_allocation* out_allocation) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || in_out_has_attempted_recovery == nullptr || out_allocation == nullptr ||
        !allocation_is_empty(out_allocation)) {
        return EINVAL;
    }
    if (required_byte_count == 0) {
        return DORA_OK;
    }
    size_t capacity = 0;
    const int status = rounded_capacity(required_byte_count, row->budget.alignment_bytes, &capacity);
    return status == DORA_OK ? allocate_rounded(row, pool, capacity, in_out_has_attempted_recovery, out_allocation)
                             : status;
}

int dora_memory_payload_allocate_capacity(struct dora_pool pool,
                                          size_t required_byte_count,
                                          size_t preferred_byte_count,
                                          struct dora_allocation* out_allocation) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_allocation == nullptr || !allocation_is_empty(out_allocation) ||
        preferred_byte_count < required_byte_count) {
        return EINVAL;
    }
    if (required_byte_count == 0) {
        return DORA_OK;
    }
    size_t exact_capacity_bytes = 0;
    size_t capacity = 0;
    int status = rounded_capacity(required_byte_count, row->budget.alignment_bytes, &exact_capacity_bytes);
    if (status == DORA_OK &&
        rounded_capacity(preferred_byte_count, row->budget.alignment_bytes, &capacity) != DORA_OK) {
        capacity = exact_capacity_bytes;
    }
    if (status != DORA_OK) {
        return status;
    }
    bool drained = false;
    if (!budget_admits(&row->budget, capacity)) {
        capacity = exact_capacity_bytes;
    }
    status = allocate_rounded(row, pool, capacity, &drained, out_allocation);
    if (status == ENOMEM && capacity != exact_capacity_bytes) {
        status = allocate_rounded(row, pool, exact_capacity_bytes, &drained, out_allocation);
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Overhead Allocations
//
//------------------------------------------------------------------------------

int dora_memory_overhead_allocate(struct dora_pool pool, size_t byte_count, void** out_address) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_address == nullptr) {
        return EINVAL;
    }
    if (byte_count == 0) {
        return DORA_OK;
    }
    size_t capacity_bytes;
    const int status = rounded_capacity(byte_count, row->budget.alignment_bytes, &capacity_bytes);
    if (status != DORA_OK) {
        return status;
    }
    bool drained = false;
    return acquire_capacity(row, capacity_bytes, &row->budget.overhead_bytes, out_address, &drained);
}

void dora_memory_overhead_release(struct dora_pool pool, void* address, size_t byte_count) {
    struct pool_row* const row = pool_row_of(pool);
    size_t capacity_bytes = 0;
    if (address == nullptr || row == nullptr ||
        rounded_capacity(byte_count, row->budget.alignment_bytes, &capacity_bytes) != DORA_OK) {
        return;
    }
    backend_retire(row, address, capacity_bytes);
    row->budget.overhead_bytes -= capacity_bytes;
}

//------------------------------------------------------------------------------
//
// Payload Allocations
//
//------------------------------------------------------------------------------

int dora_memory_payload_allocate(struct dora_pool pool,
                                 size_t required_byte_count,
                                 struct dora_allocation* out_allocation) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_allocation == nullptr || !allocation_is_empty(out_allocation)) {
        return EINVAL;
    }
    if (required_byte_count == 0) {
        return DORA_OK;
    }
    size_t capacity_bytes;
    const int status = rounded_capacity(required_byte_count, row->budget.alignment_bytes, &capacity_bytes);
    if (status != DORA_OK) {
        return status;
    }
    bool drained = false;
    return allocate_rounded(row, pool, capacity_bytes, &drained, out_allocation);
}

int dora_memory_payload_allocate_preferred(struct dora_pool pool,
                                           size_t current_capacity_bytes,
                                           size_t required_byte_count,
                                           struct dora_allocation* out_allocation) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || out_allocation == nullptr || !allocation_is_empty(out_allocation)) {
        return EINVAL;
    }
    size_t preferred_capacity_bytes = 0;
    const int status = preferred_capacity(row, current_capacity_bytes, required_byte_count, &preferred_capacity_bytes);
    if (status != DORA_OK) {
        return status;
    }
    return dora_memory_payload_allocate_capacity(pool, required_byte_count, preferred_capacity_bytes, out_allocation);
}

int dora_memory_payload_reserve(struct dora_pool pool,
                                size_t required_byte_count,
                                size_t initialized_prefix_byte_count,
                                struct dora_allocation* in_out_allocation) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || in_out_allocation == nullptr) {
        return EINVAL;
    }
    if (!allocation_is_empty(in_out_allocation) &&
        (!pool_is_same(in_out_allocation->pool, pool) || !allocation_is_owner(in_out_allocation, row) ||
         ranges_overlap(in_out_allocation,
                        sizeof(*in_out_allocation),
                        in_out_allocation->address,
                        in_out_allocation->capacity_bytes))) {
        return EINVAL;
    }
    if (initialized_prefix_byte_count > required_byte_count ||
        initialized_prefix_byte_count > in_out_allocation->capacity_bytes) {
        return EINVAL;
    }
    size_t exact_capacity_bytes;
    int status = rounded_capacity(required_byte_count, row->budget.alignment_bytes, &exact_capacity_bytes);
    if (status != DORA_OK || exact_capacity_bytes <= in_out_allocation->capacity_bytes) {
        return status;
    }
    size_t preferred_capacity_bytes = 0;
    status = preferred_capacity(row, in_out_allocation->capacity_bytes, required_byte_count, &preferred_capacity_bytes);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_allocation candidate = {};
    status = dora_memory_payload_allocate_capacity(pool, required_byte_count, preferred_capacity_bytes, &candidate);
    if (status != DORA_OK) {
        return status;
    }
    if (initialized_prefix_byte_count != 0) {
        memcpy(candidate.address, in_out_allocation->address, initialized_prefix_byte_count);
    }
    candidate.is_raw = in_out_allocation->is_raw;
    const struct dora_allocation retired = *in_out_allocation;
    *in_out_allocation = candidate;
    if (retired.address != nullptr) {
        backend_retire(row, retired.address, retired.capacity_bytes);
        row->budget.used_bytes -= retired.capacity_bytes;
    }
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Allocation
//
//------------------------------------------------------------------------------

//--------------------------------------
// Capacity
//--------------------------------------

int dora_pool_allocate(struct dora_pool pool, size_t required_byte_count, struct dora_allocation* out_allocation) {
    const int status = dora_memory_payload_allocate(pool, required_byte_count, out_allocation);
    if (status != DORA_OK || out_allocation->address == nullptr) {
        return status;
    }
    out_allocation->is_raw = true;
    ++pool_row_of(pool)->budget.raw_allocation_count;
    return DORA_OK;
}

int dora_allocation_reserve(struct dora_pool pool,
                            size_t required_byte_count,
                            size_t initialized_prefix_byte_count,
                            struct dora_allocation* in_out_allocation) {
    const bool is_birth = in_out_allocation != nullptr && in_out_allocation->address == nullptr;
    const int status =
        dora_memory_payload_reserve(pool, required_byte_count, initialized_prefix_byte_count, in_out_allocation);
    if (status != DORA_OK || !is_birth || in_out_allocation->address == nullptr) {
        return status;
    }
    in_out_allocation->is_raw = true;
    ++pool_row_of(pool)->budget.raw_allocation_count;
    return DORA_OK;
}

//--------------------------------------
// Ownership
//--------------------------------------

int dora_allocation_move(struct dora_allocation* in_out_source, struct dora_allocation* out_destination) {
    if (in_out_source == nullptr || out_destination == nullptr ||
        ranges_overlap(in_out_source, sizeof(*in_out_source), out_destination, sizeof(*out_destination)) ||
        !allocation_is_empty(out_destination)) {
        return EINVAL;
    }
    if (!allocation_is_empty(in_out_source)) {
        const struct pool_row* const row = pool_row_of(in_out_source->pool);
        if (row == nullptr || !allocation_is_owner(in_out_source, row)) {
            return EINVAL;
        }
    }
    *out_destination = *in_out_source;
    *in_out_source = (struct dora_allocation){};
    return DORA_OK;
}

void dora_allocation_internalize(struct dora_allocation* allocation) {
    if (allocation->is_raw) {
        --pool_row_of(allocation->pool)->budget.raw_allocation_count;
        allocation->is_raw = false;
    }
}

int dora_allocation_adopt(struct dora_allocation* in_out_source, struct dora_allocation* out_destination) {
    const int status = dora_allocation_move(in_out_source, out_destination);
    if (status == DORA_OK) {
        dora_allocation_internalize(out_destination);
    }
    return status;
}

void dora_allocation_release(struct dora_allocation* allocation) {
    if (allocation == nullptr || allocation->address == nullptr) {
        return;
    }
    const struct dora_allocation retired = *allocation;
    *allocation = (struct dora_allocation){};
    struct pool_row* const row = pool_row_of(retired.pool);
    if (row == nullptr) {
        return;
    }
    backend_retire(row, retired.address, retired.capacity_bytes);
    if (retired.is_raw) {
        --row->budget.raw_allocation_count;
    }
    row->budget.used_bytes -= retired.capacity_bytes;
}
