// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "memory/registry.h"

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Row Storage
//
//------------------------------------------------------------------------------

struct pool_row dora_pool_rows[DORA_POOL_CAPACITY];

//------------------------------------------------------------------------------
//
// Pool Registry
//
//------------------------------------------------------------------------------

int dora_pool_registry_open(struct dora_pool* out_pool,
                            struct dora_allocator* allocator,
                            size_t limit_bytes,
                            size_t alignment_bytes) {
    if (out_pool == nullptr || allocator == nullptr || allocator->allocate == nullptr ||
        allocator->deallocate == nullptr || !memory_alignment_is_valid(alignment_bytes)) {
        return EINVAL;
    }
    for (uint32_t index = 1; index < DORA_POOL_CAPACITY; ++index) {
        struct pool_row* const row = &dora_pool_rows[index];
        if (row->allocator != nullptr || row->generation == UINT16_MAX ||
            row->directory.generation_floor == UINT32_MAX) {
            continue;
        }
        ++row->generation;
        row->budget = (struct memory_budget){.limit_bytes = limit_bytes, .alignment_bytes = alignment_bytes};
        row->allocator = allocator;
        row->work_depth = 0;
        row->is_draining = false;
        *out_pool = (struct dora_pool){.index = (uint16_t)index, .generation = row->generation};
        return DORA_OK;
    }
    return ERANGE;
}

void dora_pool_registry_close(struct dora_pool* in_out_pool) {
    struct pool_row* const row = pool_row_of(*in_out_pool);
    if (row == nullptr) {
        return;
    }
    row->allocator = nullptr;
    *in_out_pool = (struct dora_pool){};
}

void dora_pool_callback_enter(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row != nullptr) {
        ++row->work_depth;
    }
}

void dora_pool_callback_leave(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row != nullptr) {
        --row->work_depth;
    }
}
