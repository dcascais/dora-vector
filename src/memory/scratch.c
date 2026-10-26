// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "memory/scratch.h"

#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Scratch Allocations
//
//------------------------------------------------------------------------------

static void scratch_slots_release(struct scratch_slot* slot) {
    while (slot != nullptr) {
        struct scratch_slot* const next = slot->next;
        struct dora_allocation allocation = slot->allocation;
        dora_allocation_release(&allocation);
        slot = next;
    }
}

static void scratch_workspace_release(struct scratch_workspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    struct dora_allocation allocation = workspace->allocation;
    scratch_slots_release(workspace->slots);
    scratch_slots_release(workspace->arena);
    dora_allocation_release(&allocation);
}

size_t dora_scratch_idle_bytes(struct dora_pool pool) {
    const struct pool_row* const row = pool_row_of(pool);
    const struct scratch_workspace* const workspace = row == nullptr ? nullptr : row->idle_workspace;
    if (workspace == nullptr) {
        return 0;
    }
    size_t bytes = workspace->allocation.capacity_bytes;
    for (const struct scratch_slot* slot = workspace->slots; slot != nullptr; slot = slot->next) {
        bytes += slot->allocation.capacity_bytes;
    }
    for (const struct scratch_slot* slot = workspace->arena; slot != nullptr; slot = slot->next) {
        bytes += slot->allocation.capacity_bytes;
    }
    return bytes;
}

void dora_scratch_idle_drain(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return;
    }
    struct scratch_workspace* const workspace = row->idle_workspace;
    row->idle_workspace = nullptr;
    dora_pool_callback_enter(pool);
    scratch_workspace_release(workspace);
    dora_pool_callback_leave(pool);
}

int dora_scratch_scope_begin(struct dora_pool pool, struct scratch_scope* scope) {
    struct pool_row* const row = pool_row_of(pool);
    if (scope == nullptr || row == nullptr) {
        return EINVAL;
    }
    if (row->is_draining || row->work_depth == UINT32_MAX) {
        return EBUSY;
    }
    ++row->work_depth;
    *scope = (struct scratch_scope){.pool = pool};
    return DORA_OK;
}

static int scratch_lease(struct scratch_scope* scope) {
    struct pool_row* const row = pool_row_of(scope->pool);
    if (row == nullptr || row->is_draining) {
        return EBUSY;
    }
    if (scope->workspace != nullptr) {
        return DORA_OK;
    }
    struct scratch_workspace* workspace = row->idle_workspace;
    row->idle_workspace = nullptr;
    if (workspace == nullptr) {
        struct dora_allocation allocation = {};
        const int status = dora_memory_payload_allocate(scope->pool, sizeof(*workspace), &allocation);
        if (status != DORA_OK) {
            return status;
        }
        workspace = allocation.address;
        *workspace = (struct scratch_workspace){.allocation = allocation};
    }
    scope->workspace = workspace;
    return DORA_OK;
}

static int scratch_slot_create(size_t alignment_bytes,
                               struct scratch_scope* scope,
                               size_t required,
                               size_t preferred,
                               struct scratch_slot** out_slot) {
    struct dora_allocation allocation = {};
    const int status = dora_memory_payload_allocate_capacity(scope->pool, required, preferred, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    const struct pool_row* const row = pool_row_of(scope->pool);
    if (row == nullptr || row->is_draining) {
        dora_allocation_release(&allocation);
        return EBUSY;
    }
    struct scratch_slot* const slot = allocation.address;
    const uintptr_t start = (uintptr_t)allocation.address + sizeof(*slot);
    const size_t padding = (alignment_bytes - start % alignment_bytes) % alignment_bytes;
    const size_t offset = sizeof(*slot) + padding;
    *slot = (struct scratch_slot){
        .allocation = allocation,
        .address = (unsigned char*)allocation.address + offset,
        .capacity_bytes = allocation.capacity_bytes - offset,
    };
    *out_slot = slot;
    return DORA_OK;
}

int dora_scratch_slot_require(struct scratch_scope* scope,
                              size_t slot_index,
                              void** out_address,
                              size_t byte_count,
                              size_t alignment_bytes) {
    if (scope == nullptr || out_address == nullptr || dora_bitmap_word_count_ones(alignment_bytes) != 1) {
        return EINVAL;
    }
    if (byte_count == 0) {
        *out_address = nullptr;
        return DORA_OK;
    }
    if (alignment_bytes < 64) {
        alignment_bytes = 64;
    }
    size_t rounded;
    if (ckd_add(&rounded, byte_count, alignment_bytes - 1)) {
        return EOVERFLOW;
    }
    byte_count = rounded / alignment_bytes * alignment_bytes;
    int status = scratch_lease(scope);
    if (status != DORA_OK) {
        return status;
    }
    struct scratch_slot** link = &scope->workspace->slots;
    while (*link != nullptr && (*link)->slot_index != slot_index) {
        link = &(*link)->next;
    }
    struct scratch_slot* const previous = *link;
    if (previous != nullptr && previous->extent_bytes != 0) {
        return EINVAL;
    }
    if (previous != nullptr && previous->capacity_bytes >= byte_count &&
        (uintptr_t)previous->address % alignment_bytes == 0) {
        previous->extent_bytes = byte_count;
        *out_address = previous->address;
        return DORA_OK;
    }
    size_t required;
    if (ckd_add(&required, sizeof(struct scratch_slot), alignment_bytes - 1) ||
        ckd_add(&required, required, byte_count)) {
        return EOVERFLOW;
    }
    size_t preferred = 0;
    status = dora_memory_preferred_capacity(scope->pool,
                                            previous == nullptr ? 0 : previous->allocation.capacity_bytes,
                                            required,
                                            &preferred);
    if (status != DORA_OK) {
        return status;
    }
    struct scratch_slot* const next = previous == nullptr ? nullptr : previous->next;
    if (previous != nullptr) {
        struct dora_allocation allocation = previous->allocation;
        *link = next;
        dora_allocation_release(&allocation);
    }
    struct scratch_slot* slot = nullptr;
    status = scratch_slot_create(alignment_bytes, scope, required, preferred, &slot);
    if (status != DORA_OK) {
        return status;
    }
    slot->extent_bytes = byte_count;
    slot->slot_index = slot_index;
    slot->next = next;
    *link = slot;
    *out_address = slot->address;
    return DORA_OK;
}

static constexpr size_t ARENA_INITIAL_BYTES = 4096;
static constexpr size_t ARENA_MAX_BYTES = (size_t)16 * 1024 * 1024;

int dora_scratch_allocate_growing(struct scratch_scope* scope,
                                  size_t byte_count,
                                  size_t current_capacity_bytes,
                                  void** out_address,
                                  size_t* out_capacity_bytes) {
    if (scope == nullptr || out_address == nullptr) {
        return EINVAL;
    }
    if (byte_count == 0) {
        *out_address = nullptr;
        *out_capacity_bytes = 0;
        return DORA_OK;
    }
    size_t rounded;
    if (ckd_add(&rounded, byte_count, (size_t)63)) {
        return EOVERFLOW;
    }
    byte_count = rounded / 64 * 64;
    // Geometric bounds require admitted preferred extents.
    size_t preferred_bytes = byte_count;
    if (current_capacity_bytes != 0) {
        const int growth = dora_memory_preferred_capacity(scope->pool, current_capacity_bytes, byte_count, &rounded);
        if (growth != DORA_OK) {
            return growth;
        }
        preferred_bytes = ckd_add(&rounded, rounded, (size_t)63) ? byte_count : rounded / 64 * 64;
    }
    int status = scratch_lease(scope);
    if (status != DORA_OK) {
        return status;
    }
    struct scratch_slot** link = scope->arena_cursor == nullptr ? &scope->workspace->arena : &scope->arena_cursor;
    while (*link != nullptr && byte_count > (*link)->capacity_bytes - (*link)->extent_bytes) {
        link = &(*link)->next;
    }
    if (*link == nullptr) {
        const size_t alignment_bytes = pool_row_of(scope->pool)->budget.alignment_bytes;
        const size_t unit = alignment_bytes < 64 ? alignment_bytes : 64;
        // Header space must cover every pool-aligned base address.
        const size_t header = (sizeof(struct scratch_slot) - 1) / unit * unit + 64;
        size_t required;
        if (ckd_add(&required, header, byte_count)) {
            return EOVERFLOW;
        }
        size_t capacity = ARENA_INITIAL_BYTES;
        size_t previous = 0;
        for (const struct scratch_slot* block = scope->workspace->arena; block != nullptr; block = block->next) {
            const size_t physical = block->allocation.capacity_bytes;
            if (physical <= ARENA_MAX_BYTES && physical > previous) {
                previous = physical;
            }
        }
        if (previous != 0) {
            status = dora_memory_preferred_capacity(scope->pool, previous, required, &capacity);
            if (status != DORA_OK) {
                return status;
            }
        }
        size_t preferred;
        if (ckd_add(&preferred, header, preferred_bytes)) {
            preferred = required;
        }
        if (capacity < preferred) {
            capacity = preferred;
        }
        if (capacity > ARENA_MAX_BYTES) {
            capacity = ARENA_MAX_BYTES;
        }
        if (capacity < required) {
            capacity = required;
        }
        status = scratch_slot_create(64, scope, required, capacity, link);
        if (status != DORA_OK) {
            return status;
        }
    }
    struct scratch_slot* const block = *link;
    *out_address = (unsigned char*)block->address + block->extent_bytes;
    const size_t available = (block->capacity_bytes - block->extent_bytes) / 64 * 64;
    const size_t extent = preferred_bytes < available ? preferred_bytes : available;
    block->extent_bytes += extent;
    *out_capacity_bytes = extent;
    scope->arena_cursor = block;
    return DORA_OK;
}

int dora_scratch_allocate(struct scratch_scope* scope, size_t byte_count, void** out_address) {
    size_t capacity = 0;
    return dora_scratch_allocate_growing(scope, byte_count, 0, out_address, &capacity);
}

void dora_scratch_scope_end(struct scratch_scope* scope) {
    struct pool_row* const row = pool_row_of(scope->pool);
    if (row == nullptr) {
        return;
    }
    struct scratch_workspace* const workspace = scope->workspace;
    if (workspace != nullptr) {
        for (struct scratch_slot* slot = workspace->slots; slot != nullptr; slot = slot->next) {
            slot->extent_bytes = 0;
        }
        struct scratch_slot** retained = nullptr;
        for (struct scratch_slot** link = &workspace->arena; *link != nullptr; link = &(*link)->next) {
            if ((*link)->allocation.capacity_bytes <= ARENA_MAX_BYTES &&
                (retained == nullptr || (*link)->allocation.capacity_bytes > (*retained)->allocation.capacity_bytes)) {
                retained = link;
            }
        }
        struct scratch_slot* block = nullptr;
        if (retained != nullptr) {
            block = *retained;
            *retained = block->next;
            block->next = nullptr;
            block->extent_bytes = 0;
        }
        struct scratch_slot* const overflow = workspace->arena;
        workspace->arena = block;
        struct scratch_workspace* const retired = row->idle_workspace;
        row->idle_workspace = workspace;
        *scope = (struct scratch_scope){};
        scratch_slots_release(overflow);
        scratch_workspace_release(retired);
    } else {
        *scope = (struct scratch_scope){};
    }
    --row->work_depth;
}
