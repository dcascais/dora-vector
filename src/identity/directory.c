// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Directory Storage
//
//------------------------------------------------------------------------------

static constexpr uint32_t USABLE_SLOT_LIMIT =
    (uint64_t)IDENTITY_SEGMENT_FIRST_SLOT_COUNT * ((UINT64_C(1) << IDENTITY_SEGMENT_COUNT) - 1) - 1;

//--------------------------------------
// Segment Geometry
//--------------------------------------

static uint32_t segment_of(uint32_t slot) {
    return 63U - bitmap_word_leading_zeros((uint64_t)(slot / IDENTITY_SEGMENT_FIRST_SLOT_COUNT) + 1U);
}

static uint32_t segment_first_slot(uint32_t segment) {
    return IDENTITY_SEGMENT_FIRST_SLOT_COUNT * ((UINT32_C(1) << segment) - 1U);
}

static uint32_t segment_slot_count(uint32_t segment) {
    return IDENTITY_SEGMENT_FIRST_SLOT_COUNT << segment;
}

static size_t segment_byte_count(uint32_t segment) {
    return segment == 0 ? sizeof(struct identity_head_segment)
                        : (size_t)segment_slot_count(segment) * sizeof(struct identity_slot);
}

//--------------------------------------
// Lookup
//--------------------------------------

struct identity_slot* dora_identity_slot_at(struct identity_directory* directory, uint32_t slot) {
    if (slot >= directory->slot_limit) {
        return nullptr;
    }
    const uint32_t segment = segment_of(slot);
    return &directory->segments[segment][slot - segment_first_slot(segment)];
}

static struct identity_slot* resolve_slot(const struct dora_handle* handle, enum identity_kind expected_kind) {
    struct pool_row* const row = pool_row_at(handle->pool_index);
    struct identity_slot* const slot =
        row != nullptr ? dora_identity_slot_at(&row->directory, handle->slot_index) : nullptr;
    if (slot == nullptr || slot->claims == nullptr || slot->generation != handle->generation ||
        slot->kind != expected_kind) {
        return nullptr;
    }
    return slot;
}

//------------------------------------------------------------------------------
//
// Slot Occupancy
//
//------------------------------------------------------------------------------

//--------------------------------------
// Free List
//--------------------------------------

static void free_list_append(struct identity_directory* directory, uint32_t slot_index, struct identity_slot* slot) {
    slot->next_free_slot = 0;
    if (directory->free_tail_slot == 0) {
        directory->free_head_slot = slot_index;
    } else {
        dora_identity_slot_at(directory, directory->free_tail_slot)->next_free_slot = slot_index;
    }
    directory->free_tail_slot = slot_index;
    ++directory->free_slot_count;
}

//--------------------------------------
// Growth
//--------------------------------------

// Runs no callbacks.
static void segment_link(struct identity_directory* directory, uint32_t segment, void* address) {
    struct identity_slot* slots = address;
    if (segment == 0) {
        struct identity_head_segment* const head = address;
        head->segments[0] = head->slots;
        for (uint32_t index = 1; index < IDENTITY_SEGMENT_COUNT; ++index) {
            head->segments[index] = nullptr;
        }
        directory->segments = head->segments;
        slots = head->slots;
    } else {
        directory->segments[segment] = slots;
    }
    const uint32_t first_slot = segment_first_slot(segment);
    directory->slot_limit = segment_first_slot(segment + 1);
    for (uint32_t offset = 0; offset < segment_slot_count(segment); ++offset) {
        slots[offset] = (struct identity_slot){.generation = directory->generation_floor};
        if (first_slot + offset == 0) {
            slots[offset].generation = UINT32_MAX;
        } else {
            free_list_append(directory, first_slot + offset, &slots[offset]);
        }
    }
}

// Failure keeps previously linked segments.
static int directory_grow(struct pool_row* row, struct dora_pool pool, uint32_t usable_slot_count) {
    struct identity_directory* const directory = &row->directory;
    for (;;) {
        if ((uint64_t)directory->live_slot_count + directory->free_slot_count >= usable_slot_count) {
            return DORA_OK;
        }
        const uint32_t end = segment_of(directory->slot_limit);
        if (usable_slot_count > USABLE_SLOT_LIMIT || directory->generation_floor == UINT32_MAX ||
            end == IDENTITY_SEGMENT_COUNT) {
            return ENOMEM;
        }
        void* segment = nullptr;
        const int status = dora_memory_overhead_allocate(pool, segment_byte_count(end), &segment);
        if (status != DORA_OK) {
            return status;
        }
        if (segment_of(directory->slot_limit) == end) {
            segment_link(directory, end, segment);
        } else {
            // Allocator reentry may have linked this segment.
            dora_memory_overhead_release(pool, segment, segment_byte_count(end));
        }
    }
}

//--------------------------------------
// Occupy and Vacate
//--------------------------------------

int dora_identity_slot_occupy(struct dora_pool pool,
                              enum identity_kind kind,
                              struct identity_claim_state* claims,
                              struct dora_pool holder_pool,
                              struct dora_handle* out_owner) {
    struct pool_row* const row = &dora_pool_rows[pool.index];
    struct identity_directory* const directory = &row->directory;
    if (row->is_draining) {
        return EBUSY;
    }
    if (directory->generation_floor == UINT32_MAX) {
        return ENOMEM;
    }
    // Allocator callbacks may consume newly linked slots.
    while (directory->free_slot_count == 0) {
        const int status = directory_grow(row, pool, directory->live_slot_count + 1);
        if (status != DORA_OK) {
            return status;
        }
    }
    const uint32_t slot_index = directory->free_head_slot;
    struct identity_slot* const slot = dora_identity_slot_at(directory, slot_index);
    directory->free_head_slot = slot->next_free_slot;
    if (directory->free_head_slot == 0) {
        directory->free_tail_slot = 0;
    }
    --directory->free_slot_count;
    slot->claims = claims;
    slot->next_free_slot = 0;
    slot->kind = kind;
    slot->holder_pool_index = holder_pool.index;
    if (slot->generation < directory->generation_floor) {
        slot->generation = directory->generation_floor;
    }
    if (slot->generation > directory->generation_high_water) {
        directory->generation_high_water = slot->generation;
    }
    ++directory->live_slot_count;
    *out_owner = (struct dora_handle){
        .slot_index = slot_index,
        .generation = slot->generation,
        .pool_index = pool.index,
        .is_owner = true
    };
    return DORA_OK;
}

void dora_identity_slot_vacate(struct identity_directory* directory, uint32_t slot) {
    struct identity_slot* const vacated = dora_identity_slot_at(directory, slot);
    vacated->claims = nullptr;
    vacated->holder_pool_index = 0;
    --directory->live_slot_count;
    if (vacated->generation == UINT32_MAX - 1) {
        vacated->generation = UINT32_MAX;
        return;
    }
    ++vacated->generation;
    free_list_append(directory, slot, vacated);
}

//------------------------------------------------------------------------------
//
// Slot Lifecycle
//
//------------------------------------------------------------------------------

int dora_identity_resolve(const struct dora_handle* handle,
                          enum identity_kind expected_kind,
                          const struct identity_slot** out_slot) {
    if (handle == nullptr || out_slot == nullptr) {
        return EINVAL;
    }
    const struct identity_slot* const slot = resolve_slot(handle, expected_kind);
    if (slot == nullptr) {
        return EINVAL;
    }
    *out_slot = slot;
    return DORA_OK;
}

int dora_identity_slot_publish(struct dora_pool pool,
                               enum identity_kind kind,
                               struct identity_claim_state* claims,
                               struct dora_handle* out_owner) {
    if (pool_row_of(pool) == nullptr || kind > IDENTITY_KIND_SCHEMA_INFERENCE || claims == nullptr ||
        out_owner == nullptr) {
        return EINVAL;
    }
    const int status = dora_identity_slot_occupy(pool, kind, claims, (struct dora_pool){}, out_owner);
    if (status == DORA_OK) {
        claims->claim_count = 1;
    }
    return status;
}

int dora_identity_slot_enter(const struct dora_handle* handle,
                             enum identity_kind expected_kind,
                             struct identity_slot** out_slot) {
    if (handle == nullptr || out_slot == nullptr) {
        return EINVAL;
    }
    struct identity_slot* const slot = resolve_slot(handle, expected_kind);
    if (slot == nullptr) {
        return EINVAL;
    }
    struct pool_row* const row = &dora_pool_rows[handle->pool_index];
    if (slot->is_running || row->is_draining) {
        return EBUSY;
    }
    slot->is_running = true;
    ++row->work_depth;
    *out_slot = slot;
    return DORA_OK;
}

void dora_identity_slot_leave(const struct dora_handle* handle, struct identity_slot* slot) {
    slot->is_running = false;
    --dora_pool_rows[handle->pool_index].work_depth;
}

//------------------------------------------------------------------------------
//
// Directory Lifecycle
//
//------------------------------------------------------------------------------

void dora_identity_directory_close(struct dora_pool pool) {
    struct identity_directory* const directory = &dora_pool_rows[pool.index].directory;
    // Deallocation callbacks see an empty directory.
    struct identity_slot** const segments = directory->segments;
    const uint32_t segment_count = segment_of(directory->slot_limit);
    const uint32_t generation_high_water = directory->generation_high_water;
    *directory = (struct identity_directory){
        .generation_floor = generation_high_water + 1,
        .generation_high_water = generation_high_water
    };
    for (uint32_t segment = segment_count; segment-- > 1;) {
        dora_memory_overhead_release(pool, segments[segment], segment_byte_count(segment));
    }
    dora_memory_overhead_release(pool, (void*)segments, sizeof(struct identity_head_segment));
}

// The generation floor prevents pre-reset handles from resolving.
void dora_identity_directory_reset(struct dora_pool pool) {
    struct identity_directory* const directory = &dora_pool_rows[pool.index].directory;
    directory->generation_floor = directory->generation_high_water + 1;
    if (directory->generation_floor == UINT32_MAX) {
        directory->free_head_slot = 0;
        directory->free_tail_slot = 0;
        directory->free_slot_count = 0;
    }
}

int dora_identity_directory_reserve(struct dora_pool pool, uint32_t slot_count) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return EINVAL;
    }
    return directory_grow(row, pool, slot_count);
}
