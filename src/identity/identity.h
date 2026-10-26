// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Handle claims and owner slots.

#ifndef DORA_IDENTITY_IDENTITY_H
#define DORA_IDENTITY_IDENTITY_H

#include "dora/vector/handle.h"

#include <stdint.h>

static inline bool handle_is_empty(struct dora_handle handle) {
    return handle.slot_index == 0 && handle.generation == 0 && handle.pool_index == 0 && !handle.is_owner &&
           handle.reserved == 0;
}

//------------------------------------------------------------------------------
//
// Claims and Slots
//
//------------------------------------------------------------------------------

enum identity_kind : uint8_t {
    IDENTITY_KIND_SCHEMA,
    IDENTITY_KIND_VECTOR,
    IDENTITY_KIND_BUILDER,
    IDENTITY_KIND_READER,
    IDENTITY_KIND_WRITER,
    IDENTITY_KIND_PIPELINE,
    IDENTITY_KIND_SCHEMA_INFERENCE
};

//--------------------------------------
// Data Claims
//--------------------------------------

struct identity_claim_state {
    void* payload;
    uint32_t claim_count;
};

//--------------------------------------
// Owner Slots
//--------------------------------------

struct identity_slot {
    // nullptr marks an inactive slot.
    struct identity_claim_state* claims;
    // UINT32_MAX marks a retired slot.
    uint32_t generation;
    uint32_t next_free_slot;
    // Zero denotes the caller.
    uint16_t holder_pool_index;
    enum identity_kind kind;
    bool is_running;
};

//------------------------------------------------------------------------------
//
// Directory Storage
//
//------------------------------------------------------------------------------

static constexpr uint32_t IDENTITY_SEGMENT_FIRST_SLOT_COUNT = 64;
static constexpr uint32_t IDENTITY_SEGMENT_COUNT = 26;

//--------------------------------------
// Segments
//--------------------------------------

// Segment k > 0 holds 64 << k slots.
struct identity_head_segment {
    // segments[0] aliases slots.
    struct identity_slot* segments[IDENTITY_SEGMENT_COUNT];
    struct identity_slot slots[IDENTITY_SEGMENT_FIRST_SLOT_COUNT];
};

//--------------------------------------
// Generation History
//--------------------------------------

struct identity_directory {
    // nullptr until the first growth.
    struct identity_slot** segments;
    // End of linked segments; no allocation remains unlinked.
    uint32_t slot_limit;
    uint32_t free_head_slot;
    uint32_t free_tail_slot;
    uint32_t free_slot_count;
    uint32_t live_slot_count;
    uint32_t cross_pool_claim_count;
    uint32_t generation_floor;
    uint32_t generation_high_water;
};

//------------------------------------------------------------------------------
//
// Slot Lifecycle
//
//------------------------------------------------------------------------------

int dora_identity_resolve(const struct dora_handle* handle,
                          enum identity_kind expected_kind,
                          const struct identity_slot** out_slot);

int dora_identity_slot_publish(struct dora_pool pool,
                               enum identity_kind kind,
                               struct identity_claim_state* claims,
                               struct dora_handle* out_owner);

int dora_identity_slot_enter(const struct dora_handle* handle,
                             enum identity_kind expected_kind,
                             struct identity_slot** out_slot);

void dora_identity_slot_leave(const struct dora_handle* handle, struct identity_slot* slot);

//------------------------------------------------------------------------------
//
// Claim Lifecycle
//
//------------------------------------------------------------------------------

int dora_identity_claim_acquire(struct dora_pool holder_pool,
                                const struct dora_handle* source,
                                enum identity_kind expected_kind,
                                struct dora_handle* out_owner);

int dora_identity_claim_transfer(struct dora_pool holder_pool,
                                 const struct dora_handle* owner,
                                 enum identity_kind expected_kind);

void dora_identity_claim_release(struct dora_handle* in_out_owner,
                                 enum identity_kind expected_kind,
                                 void** out_retired_payload);

//------------------------------------------------------------------------------
//
// Directory Lifecycle
//
//------------------------------------------------------------------------------

void dora_identity_directory_close(struct dora_pool pool);

void dora_identity_directory_reset(struct dora_pool pool);

int dora_identity_directory_reserve(struct dora_pool pool, uint32_t slot_count);

//------------------------------------------------------------------------------
//
// Directory Lookup
//
//------------------------------------------------------------------------------

struct identity_slot* dora_identity_slot_at(struct identity_directory* directory, uint32_t slot);

//------------------------------------------------------------------------------
//
// Slot Occupancy
//
//------------------------------------------------------------------------------

int dora_identity_slot_occupy(struct dora_pool pool,
                              enum identity_kind kind,
                              struct identity_claim_state* claims,
                              struct dora_pool holder_pool,
                              struct dora_handle* out_owner);

void dora_identity_slot_vacate(struct identity_directory* directory, uint32_t slot);

#endif
