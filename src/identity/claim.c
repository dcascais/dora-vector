// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"

#include <errno.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Claim Lifecycle
//
//------------------------------------------------------------------------------

static bool is_cross_pool_holder(uint16_t holder_pool_index, uint16_t data_pool_index) {
    return holder_pool_index != 0 && holder_pool_index != data_pool_index;
}

int dora_identity_claim_acquire(struct dora_pool holder_pool,
                                const struct dora_handle* source,
                                enum identity_kind expected_kind,
                                struct dora_handle* out_owner) {
    const struct identity_slot* source_slot = nullptr;
    if (source == nullptr || out_owner == nullptr ||
        dora_identity_resolve(source, expected_kind, &source_slot) != DORA_OK) {
        return EINVAL;
    }
    struct pool_row* const row = &dora_pool_rows[source->pool_index];
    struct identity_directory* const directory = &row->directory;
    struct identity_claim_state* const claims = source_slot->claims;
    const bool is_cross_pool = is_cross_pool_holder(holder_pool.index, source->pool_index);
    const struct dora_pool pool = {.index = source->pool_index, .generation = row->generation};
    // Source allocation may reenter the holder pool.
    if (is_cross_pool) {
        dora_pool_callback_enter(holder_pool);
    }
    const int status = dora_identity_slot_occupy(pool, expected_kind, claims, holder_pool, out_owner);
    if (is_cross_pool) {
        dora_pool_callback_leave(holder_pool);
    }
    if (status != DORA_OK) {
        return status;
    }
    ++claims->claim_count;
    if (is_cross_pool) {
        ++directory->cross_pool_claim_count;
    }
    return DORA_OK;
}

int dora_identity_claim_transfer(struct dora_pool holder_pool,
                                 const struct dora_handle* owner,
                                 enum identity_kind expected_kind) {
    const struct identity_slot* resolved = nullptr;
    if (owner == nullptr || !owner->is_owner || dora_identity_resolve(owner, expected_kind, &resolved) != DORA_OK) {
        return EINVAL;
    }
    struct identity_directory* const directory = &dora_pool_rows[owner->pool_index].directory;
    struct identity_slot* const slot = dora_identity_slot_at(directory, owner->slot_index);
    const bool is_cross_pool = is_cross_pool_holder(holder_pool.index, owner->pool_index);
    const bool was_cross_pool = is_cross_pool_holder(slot->holder_pool_index, owner->pool_index);
    if (was_cross_pool != is_cross_pool) {
        if (is_cross_pool) {
            ++directory->cross_pool_claim_count;
        } else {
            --directory->cross_pool_claim_count;
        }
    }
    slot->holder_pool_index = holder_pool.index;
    return DORA_OK;
}

void dora_identity_claim_release(struct dora_handle* in_out_owner,
                                 enum identity_kind expected_kind,
                                 void** out_retired_payload) {
    *out_retired_payload = nullptr;
    if (in_out_owner == nullptr) {
        return;
    }
    const struct dora_handle owner = *in_out_owner;
    *in_out_owner = (struct dora_handle){};
    const struct identity_slot* resolved = nullptr;
    if (!owner.is_owner || dora_identity_resolve(&owner, expected_kind, &resolved) != DORA_OK) {
        return;
    }
    struct identity_directory* const directory = &dora_pool_rows[owner.pool_index].directory;
    struct identity_claim_state* const claims = resolved->claims;
    if (is_cross_pool_holder(resolved->holder_pool_index, owner.pool_index)) {
        --directory->cross_pool_claim_count;
    }
    --claims->claim_count;
    if (claims->claim_count == 0) {
        *out_retired_payload = claims->payload;
    }
    dora_identity_slot_vacate(directory, owner.slot_index);
}
