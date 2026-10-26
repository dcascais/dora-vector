// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Row-key deduplication.

#ifndef DORA_HASH_KEYSET_H
#define DORA_HASH_KEYSET_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Row Key Set
//
//------------------------------------------------------------------------------

static constexpr uint64_t KEYSET_PAIRWISE_LIMIT = 16;

struct dora_keyset_slot {
    uint64_t hash;
    uint64_t row;
    uint64_t value;
};

// Equal rows, including nulls, must have equal hashes.
struct dora_keyset_rows {
    void* context;
    int (*hash)(void* context, uint64_t row, uint64_t* out_hash);
    int (*equal)(void* context, uint64_t left_row, uint64_t right_row, bool* out_equal);
};

struct dora_keyset {
    struct dora_allocation storage;
    struct dora_keyset_slot inline_slots[KEYSET_PAIRWISE_LIMIT];
    size_t slot_capacity;
    uint64_t key_count;
};

int dora_keyset_init(struct dora_keyset* set, struct dora_pool pool, uint64_t expected_key_count);

// Duplicate rows return the existing value.
int dora_keyset_insert(struct dora_keyset* set,
                       const struct dora_keyset_rows* rows,
                       uint64_t row,
                       uint64_t value,
                       uint64_t* out_existing_value,
                       bool* out_is_duplicate);

// Retains storage.
static inline void keyset_clear(struct dora_keyset* set) {
    if (set->slot_capacity != 0) {
        memset(set->storage.address, UINT8_MAX, set->slot_capacity * sizeof(struct dora_keyset_slot));
    }
    set->key_count = 0;
}

void dora_keyset_release(struct dora_keyset* set);

#endif
