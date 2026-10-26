// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "hash/keyset.h"

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

static constexpr uint64_t KEYSET_EMPTY_ROW = UINT64_MAX;

int dora_keyset_init(struct dora_keyset* set, struct dora_pool pool, uint64_t expected_key_count) {
    if (set == nullptr || expected_key_count == UINT64_MAX) {
        return EINVAL;
    }
    *set = (struct dora_keyset){};
    if (expected_key_count <= KEYSET_PAIRWISE_LIMIT) {
        return DORA_OK;
    }
    size_t capacity = 1;
    uint64_t minimum = 0;
    if (ckd_mul(&minimum, expected_key_count, UINT64_C(2))) {
        return EOVERFLOW;
    }
    while (capacity < minimum) {
        if (ckd_mul(&capacity, capacity, 2)) {
            return EOVERFLOW;
        }
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, capacity, sizeof(struct dora_keyset_slot))) {
        return EOVERFLOW;
    }
    const int status = dora_memory_payload_allocate(pool, bytes, &set->storage);
    if (status != DORA_OK) {
        return status;
    }
    memset(set->storage.address, UINT8_MAX, bytes);
    set->slot_capacity = capacity;
    return DORA_OK;
}

static int keyset_insert_pairwise(struct dora_keyset* set,
                                  const struct dora_keyset_rows* rows,
                                  uint64_t row,
                                  uint64_t value,
                                  uint64_t* out_existing_value,
                                  bool* out_is_duplicate) {
    for (uint64_t index = 0; index < set->key_count; ++index) {
        bool equal = false;
        const int status = rows->equal(rows->context, set->inline_slots[index].row, row, &equal);
        if (status != DORA_OK) {
            return status;
        }
        if (equal) {
            *out_existing_value = set->inline_slots[index].value;
            *out_is_duplicate = true;
            return DORA_OK;
        }
    }
    if (set->key_count == KEYSET_PAIRWISE_LIMIT) {
        return ENOSPC;
    }
    set->inline_slots[set->key_count++] = (struct dora_keyset_slot){.row = row, .value = value};
    *out_is_duplicate = false;
    return DORA_OK;
}

int dora_keyset_insert(struct dora_keyset* set,
                       const struct dora_keyset_rows* rows,
                       uint64_t row,
                       uint64_t value,
                       uint64_t* out_existing_value,
                       bool* out_is_duplicate) {
    if (set == nullptr || rows == nullptr || rows->equal == nullptr || out_existing_value == nullptr ||
        out_is_duplicate == nullptr || row == KEYSET_EMPTY_ROW) {
        return EINVAL;
    }
    if (set->slot_capacity == 0) {
        return keyset_insert_pairwise(set, rows, row, value, out_existing_value, out_is_duplicate);
    }
    if (rows->hash == nullptr || set->key_count >= set->slot_capacity / 2) {
        return rows->hash == nullptr ? EINVAL : ENOSPC;
    }
    uint64_t hash = 0;
    int status = rows->hash(rows->context, row, &hash);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_keyset_slot* const slots = set->storage.address;
    size_t index = (size_t)hash & (set->slot_capacity - 1);
    while (slots[index].row != KEYSET_EMPTY_ROW) {
        if (slots[index].hash == hash) {
            bool equal = false;
            status = rows->equal(rows->context, slots[index].row, row, &equal);
            if (status != DORA_OK) {
                return status;
            }
            if (equal) {
                *out_existing_value = slots[index].value;
                *out_is_duplicate = true;
                return DORA_OK;
            }
        }
        index = (index + 1) & (set->slot_capacity - 1);
    }
    slots[index] = (struct dora_keyset_slot){.hash = hash, .row = row, .value = value};
    ++set->key_count;
    *out_is_duplicate = false;
    return DORA_OK;
}

void dora_keyset_release(struct dora_keyset* set) {
    if (set == nullptr) {
        return;
    }
    dora_allocation_release(&set->storage);
    *set = (struct dora_keyset){};
}
