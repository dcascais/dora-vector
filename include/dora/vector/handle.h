// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Pool and object identities.

#ifndef DORA_VECTOR_HANDLE_H
#define DORA_VECTOR_HANDLE_H

#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Handles
//
//------------------------------------------------------------------------------

struct dora_handle {
    /// Registry slot.
    uint32_t slot_index;
    /// Slot generation.
    uint32_t generation;
    /// Owning pool index.
    uint16_t pool_index;
    /// Whether release drops an owning claim.
    bool is_owner;
    /// Reserved; zero.
    uint8_t reserved;
};

static inline bool dora_handle_is_same(struct dora_handle left, struct dora_handle right) {
    return left.slot_index == right.slot_index && left.generation == right.generation &&
           left.pool_index == right.pool_index;
}

struct dora_schema {
    /// Object identity.
    struct dora_handle handle;
    /// Selected schema node.
    uint32_t node_index;
};

struct dora_vector {
    /// Object identity.
    struct dora_handle handle;
    /// Selected row range.
    struct dora_range rows;
};

struct dora_builder {
    /// Object identity.
    struct dora_handle handle;
    /// Selected builder node.
    uint32_t node_index;
};

struct dora_reader {
    /// Object identity.
    struct dora_handle handle;
};

struct dora_writer {
    /// Object identity.
    struct dora_handle handle;
};

struct dora_pipeline {
    /// Object identity.
    struct dora_handle handle;
};

struct dora_pool {
    /// Registry index.
    uint16_t index;
    /// Pool generation.
    uint16_t generation;
};

struct dora_schema_inference {
    /// Object identity.
    struct dora_handle handle;
};

#endif
