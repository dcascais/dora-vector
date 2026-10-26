// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Schema construction and navigation.

#ifndef DORA_VECTOR_SCHEMA_H
#define DORA_VECTOR_SCHEMA_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Field Declarations
//
//------------------------------------------------------------------------------

enum dora_schema_field_flag : uint8_t {
    DORA_SCHEMA_FIELD_FLAG_NONNULLABLE = 1,
    DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED = 2,
    DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED = 4
};

struct dora_metadata_entry {
    /// Metadata key bytes.
    struct dora_bytes key;
    /// Metadata value bytes.
    struct dora_bytes value;
};

struct dora_schema_field {
    /// Borrowed NUL-terminated field name.
    const char* name;
    /// Logical type and parameters.
    struct dora_logical_type type;
    /// Storage layout.
    enum dora_layout layout;
    /// Dictionary index type.
    enum dora_type index_type;
    /// dora_schema_field_flag bitmask.
    uint8_t flags;
    /// Immediate child count.
    uint32_t child_count;
    /// Borrowed metadata entries.
    const struct dora_metadata_entry* metadata;
    /// Metadata entry count.
    uint32_t metadata_entry_count;
};

struct dora_schema_input {
    /// Borrowed fields in preorder.
    const struct dora_schema_field* fields;
    /// Field count.
    uint32_t field_count;
    /// Borrowed schema metadata.
    const struct dora_metadata_entry* metadata;
    /// Schema metadata entry count.
    uint32_t metadata_entry_count;
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

int dora_schema_init_copy(struct dora_schema* schema, struct dora_pool pool, const struct dora_schema_input* input);

/// Requires nonempty children in one pool; allocates in that pool.
int dora_schema_init_retain_struct(struct dora_schema* schema,
                                   const struct dora_schema* children,
                                   uint32_t child_count);

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

int dora_schema_retain(const struct dora_schema* schema, struct dora_schema* out_schema);

/// Shares the declaration into pool; children stay claimed in their original pools.
int dora_schema_share(const struct dora_schema* schema, struct dora_pool pool, struct dora_schema* out_shared);

static inline struct dora_schema dora_schema_borrow(const struct dora_schema* schema) {
    if (schema == nullptr) {
        return (struct dora_schema){};
    }
    struct dora_schema borrow = *schema;
    borrow.handle.is_owner = false;
    return borrow;
}

int dora_schema_move(struct dora_schema* in_out_source, struct dora_schema* out_destination);

void dora_schema_release(struct dora_schema* schema);

bool dora_schema_is_valid(const struct dora_schema* schema);

int dora_schema_get_pool(const struct dora_schema* schema, struct dora_pool* out_pool);

//------------------------------------------------------------------------------
//
// Navigation
//
//------------------------------------------------------------------------------

int dora_schema_get_field(const struct dora_schema* schema, struct dora_schema_field* out_field);

int dora_schema_get_child_count(const struct dora_schema* schema, uint32_t* out_child_count);

int dora_schema_get_child(const struct dora_schema* schema, uint32_t child_index, struct dora_schema* out_child);

int dora_schema_find_child(const struct dora_schema* schema, const char* name, struct dora_schema* out_child);

#endif
