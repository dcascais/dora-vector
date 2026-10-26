// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector images and buffer ownership.

#ifndef DORA_VECTOR_IMAGE_H
#define DORA_VECTOR_IMAGE_H

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "types/types_internal.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Node State
//
//------------------------------------------------------------------------------

enum vector_content_state : uint8_t {
    VECTOR_CONTENT_PROMISED,
    VECTOR_CONTENT_VALIDATED,
    VECTOR_CONTENT_VALIDATED_ASCII
};

enum vector_statistics_provenance : uint8_t {
    VECTOR_STATISTICS_IMPORTED,
    VECTOR_STATISTICS_COMPUTED
};

struct vector_deferred_validity {
    struct dora_allocation allocation;
    struct dora_vector inputs[2];
    const uint64_t* selection;
};

struct vector_node_ownership {
    struct dora_vector_view view;
    struct dora_view_buffer own_validity;
    struct dora_allocation effective_validity;
    struct dora_allocation selection_storage;

    union {
        struct dora_allocation physical_rows;
        struct dora_vector selection_source;
    };
    struct vector_deferred_validity* deferred_validity;
    uint64_t own_validity_bit_offset;
    int64_t own_null_count;
    uint8_t union_members[UNION_MEMBER_CAPACITY];
    struct dora_vector backing;
    struct dora_vector storage;
    struct dora_vector statistics;
    const struct dora_extension_definition* extension;
    struct dora_allocation extension_parameters;
    enum vector_statistics_provenance statistics_provenance;
    bool requires_effective_validity;
    bool owns_physical_rows;
};

struct vector_node {
    uint64_t offset;
    enum vector_content_state content_state;
};

//------------------------------------------------------------------------------
//
// Image Geometry
//
//------------------------------------------------------------------------------

struct vector_table_span {
    uint32_t byte_offset;
    uint32_t entry_count;
};

//------------------------------------------------------------------------------
//
// Buffer Ownership
//
//------------------------------------------------------------------------------

enum vector_buffer_owner_kind : uint8_t {
    VECTOR_BUFFER_OWNER_ALLOCATION,
    VECTOR_BUFFER_OWNER_RETAINED
};

union vector_buffer_owner {
    struct dora_allocation allocation;
    struct dora_handle retained;
};

struct vector_buffer_row {
    union vector_buffer_owner owner;
    enum vector_buffer_owner_kind owner_kind;
};

//------------------------------------------------------------------------------
//
// Vector Image
//
//------------------------------------------------------------------------------

struct vector_image {
    struct dora_schema schema;
    struct dora_allocation allocation;
    struct identity_claim_state claims;
    struct vector_node node;
    struct vector_node_ownership ownership;
    struct vector_table_span buffers;
    struct vector_table_span child_vectors;
    uint32_t view_buffers_byte_offset;
    struct vector_image* next_retired;
    struct dora_buffer_owner* wrapped_owner;
    struct vector_image* buffer_image;
    alignas(8) unsigned char table_bytes[];
};

//------------------------------------------------------------------------------
//
// Bound Inputs
//
//------------------------------------------------------------------------------

struct vector_bound_node {
    const struct vector_image* image;
    struct dora_schema_field field;
    struct dora_vector_view view;
    const struct dora_extension_definition* extension;
    const struct dora_allocation* parameters;
    uint32_t subtree_end_node_index;
    // Borrowed for settlement and scalar rebinding.
    const struct dora_vector* source;
};

struct vector_bound_input {
    struct vector_bound_node* nodes;
    uint32_t node_count;
    uint64_t length;
    bool is_constant;
};

struct vector_input {
    const struct dora_schema_field* field;
    const struct dora_schema* admitted_schema;
    const struct dora_schema* storage_schema;
    // Optional borrowed source for claims on nonempty buffers without allocations.
    const struct dora_vector* buffer_source;
    uint64_t row_count;
    int64_t null_count;
    uint64_t offset;
    uint64_t validity_bit_offset;
    const struct dora_bytes* buffers;
    const struct dora_allocation* allocations;
    const int64_t* data_buffer_sizes;
    uint32_t buffer_count;
    const struct dora_vector* children;
    const struct vector_bound_input* foreign;
    uint32_t foreign_node_index;
    const char* const* child_names;
    uint32_t child_count;
    const struct dora_bytes* ancestor_activity;
    uint64_t activity_bit_offset;
    struct dora_buffer_owner* owner;
    enum vector_content_state content_state;
    bool should_scan_geometry;
    bool are_required_rows_admitted;
    bool are_map_rows_admitted;
    bool should_check_child_activity;
    // Children with bitmaps must include ancestor nulls.
    bool are_children_folded;
    struct dora_allocation* effective_validity;
    int64_t effective_null_count;
};

static inline bool vector_layout_derives_nulls(enum dora_layout layout) {
    return layout == DORA_LAYOUT_DICTIONARY || layout == DORA_LAYOUT_DENSE_UNION ||
           layout == DORA_LAYOUT_SPARSE_UNION || layout == DORA_LAYOUT_RUN_END;
}

#endif
