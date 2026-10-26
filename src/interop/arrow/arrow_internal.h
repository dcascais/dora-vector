// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Arrow schema and array transfer.

#ifndef DORA_INTEROP_ARROW_ARROW_INTERNAL_H
#define DORA_INTEROP_ARROW_ARROW_INTERNAL_H

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Format and Buffers
//
//------------------------------------------------------------------------------

//--------------------------------------
// Formats
//--------------------------------------

static constexpr uint32_t ARROW_SIMPLE_FORMAT_COUNT = 38;

struct arrow_simple_format {
    struct dora_bytes format;
    enum dora_type type_kind;
    enum dora_layout layout_kind;
    uint8_t parameter;
};

extern const struct arrow_simple_format dora_arrow_simple_formats[ARROW_SIMPLE_FORMAT_COUNT];

//------------------------------------------------------------------------------
//
// Layout and Import Ownership
//
//------------------------------------------------------------------------------

struct arrow_adopted_array {
    struct dora_buffer_owner owner;
    struct dora_allocation self_storage;
    struct ArrowArray array;
    struct dora_arrow_device_access* access;
    uint32_t claim_count;
};

//------------------------------------------------------------------------------
//
// Traversal
//
//------------------------------------------------------------------------------

struct arrow_traversal_state {
    struct dora_allocation stack_storage;
    const void* inline_stack[16];
    struct dora_pool pool;
    uint32_t depth_capacity;
    uint64_t coordinate_work_count;
    uint64_t output_byte_count;
    uint32_t node_count;
    uint32_t edge_count;
};

void dora_arrow_traversal_init(struct dora_pool pool,
                               const struct dora_arrow_limits* limits,
                               struct arrow_traversal_state* state);
int dora_arrow_traversal_push(struct arrow_traversal_state* state, const void* source, uint32_t depth);
void dora_arrow_traversal_rewind(struct arrow_traversal_state* state);

//------------------------------------------------------------------------------
//
// Metadata Admission
//
//------------------------------------------------------------------------------

bool dora_arrow_is_extension_metadata_key(struct dora_bytes key);

//------------------------------------------------------------------------------
//
// Adapter Composition
//
//------------------------------------------------------------------------------

struct dora_arrow_limits dora_arrow_resolve_limits(const struct dora_arrow_limits* limits);

static inline int arrow_traversal_charge(const struct dora_arrow_limits* limits,
                                         struct arrow_traversal_state* state,
                                         uint64_t work,
                                         size_t output) {
    uint64_t work_total = 0;
    uint64_t output_total = 0;
    if (ckd_add(&work_total, state->coordinate_work_count, work) ||
        ckd_add(&output_total, state->output_byte_count, output)) {
        return EOVERFLOW;
    }
    if (work_total > limits->max_coordinate_work_count || output_total > limits->max_output_bytes) {
        return ERANGE;
    }
    state->coordinate_work_count = work_total;
    state->output_byte_count = output_total;
    return DORA_OK;
}

int dora_arrow_import_schema_impl(struct dora_pool pool,
                                  const struct ArrowSchema* source,
                                  const struct dora_arrow_limits* limits,
                                  struct arrow_traversal_state* state,
                                  struct dora_schema* out_schema,
                                  struct dora_error* out_error);
int dora_arrow_import_array_impl(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 struct arrow_adopted_array** out_adopted,
                                 const struct ArrowDeviceArray* device,
                                 struct dora_arrow_device_access* access,
                                 const struct dora_arrow_limits* limits,
                                 struct arrow_traversal_state* state,
                                 bool should_copy,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);
int dora_arrow_field_charge(const struct dora_arrow_limits* limits,
                            struct arrow_traversal_state* state,
                            const struct dora_schema_field* field);
int dora_arrow_device_view(const struct dora_vector* source, struct dora_vector_view* out_view);

int dora_arrow_export_array_in_pool(struct dora_pool pool,
                                    const struct dora_vector* source,
                                    struct ArrowArray* out_array);
int dora_arrow_export_device_array_in_pool(struct dora_pool pool,
                                           const struct dora_vector* source,
                                           struct ArrowDeviceArray* out_array);

#endif
