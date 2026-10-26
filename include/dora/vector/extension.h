// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Extension registration and callbacks.

#ifndef DORA_VECTOR_EXTENSION_H
#define DORA_VECTOR_EXTENSION_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Extension Capabilities
//
//------------------------------------------------------------------------------

enum dora_extension_flag : uint32_t {
    DORA_EXTENSION_FLAG_STORAGE_EQUALITY = 1,
    DORA_EXTENSION_FLAG_STORAGE_ORDER = 2
};

/// Synchronous callbacks must preserve active arguments; pool reset and release return EBUSY.
struct dora_extension_definition {
    /// Borrowed extension name.
    struct dora_bytes name;
    /// dora_extension_flag bitmask.
    uint32_t flags;
    /// Binds borrowed storage and metadata; returns empty or pool-owned parameters.
    int (*bind)(const struct dora_extension_definition* definition,
                struct dora_pool pool,
                const struct dora_schema* storage,
                struct dora_bytes metadata,
                struct dora_allocation* out_parameters,
                struct dora_error* out_error);
    /// Checks borrowed storage using bound parameters; may clear row validity.
    int (*validate)(const struct dora_extension_definition* definition,
                    struct dora_pool pool,
                    const struct dora_allocation* parameters,
                    const struct dora_vector* storage,
                    uint64_t* in_out_validity,
                    struct dora_error* out_error);
    /// Casts borrowed source; checks Utf8 results under policy.
    int (*cast)(const struct dora_extension_definition* definition,
                struct dora_pool pool,
                const struct dora_allocation* parameters,
                const struct dora_vector* source,
                const struct dora_logical_type* target,
                enum dora_rounding_mode rounding_mode,
                enum dora_failure_policy policy,
                struct dora_vector* out_result);
};

//------------------------------------------------------------------------------
//
// Registration
//
//------------------------------------------------------------------------------

/// Borrows definition and name until pool release; registrations survive reset.
int dora_extension_register(const struct dora_extension_definition* definition, struct dora_pool pool);

//------------------------------------------------------------------------------
//
// Canonical Extensions
//
//------------------------------------------------------------------------------

//--------------------------------------
// Bool8
//--------------------------------------

extern const struct dora_extension_definition dora_extension_bool8;

//--------------------------------------
// JSON
//--------------------------------------

extern const struct dora_extension_definition dora_extension_json;

//--------------------------------------
// Opaque
//--------------------------------------

extern const struct dora_extension_definition dora_extension_opaque;

//--------------------------------------
// Tensor
//--------------------------------------

extern const struct dora_extension_definition dora_extension_fixed_shape_tensor;

extern const struct dora_extension_definition dora_extension_variable_shape_tensor;

//--------------------------------------
// Timestamp With Offset
//--------------------------------------

extern const struct dora_extension_definition dora_extension_timestamp_with_offset;

//--------------------------------------
// UUID
//--------------------------------------

extern const struct dora_extension_definition dora_extension_uuid;

//--------------------------------------
// Variant
//--------------------------------------

extern const struct dora_extension_definition dora_extension_parquet_variant;

#endif
