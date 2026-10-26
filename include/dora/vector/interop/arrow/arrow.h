// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Arrow import and export.

#ifndef DORA_VECTOR_INTEROP_ARROW_ARROW_H
#define DORA_VECTOR_INTEROP_ARROW_ARROW_H

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/status.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Traversal and Device Access
//
//------------------------------------------------------------------------------

struct dora_arrow_limits {
    /// Maximum schema nesting depth.
    uint32_t max_depth;
    /// Maximum traversed nodes.
    uint32_t max_node_count;
    /// Maximum traversed edges.
    uint32_t max_edge_count;
    /// Maximum coordinate traversal work.
    uint64_t max_coordinate_work_count;
    /// Maximum output bytes.
    size_t max_output_bytes;
};

struct dora_arrow_device_access {
    /// Returns array placement and readiness.
    int (*describe)(struct dora_arrow_device_access* access,
                    const struct ArrowDeviceArray* array,
                    struct dora_placement* out_placement,
                    struct dora_readiness* out_readiness);
    /// Copies source bytes into host memory.
    int (*read_host)(struct dora_arrow_device_access* access,
                     struct dora_placement placement,
                     struct dora_device_region source,
                     uint8_t* out_bytes);
};

//------------------------------------------------------------------------------
//
// Import
//
//------------------------------------------------------------------------------

/// Copies schema declarations without inspecting array content.
int dora_arrow_import_schema(struct dora_pool pool,
                             const struct ArrowSchema* source,
                             const struct dora_arrow_limits* limits,
                             struct dora_schema* out_schema,
                             struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_copy_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_move_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 struct ArrowArray* in_out_source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_wrap_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_copy_device_array(struct dora_pool pool,
                                        const struct dora_schema* schema,
                                        const struct ArrowDeviceArray* source,
                                        struct dora_arrow_device_access* access,
                                        const struct dora_arrow_limits* limits,
                                        struct dora_vector* out_vector,
                                        struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_move_device_array(struct dora_pool pool,
                                        const struct dora_schema* schema,
                                        struct ArrowDeviceArray* in_out_source,
                                        struct dora_arrow_device_access* access,
                                        const struct dora_arrow_limits* limits,
                                        struct dora_vector* out_vector,
                                        struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_stream(struct dora_pool pool,
                             struct ArrowArrayStream* in_out_stream,
                             const struct dora_arrow_limits* limits,
                             struct dora_reader* out_reader,
                             struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; admits other content unchecked.
int dora_arrow_import_device_stream(struct dora_pool pool,
                                    struct ArrowDeviceArrayStream* in_out_stream,
                                    struct dora_arrow_device_access* access,
                                    const struct dora_arrow_limits* limits,
                                    struct dora_reader* out_reader,
                                    struct dora_error* out_error);

/// Checks layout, nullability and unique nonnull Map keys; allocates in the source pool.
int dora_arrow_import_statistics(const struct dora_vector* source,
                                 const struct ArrowSchema* statistics_schema,
                                 const struct ArrowArray* statistics_array,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Export
//
//------------------------------------------------------------------------------

int dora_arrow_export_schema(struct dora_pool pool, const struct dora_schema* source, struct ArrowSchema* out_schema);

/// Retains source pool claims until out_array release; exports stored values.
int dora_arrow_export_array(const struct dora_vector* source, struct ArrowArray* out_array);

/// Retains source pool claims until out_array release; exports stored values.
int dora_arrow_export_device_array(const struct dora_vector* source, struct ArrowDeviceArray* out_array);

/// Retains reader pool claims until out_stream release; exports stored values.
int dora_arrow_export_stream(const struct dora_schema* schema,
                             struct dora_reader* in_out_reader,
                             struct ArrowArrayStream* out_stream);

/// Retains reader pool claims until out_stream release; exports stored values.
int dora_arrow_export_device_stream(const struct dora_schema* schema,
                                    struct dora_reader* in_out_reader,
                                    struct ArrowDeviceArrayStream* out_stream);

int dora_arrow_export_statistics(struct dora_pool pool,
                                 const struct dora_vector* source,
                                 struct ArrowSchema* out_schema,
                                 struct ArrowArray* out_array);

#endif
