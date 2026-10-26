// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Incremental vector construction.

#ifndef DORA_VECTOR_BUILDER_H
#define DORA_VECTOR_BUILDER_H

#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Builder State
//
//------------------------------------------------------------------------------

struct dora_builder_options {
    /// Maximum rows; zero adds no limit.
    uint64_t row_capacity;
    /// Storage byte limit.
    size_t limit_bytes;
    /// Borrowed layouts in schema-node order.
    const enum dora_layout* node_layouts;
    /// Layout count.
    uint32_t node_layout_count;
    /// Content failure policy: ERROR, NULL or TRUST.
    enum dora_failure_policy content_policy;
    /// Borrowed statistics requests.
    const struct dora_statistics_request* statistics_requests;
    /// Statistics request count.
    uint32_t statistics_request_count;
};

struct dora_builder_metrics {
    /// Committed rows.
    uint64_t row_count;
    /// Reserved storage bytes.
    size_t capacity_bytes;
    /// Initialized storage bytes.
    size_t written_byte_count;
    /// Null rows.
    int64_t null_count;
};

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

int dora_builder_init_type(struct dora_builder* builder,
                           struct dora_pool pool,
                           const struct dora_logical_type* type,
                           const struct dora_builder_options* options);

int dora_builder_init_fields(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema_input* input,
                             const struct dora_builder_options* options);

int dora_builder_init_schema(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema* schema,
                             const struct dora_builder_options* options);

//------------------------------------------------------------------------------
//
// Navigation and Setup
//
//------------------------------------------------------------------------------

int dora_builder_get_child(const struct dora_builder* builder, uint32_t child_index, struct dora_builder* out_child);

int dora_builder_find_child(const struct dora_builder* builder, const char* name, struct dora_builder* out_child);

int dora_builder_retain_schema(const struct dora_builder* builder, struct dora_schema* out_schema);

/// Commits pending rows; failure preserves out_metrics and discards failed pending rows.
int dora_builder_get_metrics(const struct dora_builder* builder, struct dora_builder_metrics* out_metrics);

/// Retains dictionary content without rechecking it.
int dora_builder_set_dictionary(struct dora_builder* builder, const struct dora_vector* values);

int dora_builder_reserve(struct dora_builder* builder, uint64_t row_count, size_t byte_count);

//------------------------------------------------------------------------------
//
// Leaf Rows
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_builder_append_bool(struct dora_builder* builder, bool value);

int dora_builder_append_packed_bool(struct dora_builder* builder,
                                    const uint8_t* bits,
                                    uint64_t bit_offset,
                                    const uint8_t* validity,
                                    uint64_t validity_bit_offset,
                                    uint64_t row_count);

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_builder_append_i8(struct dora_builder* builder, int8_t value);

int dora_builder_append_i16(struct dora_builder* builder, int16_t value);

/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth
int dora_builder_append_i32(struct dora_builder* builder, int32_t value);

/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration
int dora_builder_append_i64(struct dora_builder* builder, int64_t value);

/// Int128, Decimal (128-bit)
int dora_builder_append_i128(struct dora_builder* builder, struct dora_i128 value);

/// Int256, Decimal (256-bit)
int dora_builder_append_i256(struct dora_builder* builder, struct dora_i256 value);

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_builder_append_u8(struct dora_builder* builder, uint8_t value);

int dora_builder_append_u16(struct dora_builder* builder, uint16_t value);

int dora_builder_append_u32(struct dora_builder* builder, uint32_t value);

int dora_builder_append_u64(struct dora_builder* builder, uint64_t value);

int dora_builder_append_u128(struct dora_builder* builder, struct dora_u128 value);

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_builder_append_f16(struct dora_builder* builder, struct dora_f16_bits value);

int dora_builder_append_f32(struct dora_builder* builder, float value);

int dora_builder_append_f64(struct dora_builder* builder, double value);

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_builder_append_interval_day_time(struct dora_builder* builder, struct dora_interval_day_time value);

int dora_builder_append_interval_month_day_nano(struct dora_builder* builder,
                                                struct dora_interval_month_day_nano value);

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
int dora_builder_append_bytes(struct dora_builder* builder, struct dora_bytes bytes);

int dora_builder_append_text(struct dora_builder* builder, const char* text, size_t byte_count);

//------------------------------------------------------------------------------
//
// Leaf Values
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_builder_append_values_bool(struct dora_builder* builder,
                                    const bool* values,
                                    const bool* validity,
                                    uint64_t row_count);

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_builder_append_values_i8(struct dora_builder* builder,
                                  const int8_t* values,
                                  const bool* validity,
                                  uint64_t row_count);

int dora_builder_append_values_i16(struct dora_builder* builder,
                                   const int16_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth
int dora_builder_append_values_i32(struct dora_builder* builder,
                                   const int32_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration
int dora_builder_append_values_i64(struct dora_builder* builder,
                                   const int64_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

/// Int128, Decimal (128-bit)
int dora_builder_append_values_i128(struct dora_builder* builder,
                                    const struct dora_i128* values,
                                    const bool* validity,
                                    uint64_t row_count);

/// Int256, Decimal (256-bit)
int dora_builder_append_values_i256(struct dora_builder* builder,
                                    const struct dora_i256* values,
                                    const bool* validity,
                                    uint64_t row_count);

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_builder_append_values_u8(struct dora_builder* builder,
                                  const uint8_t* values,
                                  const bool* validity,
                                  uint64_t row_count);

int dora_builder_append_values_u16(struct dora_builder* builder,
                                   const uint16_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

int dora_builder_append_values_u32(struct dora_builder* builder,
                                   const uint32_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

int dora_builder_append_values_u64(struct dora_builder* builder,
                                   const uint64_t* values,
                                   const bool* validity,
                                   uint64_t row_count);

int dora_builder_append_values_u128(struct dora_builder* builder,
                                    const struct dora_u128* values,
                                    const bool* validity,
                                    uint64_t row_count);

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_builder_append_values_f16(struct dora_builder* builder,
                                   const struct dora_f16_bits* values,
                                   const bool* validity,
                                   uint64_t row_count);

int dora_builder_append_values_f32(struct dora_builder* builder,
                                   const float* values,
                                   const bool* validity,
                                   uint64_t row_count);

int dora_builder_append_values_f64(struct dora_builder* builder,
                                   const double* values,
                                   const bool* validity,
                                   uint64_t row_count);

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_builder_append_values_interval_day_time(struct dora_builder* builder,
                                                 const struct dora_interval_day_time* values,
                                                 const bool* validity,
                                                 uint64_t row_count);

int dora_builder_append_values_interval_month_day_nano(struct dora_builder* builder,
                                                       const struct dora_interval_month_day_nano* values,
                                                       const bool* validity,
                                                       uint64_t row_count);

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
int dora_builder_append_values_bytes(struct dora_builder* builder,
                                     const struct dora_bytes* values,
                                     const bool* validity,
                                     uint64_t row_count);

//------------------------------------------------------------------------------
//
// Nulls and Vectors
//
//------------------------------------------------------------------------------

int dora_builder_append_null(struct dora_builder* builder);

int dora_builder_append_nulls(struct dora_builder* builder, uint64_t row_count);

/// Appends source rows without rechecking their content.
int dora_builder_append_vector(struct dora_builder* builder, const struct dora_vector* source);

//------------------------------------------------------------------------------
//
// Nested Rows
//
//------------------------------------------------------------------------------

//--------------------------------------
// Lists
//--------------------------------------

/// List; layouts: List, LargeList
int dora_builder_append_list(struct dora_builder* builder, uint64_t element_count);

/// List; layouts: ListView, LargeListView
int dora_builder_append_list_view(struct dora_builder* builder, struct dora_range elements);

int dora_builder_append_fixed_list(struct dora_builder* builder);

//--------------------------------------
// Structures
//--------------------------------------

int dora_builder_append_struct(struct dora_builder* builder);

//--------------------------------------
// Maps
//--------------------------------------

int dora_builder_append_map(struct dora_builder* builder, uint64_t entry_count);

//--------------------------------------
// Unions
//--------------------------------------

/// Union; layouts: DenseUnion, SparseUnion
int dora_builder_append_union(struct dora_builder* builder, int8_t type_id);

//------------------------------------------------------------------------------
//
// Direct Fill
//
//------------------------------------------------------------------------------

/// Borrows a fill window until commit, abort or owner release; other mutations return EBUSY.
int dora_builder_prepare(struct dora_builder* builder,
                         uint64_t row_count,
                         size_t byte_count,
                         struct dora_mutable_bytes* out_values);

/// Failure preserves the fill window for retry while its owner survives.
int dora_builder_commit(struct dora_builder* builder, const bool* validity);

/// Discards the fill window; accepted abort succeeds even after callback owner release.
int dora_builder_abort(struct dora_builder* builder);

//------------------------------------------------------------------------------
//
// Completion and Lifecycle
//
//------------------------------------------------------------------------------

/// Counts one complete Struct-root record without committing pending rows.
int dora_builder_end_row(struct dora_builder* builder);

/// Publishes rows and releases the owning root builder; failure preserves out_vector.
int dora_builder_finish(struct dora_builder* in_out_builder, struct dora_vector* out_vector);

/// Shares storage; clear and finish return EBUSY until consumers release their claims.
int dora_builder_share(struct dora_builder* builder, struct dora_vector* out_vector);

/// Discards rows, including pending rows, while retaining storage.
int dora_builder_clear(struct dora_builder* builder);

int dora_builder_move(struct dora_builder* in_out_source, struct dora_builder* out_destination);

/// Invalidates borrows; owning release during callbacks is deferred until active work completes.
void dora_builder_release(struct dora_builder* builder);

bool dora_builder_is_valid(const struct dora_builder* builder);

#endif
