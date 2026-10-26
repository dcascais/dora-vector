// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector construction and access.

#ifndef DORA_VECTOR_VECTOR_H
#define DORA_VECTOR_VECTOR_H

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Buffer Ownership
//
//------------------------------------------------------------------------------

/// Callbacks prevent reset and release of the vector pool.
struct dora_buffer_owner {
    /// Buffer placement.
    struct dora_placement placement;
    /// Buffer readiness.
    struct dora_readiness readiness;
    /// Frees owner state after the last reference.
    void (*release)(struct dora_buffer_owner* owner);
    /// Copies source bytes into host memory.
    int (*read_host)(struct dora_buffer_owner* owner, struct dora_device_region source, uint8_t* out_bytes);
};

//------------------------------------------------------------------------------
//
// Construction by Copy
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_vector_init_copy_bool(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const bool* values,
                               const bool* validity,
                               uint64_t row_count);

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_vector_init_copy_i8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const int8_t* values,
                             const bool* validity,
                             uint64_t row_count);

int dora_vector_init_copy_i16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int16_t* values,
                              const bool* validity,
                              uint64_t row_count);

/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth
int dora_vector_init_copy_i32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int32_t* values,
                              const bool* validity,
                              uint64_t row_count);

/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration
int dora_vector_init_copy_i64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int64_t* values,
                              const bool* validity,
                              uint64_t row_count);

/// Int128, Decimal (128-bit)
int dora_vector_init_copy_i128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i128* values,
                               const bool* validity,
                               uint64_t row_count);

/// Int256, Decimal (256-bit)
int dora_vector_init_copy_i256(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i256* values,
                               const bool* validity,
                               uint64_t row_count);

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_vector_init_copy_u8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const uint8_t* values,
                             const bool* validity,
                             uint64_t row_count);

int dora_vector_init_copy_u16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint16_t* values,
                              const bool* validity,
                              uint64_t row_count);

int dora_vector_init_copy_u32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint32_t* values,
                              const bool* validity,
                              uint64_t row_count);

int dora_vector_init_copy_u64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint64_t* values,
                              const bool* validity,
                              uint64_t row_count);

int dora_vector_init_copy_u128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_u128* values,
                               const bool* validity,
                               uint64_t row_count);

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_vector_init_copy_f16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const struct dora_f16_bits* values,
                              const bool* validity,
                              uint64_t row_count);

int dora_vector_init_copy_f32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const float* values,
                              const bool* validity,
                              uint64_t row_count);

int dora_vector_init_copy_f64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const double* values,
                              const bool* validity,
                              uint64_t row_count);

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_vector_init_copy_interval_day_time(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            const struct dora_logical_type* type,
                                            const struct dora_interval_day_time* values,
                                            const bool* validity,
                                            uint64_t row_count);

int dora_vector_init_copy_interval_month_day_nano(struct dora_vector* vector,
                                                  struct dora_pool pool,
                                                  const struct dora_logical_type* type,
                                                  const struct dora_interval_month_day_nano* values,
                                                  const bool* validity,
                                                  uint64_t row_count);

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
int dora_vector_init_copy_bytes(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct dora_logical_type* type,
                                enum dora_layout layout,
                                const struct dora_bytes* values,
                                const bool* validity,
                                uint64_t row_count,
                                enum dora_failure_policy content_policy);

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
int dora_vector_init_copy_cstrings(struct dora_vector* vector,
                                   struct dora_pool pool,
                                   const struct dora_logical_type* type,
                                   enum dora_layout layout,
                                   const char* const* values,
                                   uint64_t row_count,
                                   enum dora_failure_policy content_policy);

//------------------------------------------------------------------------------
//
// Construction by Wrap
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_packed_bool(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const struct dora_logical_type* type,
                                      const uint8_t* bits,
                                      uint64_t bit_offset,
                                      const uint8_t* validity,
                                      uint64_t validity_bit_offset,
                                      uint64_t row_count,
                                      struct dora_buffer_owner* owner,
                                      struct dora_error* out_error);

//--------------------------------------
// Signed Integers
//--------------------------------------

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_i8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const int8_t* values,
                             const uint8_t* validity,
                             uint64_t validity_bit_offset,
                             uint64_t row_count,
                             struct dora_buffer_owner* owner,
                             struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_i16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int16_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth
int dora_vector_init_wrap_i32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int32_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration
int dora_vector_init_wrap_i64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int64_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Int128, Decimal (128-bit)
int dora_vector_init_wrap_i128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i128* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Int256, Decimal (256-bit)
int dora_vector_init_wrap_i256(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i256* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error);

//--------------------------------------
// Unsigned Integers
//--------------------------------------

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_u8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const uint8_t* values,
                             const uint8_t* validity,
                             uint64_t validity_bit_offset,
                             uint64_t row_count,
                             struct dora_buffer_owner* owner,
                             struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_u16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint16_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_u32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint32_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_u64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint64_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_u128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_u128* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error);

//--------------------------------------
// Floating Point
//--------------------------------------

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_f16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const struct dora_f16_bits* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_f32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const float* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_f64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const double* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error);

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_interval_day_time(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            const struct dora_logical_type* type,
                                            const struct dora_interval_day_time* values,
                                            const uint8_t* validity,
                                            uint64_t validity_bit_offset,
                                            uint64_t row_count,
                                            struct dora_buffer_owner* owner,
                                            struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_interval_month_day_nano(struct dora_vector* vector,
                                                  struct dora_pool pool,
                                                  const struct dora_logical_type* type,
                                                  const struct dora_interval_month_day_nano* values,
                                                  const uint8_t* validity,
                                                  uint64_t validity_bit_offset,
                                                  uint64_t row_count,
                                                  struct dora_buffer_owner* owner,
                                                  struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
int dora_vector_init_wrap_fixed_binary(struct dora_vector* vector,
                                       struct dora_pool pool,
                                       const struct dora_logical_type* type,
                                       const uint8_t* values,
                                       const uint8_t* validity,
                                       uint64_t validity_bit_offset,
                                       uint64_t row_count,
                                       struct dora_buffer_owner* owner,
                                       struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Binary, Utf8; layouts: Binary
int dora_vector_init_wrap_binary(struct dora_vector* vector,
                                 struct dora_pool pool,
                                 const struct dora_logical_type* type,
                                 const int32_t* offsets,
                                 struct dora_bytes data,
                                 const uint8_t* validity,
                                 uint64_t validity_bit_offset,
                                 uint64_t row_count,
                                 struct dora_buffer_owner* owner,
                                 struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Binary, Utf8; layouts: LargeBinary
int dora_vector_init_wrap_large_binary(struct dora_vector* vector,
                                       struct dora_pool pool,
                                       const struct dora_logical_type* type,
                                       const int64_t* offsets,
                                       struct dora_bytes data,
                                       const uint8_t* validity,
                                       uint64_t validity_bit_offset,
                                       uint64_t row_count,
                                       struct dora_buffer_owner* owner,
                                       struct dora_error* out_error);

/// Borrows buffers; admits content without checking it.
/// Binary, Utf8; layouts: BinaryView
int dora_vector_init_wrap_binary_view(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const struct dora_logical_type* type,
                                      const struct dora_binary_view_entry* views,
                                      const struct dora_bytes* data_buffers,
                                      uint32_t data_buffer_count,
                                      const uint8_t* validity,
                                      uint64_t validity_bit_offset,
                                      uint64_t row_count,
                                      struct dora_buffer_owner* owner,
                                      struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Composition
//
//------------------------------------------------------------------------------

//--------------------------------------
// Lists
//--------------------------------------

/// Allocates in the elements pool.
/// List; layouts: List, LargeList
int dora_vector_init_retain_list(struct dora_vector* vector,
                                 enum dora_layout layout,
                                 const struct dora_vector* elements,
                                 const uint64_t* element_counts,
                                 const bool* validity,
                                 uint64_t row_count);

/// Allocates in the elements pool.
/// List; layouts: ListView, LargeListView
int dora_vector_init_retain_list_view(struct dora_vector* vector,
                                      enum dora_layout layout,
                                      const struct dora_vector* elements,
                                      const uint64_t* element_rows,
                                      const uint64_t* element_counts,
                                      const bool* validity,
                                      uint64_t row_count);

/// Allocates in the elements pool.
int dora_vector_init_retain_fixed_list(struct dora_vector* vector,
                                       uint32_t list_element_count,
                                       const struct dora_vector* elements,
                                       const bool* validity,
                                       uint64_t row_count);

//--------------------------------------
// Structures
//--------------------------------------

/// Requires nonempty children in one pool; allocates in that pool.
int dora_vector_init_retain_struct(struct dora_vector* vector,
                                   const struct dora_vector* children,
                                   const char* const* names,
                                   uint32_t child_count,
                                   const bool* validity,
                                   uint64_t row_count);

/// Allocates a zero-field Struct in pool; nullptr validity marks every row valid.
int dora_vector_init_fieldless_struct(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const bool* validity,
                                      uint64_t row_count);

//--------------------------------------
// Maps
//--------------------------------------

/// Requires keys and values in one pool; allocates in that pool.
int dora_vector_init_retain_map(struct dora_vector* vector,
                                enum dora_layout layout,
                                bool are_keys_sorted,
                                const struct dora_vector* keys,
                                const struct dora_vector* values,
                                const uint64_t* entry_counts,
                                const bool* validity,
                                uint64_t row_count);

//--------------------------------------
// Unions
//--------------------------------------

/// Requires nonempty children in one pool; allocates in that pool.
int dora_vector_init_retain_dense_union(struct dora_vector* vector,
                                        const int8_t* type_codes,
                                        const struct dora_vector* children,
                                        const char* const* names,
                                        uint32_t child_count,
                                        const int8_t* type_ids,
                                        const uint32_t* member_rows,
                                        uint64_t row_count);

/// Requires nonempty children in one pool; allocates in that pool.
int dora_vector_init_retain_sparse_union(struct dora_vector* vector,
                                         const int8_t* type_codes,
                                         const struct dora_vector* children,
                                         const char* const* names,
                                         uint32_t child_count,
                                         const int8_t* type_ids,
                                         uint64_t row_count);

//--------------------------------------
// Encodings
//--------------------------------------

/// Requires indices and values in one pool; allocates in that pool.
/// Values: all logical types; indices: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
int dora_vector_init_retain_dictionary(struct dora_vector* vector,
                                       const struct dora_vector* indices,
                                       const struct dora_vector* values);

/// Allocates in the values pool.
int dora_vector_init_retain_run_end(struct dora_vector* vector,
                                    const struct dora_vector* values,
                                    const uint64_t* run_lengths);

/// Allocates in the value pool.
int dora_vector_init_retain_constant(struct dora_vector* vector, const struct dora_vector* value, uint64_t row_count);

/// Allocates in the source pool; indices may belong to another pool.
/// Source: all logical types; indices: UInt32, UInt64
int dora_vector_init_retain_selection(struct dora_vector* vector,
                                      const struct dora_vector* source,
                                      const struct dora_vector* indices);

//--------------------------------------
// Extensions
//--------------------------------------

/// Binds name through the storage pool registry; allocates in that pool.
int dora_vector_init_retain_extension(struct dora_vector* vector,
                                      struct dora_bytes name,
                                      struct dora_bytes metadata,
                                      const struct dora_vector* storage);

//--------------------------------------
// Validation and Null
//--------------------------------------

/// Shares source buffers in its pool; checks previously unchecked content under ERROR or NULL.
int dora_vector_init_retain_validated(struct dora_vector* vector,
                                      const struct dora_vector* source,
                                      enum dora_failure_policy content_policy,
                                      struct dora_error* out_error);

int dora_vector_init_null(struct dora_vector* vector, struct dora_pool pool, uint64_t row_count);

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

int dora_vector_retain(const struct dora_vector* vector, struct dora_vector* out_vector);

static inline struct dora_vector dora_vector_borrow(const struct dora_vector* vector) {
    if (vector == nullptr) {
        return (struct dora_vector){};
    }
    struct dora_vector borrow = *vector;
    borrow.handle.is_owner = false;
    return borrow;
}

int dora_vector_move(struct dora_vector* in_out_source, struct dora_vector* out_destination);

void dora_vector_release(struct dora_vector* vector);

bool dora_vector_is_valid(const struct dora_vector* vector);

int dora_vector_get_pool(const struct dora_vector* vector, struct dora_pool* out_pool);

//------------------------------------------------------------------------------
//
// Navigation
//
//------------------------------------------------------------------------------

int dora_vector_get_length(const struct dora_vector* vector, uint64_t* out_length);

int dora_vector_get_type(const struct dora_vector* vector, struct dora_logical_type* out_type);

int dora_vector_retain_schema(const struct dora_vector* vector, struct dora_schema* out_schema);

int dora_vector_get_child_count(const struct dora_vector* vector, uint32_t* out_child_count);

int dora_vector_get_child(const struct dora_vector* vector, uint32_t child_index, struct dora_vector* out_child);

int dora_vector_find_child(const struct dora_vector* vector, const char* name, struct dora_vector* out_child);

int dora_vector_get_storage(const struct dora_vector* vector, struct dora_vector* out_storage);

int dora_vector_get_statistics(const struct dora_vector* vector, struct dora_vector* out_statistics);

int dora_vector_slice(const struct dora_vector* vector, struct dora_range rows, struct dora_vector* out_slice);

//------------------------------------------------------------------------------
//
// Reads
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_vector_read_bool(const struct dora_vector* vector, uint64_t row, bool* out_value, bool* out_is_null);

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_vector_read_i8(const struct dora_vector* vector, uint64_t row, int8_t* out_value, bool* out_is_null);

int dora_vector_read_i16(const struct dora_vector* vector, uint64_t row, int16_t* out_value, bool* out_is_null);

/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth
int dora_vector_read_i32(const struct dora_vector* vector, uint64_t row, int32_t* out_value, bool* out_is_null);

/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration
int dora_vector_read_i64(const struct dora_vector* vector, uint64_t row, int64_t* out_value, bool* out_is_null);

/// Int128, Decimal (128-bit)
int dora_vector_read_i128(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_i128* out_value,
                          bool* out_is_null);

/// Int256, Decimal (256-bit)
int dora_vector_read_i256(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_i256* out_value,
                          bool* out_is_null);

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_vector_read_u8(const struct dora_vector* vector, uint64_t row, uint8_t* out_value, bool* out_is_null);

int dora_vector_read_u16(const struct dora_vector* vector, uint64_t row, uint16_t* out_value, bool* out_is_null);

int dora_vector_read_u32(const struct dora_vector* vector, uint64_t row, uint32_t* out_value, bool* out_is_null);

int dora_vector_read_u64(const struct dora_vector* vector, uint64_t row, uint64_t* out_value, bool* out_is_null);

int dora_vector_read_u128(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_u128* out_value,
                          bool* out_is_null);

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_vector_read_f16(const struct dora_vector* vector,
                         uint64_t row,
                         struct dora_f16_bits* out_value,
                         bool* out_is_null);

int dora_vector_read_f32(const struct dora_vector* vector, uint64_t row, float* out_value, bool* out_is_null);

int dora_vector_read_f64(const struct dora_vector* vector, uint64_t row, double* out_value, bool* out_is_null);

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_vector_read_interval_day_time(const struct dora_vector* vector,
                                       uint64_t row,
                                       struct dora_interval_day_time* out_value,
                                       bool* out_is_null);

int dora_vector_read_interval_month_day_nano(const struct dora_vector* vector,
                                             uint64_t row,
                                             struct dora_interval_month_day_nano* out_value,
                                             bool* out_is_null);

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
int dora_vector_read_bytes(const struct dora_vector* vector,
                           uint64_t row,
                           struct dora_bytes* out_value,
                           bool* out_is_null);

/// Writes NUL-terminated text; nullptr out_text measures bytes excluding NUL.
/// Null, Boolean, integers, floating-point types, Decimal, Utf8, temporal types, intervals; excludes extensions
int dora_vector_format_value(const struct dora_vector* vector,
                             uint64_t row,
                             char* out_text,
                             size_t capacity_bytes,
                             size_t* out_byte_count,
                             bool* out_is_null);

//--------------------------------------
// Nullness and Equality
//--------------------------------------

int dora_vector_is_null(const struct dora_vector* vector, uint64_t row, bool* out_is_null);

int dora_vector_get_null_count(const struct dora_vector* vector, uint64_t* out_null_count);

int dora_vector_is_equal(const struct dora_vector* left, const struct dora_vector* right, bool* out_is_equal);

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

/// Borrows settled host buffers without retaining the vector.
int dora_vector_get_view(const struct dora_vector* vector, struct dora_vector_view* out_view);

/// Borrows device buffers and readiness without retaining the vector.
int dora_vector_get_device_view(const struct dora_vector* vector, struct dora_vector_view* out_view);

//------------------------------------------------------------------------------
//
// Copies
//
//------------------------------------------------------------------------------

int dora_vector_copy(const struct dora_vector* vector, struct dora_pool pool, struct dora_vector* out_copy);

/// Shares buffers into pool; source pool reset and release return EBUSY while foreign claims remain.
int dora_vector_share(const struct dora_vector* vector, struct dora_pool pool, struct dora_vector* out_shared);

/// Allocates in the vector pool.
int dora_vector_decode(const struct dora_vector* vector, struct dora_vector* out_decoded);

/// Allocates in the vector pool.
int dora_vector_recode(const struct dora_vector* vector,
                       const enum dora_layout* node_layouts,
                       uint32_t node_layout_count,
                       struct dora_vector* out_recoded);

//------------------------------------------------------------------------------
//
// Validation and Readiness
//
//------------------------------------------------------------------------------

/// Checks previously unchecked content throughout the subtree without changing state.
int dora_vector_validate(const struct dora_vector* vector, struct dora_error* out_error);

int dora_vector_settle(const struct dora_vector* vector);

#endif
