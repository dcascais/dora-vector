// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector computation and options.

#ifndef DORA_VECTOR_COMPUTE_H
#define DORA_VECTOR_COMPUTE_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Element-Wise
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

/// Requires matching input pools; allocates in that pool.
/// Boolean
int dora_compute_and(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Boolean
int dora_compute_or(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Boolean
int dora_compute_xor(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result);

/// Allocates in the source pool.
/// Boolean
int dora_compute_not(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Null Predicates
//--------------------------------------

/// Allocates in the source pool.
int dora_compute_is_null(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates in the source pool.
int dora_compute_is_not_null(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Arithmetic
//--------------------------------------

/// Requires matching input pools; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal, Date32, Time, Timestamp, Duration
int dora_compute_add(const struct dora_vector* left,
                     const struct dora_vector* right,
                     enum dora_failure_policy policy,
                     struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal, Date32, Time, Timestamp, Duration
int dora_compute_subtract(const struct dora_vector* left,
                          const struct dora_vector* right,
                          enum dora_failure_policy policy,
                          struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_multiply(const struct dora_vector* left,
                          const struct dora_vector* right,
                          enum dora_failure_policy policy,
                          struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_divide(const struct dora_vector* left,
                        const struct dora_vector* right,
                        enum dora_failure_policy policy,
                        struct dora_vector* out_result);

/// Requires matching input pools; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_remainder(const struct dora_vector* left,
                           const struct dora_vector* right,
                           enum dora_failure_policy policy,
                           struct dora_vector* out_result);

/// Requires one input pool; Decimal product/sum preserve precision, scale and width; allocates in that pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_multiply_add(const struct dora_vector* left,
                              const struct dora_vector* right,
                              const struct dora_vector* addend,
                              enum dora_failure_policy policy,
                              struct dora_vector* out_result);

/// Allocates in the source pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_abs(const struct dora_vector* source, enum dora_failure_policy policy, struct dora_vector* out_result);

/// Allocates in the source pool.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_negate(const struct dora_vector* source,
                        enum dora_failure_policy policy,
                        struct dora_vector* out_result);

//--------------------------------------
// Comparison and Conversion
//--------------------------------------

/// Requires matching input pools; allocates in that pool.
/// All logical types; ordered comparisons exclude Map subtrees
int dora_compute_compare(const struct dora_vector* left,
                         enum dora_compare_op op,
                         const struct dora_vector* right,
                         struct dora_vector* out_mask);

/// Allocates a nonnull mask in the shared input pool; null_order must be FIRST or LAST.
/// All logical types; ordered comparisons exclude Map subtrees
int dora_compute_compare_distinct(const struct dora_vector* left,
                                  enum dora_compare_op op,
                                  const struct dora_vector* right,
                                  enum dora_null_order null_order,
                                  struct dora_vector* out_mask);

/// Requires matching input pools; allocates in that pool.
/// All logical types except Map; no Map descendants
int dora_compute_order(const struct dora_vector* left,
                       const struct dora_vector* right,
                       enum dora_null_order null_order,
                       struct dora_vector* out_order);

/// Allocates in the source pool; nested schemas must match.
/// All logical types
int dora_compute_cast(const struct dora_vector* source,
                      const struct dora_logical_type* target,
                      enum dora_rounding_mode rounding_mode,
                      enum dora_failure_policy policy,
                      struct dora_vector* out_result);

/// Allocates in the source pool.
/// All logical types
int dora_compute_hash(const struct dora_vector* source, uint64_t seed, struct dora_vector* out_hashes);

/// Requires nonnull UInt64 hashes matching source length; allocates in the hashes pool.
/// Source: all logical types; hashes: UInt64
int dora_compute_hash_combine(const struct dora_vector* hashes,
                              const struct dora_vector* source,
                              struct dora_vector* out_hashes);

//--------------------------------------
// Bytes
//--------------------------------------

/// Allocates Binary in the source pool; offsets and counts are bytes.
/// Utf8, Binary, FixedBinary
int dora_compute_bytes_substring(const struct dora_vector* source,
                                 int64_t byte_offset,
                                 uint64_t byte_count,
                                 struct dora_vector* out_result);

/// Allocates nonnull Utf8; null inputs contribute no bytes; Binary and FixedBinary are unsupported.
/// Null, Boolean, signed integers, unsigned integers, floating-point types, Decimal, Utf8, temporal types, intervals
int dora_compute_utf8_concat(const struct dora_vector* inputs, uint32_t input_count, struct dora_vector* out_result);

/// Allocates nullable UInt64 byte counts in the source pool.
/// Utf8, Binary, FixedBinary
int dora_compute_bytes_count(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates checked Utf8 in the source pool; retains owned bytes or copies them.
/// Binary, FixedBinary
int dora_compute_utf8_from_binary(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Text
//
//--------------------------------------

/// Allocates nullable UInt64 scalar counts in the source pool.
int dora_compute_utf8_scalar_count(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates Utf8 by scalar range; negative offsets count backward, and ranges clamp to each row.
int dora_compute_utf8_substring(const struct dora_vector* source,
                                int64_t scalar_offset,
                                uint64_t scalar_count,
                                struct dora_vector* out_result);

/// Allocates first-match byte offsets and flags; an empty needle matches at zero.
int dora_compute_utf8_find(const struct dora_vector* source,
                           const struct dora_vector* needle,
                           struct dora_vector* out_byte_offsets,
                           struct dora_vector* out_found);

/// Requires a singleton needle; allocates nullable Boolean flags in the source pool.
int dora_compute_utf8_contains(const struct dora_vector* source,
                               const struct dora_vector* needle,
                               struct dora_vector* out_result);

/// Replaces nonoverlapping occurrences; an empty needle leaves text unchanged.
int dora_compute_utf8_replace(const struct dora_vector* source,
                              const struct dora_vector* needle,
                              const struct dora_vector* replacement,
                              struct dora_vector* out_result);

/// Allocates List<Utf8> with BinaryView children; an empty separator splits scalars.
int dora_compute_utf8_split(const struct dora_vector* source,
                            const struct dora_vector* separator,
                            struct dora_vector* out_result);

/// Joins nonnull children; absent or null separators select comma; null parents remain null.
/// List<Utf8>, FixedList<Utf8>
int dora_compute_utf8_join(const struct dora_vector* source,
                           const struct dora_vector* separator,
                           struct dora_vector* out_result);

/// Absent characters select Unicode space separators; an empty set removes nothing.
int dora_compute_utf8_trim(const struct dora_vector* source,
                           const struct dora_vector* characters,
                           struct dora_vector* out_result);

/// Requires a singleton prefix; an empty prefix matches.
int dora_compute_utf8_starts_with(const struct dora_vector* source,
                                  const struct dora_vector* prefix,
                                  struct dora_vector* out_result);

/// Requires a singleton suffix; an empty suffix matches.
int dora_compute_utf8_ends_with(const struct dora_vector* source,
                                const struct dora_vector* suffix,
                                struct dora_vector* out_result);

/// Allocates Utf8 from Unicode scalar values in the source pool.
/// Int32
int dora_compute_utf8_from_scalar(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates nullable Int32 first scalars; empty text returns -1.
int dora_compute_utf8_first_scalar(const struct dora_vector* source, struct dora_vector* out_result);

/// Pads cyclically or truncates by scalar count; empty padding returns EINVAL when growth is required.
int dora_compute_utf8_lpad(const struct dora_vector* source,
                           int64_t target_scalar_count,
                           const struct dora_vector* padding,
                           struct dora_vector* out_result);

/// Pads cyclically or truncates by scalar count; empty padding returns EINVAL when growth is required.
int dora_compute_utf8_rpad(const struct dora_vector* source,
                           int64_t target_scalar_count,
                           const struct dora_vector* padding,
                           struct dora_vector* out_result);

/// Applies Unicode simple lowercase mapping without normalization or expansion.
int dora_compute_utf8_lower(const struct dora_vector* source, struct dora_vector* out_result);

/// Applies Unicode simple uppercase mapping without normalization or expansion.
int dora_compute_utf8_upper(const struct dora_vector* source, struct dora_vector* out_result);

/// Reverses Unicode scalars in each row.
int dora_compute_utf8_reverse(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Pattern Matching
//--------------------------------------

/// Matches complete rows; % spans scalars and _ matches one; escape is absent, empty or one byte.
int dora_compute_utf8_like(const struct dora_vector* source,
                           const struct dora_vector* pattern,
                           const struct dora_vector* escape,
                           struct dora_vector* out_result);

/// Applies LIKE after Unicode simple lowercase mapping; no normalization or full casefold.
int dora_compute_utf8_ilike(const struct dora_vector* source,
                            const struct dora_vector* pattern,
                            const struct dora_vector* escape,
                            struct dora_vector* out_result);

//------------------------------------------------------------------------------
//
// Selection
//
//------------------------------------------------------------------------------

/// Requires matching input pools; allocates in that pool.
/// All logical types; ordered comparisons exclude Map subtrees
int dora_compute_select(const struct dora_vector* left,
                        enum dora_compare_op op,
                        const struct dora_vector* right,
                        struct dora_vector* out_indices);

/// Allocates in the source pool.
int dora_compute_select_null(const struct dora_vector* source, struct dora_vector* out_indices);

/// Allocates in the source pool.
int dora_compute_select_not_null(const struct dora_vector* source, struct dora_vector* out_indices);

/// Allocates in the source pool.
/// Boolean
int dora_compute_select_mask(const struct dora_vector* mask, struct dora_vector* out_indices);

/// Allocates in the shared input pool; nullptr outputs omit their partitions.
/// All logical types; ordered comparisons exclude Map subtrees
int dora_compute_partition(const struct dora_vector* left,
                           enum dora_compare_op op,
                           const struct dora_vector* right,
                           struct dora_vector* out_true,
                           struct dora_vector* out_false,
                           struct dora_vector* out_null);

/// Allocates in the source pool; null bounds select no rows.
/// All logical types except Map; no Map descendants
int dora_compute_select_range(const struct dora_vector* source,
                              const struct dora_vector* lower,
                              const struct dora_vector* upper,
                              enum dora_range_bound bounds,
                              struct dora_vector* out_indices);

//------------------------------------------------------------------------------
//
// Generation
//
//------------------------------------------------------------------------------

/// Int8, Int16, Int32, Int64
int dora_compute_sequence(struct dora_pool pool,
                          const struct dora_logical_type* type,
                          int64_t start,
                          int64_t step,
                          uint64_t row_count,
                          struct dora_vector* out_result);

//------------------------------------------------------------------------------
//
// Aggregates
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

/// Allocates in the source pool.
/// Boolean
int dora_compute_any(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates in the source pool.
/// Boolean
int dora_compute_all(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Numeric
//--------------------------------------

/// Allocates in the source pool.
/// Boolean, signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_sum(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates one source-typed row; empty or all-null input returns null.
/// All logical types except Map; no Map descendants
int dora_compute_min(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates one source-typed row; empty or all-null input returns null.
/// All logical types except Map; no Map descendants
int dora_compute_max(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates one nonnull Int64 count of nonnull rows in the source pool.
int dora_compute_count(const struct dora_vector* source, struct dora_vector* out_result);

/// Allocates one Float64 row; empty or all-null input returns null.
/// Signed integers, unsigned integers, floating-point types, Decimal
int dora_compute_product(const struct dora_vector* source, struct dora_vector* out_result);

//--------------------------------------
// Decimals
//--------------------------------------

/// Allocates in the source pool.
int dora_compute_product_decimal(const struct dora_vector* source,
                                 const struct dora_logical_type* target,
                                 struct dora_vector* out_result);

//--------------------------------------
// Text
//--------------------------------------

/// Groups exact Utf8 keys in first-occurrence order; joins nonnull values in row order.
int dora_compute_utf8_join_groups(const struct dora_vector* source,
                                  const struct dora_vector* keys,
                                  const struct dora_vector* separator,
                                  struct dora_vector* out_keys,
                                  struct dora_vector* out_result);

#endif
