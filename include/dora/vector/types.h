// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Logical types and value carriers.

#ifndef DORA_VECTOR_TYPES_H
#define DORA_VECTOR_TYPES_H

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Value Carriers
//
//------------------------------------------------------------------------------

//--------------------------------------
// Signed Integers
//--------------------------------------

/// Int128, Decimal (128-bit)
struct dora_i128 {
    /// Least-significant limb first.
    uint64_t limbs[2];
};

/// Int256, Decimal (256-bit)
struct dora_i256 {
    /// Least-significant limb first.
    uint64_t limbs[4];
};

//--------------------------------------
// Unsigned Integers
//--------------------------------------

struct dora_u128 {
    /// Least-significant limb first.
    uint64_t limbs[2];
};

//--------------------------------------
// Floating Point
//--------------------------------------

struct dora_f16_bits {
    /// IEEE 754 binary16 encoding.
    uint16_t bits;
};

//--------------------------------------
// Intervals
//--------------------------------------

struct dora_interval_day_time {
    /// Day component.
    int32_t days;
    /// Millisecond component.
    int32_t milliseconds;
};

struct dora_interval_month_day_nano {
    /// Month component.
    int32_t months;
    /// Day component.
    int32_t days;
    /// Nanosecond component.
    int64_t nanoseconds;
};

//--------------------------------------
// Binary Views
//--------------------------------------

/// Binary, Utf8; layouts: BinaryView
struct dora_binary_view_reference {
    /// First four value bytes.
    uint8_t prefix[4];
    /// External buffer index.
    int32_t buffer_index;
    /// Start within the external buffer, in bytes.
    int32_t byte_offset;
};

/// Binary, Utf8; layouts: BinaryView
union dora_binary_view_payload {
    /// Value bytes when byte_count <= 12.
    uint8_t inline_bytes[12];
    /// External location when byte_count > 12.
    struct dora_binary_view_reference reference;
};

/// Binary, Utf8; layouts: BinaryView
struct dora_binary_view_entry {
    /// Value length in bytes.
    int32_t byte_count;
    /// Inline bytes or external location.
    union dora_binary_view_payload payload;
};

//------------------------------------------------------------------------------
//
// Bytes and Rows
//
//------------------------------------------------------------------------------

struct dora_bytes {
    /// Borrowed byte start.
    const uint8_t* data;
    /// Length in bytes.
    size_t byte_count;
};

struct dora_mutable_bytes {
    /// Borrowed writable byte start.
    uint8_t* data;
    /// Length in bytes.
    size_t byte_count;
};

/// Byte initializer for ordinary or UTF-8 literals; excludes the terminating NUL.
#define DORA_BYTES_LITERAL(text)                                                                                       \
    {.data = (const uint8_t*)("" text),                                                                                \
     .byte_count = _Generic(&("" text), char (*)[]: sizeof("" text) - 1, unsigned char (*)[]: sizeof("" text) - 1)}

struct dora_range {
    /// First element.
    uint64_t offset;
    /// Element count.
    uint64_t length;
};

//------------------------------------------------------------------------------
//
// Logical Types
//
//------------------------------------------------------------------------------

enum dora_type : uint8_t {
    DORA_TYPE_NULL,
    DORA_TYPE_BOOLEAN,
    DORA_TYPE_INT8,
    DORA_TYPE_INT16,
    DORA_TYPE_INT32,
    DORA_TYPE_INT64,
    DORA_TYPE_INT128,
    DORA_TYPE_INT256,
    DORA_TYPE_UINT8,
    DORA_TYPE_UINT16,
    DORA_TYPE_UINT32,
    DORA_TYPE_UINT64,
    DORA_TYPE_UINT128,
    DORA_TYPE_FLOAT16,
    DORA_TYPE_FLOAT32,
    DORA_TYPE_FLOAT64,
    DORA_TYPE_DECIMAL,
    DORA_TYPE_UTF8,
    DORA_TYPE_BINARY,
    DORA_TYPE_FIXED_BINARY,
    DORA_TYPE_DATE32,
    DORA_TYPE_DATE64,
    DORA_TYPE_TIME,
    DORA_TYPE_TIMESTAMP,
    DORA_TYPE_DURATION,
    DORA_TYPE_INTERVAL_YEAR_MONTH,
    DORA_TYPE_INTERVAL_DAY_TIME,
    DORA_TYPE_INTERVAL_MONTH_DAY_NANO,
    DORA_TYPE_LIST,
    DORA_TYPE_FIXED_LIST,
    DORA_TYPE_STRUCT,
    DORA_TYPE_MAP,
    DORA_TYPE_UNION
};

enum dora_time_unit : uint8_t {
    DORA_TIME_UNIT_SECOND,
    DORA_TIME_UNIT_MILLISECOND,
    DORA_TIME_UNIT_MICROSECOND,
    DORA_TIME_UNIT_NANOSECOND
};

struct dora_decimal_type {
    /// Maximum decimal digits.
    int32_t precision;
    /// Power-of-ten divisor exponent.
    int32_t scale;
    /// Carrier bits; zero derives the width from precision.
    uint32_t bit_width;
};

struct dora_timestamp_type {
    /// Tick unit.
    enum dora_time_unit unit;
    /// Borrowed NUL-terminated timezone; nullptr denotes no timezone.
    const char* timezone;
};

union dora_type_parameters {
    /// Decimal precision, scale and carrier width.
    struct dora_decimal_type decimal;
    /// Timestamp unit and timezone.
    struct dora_timestamp_type timestamp;
    /// Time or Duration tick unit.
    enum dora_time_unit unit;
    /// FixedBinary bytes per value.
    uint32_t byte_width;
    /// FixedList elements per row.
    uint32_t list_element_count;
    /// Union child type codes.
    const int8_t* type_codes;
};

struct dora_logical_type {
    /// Logical type discriminator.
    enum dora_type id;
    /// Parameters selected by id.
    union dora_type_parameters parameters;
    /// Extension name; empty for a storage type.
    struct dora_bytes extension_name;
    /// Extension metadata bytes.
    struct dora_bytes extension_metadata;
};

//------------------------------------------------------------------------------
//
// Physical Types and Layouts
//
//------------------------------------------------------------------------------

enum dora_physical_type : uint8_t {
    DORA_PHYSICAL_TYPE_NONE,
    DORA_PHYSICAL_TYPE_I8,
    DORA_PHYSICAL_TYPE_I16,
    DORA_PHYSICAL_TYPE_I32,
    DORA_PHYSICAL_TYPE_I64,
    DORA_PHYSICAL_TYPE_I128,
    DORA_PHYSICAL_TYPE_I256,
    DORA_PHYSICAL_TYPE_U8,
    DORA_PHYSICAL_TYPE_U16,
    DORA_PHYSICAL_TYPE_U32,
    DORA_PHYSICAL_TYPE_U64,
    DORA_PHYSICAL_TYPE_U128,
    DORA_PHYSICAL_TYPE_F16,
    DORA_PHYSICAL_TYPE_F32,
    DORA_PHYSICAL_TYPE_F64,
    DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME,
    DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO
};

enum dora_layout : uint8_t {
    DORA_LAYOUT_DEFAULT,
    DORA_LAYOUT_NULL,
    DORA_LAYOUT_BITS,
    DORA_LAYOUT_FIXED,
    DORA_LAYOUT_BINARY,
    DORA_LAYOUT_LARGE_BINARY,
    DORA_LAYOUT_BINARY_VIEW,
    DORA_LAYOUT_LIST,
    DORA_LAYOUT_LARGE_LIST,
    DORA_LAYOUT_LIST_VIEW,
    DORA_LAYOUT_LARGE_LIST_VIEW,
    DORA_LAYOUT_FIXED_LIST,
    DORA_LAYOUT_STRUCT,
    DORA_LAYOUT_DENSE_UNION,
    DORA_LAYOUT_SPARSE_UNION,
    DORA_LAYOUT_DICTIONARY,
    DORA_LAYOUT_RUN_END
};

//------------------------------------------------------------------------------
//
// Operations
//
//------------------------------------------------------------------------------

enum dora_compare_op : uint8_t {
    DORA_COMPARE_OP_EQ,
    DORA_COMPARE_OP_NE,
    DORA_COMPARE_OP_LT,
    DORA_COMPARE_OP_LE,
    DORA_COMPARE_OP_GT,
    DORA_COMPARE_OP_GE
};

/// Treatment of failed content or operation checks.
enum dora_failure_policy : uint8_t {
    /// Publishes nothing on failure.
    DORA_FAILURE_POLICY_ERROR,
    /// Nulls failing rows; failing nonnullable rows return EINVAL.
    DORA_FAILURE_POLICY_NULL,
    /// Integer arithmetic modulo 2^width.
    DORA_FAILURE_POLICY_WRAP,
    /// Admits unchecked content as valid.
    DORA_FAILURE_POLICY_TRUST
};

enum dora_rounding_mode : uint8_t {
    DORA_ROUNDING_MODE_EXACT,
    DORA_ROUNDING_MODE_TOWARD_ZERO,
    DORA_ROUNDING_MODE_HALF_EVEN,
    DORA_ROUNDING_MODE_HALF_AWAY
};

enum dora_null_order : uint8_t {
    DORA_NULL_ORDER_LAST,
    DORA_NULL_ORDER_FIRST,
    DORA_NULL_ORDER_PROPAGATE
};

enum dora_range_bound : uint8_t {
    DORA_RANGE_BOUND_INCLUSIVE = 0,
    DORA_RANGE_BOUND_LOWER_EXCLUSIVE = 1,
    DORA_RANGE_BOUND_UPPER_EXCLUSIVE = 2,
    DORA_RANGE_BOUND_EXCLUSIVE = 3
};

#endif
