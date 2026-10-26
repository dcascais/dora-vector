// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Type admission and carrier registries.

#ifndef DORA_TYPES_TYPES_INTERNAL_H
#define DORA_TYPES_TYPES_INTERNAL_H

#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t UNION_MEMBER_CAPACITY = (uint32_t)INT8_MAX + 1;
static constexpr int32_t VIEW_INLINE_BYTE_COUNT = sizeof(((union dora_binary_view_payload){}).inline_bytes);
static constexpr int32_t DECIMAL32_MAX_PRECISION = 9;
static constexpr int32_t DECIMAL64_MAX_PRECISION = 18;
static constexpr int32_t DECIMAL128_MAX_PRECISION = 38;
static constexpr int32_t DECIMAL256_MAX_PRECISION = 76;

//------------------------------------------------------------------------------
//
// Carrier Families
//
//------------------------------------------------------------------------------

//--------------------------------------
// Fixed Carriers
//--------------------------------------

#define SIGNED_INDEX_CARRIERS(X, ...)                                                                                  \
    X(i8, int8_t, DORA_PHYSICAL_TYPE_I8 __VA_OPT__(, ) __VA_ARGS__)                                                    \
    X(i16, int16_t, DORA_PHYSICAL_TYPE_I16 __VA_OPT__(, ) __VA_ARGS__)                                                 \
    X(i32, int32_t, DORA_PHYSICAL_TYPE_I32 __VA_OPT__(, ) __VA_ARGS__)                                                 \
    X(i64, int64_t, DORA_PHYSICAL_TYPE_I64 __VA_OPT__(, ) __VA_ARGS__)

#define ROW_INDEX_CARRIERS(X, ...)                                                                                     \
    X(u32, uint32_t, DORA_PHYSICAL_TYPE_U32 __VA_OPT__(, ) __VA_ARGS__)                                                \
    X(u64, uint64_t, DORA_PHYSICAL_TYPE_U64 __VA_OPT__(, ) __VA_ARGS__)

#define UNSIGNED_INDEX_CARRIERS(X, ...)                                                                                \
    X(u8, uint8_t, DORA_PHYSICAL_TYPE_U8 __VA_OPT__(, ) __VA_ARGS__)                                                   \
    X(u16, uint16_t, DORA_PHYSICAL_TYPE_U16 __VA_OPT__(, ) __VA_ARGS__)                                                \
    ROW_INDEX_CARRIERS(X __VA_OPT__(, ) __VA_ARGS__)

#define FIXED_CARRIERS(X, ...)                                                                                         \
    SIGNED_INDEX_CARRIERS(X __VA_OPT__(, ) __VA_ARGS__)                                                                \
    X(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128 __VA_OPT__(, ) __VA_ARGS__)                                      \
    X(i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256 __VA_OPT__(, ) __VA_ARGS__)                                      \
    UNSIGNED_INDEX_CARRIERS(X __VA_OPT__(, ) __VA_ARGS__)                                                              \
    X(u128, struct dora_u128, DORA_PHYSICAL_TYPE_U128 __VA_OPT__(, ) __VA_ARGS__)                                      \
    X(f16, struct dora_f16_bits, DORA_PHYSICAL_TYPE_F16 __VA_OPT__(, ) __VA_ARGS__)                                    \
    X(f32, float, DORA_PHYSICAL_TYPE_F32 __VA_OPT__(, ) __VA_ARGS__)                                                   \
    X(f64, double, DORA_PHYSICAL_TYPE_F64 __VA_OPT__(, ) __VA_ARGS__)                                                  \
    X(interval_day_time,                                                                                               \
      struct dora_interval_day_time,                                                                                   \
      DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME __VA_OPT__(, ) __VA_ARGS__)                                                 \
    X(interval_month_day_nano,                                                                                         \
      struct dora_interval_month_day_nano,                                                                             \
      DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO __VA_OPT__(, ) __VA_ARGS__)

// Decimal (32/64/128-bit)
#define DECIMAL_RESCALE_CARRIERS(X, ...)                                                                               \
    X(i32, int32_t, DORA_PHYSICAL_TYPE_I32 __VA_OPT__(, ) __VA_ARGS__)                                                 \
    X(i64, int64_t, DORA_PHYSICAL_TYPE_I64 __VA_OPT__(, ) __VA_ARGS__)                                                 \
    X(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128 __VA_OPT__(, ) __VA_ARGS__)

#define NEGATIVE_SIGNED(VALUE) ((VALUE) < 0)
// VALUE is not expanded or evaluated.
#define NEGATIVE_UNSIGNED(VALUE) false

// Returns DORA_OK or EINVAL from the enclosing function.
#define INDEX_READ_CASE(S, T, P, NEGATIVE, DATA, ROW, OUT_INDEX)                                                       \
    case P: {                                                                                                          \
        T index = 0;                                                                                                   \
        memcpy(&index, (const uint8_t*)(DATA) + (ROW) * sizeof(index), sizeof(index));                                 \
        if (NEGATIVE(index)) {                                                                                         \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        *(OUT_INDEX) = (uint64_t)index;                                                                                \
        return DORA_OK;                                                                                                \
    }

// Returns zero without a fixed-width carrier.
static inline uint32_t physical_width(enum dora_physical_type type) {
    switch (type) {
#define PHYSICAL_WIDTH_CASE(S, T, P)                                                                                   \
    case P:                                                                                                            \
        return sizeof(T);
        FIXED_CARRIERS(PHYSICAL_WIDTH_CASE)
#undef PHYSICAL_WIDTH_CASE
    default:
        return 0;
    }
}

// Returns zero for invalid decimal types.
static inline uint32_t type_decimal_byte_width(const struct dora_logical_type* type) {
    if (type == nullptr || type->id != DORA_TYPE_DECIMAL) {
        return 0;
    }
    const struct dora_decimal_type decimal = type->parameters.decimal;
    const int32_t precision = decimal.precision;
    const uint32_t derived = precision <= 0                          ? 0
                             : precision <= DECIMAL32_MAX_PRECISION  ? 4
                             : precision <= DECIMAL64_MAX_PRECISION  ? 8
                             : precision <= DECIMAL128_MAX_PRECISION ? 16
                             : precision <= DECIMAL256_MAX_PRECISION ? 32
                                                                     : 0;
    if (derived == 0) {
        return 0;
    }
    if (decimal.bit_width == 0) {
        return derived;
    }
    if ((decimal.bit_width != 32 && decimal.bit_width != 64 && decimal.bit_width != 128 && decimal.bit_width != 256) ||
        decimal.bit_width / 8 < derived) {
        return 0;
    }
    return decimal.bit_width / 8;
}

// Non-fixed types yield DORA_PHYSICAL_TYPE_NONE.
static inline int type_physical(const struct dora_logical_type* type, enum dora_physical_type* out_type) {
    if (type == nullptr || out_type == nullptr) {
        return EINVAL;
    }
    uint32_t byte_width = 0;
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    switch (type->id) {
    case DORA_TYPE_INT8:
        physical = DORA_PHYSICAL_TYPE_I8;
        break;
    case DORA_TYPE_INT16:
        physical = DORA_PHYSICAL_TYPE_I16;
        break;
    case DORA_TYPE_INT32:
    case DORA_TYPE_DATE32:
    case DORA_TYPE_INTERVAL_YEAR_MONTH:
        physical = DORA_PHYSICAL_TYPE_I32;
        break;
    case DORA_TYPE_INT64:
    case DORA_TYPE_DATE64:
        physical = DORA_PHYSICAL_TYPE_I64;
        break;
    case DORA_TYPE_TIMESTAMP:
        if (type->parameters.timestamp.unit > DORA_TIME_UNIT_NANOSECOND) {
            return EINVAL;
        }
        physical = DORA_PHYSICAL_TYPE_I64;
        break;
    case DORA_TYPE_DURATION:
        if (type->parameters.unit > DORA_TIME_UNIT_NANOSECOND) {
            return EINVAL;
        }
        physical = DORA_PHYSICAL_TYPE_I64;
        break;
    case DORA_TYPE_INT128:
        physical = DORA_PHYSICAL_TYPE_I128;
        break;
    case DORA_TYPE_INT256:
        physical = DORA_PHYSICAL_TYPE_I256;
        break;
    case DORA_TYPE_UINT8:
        physical = DORA_PHYSICAL_TYPE_U8;
        break;
    case DORA_TYPE_UINT16:
        physical = DORA_PHYSICAL_TYPE_U16;
        break;
    case DORA_TYPE_UINT32:
        physical = DORA_PHYSICAL_TYPE_U32;
        break;
    case DORA_TYPE_UINT64:
        physical = DORA_PHYSICAL_TYPE_U64;
        break;
    case DORA_TYPE_UINT128:
        physical = DORA_PHYSICAL_TYPE_U128;
        break;
    case DORA_TYPE_FLOAT16:
        physical = DORA_PHYSICAL_TYPE_F16;
        break;
    case DORA_TYPE_FLOAT32:
        physical = DORA_PHYSICAL_TYPE_F32;
        break;
    case DORA_TYPE_FLOAT64:
        physical = DORA_PHYSICAL_TYPE_F64;
        break;
    case DORA_TYPE_INTERVAL_DAY_TIME:
        physical = DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME;
        break;
    case DORA_TYPE_INTERVAL_MONTH_DAY_NANO:
        physical = DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO;
        break;
    case DORA_TYPE_TIME:
        if (type->parameters.unit > DORA_TIME_UNIT_NANOSECOND) {
            return EINVAL;
        }
        physical =
            type->parameters.unit <= DORA_TIME_UNIT_MILLISECOND ? DORA_PHYSICAL_TYPE_I32 : DORA_PHYSICAL_TYPE_I64;
        break;
    case DORA_TYPE_DECIMAL: {
        byte_width = type_decimal_byte_width(type);
        if (byte_width == 0) {
            return EINVAL;
        }
        physical = byte_width == 4    ? DORA_PHYSICAL_TYPE_I32
                   : byte_width == 8  ? DORA_PHYSICAL_TYPE_I64
                   : byte_width == 16 ? DORA_PHYSICAL_TYPE_I128
                                      : DORA_PHYSICAL_TYPE_I256;
        break;
    }
    default:
        if (type->id > DORA_TYPE_UNION) {
            return EINVAL;
        }
        break;
    }
    *out_type = physical;
    return DORA_OK;
}

// Rejects nested types with ENOTSUP.
static inline int type_scalar_physical(const struct dora_logical_type* type, enum dora_physical_type* out_physical) {
    const int status = type_physical(type, out_physical);
    if (status != DORA_OK) {
        return status;
    }
    return type->id >= DORA_TYPE_LIST ? ENOTSUP : DORA_OK;
}

static inline bool bytes_are_well_formed(struct dora_bytes bytes) {
    return bytes.data != nullptr || bytes.byte_count == 0;
}

// Requires well-formed spans.
static inline bool bytes_are_equal(struct dora_bytes left, struct dora_bytes right) {
    return left.byte_count == right.byte_count &&
           (left.byte_count == 0 || memcmp(left.data, right.data, left.byte_count) == 0);
}

// Utf8, Binary, FixedBinary
static inline bool type_is_bytes(enum dora_type type) {
    return type == DORA_TYPE_UTF8 || type == DORA_TYPE_BINARY || type == DORA_TYPE_FIXED_BINARY;
}

static inline bool type_is_temporal(enum dora_type type) {
    return type >= DORA_TYPE_DATE32 && type <= DORA_TYPE_DURATION;
}

static inline bool type_has_timezone(const struct dora_logical_type* type) {
    return type->id == DORA_TYPE_TIMESTAMP && type->parameters.timestamp.timezone != nullptr &&
           type->parameters.timestamp.timezone[0] != '\0';
}

static inline bool type_is_native_integer(enum dora_type type) {
    return (type >= DORA_TYPE_INT8 && type <= DORA_TYPE_INT64) || (type >= DORA_TYPE_UINT8 && type <= DORA_TYPE_UINT64);
}

//------------------------------------------------------------------------------
//
// Type Admission
//
//------------------------------------------------------------------------------

int dora_type_check(const struct dora_logical_type* type);

int dora_type_resolve_layout(const struct dora_logical_type* type,
                             enum dora_layout layout,
                             enum dora_type index_type,
                             enum dora_layout* out_layout);

bool dora_type_is_equal(const struct dora_logical_type* left,
                        const struct dora_logical_type* right,
                        uint32_t union_child_count);

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(alignof(struct dora_i128) <= 8 && alignof(struct dora_i256) <= 8 && alignof(struct dora_u128) <= 8 &&
              alignof(struct dora_interval_month_day_nano) <= 8);
static_assert(sizeof(_BitInt(128)) == sizeof(struct dora_i128));
static_assert(sizeof(struct dora_i128) == 16);
static_assert(sizeof(struct dora_i256) == 32);
static_assert(sizeof(struct dora_u128) == 16);
static_assert(sizeof(struct dora_f16_bits) == 2);
static_assert(sizeof(struct dora_interval_day_time) == 8);
static_assert(sizeof(struct dora_interval_month_day_nano) == 16);
static_assert(sizeof(struct dora_binary_view_entry) == 16);
static_assert(alignof(struct dora_binary_view_entry) <= 8);

#endif
