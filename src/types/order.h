// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scalar ordering and comparison.

#ifndef DORA_TYPES_ORDER_H
#define DORA_TYPES_ORDER_H

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "numeric/float.h"
#include "numeric/limbs.h"
#include "temporal/interval.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Order
//
//------------------------------------------------------------------------------

//--------------------------------------
// Integers
//--------------------------------------

// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Decimal (32/64-bit), temporal types, IntervalYearMonth
#define TYPE_ORDER_INTEGER(S, T, P)                                                                                    \
    static inline int type_order_##S(T left, T right) {                                                                \
        return (left > right) - (left < right);                                                                        \
    }
SIGNED_INDEX_CARRIERS(TYPE_ORDER_INTEGER)
UNSIGNED_INDEX_CARRIERS(TYPE_ORDER_INTEGER)
#undef TYPE_ORDER_INTEGER

static inline int type_order_i128(struct dora_i128 left, struct dora_i128 right) {
    constexpr uint64_t sign = UINT64_C(0x80'00'00'00'00'00'00'00);
    const int high = type_order_u64(left.limbs[1] ^ sign, right.limbs[1] ^ sign);
    return high != 0 ? high : type_order_u64(left.limbs[0], right.limbs[0]);
}

static inline int type_order_i256(struct dora_i256 left, struct dora_i256 right) {
    constexpr uint64_t sign = UINT64_C(0x80'00'00'00'00'00'00'00);
    int order = type_order_u64(left.limbs[3] ^ sign, right.limbs[3] ^ sign);
    for (size_t limb = 3; order == 0 && limb != 0; --limb) {
        order = type_order_u64(left.limbs[limb - 1], right.limbs[limb - 1]);
    }
    return order;
}

static inline int type_order_u128(struct dora_u128 left, struct dora_u128 right) {
    const int high = type_order_u64(left.limbs[1], right.limbs[1]);
    return high != 0 ? high : type_order_u64(left.limbs[0], right.limbs[0]);
}

//--------------------------------------
// Floating Point
//--------------------------------------

static inline int type_order_f32(float left, float right) {
    return (left > right) - (left < right) + (int)numeric_f32_is_nan(left) - (int)numeric_f32_is_nan(right);
}

static inline int type_order_f64(double left, double right) {
    return (left > right) - (left < right) + (int)numeric_f64_is_nan(left) - (int)numeric_f64_is_nan(right);
}

static inline int32_t type_f16_order_key(struct dora_f16_bits value) {
    const int32_t magnitude = value.bits & 0x7fffU;
    return magnitude > 0x7c00 ? 0x7c01 : (value.bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

static inline int type_order_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    const int32_t a = type_f16_order_key(left);
    const int32_t b = type_f16_order_key(right);
    return (a > b) - (a < b);
}

//--------------------------------------
// Temporal
//--------------------------------------

static inline int type_order_interval_day_time(struct dora_interval_day_time left,
                                               struct dora_interval_day_time right) {
    return temporal_interval_order(
        temporal_interval_normalize(
            (struct temporal_interval_parts){.days = left.days, .nanoseconds = (int64_t)left.milliseconds * 1'000'000}),
        temporal_interval_normalize((struct temporal_interval_parts){
            .days = right.days,
            .nanoseconds = (int64_t)right.milliseconds * 1'000'000
        }));
}

static inline int type_order_interval_month_day_nano(struct dora_interval_month_day_nano left,
                                                     struct dora_interval_month_day_nano right) {
    return temporal_interval_order(
        temporal_interval_normalize((
            struct temporal_interval_parts){.months = left.months, .days = left.days, .nanoseconds = left.nanoseconds}),
        temporal_interval_normalize((struct temporal_interval_parts){
            .months = right.months,
            .days = right.days,
            .nanoseconds = right.nanoseconds
        }));
}

//--------------------------------------
// Boolean and Bytes
//--------------------------------------

static inline int type_order_bool(bool left, bool right) {
    return (int)left - (int)right;
}

static inline uint32_t type_bytes_prefix(struct dora_bytes value) {
    const size_t byte_count = value.byte_count < sizeof(uint32_t) ? value.byte_count : sizeof(uint32_t);
    uint32_t prefix = 0;
    for (size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
        prefix |= (uint32_t)value.data[byte_index] << (byte_index * 8U);
    }
    return prefix;
}

static inline int type_order_bytes(struct dora_bytes left, struct dora_bytes right) {
    const uint32_t left_prefix = type_bytes_prefix(left);
    const uint32_t right_prefix = type_bytes_prefix(right);
    if (left_prefix != right_prefix) {
        return __builtin_bswap32(left_prefix) > __builtin_bswap32(right_prefix) ? 1 : -1;
    }
    const size_t common = left.byte_count < right.byte_count ? left.byte_count : right.byte_count;
    const int order = common == 0 ? 0 : memcmp(left.data, right.data, common);
    return order != 0 ? (order > 0) - (order < 0)
                      : (left.byte_count > right.byte_count) - (left.byte_count < right.byte_count);
}

//--------------------------------------
// Decimals
//--------------------------------------

enum : uint32_t {
    TYPE_DECIMAL_DIGIT_CAPACITY = 78,
    TYPE_DECIMAL_CHUNK_DIGITS = 9,
    TYPE_DECIMAL_CHUNK_CAPACITY = 9
};

struct type_decimal_operand {
    struct dora_i256 coefficient;
    int32_t scale;
};

struct type_decimal_digits {
    uint8_t digits[TYPE_DECIMAL_DIGIT_CAPACITY];
    int64_t exponent;
    uint32_t count;
    int sign;
};

static inline size_t type_decimal_live_limbs(const uint64_t* magnitude, size_t limb_count) {
    while (limb_count != 0 && magnitude[limb_count - 1] == 0) {
        --limb_count;
    }
    return limb_count;
}

static inline struct type_decimal_digits type_decimal_digits(struct type_decimal_operand value) {
    struct type_decimal_digits result = {};
    uint64_t magnitude[4] = {};
    bool negative = false;
    dora_i256_to_mag(value.coefficient, magnitude, &negative);
    size_t live = type_decimal_live_limbs(magnitude, 4);
    if (live == 0) {
        return result;
    }
    uint32_t chunks[TYPE_DECIMAL_CHUNK_CAPACITY] = {};
    uint32_t chunk_count = 0;
    while (live != 0) {
        chunks[chunk_count++] =
            dora_limbs_div_u32((struct limbs_span){.values = magnitude, .count = live}, 1'000'000'000U);
        live = type_decimal_live_limbs(magnitude, live);
    }
    uint8_t lead[TYPE_DECIMAL_CHUNK_DIGITS] = {};
    uint32_t lead_count = 0;
    for (uint32_t top = chunks[chunk_count - 1]; top != 0; top /= 10) {
        lead[lead_count++] = (uint8_t)(top % 10);
    }
    while (lead_count != 0) {
        result.digits[result.count++] = lead[--lead_count];
    }
    for (uint32_t chunk = chunk_count - 1; chunk != 0; --chunk) {
        uint32_t remaining = chunks[chunk - 1];
        for (uint32_t place = TYPE_DECIMAL_CHUNK_DIGITS; place != 0; --place) {
            result.digits[result.count + place - 1] = (uint8_t)(remaining % 10);
            remaining /= 10;
        }
        result.count += TYPE_DECIMAL_CHUNK_DIGITS;
    }
    result.exponent = (int64_t)result.count - value.scale;
    result.sign = negative ? -1 : 1;
    return result;
}

static inline int type_order_decimal_digits(const struct type_decimal_digits* left,
                                            const struct type_decimal_digits* right) {
    if (left->sign != right->sign || left->sign == 0) {
        return (left->sign > right->sign) - (left->sign < right->sign);
    }
    int order = (left->exponent > right->exponent) - (left->exponent < right->exponent);
    const uint32_t count = left->count > right->count ? left->count : right->count;
    for (uint32_t index = 0; order == 0 && index < count; ++index) {
        const uint8_t left_digit = index < left->count ? left->digits[index] : 0;
        const uint8_t right_digit = index < right->count ? right->digits[index] : 0;
        order = (left_digit > right_digit) - (left_digit < right_digit);
    }
    return left->sign < 0 ? -order : order;
}

//--------------------------------------
// Predicates
//--------------------------------------

static inline bool type_order_matches(int order, enum dora_compare_op comparison) {
    static constexpr bool matches[DORA_COMPARE_OP_GE + 1][3] = {
        {false, true, false},
        {true, false, true},
        {true, false, false},
        {true, true, false},
        {false, false, true},
        {false, true, true},
    };
    return comparison <= DORA_COMPARE_OP_GE && matches[comparison][(order > 0) - (order < 0) + 1];
}

// Switch clause; returns from the enclosing function; evaluates VALUES and COMPARISON twice on success.
#define TYPE_ORDER_ROWS_CASE(S, T, P, VALUES, LEFT, RIGHT, COMPARISON, OUT_MATCHES)                                    \
    case P: {                                                                                                          \
        T left_key = {};                                                                                               \
        T right_key = {};                                                                                              \
        memcpy(&left_key, (const uint8_t*)(VALUES) + (LEFT) * sizeof(left_key), sizeof(left_key));                     \
        memcpy(&right_key, (const uint8_t*)(VALUES) + (RIGHT) * sizeof(right_key), sizeof(right_key));                 \
        if ((COMPARISON) > DORA_COMPARE_OP_GE) {                                                                       \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        *(OUT_MATCHES) = type_order_matches(type_order_##S(left_key, right_key), (COMPARISON));                        \
        return DORA_OK;                                                                                                \
    }

static inline int type_bytes_matches(struct dora_bytes left,
                                     struct dora_bytes right,
                                     enum dora_compare_op comparison,
                                     bool* out_matches) {
    if (comparison > DORA_COMPARE_OP_GE || (left.data == nullptr && left.byte_count != 0) ||
        (right.data == nullptr && right.byte_count != 0)) {
        return EINVAL;
    }
    const int order = left.byte_count == 0 || right.byte_count == 0
                          ? (left.byte_count > right.byte_count) - (left.byte_count < right.byte_count)
                          : type_order_bytes(left, right);
    *out_matches = type_order_matches(order, comparison);
    return DORA_OK;
}

#endif
