// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scalar numeric arithmetic.

#ifndef DORA_NUMERIC_ARITHMETIC_H
#define DORA_NUMERIC_ARITHMETIC_H

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "numeric/failure.h"
#include "numeric/float.h"
#include "numeric/limbs.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Kernels require strict floating-point semantics"
#endif

//------------------------------------------------------------------------------
//
// Native Integers
//
//------------------------------------------------------------------------------

//--------------------------------------
// Signed Integers
//--------------------------------------

#define NUMERIC_ARITHMETIC_SIGNED(X)                                                                                   \
    X(i8, int8_t, uint8_t, INT8_MIN)                                                                                   \
    X(i16, int16_t, uint16_t, INT16_MIN)                                                                               \
    X(i32, int32_t, uint32_t, INT32_MIN)                                                                               \
    X(i64, int64_t, uint64_t, INT64_MIN)

// Int8, Int16, Int32, Int64
#define NUMERIC_DEFINE_SIGNED(S, T, U, MINIMUM)                                                                        \
    static inline T numeric_signed_bits_##S(U bits) {                                                                  \
        T value = 0;                                                                                                   \
        (memcpy)(&value, &bits, sizeof(value));                                                                        \
        return value;                                                                                                  \
    }                                                                                                                  \
    static inline bool numeric_multiply_checked_##S(T left, T right, T* out_value) {                                   \
        return ckd_mul(out_value, left, right);                                                                        \
    }                                                                                                                  \
    static inline int numeric_divide_##S(T left, T right, T* out_value) {                                              \
        if (left == (MINIMUM) && right == -1) {                                                                        \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_OVERFLOW;                                                                           \
        }                                                                                                              \
        if (right == 0) {                                                                                              \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_DIVIDE_BY_ZERO;                                                                     \
        }                                                                                                              \
        *out_value = (T)(left / right);                                                                                \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static inline int numeric_remainder_##S(T left, T right, T* out_value) {                                           \
        if (left == (MINIMUM) && right == -1) {                                                                        \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_OVERFLOW;                                                                           \
        }                                                                                                              \
        if (right == 0) {                                                                                              \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_DIVIDE_BY_ZERO;                                                                     \
        }                                                                                                              \
        *out_value = (T)(left % right);                                                                                \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static inline T numeric_add_wrap_##S(T left, T right) {                                                            \
        return numeric_signed_bits_##S((U)((uint64_t)(U)left + (uint64_t)(U)right));                                   \
    }                                                                                                                  \
    static inline T numeric_subtract_wrap_##S(T left, T right) {                                                       \
        return numeric_signed_bits_##S((U)((uint64_t)(U)left - (uint64_t)(U)right));                                   \
    }                                                                                                                  \
    static inline T numeric_multiply_wrap_##S(T left, T right) {                                                       \
        return numeric_signed_bits_##S((U)((uint64_t)(U)left * (uint64_t)(U)right));                                   \
    }                                                                                                                  \
    static inline T numeric_negate_wrap_##S(T value) {                                                                 \
        return numeric_signed_bits_##S((U)(0 - (uint64_t)(U)value));                                                   \
    }                                                                                                                  \
    static inline T numeric_abs_wrap_##S(T value) {                                                                    \
        return value < 0 ? numeric_negate_wrap_##S(value) : value;                                                     \
    }                                                                                                                  \
    static inline bool numeric_negate_checked_##S(T value, T* out_value) {                                             \
        *out_value = numeric_negate_wrap_##S(value);                                                                   \
        return value == (MINIMUM);                                                                                     \
    }                                                                                                                  \
    static inline bool numeric_abs_checked_##S(T value, T* out_value) {                                                \
        *out_value = numeric_abs_wrap_##S(value);                                                                      \
        return value == (MINIMUM);                                                                                     \
    }
NUMERIC_ARITHMETIC_SIGNED(NUMERIC_DEFINE_SIGNED)
#undef NUMERIC_DEFINE_SIGNED
#undef NUMERIC_ARITHMETIC_SIGNED

#define NUMERIC_ARITHMETIC_WIDENED(X)                                                                                  \
    X(i8, int8_t, int16_t, INT8_MIN, INT8_MAX)                                                                         \
    X(i16, int16_t, int32_t, INT16_MIN, INT16_MAX)

// Int8, Int16
#define NUMERIC_DEFINE_WIDENED(S, T, W, MINIMUM, MAXIMUM)                                                              \
    static inline bool numeric_add_checked_##S(T left, T right, T* out_value) {                                        \
        const W result = (W)((W)left + (W)right);                                                                      \
        *out_value = (T)result;                                                                                        \
        return result < (MINIMUM) || result > (MAXIMUM);                                                               \
    }                                                                                                                  \
    static inline bool numeric_subtract_checked_##S(T left, T right, T* out_value) {                                   \
        const W result = (W)((W)left - (W)right);                                                                      \
        *out_value = (T)result;                                                                                        \
        return result < (MINIMUM) || result > (MAXIMUM);                                                               \
    }
NUMERIC_ARITHMETIC_WIDENED(NUMERIC_DEFINE_WIDENED)
#undef NUMERIC_DEFINE_WIDENED
#undef NUMERIC_ARITHMETIC_WIDENED

#define NUMERIC_ARITHMETIC_CARRIED(X)                                                                                  \
    X(i32, int32_t)                                                                                                    \
    X(i64, int64_t)

// Int32, Int64
#define NUMERIC_DEFINE_CARRIED(S, T)                                                                                   \
    static inline bool numeric_add_checked_##S(T left, T right, T* out_value) {                                        \
        return ckd_add(out_value, left, right);                                                                        \
    }                                                                                                                  \
    static inline bool numeric_subtract_checked_##S(T left, T right, T* out_value) {                                   \
        return ckd_sub(out_value, left, right);                                                                        \
    }
NUMERIC_ARITHMETIC_CARRIED(NUMERIC_DEFINE_CARRIED)
#undef NUMERIC_DEFINE_CARRIED
#undef NUMERIC_ARITHMETIC_CARRIED

//--------------------------------------
// Unsigned Integers
//--------------------------------------

#define NUMERIC_ARITHMETIC_UNSIGNED(X)                                                                                 \
    X(u8, uint8_t)                                                                                                     \
    X(u16, uint16_t)                                                                                                   \
    X(u32, uint32_t)                                                                                                   \
    X(u64, uint64_t)

// UInt8, UInt16, UInt32, UInt64
#define NUMERIC_DEFINE_UNSIGNED(S, T)                                                                                  \
    static inline bool numeric_multiply_checked_##S(T left, T right, T* out_value) {                                   \
        return ckd_mul(out_value, left, right);                                                                        \
    }                                                                                                                  \
    static inline int numeric_divide_##S(T left, T right, T* out_value) {                                              \
        if (right == 0) {                                                                                              \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_DIVIDE_BY_ZERO;                                                                     \
        }                                                                                                              \
        *out_value = (T)(left / right);                                                                                \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static inline int numeric_remainder_##S(T left, T right, T* out_value) {                                           \
        if (right == 0) {                                                                                              \
            *out_value = left;                                                                                         \
            return NUMERIC_FAILURE_DIVIDE_BY_ZERO;                                                                     \
        }                                                                                                              \
        *out_value = (T)(left % right);                                                                                \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static inline T numeric_add_wrap_##S(T left, T right) {                                                            \
        return (T)((uint64_t)left + (uint64_t)right);                                                                  \
    }                                                                                                                  \
    static inline T numeric_subtract_wrap_##S(T left, T right) {                                                       \
        return (T)((uint64_t)left - (uint64_t)right);                                                                  \
    }                                                                                                                  \
    static inline T numeric_multiply_wrap_##S(T left, T right) {                                                       \
        return (T)((uint64_t)left * (uint64_t)right);                                                                  \
    }                                                                                                                  \
    static inline T numeric_negate_wrap_##S(T value) {                                                                 \
        return (T)(0 - (uint64_t)value);                                                                               \
    }                                                                                                                  \
    static inline T numeric_abs_wrap_##S(T value) {                                                                    \
        return value;                                                                                                  \
    }                                                                                                                  \
    static inline bool numeric_flag_set_##S(T flags) {                                                                 \
        return ((uint64_t)flags >> (sizeof(flags) * 8U - 1U)) != 0U;                                                   \
    }                                                                                                                  \
    static inline T numeric_add_flag_##S(T left, T right, T* out_value) {                                              \
        const T result = numeric_add_wrap_##S(left, right);                                                            \
        *out_value = result;                                                                                           \
        return (T)(0U - (T)(result < left));                                                                           \
    }                                                                                                                  \
    static inline T numeric_subtract_flag_##S(T left, T right, T* out_value) {                                         \
        *out_value = numeric_subtract_wrap_##S(left, right);                                                           \
        return (T)(0U - (T)(left < right));                                                                            \
    }                                                                                                                  \
    static inline T numeric_negate_flag_##S(T value, T* out_value) {                                                   \
        *out_value = numeric_negate_wrap_##S(value);                                                                   \
        return (T)(0U - (T)(value != 0));                                                                              \
    }                                                                                                                  \
    static inline T numeric_abs_flag_##S(T value, T* out_value) {                                                      \
        *out_value = value;                                                                                            \
        return 0;                                                                                                      \
    }                                                                                                                  \
    static inline bool numeric_add_checked_##S(T left, T right, T* out_value) {                                        \
        return numeric_flag_set_##S(numeric_add_flag_##S(left, right, out_value));                                     \
    }                                                                                                                  \
    static inline bool numeric_subtract_checked_##S(T left, T right, T* out_value) {                                   \
        return numeric_flag_set_##S(numeric_subtract_flag_##S(left, right, out_value));                                \
    }                                                                                                                  \
    static inline bool numeric_negate_checked_##S(T value, T* out_value) {                                             \
        return numeric_flag_set_##S(numeric_negate_flag_##S(value, out_value));                                        \
    }                                                                                                                  \
    static inline bool numeric_abs_checked_##S(T value, T* out_value) {                                                \
        return numeric_flag_set_##S(numeric_abs_flag_##S(value, out_value));                                           \
    }
NUMERIC_ARITHMETIC_UNSIGNED(NUMERIC_DEFINE_UNSIGNED)
#undef NUMERIC_DEFINE_UNSIGNED
#undef NUMERIC_ARITHMETIC_UNSIGNED

//------------------------------------------------------------------------------
//
// Wide Integers
//
//------------------------------------------------------------------------------

static constexpr _BitInt(128) NUMERIC_I128_MAX = (_BitInt(128))((((unsigned _BitInt(128))1) << 127U) - 1U);
static constexpr _BitInt(128) NUMERIC_I128_MIN = -NUMERIC_I128_MAX - 1;

//--------------------------------------
// Signed Integers
//--------------------------------------

static inline struct dora_i128 numeric_add_wrap_i128(struct dora_i128 left, struct dora_i128 right) {
    const unsigned _BitInt(128) a = (unsigned _BitInt(128))type_i128_load(left);
    const unsigned _BitInt(128) b = (unsigned _BitInt(128))type_i128_load(right);
    const unsigned _BitInt(128) result = a + b;
    return type_i128_store(result);
}

static inline struct dora_i128 numeric_subtract_wrap_i128(struct dora_i128 left, struct dora_i128 right) {
    const unsigned _BitInt(128) a = (unsigned _BitInt(128))type_i128_load(left);
    const unsigned _BitInt(128) b = (unsigned _BitInt(128))type_i128_load(right);
    const unsigned _BitInt(128) result = a - b;
    return type_i128_store(result);
}

static inline struct dora_i128 numeric_multiply_wrap_i128(struct dora_i128 left, struct dora_i128 right) {
    const unsigned _BitInt(128) a = (unsigned _BitInt(128))type_i128_load(left);
    const unsigned _BitInt(128) b = (unsigned _BitInt(128))type_i128_load(right);
    const unsigned _BitInt(128) result = a * b;
    return type_i128_store(result);
}

static inline struct dora_i128 numeric_negate_wrap_i128(struct dora_i128 value) {
    const unsigned _BitInt(128) v = (unsigned _BitInt(128))type_i128_load(value);
    const unsigned _BitInt(128) result = 0U - v;
    return type_i128_store(result);
}

static inline struct dora_i128 numeric_abs_wrap_i128(struct dora_i128 value) {
    const _BitInt(128) v = type_i128_load(value);
    const unsigned _BitInt(128) uv = (unsigned _BitInt(128))v;
    const unsigned _BitInt(128) result = v < 0 ? (0U - uv) : uv;
    return type_i128_store(result);
}

static inline bool
numeric_add_checked_i128(struct dora_i128 left, struct dora_i128 right, struct dora_i128* out_value) {
    const unsigned _BitInt(128) ua = (unsigned _BitInt(128))type_i128_load(left);
    const unsigned _BitInt(128) ub = (unsigned _BitInt(128))type_i128_load(right);
    const unsigned _BitInt(128) ur = ua + ub;
    *out_value = type_i128_store(ur);
    return (((ur ^ ua) & (ur ^ ub)) >> 127U) != 0;
}

static inline bool
numeric_subtract_checked_i128(struct dora_i128 left, struct dora_i128 right, struct dora_i128* out_value) {
    const unsigned _BitInt(128) ua = (unsigned _BitInt(128))type_i128_load(left);
    const unsigned _BitInt(128) ub = (unsigned _BitInt(128))type_i128_load(right);
    const unsigned _BitInt(128) ur = ua - ub;
    *out_value = type_i128_store(ur);
    return (((ua ^ ub) & (ur ^ ua)) >> 127U) != 0;
}

static inline bool numeric_negate_checked_i128(struct dora_i128 value, struct dora_i128* out_value) {
    const _BitInt(128) a = type_i128_load(value);
    *out_value = numeric_negate_wrap_i128(value);
    return a == NUMERIC_I128_MIN;
}

static inline bool numeric_abs_checked_i128(struct dora_i128 value, struct dora_i128* out_value) {
    const _BitInt(128) a = type_i128_load(value);
    *out_value = numeric_abs_wrap_i128(value);
    return a == NUMERIC_I128_MIN;
}

static inline struct dora_i256 numeric_add_wrap_i256(struct dora_i256 left, struct dora_i256 right) {
    struct dora_i256 result = {};
    dora_limbs_add(left.limbs, right.limbs, 4, result.limbs);
    return result;
}

static inline struct dora_i256 numeric_subtract_wrap_i256(struct dora_i256 left, struct dora_i256 right) {
    struct dora_i256 result = {};
    dora_limbs_subtract(left.limbs, right.limbs, 4, result.limbs);
    return result;
}

static inline struct dora_i256 numeric_multiply_wrap_i256(struct dora_i256 left, struct dora_i256 right) {
    uint64_t product[8] = {};
    struct dora_i256 result = {};
    dora_limbs_multiply((struct limbs_multiplication_input){.left = left.limbs, .right = right.limbs, .count = 4},
                        product);
    (memcpy)(result.limbs, product, sizeof(result.limbs));
    return result;
}

static inline struct dora_i256 numeric_negate_wrap_i256(struct dora_i256 value) {
    struct dora_i256 result = {};
    dora_limbs_negate(value.limbs, 4, result.limbs);
    return result;
}

static inline struct dora_i256 numeric_abs_wrap_i256(struct dora_i256 value) {
    return dora_limbs_is_negative(value.limbs, 4) ? numeric_negate_wrap_i256(value) : value;
}

static inline bool
numeric_add_checked_i256(struct dora_i256 left, struct dora_i256 right, struct dora_i256* out_value) {
    const struct dora_i256 result = numeric_add_wrap_i256(left, right);
    *out_value = result;
    return dora_limbs_is_negative(left.limbs, 4) == dora_limbs_is_negative(right.limbs, 4) &&
           dora_limbs_is_negative(result.limbs, 4) != dora_limbs_is_negative(left.limbs, 4);
}

static inline bool
numeric_subtract_checked_i256(struct dora_i256 left, struct dora_i256 right, struct dora_i256* out_value) {
    const struct dora_i256 result = numeric_subtract_wrap_i256(left, right);
    *out_value = result;
    return dora_limbs_is_negative(left.limbs, 4) != dora_limbs_is_negative(right.limbs, 4) &&
           dora_limbs_is_negative(result.limbs, 4) != dora_limbs_is_negative(left.limbs, 4);
}

static inline bool numeric_negate_checked_i256(struct dora_i256 value, struct dora_i256* out_value) {
    *out_value = numeric_negate_wrap_i256(value);
    return dora_limbs_is_minimum(value.limbs, 4);
}

static inline bool numeric_abs_checked_i256(struct dora_i256 value, struct dora_i256* out_value) {
    *out_value = numeric_abs_wrap_i256(value);
    return dora_limbs_is_minimum(value.limbs, 4);
}

static inline struct dora_i256 numeric_i256_from_i64(int64_t value) {
    const uint64_t fill = value < 0 ? UINT64_MAX : 0;
    return (struct dora_i256){.limbs = {(uint64_t)value, fill, fill, fill}};
}

static inline struct dora_i256 numeric_i256_from_u64(uint64_t value) {
    return (struct dora_i256){.limbs = {value}};
}

static inline struct dora_i256 numeric_i256_from_i128(struct dora_i128 value) {
    const uint64_t fill = (value.limbs[1] >> 63U) != 0 ? UINT64_MAX : 0;
    return (struct dora_i256){.limbs = {value.limbs[0], value.limbs[1], fill, fill}};
}

static inline struct dora_i256 numeric_i256_from_u128(struct dora_u128 value) {
    return (struct dora_i256){.limbs = {value.limbs[0], value.limbs[1]}};
}

static inline int
numeric_load_integer(enum dora_physical_type physical, const void* value, struct dora_i256* out_value) {
    switch (physical) {
#define LOAD_INTEGER(S, T, P, CONVERT)                                                                                 \
    case P:                                                                                                            \
        *out_value = CONVERT(type_load_##S(value, 0));                                                                 \
        return DORA_OK;
        SIGNED_INDEX_CARRIERS(LOAD_INTEGER, numeric_i256_from_i64)
        UNSIGNED_INDEX_CARRIERS(LOAD_INTEGER, numeric_i256_from_u64)
        LOAD_INTEGER(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128, numeric_i256_from_i128)
        LOAD_INTEGER(u128, struct dora_u128, DORA_PHYSICAL_TYPE_U128, numeric_i256_from_u128)
#undef LOAD_INTEGER
    case DORA_PHYSICAL_TYPE_I256:
        *out_value = type_load_i256(value, 0);
        return DORA_OK;
    default:
        return ENOTSUP;
    }
}

static inline bool
numeric_multiply_checked_i128(struct dora_i128 left, struct dora_i128 right, struct dora_i128* out_value) {
    const _BitInt(128) a = type_i128_load(left);
    const _BitInt(128) b = type_i128_load(right);
    if (a == NUMERIC_I128_MIN || b == NUMERIC_I128_MIN) {
        const bool zero = a == 0 || b == 0;
        const bool one = a == 1 || b == 1;
        *out_value = type_i128_store(zero ? 0 : NUMERIC_I128_MIN);
        return !zero && !one;
    }
    const bool negative = (a < 0) != (b < 0);
    const unsigned _BitInt(128) left_magnitude = a < 0 ? -(unsigned _BitInt(128))a : (unsigned _BitInt(128))a;
    const unsigned _BitInt(128) right_magnitude = b < 0 ? -(unsigned _BitInt(128))b : (unsigned _BitInt(128))b;
    const unsigned _BitInt(128) digit_mask = UINT64_MAX;
    const unsigned _BitInt(128) left_low = left_magnitude & digit_mask;
    const unsigned _BitInt(128) left_high = left_magnitude >> 64U;
    const unsigned _BitInt(128) right_low = right_magnitude & digit_mask;
    const unsigned _BitInt(128) right_high = right_magnitude >> 64U;
    const unsigned _BitInt(128) low_low = left_low * right_low;
    const unsigned _BitInt(128) low_high = left_low * right_high;
    const unsigned _BitInt(128) high_low = left_high * right_low;
    const unsigned _BitInt(128) high_high = left_high * right_high;
    const unsigned _BitInt(128) middle = (low_low >> 64U) + (low_high & digit_mask) + (high_low & digit_mask);
    const unsigned _BitInt(128) product_low = (low_low & digit_mask) | (middle << 64U);
    const unsigned _BitInt(128) product_high = high_high + (low_high >> 64U) + (high_low >> 64U) + (middle >> 64U);
    *out_value = type_i128_store(negative ? -product_low : product_low);
    return product_high != 0 || product_low > (unsigned _BitInt(128))NUMERIC_I128_MAX;
}

static inline int numeric_divide_i128(struct dora_i128 left, struct dora_i128 right, struct dora_i128* out_value) {
    const _BitInt(128) a = type_i128_load(left);
    const _BitInt(128) b = type_i128_load(right);
    *out_value = left;
    if (a == NUMERIC_I128_MIN && b == -1) {
        return NUMERIC_FAILURE_OVERFLOW;
    }
    if (b == 0) {
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    }
    *out_value = type_i128_store(a / b);
    return 0;
}

static inline int numeric_remainder_i128(struct dora_i128 left, struct dora_i128 right, struct dora_i128* out_value) {
    const _BitInt(128) a = type_i128_load(left);
    const _BitInt(128) b = type_i128_load(right);
    *out_value = left;
    if (a == NUMERIC_I128_MIN && b == -1) {
        return NUMERIC_FAILURE_OVERFLOW;
    }
    if (b == 0) {
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    }
    *out_value = type_i128_store(a % b);
    return 0;
}

static inline bool
numeric_multiply_checked_i256(struct dora_i256 left, struct dora_i256 right, struct dora_i256* out_value) {
    uint64_t left_magnitude[4] = {};
    uint64_t right_magnitude[4] = {};
    uint64_t product[8] = {};
    uint64_t limit[4] = {};
    const bool negative = dora_limbs_is_negative(left.limbs, 4) != dora_limbs_is_negative(right.limbs, 4);
    dora_limbs_magnitude(left.limbs, 4, left_magnitude);
    dora_limbs_magnitude(right.limbs, 4, right_magnitude);
    dora_limbs_multiply(
        (struct limbs_multiplication_input){.left = left_magnitude, .right = right_magnitude, .count = 4},
        product);
    for (size_t limb = 0; limb < 4; ++limb) {
        limit[limb] = negative ? 0 : UINT64_MAX;
    }
    limit[3] = negative ? UINT64_C(0x80'00'00'00'00'00'00'00) : UINT64_C(0x7f'ff'ff'ff'ff'ff'ff'ff);
    struct dora_i256 result = {};
    if (negative) {
        dora_limbs_negate(product, 4, result.limbs);
    } else {
        (memcpy)(result.limbs, product, sizeof(result.limbs));
    }
    *out_value = result;
    return !dora_limbs_is_zero(product + 4, 4) || dora_limbs_compare(product, limit, 4) > 0;
}

static inline int
numeric_divmod_i256(struct dora_i256 left, struct dora_i256 right, bool remainder, struct dora_i256* out_value) {
    *out_value = (struct dora_i256){};
    if (dora_limbs_is_zero(right.limbs, 4)) {
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    }
    if (!remainder && dora_limbs_is_minimum(left.limbs, 4) && dora_limbs_is_minus_one(right.limbs, 4)) {
        return NUMERIC_FAILURE_OVERFLOW;
    }
    uint64_t numerator[4] = {};
    uint64_t denominator[4] = {};
    uint64_t quotient[4] = {};
    uint64_t residual[4] = {};
    dora_limbs_magnitude(left.limbs, 4, numerator);
    dora_limbs_magnitude(right.limbs, 4, denominator);
    dora_limbs_divmod((struct limbs_division_input){.numerator = numerator, .denominator = denominator, .count = 4},
                      (struct limbs_division_output){.quotient = quotient, .remainder = residual});
    const uint64_t* const magnitude = remainder ? residual : quotient;
    const bool left_negative = dora_limbs_is_negative(left.limbs, 4);
    const bool right_negative = !remainder && dora_limbs_is_negative(right.limbs, 4);
    const bool negative = left_negative != right_negative;
    if (negative) {
        dora_limbs_negate(magnitude, 4, out_value->limbs);
    } else {
        (memcpy)(out_value->limbs, magnitude, sizeof(out_value->limbs));
    }
    return 0;
}

static inline int numeric_divide_i256(struct dora_i256 left, struct dora_i256 right, struct dora_i256* out_value) {
    return numeric_divmod_i256(left, right, false, out_value);
}

static inline int numeric_remainder_i256(struct dora_i256 left, struct dora_i256 right, struct dora_i256* out_value) {
    return numeric_divmod_i256(left, right, true, out_value);
}

//--------------------------------------
// Unsigned Integers
//--------------------------------------

static inline struct dora_u128 numeric_add_wrap_u128(struct dora_u128 left, struct dora_u128 right) {
    return type_u128_store(type_u128_load(left) + type_u128_load(right));
}

static inline struct dora_u128 numeric_subtract_wrap_u128(struct dora_u128 left, struct dora_u128 right) {
    return type_u128_store(type_u128_load(left) - type_u128_load(right));
}

static inline struct dora_u128 numeric_multiply_wrap_u128(struct dora_u128 left, struct dora_u128 right) {
    return type_u128_store(type_u128_load(left) * type_u128_load(right));
}

static inline struct dora_u128 numeric_negate_wrap_u128(struct dora_u128 value) {
    return type_u128_store(0U - type_u128_load(value));
}

static inline struct dora_u128 numeric_abs_wrap_u128(struct dora_u128 value) {
    return value;
}

static inline bool
numeric_add_checked_u128(struct dora_u128 left, struct dora_u128 right, struct dora_u128* out_value) {
    const unsigned _BitInt(128) a = type_u128_load(left);
    const unsigned _BitInt(128) b = type_u128_load(right);
    const unsigned _BitInt(128) r = a + b;
    *out_value = type_u128_store(r);
    return r < a;
}

static inline bool
numeric_subtract_checked_u128(struct dora_u128 left, struct dora_u128 right, struct dora_u128* out_value) {
    const unsigned _BitInt(128) a = type_u128_load(left);
    const unsigned _BitInt(128) b = type_u128_load(right);
    const unsigned _BitInt(128) r = a - b;
    *out_value = type_u128_store(r);
    return a < b;
}

static inline bool
numeric_multiply_checked_u128(struct dora_u128 left, struct dora_u128 right, struct dora_u128* out_value) {
    const bool high_overflow = left.limbs[1] != 0 && right.limbs[1] != 0;
    const unsigned _BitInt(128) lo = (unsigned _BitInt(128))left.limbs[0] * right.limbs[0];
    const unsigned _BitInt(128) c1 = (unsigned _BitInt(128))left.limbs[0] * right.limbs[1];
    const unsigned _BitInt(128) c2 = (unsigned _BitInt(128))left.limbs[1] * right.limbs[0];
    const unsigned _BitInt(128) cross = c1 + c2;
    const unsigned _BitInt(128) total = lo + (cross << 64U);
    *out_value = type_u128_store(total);
    return high_overflow || (cross >> 64U) != 0 || total < lo;
}

static inline bool numeric_negate_checked_u128(struct dora_u128 value, struct dora_u128* out_value) {
    const unsigned _BitInt(128) a = type_u128_load(value);
    *out_value = type_u128_store(0U - a);
    return a != 0;
}

static inline bool numeric_abs_checked_u128(struct dora_u128 value, struct dora_u128* out_value) {
    *out_value = value;
    return false;
}

static inline int numeric_divide_u128(struct dora_u128 left, struct dora_u128 right, struct dora_u128* out_value) {
    const unsigned _BitInt(128) b = type_u128_load(right);
    *out_value = left;
    if (b == 0) {
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    }
    *out_value = type_u128_store(type_u128_load(left) / b);
    return 0;
}

static inline int numeric_remainder_u128(struct dora_u128 left, struct dora_u128 right, struct dora_u128* out_value) {
    const unsigned _BitInt(128) b = type_u128_load(right);
    *out_value = left;
    if (b == 0) {
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    }
    *out_value = type_u128_store(type_u128_load(left) % b);
    return 0;
}

//------------------------------------------------------------------------------
//
// Floating Point
//
//------------------------------------------------------------------------------

//--------------------------------------
// Single and Double Precision
//--------------------------------------

#define NUMERIC_ARITHMETIC_FLOAT(X)                                                                                    \
    X(f32, float, fmodf, fabsf)                                                                                        \
    X(f64, double, fmod, fabs)

// Float32, Float64
#define NUMERIC_DEFINE_FLOAT(S, T, REMAINDER, ABSOLUTE)                                                                \
    static inline T numeric_add_##S(T left, T right) {                                                                 \
        return left + right;                                                                                           \
    }                                                                                                                  \
    static inline T numeric_subtract_##S(T left, T right) {                                                            \
        return left - right;                                                                                           \
    }                                                                                                                  \
    static inline T numeric_multiply_##S(T left, T right) {                                                            \
        return left * right;                                                                                           \
    }                                                                                                                  \
    static inline T numeric_divide_##S(T left, T right) {                                                              \
        return left / right;                                                                                           \
    }                                                                                                                  \
    static inline T numeric_remainder_##S(T left, T right) {                                                           \
        return REMAINDER(left, right);                                                                                 \
    }                                                                                                                  \
    static inline T numeric_negate_##S(T value) {                                                                      \
        return -value;                                                                                                 \
    }                                                                                                                  \
    static inline T numeric_abs_##S(T value) {                                                                         \
        return ABSOLUTE(value);                                                                                        \
    }
NUMERIC_ARITHMETIC_FLOAT(NUMERIC_DEFINE_FLOAT)
#undef NUMERIC_DEFINE_FLOAT
#undef NUMERIC_ARITHMETIC_FLOAT

//--------------------------------------
// Half Precision
//--------------------------------------

static inline struct dora_f16_bits numeric_add_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    return numeric_half_narrow(numeric_half_widen(left) + numeric_half_widen(right));
}

static inline struct dora_f16_bits numeric_subtract_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    return numeric_half_narrow(numeric_half_widen(left) - numeric_half_widen(right));
}

static inline struct dora_f16_bits numeric_multiply_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    return numeric_half_narrow(numeric_half_widen(left) * numeric_half_widen(right));
}

static inline struct dora_f16_bits numeric_divide_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    return numeric_half_narrow(numeric_half_widen(left) / numeric_half_widen(right));
}

static inline struct dora_f16_bits numeric_remainder_f16(struct dora_f16_bits left, struct dora_f16_bits right) {
    return numeric_half_narrow(fmodf(numeric_half_widen(left), numeric_half_widen(right)));
}

static inline struct dora_f16_bits numeric_negate_f16(struct dora_f16_bits value) {
    return (struct dora_f16_bits){.bits = (uint16_t)(value.bits ^ 0x8000U)};
}

static inline struct dora_f16_bits numeric_abs_f16(struct dora_f16_bits value) {
    return (struct dora_f16_bits){.bits = (uint16_t)(value.bits & 0x7fffU)};
}

int dora_numeric_store_integer(struct dora_i256 value, enum dora_physical_type physical, void* out_value);
int64_t dora_numeric_floor_div(int64_t value, int64_t divisor);

#endif
