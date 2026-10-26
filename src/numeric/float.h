// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Floating-point representation and conversion.

#ifndef DORA_NUMERIC_FLOAT_H
#define DORA_NUMERIC_FLOAT_H

#include "dora/vector/types.h"
#include "numeric/limbs.h"

#include <float.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Kernels require strict floating-point semantics"
#endif

static inline uint32_t numeric_float_bits(float value) {
    uint32_t bits = 0;
    (memcpy)(&bits, &value, sizeof(bits));
    return bits;
}

static inline uint64_t numeric_double_bits(double value) {
    uint64_t bits = 0;
    (memcpy)(&bits, &value, sizeof(bits));
    return bits;
}

static inline bool numeric_f32_is_nan(float value) {
    return (numeric_float_bits(value) & 0x7f'ff'ff'ffU) > 0x7f'80'00'00U;
}

static inline bool numeric_f64_is_nan(double value) {
    return (numeric_double_bits(value) & UINT64_C(0x7f'ff'ff'ff'ff'ff'ff'ff)) > UINT64_C(0x7f'f0'00'00'00'00'00'00);
}

static inline uint32_t numeric_f32_key(float value) {
    const uint32_t bits = numeric_float_bits(value);
    const uint32_t magnitude = bits & 0x7f'ff'ff'ffU;
    return magnitude == 0 ? 0 : magnitude > 0x7f'80'00'00U ? 0x7f'c0'00'00U : bits;
}

static inline uint64_t numeric_f64_key(double value) {
    const uint64_t bits = numeric_double_bits(value);
    const uint64_t magnitude = bits & UINT64_C(0x7f'ff'ff'ff'ff'ff'ff'ff);
    return magnitude == 0                                    ? 0
           : magnitude > UINT64_C(0x7f'f0'00'00'00'00'00'00) ? UINT64_C(0x7f'f8'00'00'00'00'00'00)
                                                             : bits;
}

static inline float numeric_half_widen(struct dora_f16_bits value) {
    uint32_t sign = (uint32_t)(value.bits & 0x8000U) << 16U;
    uint32_t exponent = ((uint32_t)value.bits >> 10U) & 31U;
    uint32_t fraction = value.bits & 1023U;
    uint32_t bits = sign;
    if (exponent == 31) {
        bits |= 0x7f'80'00'00U | (fraction << 13U);
    } else if (exponent != 0) {
        bits |= ((exponent + 112) << 23U) | (fraction << 13U);
    } else if (fraction != 0) {
        exponent = 113;
        while ((fraction & 1024U) == 0) {
            fraction <<= 1U;
            --exponent;
        }
        bits |= (exponent << 23U) | ((fraction & 1023U) << 13U);
    }
    float result = 0;
    (memcpy)(&result, &bits, sizeof(result));
    return result;
}

static inline uint32_t numeric_round_shift_even(uint32_t value, unsigned int shift) {
    uint32_t quotient = value >> shift;
    uint32_t remainder = value & ((1U << shift) - 1);
    uint32_t half = 1U << (shift - 1);
    return quotient + (uint32_t)(remainder > half || (remainder == half && (quotient & 1U) != 0));
}

static inline struct dora_f16_bits numeric_half_narrow(float value) {
    uint32_t bits = numeric_float_bits(value);
    uint16_t sign = (uint16_t)((bits >> 16U) & 0x8000U);
    uint32_t exponent = (bits >> 23U) & 255U;
    uint32_t fraction = bits & 0x7f'ff'ffU;
    if (exponent == 255) {
        return (struct dora_f16_bits){.bits = (uint16_t)(sign | (fraction == 0 ? 0x7c00U : 0x7e00U))};
    }
    int unbiased = (int)exponent - 127;
    if (unbiased > 15) {
        return (struct dora_f16_bits){.bits = (uint16_t)(sign | 0x7c00U)};
    }
    if (unbiased < -25) {
        return (struct dora_f16_bits){.bits = sign};
    }
    uint32_t significand = fraction | 0x80'00'00U;
    uint32_t result = 0;
    if (unbiased < -14) {
        result = numeric_round_shift_even(significand, (unsigned int)(-unbiased - 1));
    } else {
        uint32_t rounded = numeric_round_shift_even(significand, 13);
        result = ((uint32_t)(unbiased + 15) << 10U) + rounded - 1024;
    }
    return (struct dora_f16_bits){.bits = (uint16_t)(sign | result)};
}

static inline double numeric_i256_to_double(struct dora_i256 value) {
    uint64_t magnitude[4] = {};
    bool negative = false;
    dora_i256_to_mag(value, magnitude, &negative);
    double result = 0;
    for (uint32_t index = 4; index != 0; --index) {
        result = result * 0x1p64 + (double)magnitude[index - 1];
    }
    return negative ? -result : result;
}

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && FLT_MIN_EXP == -125);
static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021);
static_assert(FLT_EVAL_METHOD == 0);
static_assert(FLT_HAS_SUBNORM == 1);
static_assert(DBL_HAS_SUBNORM == 1);

#endif
