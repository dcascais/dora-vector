// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Decimal and floating-point conversion.

#ifndef DORA_NUMERIC_DECIMAL_FLOAT_H
#define DORA_NUMERIC_DECIMAL_FLOAT_H

#include "dora/vector/types.h"

#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Floating Point
//--------------------------------------

static constexpr uint32_t DECIMAL_FLOAT_DIGIT_CAPACITY = 768;

struct decimal_float_digits {
    uint32_t num_digits;
    int32_t decimal_point;
    bool was_truncated;
    uint8_t digits[DECIMAL_FLOAT_DIGIT_CAPACITY];
};

void dora_decimal_digits_trim(struct decimal_float_digits* value);
uint64_t dora_decimal_digits_round(const struct decimal_float_digits* value);
extern const double dora_decimal_double_powers[];

void dora_decimal_digits_shift_left(struct decimal_float_digits* value, uint32_t shift);
void dora_decimal_digits_shift_right(struct decimal_float_digits* value, uint32_t shift);

#define DECIMAL_BINARY_FORMATS(X)                                                                                      \
    X(binary, 52U, 1023)                                                                                               \
    X(binary32, 23U, 127)

#define DEFINE_DECIMAL_DIGITS_BINARY(NAME, MANTISSA, BIAS)                                                             \
    static inline uint64_t decimal_digits_to_##NAME(struct decimal_float_digits* value) {                              \
        if (value->num_digits == 0 || value->decimal_point < -324) {                                                   \
            return 0;                                                                                                  \
        }                                                                                                              \
        if (value->decimal_point >= 310) {                                                                             \
            return ((uint64_t)(2 * (BIAS) + 1) << (MANTISSA));                                                         \
        }                                                                                                              \
        static const uint8_t decimal_powers[19] =                                                                      \
            {0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 43, 46, 49, 53, 56, 59};                                  \
        int32_t exponent = 0;                                                                                          \
        while (value->decimal_point > 0) {                                                                             \
            uint32_t point = (uint32_t)value->decimal_point;                                                           \
            uint32_t shift = point < 19 ? decimal_powers[point] : 60;                                                  \
            dora_decimal_digits_shift_right(value, shift);                                                             \
            exponent += (int32_t)shift;                                                                                \
        }                                                                                                              \
        while (value->decimal_point <= 0) {                                                                            \
            uint32_t shift = 0;                                                                                        \
            if (value->decimal_point == 0) {                                                                           \
                if (value->digits[0] >= 5) {                                                                           \
                    break;                                                                                             \
                }                                                                                                      \
                shift = value->digits[0] < 2 ? 2 : 1;                                                                  \
            } else {                                                                                                   \
                uint32_t point = (uint32_t)-value->decimal_point;                                                      \
                shift = point < 19 ? decimal_powers[point] : 60;                                                       \
            }                                                                                                          \
            dora_decimal_digits_shift_left(value, shift);                                                              \
            exponent -= (int32_t)shift;                                                                                \
        }                                                                                                              \
        --exponent;                                                                                                    \
        while (exponent < (1 - (BIAS))) {                                                                              \
            uint32_t shift = (uint32_t)((1 - (BIAS)) - exponent);                                                      \
            if (shift > 60) {                                                                                          \
                shift = 60;                                                                                            \
            }                                                                                                          \
            dora_decimal_digits_shift_right(value, shift);                                                             \
            exponent += (int32_t)shift;                                                                                \
        }                                                                                                              \
        if (exponent + (BIAS) >= 2 * (BIAS) + 1) {                                                                     \
            return ((uint64_t)(2 * (BIAS) + 1) << (MANTISSA));                                                         \
        }                                                                                                              \
        dora_decimal_digits_shift_left(value, (MANTISSA) + 1);                                                         \
        uint64_t significand = dora_decimal_digits_round(value);                                                       \
        if (significand >= (UINT64_C(1) << ((MANTISSA) + 1U))) {                                                       \
            dora_decimal_digits_shift_right(value, 1);                                                                 \
            ++exponent;                                                                                                \
            significand = dora_decimal_digits_round(value);                                                            \
            if (exponent + (BIAS) >= 2 * (BIAS) + 1) {                                                                 \
                return ((uint64_t)(2 * (BIAS) + 1) << (MANTISSA));                                                     \
            }                                                                                                          \
        }                                                                                                              \
        int32_t encoded_exponent = exponent + (BIAS);                                                                  \
        if (significand < (UINT64_C(1) << (MANTISSA))) {                                                               \
            --encoded_exponent;                                                                                        \
        }                                                                                                              \
        return ((uint64_t)encoded_exponent << (MANTISSA)) | (significand & ((UINT64_C(1) << (MANTISSA)) - 1));         \
    }
DECIMAL_BINARY_FORMATS(DEFINE_DECIMAL_DIGITS_BINARY)
#undef DEFINE_DECIMAL_DIGITS_BINARY
#undef DECIMAL_BINARY_FORMATS

double dora_decimal_to_double_i64(int64_t value, int32_t scale);
double dora_decimal_to_double_i128(struct dora_i128 value, int32_t scale);
float dora_decimal_to_float_i64(int64_t value, int32_t scale);
float dora_decimal_to_float_i128(struct dora_i128 value, int32_t scale);

static constexpr int64_t DECIMAL_DOUBLE_EXACT = INT64_C(0x0020000000000000);
static constexpr int32_t DECIMAL_DOUBLE_FAST_SCALE = 22;

static inline double decimal_double_i64(int32_t scale, int64_t value) {
    if (scale == 0 || (scale > 0 && scale <= DECIMAL_DOUBLE_FAST_SCALE && value >= -DECIMAL_DOUBLE_EXACT &&
                       value <= DECIMAL_DOUBLE_EXACT)) {
        return (double)value / dora_decimal_double_powers[scale];
    }
    return dora_decimal_to_double_i64(value, scale);
}

static inline double decimal_double_i32(int32_t scale, int32_t value) {
    return decimal_double_i64(scale, value);
}

static inline double decimal_double_i128(int32_t scale, struct dora_i128 value) {
    _BitInt(128) number = 0;
    (memcpy)(&number, value.limbs, sizeof(number));
    if (scale == 0 || (scale > 0 && scale <= DECIMAL_DOUBLE_FAST_SCALE && number >= -DECIMAL_DOUBLE_EXACT &&
                       number <= DECIMAL_DOUBLE_EXACT)) {
        const int64_t upper = (int64_t)value.limbs[1];
        const double whole = upper == -1 ? -(double)(UINT64_MAX - value.limbs[0]) - 1
                                         : (double)value.limbs[0] + (double)upper * ((double)UINT64_MAX + 1);
        return whole / dora_decimal_double_powers[scale];
    }
    return dora_decimal_to_double_i128(value, scale);
}

#endif
