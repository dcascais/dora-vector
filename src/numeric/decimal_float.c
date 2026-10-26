// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/decimal_float.h"

#include "dora/vector/types.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <float.h>
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

static constexpr uint32_t DECIMAL_FLOAT_MAX_DIGIT_WITHOUT_OVERFLOW = 19;
static constexpr int32_t DECIMAL_FLOAT_DECIMAL_POINT_RANGE = 2047;

const double dora_decimal_double_powers[DECIMAL256_MAX_PRECISION + 1] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30, 1e31,
    1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39, 1e40, 1e41, 1e42, 1e43, 1e44, 1e45, 1e46, 1e47,
    1e48, 1e49, 1e50, 1e51, 1e52, 1e53, 1e54, 1e55, 1e56, 1e57, 1e58, 1e59, 1e60, 1e61, 1e62, 1e63,
    1e64, 1e65, 1e66, 1e67, 1e68, 1e69, 1e70, 1e71, 1e72, 1e73, 1e74, 1e75, 1e76,
};

void dora_decimal_digits_trim(struct decimal_float_digits* value) {
    while (value->num_digits > 0 && value->digits[value->num_digits - 1] == 0) {
        value->num_digits--;
    }
}

static uint32_t decimal_float_left_shift_digits(const struct decimal_float_digits* h, uint32_t shift) {
    shift &= 63U;
    static const uint16_t number_of_digits_decimal_left_shift_table[65] = {
        0x0000, 0x0800, 0x0801, 0x0803, 0x1006, 0x1009, 0x100D, 0x1812, 0x1817, 0x181D, 0x2024, 0x202B, 0x2033,
        0x203C, 0x2846, 0x2850, 0x285B, 0x3067, 0x3073, 0x3080, 0x388E, 0x389C, 0x38AB, 0x38BB, 0x40CC, 0x40DD,
        0x40EF, 0x4902, 0x4915, 0x4929, 0x513E, 0x5153, 0x5169, 0x5180, 0x5998, 0x59B0, 0x59C9, 0x61E3, 0x61FD,
        0x6218, 0x6A34, 0x6A50, 0x6A6D, 0x6A8B, 0x72AA, 0x72C9, 0x72E9, 0x7B0A, 0x7B2B, 0x7B4D, 0x8370, 0x8393,
        0x83B7, 0x83DC, 0x8C02, 0x8C28, 0x8C4F, 0x9477, 0x949F, 0x94C8, 0x9CF2, 0x051C, 0x051C, 0x051C, 0x051C,
    };
    const uint32_t x_a = number_of_digits_decimal_left_shift_table[shift];
    const uint32_t x_b = number_of_digits_decimal_left_shift_table[shift + 1];
    const uint32_t num_new_digits = x_a >> 11U;
    const uint32_t pow5_a = 0x7FFU & x_a;
    const uint32_t pow5_b = 0x7FFU & x_b;
    static const uint8_t number_of_digits_decimal_left_shift_table_powers_of_5[0x051C] = {
        5, 2, 5, 1, 2, 5, 6, 2, 5, 3, 1, 2, 5, 1, 5, 6, 2, 5, 7, 8, 1, 2, 5, 3, 9, 0, 6, 2, 5, 1, 9, 5, 3, 1, 2, 5, 9,
        7, 6, 5, 6, 2, 5, 4, 8, 8, 2, 8, 1, 2, 5, 2, 4, 4, 1, 4, 0, 6, 2, 5, 1, 2, 2, 0, 7, 0, 3, 1, 2, 5, 6, 1, 0, 3,
        5, 1, 5, 6, 2, 5, 3, 0, 5, 1, 7, 5, 7, 8, 1, 2, 5, 1, 5, 2, 5, 8, 7, 8, 9, 0, 6, 2, 5, 7, 6, 2, 9, 3, 9, 4, 5,
        3, 1, 2, 5, 3, 8, 1, 4, 6, 9, 7, 2, 6, 5, 6, 2, 5, 1, 9, 0, 7, 3, 4, 8, 6, 3, 2, 8, 1, 2, 5, 9, 5, 3, 6, 7, 4,
        3, 1, 6, 4, 0, 6, 2, 5, 4, 7, 6, 8, 3, 7, 1, 5, 8, 2, 0, 3, 1, 2, 5, 2, 3, 8, 4, 1, 8, 5, 7, 9, 1, 0, 1, 5, 6,
        2, 5, 1, 1, 9, 2, 0, 9, 2, 8, 9, 5, 5, 0, 7, 8, 1, 2, 5, 5, 9, 6, 0, 4, 6, 4, 4, 7, 7, 5, 3, 9, 0, 6, 2, 5, 2,
        9, 8, 0, 2, 3, 2, 2, 3, 8, 7, 6, 9, 5, 3, 1, 2, 5, 1, 4, 9, 0, 1, 1, 6, 1, 1, 9, 3, 8, 4, 7, 6, 5, 6, 2, 5, 7,
        4, 5, 0, 5, 8, 0, 5, 9, 6, 9, 2, 3, 8, 2, 8, 1, 2, 5, 3, 7, 2, 5, 2, 9, 0, 2, 9, 8, 4, 6, 1, 9, 1, 4, 0, 6, 2,
        5, 1, 8, 6, 2, 6, 4, 5, 1, 4, 9, 2, 3, 0, 9, 5, 7, 0, 3, 1, 2, 5, 9, 3, 1, 3, 2, 2, 5, 7, 4, 6, 1, 5, 4, 7, 8,
        5, 1, 5, 6, 2, 5, 4, 6, 5, 6, 6, 1, 2, 8, 7, 3, 0, 7, 7, 3, 9, 2, 5, 7, 8, 1, 2, 5, 2, 3, 2, 8, 3, 0, 6, 4, 3,
        6, 5, 3, 8, 6, 9, 6, 2, 8, 9, 0, 6, 2, 5, 1, 1, 6, 4, 1, 5, 3, 2, 1, 8, 2, 6, 9, 3, 4, 8, 1, 4, 4, 5, 3, 1, 2,
        5, 5, 8, 2, 0, 7, 6, 6, 0, 9, 1, 3, 4, 6, 7, 4, 0, 7, 2, 2, 6, 5, 6, 2, 5, 2, 9, 1, 0, 3, 8, 3, 0, 4, 5, 6, 7,
        3, 3, 7, 0, 3, 6, 1, 3, 2, 8, 1, 2, 5, 1, 4, 5, 5, 1, 9, 1, 5, 2, 2, 8, 3, 6, 6, 8, 5, 1, 8, 0, 6, 6, 4, 0, 6,
        2, 5, 7, 2, 7, 5, 9, 5, 7, 6, 1, 4, 1, 8, 3, 4, 2, 5, 9, 0, 3, 3, 2, 0, 3, 1, 2, 5, 3, 6, 3, 7, 9, 7, 8, 8, 0,
        7, 0, 9, 1, 7, 1, 2, 9, 5, 1, 6, 6, 0, 1, 5, 6, 2, 5, 1, 8, 1, 8, 9, 8, 9, 4, 0, 3, 5, 4, 5, 8, 5, 6, 4, 7, 5,
        8, 3, 0, 0, 7, 8, 1, 2, 5, 9, 0, 9, 4, 9, 4, 7, 0, 1, 7, 7, 2, 9, 2, 8, 2, 3, 7, 9, 1, 5, 0, 3, 9, 0, 6, 2, 5,
        4, 5, 4, 7, 4, 7, 3, 5, 0, 8, 8, 6, 4, 6, 4, 1, 1, 8, 9, 5, 7, 5, 1, 9, 5, 3, 1, 2, 5, 2, 2, 7, 3, 7, 3, 6, 7,
        5, 4, 4, 3, 2, 3, 2, 0, 5, 9, 4, 7, 8, 7, 5, 9, 7, 6, 5, 6, 2, 5, 1, 1, 3, 6, 8, 6, 8, 3, 7, 7, 2, 1, 6, 1, 6,
        0, 2, 9, 7, 3, 9, 3, 7, 9, 8, 8, 2, 8, 1, 2, 5, 5, 6, 8, 4, 3, 4, 1, 8, 8, 6, 0, 8, 0, 8, 0, 1, 4, 8, 6, 9, 6,
        8, 9, 9, 4, 1, 4, 0, 6, 2, 5, 2, 8, 4, 2, 1, 7, 0, 9, 4, 3, 0, 4, 0, 4, 0, 0, 7, 4, 3, 4, 8, 4, 4, 9, 7, 0, 7,
        0, 3, 1, 2, 5, 1, 4, 2, 1, 0, 8, 5, 4, 7, 1, 5, 2, 0, 2, 0, 0, 3, 7, 1, 7, 4, 2, 2, 4, 8, 5, 3, 5, 1, 5, 6, 2,
        5, 7, 1, 0, 5, 4, 2, 7, 3, 5, 7, 6, 0, 1, 0, 0, 1, 8, 5, 8, 7, 1, 1, 2, 4, 2, 6, 7, 5, 7, 8, 1, 2, 5, 3, 5, 5,
        2, 7, 1, 3, 6, 7, 8, 8, 0, 0, 5, 0, 0, 9, 2, 9, 3, 5, 5, 6, 2, 1, 3, 3, 7, 8, 9, 0, 6, 2, 5, 1, 7, 7, 6, 3, 5,
        6, 8, 3, 9, 4, 0, 0, 2, 5, 0, 4, 6, 4, 6, 7, 7, 8, 1, 0, 6, 6, 8, 9, 4, 5, 3, 1, 2, 5, 8, 8, 8, 1, 7, 8, 4, 1,
        9, 7, 0, 0, 1, 2, 5, 2, 3, 2, 3, 3, 8, 9, 0, 5, 3, 3, 4, 4, 7, 2, 6, 5, 6, 2, 5, 4, 4, 4, 0, 8, 9, 2, 0, 9, 8,
        5, 0, 0, 6, 2, 6, 1, 6, 1, 6, 9, 4, 5, 2, 6, 6, 7, 2, 3, 6, 3, 2, 8, 1, 2, 5, 2, 2, 2, 0, 4, 4, 6, 0, 4, 9, 2,
        5, 0, 3, 1, 3, 0, 8, 0, 8, 4, 7, 2, 6, 3, 3, 3, 6, 1, 8, 1, 6, 4, 0, 6, 2, 5, 1, 1, 1, 0, 2, 2, 3, 0, 2, 4, 6,
        2, 5, 1, 5, 6, 5, 4, 0, 4, 2, 3, 6, 3, 1, 6, 6, 8, 0, 9, 0, 8, 2, 0, 3, 1, 2, 5, 5, 5, 5, 1, 1, 1, 5, 1, 2, 3,
        1, 2, 5, 7, 8, 2, 7, 0, 2, 1, 1, 8, 1, 5, 8, 3, 4, 0, 4, 5, 4, 1, 0, 1, 5, 6, 2, 5, 2, 7, 7, 5, 5, 5, 7, 5, 6,
        1, 5, 6, 2, 8, 9, 1, 3, 5, 1, 0, 5, 9, 0, 7, 9, 1, 7, 0, 2, 2, 7, 0, 5, 0, 7, 8, 1, 2, 5, 1, 3, 8, 7, 7, 7, 8,
        7, 8, 0, 7, 8, 1, 4, 4, 5, 6, 7, 5, 5, 2, 9, 5, 3, 9, 5, 8, 5, 1, 1, 3, 5, 2, 5, 3, 9, 0, 6, 2, 5, 6, 9, 3, 8,
        8, 9, 3, 9, 0, 3, 9, 0, 7, 2, 2, 8, 3, 7, 7, 6, 4, 7, 6, 9, 7, 9, 2, 5, 5, 6, 7, 6, 2, 6, 9, 5, 3, 1, 2, 5, 3,
        4, 6, 9, 4, 4, 6, 9, 5, 1, 9, 5, 3, 6, 1, 4, 1, 8, 8, 8, 2, 3, 8, 4, 8, 9, 6, 2, 7, 8, 3, 8, 1, 3, 4, 7, 6, 5,
        6, 2, 5, 1, 7, 3, 4, 7, 2, 3, 4, 7, 5, 9, 7, 6, 8, 0, 7, 0, 9, 4, 4, 1, 1, 9, 2, 4, 4, 8, 1, 3, 9, 1, 9, 0, 6,
        7, 3, 8, 2, 8, 1, 2, 5, 8, 6, 7, 3, 6, 1, 7, 3, 7, 9, 8, 8, 4, 0, 3, 5, 4, 7, 2, 0, 5, 9, 6, 2, 2, 4, 0, 6, 9,
        5, 9, 5, 3, 3, 6, 9, 1, 4, 0, 6, 2, 5,
    };
    const uint8_t* pow5 = &number_of_digits_decimal_left_shift_table_powers_of_5[pow5_a];
    const uint32_t n = pow5_b - pow5_a;
    for (uint32_t i = 0; i < n; i++) {
        if (i >= h->num_digits) {
            return num_new_digits - 1;
        }
        if (h->digits[i] == pow5[i]) {
            continue;
        }
        if (h->digits[i] < pow5[i]) {
            return num_new_digits - 1;
        }
        return num_new_digits;
    }
    return num_new_digits;
}

uint64_t dora_decimal_digits_round(const struct decimal_float_digits* value) {
    if (value->num_digits == 0 || value->decimal_point < 0) {
        return 0;
    }
    if (value->decimal_point > 18) {
        return UINT64_MAX;
    }
    const uint32_t dp = (uint32_t)value->decimal_point;
    uint64_t n = 0;
    for (uint32_t i = 0; i < dp; i++) {
        n = (10 * n) + (i < value->num_digits ? value->digits[i] : 0U);
    }
    bool round_up = false;
    if (dp < value->num_digits) {
        round_up = value->digits[dp] >= 5;
        if (value->digits[dp] == 5 && dp + 1 == value->num_digits) {
            round_up = value->was_truncated || (dp > 0 && (1U & value->digits[dp - 1]) != 0);
        }
    }
    if (round_up) {
        n++;
    }
    return n;
}

void dora_decimal_digits_shift_left(struct decimal_float_digits* value, uint32_t shift) {
    if (value->num_digits == 0) {
        return;
    }
    const uint32_t num_new_digits = decimal_float_left_shift_digits(value, shift);
    int32_t read_index = (int32_t)(value->num_digits - 1);
    uint32_t write_index = value->num_digits - 1 + num_new_digits;
    uint64_t n = 0;
    while (read_index >= 0) {
        n += (uint64_t)value->digits[read_index] << shift;
        const uint64_t quotient = n / 10;
        const uint64_t remainder = n - (10 * quotient);
        if (write_index < DECIMAL_FLOAT_DIGIT_CAPACITY) {
            value->digits[write_index] = (uint8_t)remainder;
        } else if (remainder > 0) {
            value->was_truncated = true;
        }
        n = quotient;
        write_index--;
        read_index--;
    }
    while (n > 0) {
        const uint64_t quotient = n / 10;
        const uint64_t remainder = n - (10 * quotient);
        if (write_index < DECIMAL_FLOAT_DIGIT_CAPACITY) {
            value->digits[write_index] = (uint8_t)remainder;
        } else if (remainder > 0) {
            value->was_truncated = true;
        }
        n = quotient;
        write_index--;
    }
    value->num_digits += num_new_digits;
    if (value->num_digits > DECIMAL_FLOAT_DIGIT_CAPACITY) {
        value->num_digits = DECIMAL_FLOAT_DIGIT_CAPACITY;
    }
    value->decimal_point += (int32_t)num_new_digits;
    dora_decimal_digits_trim(value);
}

void dora_decimal_digits_shift_right(struct decimal_float_digits* value, uint32_t shift) {
    uint32_t read_index = 0;
    uint32_t write_index = 0;
    uint64_t n = 0;
    while ((n >> shift) == 0) {
        if (read_index < value->num_digits) {
            n = (10 * n) + value->digits[read_index++];
        } else if (n == 0) {
            return;
        } else {
            while ((n >> shift) == 0) {
                n = 10 * n;
                read_index++;
            }
            break;
        }
    }
    value->decimal_point -= (int32_t)(read_index - 1);
    if (value->decimal_point < -DECIMAL_FLOAT_DECIMAL_POINT_RANGE) {
        value->num_digits = 0;
        value->decimal_point = 0;
        value->was_truncated = false;
        return;
    }
    const uint64_t mask = (UINT64_C(1) << shift) - 1;
    while (read_index < value->num_digits) {
        const uint8_t new_digit = (uint8_t)(n >> shift);
        n = (10 * (n & mask)) + value->digits[read_index++];
        value->digits[write_index++] = new_digit;
    }
    while (n > 0) {
        const uint8_t new_digit = (uint8_t)(n >> shift);
        n = 10 * (n & mask);
        if (write_index < DECIMAL_FLOAT_DIGIT_CAPACITY) {
            value->digits[write_index++] = new_digit;
        } else if (new_digit > 0) {
            value->was_truncated = true;
        }
    }
    value->num_digits = write_index;
    dora_decimal_digits_trim(value);
}

#define DECIMAL_COEFFICIENTS(X)                                                                                        \
    X(i64, int64_t, uint64_t, 20)                                                                                      \
    X(i128, struct dora_i128, unsigned _BitInt(128), 40)

#define DECIMAL_COEFFICIENT_LOAD_i64 const int64_t number = value
#define DECIMAL_COEFFICIENT_LOAD_i128 auto const number = type_i128_load(value)

// Decimal (64/128-bit)
#define DEFINE_DECIMAL_DIGITS(S, T, U, CAPACITY)                                                                       \
    static bool decimal_digits_##S(T value, struct decimal_float_digits* decimal, int32_t scale) {                     \
        DECIMAL_COEFFICIENT_LOAD_##S;                                                                                  \
        const bool negative = number < 0;                                                                              \
        U magnitude = negative ? -(U)number : (U)number;                                                               \
        uint8_t digits[CAPACITY];                                                                                      \
        while (magnitude > 0) {                                                                                        \
            digits[decimal->num_digits++] = (uint8_t)(magnitude % 10U);                                                \
            magnitude /= 10U;                                                                                          \
        }                                                                                                              \
        for (uint32_t i = 0; i < decimal->num_digits; i++) {                                                           \
            decimal->digits[i] = digits[decimal->num_digits - i - 1];                                                  \
        }                                                                                                              \
        const int64_t point = (int64_t)decimal->num_digits - scale;                                                    \
        decimal->decimal_point =                                                                                       \
            point > DECIMAL_FLOAT_DECIMAL_POINT_RANGE ? DECIMAL_FLOAT_DECIMAL_POINT_RANGE : (int32_t)point;            \
        dora_decimal_digits_trim(decimal);                                                                             \
        for (uint32_t i = decimal->num_digits; i < DECIMAL_FLOAT_MAX_DIGIT_WITHOUT_OVERFLOW; i++) {                    \
            decimal->digits[i] = 0;                                                                                    \
        }                                                                                                              \
        return negative;                                                                                               \
    }
DECIMAL_COEFFICIENTS(DEFINE_DECIMAL_DIGITS)
#undef DEFINE_DECIMAL_DIGITS

// Decimal (64/128-bit) to Float64
#define DEFINE_DECIMAL_DOUBLE(S, T, U, CAPACITY)                                                                       \
    double dora_decimal_to_double_##S(T value, int32_t scale) {                                                        \
        struct decimal_float_digits decimal = {};                                                                      \
        const bool negative = decimal_digits_##S(value, &decimal, scale);                                              \
        const uint64_t bits = decimal_digits_to_binary(&decimal) | ((uint64_t)negative << 63U);                        \
        double result = 0;                                                                                             \
        (memcpy)(&result, &bits, sizeof(result));                                                                      \
        return result;                                                                                                 \
    }
DECIMAL_COEFFICIENTS(DEFINE_DECIMAL_DOUBLE)
#undef DEFINE_DECIMAL_DOUBLE

#if FLT_EVAL_METHOD == 0 || FLT_EVAL_METHOD == 1
static constexpr int32_t DECIMAL_FLOAT_FAST_SCALE = 10;
#else
static constexpr int32_t DECIMAL_FLOAT_FAST_SCALE = 0;
#endif

#define DECIMAL_FLOAT_INTEGER_i64(VALUE) ((float)(VALUE))
// Scale-zero Int128 conversion preserves binary64 intermediate rounding.
#define DECIMAL_FLOAT_INTEGER_i128(VALUE) ((float)decimal_double_i128(0, VALUE))

// Decimal (64/128-bit) to Float32
#define DEFINE_DECIMAL_FLOAT(S, T, U, CAPACITY)                                                                        \
    float dora_decimal_to_float_##S(T value, int32_t scale) {                                                          \
        DECIMAL_COEFFICIENT_LOAD_##S;                                                                                  \
        if (scale == 0 ||                                                                                              \
            (scale > 0 && scale <= DECIMAL_FLOAT_FAST_SCALE && number >= -0x1000000 && number <= 0x1000000)) {         \
            return DECIMAL_FLOAT_INTEGER_##S(value) / (float)dora_decimal_double_powers[scale];                        \
        }                                                                                                              \
        struct decimal_float_digits decimal = {};                                                                      \
        const bool negative = decimal_digits_##S(value, &decimal, scale);                                              \
        const uint32_t bits = (uint32_t)decimal_digits_to_binary32(&decimal) | ((uint32_t)negative << 31U);            \
        float result = 0;                                                                                              \
        (memcpy)(&result, &bits, sizeof(result));                                                                      \
        return result;                                                                                                 \
    }
DECIMAL_COEFFICIENTS(DEFINE_DECIMAL_FLOAT)
#undef DEFINE_DECIMAL_FLOAT
#undef DECIMAL_FLOAT_INTEGER_i128
#undef DECIMAL_FLOAT_INTEGER_i64
#undef DECIMAL_COEFFICIENT_LOAD_i128
#undef DECIMAL_COEFFICIENT_LOAD_i64
#undef DECIMAL_COEFFICIENTS
