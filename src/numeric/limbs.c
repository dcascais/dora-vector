// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/limbs.h"

#include "dora/vector/types.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Integer Arithmetic
//
//------------------------------------------------------------------------------

int dora_limbs_compare(const uint64_t* left, const uint64_t* right, size_t count) {
    for (size_t i = count; i != 0; --i) {
        if (left[i - 1] != right[i - 1]) {
            return left[i - 1] < right[i - 1] ? -1 : 1;
        }
    }
    return 0;
}

bool dora_limbs_is_zero(const uint64_t* value, size_t count) {
    uint64_t combined = 0;
    for (size_t i = 0; i < count; ++i) {
        combined |= value[i];
    }
    return combined == 0;
}

bool dora_limbs_is_negative(const uint64_t* value, size_t count) {
    return (value[count - 1] >> 63U) != 0;
}

void dora_limbs_add(const uint64_t* left, const uint64_t* right, size_t count, uint64_t* out_value) {
    uint64_t carry = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t sum = left[i] + right[i];
        uint64_t next = (uint64_t)(sum < left[i]);
        uint64_t result = sum + carry;
        carry = next | (uint64_t)(result < sum);
        out_value[i] = result;
    }
}

void dora_limbs_subtract(const uint64_t* left, const uint64_t* right, size_t count, uint64_t* out_value) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t difference = left[i] - right[i];
        uint64_t next = (uint64_t)(left[i] < right[i]);
        uint64_t result = difference - borrow;
        borrow = next | (uint64_t)(difference < borrow);
        out_value[i] = result;
    }
}

void dora_limbs_negate(const uint64_t* value, size_t count, uint64_t* out_value) {
    uint64_t carry = 1;
    for (size_t i = 0; i < count; ++i) {
        uint64_t inverted = ~value[i];
        uint64_t result = inverted + carry;
        carry = (uint64_t)(result < inverted);
        out_value[i] = result;
    }
}

void dora_limbs_magnitude(const uint64_t* value, size_t count, uint64_t* out_value) {
    if (dora_limbs_is_negative(value, count)) {
        dora_limbs_negate(value, count, out_value);
    } else {
        memcpy(out_value, value, count * sizeof(*value));
    }
}

int dora_limbs_mul_u32(struct limbs_span magnitude, struct limbs_affine_u32 affine) {
    uint64_t carry = affine.addend;
    for (size_t i = 0; i < magnitude.count; ++i) {
        uint64_t lo = (magnitude.values[i] & UINT32_MAX) * affine.factor + (carry & UINT32_MAX);
        uint64_t hi = (magnitude.values[i] >> 32U) * affine.factor + (carry >> 32U) + (lo >> 32U);
        magnitude.values[i] = (lo & UINT32_MAX) | (hi << 32U);
        carry = hi >> 32U;
    }
    return carry == 0 ? 0 : ERANGE;
}

uint32_t dora_limbs_div_u32(struct limbs_span magnitude, uint32_t divisor) {
    uint64_t rem = 0;
    for (size_t i = magnitude.count; i != 0; --i) {
        uint64_t hi_num = (rem << 32U) | (magnitude.values[i - 1] >> 32U);
        uint64_t q_hi = hi_num / divisor;
        rem = hi_num % divisor;
        uint64_t lo_num = (rem << 32U) | (magnitude.values[i - 1] & UINT32_MAX);
        uint64_t q_lo = lo_num / divisor;
        rem = lo_num % divisor;
        magnitude.values[i - 1] = (q_hi << 32U) | q_lo;
    }
    return (uint32_t)rem;
}

int dora_i256_from_mag(const uint64_t mag[4], bool neg, struct dora_i256* out_val) {
    if (!neg) {
        if (dora_limbs_is_negative(mag, 4)) {
            return ERANGE;
        }
        for (size_t i = 0; i < 4; ++i) {
            out_val->limbs[i] = mag[i];
        }
        return 0;
    }
    if (dora_limbs_is_negative(mag, 4)) {
        if (mag[3] != (1ULL << 63U) || mag[0] != 0 || mag[1] != 0 || mag[2] != 0) {
            return ERANGE;
        }
    }
    dora_limbs_negate(mag, 4, out_val->limbs);
    return 0;
}

void dora_i256_to_mag(struct dora_i256 val, uint64_t out_mag[4], bool* out_neg) {
    *out_neg = dora_limbs_is_negative(val.limbs, 4);
    dora_limbs_magnitude(val.limbs, 4, out_mag);
}

void dora_limbs_multiply(struct limbs_multiplication_input input, uint64_t* out_product) {
    uint32_t product[16] = {};
    size_t digits = input.count * 2;
    for (size_t i = 0; i < digits; ++i) {
        uint64_t left_digit = (input.left[i / 2] >> ((i % 2) * 32)) & UINT32_MAX;
        uint64_t carry = 0;
        for (size_t j = 0; j < digits; ++j) {
            uint64_t right_digit = (input.right[j / 2] >> ((j % 2) * 32)) & UINT32_MAX;
            uint64_t total = left_digit * right_digit + product[i + j] + carry;
            product[i + j] = (uint32_t)total;
            carry = total >> 32U;
        }
        product[i + digits] = (uint32_t)carry;
    }
    for (size_t i = 0; i < input.count * 2; ++i) {
        out_product[i] = product[2 * i] | ((uint64_t)product[2 * i + 1] << 32U);
    }
}

void dora_limbs_divmod(struct limbs_division_input input, struct limbs_division_output output) {
    for (size_t bit = input.count * 64; bit != 0; --bit) {
        size_t index = bit - 1;
        uint64_t carry = (input.numerator[index / 64] >> (index % 64)) & 1U;
        for (size_t i = 0; i < input.count; ++i) {
            uint64_t next = output.remainder[i] >> 63U;
            output.remainder[i] = (output.remainder[i] << 1U) | carry;
            carry = next;
        }
        if (carry != 0 || dora_limbs_compare(output.remainder, input.denominator, input.count) >= 0) {
            dora_limbs_subtract(output.remainder, input.denominator, input.count, output.remainder);
            output.quotient[index / 64] |= 1ULL << (index % 64);
        }
    }
}

bool dora_limbs_is_minimum(const uint64_t* value, size_t count) {
    return value[count - 1] == (1ULL << 63U) && dora_limbs_is_zero(value, count - 1);
}

bool dora_limbs_is_minus_one(const uint64_t* value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (value[i] != UINT64_MAX) {
            return false;
        }
    }
    return true;
}
