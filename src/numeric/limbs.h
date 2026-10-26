// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Multiword integer arithmetic.

#ifndef DORA_NUMERIC_LIMBS_H
#define DORA_NUMERIC_LIMBS_H

#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Arithmetic Buffers
//
//------------------------------------------------------------------------------

struct limbs_span {
    uint64_t* values;
    size_t count;
};

struct limbs_affine_u32 {
    uint32_t factor;
    uint32_t addend;
};

struct limbs_multiplication_input {
    const uint64_t* left;
    const uint64_t* right;
    size_t count;
};

struct limbs_division_input {
    const uint64_t* numerator;
    const uint64_t* denominator;
    size_t count;
};

struct limbs_division_output {
    uint64_t* quotient;
    uint64_t* remainder;
};

//------------------------------------------------------------------------------
//
// Integer Arithmetic
//
//------------------------------------------------------------------------------

bool dora_limbs_is_zero(const uint64_t* value, size_t count);
int dora_limbs_mul_u32(struct limbs_span magnitude, struct limbs_affine_u32 affine);
uint32_t dora_limbs_div_u32(struct limbs_span magnitude, uint32_t divisor);
int dora_i256_from_mag(const uint64_t mag[4], bool neg, struct dora_i256* out_val);
void dora_i256_to_mag(struct dora_i256 val, uint64_t out_mag[4], bool* out_neg);
int dora_limbs_compare(const uint64_t* left, const uint64_t* right, size_t count);
bool dora_limbs_is_negative(const uint64_t* value, size_t count);
void dora_limbs_negate(const uint64_t* value, size_t count, uint64_t* out_value);
void dora_limbs_magnitude(const uint64_t* value, size_t count, uint64_t* out_value);
void dora_limbs_divmod(struct limbs_division_input input, struct limbs_division_output output);
void dora_limbs_add(const uint64_t* left, const uint64_t* right, size_t count, uint64_t* out_value);
void dora_limbs_subtract(const uint64_t* left, const uint64_t* right, size_t count, uint64_t* out_value);
void dora_limbs_multiply(struct limbs_multiplication_input input, uint64_t* out_product);
bool dora_limbs_is_minimum(const uint64_t* value, size_t count);
bool dora_limbs_is_minus_one(const uint64_t* value, size_t count);

#endif
