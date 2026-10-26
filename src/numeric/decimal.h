// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Decimal arithmetic and rescaling.

#ifndef DORA_NUMERIC_DECIMAL_H
#define DORA_NUMERIC_DECIMAL_H

#include "dora/vector/types.h"

#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Decimal Coefficients
//
//------------------------------------------------------------------------------

struct decimal_unsigned {
    uint64_t limbs[8];
};

struct decimal_signed {
    struct decimal_unsigned magnitude;
    bool is_negative;
};

//------------------------------------------------------------------------------
//
// Decimal Arithmetic
//
//------------------------------------------------------------------------------

//--------------------------------------
// Unsigned Magnitudes
//--------------------------------------

int dora_decimal_unsigned_compare(struct decimal_unsigned left, struct decimal_unsigned right);

//--------------------------------------
// Signed Coefficients
//--------------------------------------

struct decimal_signed dora_decimal_signed_i256(struct dora_i256 value);
struct dora_i256 dora_decimal_i256(struct decimal_signed value);

//--------------------------------------
// Exact Arithmetic
//--------------------------------------

int dora_decimal_multiply_exact(struct decimal_signed left,
                                struct decimal_signed right,
                                struct dora_decimal_type type,
                                struct decimal_signed* out_value);
int dora_decimal_divide_exact(struct decimal_signed left,
                              struct decimal_signed right,
                              struct dora_decimal_type type,
                              struct decimal_signed* out_value);
int dora_decimal_remainder_exact(struct decimal_signed left,
                                 struct decimal_signed right,
                                 struct decimal_signed* out_value);

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Decimals
//--------------------------------------

int dora_decimal_bound_prepare(struct dora_decimal_type type, int32_t maximum, struct dora_i256* out_bound);
int dora_decimal_bound_prepare_width(struct dora_decimal_type type, uint32_t width_bits, struct dora_i256* out_bound);

static inline bool decimal_bound_admits_i32(int32_t value, struct dora_i256 bound) {
    uint32_t bits = 0;
    (memcpy)(&bits, &value, sizeof(bits));
    const uint32_t sign_mask = UINT32_MAX * (bits >> 31U);
    const uint32_t magnitude = (bits ^ sign_mask) + (sign_mask & 1U);
    return magnitude <= bound.limbs[0];
}

static inline bool decimal_bound_admits_i64(int64_t value, struct dora_i256 bound) {
    uint64_t bits = 0;
    (memcpy)(&bits, &value, sizeof(bits));
    const uint64_t sign_mask = UINT64_MAX * (bits >> 63U);
    const uint64_t magnitude = (bits ^ sign_mask) + (sign_mask & 1U);
    return magnitude <= bound.limbs[0];
}

static inline bool decimal_bound_admits_i128(struct dora_i128 value, struct dora_i256 bound) {
    const uint64_t sign_mask = UINT64_MAX * (value.limbs[1] >> 63U);
    const uint64_t inverted_0 = value.limbs[0] ^ sign_mask;
    const uint64_t inverted_1 = value.limbs[1] ^ sign_mask;
    const uint64_t magnitude_0 = inverted_0 + (sign_mask & 1U);
    const uint64_t magnitude_1 = inverted_1 + (uint64_t)(magnitude_0 < inverted_0);
    const bool high_less = magnitude_1 < bound.limbs[1];
    const bool high_equal = magnitude_1 == bound.limbs[1];
    const bool low_less_equal = magnitude_0 <= bound.limbs[0];
    return high_less || (high_equal && low_less_equal);
}

static inline bool decimal_bound_admits_i256(struct dora_i256 value, struct dora_i256 bound) {
    const uint64_t sign_mask = UINT64_MAX * (value.limbs[3] >> 63U);
    const uint64_t inverted_0 = value.limbs[0] ^ sign_mask;
    const uint64_t inverted_1 = value.limbs[1] ^ sign_mask;
    const uint64_t inverted_2 = value.limbs[2] ^ sign_mask;
    const uint64_t inverted_3 = value.limbs[3] ^ sign_mask;
    const uint64_t magnitude_0 = inverted_0 + (sign_mask & 1U);
    const uint64_t magnitude_1 = inverted_1 + (uint64_t)(magnitude_0 < inverted_0);
    const uint64_t magnitude_2 = inverted_2 + (uint64_t)(magnitude_1 < inverted_1);
    const uint64_t magnitude_3 = inverted_3 + (uint64_t)(magnitude_2 < inverted_2);
    const bool less_3 = magnitude_3 < bound.limbs[3];
    const bool equal_3 = magnitude_3 == bound.limbs[3];
    const bool less_2 = magnitude_2 < bound.limbs[2];
    const bool equal_2 = magnitude_2 == bound.limbs[2];
    const bool less_1 = magnitude_1 < bound.limbs[1];
    const bool equal_1 = magnitude_1 == bound.limbs[1];
    const bool less_equal_0 = magnitude_0 <= bound.limbs[0];
    return less_3 || (equal_3 && less_2) || (equal_3 && equal_2 && less_1) ||
           (equal_3 && equal_2 && equal_1 && less_equal_0);
}

//--------------------------------------
// Rescaling
//--------------------------------------

struct decimal_scale_pair {
    int32_t source;
    int32_t target;
};

int dora_decimal_rescale_i256(struct dora_i256 value,
                              struct decimal_scale_pair scale,
                              enum dora_rounding_mode mode,
                              struct dora_i256* out_value);

//------------------------------------------------------------------------------
//
// Prepared Descriptors
//
//------------------------------------------------------------------------------

struct decimal_prepared {
    struct dora_decimal_type type;
    struct decimal_unsigned precision_limit;
    struct dora_i256 bound;
    // Common-scale factor for comparisons; 10^precision otherwise.
    struct dora_i128 limit128;
};

int dora_decimal_prepare(struct dora_decimal_type type,
                         enum dora_physical_type physical,
                         struct decimal_prepared* out_prepared);

#endif
