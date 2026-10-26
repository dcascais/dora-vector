// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/decimal.h"

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "numeric/limbs.h"
#include "types/types_internal.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr int64_t DECIMAL256_COEFFICIENT_DIGITS = DECIMAL256_MAX_PRECISION + 1;

//------------------------------------------------------------------------------
//
// Decimal Arithmetic
//
//------------------------------------------------------------------------------

//--------------------------------------
// Unsigned Magnitudes
//--------------------------------------

static bool decimal_unsigned_is_zero(struct decimal_unsigned value) {
    return dora_limbs_is_zero(value.limbs, 8);
}

static bool decimal_unsigned_is_one(struct decimal_unsigned value) {
    return value.limbs[0] == 1 && dora_limbs_is_zero(value.limbs + 1, 7);
}

int dora_decimal_unsigned_compare(struct decimal_unsigned left, struct decimal_unsigned right) {
    return dora_limbs_compare(left.limbs, right.limbs, 8);
}

static bool decimal_unsigned_multiply_small(struct decimal_unsigned* value, uint32_t factor) {
    uint64_t carry = 0;
    for (size_t i = 0; i < 8; ++i) {
        uint64_t low = (value->limbs[i] & UINT32_MAX) * factor + carry;
        uint64_t high = (value->limbs[i] >> 32U) * factor + (low >> 32U);
        value->limbs[i] = (low & UINT32_MAX) | (high << 32U);
        carry = high >> 32U;
    }
    return carry != 0;
}

static uint32_t
decimal_unsigned_divide_small(struct decimal_unsigned value, uint32_t divisor, struct decimal_unsigned* out_quotient) {
    uint64_t remainder = 0;
    *out_quotient = (struct decimal_unsigned){};
    for (size_t i = 8; i != 0; --i) {
        uint64_t high = (remainder << 32U) | (value.limbs[i - 1] >> 32U);
        uint64_t high_quotient = high / divisor;
        remainder = high % divisor;
        uint64_t low = (remainder << 32U) | (value.limbs[i - 1] & UINT32_MAX);
        uint64_t low_quotient = low / divisor;
        remainder = low % divisor;
        out_quotient->limbs[i - 1] = (high_quotient << 32U) | low_quotient;
    }
    return (uint32_t)remainder;
}

static void decimal_unsigned_divmod(struct decimal_unsigned numerator,
                                    struct decimal_unsigned denominator,
                                    struct decimal_unsigned* out_quotient,
                                    struct decimal_unsigned* out_remainder) {
    *out_quotient = (struct decimal_unsigned){};
    *out_remainder = (struct decimal_unsigned){};
    dora_limbs_divmod(
        (struct limbs_division_input){.numerator = numerator.limbs, .denominator = denominator.limbs, .count = 8},
        (struct limbs_division_output){.quotient = out_quotient->limbs, .remainder = out_remainder->limbs});
}

static struct decimal_unsigned decimal_unsigned_gcd(struct decimal_unsigned left, struct decimal_unsigned right) {
    for (size_t iteration = 0;
         iteration < sizeof(struct decimal_unsigned) * CHAR_BIT && !decimal_unsigned_is_zero(right);
         ++iteration) {
        struct decimal_unsigned quotient = {};
        struct decimal_unsigned remainder = {};
        decimal_unsigned_divmod(left, right, &quotient, &remainder);
        left = right;
        right = remainder;
    }
    return left;
}

static int decimal_unsigned_power10(int32_t exponent, struct decimal_unsigned* out_value) {
    struct decimal_unsigned result = {.limbs = {1}};
    for (int32_t i = 0; i < DECIMAL256_MAX_PRECISION && i < exponent; ++i) {
        if (decimal_unsigned_multiply_small(&result, 10)) {
            return EOVERFLOW;
        }
    }
    *out_value = result;
    return 0;
}

static int decimal_unsigned_fits_precision(struct decimal_unsigned value, int32_t precision, bool* out_fits) {
    struct decimal_unsigned limit = {};
    const int status = decimal_unsigned_power10(precision, &limit);
    if (status == 0) {
        *out_fits = dora_decimal_unsigned_compare(value, limit) < 0;
    }
    return status;
}

//--------------------------------------
// Signed Coefficients
//--------------------------------------

static struct decimal_signed decimal_signed_normalize(struct decimal_signed value) {
    if (decimal_unsigned_is_zero(value.magnitude)) {
        value.is_negative = false;
    }
    return value;
}

struct decimal_signed dora_decimal_signed_i256(struct dora_i256 value) {
    struct decimal_signed result = {.is_negative = dora_limbs_is_negative(value.limbs, 4)};
    dora_limbs_magnitude(value.limbs, 4, result.magnitude.limbs);
    return result;
}

struct dora_i256 dora_decimal_i256(struct decimal_signed value) {
    struct dora_i256 result = {};
    memcpy(result.limbs, value.magnitude.limbs, sizeof(result.limbs));
    if (value.is_negative) {
        dora_limbs_negate(result.limbs, 4, result.limbs);
    }
    return result;
}

//--------------------------------------
// Admission
//--------------------------------------

static int decimal_storage_admit(struct dora_decimal_type type, uint32_t storage_width) {
    if (type.precision < 1 || type.precision > DECIMAL256_MAX_PRECISION) {
        return EINVAL;
    }
    uint32_t minimum_width = type.precision <= DECIMAL32_MAX_PRECISION    ? 32
                             : type.precision <= DECIMAL64_MAX_PRECISION  ? 64
                             : type.precision <= DECIMAL128_MAX_PRECISION ? 128
                                                                          : 256;
    uint32_t declared_width = type.bit_width == 0 ? minimum_width : type.bit_width;
    return declared_width == storage_width && declared_width >= minimum_width ? 0 : EINVAL;
}

//--------------------------------------
// Exact Arithmetic
//--------------------------------------

int dora_decimal_multiply_exact(struct decimal_signed left,
                                struct decimal_signed right,
                                struct dora_decimal_type type,
                                struct decimal_signed* out_value) {
    if (decimal_unsigned_is_zero(left.magnitude) || decimal_unsigned_is_zero(right.magnitude)) {
        *out_value = (struct decimal_signed){};
        return 0;
    }
    struct decimal_signed result = {.is_negative = left.is_negative != right.is_negative};
    dora_limbs_multiply(
        (struct limbs_multiplication_input){.left = left.magnitude.limbs, .right = right.magnitude.limbs, .count = 4},
        result.magnitude.limbs);
    if (type.scale > 0) {
        if (type.scale > 2 * type.precision) {
            return ERANGE;
        }
        for (int32_t i = 0; i < (2 * DECIMAL256_MAX_PRECISION) && i < type.scale; ++i) {
            struct decimal_unsigned quotient = {};
            if (decimal_unsigned_divide_small(result.magnitude, 10, &quotient) != 0) {
                return ERANGE;
            }
            result.magnitude = quotient;
        }
    } else if (type.scale < 0) {
        int64_t scale_magnitude = -(int64_t)type.scale;
        if (scale_magnitude >= type.precision) {
            return EOVERFLOW;
        }
        struct decimal_unsigned unscaled_limit = {};
        const int status = decimal_unsigned_power10(type.precision - (int32_t)scale_magnitude, &unscaled_limit);
        if (status != 0) {
            return status;
        }
        if (dora_decimal_unsigned_compare(result.magnitude, unscaled_limit) >= 0) {
            return EOVERFLOW;
        }
        for (int64_t i = 0; i < DECIMAL256_MAX_PRECISION && i < scale_magnitude; ++i) {
            if (decimal_unsigned_multiply_small(&result.magnitude, 10)) {
                return EOVERFLOW;
            }
        }
    }
    bool fits = false;
    const int status = decimal_unsigned_fits_precision(result.magnitude, type.precision, &fits);
    if (status != 0) {
        return status;
    }
    if (!fits) {
        return EOVERFLOW;
    }
    *out_value = result;
    return 0;
}

int dora_decimal_divide_exact(struct decimal_signed left,
                              struct decimal_signed right,
                              struct dora_decimal_type type,
                              struct decimal_signed* out_value) {
    if (decimal_unsigned_is_zero(right.magnitude)) {
        return EDOM;
    }
    if (decimal_unsigned_is_zero(left.magnitude)) {
        *out_value = (struct decimal_signed){};
        return 0;
    }
    struct decimal_signed result = {.is_negative = left.is_negative != right.is_negative};
    if (type.scale <= 0) {
        int64_t scale_magnitude = -(int64_t)type.scale;
        if (scale_magnitude >= type.precision) {
            return ERANGE;
        }
        struct decimal_unsigned denominator = right.magnitude;
        for (int64_t i = 0; i < DECIMAL256_MAX_PRECISION && i < scale_magnitude; ++i) {
            if (decimal_unsigned_multiply_small(&denominator, 10)) {
                return EOVERFLOW;
            }
        }
        struct decimal_unsigned remainder = {};
        decimal_unsigned_divmod(left.magnitude, denominator, &result.magnitude, &remainder);
        if (!decimal_unsigned_is_zero(remainder)) {
            return ERANGE;
        }
        *out_value = decimal_signed_normalize(result);
        return 0;
    }

    struct decimal_unsigned gcd = decimal_unsigned_gcd(left.magnitude, right.magnitude);
    struct decimal_unsigned ignored = {};
    struct decimal_unsigned numerator = {};
    struct decimal_unsigned denominator = {};
    decimal_unsigned_divmod(left.magnitude, gcd, &numerator, &ignored);
    decimal_unsigned_divmod(right.magnitude, gcd, &denominator, &ignored);

    uint32_t twos = 0;
    uint32_t fives = 0;
    for (uint32_t factor_count = 0; factor_count < sizeof(struct dora_i256) * CHAR_BIT; ++factor_count) {
        struct decimal_unsigned quotient = {};
        if (decimal_unsigned_divide_small(denominator, 2, &quotient) != 0) {
            break;
        }
        denominator = quotient;
        ++twos;
    }
    for (uint32_t factor_count = 0; factor_count < sizeof(struct dora_i256) * CHAR_BIT; ++factor_count) {
        struct decimal_unsigned quotient = {};
        if (decimal_unsigned_divide_small(denominator, 5, &quotient) != 0) {
            break;
        }
        denominator = quotient;
        ++fives;
    }
    uint32_t required_scale = twos > fives ? twos : fives;
    if (!decimal_unsigned_is_one(denominator) || (uint32_t)type.scale < required_scale) {
        return ERANGE;
    }

    uint64_t remaining_twos = (uint32_t)type.scale - twos;
    uint64_t remaining_fives = (uint32_t)type.scale - fives;
    uint64_t tens = remaining_twos < remaining_fives ? remaining_twos : remaining_fives;
    if (tens >= (uint32_t)type.precision) {
        return EOVERFLOW;
    }
    result.magnitude = numerator;
    struct decimal_unsigned limit = {};
    const int limit_status = decimal_unsigned_power10(type.precision, &limit);
    if (limit_status != 0) {
        return limit_status;
    }
    for (uint64_t i = 0; i < DECIMAL256_MAX_PRECISION && i < tens; ++i) {
        if (decimal_unsigned_multiply_small(&result.magnitude, 10)) {
            return EOVERFLOW;
        }
    }
    uint64_t remaining =
        remaining_twos > remaining_fives ? remaining_twos - remaining_fives : remaining_fives - remaining_twos;
    uint32_t factor = remaining_twos > remaining_fives ? 2 : 5;
    for (uint64_t i = 0; i < sizeof(struct dora_i256) * CHAR_BIT && i < remaining; ++i) {
        if (dora_decimal_unsigned_compare(result.magnitude, limit) >= 0 ||
            decimal_unsigned_multiply_small(&result.magnitude, factor)) {
            return EOVERFLOW;
        }
    }
    if (dora_decimal_unsigned_compare(result.magnitude, limit) >= 0) {
        return EOVERFLOW;
    }
    *out_value = decimal_signed_normalize(result);
    return 0;
}

int dora_decimal_remainder_exact(struct decimal_signed left,
                                 struct decimal_signed right,
                                 struct decimal_signed* out_value) {
    if (decimal_unsigned_is_zero(right.magnitude)) {
        return EDOM;
    }
    struct decimal_unsigned quotient = {};
    struct decimal_signed result = {.is_negative = left.is_negative};
    decimal_unsigned_divmod(left.magnitude, right.magnitude, &quotient, &result.magnitude);
    *out_value = decimal_signed_normalize(result);
    return 0;
}

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Decimals
//--------------------------------------

int dora_decimal_bound_prepare(struct dora_decimal_type type, int32_t maximum, struct dora_i256* out_bound) {
    if (type.precision < 1 || type.precision > maximum) {
        return EINVAL;
    }
    uint64_t limit[4] = {1};
    for (int32_t i = 0; i < type.precision; ++i) {
        uint64_t carry = 0;
        for (size_t limb = 0; limb < 4; ++limb) {
            uint64_t low = (limit[limb] & UINT32_MAX) * 10 + carry;
            uint64_t high = (limit[limb] >> 32U) * 10 + (low >> 32U);
            limit[limb] = (low & UINT32_MAX) | (high << 32U);
            carry = high >> 32U;
        }
    }
    const uint64_t one[4] = {1};
    dora_limbs_subtract(limit, one, 4, out_bound->limbs);
    return DORA_OK;
}

int dora_decimal_bound_prepare_width(struct dora_decimal_type type, uint32_t width_bits, struct dora_i256* out_bound) {
    switch (width_bits) {
    case 32:
        return dora_decimal_bound_prepare(type, DECIMAL32_MAX_PRECISION, out_bound);
    case 64:
        return dora_decimal_bound_prepare(type, DECIMAL64_MAX_PRECISION, out_bound);
    case 128:
        return dora_decimal_bound_prepare(type, DECIMAL128_MAX_PRECISION, out_bound);
    case 256:
        return dora_decimal_bound_prepare(type, DECIMAL256_MAX_PRECISION, out_bound);
    default:
        return EINVAL;
    }
}

//--------------------------------------
// Rescaling
//--------------------------------------

int dora_decimal_rescale_i256(struct dora_i256 value,
                              struct decimal_scale_pair scale,
                              enum dora_rounding_mode mode,
                              struct dora_i256* out_value) {
    if (out_value == nullptr || mode > DORA_ROUNDING_MODE_HALF_AWAY) {
        return EINVAL;
    }
    if (scale.source == scale.target) {
        *out_value = value;
        return DORA_OK;
    }
    uint64_t magnitude[4] = {};
    bool negative = false;
    dora_i256_to_mag(value, magnitude, &negative);
    if (dora_limbs_is_zero(magnitude, 4)) {
        *out_value = (struct dora_i256){};
        return DORA_OK;
    }
    const struct limbs_span span = {.values = magnitude, .count = 4};
    if (scale.target > scale.source) {
        const int64_t steps = (int64_t)scale.target - scale.source;
        if (steps > DECIMAL256_COEFFICIENT_DIGITS) {
            return ERANGE;
        }
        for (int64_t step = 0; step < steps; ++step) {
            if (dora_limbs_mul_u32(span, (struct limbs_affine_u32){.factor = 10}) != DORA_OK) {
                return ERANGE;
            }
        }
        return dora_i256_from_mag(magnitude, negative, out_value);
    }
    const int64_t steps = (int64_t)scale.source - scale.target;
    const int64_t divisions = steps < DECIMAL256_COEFFICIENT_DIGITS + 1 ? steps : DECIMAL256_COEFFICIENT_DIGITS + 1;
    uint32_t last = 0;
    bool sticky = false;
    for (int64_t step = 0; step < divisions; ++step) {
        sticky = sticky || last != 0;
        last = dora_limbs_div_u32(span, 10);
    }
    if (steps > divisions) {
        sticky = sticky || last != 0;
        last = 0;
    }
    if (mode == DORA_ROUNDING_MODE_EXACT && (last != 0 || sticky)) {
        return ERANGE;
    }
    const bool increment =
        (mode == DORA_ROUNDING_MODE_HALF_EVEN && (last > 5 || (last == 5 && (sticky || (magnitude[0] & 1U) != 0)))) ||
        (mode == DORA_ROUNDING_MODE_HALF_AWAY && last >= 5);
    if (increment && dora_limbs_mul_u32(span, (struct limbs_affine_u32){.factor = 1, .addend = 1}) != DORA_OK) {
        return ERANGE;
    }
    return dora_i256_from_mag(magnitude, negative && !dora_limbs_is_zero(magnitude, 4), out_value);
}

//------------------------------------------------------------------------------
//
// Prepared Descriptors
//
//------------------------------------------------------------------------------

int dora_decimal_prepare(struct dora_decimal_type type,
                         enum dora_physical_type physical,
                         struct decimal_prepared* out_prepared) {
    if (out_prepared == nullptr) {
        return EINVAL;
    }
    const uint32_t width = physical_width(physical) * 8U;
    if (physical != DORA_PHYSICAL_TYPE_I32 && physical != DORA_PHYSICAL_TYPE_I64 &&
        physical != DORA_PHYSICAL_TYPE_I128 && physical != DORA_PHYSICAL_TYPE_I256) {
        return ENOTSUP;
    }
    if (type.bit_width == 0) {
        type.bit_width = width;
    }
    const int status = decimal_storage_admit(type, width);
    if (status != DORA_OK) {
        return status;
    }
    struct decimal_unsigned limit = {};
    const int power_status = decimal_unsigned_power10(type.precision, &limit);
    if (power_status != DORA_OK) {
        return power_status;
    }
    struct decimal_prepared prepared = {.type = type, .precision_limit = limit};
    const int bound_status = dora_decimal_bound_prepare(type, DECIMAL256_MAX_PRECISION, &prepared.bound);
    if (bound_status != DORA_OK) {
        return bound_status;
    }
    if (width <= 128) {
        memcpy(prepared.limit128.limbs, limit.limbs, sizeof(prepared.limit128.limbs));
    }
    *out_prepared = prepared;
    return DORA_OK;
}
