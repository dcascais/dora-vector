// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "types/text.h"

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "numeric/arithmetic.h"
#include "numeric/decimal.h"
#include "numeric/decimal_float.h"
#include "numeric/float.h"
#include "numeric/limbs.h"
#include "temporal/temporal.h"
#include "types/types_internal.h"
#include "utf8/ascii.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t FORMAT_FLOAT_LIMB_COUNT = 40;
static constexpr size_t FORMAT_FLOAT_DIGIT_CAPACITY = 20;
static constexpr int32_t PARSE_FLOAT_EXPONENT_LIMIT = 2047;
static constexpr int64_t PARSE_DECIMAL_EXPONENT_LIMIT = 1'000'000;
static constexpr int64_t PARSE_DECIMAL_SCALE_LIMIT = 1000;
static constexpr int FORMAT_FLOAT_FIXED_MIN_EXPONENT = -4;
static constexpr int FORMAT_FLOAT_FIXED_MAX_EXPONENT = 16;

#ifdef __FAST_MATH__
#error "Kernels require strict floating-point semantics"
#endif

struct format_unsigned {
    uint32_t limbs[FORMAT_FLOAT_LIMB_COUNT];
};

static struct format_unsigned format_unsigned_u64(uint64_t value) {
    return (struct format_unsigned){.limbs = {(uint32_t)value, (uint32_t)(value >> 32U)}};
}

static int format_unsigned_compare(const struct format_unsigned* left, const struct format_unsigned* right) {
    for (size_t i = FORMAT_FLOAT_LIMB_COUNT; i != 0; --i) {
        if (left->limbs[i - 1] != right->limbs[i - 1]) {
            return left->limbs[i - 1] > right->limbs[i - 1] ? 1 : -1;
        }
    }
    return 0;
}

static void format_unsigned_multiply(struct format_unsigned* value, uint32_t factor) {
    uint64_t carry = 0;
    for (size_t i = 0; i < FORMAT_FLOAT_LIMB_COUNT; ++i) {
        uint64_t product = (uint64_t)value->limbs[i] * factor + carry;
        value->limbs[i] = (uint32_t)product;
        carry = product >> 32U;
    }
}

static void format_unsigned_shift(struct format_unsigned* value, int shift) {
    for (int i = 0; i < shift; ++i) {
        format_unsigned_multiply(value, 2);
    }
}

static void format_unsigned_subtract(struct format_unsigned* value, const struct format_unsigned* right) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < FORMAT_FLOAT_LIMB_COUNT; ++i) {
        uint64_t subtrahend = (uint64_t)right->limbs[i] + borrow;
        uint64_t minuend = value->limbs[i];
        value->limbs[i] = (uint32_t)(minuend - subtrahend);
        borrow = minuend < subtrahend;
    }
}

static int format_unsigned_add_compare(const struct format_unsigned* left,
                                       const struct format_unsigned* addend,
                                       struct format_unsigned right) {
    struct format_unsigned sum = {};
    uint64_t carry = 0;
    for (size_t i = 0; i < FORMAT_FLOAT_LIMB_COUNT; ++i) {
        uint64_t value = (uint64_t)left->limbs[i] + addend->limbs[i] + carry;
        sum.limbs[i] = (uint32_t)value;
        carry = value >> 32U;
    }
    return format_unsigned_compare(&sum, &right);
}

static size_t format_float_digits(uint64_t significand,
                                  bool lower_closer,
                                  int exponent,
                                  char out_digits[FORMAT_FLOAT_DIGIT_CAPACITY],
                                  int* out_exponent) {
    int shift = lower_closer ? 2 : 1;
    struct format_unsigned numerator = format_unsigned_u64(significand << (unsigned)shift);
    struct format_unsigned denominator = format_unsigned_u64(1);
    struct format_unsigned lower = format_unsigned_u64(1);
    struct format_unsigned upper = format_unsigned_u64(lower_closer ? 2 : 1);
    if (exponent >= 0) {
        format_unsigned_shift(&numerator, exponent);
        format_unsigned_shift(&lower, exponent);
        format_unsigned_shift(&upper, exponent);
        format_unsigned_shift(&denominator, shift);
    } else {
        format_unsigned_shift(&denominator, shift - exponent);
    }
    int decimal_exponent = 0;
    while (format_unsigned_compare(&numerator, &denominator) < 0) {
        format_unsigned_multiply(&numerator, 10);
        format_unsigned_multiply(&lower, 10);
        format_unsigned_multiply(&upper, 10);
        --decimal_exponent;
    }
    for (;;) {
        struct format_unsigned next = denominator;
        format_unsigned_multiply(&next, 10);
        if (format_unsigned_compare(&numerator, &next) < 0) {
            break;
        }
        denominator = next;
        ++decimal_exponent;
    }
    bool even = (significand & 1U) == 0;
    size_t count = 0;
    while (count < FORMAT_FLOAT_DIGIT_CAPACITY) {
        unsigned digit = 0;
        while (format_unsigned_compare(&numerator, &denominator) >= 0) {
            format_unsigned_subtract(&numerator, &denominator);
            ++digit;
        }
        bool low = format_unsigned_compare(&numerator, &lower) < (int)even;
        bool high = format_unsigned_add_compare(&numerator, &upper, denominator) > -(int)even;
        out_digits[count++] = (char)('0' + digit);
        if (low || high) {
            int midpoint = format_unsigned_add_compare(&numerator, &numerator, denominator);
            if (!low || (high && (midpoint > 0 || (midpoint == 0 && (digit & 1U) != 0)))) {
                size_t position = count;
                while (position != 0 && out_digits[position - 1] == '9') {
                    out_digits[--position] = '0';
                }
                if (position == 0) {
                    out_digits[0] = '1';
                    ++decimal_exponent;
                } else {
                    ++out_digits[position - 1];
                }
            }
            while (count > 1 && out_digits[count - 1] == '0') {
                --count;
            }
            *out_exponent = decimal_exponent;
            return count;
        }
        format_unsigned_multiply(&numerator, 10);
        format_unsigned_multiply(&lower, 10);
        format_unsigned_multiply(&upper, 10);
    }
    return SIZE_MAX;
}

struct binary_float_type {
    unsigned fraction_bits;
    unsigned exponent_bits;
    int bias;
};

static size_t format_float_bits(uint64_t bits, struct binary_float_type type, char out_text[TYPE_FLOAT_TEXT_CAPACITY]) {
    unsigned fraction_bits = type.fraction_bits;
    unsigned exponent_bits = type.exponent_bits;
    int bias = type.bias;
    if (out_text == nullptr) {
        return SIZE_MAX;
    }
    uint64_t fraction_mask = (UINT64_C(1) << fraction_bits) - 1;
    uint64_t fraction = bits & fraction_mask;
    uint64_t encoded_exponent = (bits >> fraction_bits) & ((UINT64_C(1) << exponent_bits) - 1);
    bool negative = ((bits >> (fraction_bits + exponent_bits)) & 1U) != 0;
    if (encoded_exponent == (UINT64_C(1) << exponent_bits) - 1) {
        const char* text = fraction != 0 ? "nan" : negative ? "-inf" : "inf";
        size_t length = strlen(text);
        memcpy(out_text, text, length + 1);
        return length;
    }
    size_t position = 0;
    if (negative) {
        out_text[position++] = '-';
    }
    if (encoded_exponent == 0 && fraction == 0) {
        memcpy(out_text + position, "0.0", 4);
        return position + 3;
    }
    uint64_t significand = fraction;
    if (encoded_exponent != 0) {
        significand |= UINT64_C(1) << fraction_bits;
    }
    int exponent = (int)(encoded_exponent == 0 ? 1 : encoded_exponent) - bias - (int)fraction_bits;
    char digits[FORMAT_FLOAT_DIGIT_CAPACITY] = {};
    int decimal_exponent = 0;
    size_t count =
        format_float_digits(significand, fraction == 0 && encoded_exponent > 1, exponent, digits, &decimal_exponent);
    if (count == SIZE_MAX) {
        return SIZE_MAX;
    }
    if (decimal_exponent < FORMAT_FLOAT_FIXED_MIN_EXPONENT || decimal_exponent >= FORMAT_FLOAT_FIXED_MAX_EXPONENT) {
        out_text[position++] = digits[0];
        if (count > 1) {
            out_text[position++] = '.';
            memcpy(out_text + position, digits + 1, count - 1);
            position += count - 1;
        }
        out_text[position++] = 'e';
        out_text[position++] = decimal_exponent < 0 ? '-' : '+';
        unsigned magnitude = (unsigned)(decimal_exponent < 0 ? -decimal_exponent : decimal_exponent);
        if (magnitude >= 100) {
            out_text[position++] = (char)('0' + magnitude / 100);
        }
        out_text[position++] = (char)('0' + magnitude / 10 % 10);
        out_text[position++] = (char)('0' + magnitude % 10);
    } else if (decimal_exponent < 0) {
        out_text[position++] = '0';
        out_text[position++] = '.';
        for (int i = -1; i > decimal_exponent; --i) {
            out_text[position++] = '0';
        }
        memcpy(out_text + position, digits, count);
        position += count;
    } else {
        size_t integral = (size_t)decimal_exponent + 1;
        for (size_t i = 0; i < integral; ++i) {
            out_text[position++] = i < count ? digits[i] : '0';
        }
        out_text[position++] = '.';
        if (count > integral) {
            memcpy(out_text + position, digits + integral, count - integral);
            position += count - integral;
        } else {
            out_text[position++] = '0';
        }
    }
    out_text[position] = '\0';
    return position;
}

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

static int parse_text_bool(const char* str, size_t len, bool* out_val) {
    if (out_val == nullptr || (str == nullptr && len != 0)) {
        return EINVAL;
    }
    if (len == 1) {
        if (str[0] == 't' || str[0] == 'T' || str[0] == '1') {
            *out_val = true;
            return 0;
        }
        if (str[0] == 'f' || str[0] == 'F' || str[0] == '0') {
            *out_val = false;
            return 0;
        }
        return EINVAL;
    }
    if (len == 4 && (str[0] == 't' || str[0] == 'T') && (str[1] == 'r' || str[1] == 'R') &&
        (str[2] == 'u' || str[2] == 'U') && (str[3] == 'e' || str[3] == 'E')) {
        *out_val = true;
        return 0;
    }
    if (len == 5 && (str[0] == 'f' || str[0] == 'F') && (str[1] == 'a' || str[1] == 'A') &&
        (str[2] == 'l' || str[2] == 'L') && (str[3] == 's' || str[3] == 'S') && (str[4] == 'e' || str[4] == 'E')) {
        *out_val = false;
        return 0;
    }
    return EINVAL;
}

//--------------------------------------
// Signed Integers
//--------------------------------------

static int parse_text_exact_numeric(const char* str, size_t len, struct dora_i256* out_coeff, int32_t* out_scale) {
    if (out_coeff == nullptr || out_scale == nullptr || (str == nullptr && len != 0)) {
        return EINVAL;
    }
    size_t i = 0;
    while (i < len && (str[i] == ' ' || str[i] == '\t' || str[i] == '\n' || str[i] == '\r')) {
        i++;
    }
    if (i >= len) {
        return EINVAL;
    }
    bool neg = false;
    if (str[i] == '+' || str[i] == '-') {
        neg = (str[i] == '-');
        i++;
    }
    if (i >= len) {
        return EINVAL;
    }
    uint64_t mag[4] = {};
    size_t digits_before = 0;
    while (i < len && str[i] >= '0' && str[i] <= '9') {
        if (dora_limbs_mul_u32((struct limbs_span){.values = mag, .count = 4},
                               (struct limbs_affine_u32){.factor = 10, .addend = (uint32_t)(str[i] - '0')}) != 0) {
            return ERANGE;
        }
        digits_before++;
        i++;
    }
    int64_t frac_digits = 0;
    if (i < len && str[i] == '.') {
        i++;
        size_t digits_after = 0;
        while (i < len && str[i] >= '0' && str[i] <= '9') {
            if (dora_limbs_mul_u32((struct limbs_span){.values = mag, .count = 4},
                                   (struct limbs_affine_u32){.factor = 10, .addend = (uint32_t)(str[i] - '0')}) != 0) {
                return ERANGE;
            }
            digits_after++;
            frac_digits++;
            i++;
        }
        if (digits_before == 0 && digits_after == 0) {
            return EINVAL;
        }
        if (digits_after == 0 && i < len && str[i] != 'e' && str[i] != 'E' && str[i] != ' ' && str[i] != '\t' &&
            str[i] != '\n' && str[i] != '\r') {
            return EINVAL;
        }
    } else if (digits_before == 0) {
        return EINVAL;
    }
    int64_t exp_val = 0;
    if (i < len && (str[i] == 'e' || str[i] == 'E')) {
        i++;
        bool exp_neg = false;
        if (i < len && (str[i] == '+' || str[i] == '-')) {
            exp_neg = (str[i] == '-');
            i++;
        }
        size_t exp_digits = 0;
        while (i < len && str[i] >= '0' && str[i] <= '9') {
            if (exp_val < PARSE_DECIMAL_EXPONENT_LIMIT) {
                exp_val = exp_val * 10 + (str[i] - '0');
            }
            exp_digits++;
            i++;
        }
        if (exp_digits == 0) {
            return EINVAL;
        }
        if (exp_neg) {
            exp_val = -exp_val;
        }
    }
    while (i < len && (str[i] == ' ' || str[i] == '\t' || str[i] == '\n' || str[i] == '\r')) {
        i++;
    }
    if (i < len) {
        return EINVAL;
    }
    if (dora_limbs_is_zero(mag, 4)) {
        *out_coeff = (struct dora_i256){};
        *out_scale = 0;
        return 0;
    }
    int64_t eff_scale = frac_digits - exp_val;
    while (eff_scale < 0) {
        if (dora_limbs_mul_u32((struct limbs_span){.values = mag, .count = 4},
                               (struct limbs_affine_u32){.factor = 10, .addend = 0}) != 0) {
            return ERANGE;
        }
        eff_scale++;
    }
    if (eff_scale > PARSE_DECIMAL_SCALE_LIMIT) {
        return ERANGE;
    }
    int status = dora_i256_from_mag(mag, neg, out_coeff);
    if (status != 0) {
        return status;
    }
    *out_scale = (int32_t)eff_scale;
    return 0;
}

static const char format_digit_pairs[] = "0001020304050607080910111213141516171819"
                                         "2021222324252627282930313233343536373839"
                                         "4041424344454647484950515253545556575859"
                                         "6061626364656667686970717273747576777879"
                                         "8081828384858687888990919293949596979899";

static char* format_unsigned(uint64_t value, char* ptr) {
    while (value >= 100) {
        const unsigned index = (unsigned)((value % 100) * 2);
        value /= 100;
        *--ptr = format_digit_pairs[index + 1];
        *--ptr = format_digit_pairs[index];
    }
    if (value < 10) {
        *--ptr = (char)('0' + value);
        return ptr;
    }
    const unsigned index = (unsigned)(value * 2);
    *--ptr = format_digit_pairs[index + 1];
    *--ptr = format_digit_pairs[index];
    return ptr;
}

static char* format_unsigned_wide(unsigned _BitInt(128) value, char* ptr) {
    static constexpr uint64_t chunk = UINT64_C(100'000'000'000'000'000);
    while (value > UINT64_MAX) {
        const uint64_t remainder = (uint64_t)(value % chunk);
        value /= chunk;
        char* const start = ptr;
        ptr = format_unsigned(remainder, ptr);
        while (start - ptr < 17) {
            *--ptr = '0';
        }
    }
    return format_unsigned((uint64_t)value, ptr);
}

static int unsigned_length(uint64_t value) {
    if (value >= UINT64_C(10'000'000'000)) {
        if (value >= UINT64_C(1'000'000'000'000'000)) {
            int length = 16;
            length += value >= UINT64_C(10'000'000'000'000'000);
            length += value >= UINT64_C(100'000'000'000'000'000);
            length += value >= UINT64_C(1'000'000'000'000'000'000);
            length += value >= UINT64_C(10'000'000'000'000'000'000);
            return length;
        }
        int length = 11;
        length += value >= UINT64_C(100'000'000'000);
        length += value >= UINT64_C(1'000'000'000'000);
        length += value >= UINT64_C(10'000'000'000'000);
        length += value >= UINT64_C(100'000'000'000'000);
        return length;
    }
    if (value >= UINT64_C(100'000)) {
        int length = 6;
        length += value >= UINT64_C(1'000'000);
        length += value >= UINT64_C(10'000'000);
        length += value >= UINT64_C(100'000'000);
        length += value >= UINT64_C(1'000'000'000);
        return length;
    }
    int length = 1;
    length += value >= UINT64_C(10);
    length += value >= UINT64_C(100);
    length += value >= UINT64_C(1'000);
    length += value >= UINT64_C(10'000);
    return length;
}

size_t dora_type_signed_length(int64_t value) {
    const int64_t sign = -(int64_t)(value < 0);
    const uint64_t magnitude = ((uint64_t)value ^ (uint64_t)sign) + (uint64_t)-sign;
    return (size_t)unsigned_length(magnitude) + (size_t)-sign;
}

size_t dora_type_unsigned_length(uint64_t value) {
    return (size_t)unsigned_length(value);
}

static void format_magnitude(uint64_t magnitude, char* out_text_end, int64_t sign) {
    char* const begin = format_unsigned(magnitude, out_text_end);
    if (sign < 0) {
        begin[-1] = '-';
    }
}

void dora_type_format_signed(int64_t value, char* out_text_end) {
    const int64_t mask = -(int64_t)(value < 0);
    const uint64_t magnitude = ((uint64_t)value ^ (uint64_t)mask) + (uint64_t)-mask;
    format_magnitude(magnitude, out_text_end, value);
}

void dora_type_format_unsigned(uint64_t value, char* out_text_end) {
    format_magnitude(value, out_text_end, 0);
}

int dora_type_integer_text_prepare(struct dora_i256 coefficient, int32_t scale, struct type_integer_text* out_text) {
    uint64_t mag[4] = {};
    bool neg = false;
    dora_i256_to_mag(coefficient, mag, &neg);
    char* const end = out_text->digits + sizeof out_text->digits;
    char* begin = end;
    if (mag[2] == 0 && mag[3] == 0 && mag[1] == 0) {
        begin = format_unsigned(mag[0], end);
    } else if (mag[2] == 0 && mag[3] == 0) {
        begin = format_unsigned_wide(((unsigned _BitInt(128))mag[1] << 64U) | mag[0], end);
    } else {
        while (!dora_limbs_is_zero(mag, 4)) {
            *--begin = (char)('0' + dora_limbs_div_u32((struct limbs_span){.values = mag, .count = 4}, 10));
        }
    }
    const size_t digits = (size_t)(end - begin);
    size_t required = digits;
    size_t extension = scale < 0 ? (size_t)(-(int64_t)scale) : scale > 0 ? 1 : 0;
    if (scale > 0 && (uint64_t)scale >= digits) {
        required = (size_t)scale;
        extension = 2;
    }
    if (ckd_add(&required, required, extension) || ckd_add(&required, required, (size_t)neg) || required == SIZE_MAX) {
        return EOVERFLOW;
    }
    out_text->first = (uint32_t)(begin - out_text->digits);
    out_text->digit_count = (uint32_t)digits;
    out_text->byte_count = required;
    out_text->scale = scale;
    out_text->is_negative = neg;
    return DORA_OK;
}

void dora_type_integer_text_write(const struct type_integer_text* text, char* out_text) {
    const char* const begin = text->digits + text->first;
    const size_t digits = text->digit_count;
    const int32_t scale = text->scale;
    size_t position = 0;
    if (text->is_negative) {
        out_text[position++] = '-';
    }
    if (scale > 0 && (uint64_t)scale >= digits) {
        out_text[position++] = '0';
        out_text[position++] = '.';
        for (size_t zeros = (size_t)scale - digits; zeros != 0; --zeros) {
            out_text[position++] = '0';
        }
    }
    for (size_t index = 0; index < digits; ++index) {
        if (scale > 0 && (uint64_t)scale < digits && digits - index == (size_t)scale) {
            out_text[position++] = '.';
        }
        out_text[position++] = begin[index];
    }
    for (int64_t zeros = scale < 0 ? -(int64_t)scale : 0; zeros > 0; --zeros) {
        out_text[position++] = '0';
    }
}

static int format_integer(struct dora_i256 coeff, int32_t scale, char* buf, size_t cap, size_t* out_count) {
    struct type_integer_text text;
    const int status = dora_type_integer_text_prepare(coeff, scale, &text);
    if (status != DORA_OK) {
        return status;
    }
    *out_count = text.byte_count;
    if (buf == nullptr) {
        return DORA_OK;
    }
    if (cap <= text.byte_count) {
        return ENOBUFS;
    }
    dora_type_integer_text_write(&text, buf);
    buf[text.byte_count] = '\0';
    return DORA_OK;
}

//--------------------------------------
// Floating Point
//--------------------------------------

static void parse_decimal_append(struct decimal_float_digits* value, uint8_t digit, size_t* leading_zeros) {
    if (value->num_digits == 0 && digit == 0) {
        ++*leading_zeros;
    } else if (value->num_digits < DECIMAL_FLOAT_DIGIT_CAPACITY) {
        value->digits[value->num_digits++] = digit;
    } else if (digit != 0) {
        value->was_truncated = true;
    }
}

struct parsed_exponent {
    uint64_t limbs[2];
    bool is_negative;
    bool has_overflow;
};

static int32_t parse_decimal_point(size_t integer_digits, size_t leading_zeros, struct parsed_exponent exponent) {
    bool negative = leading_zeros > integer_digits;
    uint64_t magnitude = (uint64_t)(negative ? leading_zeros - integer_digits : integer_digits - leading_zeros);
    if (exponent.has_overflow || exponent.limbs[1] > 1 ||
        (exponent.limbs[1] == 1 && exponent.is_negative == negative)) {
        return exponent.is_negative ? -PARSE_FLOAT_EXPONENT_LIMIT : PARSE_FLOAT_EXPONENT_LIMIT;
    }
    if (exponent.limbs[1] == 1) {
        magnitude = UINT64_MAX - magnitude;
        if (ckd_add(&magnitude, magnitude, exponent.limbs[0]) || ckd_add(&magnitude, magnitude, UINT64_C(1))) {
            return exponent.is_negative ? -PARSE_FLOAT_EXPONENT_LIMIT : PARSE_FLOAT_EXPONENT_LIMIT;
        }
        negative = exponent.is_negative;
    } else if (negative == exponent.is_negative) {
        if (ckd_add(&magnitude, magnitude, exponent.limbs[0])) {
            return negative ? -PARSE_FLOAT_EXPONENT_LIMIT : PARSE_FLOAT_EXPONENT_LIMIT;
        }
    } else if (magnitude >= exponent.limbs[0]) {
        magnitude -= exponent.limbs[0];
    } else {
        magnitude = exponent.limbs[0] - magnitude;
        negative = exponent.is_negative;
    }
    if (magnitude > PARSE_FLOAT_EXPONENT_LIMIT) {
        magnitude = PARSE_FLOAT_EXPONENT_LIMIT;
    }
    return negative ? -(int32_t)magnitude : (int32_t)magnitude;
}

int dora_type_parse_double(const char* text, size_t byte_count, double* out_value) {
    if (out_value == nullptr || (text == nullptr && byte_count != 0)) {
        return EINVAL;
    }
    size_t position = 0;
    while (position < byte_count && dora_utf8_ascii_is_text_space(text[position])) {
        ++position;
    }
    bool negative = false;
    if (position < byte_count && (text[position] == '+' || text[position] == '-')) {
        negative = text[position++] == '-';
    }
    struct decimal_float_digits value = {};
    size_t leading_zeros = 0;
    size_t integer_start = position;
    while (position < byte_count && text[position] >= '0' && text[position] <= '9') {
        parse_decimal_append(&value, (uint8_t)(text[position++] - '0'), &leading_zeros);
    }
    size_t integer_digits = position - integer_start;
    size_t fraction_digits = 0;
    if (position < byte_count && text[position] == '.') {
        ++position;
        size_t fraction_start = position;
        while (position < byte_count && text[position] >= '0' && text[position] <= '9') {
            parse_decimal_append(&value, (uint8_t)(text[position++] - '0'), &leading_zeros);
        }
        fraction_digits = position - fraction_start;
    }
    if (integer_digits == 0 && fraction_digits == 0) {
        return EINVAL;
    }
    struct parsed_exponent exponent = {};
    if (position < byte_count && (text[position] == 'e' || text[position] == 'E')) {
        ++position;
        if (position < byte_count && (text[position] == '+' || text[position] == '-')) {
            exponent.is_negative = text[position++] == '-';
        }
        size_t exponent_start = position;
        while (position < byte_count && text[position] >= '0' && text[position] <= '9') {
            uint32_t digit = (uint32_t)(text[position++] - '0');
            if (!exponent.has_overflow &&
                dora_limbs_mul_u32((struct limbs_span){.values = exponent.limbs, .count = 2},
                                   (struct limbs_affine_u32){.factor = 10, .addend = digit}) != 0) {
                exponent.has_overflow = true;
            }
        }
        if (position == exponent_start) {
            return EINVAL;
        }
    }
    while (position < byte_count && dora_utf8_ascii_is_text_space(text[position])) {
        ++position;
    }
    if (position != byte_count) {
        return EINVAL;
    }
    dora_decimal_digits_trim(&value);
    const bool nonzero = value.num_digits != 0;
    value.decimal_point = parse_decimal_point(integer_digits, leading_zeros, exponent);
    uint64_t bits = decimal_digits_to_binary(&value);
    const bool range = bits == UINT64_C(0x7f'f0'00'00'00'00'00'00) || (nonzero && bits == 0);
    bits |= (uint64_t)negative << 63U;
    memcpy(out_value, &bits, sizeof(bits));
    return range ? ERANGE : DORA_OK;
}

size_t dora_type_format_f32(float value, char out_text[TYPE_FLOAT_TEXT_CAPACITY]) {
    return format_float_bits(numeric_float_bits(value), (struct binary_float_type){23, 8, 127}, out_text);
}

size_t dora_type_format_f64(double value, char out_text[TYPE_FLOAT_TEXT_CAPACITY]) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return format_float_bits(bits, (struct binary_float_type){52, 11, 1023}, out_text);
}

//------------------------------------------------------------------------------
//
// Scalar Text
//
//------------------------------------------------------------------------------

static int format_scalar_bytes(struct dora_bytes bytes, char* out_text, size_t capacity_bytes, size_t* out_byte_count) {
    if (out_byte_count == nullptr || (bytes.data == nullptr && bytes.byte_count != 0)) {
        return EINVAL;
    }
    if (bytes.byte_count == SIZE_MAX) {
        return EOVERFLOW;
    }
    *out_byte_count = bytes.byte_count;
    if (out_text == nullptr) {
        return DORA_OK;
    }
    if (capacity_bytes <= bytes.byte_count) {
        return ENOBUFS;
    }
    if (bytes.byte_count != 0) {
        (memcpy)(out_text, bytes.data, bytes.byte_count);
    }
    out_text[bytes.byte_count] = '\0';
    return DORA_OK;
}

int dora_type_format_scalar(const struct dora_logical_type* type,
                            enum dora_physical_type physical_type,
                            const void* value,
                            bool is_null,
                            char* out_text,
                            size_t capacity_bytes,
                            size_t* out_byte_count) {
    if (type == nullptr || out_byte_count == nullptr || (value == nullptr && !is_null)) {
        return EINVAL;
    }
    if (type->extension_name.byte_count != 0 || type->id == DORA_TYPE_BINARY || type->id == DORA_TYPE_FIXED_BINARY ||
        type->id >= DORA_TYPE_LIST) {
        return ENOTSUP;
    }
    if (is_null || type->id == DORA_TYPE_NULL) {
        return format_scalar_bytes((struct dora_bytes){}, out_text, capacity_bytes, out_byte_count);
    }
    if (type->id == DORA_TYPE_UTF8) {
        struct dora_bytes bytes = {};
        memcpy(&bytes, value, sizeof(bytes));
        return format_scalar_bytes(bytes, out_text, capacity_bytes, out_byte_count);
    }
    if (type->id == DORA_TYPE_BOOLEAN) {
        bool boolean = false;
        memcpy(&boolean, value, sizeof(boolean));
        const struct dora_bytes bytes = {
            .data = (const uint8_t*)(boolean ? "true" : "false"),
            .byte_count = boolean ? 4 : 5
        };
        return format_scalar_bytes(bytes, out_text, capacity_bytes, out_byte_count);
    }
    if (type->id == DORA_TYPE_FLOAT16 || type->id == DORA_TYPE_FLOAT32 || type->id == DORA_TYPE_FLOAT64) {
        char text[TYPE_FLOAT_TEXT_CAPACITY] = {};
        size_t count = 0;
        if (physical_type == DORA_PHYSICAL_TYPE_F16) {
            struct dora_f16_bits half = {};
            memcpy(&half, value, sizeof(half));
            count = dora_type_format_f32(numeric_half_widen(half), text);
        } else if (physical_type == DORA_PHYSICAL_TYPE_F32) {
            float single = 0;
            memcpy(&single, value, sizeof(single));
            count = dora_type_format_f32(single, text);
        } else if (physical_type == DORA_PHYSICAL_TYPE_F64) {
            double number = 0;
            memcpy(&number, value, sizeof(number));
            count = dora_type_format_f64(number, text);
        } else {
            return EINVAL;
        }
        return format_scalar_bytes((struct dora_bytes){.data = (const uint8_t*)text, .byte_count = count},
                                   out_text,
                                   capacity_bytes,
                                   out_byte_count);
    }
    if (type->id >= DORA_TYPE_INTERVAL_YEAR_MONTH && type->id <= DORA_TYPE_INTERVAL_MONTH_DAY_NANO) {
        int64_t parts[3] = {};
        uint32_t part_count = 1;
        if (type->id == DORA_TYPE_INTERVAL_YEAR_MONTH) {
            int32_t months = 0;
            memcpy(&months, value, sizeof(months));
            parts[0] = months;
        } else if (type->id == DORA_TYPE_INTERVAL_DAY_TIME) {
            struct dora_interval_day_time interval = {};
            memcpy(&interval, value, sizeof(interval));
            parts[0] = interval.days;
            parts[1] = interval.milliseconds;
            part_count = 2;
        } else {
            struct dora_interval_month_day_nano interval = {};
            memcpy(&interval, value, sizeof(interval));
            parts[0] = interval.months;
            parts[1] = interval.days;
            parts[2] = interval.nanoseconds;
            part_count = 3;
        }
        struct temporal_text text = {};
        int status = dora_temporal_append_byte(&text, '(');
        for (uint32_t part = 0; status == 0 && part < part_count; ++part) {
            if (part != 0) {
                status = dora_temporal_append_byte(&text, ',');
                if (status == 0) {
                    status = dora_temporal_append_byte(&text, ' ');
                }
            }
            if (status == 0) {
                status = dora_temporal_append_signed(&text, parts[part], false, 1);
            }
        }
        if (status == 0) {
            status = dora_temporal_append_byte(&text, ')');
        }
        return status != 0 ? status
                           : format_scalar_bytes(
                                 (struct dora_bytes){.data = (const uint8_t*)text.bytes, .byte_count = text.byte_count},
                                 out_text,
                                 capacity_bytes,
                                 out_byte_count);
    }
    struct dora_i256 coefficient = {};
    if (numeric_load_integer(physical_type, value, &coefficient) != DORA_OK) {
        return EINVAL;
    }
    if (type->id >= DORA_TYPE_DATE32 && type->id <= DORA_TYPE_DURATION) {
        int64_t ticks = 0;
        memcpy(&ticks, &coefficient.limbs[0], sizeof(ticks));
        const enum dora_time_unit unit = type->id == DORA_TYPE_TIMESTAMP ? type->parameters.timestamp.unit
                                         : type->id == DORA_TYPE_DATE32 || type->id == DORA_TYPE_DATE64
                                             ? DORA_TIME_UNIT_SECOND
                                             : type->parameters.unit;
        const bool instant = type->id == DORA_TYPE_TIMESTAMP && type->parameters.timestamp.timezone != nullptr &&
                             type->parameters.timestamp.timezone[0] != 0;
        return dora_temporal_format(ticks,
                                    type->id,
                                    physical_type,
                                    unit,
                                    instant,
                                    out_text,
                                    capacity_bytes,
                                    out_byte_count);
    }
    return format_integer(coefficient,
                          type->id == DORA_TYPE_DECIMAL ? type->parameters.decimal.scale : 0,
                          out_text,
                          capacity_bytes,
                          out_byte_count);
}

static int parse_interval(enum dora_type type, const char* text, size_t bytes, void* out_value) {
    const uint32_t count = type == DORA_TYPE_INTERVAL_YEAR_MONTH ? 1 : type == DORA_TYPE_INTERVAL_DAY_TIME ? 2 : 3;
    if (bytes < 3 || (unsigned char)text[0] != '(' || (unsigned char)text[bytes - 1] != ')') {
        return EILSEQ;
    }
    int64_t values[3] = {};
    size_t start = 1;
    for (uint32_t part = 0; part < count; ++part) {
        size_t end = start;
        while (end < bytes - 1 && (unsigned char)text[end] != ',') {
            ++end;
        }
        if ((part + 1 < count) != (end < bytes - 1)) {
            return EILSEQ;
        }
        struct dora_i256 coefficient = {};
        int32_t scale = 0;
        int status = parse_text_exact_numeric(text + start, end - start, &coefficient, &scale);
        if (status == DORA_OK) {
            status = dora_decimal_rescale_i256(coefficient,
                                               (struct decimal_scale_pair){.source = scale},
                                               DORA_ROUNDING_MODE_EXACT,
                                               &coefficient);
        }
        if (status == DORA_OK) {
            status = dora_numeric_store_integer(coefficient, DORA_PHYSICAL_TYPE_I64, &values[part]);
        }
        if (status != DORA_OK) {
            return status == EINVAL ? EILSEQ : status;
        }
        if ((part < 2 || type != DORA_TYPE_INTERVAL_MONTH_DAY_NANO) &&
            (values[part] < INT32_MIN || values[part] > INT32_MAX)) {
            return ERANGE;
        }
        start = end + 1;
    }
    if (type == DORA_TYPE_INTERVAL_YEAR_MONTH) {
        const int32_t result = (int32_t)values[0];
        memcpy(out_value, &result, sizeof(result));
    } else if (type == DORA_TYPE_INTERVAL_DAY_TIME) {
        const struct dora_interval_day_time result = {.days = (int32_t)values[0], .milliseconds = (int32_t)values[1]};
        memcpy(out_value, &result, sizeof(result));
    } else {
        const struct dora_interval_month_day_nano result =
            {.months = (int32_t)values[0], .days = (int32_t)values[1], .nanoseconds = values[2]};
        memcpy(out_value, &result, sizeof(result));
    }
    return DORA_OK;
}

//--------------------------------------
// Typed Parsers
//--------------------------------------

static int parse_boolean(const struct dora_logical_type*,
                         enum dora_physical_type,
                         const char* text,
                         size_t byte_count,
                         enum dora_rounding_mode,
                         const struct dora_i256*,
                         void* out_value) {
    if (byte_count == 0) {
        return EILSEQ;
    }
    bool value = false;
    const int status = parse_text_bool(text, byte_count, &value);
    if (status == DORA_OK) {
        memcpy(out_value, &value, sizeof(value));
    }
    return status == EINVAL ? EILSEQ : status;
}

static int parse_float(const struct dora_logical_type*,
                       enum dora_physical_type physical_type,
                       const char* text,
                       const size_t byte_count,
                       enum dora_rounding_mode rounding_mode,
                       const struct dora_i256*,
                       void* out_value) {
    if (byte_count == 0) {
        return EILSEQ;
    }
    double value = 0;
    const int status = dora_type_parse_double(text, byte_count, &value);
    if (status != DORA_OK) {
        return status == EINVAL ? EILSEQ : status;
    }
    if (physical_type == DORA_PHYSICAL_TYPE_F64) {
        memcpy(out_value, &value, sizeof(value));
        return DORA_OK;
    }
    if (physical_type != DORA_PHYSICAL_TYPE_F32 && physical_type != DORA_PHYSICAL_TYPE_F16) {
        return EINVAL;
    }
    const float single = (float)value;
    if (!isfinite(single)) {
        return ERANGE;
    }
    if (physical_type == DORA_PHYSICAL_TYPE_F32) {
        if (rounding_mode == DORA_ROUNDING_MODE_EXACT && (double)single != value) {
            return ERANGE;
        }
        memcpy(out_value, &single, sizeof(single));
        return DORA_OK;
    }
    const struct dora_f16_bits half = numeric_half_narrow(single);
    if ((half.bits & 0x7c00U) == 0x7c00U ||
        (rounding_mode == DORA_ROUNDING_MODE_EXACT && (double)numeric_half_widen(half) != value)) {
        return ERANGE;
    }
    memcpy(out_value, &half, sizeof(half));
    return DORA_OK;
}

static int parse_intervals(const struct dora_logical_type* type,
                           enum dora_physical_type,
                           const char* text,
                           size_t byte_count,
                           enum dora_rounding_mode,
                           const struct dora_i256*,
                           void* out_value) {
    if (byte_count == 0) {
        return EILSEQ;
    }
    return parse_interval(type->id, text, byte_count, out_value);
}

static int parse_temporal(const struct dora_logical_type* type,
                          enum dora_physical_type physical_type,
                          const char* text,
                          size_t byte_count,
                          enum dora_rounding_mode rounding_mode,
                          const struct dora_i256*,
                          void* out_value) {
    if (byte_count == 0) {
        return EILSEQ;
    }
    const int status = dora_temporal_parse_type(type,
                                                physical_type,
                                                text,
                                                byte_count,
                                                rounding_mode,
                                                TEMPORAL_PARSE_CAST,
                                                nullptr,
                                                out_value);
    return status == EINVAL ? EILSEQ : status;
}

int dora_type_parse_exact(const struct dora_logical_type* type,
                          enum dora_physical_type physical_type,
                          const char* text,
                          const size_t byte_count,
                          enum dora_rounding_mode rounding_mode,
                          const struct dora_i256* decimal_bound,
                          void* out_value) {
    if (byte_count == 0) {
        return EILSEQ;
    }
    struct dora_i256 coefficient = {};
    int32_t scale = 0;
    int status = parse_text_exact_numeric(text, byte_count, &coefficient, &scale);
    if (status == DORA_OK) {
        status =
            dora_decimal_rescale_i256(coefficient,
                                      (struct decimal_scale_pair){
                                          .source = scale,
                                          .target = type->id == DORA_TYPE_DECIMAL ? type->parameters.decimal.scale : 0
                                      },
                                      rounding_mode,
                                      &coefficient);
    }
    if (status == DORA_OK && type->id == DORA_TYPE_DECIMAL) {
        status = decimal_bound == nullptr                                 ? EINVAL
                 : decimal_bound_admits_i256(coefficient, *decimal_bound) ? DORA_OK
                                                                          : ERANGE;
    }
    if (status == DORA_OK) {
        status = dora_numeric_store_integer(coefficient, physical_type, out_value);
    }
    return status == EINVAL ? EILSEQ : status;
}

// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_PARSE_INTEGER(S, T, P, IS_SIGNED)                                                                       \
    int dora_type_parse_##S(const struct dora_logical_type* type,                                                      \
                            enum dora_physical_type physical_type,                                                     \
                            const char* text,                                                                          \
                            size_t byte_count,                                                                         \
                            enum dora_rounding_mode rounding_mode,                                                     \
                            const struct dora_i256* decimal_bound,                                                     \
                            void* out_value) {                                                                         \
        if (byte_count == 0) {                                                                                         \
            return EILSEQ;                                                                                             \
        }                                                                                                              \
        size_t position = 0;                                                                                           \
        while (position < byte_count && dora_utf8_ascii_is_text_space(text[position])) {                               \
            ++position;                                                                                                \
        }                                                                                                              \
        const bool negative = position < byte_count && text[position] == '-';                                          \
        if (position < byte_count && (text[position] == '-' || text[position] == '+')) {                               \
            ++position;                                                                                                \
        }                                                                                                              \
        if (negative && !(IS_SIGNED)) {                                                                                \
            goto exact;                                                                                                \
        }                                                                                                              \
        const size_t first = position;                                                                                 \
        T result = 0;                                                                                                  \
        bool overflow = false;                                                                                         \
        while (position < byte_count && text[position] >= '0' && text[position] <= '9') {                              \
            const T digit = (T)(text[position++] - '0');                                                               \
            T scaled;                                                                                                  \
            overflow |= ckd_mul(&scaled, result, 10);                                                                  \
            const bool adjusted = (IS_SIGNED) ? ckd_sub(&result, scaled, digit) : ckd_add(&result, scaled, digit);     \
            overflow |= adjusted;                                                                                      \
        }                                                                                                              \
        const size_t last = position;                                                                                  \
        while (position < byte_count && dora_utf8_ascii_is_text_space(text[position])) {                               \
            ++position;                                                                                                \
        }                                                                                                              \
        if (last == first || position != byte_count) {                                                                 \
            goto exact;                                                                                                \
        }                                                                                                              \
        if ((IS_SIGNED) && !negative) {                                                                                \
            overflow |= ckd_sub(&result, 0, result);                                                                   \
        }                                                                                                              \
        if (overflow) {                                                                                                \
            return ERANGE;                                                                                             \
        }                                                                                                              \
        memcpy(out_value, &result, sizeof(result));                                                                    \
        return DORA_OK;                                                                                                \
exact:                                                                                                                 \
        return dora_type_parse_exact(type, physical_type, text, byte_count, rounding_mode, decimal_bound, out_value);  \
    }
SIGNED_INDEX_CARRIERS(DEFINE_PARSE_INTEGER, true)
UNSIGNED_INDEX_CARRIERS(DEFINE_PARSE_INTEGER, false)
#undef DEFINE_PARSE_INTEGER

static typeof(&dora_type_parse_exact) const parse_integers[DORA_TYPE_UINT64 + 1] = {
    [DORA_TYPE_INT8] = dora_type_parse_i8,
    [DORA_TYPE_INT16] = dora_type_parse_i16,
    [DORA_TYPE_INT32] = dora_type_parse_i32,
    [DORA_TYPE_INT64] = dora_type_parse_i64,
    [DORA_TYPE_UINT8] = dora_type_parse_u8,
    [DORA_TYPE_UINT16] = dora_type_parse_u16,
    [DORA_TYPE_UINT32] = dora_type_parse_u32,
    [DORA_TYPE_UINT64] = dora_type_parse_u64,
};

static constexpr enum dora_physical_type parse_integer_physical[DORA_TYPE_UINT64 + 1] = {
    [DORA_TYPE_INT8] = DORA_PHYSICAL_TYPE_I8,
    [DORA_TYPE_INT16] = DORA_PHYSICAL_TYPE_I16,
    [DORA_TYPE_INT32] = DORA_PHYSICAL_TYPE_I32,
    [DORA_TYPE_INT64] = DORA_PHYSICAL_TYPE_I64,
    [DORA_TYPE_UINT8] = DORA_PHYSICAL_TYPE_U8,
    [DORA_TYPE_UINT16] = DORA_PHYSICAL_TYPE_U16,
    [DORA_TYPE_UINT32] = DORA_PHYSICAL_TYPE_U32,
    [DORA_TYPE_UINT64] = DORA_PHYSICAL_TYPE_U64,
};

typeof(&dora_type_parse_exact) dora_type_parse_select(const struct dora_logical_type* type,
                                                      enum dora_physical_type physical_type) {
    if (type->extension_name.byte_count != 0 || type->id == DORA_TYPE_NULL || type->id == DORA_TYPE_UTF8 ||
        type->id == DORA_TYPE_BINARY || type->id == DORA_TYPE_FIXED_BINARY || type->id >= DORA_TYPE_LIST) {
        return nullptr;
    }
    if (type->id == DORA_TYPE_BOOLEAN) {
        return parse_boolean;
    }
    if (type->id >= DORA_TYPE_FLOAT16 && type->id <= DORA_TYPE_FLOAT64) {
        return parse_float;
    }
    if (type->id >= DORA_TYPE_INTERVAL_YEAR_MONTH && type->id <= DORA_TYPE_INTERVAL_MONTH_DAY_NANO) {
        return parse_intervals;
    }
    if (type->id >= DORA_TYPE_DATE32 && type->id <= DORA_TYPE_DURATION) {
        return parse_temporal;
    }
    if (type->id <= DORA_TYPE_UINT64 && parse_integers[type->id] != nullptr &&
        parse_integer_physical[type->id] == physical_type) {
        return parse_integers[type->id];
    }
    return dora_type_parse_exact;
}

int dora_type_parse_scalar(const struct dora_logical_type* type,
                           enum dora_physical_type physical_type,
                           const char* text,
                           size_t byte_count,
                           enum dora_rounding_mode rounding_mode,
                           void* out_value) {
    if (type == nullptr || out_value == nullptr || (text == nullptr && byte_count != 0) ||
        rounding_mode > DORA_ROUNDING_MODE_HALF_AWAY) {
        return EINVAL;
    }
    typeof(&dora_type_parse_exact) const parse = dora_type_parse_select(type, physical_type);
    if (parse == nullptr) {
        return ENOTSUP;
    }
    struct dora_i256 bound = {};
    const bool bounded =
        type->id != DORA_TYPE_DECIMAL ||
        dora_decimal_bound_prepare(type->parameters.decimal, DECIMAL256_MAX_PRECISION, &bound) == DORA_OK;
    return parse(type, physical_type, text, byte_count, rounding_mode, bounded ? &bound : nullptr, out_value);
}
