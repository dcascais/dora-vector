// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scalar text parsing and formatting.

#ifndef DORA_TYPES_TEXT_H
#define DORA_TYPES_TEXT_H

#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

static constexpr size_t TYPE_FLOAT_TEXT_CAPACITY = 32;
static constexpr size_t TYPE_INTEGER_DIGIT_CAPACITY = 80;

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

// Boolean, signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals
int dora_type_parse_scalar(const struct dora_logical_type* type,
                           enum dora_physical_type physical_type,
                           const char* text,
                           size_t byte_count,
                           enum dora_rounding_mode rounding_mode,
                           void* out_value);
// Boolean, signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals
typeof(int (*)(const struct dora_logical_type* type,
               enum dora_physical_type physical_type,
               const char* text,
               size_t byte_count,
               enum dora_rounding_mode rounding_mode,
               const struct dora_i256* decimal_bound,
               void* out_value))
dora_type_parse_select(const struct dora_logical_type* type, enum dora_physical_type physical_type);

// Signed integers, unsigned integers, Decimal
int dora_type_parse_exact(const struct dora_logical_type* type,
                          enum dora_physical_type physical_type,
                          const char* text,
                          size_t byte_count,
                          enum dora_rounding_mode rounding_mode,
                          const struct dora_i256* decimal_bound,
                          void* out_value);
typeof(dora_type_parse_exact) dora_type_parse_i8;
typeof(dora_type_parse_exact) dora_type_parse_i16;
typeof(dora_type_parse_exact) dora_type_parse_i32;
typeof(dora_type_parse_exact) dora_type_parse_i64;
typeof(dora_type_parse_exact) dora_type_parse_u8;
typeof(dora_type_parse_exact) dora_type_parse_u16;
typeof(dora_type_parse_exact) dora_type_parse_u32;
typeof(dora_type_parse_exact) dora_type_parse_u64;

// Null, Boolean, signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals, Utf8
int dora_type_format_scalar(const struct dora_logical_type* type,
                            enum dora_physical_type physical_type,
                            const void* value,
                            bool is_null,
                            char* out_text,
                            size_t capacity_bytes,
                            size_t* out_byte_count);

//--------------------------------------
// Signed Integers
//--------------------------------------

struct type_integer_text {
    char digits[TYPE_INTEGER_DIGIT_CAPACITY];
    uint32_t first;
    uint32_t digit_count;
    size_t byte_count;
    int32_t scale;
    bool is_negative;
};

int dora_type_integer_text_prepare(struct dora_i256 coefficient, int32_t scale, struct type_integer_text* out_text);
void dora_type_integer_text_write(const struct type_integer_text* text, char* out_text);
size_t dora_type_signed_length(int64_t value);
size_t dora_type_unsigned_length(uint64_t value);
void dora_type_format_signed(int64_t value, char* out_text_end);
void dora_type_format_unsigned(uint64_t value, char* out_text_end);

//--------------------------------------
// Floating Point
//--------------------------------------

// ERANGE writes the rounded value; other failures preserve out_value.
int dora_type_parse_double(const char* text, size_t byte_count, double* out_value);

size_t dora_type_format_f32(float value, char out_text[TYPE_FLOAT_TEXT_CAPACITY]);
size_t dora_type_format_f64(double value, char out_text[TYPE_FLOAT_TEXT_CAPACITY]);

#endif
