// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/binding/reduce.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/request.h"
#include "kernel/cpu/packet_bits.h"
#include "kernel/packet.h"
#include "numeric/arithmetic.h"
#include "numeric/decimal.h"
#include "numeric/decimal_float.h"
#include "numeric/float.h"
#include "numeric/limbs.h"
#include "types/load.h"
#include "types/order.h"
#include "types/types_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Packets
//
//------------------------------------------------------------------------------

static uint32_t packet_valid_count(const struct kernel_packet* source) {
    return source->validity == nullptr ? source->count
                                       : (uint32_t)dora_bitmap_count_set_words(source->validity, source->count);
}

static uint64_t packet_boolean_count(const struct kernel_packet* source, uint32_t valid, bool clear) {
    if (valid == 0) {
        return 0;
    }
    if (source->shape == KERNEL_SHAPE_CONSTANT) {
        return dora_bitmap_test(source->values, source->value_offset) != clear ? valid : 0;
    }
    uint64_t words[KERNEL_CHUNK_WORD_COUNT];
    if (source->shape == KERNEL_SHAPE_FLAT) {
        dora_bitmap_load_words(source->values, source->value_offset, source->count, words);
    } else {
        uint64_t all[KERNEL_CHUNK_WORD_COUNT];
        if (source->validity == nullptr) {
            bitmap_fill_bytes((uint8_t*)all, sizeof all, true);
        }
        dora_bitmap_gather_words_u32(words,
                                     source->values,
                                     source->value_offset,
                                     source->rows,
                                     source->validity == nullptr ? all : source->validity,
                                     source->count);
    }
    if (source->validity == nullptr) {
        const uint64_t ones = dora_bitmap_count_set_words(words, source->count);
        return clear ? valid - ones : ones;
    }
    return clear ? dora_bitmap_count_and_not_words(source->validity, words, source->count)
                 : dora_bitmap_count_and_words(words, source->validity, source->count);
}

#define FLAT_AT(S, ROW) type_load_##S(values, offset + (ROW))
#define MAP_AT(S, ROW) type_load_##S(values, offset + rows[ROW])

#define REDUCE_LANES(S, AT, LANE)                                                                                      \
    if (source->validity == nullptr) {                                                                                 \
        for (uint32_t row = 0; row < count; ++row) {                                                                   \
            LANE(S, row, AT(S, row));                                                                                  \
        }                                                                                                              \
    } else {                                                                                                           \
        for (uint32_t word = 0; word * 64 < count; ++word) {                                                           \
            const uint64_t tail = bitmap_word_tail_mask(count, word);                                                  \
            uint64_t active = source->validity[word] & tail;                                                           \
            if (active == tail) {                                                                                      \
                const uint32_t end = count - word * 64 < 64 ? count : word * 64 + 64;                                  \
                for (uint32_t row = word * 64; row < end; ++row) {                                                     \
                    LANE(S, row, AT(S, row));                                                                          \
                }                                                                                                      \
            } else {                                                                                                   \
                while (active != 0) {                                                                                  \
                    const uint32_t row = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                         \
                    LANE(S, row, AT(S, row));                                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define REDUCE_SHAPES(S, LANE, CONSTANT)                                                                               \
    switch (source->shape) {                                                                                           \
    case KERNEL_SHAPE_FLAT:                                                                                            \
        REDUCE_LANES(S, FLAT_AT, LANE)                                                                                 \
        break;                                                                                                         \
    case KERNEL_SHAPE_GATHERED:                                                                                        \
        REDUCE_LANES(S, MAP_AT, LANE)                                                                                  \
        break;                                                                                                         \
    case KERNEL_SHAPE_CONSTANT:                                                                                        \
        CONSTANT;                                                                                                      \
        break;                                                                                                         \
    }

#define REDUCE_MASKED(S, AT, LANE, MASKED)                                                                             \
    if (source->validity == nullptr) {                                                                                 \
        for (uint32_t row = 0; row < count; ++row) {                                                                   \
            LANE(S, row, AT(S, row));                                                                                  \
        }                                                                                                              \
    } else {                                                                                                           \
        for (uint32_t word = 0; word * 64 < count; ++word) {                                                           \
            const uint64_t tail = bitmap_word_tail_mask(count, word);                                                  \
            const uint64_t active = source->validity[word] & tail;                                                     \
            const uint64_t base = (uint64_t)word * 64;                                                                 \
            const uint64_t lanes = count - base < 64 ? count - base : 64;                                              \
            if (active == tail) {                                                                                      \
                for (uint64_t lane = 0; lane < lanes; ++lane) {                                                        \
                    LANE(S, base + lane, AT(S, base + lane));                                                          \
                }                                                                                                      \
            } else if (active != 0) {                                                                                  \
                for (uint64_t lane = 0; lane < lanes; ++lane) {                                                        \
                    MASKED(S, base + lane, AT(S, base + lane), bitmap_word_bit_mask(active, lane));                    \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define REDUCE_MASKED_SHAPES(S, LANE, MASKED, CONSTANT)                                                                \
    switch (source->shape) {                                                                                           \
    case KERNEL_SHAPE_FLAT:                                                                                            \
        REDUCE_MASKED(S, FLAT_AT, LANE, MASKED)                                                                        \
        break;                                                                                                         \
    case KERNEL_SHAPE_GATHERED:                                                                                        \
        REDUCE_MASKED(S, MAP_AT, LANE, MASKED)                                                                         \
        break;                                                                                                         \
    case KERNEL_SHAPE_CONSTANT:                                                                                        \
        CONSTANT;                                                                                                      \
        break;                                                                                                         \
    }

#define CONSTANT_REPEAT(S, LANE)                                                                                       \
    do {                                                                                                               \
        const typeof(type_load_##S(values, 0)) repeated =                                                              \
            valid == 0 ? (typeof(repeated)){} : type_load_##S(values, offset);                                         \
        for (uint32_t index = 0; index < valid; ++index) {                                                             \
            LANE(S, index, repeated);                                                                                  \
        }                                                                                                              \
    } while (0)

#define PACKET_LOCALS                                                                                                  \
    const uint32_t count = source->count;                                                                              \
    const uint8_t* const values = source->values;                                                                      \
    const uint64_t offset = source->value_offset;                                                                      \
    const uint32_t* const rows = source->rows

//------------------------------------------------------------------------------
//
// State
//
//------------------------------------------------------------------------------

static int reduce_begin(const struct kernel_packet* source, struct kernel_reduction_state* in_out_state) {
    if (source == nullptr || source->count > KERNEL_CHUNK_ROW_COUNT) {
        return EINVAL;
    }
    uint64_t input_count = 0;
    if (ckd_add(&input_count, in_out_state->input_count, source->count) || input_count > INT64_MAX) {
        return EOVERFLOW;
    }
    in_out_state->input_count = input_count;
    return DORA_OK;
}

static inline struct dora_i256 widen_i256(struct dora_i256 value) {
    return value;
}

static inline struct dora_i128 narrow_i128(struct dora_i256 value) {
    return (struct dora_i128){.limbs = {value.limbs[0], value.limbs[1]}};
}

#define WIDEN_i32 numeric_i256_from_i64
#define WIDEN_i64 numeric_i256_from_i64
#define WIDEN_i128 numeric_i256_from_i128
#define WIDEN_i256 widen_i256
#define NARROW_i128 narrow_i128
#define NARROW_i256 widen_i256
#define IS_NAN_f16(VALUE) numeric_f32_is_nan(numeric_half_widen(VALUE))
#define IS_NAN_f32(VALUE) numeric_f32_is_nan(VALUE)
#define IS_NAN_f64(VALUE) numeric_f64_is_nan(VALUE)
#define TO_DOUBLE_i8(VALUE) (double)(VALUE)
#define TO_DOUBLE_i16(VALUE) (double)(VALUE)
#define TO_DOUBLE_i32(VALUE) (double)(VALUE)
#define TO_DOUBLE_i64(VALUE) (double)(VALUE)
#define TO_DOUBLE_u8(VALUE) (double)(VALUE)
#define TO_DOUBLE_u16(VALUE) (double)(VALUE)
#define TO_DOUBLE_u32(VALUE) (double)(VALUE)
#define TO_DOUBLE_u64(VALUE) (double)(VALUE)
#define TO_DOUBLE_i128(VALUE) numeric_i256_to_double(numeric_i256_from_i128(VALUE))
#define TO_DOUBLE_u128(VALUE) numeric_i256_to_double(numeric_i256_from_u128(VALUE))
#define TO_DOUBLE_i256(VALUE) numeric_i256_to_double(VALUE)
#define TO_DOUBLE_f16(VALUE) (double)numeric_half_widen(VALUE)
#define TO_DOUBLE_f32(VALUE) (double)(VALUE)
#define TO_DOUBLE_f64(VALUE) (VALUE)

void dora_kernel_reduction_state_init(const struct kernel_reduction_binding* binding,
                                      struct kernel_reduction_state* out_state) {
    struct kernel_reduction_state state = {.minimum_width = UINT64_MAX};
    if (binding->operation == KERNEL_REDUCTION_PRODUCT) {
        state.value.floating = 1.0;
    } else if (binding->operation == KERNEL_REDUCTION_ALL) {
        state.value.boolean = true;
    }
    *out_state = state;
}

//------------------------------------------------------------------------------
//
// Count
//
//------------------------------------------------------------------------------

// All logical types
static int count_update(const struct kernel_packet* source,
                        const struct kernel_reduction_binding*,
                        struct kernel_reduction_state* in_out_state) {
    struct kernel_reduction_state state = *in_out_state;
    const int status = reduce_begin(source, &state);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t valid = packet_valid_count(source);
    state.value.count += valid;
    state.has_value = state.has_value || valid != 0;
    *in_out_state = state;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Sum
//
//------------------------------------------------------------------------------

//--------------------------------------
// Narrow Integers
//--------------------------------------

static int sum_fold(struct kernel_reduction_state* in_out_state, int64_t partial) {
    struct dora_i128 total = narrow_i128(in_out_state->value.integer);
    if (numeric_add_checked_i128(total, narrow_i128(numeric_i256_from_i64(partial)), &total)) {
        return EOVERFLOW;
    }
    in_out_state->value.integer = numeric_i256_from_i128(total);
    return DORA_OK;
}

#define SUM_NARROW_LANE(S, ROW, VALUE) partial += (int64_t)(VALUE)
#define SUM_NARROW_MASKED(S, ROW, VALUE, KEEP) partial += (int64_t)((uint64_t)(int64_t)(VALUE) & (KEEP))

// Int8, Int16, Int32, UInt8, UInt16, UInt32, Decimal (32-bit)
#define DEFINE_SUM_NARROW(S)                                                                                           \
    static int sum_##S(const struct kernel_packet* source,                                                             \
                       const struct kernel_reduction_binding*,                                                         \
                       struct kernel_reduction_state* in_out_state) {                                                  \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        int status = reduce_begin(source, &state);                                                                     \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        int64_t partial = 0;                                                                                           \
        REDUCE_MASKED_SHAPES(S,                                                                                        \
                             SUM_NARROW_LANE,                                                                          \
                             SUM_NARROW_MASKED,                                                                        \
                             partial = valid == 0 ? 0 : (int64_t)type_load_##S(values, offset) * valid)                \
        status = sum_fold(&state, partial);                                                                            \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_SUM_NARROW(i8)
DEFINE_SUM_NARROW(i16)
DEFINE_SUM_NARROW(i32)
DEFINE_SUM_NARROW(u8)
DEFINE_SUM_NARROW(u16)
DEFINE_SUM_NARROW(u32)
#undef DEFINE_SUM_NARROW

// Int32, Int64
#define DEFINE_SUM_NO_OVERFLOW(S)                                                                                      \
    static int sum_no_overflow_##S(const struct kernel_packet* source,                                                 \
                                   const struct kernel_reduction_binding*,                                             \
                                   struct kernel_reduction_state* in_out_state) {                                      \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        int64_t partial = (int64_t)state.value.integer.limbs[0];                                                       \
        REDUCE_MASKED_SHAPES(S,                                                                                        \
                             SUM_NARROW_LANE,                                                                          \
                             SUM_NARROW_MASKED,                                                                        \
                             partial += valid == 0 ? 0 : (int64_t)type_load_##S(values, offset) * valid)               \
        state.value.integer = numeric_i256_from_i64(partial);                                                          \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_SUM_NO_OVERFLOW(i32)
DEFINE_SUM_NO_OVERFLOW(i64)
#undef DEFINE_SUM_NO_OVERFLOW
#undef SUM_NARROW_MASKED
#undef SUM_NARROW_LANE

//--------------------------------------
// Word Integers
//--------------------------------------

#define SUM_WIDE_LANE(S, ROW, VALUE) total += (_BitInt(128))(VALUE)

// Int64, UInt64, Decimal (64-bit)
#define DEFINE_SUM_WIDE(S)                                                                                             \
    static int sum_##S(const struct kernel_packet* source,                                                             \
                       const struct kernel_reduction_binding*,                                                         \
                       struct kernel_reduction_state* in_out_state) {                                                  \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        _BitInt(128) total = type_i128_load(narrow_i128(state.value.integer));                                         \
        REDUCE_SHAPES(S, SUM_WIDE_LANE, total += valid == 0 ? 0 : (_BitInt(128))type_load_##S(values, offset) * valid) \
        state.value.integer = numeric_i256_from_i128(type_i128_store(total));                                          \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_SUM_WIDE(i64)
DEFINE_SUM_WIDE(u64)
#undef DEFINE_SUM_WIDE
#undef SUM_WIDE_LANE

//--------------------------------------
// Checked Integers
//--------------------------------------

#define CHECKED_LANE(S, ROW, VALUE)                                                                                    \
    do {                                                                                                               \
        if (numeric_add_checked_##S(total, VALUE, &total)) {                                                           \
            return EOVERFLOW;                                                                                          \
        }                                                                                                              \
    } while (0)

static int
sum_repeat_i128(struct dora_i128 total, struct dora_i128 value, uint32_t count, struct dora_i128* out_total) {
    uint64_t magnitude[4] = {};
    bool negative = false;
    dora_i256_to_mag(numeric_i256_from_i128(value), magnitude, &negative);
    int status = dora_limbs_mul_u32((struct limbs_span){.values = magnitude, .count = 4},
                                    (struct limbs_affine_u32){.factor = count});
    struct dora_i256 repeated = {};
    struct dora_i256 result = {};
    if (status == DORA_OK) {
        status = dora_i256_from_mag(magnitude, negative, &repeated);
    }
    if (status == DORA_OK && numeric_add_checked_i256(numeric_i256_from_i128(total), repeated, &result)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_numeric_store_integer(result, DORA_PHYSICAL_TYPE_I128, out_total);
    }
    return status == ERANGE ? EOVERFLOW : status;
}

// Int128, Int256, Decimal (128/256-bit)
#define DEFINE_SUM_CHECKED(S, CONSTANT)                                                                                \
    static int sum_##S(const struct kernel_packet* source,                                                             \
                       const struct kernel_reduction_binding*,                                                         \
                       struct kernel_reduction_state* in_out_state) {                                                  \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        int status = reduce_begin(source, &state);                                                                     \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        auto total = NARROW_##S(state.value.integer);                                                                  \
        REDUCE_SHAPES(S, CHECKED_LANE, CONSTANT)                                                                       \
        state.value.integer = WIDEN_##S(total);                                                                        \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_SUM_CHECKED(
    i128,
    if (valid != 0) {
        status = sum_repeat_i128(total, type_load_i128(values, offset), valid, &total);
        if (status != DORA_OK) {
            return status;
        }
    })
DEFINE_SUM_CHECKED(i256, CONSTANT_REPEAT(i256, CHECKED_LANE))
#undef DEFINE_SUM_CHECKED
#undef CHECKED_LANE

//--------------------------------------
// Floating Point
//--------------------------------------

// SUM requires low-plus-high-times-2^64 rounding.
static inline double uhugeint_double(struct dora_u128 value) {
    return (double)value.limbs[0] + ((double)value.limbs[1] * 0x1p64);
}

#define SUM_DOUBLE_f16(VALUE) TO_DOUBLE_f16(VALUE)
#define SUM_DOUBLE_f32(VALUE) TO_DOUBLE_f32(VALUE)
#define SUM_DOUBLE_f64(VALUE) TO_DOUBLE_f64(VALUE)
#define SUM_DOUBLE_u128(VALUE) uhugeint_double(VALUE)
#define SUM_FLOAT_LANE(S, ROW, VALUE) total += SUM_DOUBLE_##S(VALUE)

// Float16, Float32, Float64, UInt128
#define DEFINE_SUM_FLOAT(S)                                                                                            \
    static int sum_##S(const struct kernel_packet* source,                                                             \
                       const struct kernel_reduction_binding*,                                                         \
                       struct kernel_reduction_state* in_out_state) {                                                  \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        double total = state.value.floating;                                                                           \
        REDUCE_SHAPES(                                                                                                 \
            S,                                                                                                         \
            SUM_FLOAT_LANE,                                                                                            \
            if (valid != 0) { total += SUM_DOUBLE_##S(type_load_##S(values, offset)) * (double)valid; })               \
        state.value.floating = total;                                                                                  \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_SUM_FLOAT(f16)
DEFINE_SUM_FLOAT(f32)
DEFINE_SUM_FLOAT(f64)
DEFINE_SUM_FLOAT(u128)
#undef DEFINE_SUM_FLOAT
#undef SUM_FLOAT_LANE
#undef SUM_DOUBLE_u128
#undef SUM_DOUBLE_f64
#undef SUM_DOUBLE_f32
#undef SUM_DOUBLE_f16

//--------------------------------------
// Decimals
//--------------------------------------

static inline bool decimal_magnitude_is_below_limit(struct dora_i256 value, const struct decimal_unsigned* limit) {
    uint64_t magnitude[4] = {};
    dora_limbs_magnitude(value.limbs, 4, magnitude);
    return dora_limbs_compare(magnitude, limit->limbs, 4) < 0;
}

//--------------------------------------
// Boolean
//--------------------------------------

// Boolean
static int sum_boolean(const struct kernel_packet* source,
                       const struct kernel_reduction_binding*,
                       struct kernel_reduction_state* in_out_state) {
    struct kernel_reduction_state state = *in_out_state;
    int status = reduce_begin(source, &state);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t valid = packet_valid_count(source);
    status = sum_fold(&state, (int64_t)packet_boolean_count(source, valid, false));
    if (status != DORA_OK) {
        return status;
    }
    state.has_value = state.has_value || valid != 0;
    *in_out_state = state;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Product
//
//------------------------------------------------------------------------------

//--------------------------------------
// Numbers
//--------------------------------------

#define PRODUCT_DOUBLE_LANE(S, ROW, VALUE) total *= TO_DOUBLE_##S(VALUE)

// Signed integers, unsigned integers, floating-point types to Float64
#define DEFINE_PRODUCT_DOUBLE(S)                                                                                       \
    static int product_##S(const struct kernel_packet* source,                                                         \
                           const struct kernel_reduction_binding*,                                                     \
                           struct kernel_reduction_state* in_out_state) {                                              \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        double total = state.value.floating;                                                                           \
        REDUCE_SHAPES(S, PRODUCT_DOUBLE_LANE, CONSTANT_REPEAT(S, PRODUCT_DOUBLE_LANE))                                 \
        state.value.floating = total;                                                                                  \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_PRODUCT_DOUBLE(i8)
DEFINE_PRODUCT_DOUBLE(i16)
DEFINE_PRODUCT_DOUBLE(i32)
DEFINE_PRODUCT_DOUBLE(i64)
DEFINE_PRODUCT_DOUBLE(i128)
DEFINE_PRODUCT_DOUBLE(i256)
DEFINE_PRODUCT_DOUBLE(u8)
DEFINE_PRODUCT_DOUBLE(u16)
DEFINE_PRODUCT_DOUBLE(u32)
DEFINE_PRODUCT_DOUBLE(u64)
DEFINE_PRODUCT_DOUBLE(u128)
DEFINE_PRODUCT_DOUBLE(f16)
DEFINE_PRODUCT_DOUBLE(f32)
DEFINE_PRODUCT_DOUBLE(f64)
#undef DEFINE_PRODUCT_DOUBLE

//--------------------------------------
// Decimals into Float64
//--------------------------------------

static inline double product_decimal_double_i256(double factor, struct dora_i256 value) {
    return numeric_i256_to_double(value) * factor;
}

#define PRODUCT_DECIMAL_CONVERT_i32 decimal_double_i32
#define PRODUCT_DECIMAL_CONVERT_i64 decimal_double_i64
#define PRODUCT_DECIMAL_CONVERT_i128 decimal_double_i128
#define PRODUCT_DECIMAL_CONVERT_i256 product_decimal_double_i256
#define PRODUCT_DECIMAL_DOUBLE_LANE(S, ROW, VALUE) total *= PRODUCT_DECIMAL_CONVERT_##S(convert, VALUE)

// Decimal (32/64/128/256-bit) to Float64
#define DEFINE_PRODUCT_DECIMAL_DOUBLE(S, D, T, FACTOR)                                                                 \
    static int product_double_##D(const struct kernel_packet* source,                                                  \
                                  const struct kernel_reduction_binding* binding,                                      \
                                  struct kernel_reduction_state* in_out_state) {                                       \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        const int32_t scale = binding->input_decimal.type.scale;                                                       \
        const T convert = FACTOR;                                                                                      \
        double total = state.value.floating;                                                                           \
        REDUCE_SHAPES(S, PRODUCT_DECIMAL_DOUBLE_LANE, CONSTANT_REPEAT(S, PRODUCT_DECIMAL_DOUBLE_LANE))                 \
        state.value.floating = total;                                                                                  \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_PRODUCT_DECIMAL_DOUBLE(i32, decimal32, int32_t, scale)
DEFINE_PRODUCT_DECIMAL_DOUBLE(i64, decimal64, int32_t, scale)
DEFINE_PRODUCT_DECIMAL_DOUBLE(i128, decimal128, int32_t, scale)
DEFINE_PRODUCT_DECIMAL_DOUBLE(i256, decimal256, double, pow(10.0, -(double)scale))
#undef PRODUCT_DECIMAL_CONVERT_i128
#define PRODUCT_DECIMAL_CONVERT_i128(SCALE, VALUE) decimal_double_i64(SCALE, (int64_t)type_i128_load(VALUE))
DEFINE_PRODUCT_DECIMAL_DOUBLE(i128, decimal128_integer, int32_t, scale)
#undef DEFINE_PRODUCT_DECIMAL_DOUBLE
#undef PRODUCT_DECIMAL_DOUBLE_LANE
#undef PRODUCT_DECIMAL_CONVERT_i256
#undef PRODUCT_DECIMAL_CONVERT_i128
#undef PRODUCT_DECIMAL_CONVERT_i64
#undef PRODUCT_DECIMAL_CONVERT_i32
#undef PRODUCT_DOUBLE_LANE
#undef TO_DOUBLE_f64
#undef TO_DOUBLE_f32
#undef TO_DOUBLE_f16
#undef TO_DOUBLE_i256
#undef TO_DOUBLE_u128
#undef TO_DOUBLE_i128
#undef TO_DOUBLE_u64
#undef TO_DOUBLE_u32
#undef TO_DOUBLE_u16
#undef TO_DOUBLE_u8
#undef TO_DOUBLE_i64
#undef TO_DOUBLE_i32
#undef TO_DOUBLE_i16
#undef TO_DOUBLE_i8

//--------------------------------------
// Decimals
//--------------------------------------

struct product_decimal_step {
    const struct decimal_unsigned* target_limit;
    struct decimal_scale_pair first;
    struct decimal_scale_pair subsequent;
};

static int product_decimal_term(const struct product_decimal_step* step,
                                struct dora_i256 term,
                                struct kernel_reduction_state* in_out_state) {
    struct dora_i256 product = term;
    int status = in_out_state->has_value && numeric_multiply_checked_i256(in_out_state->value.integer, term, &product)
                     ? EOVERFLOW
                     : DORA_OK;
    if (status == DORA_OK) {
        status = dora_decimal_rescale_i256(product,
                                           in_out_state->has_value ? step->subsequent : step->first,
                                           DORA_ROUNDING_MODE_HALF_EVEN,
                                           &product);
    }
    if (status == DORA_OK && !decimal_magnitude_is_below_limit(product, step->target_limit)) {
        status = EOVERFLOW;
    }
    if (status != DORA_OK) {
        return status == ERANGE ? EOVERFLOW : status;
    }
    in_out_state->value.integer = product;
    in_out_state->has_value = true;
    return DORA_OK;
}

#define PRODUCT_DECIMAL_LANE(S, ROW, VALUE)                                                                            \
    do {                                                                                                               \
        status = product_decimal_term(&step, WIDEN_##S(VALUE), &state);                                                \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

// Decimal (32/64/128/256-bit)
#define DEFINE_PRODUCT_DECIMAL(S, D)                                                                                   \
    static int product_##D(const struct kernel_packet* source,                                                         \
                           const struct kernel_reduction_binding* binding,                                             \
                           struct kernel_reduction_state* in_out_state) {                                              \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        int status = reduce_begin(source, &state);                                                                     \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        const struct product_decimal_step step = {                                                                     \
            .target_limit = &binding->result_decimal.precision_limit,                                                  \
            .first = {.source = binding->input_decimal.type.scale, .target = binding->result_decimal.type.scale},      \
            .subsequent = {.source = binding->input_decimal.type.scale},                                               \
        };                                                                                                             \
        REDUCE_SHAPES(S, PRODUCT_DECIMAL_LANE, CONSTANT_REPEAT(S, PRODUCT_DECIMAL_LANE))                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_PRODUCT_DECIMAL(i32, decimal32)
DEFINE_PRODUCT_DECIMAL(i64, decimal64)
DEFINE_PRODUCT_DECIMAL(i128, decimal128)
DEFINE_PRODUCT_DECIMAL(i256, decimal256)
#undef DEFINE_PRODUCT_DECIMAL
#undef PRODUCT_DECIMAL_LANE
#undef NARROW_i256
#undef NARROW_i128
#undef WIDEN_i256
#undef WIDEN_i128
#undef WIDEN_i64
#undef WIDEN_i32

//------------------------------------------------------------------------------
//
// Extrema
//
//------------------------------------------------------------------------------

static int extremum_begin(const struct kernel_packet* source) {
    return source == nullptr || source->count > KERNEL_CHUNK_ROW_COUNT ? EINVAL : DORA_OK;
}

#define EXTREMUM_LANE(S, ROW, VALUE, REPLACE)                                                                          \
    do {                                                                                                               \
        const typeof(winner) candidate = (VALUE);                                                                      \
        const int order = type_order_##S(candidate, winner);                                                           \
        winner = !seen || (REPLACE) ? candidate : winner;                                                              \
        seen = true;                                                                                                   \
    } while (0)

#define MIN_LANE(S, ROW, VALUE) EXTREMUM_LANE(S, ROW, VALUE, order <= 0)
#define MAX_LANE(S, ROW, VALUE) EXTREMUM_LANE(S, ROW, VALUE, order >= 0)
#define LOWER_LANE(S, ROW, VALUE) EXTREMUM_LANE(S, ROW, VALUE, order < 0)
#define UPPER_LANE(S, ROW, VALUE) EXTREMUM_LANE(S, ROW, VALUE, order > 0)

#define DEFINE_EXTREMUM(NAME, S, LANE)                                                                                 \
    static int NAME##_##S(const struct kernel_packet* source,                                                          \
                          const struct kernel_reduction_binding*,                                                      \
                          struct kernel_reduction_state* in_out_state) {                                               \
        const int status = extremum_begin(source);                                                                     \
        if (status != DORA_OK || source->count == 0) {                                                                 \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        typeof(type_load_##S(values, 0)) winner = {};                                                                  \
        memcpy(&winner, in_out_state->value.extremum, sizeof(winner));                                                 \
        bool seen = in_out_state->has_value;                                                                           \
        REDUCE_SHAPES(S, LANE, if (packet_valid_count(source) != 0) { LANE(S, 0, type_load_##S(values, offset)); })    \
        memcpy(in_out_state->value.extremum, &winner, sizeof(winner));                                                 \
        in_out_state->has_value = seen;                                                                                \
        return DORA_OK;                                                                                                \
    }

// Signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals
#define DEFINE_EXTREMA(S, T, P)                                                                                        \
    DEFINE_EXTREMUM(min, S, MIN_LANE)                                                                                  \
    DEFINE_EXTREMUM(max, S, MAX_LANE)
FIXED_CARRIERS(DEFINE_EXTREMA)
#undef DEFINE_EXTREMA

// Float16, Float32, Float64, IntervalDayTime, IntervalMonthDayNano, Utf8, Binary, FixedBinary
#define DEFINE_BOUNDS(S)                                                                                               \
    DEFINE_EXTREMUM(lower, S, LOWER_LANE)                                                                              \
    DEFINE_EXTREMUM(upper, S, UPPER_LANE)

DEFINE_BOUNDS(f16)
DEFINE_BOUNDS(f32)
DEFINE_BOUNDS(f64)
DEFINE_BOUNDS(interval_day_time)
DEFINE_BOUNDS(interval_month_day_nano)
DEFINE_BOUNDS(bytes)
#undef DEFINE_BOUNDS
#undef DEFINE_EXTREMUM
#undef UPPER_LANE
#undef LOWER_LANE
#undef MAX_LANE
#undef MIN_LANE
#undef EXTREMUM_LANE

static int
extremum_boolean(const struct kernel_packet* source, bool maximum, struct kernel_reduction_state* in_out_state) {
    const int status = extremum_begin(source);
    if (status != DORA_OK) {
        return status;
    }
    struct kernel_reduction_state state = *in_out_state;
    for (uint32_t word = 0; word * 64 < source->count && !(state.has_value && state.value.boolean == maximum); ++word) {
        const uint64_t tail = bitmap_word_tail_mask(source->count, word);
        const uint64_t active = source->validity == nullptr ? tail : source->validity[word] & tail;
        const uint64_t ones = kernel_packet_bits_word(source, word, active);
        const uint64_t best = maximum ? ones : active & ~ones;
        if (best != 0) {
            state.value.boolean = maximum;
            state.has_value = true;
        } else if (!state.has_value && active != 0) {
            state.value.boolean = !maximum;
            state.has_value = true;
        }
    }
    *in_out_state = state;
    return DORA_OK;
}

// Boolean
static int min_boolean(const struct kernel_packet* source,
                       const struct kernel_reduction_binding*,
                       struct kernel_reduction_state* in_out_state) {
    return extremum_boolean(source, false, in_out_state);
}

// Boolean
static int max_boolean(const struct kernel_packet* source,
                       const struct kernel_reduction_binding*,
                       struct kernel_reduction_state* in_out_state) {
    return extremum_boolean(source, true, in_out_state);
}

//------------------------------------------------------------------------------
//
// Any and All
//
//------------------------------------------------------------------------------

static int
boolean_aggregate(const struct kernel_packet* source, bool all, struct kernel_reduction_state* in_out_state) {
    struct kernel_reduction_state state = *in_out_state;
    const int status = reduce_begin(source, &state);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t valid = packet_valid_count(source);
    const uint64_t counted = packet_boolean_count(source, valid, all);
    if (all) {
        state.value.boolean = state.value.boolean && counted == 0;
    } else {
        state.value.boolean = state.value.boolean || counted != 0;
    }
    state.has_value = state.has_value || valid != 0;
    *in_out_state = state;
    return DORA_OK;
}

// Boolean
static int any_update(const struct kernel_packet* source,
                      const struct kernel_reduction_binding*,
                      struct kernel_reduction_state* in_out_state) {
    return boolean_aggregate(source, false, in_out_state);
}

// Boolean
static int all_update(const struct kernel_packet* source,
                      const struct kernel_reduction_binding*,
                      struct kernel_reduction_state* in_out_state) {
    return boolean_aggregate(source, true, in_out_state);
}

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

//--------------------------------------
// NaN Count
//--------------------------------------

#define NAN_LANE(S, ROW, VALUE) nans += (uint64_t)IS_NAN_##S(VALUE)
#define NAN_MASKED(S, ROW, VALUE, KEEP) nans += (uint64_t)IS_NAN_##S(VALUE) & (KEEP) & 1U

// Float16, Float32, Float64
#define DEFINE_NAN_COUNT(S)                                                                                            \
    static int nan_count_##S(const struct kernel_packet* source,                                                       \
                             const struct kernel_reduction_binding*,                                                   \
                             struct kernel_reduction_state* in_out_state) {                                            \
        struct kernel_reduction_state state = *in_out_state;                                                           \
        const int status = reduce_begin(source, &state);                                                               \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        PACKET_LOCALS;                                                                                                 \
        const uint32_t valid = packet_valid_count(source);                                                             \
        uint64_t nans = 0;                                                                                             \
        REDUCE_MASKED_SHAPES(                                                                                          \
            S,                                                                                                         \
            NAN_LANE,                                                                                                  \
            NAN_MASKED,                                                                                                \
            if (valid != 0 && IS_NAN_##S(type_load_##S(values, offset))) { nans += valid; })                           \
        state.value.count += nans;                                                                                     \
        state.has_value = state.has_value || valid != 0;                                                               \
        *in_out_state = state;                                                                                         \
        return DORA_OK;                                                                                                \
    }
DEFINE_NAN_COUNT(f16)
DEFINE_NAN_COUNT(f32)
DEFINE_NAN_COUNT(f64)
#undef DEFINE_NAN_COUNT
#undef NAN_MASKED
#undef NAN_LANE
#undef IS_NAN_f64
#undef IS_NAN_f32
#undef IS_NAN_f16
#undef REDUCE_MASKED_SHAPES
#undef REDUCE_MASKED

//--------------------------------------
// Byte Widths
//--------------------------------------

struct width_totals {
    uint64_t minimum;
    uint64_t maximum;
    struct dora_u128 total;
};

static inline void width_add(struct width_totals* in_out_totals, struct dora_bytes value) {
    const uint64_t size = value.byte_count;
    const uint64_t low = in_out_totals->total.limbs[0] + size;
    in_out_totals->total.limbs[1] += (uint64_t)(low < size);
    in_out_totals->total.limbs[0] = low;
    in_out_totals->minimum = size < in_out_totals->minimum ? size : in_out_totals->minimum;
    in_out_totals->maximum = size > in_out_totals->maximum ? size : in_out_totals->maximum;
}

static inline void width_add_repeated(struct width_totals* totals, struct dora_bytes value, uint32_t repeat) {
    const uint64_t size = value.byte_count;
    totals->minimum = size < totals->minimum ? size : totals->minimum;
    totals->maximum = size > totals->maximum ? size : totals->maximum;
    totals->total = type_u128_store(type_u128_load(totals->total) + (unsigned _BitInt(128))size * repeat);
}

#define WIDTH_LANE(S, ROW, VALUE) width_add(&totals, VALUE)

// Utf8, Binary, FixedBinary
static int byte_width_update(const struct kernel_packet* source,
                             const struct kernel_reduction_binding*,
                             struct kernel_reduction_state* in_out_state) {
    struct kernel_reduction_state state = *in_out_state;
    const int status = reduce_begin(source, &state);
    if (status != DORA_OK) {
        return status;
    }
    PACKET_LOCALS;
    const uint32_t valid = packet_valid_count(source);
    struct width_totals totals = {
        .minimum = state.minimum_width,
        .maximum = state.maximum_width,
        .total = state.total_width,
    };
    REDUCE_SHAPES(
        bytes,
        WIDTH_LANE,
        if (valid != 0) { width_add_repeated(&totals, type_load_bytes(values, offset), valid); })
    state.minimum_width = totals.minimum;
    state.maximum_width = totals.maximum;
    state.total_width = totals.total;
    state.has_value = state.has_value || valid != 0;
    *in_out_state = state;
    return DORA_OK;
}

#undef WIDTH_LANE
#undef PACKET_LOCALS
#undef CONSTANT_REPEAT
#undef REDUCE_SHAPES
#undef REDUCE_LANES
#undef MAP_AT
#undef FLAT_AT

//------------------------------------------------------------------------------
//
// Finish
//
//------------------------------------------------------------------------------

struct reduce_output {
    const void* value;
    size_t byte_count;
    bool is_valid;
};

static int reduce_publish(const struct reduce_output* output, struct kernel_result* out_result) {
    if (out_result == nullptr || (output->is_valid && out_result->values == nullptr) ||
        (!output->is_valid && out_result->validity == nullptr)) {
        return EINVAL;
    }
    if (output->is_valid) {
        memcpy(out_result->values, output->value, output->byte_count);
    }
    if (out_result->validity != nullptr) {
        out_result->validity[0] = output->is_valid ? 1 : 0;
    }
    out_result->null_count = output->is_valid ? 0 : 1;
    out_result->failure_bits = 0;
    return DORA_OK;
}

static int finish_count(const struct kernel_reduction_binding* binding,
                        const struct kernel_reduction_state* state,
                        struct kernel_result* out_result) {
    const struct reduce_output output =
        {.value = &state->value.count, .byte_count = binding->result_byte_width, .is_valid = true};
    return reduce_publish(&output, out_result);
}

static int finish_value(const struct kernel_reduction_binding* binding,
                        const struct kernel_reduction_state* state,
                        struct kernel_result* out_result) {
    const struct reduce_output output = {
        .value = &state->value,
        .byte_count = binding->result_byte_width,
        .is_valid = state->has_value,
    };
    return reduce_publish(&output, out_result);
}

static int finish_boolean(const struct kernel_reduction_binding*,
                          const struct kernel_reduction_state* state,
                          struct kernel_result* out_result) {
    const uint64_t word = state->value.boolean ? 1 : 0;
    const struct reduce_output output = {.value = &word, .byte_count = sizeof(word), .is_valid = state->has_value};
    return reduce_publish(&output, out_result);
}

static int finish_decimal(const struct kernel_reduction_binding* binding,
                          const struct kernel_reduction_state* state,
                          struct kernel_result* out_result) {
    struct dora_i256 stored = {};
    if (state->has_value) {
        if (!decimal_magnitude_is_below_limit(state->value.integer, &binding->result_decimal.precision_limit)) {
            return EOVERFLOW;
        }
        const int status = dora_numeric_store_integer(state->value.integer, binding->result_physical, &stored);
        if (status != DORA_OK) {
            return status == ERANGE ? EOVERFLOW : status;
        }
    }
    const struct reduce_output output =
        {.value = &stored, .byte_count = binding->result_byte_width, .is_valid = state->has_value};
    return reduce_publish(&output, out_result);
}

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

//--------------------------------------
// Tables
//--------------------------------------

static constexpr uint32_t REDUCE_CARRIER_COUNT = DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO + 1;

static typeof(&count_update) const sum_leaves[REDUCE_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = sum_i8,
    [DORA_PHYSICAL_TYPE_I16] = sum_i16,
    [DORA_PHYSICAL_TYPE_I32] = sum_i32,
    [DORA_PHYSICAL_TYPE_I64] = sum_i64,
    [DORA_PHYSICAL_TYPE_I128] = sum_i128,
    [DORA_PHYSICAL_TYPE_I256] = sum_i256,
    [DORA_PHYSICAL_TYPE_U8] = sum_u8,
    [DORA_PHYSICAL_TYPE_U16] = sum_u16,
    [DORA_PHYSICAL_TYPE_U32] = sum_u32,
    [DORA_PHYSICAL_TYPE_U64] = sum_u64,
    [DORA_PHYSICAL_TYPE_U128] = sum_u128,
    [DORA_PHYSICAL_TYPE_F16] = sum_f16,
    [DORA_PHYSICAL_TYPE_F32] = sum_f32,
    [DORA_PHYSICAL_TYPE_F64] = sum_f64,
};

static typeof(&count_update) const product_leaves[REDUCE_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = product_i8,
    [DORA_PHYSICAL_TYPE_I16] = product_i16,
    [DORA_PHYSICAL_TYPE_I32] = product_i32,
    [DORA_PHYSICAL_TYPE_I64] = product_i64,
    [DORA_PHYSICAL_TYPE_I128] = product_i128,
    [DORA_PHYSICAL_TYPE_I256] = product_i256,
    [DORA_PHYSICAL_TYPE_U8] = product_u8,
    [DORA_PHYSICAL_TYPE_U16] = product_u16,
    [DORA_PHYSICAL_TYPE_U32] = product_u32,
    [DORA_PHYSICAL_TYPE_U64] = product_u64,
    [DORA_PHYSICAL_TYPE_U128] = product_u128,
    [DORA_PHYSICAL_TYPE_F16] = product_f16,
    [DORA_PHYSICAL_TYPE_F32] = product_f32,
    [DORA_PHYSICAL_TYPE_F64] = product_f64,
};

static typeof(&count_update) const decimal_double_product_leaves[REDUCE_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I32] = product_double_decimal32,
    [DORA_PHYSICAL_TYPE_I64] = product_double_decimal64,
    [DORA_PHYSICAL_TYPE_I128] = product_double_decimal128,
    [DORA_PHYSICAL_TYPE_I256] = product_double_decimal256,
};

static typeof(&count_update) const decimal_product_leaves[REDUCE_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I32] = product_decimal32,
    [DORA_PHYSICAL_TYPE_I64] = product_decimal64,
    [DORA_PHYSICAL_TYPE_I128] = product_decimal128,
    [DORA_PHYSICAL_TYPE_I256] = product_decimal256,
};

static typeof(&count_update) const extremum_leaves[4][REDUCE_CARRIER_COUNT] = {
    {
        [DORA_PHYSICAL_TYPE_NONE] = lower_bytes,
        [DORA_PHYSICAL_TYPE_I8] = min_i8,
        [DORA_PHYSICAL_TYPE_I16] = min_i16,
        [DORA_PHYSICAL_TYPE_I32] = min_i32,
        [DORA_PHYSICAL_TYPE_I64] = min_i64,
        [DORA_PHYSICAL_TYPE_I128] = min_i128,
        [DORA_PHYSICAL_TYPE_I256] = min_i256,
        [DORA_PHYSICAL_TYPE_U8] = min_u8,
        [DORA_PHYSICAL_TYPE_U16] = min_u16,
        [DORA_PHYSICAL_TYPE_U32] = min_u32,
        [DORA_PHYSICAL_TYPE_U64] = min_u64,
        [DORA_PHYSICAL_TYPE_U128] = min_u128,
        [DORA_PHYSICAL_TYPE_F16] = min_f16,
        [DORA_PHYSICAL_TYPE_F32] = min_f32,
        [DORA_PHYSICAL_TYPE_F64] = min_f64,
        [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] = min_interval_day_time,
        [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] = min_interval_month_day_nano,
    },
    {
        [DORA_PHYSICAL_TYPE_NONE] = upper_bytes,
        [DORA_PHYSICAL_TYPE_I8] = max_i8,
        [DORA_PHYSICAL_TYPE_I16] = max_i16,
        [DORA_PHYSICAL_TYPE_I32] = max_i32,
        [DORA_PHYSICAL_TYPE_I64] = max_i64,
        [DORA_PHYSICAL_TYPE_I128] = max_i128,
        [DORA_PHYSICAL_TYPE_I256] = max_i256,
        [DORA_PHYSICAL_TYPE_U8] = max_u8,
        [DORA_PHYSICAL_TYPE_U16] = max_u16,
        [DORA_PHYSICAL_TYPE_U32] = max_u32,
        [DORA_PHYSICAL_TYPE_U64] = max_u64,
        [DORA_PHYSICAL_TYPE_U128] = max_u128,
        [DORA_PHYSICAL_TYPE_F16] = max_f16,
        [DORA_PHYSICAL_TYPE_F32] = max_f32,
        [DORA_PHYSICAL_TYPE_F64] = max_f64,
        [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] = max_interval_day_time,
        [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] = max_interval_month_day_nano,
    },
    {
        [DORA_PHYSICAL_TYPE_NONE] = lower_bytes,
        [DORA_PHYSICAL_TYPE_I8] = min_i8,
        [DORA_PHYSICAL_TYPE_I16] = min_i16,
        [DORA_PHYSICAL_TYPE_I32] = min_i32,
        [DORA_PHYSICAL_TYPE_I64] = min_i64,
        [DORA_PHYSICAL_TYPE_I128] = min_i128,
        [DORA_PHYSICAL_TYPE_I256] = min_i256,
        [DORA_PHYSICAL_TYPE_U8] = min_u8,
        [DORA_PHYSICAL_TYPE_U16] = min_u16,
        [DORA_PHYSICAL_TYPE_U32] = min_u32,
        [DORA_PHYSICAL_TYPE_U64] = min_u64,
        [DORA_PHYSICAL_TYPE_U128] = min_u128,
        [DORA_PHYSICAL_TYPE_F16] = lower_f16,
        [DORA_PHYSICAL_TYPE_F32] = lower_f32,
        [DORA_PHYSICAL_TYPE_F64] = lower_f64,
        [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] = lower_interval_day_time,
        [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] = lower_interval_month_day_nano,
    },
    {
        [DORA_PHYSICAL_TYPE_NONE] = upper_bytes,
        [DORA_PHYSICAL_TYPE_I8] = max_i8,
        [DORA_PHYSICAL_TYPE_I16] = max_i16,
        [DORA_PHYSICAL_TYPE_I32] = max_i32,
        [DORA_PHYSICAL_TYPE_I64] = max_i64,
        [DORA_PHYSICAL_TYPE_I128] = max_i128,
        [DORA_PHYSICAL_TYPE_I256] = max_i256,
        [DORA_PHYSICAL_TYPE_U8] = max_u8,
        [DORA_PHYSICAL_TYPE_U16] = max_u16,
        [DORA_PHYSICAL_TYPE_U32] = max_u32,
        [DORA_PHYSICAL_TYPE_U64] = max_u64,
        [DORA_PHYSICAL_TYPE_U128] = max_u128,
        [DORA_PHYSICAL_TYPE_F16] = upper_f16,
        [DORA_PHYSICAL_TYPE_F32] = upper_f32,
        [DORA_PHYSICAL_TYPE_F64] = upper_f64,
        [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] = upper_interval_day_time,
        [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] = upper_interval_month_day_nano,
    },
};

static typeof(&count_update) const nan_count_leaves[REDUCE_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_F16] = nan_count_f16,
    [DORA_PHYSICAL_TYPE_F32] = nan_count_f32,
    [DORA_PHYSICAL_TYPE_F64] = nan_count_f64,
};

//--------------------------------------
// Admission
//--------------------------------------

static int reduce_result(struct kernel_reduction_binding* in_out_binding, struct dora_logical_type type) {
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    const int status = type_scalar_physical(&type, &physical);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_logical_type* const result = in_out_binding->request.result_type;
    if (result != nullptr &&
        (result->id != type.id ||
         (type.id == DORA_TYPE_DECIMAL && (result->parameters.decimal.precision != type.parameters.decimal.precision ||
                                           result->parameters.decimal.scale != type.parameters.decimal.scale ||
                                           type_decimal_byte_width(result) != physical_width(physical))) ||
         (type.id == DORA_TYPE_TIME && result->parameters.unit != type.parameters.unit))) {
        return EINVAL;
    }
    in_out_binding->result_type = type;
    in_out_binding->result_physical = physical;
    in_out_binding->result_byte_width = physical_width(physical);
    return DORA_OK;
}

static int reduce_floating(struct kernel_reduction_binding* in_out_binding) {
    return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_FLOAT64});
}

//--------------------------------------
// Operations
//--------------------------------------

static int bind_sum_decimal(struct kernel_reduction_binding* in_out_binding) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    int status = dora_decimal_prepare(request->input_type->parameters.decimal,
                                      in_out_binding->input_physical,
                                      &in_out_binding->input_decimal);
    if (status != DORA_OK) {
        return status;
    }
    const bool wide = in_out_binding->input_decimal.type.precision > DECIMAL128_MAX_PRECISION;
    const struct dora_decimal_type expected = {
        .precision = wide ? DECIMAL256_MAX_PRECISION : DECIMAL128_MAX_PRECISION,
        .scale = in_out_binding->input_decimal.type.scale,
        .bit_width = wide ? 256 : 128,
    };
    const struct dora_logical_type* const result = request->result_type;
    if (result != nullptr && (result->parameters.decimal.precision != expected.precision ||
                              result->parameters.decimal.scale != expected.scale)) {
        return EINVAL;
    }
    status = reduce_result(in_out_binding,
                           (struct dora_logical_type){.id = DORA_TYPE_DECIMAL, .parameters.decimal = expected});
    if (status == DORA_OK) {
        status = dora_decimal_prepare(expected, in_out_binding->result_physical, &in_out_binding->result_decimal);
    }
    in_out_binding->update = sum_leaves[in_out_binding->input_physical];
    in_out_binding->finish = finish_decimal;
    return status;
}

static int bind_sum(struct kernel_reduction_binding* in_out_binding) {
    const enum dora_type type = in_out_binding->request.input_type->id;
    const enum dora_physical_type physical = in_out_binding->input_physical;
    in_out_binding->finish = finish_value;
    in_out_binding->update = sum_leaves[physical];
    // Statistics-proven bounds make WRAP accumulation equivalent to checked accumulation.
    if (in_out_binding->request.policy == DORA_FAILURE_POLICY_WRAP) {
        if (type != DORA_TYPE_INT32 && type != DORA_TYPE_INT64) {
            return EINVAL;
        }
        in_out_binding->update = type == DORA_TYPE_INT32 ? sum_no_overflow_i32 : sum_no_overflow_i64;
        return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT128});
    }
    if (type == DORA_TYPE_BOOLEAN) {
        in_out_binding->update = sum_boolean;
        return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT128});
    }
    if (type == DORA_TYPE_DECIMAL) {
        return bind_sum_decimal(in_out_binding);
    }
    if ((type >= DORA_TYPE_FLOAT16 && type <= DORA_TYPE_FLOAT64) || type == DORA_TYPE_UINT128) {
        return reduce_floating(in_out_binding);
    }
    if (type == DORA_TYPE_INT256) {
        return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT256});
    }
    if (type >= DORA_TYPE_INT8 && type <= DORA_TYPE_UINT64) {
        return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT128});
    }
    return ENOTSUP;
}

static int bind_product(struct kernel_reduction_binding* in_out_binding) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    const enum dora_type type = request->input_type->id;
    in_out_binding->finish = finish_value;
    in_out_binding->update = product_leaves[in_out_binding->input_physical];
    if (type == DORA_TYPE_DECIMAL) {
        const int status = dora_decimal_prepare(request->input_type->parameters.decimal,
                                                in_out_binding->input_physical,
                                                &in_out_binding->input_decimal);
        if (status != DORA_OK) {
            return status;
        }
        in_out_binding->update = decimal_double_product_leaves[in_out_binding->input_physical];
        if (request->input_type->parameters.decimal.scale == 0 &&
            request->input_type->parameters.decimal.precision <= DECIMAL64_MAX_PRECISION &&
            in_out_binding->input_physical == DORA_PHYSICAL_TYPE_I128) {
            in_out_binding->update = product_double_decimal128_integer;
        }
        return reduce_floating(in_out_binding);
    }
    if (type >= DORA_TYPE_INT8 && type <= DORA_TYPE_FLOAT64) {
        return reduce_floating(in_out_binding);
    }
    return ENOTSUP;
}

static int bind_product_decimal(struct kernel_reduction_binding* in_out_binding) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    if (request->input_type->id != DORA_TYPE_DECIMAL) {
        return ENOTSUP;
    }
    if (request->result_type == nullptr || request->result_type->id != DORA_TYPE_DECIMAL) {
        return EINVAL;
    }
    int status = reduce_result(in_out_binding, *request->result_type);
    if (status == DORA_OK) {
        status = dora_decimal_prepare(request->input_type->parameters.decimal,
                                      in_out_binding->input_physical,
                                      &in_out_binding->input_decimal);
    }
    if (status == DORA_OK) {
        status = dora_decimal_prepare(request->result_type->parameters.decimal,
                                      in_out_binding->result_physical,
                                      &in_out_binding->result_decimal);
    }
    in_out_binding->update = decimal_product_leaves[in_out_binding->input_physical];
    in_out_binding->finish = finish_decimal;
    return status;
}

static int bind_extremum(struct kernel_reduction_binding* in_out_binding, bool maximum, bool bound) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    const struct dora_logical_type* const type = request->input_type;
    const struct dora_logical_type* const result = request->result_type;
    int status = type->id == DORA_TYPE_DECIMAL ? dora_decimal_prepare(type->parameters.decimal,
                                                                      in_out_binding->input_physical,
                                                                      &in_out_binding->input_decimal)
                                               : DORA_OK;
    if (status == DORA_OK && type->id == DORA_TYPE_DECIMAL && result != nullptr &&
        (result->parameters.decimal.precision != type->parameters.decimal.precision ||
         result->parameters.decimal.scale != type->parameters.decimal.scale)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = reduce_result(in_out_binding, *type);
    }
    in_out_binding->result_decimal = in_out_binding->input_decimal;
    in_out_binding->update = extremum_leaves[(bound ? 2 : 0) + (maximum ? 1 : 0)][in_out_binding->input_physical];
    in_out_binding->finish = finish_value;
    if (type->id == DORA_TYPE_BOOLEAN) {
        in_out_binding->update = maximum ? max_boolean : min_boolean;
        in_out_binding->finish = finish_boolean;
    } else if (type_is_bytes(type->id)) {
        in_out_binding->input_byte_width = sizeof(struct dora_bytes);
        in_out_binding->result_byte_width = sizeof(struct dora_bytes);
    }
    return status;
}

static int bind_boolean(struct kernel_reduction_binding* in_out_binding, bool all) {
    if (in_out_binding->request.input_type->id != DORA_TYPE_BOOLEAN) {
        return ENOTSUP;
    }
    in_out_binding->update = all ? all_update : any_update;
    in_out_binding->finish = finish_boolean;
    return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_BOOLEAN});
}

static int bind_nan_count(struct kernel_reduction_binding* in_out_binding) {
    const enum dora_type type = in_out_binding->request.input_type->id;
    if (type < DORA_TYPE_FLOAT16 || type > DORA_TYPE_FLOAT64) {
        return ENOTSUP;
    }
    in_out_binding->update = nan_count_leaves[in_out_binding->input_physical];
    in_out_binding->finish = finish_count;
    return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT64});
}

static int bind_byte_width(struct kernel_reduction_binding* in_out_binding) {
    if (!type_is_bytes(in_out_binding->request.input_type->id)) {
        return ENOTSUP;
    }
    in_out_binding->update = byte_width_update;
    in_out_binding->input_byte_width = sizeof(struct dora_bytes);
    in_out_binding->result_byte_width = sizeof(struct dora_u128);
    return DORA_OK;
}

static int bind_operation(enum kernel_reduction operation, struct kernel_reduction_binding* in_out_binding) {
    switch (operation) {
    case KERNEL_REDUCTION_COUNT_VALID:
        in_out_binding->update = count_update;
        in_out_binding->finish = finish_count;
        return reduce_result(in_out_binding, (struct dora_logical_type){.id = DORA_TYPE_INT64});
    case KERNEL_REDUCTION_SUM:
        return bind_sum(in_out_binding);
    case KERNEL_REDUCTION_PRODUCT:
        return bind_product(in_out_binding);
    case KERNEL_REDUCTION_PRODUCT_DECIMAL:
        return bind_product_decimal(in_out_binding);
    case KERNEL_REDUCTION_MIN:
        return bind_extremum(in_out_binding, false, false);
    case KERNEL_REDUCTION_MAX:
        return bind_extremum(in_out_binding, true, false);
    case KERNEL_REDUCTION_LOWER_BOUND:
        return bind_extremum(in_out_binding, false, true);
    case KERNEL_REDUCTION_UPPER_BOUND:
        return bind_extremum(in_out_binding, true, true);
    case KERNEL_REDUCTION_ANY:
        return bind_boolean(in_out_binding, false);
    case KERNEL_REDUCTION_ALL:
        return bind_boolean(in_out_binding, true);
    case KERNEL_REDUCTION_NAN_COUNT:
        return bind_nan_count(in_out_binding);
    case KERNEL_REDUCTION_BYTE_WIDTH:
        return bind_byte_width(in_out_binding);
    }
    return EINVAL;
}

int dora_kernel_reduction_bind(enum kernel_reduction operation,
                               const struct kernel_bind_request* request,
                               struct kernel_reduction_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || operation > KERNEL_REDUCTION_UPPER_BOUND ||
        (request->policy != DORA_FAILURE_POLICY_ERROR &&
         (request->policy != DORA_FAILURE_POLICY_WRAP || operation != KERNEL_REDUCTION_SUM)) ||
        (operation != KERNEL_REDUCTION_COUNT_VALID && request->input_type == nullptr)) {
        return EINVAL;
    }
    struct kernel_reduction_binding binding = {.request = *request, .operation = operation};
    int status = DORA_OK;
    if (request->input_type != nullptr) {
        status = type_scalar_physical(request->input_type, &binding.input_physical);
        if (status == DORA_OK && request->input_type->id == DORA_TYPE_NULL) {
            status = ENOTSUP;
        }
        binding.input_byte_width = physical_width(binding.input_physical);
    }
    if (status == DORA_OK || (status == ENOTSUP && operation == KERNEL_REDUCTION_COUNT_VALID)) {
        status = bind_operation(operation, &binding);
    }
    if (status == DORA_OK && binding.update == nullptr) {
        status = ENOTSUP;
    }
    if (status == DORA_OK) {
        *out_binding = binding;
    }
    return status;
}
