// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/arithmetic.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/cpu/packet_bits.h"
#include "kernel/packet.h"
#include "numeric/decimal.h"
#include "numeric/decimal_float.h"
#include "numeric/failure.h"
#include "types/array_count.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

#ifdef __FAST_MATH__
#error "Kernels require strict floating-point semantics"
#endif

// Carriers: signed integers, unsigned integers, floating-point types
#define DEFINE_CARRIER_STORE(S, T, P)                                                                                  \
    static inline void carrier_store_##S(uint8_t* values, uint64_t row, T value) {                                     \
        (memcpy)(values + row * sizeof(T), &value, sizeof(T));                                                         \
    }
SIGNED_INDEX_CARRIERS(DEFINE_CARRIER_STORE)
DEFINE_CARRIER_STORE(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128)
DEFINE_CARRIER_STORE(i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256)
UNSIGNED_INDEX_CARRIERS(DEFINE_CARRIER_STORE)
DEFINE_CARRIER_STORE(u128, struct dora_u128, DORA_PHYSICAL_TYPE_U128)
DEFINE_CARRIER_STORE(f16, struct dora_f16_bits, DORA_PHYSICAL_TYPE_F16)
DEFINE_CARRIER_STORE(f32, float, DORA_PHYSICAL_TYPE_F32)
DEFINE_CARRIER_STORE(f64, double, DORA_PHYSICAL_TYPE_F64)
#undef DEFINE_CARRIER_STORE

//------------------------------------------------------------------------------
//
// Helpers
//
//------------------------------------------------------------------------------

static int arithmetic_fill_all_valid(uint32_t count, struct kernel_result* out_result) {
    if (out_result->validity != nullptr) {
        bitmap_prefix_words(count, out_result->validity, (size_t)bitmap_words_for_bits(count));
    }
    out_result->null_count = 0;
    return DORA_OK;
}

static int arithmetic_fill_null(uint32_t count, struct kernel_result* out_result) {
    if (out_result->validity != nullptr) {
        bitmap_fill_bytes((uint8_t*)out_result->validity,
                          (size_t)bitmap_words_for_bits(count) * sizeof(uint64_t),
                          false);
    }
    out_result->null_count = count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

static uint32_t arithmetic_lane_count(const struct kernel_packet* left, const struct kernel_packet* right) {
    return left->count == 0                                                                ? 0
           : left->shape == KERNEL_SHAPE_CONSTANT && right->shape == KERNEL_SHAPE_CONSTANT ? 1
                                                                                           : left->count;
}

//------------------------------------------------------------------------------
//
// Decimal Lanes
//
//------------------------------------------------------------------------------

static inline uint32_t decimal_failure(int status) {
    switch (status) {
    case DORA_OK:
        return 0;
    case EOVERFLOW:
        return NUMERIC_FAILURE_OVERFLOW;
    case EDOM:
        return NUMERIC_FAILURE_DIVIDE_BY_ZERO;
    case ERANGE:
        return NUMERIC_FAILURE_RANGE;
    default:
        return NUMERIC_FAILURE_INVALID_VALUE;
    }
}

#define DECIMAL_WIDTH_ROW(S, T, P, X, ARGUMENT) X(ARGUMENT, S, T, P)
#define DECIMAL_WIDTHS(X, ARGUMENT)                                                                                    \
    DECIMAL_RESCALE_CARRIERS(DECIMAL_WIDTH_ROW, X, ARGUMENT)                                                           \
    X(ARGUMENT, i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256)

static inline uint32_t decimal_admit_i32(const struct decimal_prepared* decimal, int32_t value) {
    return decimal_bound_admits_i32(value, decimal->bound) ? 0 : NUMERIC_FAILURE_OVERFLOW;
}

static inline uint32_t decimal_admit_i64(const struct decimal_prepared* decimal, int64_t value) {
    return decimal_bound_admits_i64(value, decimal->bound) ? 0 : NUMERIC_FAILURE_OVERFLOW;
}

static inline uint32_t decimal_admit_i128(const struct decimal_prepared* decimal, struct dora_i128 value) {
    const _BitInt(128) number = type_i128_load(value);
    const _BitInt(128) limit = type_i128_load(decimal->limit128);
    return number < limit && number > -limit ? 0 : NUMERIC_FAILURE_OVERFLOW;
}

static inline uint32_t decimal_admit_i256(const struct decimal_prepared* decimal, struct dora_i256 value) {
    return dora_decimal_unsigned_compare(dora_decimal_signed_i256(value).magnitude, decimal->precision_limit) < 0
               ? 0
               : NUMERIC_FAILURE_OVERFLOW;
}

#define DECIMAL_CAPPED(X, ARGUMENT)                                                                                    \
    X(ARGUMENT, i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128)                                                       \
    X(ARGUMENT, i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256)

// Decimal (128/256-bit)
#define DEFINE_DECIMAL_CAPPED(ARGUMENT, S, T, P)                                                                       \
    static inline uint32_t decimal_add_##S(const struct kernel_binary_binding* binding,                                \
                                           T left,                                                                     \
                                           T right,                                                                    \
                                           T* out_value) {                                                             \
        return numeric_add_checked_##S(left, right, out_value)                                                         \
                   ? NUMERIC_FAILURE_OVERFLOW                                                                          \
                   : decimal_admit_##S(&binding->result_decimal, *out_value);                                          \
    }                                                                                                                  \
    static inline uint32_t decimal_subtract_##S(const struct kernel_binary_binding* binding,                           \
                                                T left,                                                                \
                                                T right,                                                               \
                                                T* out_value) {                                                        \
        return numeric_subtract_checked_##S(left, right, out_value)                                                    \
                   ? NUMERIC_FAILURE_OVERFLOW                                                                          \
                   : decimal_admit_##S(&binding->result_decimal, *out_value);                                          \
    }
DECIMAL_CAPPED(DEFINE_DECIMAL_CAPPED, )
#undef DEFINE_DECIMAL_CAPPED

// Decimal (32/64/128-bit)
#define DEFINE_DECIMAL_REMAINDER(S, T)                                                                                 \
    static inline uint32_t decimal_remainder_##S(const struct kernel_binary_binding*, T left, T right, T* out_value) { \
        return (uint32_t)numeric_remainder_##S(left, right, out_value);                                                \
    }
DEFINE_DECIMAL_REMAINDER(i32, int32_t)
DEFINE_DECIMAL_REMAINDER(i64, int64_t)
DEFINE_DECIMAL_REMAINDER(i128, struct dora_i128)
#undef DEFINE_DECIMAL_REMAINDER

static inline uint32_t decimal_remainder_i256(const struct kernel_binary_binding*,
                                              struct dora_i256 left,
                                              struct dora_i256 right,
                                              struct dora_i256* out_value) {
    struct decimal_signed result = {};
    const int status =
        dora_decimal_remainder_exact(dora_decimal_signed_i256(left), dora_decimal_signed_i256(right), &result);
    *out_value = dora_decimal_i256(result);
    return decimal_failure(status);
}

static inline uint32_t decimal_multiply_i256(const struct kernel_binary_binding* binding,
                                             struct dora_i256 left,
                                             struct dora_i256 right,
                                             struct dora_i256* out_value) {
    const struct decimal_prepared* const decimal = &binding->result_decimal;
    struct decimal_signed result = {};
    const int status = dora_decimal_multiply_exact(dora_decimal_signed_i256(left),
                                                   dora_decimal_signed_i256(right),
                                                   decimal->type,
                                                   &result);
    *out_value = dora_decimal_i256(result);
    return decimal_failure(status);
}

static inline uint32_t decimal_divide_i256(const struct kernel_binary_binding* binding,
                                           struct dora_i256 left,
                                           struct dora_i256 right,
                                           struct dora_i256* out_value) {
    const struct decimal_prepared* const decimal = &binding->result_decimal;
    struct decimal_signed result = {};
    const int status = dora_decimal_divide_exact(dora_decimal_signed_i256(left),
                                                 dora_decimal_signed_i256(right),
                                                 decimal->type,
                                                 &result);
    *out_value = dora_decimal_i256(result);
    return decimal_failure(status);
}

// Decimal (32/64/128-bit)
#define DEFINE_DECIMAL_QUOTIENT(S, T)                                                                                  \
    static inline double decimal_quotient_##S(int32_t scale, T left, T right) {                                        \
        return numeric_divide_f64(decimal_double_##S(scale, left), decimal_double_##S(scale, right));                  \
    }
DEFINE_DECIMAL_QUOTIENT(i32, int32_t)
DEFINE_DECIMAL_QUOTIENT(i64, int64_t)
DEFINE_DECIMAL_QUOTIENT(i128, struct dora_i128)
#undef DEFINE_DECIMAL_QUOTIENT

static inline int64_t decimal_widen_i64(int32_t value) {
    return value;
}

static inline struct dora_i128 decimal_widen_i128(int64_t value) {
    return type_i128_store((_BitInt(128))value);
}

#define DECIMAL_SAME(VALUE) (VALUE)

#define DECIMAL_WIDENINGS(X, ARGUMENT)                                                                                 \
    X(ARGUMENT, i32, int32_t, i64, int64_t, decimal_widen_i64)                                                         \
    X(ARGUMENT, i64, int64_t, i128, struct dora_i128, decimal_widen_i128)

#define DECIMAL_PRODUCTS(X, ARGUMENT)                                                                                  \
    X(ARGUMENT, i32, int32_t, i32, int32_t, DECIMAL_SAME)                                                              \
    X(ARGUMENT, i64, int64_t, i64, int64_t, DECIMAL_SAME)                                                              \
    X(ARGUMENT, i128, struct dora_i128, i128, struct dora_i128, DECIMAL_SAME)                                          \
    DECIMAL_WIDENINGS(X, ARGUMENT)

// Decimal (32/64/128-bit); widening: 32 to 64, 64 to 128
#define DEFINE_DECIMAL_PRODUCT(ARGUMENT, LS, LT, OS, OT, WIDEN)                                                        \
    static inline uint32_t decimal_multiply_##LS##_##OS(const struct kernel_binary_binding* binding,                   \
                                                        LT left,                                                       \
                                                        LT right,                                                      \
                                                        OT* restrict out_value) {                                      \
        const bool overflow = numeric_multiply_checked_##OS(WIDEN(left), WIDEN(right), out_value);                     \
        return overflow || decimal_admit_##OS(&binding->result_decimal, *out_value) != 0 ? NUMERIC_FAILURE_RANGE : 0;  \
    }
DECIMAL_PRODUCTS(DEFINE_DECIMAL_PRODUCT, )
#undef DEFINE_DECIMAL_PRODUCT

// Decimal; widening: 32 to 64, 64 to 128 bits
#define DEFINE_DECIMAL_SUM(ARGUMENT, LS, LT, OS, OT, WIDEN)                                                            \
    static inline OT decimal_add_##LS##_##OS(LT left, LT right) {                                                      \
        return numeric_add_wrap_##OS(WIDEN(left), WIDEN(right));                                                       \
    }                                                                                                                  \
    static inline OT decimal_subtract_##LS##_##OS(LT left, LT right) {                                                 \
        return numeric_subtract_wrap_##OS(WIDEN(left), WIDEN(right));                                                  \
    }
DECIMAL_WIDENINGS(DEFINE_DECIMAL_SUM, )
#undef DEFINE_DECIMAL_SUM

//------------------------------------------------------------------------------
//
// Leaf Bodies
//
//------------------------------------------------------------------------------

//--------------------------------------
// Lanes
//--------------------------------------

#define CALL_PLAIN(FUNCTION, ...) FUNCTION(__VA_ARGS__)
#define CALL_BOUND(FUNCTION, ...) FUNCTION(binding, __VA_ARGS__)
#define CALL_SCALED(FUNCTION, ...) FUNCTION(decimal_scale, __VA_ARGS__)
#define CALL_PLAIN_BEGIN
#define CALL_BOUND_BEGIN
#define CALL_SCALED_BEGIN const int32_t decimal_scale = binding->left_decimal.type.scale

#define LANE_CHECKED(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                            \
    do {                                                                                                               \
        OT lane_value = {};                                                                                            \
        failure |= (uint32_t)CALL(FUNCTION, __VA_ARGS__, &lane_value);                                                 \
        carrier_store_##OS(out_values, (ROW), lane_value);                                                             \
    } while (0)

#define LANE_NULLING(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                            \
    do {                                                                                                               \
        OT lane_value = {};                                                                                            \
        failed |= (uint64_t)((uint32_t)CALL(FUNCTION, __VA_ARGS__, &lane_value) != 0U) << (BIT);                       \
        carrier_store_##OS(out_values, (ROW), lane_value);                                                             \
    } while (0)

#define LANE_UNCHECKED(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                          \
    carrier_store_##OS(out_values, (ROW), FUNCTION(__VA_ARGS__))

#define LANE_COMPUTED(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                           \
    carrier_store_##OS(out_values, (ROW), CALL(FUNCTION, __VA_ARGS__))

#define LANE_FLAGGED(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                            \
    do {                                                                                                               \
        OT lane_value = {};                                                                                            \
        lane_flags |= CALL(FUNCTION, __VA_ARGS__, &lane_value);                                                        \
        carrier_store_##OS(out_values, (ROW), lane_value);                                                             \
    } while (0)

#define LANE_FLAGGED_NULLING(CALL, FUNCTION, OT, OS, ROW, BIT, ...)                                                    \
    do {                                                                                                               \
        OT lane_value = {};                                                                                            \
        const FLAG_TYPE_##OS lane_flag = CALL(FUNCTION, __VA_ARGS__, &lane_value);                                     \
        failed |= ((uint64_t)lane_flag >> (sizeof(lane_flag) * 8U - 1U)) << (BIT);                                     \
        carrier_store_##OS(out_values, (ROW), lane_value);                                                             \
    } while (0)

#define FLAG_TYPE_u8 uint8_t
#define FLAG_TYPE_u16 uint16_t
#define FLAG_TYPE_u32 uint32_t
#define FLAG_TYPE_u64 uint64_t

#define LANE_CHECKED_BEGIN(S)
#define LANE_CHECKED_END
#define LANE_NULLING_BEGIN(S)
#define LANE_NULLING_END
#define LANE_UNCHECKED_BEGIN(S)
#define LANE_UNCHECKED_END
#define LANE_COMPUTED_BEGIN(S)
#define LANE_COMPUTED_END
#define LANE_FLAGGED_BEGIN(S) FLAG_TYPE_##S lane_flags = 0
#define LANE_FLAGGED_END                                                                                               \
    failure |= ((uint64_t)lane_flags >> (sizeof(lane_flags) * 8U - 1U)) != 0U ? (uint32_t)NUMERIC_FAILURE_OVERFLOW : 0U
#define LANE_FLAGGED_NULLING_BEGIN(S)
#define LANE_FLAGGED_NULLING_END

//--------------------------------------
// Operands
//--------------------------------------

#define LEFT_FLAT(S, ROW) type_load_##S(left_values, left_offset + (ROW))
#define LEFT_CONSTANT(S, ROW) left_constant
#define LEFT_GATHERED(S, ROW) type_load_##S(left_values, left_offset + left_rows[ROW])
#define LEFT_GENERIC(S, ROW) type_load_##S(left_values, kernel_row_index(left_offset, left_rows, left_mask, (ROW)))
#define RIGHT_FLAT(S, ROW) type_load_##S(right_values, right_offset + (ROW))
#define RIGHT_CONSTANT(S, ROW) right_constant
#define RIGHT_GATHERED(S, ROW) type_load_##S(right_values, right_offset + right_rows[ROW])
#define RIGHT_GENERIC(S, ROW) type_load_##S(right_values, kernel_row_index(right_offset, right_rows, right_mask, (ROW)))
#define SOURCE_FLAT(S, ROW) type_load_##S(source_values, source_offset + (ROW))
#define SOURCE_GENERIC(S, ROW) type_load_##S(source_values, source_offset + source_rows[ROW])
#define CARRIER_TYPE(S) typeof(type_load_##S(nullptr, 0))
#define BINARY_LANE(LANE, CALL, FUNCTION, LS, RS, OS, LEFT_AT, RIGHT_AT, ROW, BIT)                                     \
    LANE(CALL, FUNCTION, CARRIER_TYPE(OS), OS, ROW, BIT, LEFT_AT(LS, ROW), RIGHT_AT(RS, ROW))

#define BINARY_LOCALS(LS, RS)                                                                                          \
    const uint8_t* const left_values = left->values;                                                                   \
    const uint8_t* const right_values = right->values;                                                                 \
    const uint64_t left_offset = left->value_offset;                                                                   \
    const uint64_t right_offset = right->value_offset;                                                                 \
    const uint32_t* const left_rows = left->shape == KERNEL_SHAPE_GATHERED ? left->rows : nullptr;                     \
    const uint32_t* const right_rows = right->shape == KERNEL_SHAPE_GATHERED ? right->rows : nullptr;                  \
    const CARRIER_TYPE(LS) left_constant =                                                                             \
        left->shape == KERNEL_SHAPE_CONSTANT ? type_load_##LS(left_values, left_offset) : (CARRIER_TYPE(LS)){};        \
    const CARRIER_TYPE(RS) right_constant =                                                                            \
        right->shape == KERNEL_SHAPE_CONSTANT ? type_load_##RS(right_values, right_offset) : (CARRIER_TYPE(RS)){};     \
    uint8_t* const out_values = out_result->values;                                                                    \
    uint32_t failure = 0

#define BINARY_SPAN(LS, RS, OS, LEFT_AT, RIGHT_AT, LANE, CALL, FUNCTION)                                               \
    {                                                                                                                  \
        LANE##_BEGIN(OS);                                                                                              \
        for (uint32_t row = 0; row < count; ++row) {                                                                   \
            BINARY_LANE(LANE, CALL, FUNCTION, LS, RS, OS, LEFT_AT, RIGHT_AT, row, 0U);                                 \
        }                                                                                                              \
        LANE##_END;                                                                                                    \
    }

#define BINARY_TIGHT_ON(LS, RS, OS, LANE, CALL, FUNCTION)                                                              \
    if (left->validity == nullptr && right->validity == nullptr) {                                                     \
        BINARY_LOCALS(LS, RS);                                                                                         \
        CALL##_BEGIN;                                                                                                  \
        const enum kernel_shape_pair shapes =                                                                          \
            left->shape == KERNEL_SHAPE_CONSTANT && right->shape == KERNEL_SHAPE_CONSTANT                              \
                ? KERNEL_PAIR_FLAT_FLAT                                                                                \
                : kernel_shape_pair_of(left->shape, right->shape);                                                     \
        switch (shapes) {                                                                                              \
        case KERNEL_PAIR_FLAT_FLAT:                                                                                    \
            BINARY_SPAN(LS, RS, OS, LEFT_FLAT, RIGHT_FLAT, LANE, CALL, FUNCTION)                                       \
            break;                                                                                                     \
        case KERNEL_PAIR_FLAT_CONSTANT:                                                                                \
            BINARY_SPAN(LS, RS, OS, LEFT_FLAT, RIGHT_CONSTANT, LANE, CALL, FUNCTION)                                   \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_FLAT:                                                                                \
            BINARY_SPAN(LS, RS, OS, LEFT_CONSTANT, RIGHT_FLAT, LANE, CALL, FUNCTION)                                   \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_CONSTANT:                                                                            \
            break;                                                                                                     \
        case KERNEL_PAIR_GENERIC:                                                                                      \
            if (left->shape == KERNEL_SHAPE_GATHERED) {                                                                \
                if (right->shape == KERNEL_SHAPE_GATHERED) {                                                           \
                    BINARY_SPAN(LS, RS, OS, LEFT_GATHERED, RIGHT_GATHERED, LANE, CALL, FUNCTION)                       \
                } else if (right->shape == KERNEL_SHAPE_CONSTANT) {                                                    \
                    BINARY_SPAN(LS, RS, OS, LEFT_GATHERED, RIGHT_CONSTANT, LANE, CALL, FUNCTION)                       \
                } else {                                                                                               \
                    BINARY_SPAN(LS, RS, OS, LEFT_GATHERED, RIGHT_FLAT, LANE, CALL, FUNCTION)                           \
                }                                                                                                      \
            } else if (left->shape == KERNEL_SHAPE_CONSTANT) {                                                         \
                BINARY_SPAN(LS, RS, OS, LEFT_CONSTANT, RIGHT_GATHERED, LANE, CALL, FUNCTION)                           \
            } else {                                                                                                   \
                BINARY_SPAN(LS, RS, OS, LEFT_FLAT, RIGHT_GATHERED, LANE, CALL, FUNCTION)                               \
            }                                                                                                          \
            break;                                                                                                     \
        }                                                                                                              \
        out_result->failure_bits = (uint8_t)failure;                                                                   \
        return arithmetic_fill_all_valid(count, out_result);                                                           \
    }
#define BINARY_TIGHT_OFF(LS, RS, OS, LANE, CALL, FUNCTION)
#define LANE_CHECKED_TIGHT BINARY_TIGHT_ON
#define LANE_UNCHECKED_TIGHT BINARY_TIGHT_ON
#define LANE_COMPUTED_TIGHT BINARY_TIGHT_ON
#define LANE_FLAGGED_TIGHT BINARY_TIGHT_ON
#define LANE_NULLING_TIGHT BINARY_TIGHT_OFF
#define LANE_FLAGGED_NULLING_TIGHT BINARY_TIGHT_OFF

#define BINARY_WORD_LOOP(LS, RS, OS, LEFT_AT, RIGHT_AT, LANE, CALL, FUNCTION)                                          \
    for (uint32_t word = 0; word < word_count; ++word) {                                                               \
        const uint64_t active = merged[word];                                                                          \
        const uint32_t first = word * 64;                                                                              \
        uint64_t failed = 0;                                                                                           \
        LANE##_BEGIN(OS);                                                                                              \
        if (active == UINT64_MAX) {                                                                                    \
            for (uint32_t bit = 0; bit < 64; ++bit) {                                                                  \
                BINARY_LANE(LANE, CALL, FUNCTION, LS, RS, OS, LEFT_AT, RIGHT_AT, first + bit, bit);                    \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint64_t pending = active; pending != 0;) {                                                           \
                const uint32_t bit = dora_bitmap_word_pop_lowest_set(&pending);                                        \
                BINARY_LANE(LANE, CALL, FUNCTION, LS, RS, OS, LEFT_AT, RIGHT_AT, first + bit, bit);                    \
            }                                                                                                          \
        }                                                                                                              \
        LANE##_END;                                                                                                    \
        words_out[word] = active & ~failed;                                                                            \
    }

#define DEFINE_BINARY_WORD(NAME, LANE, CALL, FUNCTION, LS, RS, OS, PARAMETER)                                          \
    static int NAME##_words(const struct kernel_packet* left,                                                          \
                            const struct kernel_packet* right,                                                         \
                            PARAMETER,                                                                                 \
                            struct kernel_result* out_result) {                                                        \
        const uint32_t count = arithmetic_lane_count(left, right);                                                     \
        const size_t word_count = (size_t)bitmap_words_for_bits(count);                                                \
        uint64_t merged[KERNEL_CHUNK_WORD_COUNT];                                                                      \
        bitmap_prefix_words(count, merged, word_count);                                                                \
        if (left->shape != KERNEL_SHAPE_CONSTANT && left->validity != nullptr) {                                       \
            dora_bitmap_and_aligned_words(merged, left->validity, word_count);                                         \
        }                                                                                                              \
        if (right->shape != KERNEL_SHAPE_CONSTANT && right->validity != nullptr) {                                     \
            dora_bitmap_and_aligned_words(merged, right->validity, word_count);                                        \
        }                                                                                                              \
        uint64_t* const words_out = out_result->validity != nullptr ? out_result->validity : merged;                   \
        BINARY_LOCALS(LS, RS);                                                                                         \
        CALL##_BEGIN;                                                                                                  \
        const uint32_t left_mask = left->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                  \
        const uint32_t right_mask = right->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                \
        const enum kernel_shape_pair shapes =                                                                          \
            left->shape == KERNEL_SHAPE_CONSTANT && right->shape == KERNEL_SHAPE_CONSTANT                              \
                ? KERNEL_PAIR_FLAT_FLAT                                                                                \
                : kernel_shape_pair_of(left->shape, right->shape);                                                     \
        switch (shapes) {                                                                                              \
        case KERNEL_PAIR_FLAT_FLAT:                                                                                    \
            BINARY_WORD_LOOP(LS, RS, OS, LEFT_FLAT, RIGHT_FLAT, LANE, CALL, FUNCTION)                                  \
            break;                                                                                                     \
        case KERNEL_PAIR_FLAT_CONSTANT:                                                                                \
            BINARY_WORD_LOOP(LS, RS, OS, LEFT_FLAT, RIGHT_CONSTANT, LANE, CALL, FUNCTION)                              \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_FLAT:                                                                                \
            BINARY_WORD_LOOP(LS, RS, OS, LEFT_CONSTANT, RIGHT_FLAT, LANE, CALL, FUNCTION)                              \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_CONSTANT:                                                                            \
        case KERNEL_PAIR_GENERIC:                                                                                      \
            BINARY_WORD_LOOP(LS, RS, OS, LEFT_GENERIC, RIGHT_GENERIC, LANE, CALL, FUNCTION)                            \
            break;                                                                                                     \
        }                                                                                                              \
        out_result->null_count = count - (uint32_t)dora_bitmap_count_set_words(words_out, count);                      \
        out_result->failure_bits = (uint8_t)failure;                                                                   \
        return DORA_OK;                                                                                                \
    }

#define CALL_PLAIN_BINARY(NAME, LANE, CALL, FUNCTION, LS, RS, OS)                                                      \
    DEFINE_BINARY_WORD(NAME, LANE, CALL, FUNCTION, LS, RS, OS, const struct kernel_binary_binding*)
#define CALL_BOUND_BINARY(NAME, LANE, CALL, FUNCTION, LS, RS, OS)                                                      \
    DEFINE_BINARY_WORD(NAME, LANE, CALL, FUNCTION, LS, RS, OS, const struct kernel_binary_binding* binding)
#define CALL_SCALED_BINARY(NAME, LANE, CALL, FUNCTION, LS, RS, OS)                                                     \
    DEFINE_BINARY_WORD(NAME, LANE, CALL, FUNCTION, LS, RS, OS, const struct kernel_binary_binding* binding)

#define DEFINE_BINARY_LEAF(NAME, LANE, CALL, FUNCTION, LS, RS, OS)                                                     \
    CALL##_BINARY(NAME, LANE, CALL, FUNCTION, LS, RS, OS) static int NAME(const struct kernel_packet* left,            \
                                                                          const struct kernel_packet* right,           \
                                                                          const struct kernel_binary_binding* binding, \
                                                                          struct kernel_result* out_result) {          \
        const uint32_t count = arithmetic_lane_count(left, right);                                                     \
        if (count == 0) {                                                                                              \
            out_result->failure_bits = 0;                                                                              \
            return arithmetic_fill_all_valid(count, out_result);                                                       \
        }                                                                                                              \
        if ((left->shape == KERNEL_SHAPE_CONSTANT && !kernel_packet_is_valid_constant(left)) ||                        \
            (right->shape == KERNEL_SHAPE_CONSTANT && !kernel_packet_is_valid_constant(right))) {                      \
            return arithmetic_fill_null(count, out_result);                                                            \
        }                                                                                                              \
        LANE##_TIGHT(LS, RS, OS, LANE, CALL, FUNCTION) return NAME##_words(left, right, binding, out_result);          \
    }

#define UNARY_SPAN(T, S, LANE, CALL, FUNCTION, SOURCE_AT)                                                              \
    {                                                                                                                  \
        LANE##_BEGIN(S);                                                                                               \
        for (uint32_t row = 0; row < count; ++row) {                                                                   \
            LANE(CALL, FUNCTION, T, S, row, 0U, SOURCE_AT(S, row));                                                    \
        }                                                                                                              \
        LANE##_END;                                                                                                    \
    }

#define UNARY_TIGHT_ON(T, S, LANE, CALL, FUNCTION)                                                                     \
    if (source->validity == nullptr) {                                                                                 \
        const uint8_t* const source_values = source->values;                                                           \
        const uint64_t source_offset = source->value_offset;                                                           \
        const uint32_t* const source_rows = source->rows;                                                              \
        uint8_t* const out_values = out_result->values;                                                                \
        uint32_t failure = 0;                                                                                          \
        if (source->shape == KERNEL_SHAPE_GATHERED) {                                                                  \
            UNARY_SPAN(T, S, LANE, CALL, FUNCTION, SOURCE_GENERIC)                                                     \
        } else {                                                                                                       \
            UNARY_SPAN(T, S, LANE, CALL, FUNCTION, SOURCE_FLAT)                                                        \
        }                                                                                                              \
        out_result->failure_bits = (uint8_t)failure;                                                                   \
        return arithmetic_fill_all_valid(count, out_result);                                                           \
    }
#define UNARY_TIGHT_OFF(T, S, LANE, CALL, FUNCTION)
#define LANE_CHECKED_UNARY_TIGHT UNARY_TIGHT_ON
#define LANE_UNCHECKED_UNARY_TIGHT UNARY_TIGHT_ON
#define LANE_FLAGGED_UNARY_TIGHT UNARY_TIGHT_ON
#define LANE_NULLING_UNARY_TIGHT UNARY_TIGHT_OFF
#define LANE_FLAGGED_NULLING_UNARY_TIGHT UNARY_TIGHT_OFF

#define UNARY_WORD_LOOP(T, S, LANE, CALL, FUNCTION, SOURCE_AT)                                                         \
    for (uint32_t word = 0; word < word_count; ++word) {                                                               \
        const uint64_t active = merged[word];                                                                          \
        const uint32_t first = word * 64;                                                                              \
        uint64_t failed = 0;                                                                                           \
        LANE##_BEGIN(S);                                                                                               \
        if (active == UINT64_MAX) {                                                                                    \
            for (uint32_t bit = 0; bit < 64; ++bit) {                                                                  \
                LANE(CALL, FUNCTION, T, S, first + bit, bit, SOURCE_AT(S, first + bit));                               \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint64_t pending = active; pending != 0;) {                                                           \
                const uint32_t bit = dora_bitmap_word_pop_lowest_set(&pending);                                        \
                LANE(CALL, FUNCTION, T, S, first + bit, bit, SOURCE_AT(S, first + bit));                               \
            }                                                                                                          \
        }                                                                                                              \
        LANE##_END;                                                                                                    \
        words_out[word] = active & ~failed;                                                                            \
    }

#define DEFINE_UNARY_WORD(NAME, LANE, CALL, FUNCTION, T, S)                                                            \
    static int NAME##_words(const struct kernel_packet* source,                                                        \
                            const struct kernel_unary_binding*,                                                        \
                            struct kernel_result* out_result) {                                                        \
        const uint32_t count = source->count == 0 ? 0 : source->shape == KERNEL_SHAPE_CONSTANT ? 1 : source->count;    \
        const size_t word_count = (size_t)bitmap_words_for_bits(count);                                                \
        uint64_t merged[KERNEL_CHUNK_WORD_COUNT];                                                                      \
        bitmap_prefix_words(count, merged, word_count);                                                                \
        if (source->shape != KERNEL_SHAPE_CONSTANT && source->validity != nullptr) {                                   \
            dora_bitmap_and_aligned_words(merged, source->validity, word_count);                                       \
        }                                                                                                              \
        uint64_t* const words_out = out_result->validity != nullptr ? out_result->validity : merged;                   \
        const uint8_t* const source_values = source->values;                                                           \
        const uint64_t source_offset = source->value_offset;                                                           \
        const uint32_t* const source_rows = source->rows;                                                              \
        uint8_t* const out_values = out_result->values;                                                                \
        uint32_t failure = 0;                                                                                          \
        if (source->shape == KERNEL_SHAPE_GATHERED) {                                                                  \
            UNARY_WORD_LOOP(T, S, LANE, CALL, FUNCTION, SOURCE_GENERIC)                                                \
        } else {                                                                                                       \
            UNARY_WORD_LOOP(T, S, LANE, CALL, FUNCTION, SOURCE_FLAT)                                                   \
        }                                                                                                              \
        out_result->null_count = count - (uint32_t)dora_bitmap_count_set_words(words_out, count);                      \
        out_result->failure_bits = (uint8_t)failure;                                                                   \
        return DORA_OK;                                                                                                \
    }

#define DEFINE_UNARY_LEAF(NAME, LANE, CALL, FUNCTION, T, S)                                                            \
    DEFINE_UNARY_WORD(NAME, LANE, CALL, FUNCTION, T, S)                                                                \
    static int NAME(const struct kernel_packet* source,                                                                \
                    const struct kernel_unary_binding* binding,                                                        \
                    struct kernel_result* out_result) {                                                                \
        const uint32_t count = source->count == 0 ? 0 : source->shape == KERNEL_SHAPE_CONSTANT ? 1 : source->count;    \
        if (count == 0) {                                                                                              \
            out_result->failure_bits = 0;                                                                              \
            return arithmetic_fill_all_valid(count, out_result);                                                       \
        }                                                                                                              \
        if (source->shape == KERNEL_SHAPE_CONSTANT && !kernel_packet_is_valid_constant(source)) {                      \
            return arithmetic_fill_null(count, out_result);                                                            \
        }                                                                                                              \
        LANE##_UNARY_TIGHT(T, S, LANE, CALL, FUNCTION) return NAME##_words(source, binding, out_result);               \
    }

//------------------------------------------------------------------------------
//
// Leaves
//
//------------------------------------------------------------------------------

//--------------------------------------
// Integers
//--------------------------------------

#define ARITHMETIC_INTEGERS(X, ARGUMENT)                                                                               \
    X(ARGUMENT, i8, int8_t, I8, checked, checked)                                                                      \
    X(ARGUMENT, i16, int16_t, I16, checked, checked)                                                                   \
    X(ARGUMENT, i32, int32_t, I32, checked, checked)                                                                   \
    X(ARGUMENT, i64, int64_t, I64, checked, checked)                                                                   \
    X(ARGUMENT, i128, struct dora_i128, I128, checked, checked)                                                        \
    X(ARGUMENT, i256, struct dora_i256, I256, checked, checked)                                                        \
    X(ARGUMENT, u8, uint8_t, U8, flag, flag)                                                                           \
    X(ARGUMENT, u16, uint16_t, U16, flag, flag)                                                                        \
    X(ARGUMENT, u32, uint32_t, U32, flag, flag)                                                                        \
    X(ARGUMENT, u64, uint64_t, U64, flag, flag)                                                                        \
    X(ARGUMENT, u128, struct dora_u128, U128, checked, checked)

#define LANE_ERROR_flag LANE_FLAGGED
#define LANE_ERROR_checked LANE_CHECKED
#define LANE_NULL_flag LANE_FLAGGED_NULLING
#define LANE_NULL_checked LANE_NULLING
#define EXPANDED_BINARY_LEAF(...) DEFINE_BINARY_LEAF(__VA_ARGS__)
#define EXPANDED_UNARY_LEAF(...) DEFINE_UNARY_LEAF(__VA_ARGS__)

#define DEFINE_INTEGER_BINARY(OP, KIND, CHECKED, S, T)                                                                 \
    EXPANDED_BINARY_LEAF(arithmetic_##OP##_##S##_error, LANE_ERROR_##KIND, CALL_PLAIN, CHECKED, S, S, S)               \
    EXPANDED_BINARY_LEAF(arithmetic_##OP##_##S##_null, LANE_NULL_##KIND, CALL_PLAIN, CHECKED, S, S, S)

#define DEFINE_INTEGER_WRAP(OP, S, T)                                                                                  \
    DEFINE_BINARY_LEAF(arithmetic_##OP##_##S##_wrap, LANE_UNCHECKED, CALL_PLAIN, numeric_##OP##_wrap_##S, S, S, S)

#define DEFINE_INTEGER_UNARY(OP, KIND, S, T)                                                                           \
    EXPANDED_UNARY_LEAF(arithmetic_##OP##_##S##_error,                                                                 \
                        LANE_ERROR_##KIND,                                                                             \
                        CALL_PLAIN,                                                                                    \
                        numeric_##OP##_##KIND##_##S,                                                                   \
                        T,                                                                                             \
                        S)                                                                                             \
    EXPANDED_UNARY_LEAF(arithmetic_##OP##_##S##_null, LANE_NULL_##KIND, CALL_PLAIN, numeric_##OP##_##KIND##_##S, T, S) \
    DEFINE_UNARY_LEAF(arithmetic_##OP##_##S##_wrap, LANE_UNCHECKED, CALL_PLAIN, numeric_##OP##_wrap_##S, T, S)

// Signed integers, unsigned integers
#define DEFINE_INTEGER_LEAVES(ARGUMENT, S, T, P, KIND, UNARY_KIND)                                                     \
    DEFINE_INTEGER_BINARY(add, KIND, numeric_add_##KIND##_##S, S, T)                                                   \
    DEFINE_INTEGER_BINARY(subtract, KIND, numeric_subtract_##KIND##_##S, S, T)                                         \
    DEFINE_INTEGER_BINARY(multiply, checked, numeric_multiply_checked_##S, S, T)                                       \
    DEFINE_INTEGER_BINARY(divide, checked, numeric_divide_##S, S, T)                                                   \
    DEFINE_INTEGER_BINARY(remainder, checked, numeric_remainder_##S, S, T)                                             \
    DEFINE_INTEGER_WRAP(add, S, T)                                                                                     \
    DEFINE_INTEGER_WRAP(subtract, S, T)                                                                                \
    DEFINE_INTEGER_WRAP(multiply, S, T)                                                                                \
    DEFINE_INTEGER_UNARY(negate, UNARY_KIND, S, T)                                                                     \
    DEFINE_INTEGER_UNARY(abs, UNARY_KIND, S, T)
ARITHMETIC_INTEGERS(DEFINE_INTEGER_LEAVES, )
#undef DEFINE_INTEGER_LEAVES
#undef DEFINE_INTEGER_UNARY
#undef DEFINE_INTEGER_WRAP
#undef DEFINE_INTEGER_BINARY

//--------------------------------------
// Floating Point
//--------------------------------------

#define ARITHMETIC_FLOATS(X, ARGUMENT)                                                                                 \
    X(ARGUMENT, f16, struct dora_f16_bits, F16)                                                                        \
    X(ARGUMENT, f32, float, F32)                                                                                       \
    X(ARGUMENT, f64, double, F64)

#define DEFINE_FLOAT_BINARY(OP, S, T)                                                                                  \
    DEFINE_BINARY_LEAF(arithmetic_##OP##_##S, LANE_UNCHECKED, CALL_PLAIN, numeric_##OP##_##S, S, S, S)

#define DEFINE_FLOAT_UNARY(OP, S, T)                                                                                   \
    DEFINE_UNARY_LEAF(arithmetic_##OP##_##S, LANE_UNCHECKED, CALL_PLAIN, numeric_##OP##_##S, T, S)

// Float16, Float32, Float64
#define DEFINE_FLOAT_LEAVES(ARGUMENT, S, T, P)                                                                         \
    DEFINE_FLOAT_BINARY(add, S, T)                                                                                     \
    DEFINE_FLOAT_BINARY(subtract, S, T)                                                                                \
    DEFINE_FLOAT_BINARY(multiply, S, T)                                                                                \
    DEFINE_FLOAT_BINARY(divide, S, T)                                                                                  \
    DEFINE_FLOAT_BINARY(remainder, S, T)                                                                               \
    DEFINE_FLOAT_UNARY(negate, S, T)                                                                                   \
    DEFINE_FLOAT_UNARY(abs, S, T)
ARITHMETIC_FLOATS(DEFINE_FLOAT_LEAVES, )
#undef DEFINE_FLOAT_LEAVES
#undef DEFINE_FLOAT_UNARY
#undef DEFINE_FLOAT_BINARY

//--------------------------------------
// Decimals
//--------------------------------------

#define DEFINE_DECIMAL_BINARY(OP, S, T)                                                                                \
    DEFINE_BINARY_LEAF(arithmetic_##OP##_decimal_##S##_error, LANE_CHECKED, CALL_BOUND, decimal_##OP##_##S, S, S, S)   \
    DEFINE_BINARY_LEAF(arithmetic_##OP##_decimal_##S##_null, LANE_NULLING, CALL_BOUND, decimal_##OP##_##S, S, S, S)

// Decimal (32/64/128/256-bit)
#define DEFINE_DECIMAL_LEAVES(ARGUMENT, S, T, P) DEFINE_DECIMAL_BINARY(remainder, S, T)
DECIMAL_WIDTHS(DEFINE_DECIMAL_LEAVES, )
DEFINE_DECIMAL_BINARY(multiply, i256, struct dora_i256)
DEFINE_DECIMAL_BINARY(divide, i256, struct dora_i256)
#undef DEFINE_DECIMAL_LEAVES

// Decimal (128/256-bit)
#define DEFINE_DECIMAL_CAPPED_LEAVES(ARGUMENT, S, T, P)                                                                \
    DEFINE_DECIMAL_BINARY(add, S, T)                                                                                   \
    DEFINE_DECIMAL_BINARY(subtract, S, T)
DECIMAL_CAPPED(DEFINE_DECIMAL_CAPPED_LEAVES, )
#undef DEFINE_DECIMAL_CAPPED_LEAVES

// Decimal (32/64/128-bit)
#define DEFINE_DECIMAL_QUOTIENT_LEAF(S, T)                                                                             \
    DEFINE_BINARY_LEAF(arithmetic_divide_decimal_##S, LANE_COMPUTED, CALL_SCALED, decimal_quotient_##S, S, S, f64)
DEFINE_DECIMAL_QUOTIENT_LEAF(i32, int32_t)
DEFINE_DECIMAL_QUOTIENT_LEAF(i64, int64_t)
DEFINE_DECIMAL_QUOTIENT_LEAF(i128, struct dora_i128)
#undef DEFINE_DECIMAL_QUOTIENT_LEAF

// Decimal (32/64/128-bit); widening: 32 to 64, 64 to 128
#define DEFINE_DECIMAL_PRODUCT_LEAVES(ARGUMENT, LS, LT, OS, OT, WIDEN)                                                 \
    DEFINE_BINARY_LEAF(arithmetic_multiply_decimal_##LS##_##OS##_error,                                                \
                       LANE_CHECKED,                                                                                   \
                       CALL_BOUND,                                                                                     \
                       decimal_multiply_##LS##_##OS,                                                                   \
                       LS,                                                                                             \
                       LS,                                                                                             \
                       OS)                                                                                             \
    DEFINE_BINARY_LEAF(arithmetic_multiply_decimal_##LS##_##OS##_null,                                                 \
                       LANE_NULLING,                                                                                   \
                       CALL_BOUND,                                                                                     \
                       decimal_multiply_##LS##_##OS,                                                                   \
                       LS,                                                                                             \
                       LS,                                                                                             \
                       OS)
DECIMAL_PRODUCTS(DEFINE_DECIMAL_PRODUCT_LEAVES, )
#undef DEFINE_DECIMAL_PRODUCT_LEAVES

// Decimal; widening: 32 to 64, 64 to 128 bits
#define DEFINE_DECIMAL_SUM_LEAVES(ARGUMENT, LS, LT, OS, OT, WIDEN)                                                     \
    DEFINE_BINARY_LEAF(arithmetic_add_decimal_##LS##_##OS,                                                             \
                       LANE_UNCHECKED,                                                                                 \
                       CALL_PLAIN,                                                                                     \
                       decimal_add_##LS##_##OS,                                                                        \
                       LS,                                                                                             \
                       LS,                                                                                             \
                       OS)                                                                                             \
    DEFINE_BINARY_LEAF(arithmetic_subtract_decimal_##LS##_##OS,                                                        \
                       LANE_UNCHECKED,                                                                                 \
                       CALL_PLAIN,                                                                                     \
                       decimal_subtract_##LS##_##OS,                                                                   \
                       LS,                                                                                             \
                       LS,                                                                                             \
                       OS)
DECIMAL_WIDENINGS(DEFINE_DECIMAL_SUM_LEAVES, )
#undef DEFINE_DECIMAL_SUM_LEAVES
#undef DEFINE_DECIMAL_BINARY

//--------------------------------------
// Temporal
//--------------------------------------

#define TEMPORAL_OPERATIONS(X, ARGUMENT)                                                                               \
    X(ARGUMENT, add, numeric_add_checked_i64)                                                                          \
    X(ARGUMENT, subtract, numeric_subtract_checked_i64)

#define DEFINE_TEMPORAL_LANES(ARGUMENT, OP, CHECKED)                                                                   \
    static inline uint32_t temporal_date32_##OP(int64_t left, int64_t right, int32_t* out_value) {                     \
        int64_t value = 0;                                                                                             \
        const bool fits = !CHECKED(left, right, &value) && value >= INT32_MIN && value <= INT32_MAX;                   \
        *out_value = fits ? (int32_t)value : 0;                                                                        \
        return fits ? 0 : NUMERIC_FAILURE_OVERFLOW;                                                                    \
    }                                                                                                                  \
    static inline uint32_t temporal_time32_##OP(const struct kernel_binary_binding* binding,                           \
                                                int64_t left,                                                          \
                                                int64_t right,                                                         \
                                                int32_t* out_value) {                                                  \
        int64_t value = 0;                                                                                             \
        const bool fits = !CHECKED(left, right, &value) && value >= 0 && value < binding->ticks_per_day;               \
        *out_value = fits ? (int32_t)value : 0;                                                                        \
        return fits ? 0 : NUMERIC_FAILURE_RANGE;                                                                       \
    }                                                                                                                  \
    static inline uint32_t temporal_time64_##OP(const struct kernel_binary_binding* binding,                           \
                                                int64_t left,                                                          \
                                                int64_t right,                                                         \
                                                int64_t* out_value) {                                                  \
        int64_t value = 0;                                                                                             \
        const bool fits = !CHECKED(left, right, &value) && value >= 0 && value < binding->ticks_per_day;               \
        *out_value = fits ? value : 0;                                                                                 \
        return fits ? 0 : NUMERIC_FAILURE_RANGE;                                                                       \
    }
TEMPORAL_OPERATIONS(DEFINE_TEMPORAL_LANES, )
#undef DEFINE_TEMPORAL_LANES

#define DEFINE_TEMPORAL_LEAF(NAME, CALL, FUNCTION, LS, RS, OS)                                                         \
    DEFINE_BINARY_LEAF(temporal_##NAME##_error, LANE_CHECKED, CALL, FUNCTION, LS, RS, OS)                              \
    DEFINE_BINARY_LEAF(temporal_##NAME##_null, LANE_NULLING, CALL, FUNCTION, LS, RS, OS)

#define TEMPORAL_DELTAS(X, ARGUMENT)                                                                                   \
    X(ARGUMENT, i8, int8_t)                                                                                            \
    X(ARGUMENT, i16, int16_t)                                                                                          \
    X(ARGUMENT, i32, int32_t)                                                                                          \
    X(ARGUMENT, i64, int64_t)

// Date32 (days) with Int8, Int16, Int32, Int64 day counts
#define DEFINE_DATE_LEAVES(ARGUMENT, S, T)                                                                             \
    DEFINE_TEMPORAL_LEAF(add_date32_##S, CALL_PLAIN, temporal_date32_add, i32, S, i32)                                 \
    DEFINE_TEMPORAL_LEAF(subtract_date32_##S, CALL_PLAIN, temporal_date32_subtract, i32, S, i32)                       \
    DEFINE_TEMPORAL_LEAF(add_##S##_date32, CALL_PLAIN, temporal_date32_add, S, i32, i32)
TEMPORAL_DELTAS(DEFINE_DATE_LEAVES, )
#undef DEFINE_DATE_LEAVES

// Time with Duration (matching units)
DEFINE_TEMPORAL_LEAF(add_time_i32_duration, CALL_BOUND, temporal_time32_add, i32, i64, i32)
DEFINE_TEMPORAL_LEAF(subtract_time_i32_duration, CALL_BOUND, temporal_time32_subtract, i32, i64, i32)
DEFINE_TEMPORAL_LEAF(add_duration_time_i32, CALL_BOUND, temporal_time32_add, i64, i32, i32)
DEFINE_TEMPORAL_LEAF(add_time_i64_duration, CALL_BOUND, temporal_time64_add, i64, i64, i64)
DEFINE_TEMPORAL_LEAF(subtract_time_i64_duration, CALL_BOUND, temporal_time64_subtract, i64, i64, i64)
DEFINE_TEMPORAL_LEAF(add_duration_time_i64, CALL_BOUND, temporal_time64_add, i64, i64, i64)

// Timestamp with Duration, Timestamp minus Timestamp, Duration with Duration
DEFINE_TEMPORAL_LEAF(add_ticks_i64, CALL_PLAIN, numeric_add_checked_i64, i64, i64, i64)
DEFINE_TEMPORAL_LEAF(subtract_ticks_i64, CALL_PLAIN, numeric_subtract_checked_i64, i64, i64, i64)
#undef DEFINE_TEMPORAL_LEAF
#undef TEMPORAL_DELTAS
#undef TEMPORAL_OPERATIONS
#undef EXPANDED_UNARY_LEAF
#undef EXPANDED_BINARY_LEAF
#undef LANE_NULL_checked
#undef LANE_NULL_flag
#undef LANE_ERROR_checked
#undef LANE_ERROR_flag
#undef DEFINE_UNARY_LEAF
#undef DEFINE_BINARY_LEAF
#undef DEFINE_BINARY_WORD
#undef DEFINE_UNARY_WORD
#undef CALL_PLAIN_BINARY
#undef CALL_BOUND_BINARY
#undef CALL_SCALED_BINARY
#undef BINARY_LOCALS
#undef BINARY_TIGHT_ON
#undef BINARY_TIGHT_OFF
#undef BINARY_WORD_LOOP
#undef LEFT_GATHERED
#undef RIGHT_GATHERED
#undef UNARY_SPAN
#undef UNARY_WORD_LOOP
#undef UNARY_TIGHT_ON
#undef UNARY_TIGHT_OFF
#undef LANE_CHECKED_UNARY_TIGHT
#undef LANE_UNCHECKED_UNARY_TIGHT
#undef LANE_FLAGGED_UNARY_TIGHT
#undef LANE_NULLING_UNARY_TIGHT
#undef LANE_FLAGGED_NULLING_UNARY_TIGHT
#undef CALL_SCALED
#undef CALL_PLAIN_BEGIN
#undef CALL_BOUND_BEGIN
#undef CALL_SCALED_BEGIN
#undef BINARY_SPAN
#undef LANE_CHECKED_TIGHT
#undef LANE_UNCHECKED_TIGHT
#undef LANE_COMPUTED_TIGHT
#undef LANE_FLAGGED_TIGHT
#undef LANE_NULLING_TIGHT
#undef LANE_FLAGGED_NULLING_TIGHT
#undef BINARY_LANE
#undef CARRIER_TYPE
#undef SOURCE_GENERIC
#undef SOURCE_FLAT
#undef RIGHT_GENERIC
#undef RIGHT_CONSTANT
#undef RIGHT_FLAT
#undef LEFT_GENERIC
#undef LEFT_CONSTANT
#undef LEFT_FLAT
#undef LANE_FLAGGED_NULLING_END
#undef LANE_FLAGGED_NULLING_BEGIN
#undef LANE_FLAGGED_END
#undef LANE_FLAGGED_BEGIN
#undef LANE_COMPUTED_END
#undef LANE_COMPUTED_BEGIN
#undef LANE_UNCHECKED_END
#undef LANE_UNCHECKED_BEGIN
#undef LANE_NULLING_END
#undef LANE_NULLING_BEGIN
#undef LANE_CHECKED_END
#undef LANE_CHECKED_BEGIN
#undef FLAG_TYPE_u64
#undef FLAG_TYPE_u32
#undef FLAG_TYPE_u16
#undef FLAG_TYPE_u8
#undef LANE_FLAGGED_NULLING
#undef LANE_FLAGGED
#undef LANE_COMPUTED
#undef LANE_UNCHECKED
#undef LANE_NULLING
#undef LANE_CHECKED
#undef CALL_BOUND
#undef CALL_PLAIN

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

//--------------------------------------
// Tables
//--------------------------------------

enum : uint32_t {
    ARITHMETIC_BINARY_COUNT = KERNEL_OPERATION_NEGATE,
    ARITHMETIC_UNARY_COUNT = KERNEL_OPERATION_COUNT - KERNEL_OPERATION_NEGATE,
    ARITHMETIC_PHYSICAL_COUNT = DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO + 1,
    ARITHMETIC_POLICY_COUNT = DORA_FAILURE_POLICY_WRAP + 1
};

#define WRAPPING_ENTRY(OP, S, T, P, ...)                                                                               \
    [DORA_PHYSICAL_TYPE_##P] = {                                                                                       \
        [DORA_FAILURE_POLICY_ERROR] = arithmetic_##OP##_##S##_error,                                                   \
        [DORA_FAILURE_POLICY_NULL] = arithmetic_##OP##_##S##_null,                                                     \
        [DORA_FAILURE_POLICY_WRAP] = arithmetic_##OP##_##S##_wrap                                                      \
    },
#define CHECKED_ENTRY(OP, S, T, P, ...)                                                                                \
    [DORA_PHYSICAL_TYPE_##P] = {                                                                                       \
        [DORA_FAILURE_POLICY_ERROR] = arithmetic_##OP##_##S##_error,                                                   \
        [DORA_FAILURE_POLICY_NULL] = arithmetic_##OP##_##S##_null,                                                     \
        [DORA_FAILURE_POLICY_WRAP] = nullptr                                                                           \
    },
#define FLOAT_ENTRY(OP, S, T, P)                                                                                       \
    [DORA_PHYSICAL_TYPE_##P] = {                                                                                       \
        [DORA_FAILURE_POLICY_ERROR] = arithmetic_##OP##_##S,                                                           \
        [DORA_FAILURE_POLICY_NULL] = arithmetic_##OP##_##S,                                                            \
        [DORA_FAILURE_POLICY_WRAP] = nullptr                                                                           \
    },
#define DECIMAL_ENTRY(OP, S, T, P)                                                                                     \
    [P] = {                                                                                                            \
        [DORA_FAILURE_POLICY_ERROR] = arithmetic_##OP##_decimal_##S##_error,                                           \
        [DORA_FAILURE_POLICY_NULL] = arithmetic_##OP##_decimal_##S##_null                                              \
    },

static typeof(&arithmetic_add_i64_error)
    const arithmetic_binary_leaves[ARITHMETIC_BINARY_COUNT][ARITHMETIC_PHYSICAL_COUNT][ARITHMETIC_POLICY_COUNT] = {
        [KERNEL_OPERATION_ADD] = {ARITHMETIC_INTEGERS(WRAPPING_ENTRY, add) ARITHMETIC_FLOATS(FLOAT_ENTRY, add)},
        [KERNEL_OPERATION_SUBTRACT] = {ARITHMETIC_INTEGERS(WRAPPING_ENTRY, subtract)
                                           ARITHMETIC_FLOATS(FLOAT_ENTRY, subtract)},
        [KERNEL_OPERATION_MULTIPLY] = {ARITHMETIC_INTEGERS(WRAPPING_ENTRY, multiply)
                                           ARITHMETIC_FLOATS(FLOAT_ENTRY, multiply)},
        [KERNEL_OPERATION_DIVIDE] = {ARITHMETIC_INTEGERS(CHECKED_ENTRY, divide) ARITHMETIC_FLOATS(FLOAT_ENTRY, divide)},
        [KERNEL_OPERATION_REMAINDER] = {ARITHMETIC_INTEGERS(CHECKED_ENTRY, remainder)
                                            ARITHMETIC_FLOATS(FLOAT_ENTRY, remainder)},
};

static typeof(&arithmetic_add_decimal_i128_error)
    const arithmetic_decimal_leaves[ARITHMETIC_BINARY_COUNT][ARITHMETIC_PHYSICAL_COUNT][ARITHMETIC_POLICY_COUNT] = {
        [KERNEL_OPERATION_ADD] = {DECIMAL_CAPPED(DECIMAL_ENTRY, add)},
        [KERNEL_OPERATION_SUBTRACT] = {DECIMAL_CAPPED(DECIMAL_ENTRY, subtract)},
        [KERNEL_OPERATION_MULTIPLY] = {DECIMAL_ENTRY(multiply, i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256)},
        [KERNEL_OPERATION_DIVIDE] = {DECIMAL_ENTRY(divide, i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256)},
        [KERNEL_OPERATION_REMAINDER] = {DECIMAL_WIDTHS(DECIMAL_ENTRY, remainder)},
};

static typeof(&arithmetic_divide_decimal_i64) const arithmetic_decimal_quotients[ARITHMETIC_PHYSICAL_COUNT] = {
    [DORA_PHYSICAL_TYPE_I32] = arithmetic_divide_decimal_i32,
    [DORA_PHYSICAL_TYPE_I64] = arithmetic_divide_decimal_i64,
    [DORA_PHYSICAL_TYPE_I128] = arithmetic_divide_decimal_i128,
};

static typeof(&arithmetic_negate_i64_error)
    const arithmetic_unary_leaves[ARITHMETIC_UNARY_COUNT][ARITHMETIC_PHYSICAL_COUNT][ARITHMETIC_POLICY_COUNT] = {
        {ARITHMETIC_INTEGERS(WRAPPING_ENTRY, negate) ARITHMETIC_FLOATS(FLOAT_ENTRY, negate)},
        {ARITHMETIC_INTEGERS(WRAPPING_ENTRY, abs) ARITHMETIC_FLOATS(FLOAT_ENTRY, abs)},
};

#undef DECIMAL_ENTRY
#undef FLOAT_ENTRY
#undef CHECKED_ENTRY
#undef WRAPPING_ENTRY
#undef DECIMAL_CAPPED
#undef DECIMAL_WIDTH_ROW
#undef DECIMAL_WIDTHS
#undef ARITHMETIC_FLOATS
#undef ARITHMETIC_INTEGERS

//--------------------------------------
// Carriers
//--------------------------------------

struct arithmetic_carrier {
    struct decimal_prepared decimal;
    enum dora_type id;
    enum dora_physical_type physical;
    uint32_t width;
};

static int arithmetic_carrier(const struct dora_logical_type* type, struct arithmetic_carrier* out_carrier) {
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    const int status = type_scalar_physical(type, &physical);
    if (status != DORA_OK) {
        return status;
    }
    struct arithmetic_carrier carrier = {.id = type->id, .physical = physical, .width = physical_width(physical)};
    if (type->id == DORA_TYPE_DECIMAL) {
        const int status = dora_decimal_prepare(type->parameters.decimal, physical, &carrier.decimal);
        if (status != DORA_OK) {
            return status;
        }
    } else {
        if (!((type->id >= DORA_TYPE_INT8 && type->id <= DORA_TYPE_FLOAT64) || type->id == DORA_TYPE_DURATION)) {
            return ENOTSUP;
        }
    }
    *out_carrier = carrier;
    return DORA_OK;
}

static int arithmetic_admit(const struct arithmetic_carrier* operand, const struct arithmetic_carrier* result) {
    if (operand->id != result->id || operand->physical != result->physical) {
        return ENOTSUP;
    }
    if (operand->id == DORA_TYPE_DECIMAL && (operand->decimal.type.precision != result->decimal.type.precision ||
                                             operand->decimal.type.scale != result->decimal.type.scale)) {
        return EINVAL;
    }
    return DORA_OK;
}

static int arithmetic_policy(const struct arithmetic_carrier* carrier,
                             enum kernel_operation operation,
                             enum dora_failure_policy policy) {
    const bool floating = carrier->physical >= DORA_PHYSICAL_TYPE_F16 && carrier->physical <= DORA_PHYSICAL_TYPE_F64;
    const bool exact = carrier->id == DORA_TYPE_DECIMAL || carrier->id == DORA_TYPE_DURATION;
    if (carrier->id == DORA_TYPE_DURATION && operation != KERNEL_OPERATION_ADD &&
        operation != KERNEL_OPERATION_SUBTRACT) {
        return ENOTSUP;
    }
    if (policy == DORA_FAILURE_POLICY_WRAP &&
        (floating || exact || operation == KERNEL_OPERATION_DIVIDE || operation == KERNEL_OPERATION_REMAINDER)) {
        return EINVAL;
    }
    return DORA_OK;
}

//--------------------------------------
// Decimal Results
//--------------------------------------

#define DECIMAL_PHYSICAL_i32 DORA_PHYSICAL_TYPE_I32
#define DECIMAL_PHYSICAL_i64 DORA_PHYSICAL_TYPE_I64
#define DECIMAL_PHYSICAL_i128 DORA_PHYSICAL_TYPE_I128
#define DECIMAL_WIDENED_ENTRY(OP, LS, LT, OS, OT, WIDEN)                                                               \
    if (operand == DECIMAL_PHYSICAL_##LS && result == DECIMAL_PHYSICAL_##OS) {                                         \
        return policy == DORA_FAILURE_POLICY_ERROR ? arithmetic_##OP##_decimal_##LS##_##OS##_error                     \
                                                   : arithmetic_##OP##_decimal_##LS##_##OS##_null;                     \
    }

#define DECIMAL_SUM_ENTRY(OP, LS, LT, OS, OT, WIDEN)                                                                   \
    if (operand == DECIMAL_PHYSICAL_##LS && result == DECIMAL_PHYSICAL_##OS) {                                         \
        return arithmetic_##OP##_decimal_##LS##_##OS;                                                                  \
    }

static typeof(&arithmetic_multiply_decimal_i64_i64_error) decimal_widened_fill(enum kernel_operation operation,
                                                                               enum dora_physical_type operand,
                                                                               enum dora_physical_type result,
                                                                               enum dora_failure_policy policy) {
    switch (operation) {
    case KERNEL_OPERATION_ADD:
        DECIMAL_WIDENINGS(DECIMAL_SUM_ENTRY, add)
        break;
    case KERNEL_OPERATION_SUBTRACT:
        DECIMAL_WIDENINGS(DECIMAL_SUM_ENTRY, subtract)
        break;
    case KERNEL_OPERATION_MULTIPLY:
        DECIMAL_PRODUCTS(DECIMAL_WIDENED_ENTRY, multiply)
        break;
    default:
        break;
    }
    return nullptr;
}

#undef DECIMAL_SUM_ENTRY
#undef DECIMAL_WIDENED_ENTRY
#undef DECIMAL_PHYSICAL_i128
#undef DECIMAL_PHYSICAL_i64
#undef DECIMAL_PHYSICAL_i32
#undef DECIMAL_PRODUCTS
#undef DECIMAL_WIDENINGS
#undef DECIMAL_SAME

static int arithmetic_result_type(enum kernel_operation operation,
                                  const struct dora_logical_type* operand,
                                  enum dora_physical_type physical,
                                  struct dora_logical_type* out_type) {
    if (operand == nullptr || out_type == nullptr) {
        return EINVAL;
    }
    *out_type = *operand;
    const uint32_t operand_bits = physical_width(physical) * 8U;
    if (operand->id != DORA_TYPE_DECIMAL || operand_bits == 0 || operand_bits > 128) {
        return DORA_OK;
    }
    if (operation == KERNEL_OPERATION_DIVIDE) {
        *out_type = (struct dora_logical_type){.id = DORA_TYPE_FLOAT64};
        return DORA_OK;
    }
    const bool sum = operation == KERNEL_OPERATION_ADD || operation == KERNEL_OPERATION_SUBTRACT;
    if (!sum && operation != KERNEL_OPERATION_MULTIPLY) {
        return DORA_OK;
    }
    const int32_t width = operand->parameters.decimal.precision;
    const int32_t scale = operand->parameters.decimal.scale;
    int32_t result_width = sum ? width + 1 : 2 * width;
    int32_t result_scale = scale;
    if (!sum && (ckd_mul(&result_scale, scale, 2) || result_scale > DECIMAL128_MAX_PRECISION)) {
        return ERANGE;
    }
    if (!sum && result_width > DECIMAL64_MAX_PRECISION && width <= DECIMAL64_MAX_PRECISION &&
        result_scale < DECIMAL64_MAX_PRECISION) {
        result_width = DECIMAL64_MAX_PRECISION;
    }
    if (result_width > DECIMAL128_MAX_PRECISION) {
        result_width = DECIMAL128_MAX_PRECISION;
    }
    const uint32_t minimum_bits = result_width <= DECIMAL32_MAX_PRECISION   ? 32
                                  : result_width <= DECIMAL64_MAX_PRECISION ? 64
                                                                            : 128;
    const uint32_t result_bits = minimum_bits > operand_bits ? minimum_bits : operand_bits;
    out_type->parameters.decimal =
        (struct dora_decimal_type){.precision = result_width, .scale = result_scale, .bit_width = result_bits};
    return DORA_OK;
}

//--------------------------------------
// Bind Functions
//--------------------------------------

int dora_kernel_arithmetic_bind(enum kernel_operation operation,
                                const struct kernel_bind_request* request,
                                struct kernel_binary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || operation >= KERNEL_OPERATION_NEGATE ||
        request->policy > DORA_FAILURE_POLICY_WRAP) {
        return EINVAL;
    }
    struct arithmetic_carrier left = {};
    struct arithmetic_carrier right = {};
    struct arithmetic_carrier result = {};
    int status = arithmetic_carrier(request->input_type, &left);
    if (status == DORA_OK) {
        status = arithmetic_carrier(request->right_type, &right);
    }
    struct dora_logical_type result_type = {};
    if (status == DORA_OK) {
        status = arithmetic_admit(&right, &left);
    }
    if (status == DORA_OK) {
        status = arithmetic_result_type(operation, request->input_type, left.physical, &result_type);
    }
    if (status == DORA_OK) {
        if (left.id != DORA_TYPE_DECIMAL) {
            result = left;
        } else {
            status = arithmetic_carrier(&result_type, &result);
        }
    }
    if (status == DORA_OK) {
        status = arithmetic_policy(&result, operation, request->policy);
    }
    if (status != DORA_OK) {
        return status;
    }
    const bool decimal = left.id == DORA_TYPE_DECIMAL;
    const bool wide = left.physical == DORA_PHYSICAL_TYPE_I256;
    const bool capped = left.decimal.type.precision == result.decimal.type.precision;
    const bool infallible_sum =
        decimal && !capped && (operation == KERNEL_OPERATION_ADD || operation == KERNEL_OPERATION_SUBTRACT);
    const bool widened = decimal && !wide &&
                         (operation == KERNEL_OPERATION_MULTIPLY ||
                          ((operation == KERNEL_OPERATION_ADD || operation == KERNEL_OPERATION_SUBTRACT) && !capped));
    const bool quotient = decimal && operation == KERNEL_OPERATION_DIVIDE && !wide;
    typeof(&arithmetic_add_i64_error) const fill =
        infallible_sum && left.physical == result.physical
            ? arithmetic_binary_leaves[operation][result.physical][DORA_FAILURE_POLICY_WRAP]
        : widened  ? decimal_widened_fill(operation, left.physical, result.physical, request->policy)
        : quotient ? arithmetic_decimal_quotients[left.physical]
        : decimal  ? arithmetic_decimal_leaves[operation][result.physical][request->policy]
                   : arithmetic_binary_leaves[operation][result.physical][request->policy];
    if (fill == nullptr) {
        return ENOTSUP;
    }
    *out_binding = (struct kernel_binary_binding){
        .fill = fill,
        .request = *request,
        .result_type = result_type,
        .input_physical = left.physical,
        .right_physical = right.physical,
        .result_physical = result.physical,
        .left_decimal = left.decimal,
        .right_decimal = right.decimal,
        .result_decimal = result.decimal,
        .left_byte_width = left.width,
        .right_byte_width = right.width,
        .result_byte_width = result.width,
        .is_result_all_valid = false,
        .can_fail = request->policy == DORA_FAILURE_POLICY_ERROR && !infallible_sum &&
                    (result.physical < DORA_PHYSICAL_TYPE_F16 || result.physical > DORA_PHYSICAL_TYPE_F64),
    };
    return DORA_OK;
}

int dora_kernel_unary_arithmetic_bind(enum kernel_operation operation,
                                      const struct kernel_bind_request* request,
                                      struct kernel_unary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || operation < KERNEL_OPERATION_NEGATE ||
        operation >= KERNEL_OPERATION_COUNT || request->policy > DORA_FAILURE_POLICY_WRAP) {
        return EINVAL;
    }
    struct arithmetic_carrier source = {};
    struct arithmetic_carrier result = {};
    int status = arithmetic_carrier(request->input_type, &source);
    result = source;
    if (status == DORA_OK) {
        status = arithmetic_policy(&result, operation, request->policy);
    }
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t index = operation - KERNEL_OPERATION_NEGATE;
    typeof(&arithmetic_negate_i64_error) const fill =
        arithmetic_unary_leaves[index][result.physical]
                               [result.id == DORA_TYPE_DECIMAL ? DORA_FAILURE_POLICY_WRAP : request->policy];
    if (fill == nullptr) {
        return ENOTSUP;
    }
    *out_binding = (struct kernel_unary_binding){
        .fill = fill,
        .request = *request,
        .result_type = *request->input_type,
        .input_physical = source.physical,
        .result_physical = result.physical,
        .input_decimal = source.decimal,
        .result_decimal = result.decimal,
        .input_byte_width = source.width,
        .result_byte_width = result.width,
        .is_result_all_valid = false,
        .can_fail = result.id != DORA_TYPE_DECIMAL && request->policy == DORA_FAILURE_POLICY_ERROR &&
                    (result.physical < DORA_PHYSICAL_TYPE_F16 || result.physical > DORA_PHYSICAL_TYPE_F64),
    };
    return DORA_OK;
}

//--------------------------------------
// Multiply-Add
//--------------------------------------

static const uint64_t arithmetic_null_words[KERNEL_CHUNK_WORD_COUNT] = {};

static int multiply_add_fill(const struct kernel_packet* left,
                             const struct kernel_packet* right,
                             const struct kernel_packet* addend,
                             const struct kernel_ternary_binding* binding,
                             void* product_scratch,
                             struct kernel_result* out_result) {
    if (left->count != right->count || right->count != addend->count) {
        return EINVAL;
    }
    const struct kernel_binary_binding* const product = &binding->product;
    const struct kernel_binary_binding* const sum = &binding->sum;
    const bool constant = left->shape == KERNEL_SHAPE_CONSTANT && right->shape == KERNEL_SHAPE_CONSTANT;
    const bool all_valid =
        left->validity == nullptr && right->validity == nullptr && product->request.policy != DORA_FAILURE_POLICY_NULL;
    uint64_t* const product_words = product_scratch;
    uint8_t* const product_values = (uint8_t*)(product_words + KERNEL_CHUNK_WORD_COUNT);
    struct kernel_result product_result = {
        .values = product_values,
        .validity = all_valid ? nullptr : product_words,
        .row_capacity = constant ? 1 : left->count,
    };
    const int status = product->fill(left, right, product, &product_result);
    if (status != DORA_OK || product_result.failure_bits != 0) {
        out_result->null_count = 0;
        out_result->failure_bits = product_result.failure_bits;
        return status;
    }
    const struct kernel_packet product_packet = {
        .values = product_values,
        .validity = all_valid                        ? nullptr
                    : !constant                      ? product_words
                    : product_result.null_count == 0 ? nullptr
                                                     : arithmetic_null_words,
        .count = addend->count,
        .shape = constant ? KERNEL_SHAPE_CONSTANT : KERNEL_SHAPE_FLAT,
        .placement = left->placement,
    };
    return sum->fill(&product_packet, addend, sum, out_result);
}

int dora_kernel_multiply_add_bind(const struct kernel_bind_request* request,
                                  struct kernel_ternary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    int status = request->input_type == nullptr ? EINVAL : DORA_OK;
    struct kernel_ternary_binding binding = {.fill = multiply_add_fill};
    if (status == DORA_OK) {
        status = dora_kernel_arithmetic_bind(KERNEL_OPERATION_MULTIPLY, request, &binding.product);
    }
    if (status == DORA_OK) {
        status = dora_kernel_arithmetic_bind(KERNEL_OPERATION_ADD, request, &binding.sum);
    }
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_logical_type* const input = request->input_type;
    const struct kernel_binary_binding* const results[] = {&binding.product, &binding.sum};
    for (size_t index = 0; index < ARRAY_COUNT(results); ++index) {
        const struct dora_logical_type* const result = &results[index]->result_type;
        if (results[index]->result_physical != results[index]->input_physical ||
            (result->id == DORA_TYPE_DECIMAL &&
             (result->parameters.decimal.precision != input->parameters.decimal.precision ||
              result->parameters.decimal.scale != input->parameters.decimal.scale))) {
            return ENOTSUP;
        }
    }
    binding.scratch_bytes = KERNEL_CHUNK_WORD_COUNT * (uint32_t)sizeof(uint64_t) +
                            KERNEL_CHUNK_ROW_COUNT * binding.product.result_byte_width;
    *out_binding = binding;
    return DORA_OK;
}

//--------------------------------------
// Temporal Binding
//--------------------------------------

struct temporal_leaf {
    typeof(&temporal_add_ticks_i64_error) error;
    typeof(&temporal_add_ticks_i64_error) null;
};

#define TEMPORAL_LEAF(NAME)                                                                                            \
    (struct temporal_leaf) {                                                                                           \
        .error = temporal_##NAME##_error, .null = temporal_##NAME##_null                                               \
    }

static int64_t temporal_day_limit(enum dora_time_unit unit) {
    switch (unit) {
    case DORA_TIME_UNIT_SECOND:
        return 86'400;
    case DORA_TIME_UNIT_MILLISECOND:
        return 86'400'000;
    case DORA_TIME_UNIT_MICROSECOND:
        return 86'400'000'000;
    case DORA_TIME_UNIT_NANOSECOND:
        return 86'400'000'000'000;
    }
    return 0;
}

static bool temporal_has_extension(const struct dora_logical_type* type) {
    return type->extension_name.data != nullptr || type->extension_name.byte_count != 0 ||
           type->extension_metadata.data != nullptr || type->extension_metadata.byte_count != 0;
}

static int temporal_date(enum dora_type delta, bool reversed, bool subtract, struct temporal_leaf* out_leaf) {
    switch (delta) {
#define TEMPORAL_DATE_CASE(S, TYPE)                                                                                    \
    case DORA_TYPE_##TYPE:                                                                                             \
        *out_leaf = reversed   ? TEMPORAL_LEAF(add_##S##_date32)                                                       \
                    : subtract ? TEMPORAL_LEAF(subtract_date32_##S)                                                    \
                               : TEMPORAL_LEAF(add_date32_##S);                                                        \
        return DORA_OK;
        TEMPORAL_DATE_CASE(i8, INT8)
        TEMPORAL_DATE_CASE(i16, INT16)
        TEMPORAL_DATE_CASE(i32, INT32)
        TEMPORAL_DATE_CASE(i64, INT64)
#undef TEMPORAL_DATE_CASE
    default:
        return ENOTSUP;
    }
}

static int temporal_plan(const struct dora_logical_type* left_type,
                         const struct dora_logical_type* right_type,
                         bool subtract,
                         struct kernel_binary_binding* in_out_binding,
                         struct temporal_leaf* out_leaf) {
    const bool reversed = !subtract && (right_type->id == DORA_TYPE_DATE32 || right_type->id == DORA_TYPE_TIME ||
                                        right_type->id == DORA_TYPE_TIMESTAMP);
    const struct dora_logical_type* const temporal_type = reversed ? right_type : left_type;
    const struct dora_logical_type* const delta_type = reversed ? left_type : right_type;
    if (temporal_type->id == DORA_TYPE_DATE32) {
        in_out_binding->result_type = (struct dora_logical_type){.id = DORA_TYPE_DATE32};

        return temporal_date(delta_type->id, reversed, subtract, out_leaf);
    }
    if (temporal_type->id == DORA_TYPE_TIME && delta_type->id == DORA_TYPE_DURATION) {
        const enum dora_time_unit unit = temporal_type->parameters.unit;
        in_out_binding->ticks_per_day = temporal_day_limit(unit);
        if (in_out_binding->ticks_per_day == 0 || unit != delta_type->parameters.unit) {
            return EINVAL;
        }
        in_out_binding->result_type = (struct dora_logical_type){.id = DORA_TYPE_TIME, .parameters.unit = unit};
        if (unit <= DORA_TIME_UNIT_MILLISECOND) {
            *out_leaf = reversed   ? TEMPORAL_LEAF(add_duration_time_i32)
                        : subtract ? TEMPORAL_LEAF(subtract_time_i32_duration)
                                   : TEMPORAL_LEAF(add_time_i32_duration);
        } else {
            *out_leaf = reversed   ? TEMPORAL_LEAF(add_duration_time_i64)
                        : subtract ? TEMPORAL_LEAF(subtract_time_i64_duration)
                                   : TEMPORAL_LEAF(add_time_i64_duration);
        }
        return DORA_OK;
    }
    *out_leaf = subtract ? TEMPORAL_LEAF(subtract_ticks_i64) : TEMPORAL_LEAF(add_ticks_i64);

    if (temporal_type->id == DORA_TYPE_TIMESTAMP && delta_type->id == DORA_TYPE_DURATION) {
        if (temporal_type->parameters.timestamp.unit > DORA_TIME_UNIT_NANOSECOND ||
            temporal_type->parameters.timestamp.unit != delta_type->parameters.unit) {
            return EINVAL;
        }
        in_out_binding->result_type = (struct dora_logical_type){
            .id = DORA_TYPE_TIMESTAMP,
            .parameters.timestamp = temporal_type->parameters.timestamp,
        };
        return DORA_OK;
    }
    if (subtract && left_type->id == DORA_TYPE_TIMESTAMP && right_type->id == DORA_TYPE_TIMESTAMP) {
        const struct dora_timestamp_type* const left_timestamp = &left_type->parameters.timestamp;
        const struct dora_timestamp_type* const right_timestamp = &right_type->parameters.timestamp;
        if (left_timestamp->unit > DORA_TIME_UNIT_NANOSECOND || left_timestamp->unit != right_timestamp->unit ||
            strcmp(left_timestamp->timezone == nullptr ? "" : left_timestamp->timezone,
                   right_timestamp->timezone == nullptr ? "" : right_timestamp->timezone) != 0) {
            return EINVAL;
        }
        in_out_binding->result_type =
            (struct dora_logical_type){.id = DORA_TYPE_DURATION, .parameters.unit = left_timestamp->unit};
        return DORA_OK;
    }
    if (left_type->id == DORA_TYPE_DURATION && right_type->id == DORA_TYPE_DURATION) {
        if (left_type->parameters.unit > DORA_TIME_UNIT_NANOSECOND ||
            left_type->parameters.unit != right_type->parameters.unit) {
            return EINVAL;
        }
        in_out_binding->result_type =
            (struct dora_logical_type){.id = DORA_TYPE_DURATION, .parameters.unit = left_type->parameters.unit};
        return DORA_OK;
    }
    return ENOTSUP;
}

#undef TEMPORAL_LEAF

int dora_kernel_temporal_bind(enum kernel_operation operation,
                              const struct dora_logical_type* left_type,
                              const struct dora_logical_type* right_type,
                              enum dora_failure_policy policy,
                              struct kernel_binary_binding* out_binding) {
    if (left_type == nullptr || right_type == nullptr || out_binding == nullptr ||
        operation >= KERNEL_OPERATION_COUNT || policy > DORA_FAILURE_POLICY_NULL) {
        return EINVAL;
    }
    if (operation != KERNEL_OPERATION_ADD && operation != KERNEL_OPERATION_SUBTRACT) {
        return ENOTSUP;
    }
    if (temporal_has_extension(left_type) || temporal_has_extension(right_type)) {
        return ENOTSUP;
    }
    struct kernel_binary_binding binding = {};
    struct temporal_leaf leaf = {};
    int status = temporal_plan(left_type, right_type, operation == KERNEL_OPERATION_SUBTRACT, &binding, &leaf);
    if (status != DORA_OK) {
        return status;
    }
    status = type_scalar_physical(left_type, &binding.input_physical);
    if (status == DORA_OK) {
        status = type_scalar_physical(right_type, &binding.right_physical);
    }
    if (status == DORA_OK) {
        status = type_scalar_physical(&binding.result_type, &binding.result_physical);
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_binding = binding;
    out_binding->fill = policy == DORA_FAILURE_POLICY_ERROR ? leaf.error : leaf.null;
    out_binding->request = (struct kernel_bind_request){
        .input_type = left_type,
        .right_type = right_type,
        .policy = policy,
    };
    out_binding->left_byte_width = physical_width(binding.input_physical);
    out_binding->right_byte_width = physical_width(binding.right_physical);
    out_binding->result_byte_width = physical_width(binding.result_physical);
    out_binding->is_result_all_valid = false;
    out_binding->can_fail = policy == DORA_FAILURE_POLICY_ERROR;
    return DORA_OK;
}
