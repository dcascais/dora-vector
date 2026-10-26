// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scalar hashing.

#ifndef DORA_HASH_HASH_H
#define DORA_HASH_HASH_H

#include "dora/vector/types.h"
#include "numeric/float.h"
#include "temporal/interval.h"
#include "types/types_internal.h"

#include <stddef.h>
#include <stdint.h>

static constexpr uint64_t HASH_NULL_VALUE = UINT64_C(0xbf58476d1ce4e5b9);

//------------------------------------------------------------------------------
//
// Mixing
//
//------------------------------------------------------------------------------

static inline uint64_t hash_value_word(uint64_t value, uint64_t seed) {
    uint64_t mixed = value ^ seed;
    mixed ^= mixed >> 30U;
    mixed *= UINT64_C(0xbf'58'47'6d'1c'e4'e5'b9);
    mixed ^= mixed >> 27U;
    mixed *= UINT64_C(0x94'd0'49'bb'13'31'11'eb);
    return mixed ^ (mixed >> 31U);
}

static inline uint64_t hash_value_combine(uint64_t left, uint64_t right) {
    left ^= left >> 32U;
    left *= UINT64_C(0xd6e8feb86659fd93);
    return left ^ right;
}

static inline uint64_t hash_value_interval(struct temporal_interval_parts value, uint64_t seed) {
    return hash_value_word((uint64_t)value.nanoseconds,
                           hash_value_word((uint64_t)value.days, hash_value_word((uint64_t)value.months, seed)));
}

//------------------------------------------------------------------------------
//
// Values
//
//------------------------------------------------------------------------------

//--------------------------------------
// Integers
//--------------------------------------

#define HASH_VALUE_INTEGER(S, T, P)                                                                                    \
    static inline uint64_t hash_value_##S(T value, uint64_t seed) {                                                    \
        return hash_value_word((uint64_t)value, seed);                                                                 \
    }
SIGNED_INDEX_CARRIERS(HASH_VALUE_INTEGER)
UNSIGNED_INDEX_CARRIERS(HASH_VALUE_INTEGER)
HASH_VALUE_INTEGER(bool, bool, DORA_PHYSICAL_TYPE_NONE)
#undef HASH_VALUE_INTEGER

#define HASH_VALUE_WIDE(S, T, N)                                                                                       \
    static inline uint64_t hash_value_##S(T value, uint64_t seed) {                                                    \
        for (size_t i = 0; i < (N); ++i) {                                                                             \
            seed = hash_value_word(value.limbs[i], seed);                                                              \
        }                                                                                                              \
        return seed;                                                                                                   \
    }
HASH_VALUE_WIDE(i128, struct dora_i128, 2)
HASH_VALUE_WIDE(i256, struct dora_i256, 4)
HASH_VALUE_WIDE(u128, struct dora_u128, 2)
#undef HASH_VALUE_WIDE

//--------------------------------------
// Floating Point
//--------------------------------------

static inline uint64_t hash_value_f32(float value, uint64_t seed) {
    return hash_value_word(numeric_f32_key(value), seed);
}

static inline uint64_t hash_value_f64(double value, uint64_t seed) {
    return hash_value_word(numeric_f64_key(value), seed);
}

static inline uint64_t hash_value_f16(struct dora_f16_bits value, uint64_t seed) {
    return hash_value_f32(numeric_half_widen(value), seed);
}

//--------------------------------------
// Temporal
//--------------------------------------

static inline uint64_t hash_value_interval_year_month(int32_t value, uint64_t seed) {
    return hash_value_interval((struct temporal_interval_parts){.months = value}, seed);
}

static inline uint64_t hash_value_interval_day_time(struct dora_interval_day_time value, uint64_t seed) {
    return hash_value_interval(temporal_interval_normalize((struct temporal_interval_parts){
                                   .days = value.days,
                                   .nanoseconds = (int64_t)value.milliseconds * 1'000'000
                               }),
                               seed);
}

static inline uint64_t hash_value_interval_month_day_nano(struct dora_interval_month_day_nano value, uint64_t seed) {
    return hash_value_interval(temporal_interval_normalize((struct temporal_interval_parts){
                                   .months = value.months,
                                   .days = value.days,
                                   .nanoseconds = value.nanoseconds
                               }),
                               seed);
}

//--------------------------------------
// Bytes
//--------------------------------------

static inline uint64_t hash_value_bytes(struct dora_bytes value, uint64_t seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < value.byte_count; ++i) {
        hash = hash_value_word((uint64_t)value.data[i] + 1, hash);
    }
    return hash_value_word(value.byte_count, hash);
}

#endif
