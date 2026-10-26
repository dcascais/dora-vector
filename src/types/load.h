// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Scalar carrier loads and stores.

#ifndef DORA_TYPES_LOAD_H
#define DORA_TYPES_LOAD_H

#include "dora/vector/types.h"
#include "types/types_internal.h"

#include <stdint.h>
#include <string.h>

static inline _BitInt(128) type_i128_load(struct dora_i128 value) {
    _BitInt(128) result = 0;
    (memcpy)(&result, value.limbs, sizeof(result));
    return result;
}

static inline struct dora_i128 type_i128_store(unsigned _BitInt(128) value) {
    struct dora_i128 result = {};
    (memcpy)(result.limbs, &value, sizeof(result.limbs));
    return result;
}

static inline unsigned _BitInt(128) type_u128_load(struct dora_u128 value) {
    unsigned _BitInt(128) result = 0;
    (memcpy)(&result, value.limbs, sizeof(result));
    return result;
}

static inline struct dora_u128 type_u128_store(unsigned _BitInt(128) value) {
    struct dora_u128 result = {};
    (memcpy)(result.limbs, &value, sizeof(result.limbs));
    return result;
}

// Integers, floating-point types, Decimal, temporal types, intervals, Utf8, Binary, FixedBinary
#define DEFINE_LOAD(S, T, P)                                                                                           \
    static inline T type_load_##S(const uint8_t* values, uint64_t element) {                                           \
        T value;                                                                                                       \
        (memcpy)(&value, values + element * sizeof(T), sizeof(T));                                                     \
        return value;                                                                                                  \
    }

FIXED_CARRIERS(DEFINE_LOAD)
DEFINE_LOAD(bytes, struct dora_bytes, DORA_PHYSICAL_TYPE_NONE)

#undef DEFINE_LOAD

#define TYPE_LOADER(T)                                                                                                 \
    _Generic((T){},                                                                                                    \
        int8_t: type_load_i8,                                                                                          \
        int16_t: type_load_i16,                                                                                        \
        int32_t: type_load_i32,                                                                                        \
        int64_t: type_load_i64,                                                                                        \
        struct dora_i128: type_load_i128,                                                                              \
        struct dora_i256: type_load_i256,                                                                              \
        uint8_t: type_load_u8,                                                                                         \
        uint16_t: type_load_u16,                                                                                       \
        uint32_t: type_load_u32,                                                                                       \
        uint64_t: type_load_u64,                                                                                       \
        struct dora_u128: type_load_u128,                                                                              \
        struct dora_f16_bits: type_load_f16,                                                                           \
        float: type_load_f32,                                                                                          \
        double: type_load_f64,                                                                                         \
        struct dora_interval_day_time: type_load_interval_day_time,                                                    \
        struct dora_interval_month_day_nano: type_load_interval_month_day_nano,                                        \
        struct dora_bytes: type_load_bytes)

#endif
