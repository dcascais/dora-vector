// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "numeric/arithmetic.h"
#include "numeric/decimal.h"
#include "numeric/decimal_float.h"
#include "numeric/float.h"
#include "numeric/limbs.h"
#include "temporal/temporal.h"
#include "types/load.h"
#include "types/text.h"
#include "types/types_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Values
//
//------------------------------------------------------------------------------

union cast_value {
    bool boolean;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    struct dora_i128 i128;
    struct dora_i256 i256;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    struct dora_u128 u128;
    struct dora_f16_bits f16;
    float f32;
    double f64;
    struct dora_bytes bytes;
};

static inline bool cast_should_null_failure(const struct kernel_unary_binding* binding, int status) {
    return binding->request.policy == DORA_FAILURE_POLICY_NULL &&
           (status == EINVAL || status == ERANGE || status == EOVERFLOW || status == EILSEQ);
}

//------------------------------------------------------------------------------
//
// Scalar Core
//
//------------------------------------------------------------------------------

static bool cast_is_float(enum dora_physical_type physical) {
    return physical >= DORA_PHYSICAL_TYPE_F16 && physical <= DORA_PHYSICAL_TYPE_F64;
}

static double cast_double(enum dora_physical_type physical, const union cast_value* value) {
    return physical == DORA_PHYSICAL_TYPE_F16   ? (double)numeric_half_widen(value->f16)
           : physical == DORA_PHYSICAL_TYPE_F32 ? value->f32
                                                : value->f64;
}

static bool cast_round(double value, double* out_integral, enum dora_rounding_mode mode) {
    if (!isfinite(value)) {
        return false;
    }
    const bool negative = value < 0;
    const double magnitude = fabs(value);
    double integral = floor(magnitude);
    const double fraction = magnitude - integral;
    if (mode == DORA_ROUNDING_MODE_EXACT && fraction != 0) {
        return false;
    }
    if ((mode == DORA_ROUNDING_MODE_HALF_AWAY || mode == DORA_ROUNDING_MODE_HALF_EVEN) &&
        (fraction > 0.5 || (fraction == 0.5 && (mode == DORA_ROUNDING_MODE_HALF_AWAY || fmod(integral, 2.0) != 0)))) {
        integral += 1;
    }
    *out_integral = negative ? -integral : integral;
    return true;
}

static int cast_double_integer(double value, struct dora_i256* out_value, enum dora_rounding_mode mode) {
    double integral = 0;
    if (!cast_round(value, &integral, mode)) {
        return ERANGE;
    }
    const bool negative = integral < 0;
    double magnitude = fabs(integral);
    if (magnitude > 0x1p255 || (magnitude == 0x1p255 && !negative)) {
        return ERANGE;
    }
    uint64_t limbs[4] = {};
    for (uint32_t index = 0; index < 4; ++index) {
        limbs[index] = (uint64_t)fmod(magnitude, 0x1p64);
        magnitude = floor(magnitude * 0x1p-64);
    }
    return dora_i256_from_mag(limbs, negative, out_value);
}

static int cast_decimal_check(struct dora_i256 value, const struct decimal_prepared* decimal) {
    uint64_t magnitude[4] = {};
    dora_limbs_magnitude(value.limbs, 4, magnitude);
    return dora_limbs_compare(magnitude, decimal->precision_limit.limbs, 4) < 0 ? DORA_OK : EOVERFLOW;
}

static int cast_tick_scale(const struct dora_logical_type* type) {
    if (type->id == DORA_TYPE_DATE32) {
        return 0;
    }
    if (type->id == DORA_TYPE_DATE64) {
        return 3;
    }
    return 3 * (int)(type->id == DORA_TYPE_TIMESTAMP ? type->parameters.timestamp.unit : type->parameters.unit);
}

static int64_t cast_day_ticks(const struct dora_logical_type* type) {
    static constexpr int64_t day_ticks[4] = {86'400, 86'400'000, 86'400'000'000, 86'400'000'000'000};
    return day_ticks[cast_tick_scale(type) / 3];
}

static int cast_temporal_admit(const struct dora_logical_type* source, const struct dora_logical_type* target) {
    const bool source_date = source->id == DORA_TYPE_DATE32 || source->id == DORA_TYPE_DATE64;
    const bool target_date = target->id == DORA_TYPE_DATE32 || target->id == DORA_TYPE_DATE64;
    if (source->id == DORA_TYPE_TIMESTAMP && target->id == DORA_TYPE_TIMESTAMP) {
        const char* first = source->parameters.timestamp.timezone;
        const char* second = target->parameters.timestamp.timezone;
        if (strcmp(first != nullptr ? first : "", second != nullptr ? second : "") != 0) {
            return EINVAL;
        }
    }
    if ((source_date && target->id == DORA_TYPE_TIMESTAMP) || (target_date && source->id == DORA_TYPE_TIMESTAMP)) {
        return type_has_timezone(source) || type_has_timezone(target) ? ENOTSUP : DORA_OK;
    }
    return source->id != target->id && !(source_date && target_date) ? EINVAL : DORA_OK;
}

static int
cast_temporal_value(const struct kernel_unary_binding* binding, struct dora_i256 value, struct dora_i256* out_value) {
    const struct dora_logical_type* const source = binding->request.input_type;
    const struct dora_logical_type* const target = binding->request.result_type;
    if (source->id == DORA_TYPE_DATE64 && (int64_t)value.limbs[0] % 86400000 != 0) {
        return ERANGE;
    }
    if (source->id == DORA_TYPE_TIME) {
        const int64_t ticks = (int64_t)value.limbs[0];
        if (ticks < 0 || ticks >= cast_day_ticks(source)) {
            return ERANGE;
        }
    }
    const bool source_date = source->id == DORA_TYPE_DATE32 || source->id == DORA_TYPE_DATE64;
    const bool target_date = target->id == DORA_TYPE_DATE32 || target->id == DORA_TYPE_DATE64;
    if (source_date) {
        int64_t days = (int64_t)value.limbs[0];
        if (source->id == DORA_TYPE_DATE64) {
            days /= 86400000;
        }
        if (target->id == DORA_TYPE_DATE32) {
            *out_value = numeric_i256_from_i64(days);
            return DORA_OK;
        }
        value = numeric_i256_from_i128(type_i128_store((_BitInt(128))days * (target_date ? 86400000 : 86400)));
        const int status =
            target_date ? DORA_OK
                        : dora_decimal_rescale_i256(value, binding->cast.scale, binding->request.rounding, &value);
        if (status == DORA_OK) {
            *out_value = value;
        }
        return status;
    }
    if (target_date) {
        const int64_t divisor = binding->cast.ticks_per_day;
        const int64_t ticks = (int64_t)value.limbs[0];
        int64_t days = ticks / divisor;
        if (ticks % divisor < 0) {
            --days;
        }
        *out_value = target->id == DORA_TYPE_DATE64
                         ? numeric_i256_from_i128(type_i128_store((_BitInt(128))days * 86400000))
                         : numeric_i256_from_i64(days);
        return DORA_OK;
    }
    return dora_decimal_rescale_i256(value, binding->cast.scale, binding->request.rounding, out_value);
}

static int cast_to_float(const struct kernel_unary_binding* binding,
                         const union cast_value* value,
                         struct dora_i256 integer,
                         union cast_value* out_value) {
    const enum dora_physical_type physical = binding->result_physical;
    const enum dora_rounding_mode mode = binding->request.rounding;
    const bool floating = cast_is_float(binding->input_physical);
    const int32_t scale = binding->input_decimal.type.scale;
    const enum dora_physical_type input = binding->input_physical;
    const bool decimal = binding->input_decimal.type.precision != 0;
    if (physical == DORA_PHYSICAL_TYPE_F32 && decimal && input != DORA_PHYSICAL_TYPE_I256) {
        out_value->f32 =
            input == DORA_PHYSICAL_TYPE_I128
                ? dora_decimal_to_float_i128(value->i128, scale)
                : dora_decimal_to_float_i64(input == DORA_PHYSICAL_TYPE_I32 ? value->i32 : value->i64, scale);
        if (mode != DORA_ROUNDING_MODE_EXACT && isfinite(out_value->f32)) {
            return DORA_OK;
        }
    }
    double number = 0;
    if (floating) {
        number = cast_double(input, value);
    } else if (decimal && input == DORA_PHYSICAL_TYPE_I32) {
        number = decimal_double_i32(scale, value->i32);
    } else if (decimal && input == DORA_PHYSICAL_TYPE_I64) {
        number = decimal_double_i64(scale, value->i64);
    } else if (decimal && input == DORA_PHYSICAL_TYPE_I128) {
        number = decimal_double_i128(scale, value->i128);
    } else {
        number = numeric_i256_to_double(integer);
        if (number != 0 && decimal) {
            number *= pow(10.0, -(double)scale);
        }
    }
    double restored = number;
    if (physical == DORA_PHYSICAL_TYPE_F16) {
        out_value->f16 = numeric_half_narrow((float)number);
        restored = numeric_half_widen(out_value->f16);
    } else if (physical == DORA_PHYSICAL_TYPE_F32) {
        if (!decimal || input == DORA_PHYSICAL_TYPE_I256) {
            out_value->f32 = (float)number;
        }
        restored = out_value->f32;
    } else {
        out_value->f64 = number;
    }
    if (!floating && scale == 0 && mode == DORA_ROUNDING_MODE_EXACT) {
        struct dora_i256 check = {};
        return cast_double_integer(restored, &check, mode) != DORA_OK || memcmp(&integer, &check, sizeof(integer)) != 0
                   ? ERANGE
                   : DORA_OK;
    }
    if ((isfinite(number) && !isfinite(restored)) ||
        (mode == DORA_ROUNDING_MODE_EXACT && !isnan(number) && restored != number)) {
        return ERANGE;
    }
    return DORA_OK;
}

static int cast_to_integer(const struct kernel_unary_binding* binding,
                           const union cast_value* value,
                           struct dora_i256 integer,
                           union cast_value* out_value) {
    const struct dora_logical_type* const source = binding->request.input_type;
    const struct dora_logical_type* const target = binding->request.result_type;
    const enum dora_physical_type physical = binding->result_physical;
    const bool floating = cast_is_float(binding->input_physical);
    int status = DORA_OK;
    if (type_is_temporal(target->id)) {
        if (type_is_temporal(source->id)) {
            status = cast_temporal_value(binding, integer, &integer);
        }
    } else if (!type_is_temporal(source->id) && floating) {
        double number = cast_double(binding->input_physical, value);
        if (number != 0) {
            const int32_t target_scale = binding->cast.scale.target;
            number *= target_scale >= 0 && target_scale <= DECIMAL256_MAX_PRECISION
                          ? dora_decimal_double_powers[target_scale]
                          : pow(10.0, target_scale);
        }
        status = cast_double_integer(number, &integer, binding->request.rounding);
    } else if (!type_is_temporal(source->id)) {
        status = dora_decimal_rescale_i256(integer, binding->cast.scale, binding->request.rounding, &integer);
    }
    if (status == DORA_OK && binding->result_decimal.type.precision != 0) {
        status = cast_decimal_check(integer, &binding->result_decimal);
    }
    if (status == DORA_OK) {
        status = dora_numeric_store_integer(integer, physical, out_value);
    }
    if (status == DORA_OK && target->id == DORA_TYPE_DATE64 && out_value->i64 % 86400000 != 0) {
        status = ERANGE;
    }
    if (status == DORA_OK && target->id == DORA_TYPE_TIME) {
        const int64_t ticks = physical == DORA_PHYSICAL_TYPE_I32 ? out_value->i32 : out_value->i64;
        if (ticks < 0 || ticks >= binding->cast.ticks_per_day) {
            status = ERANGE;
        }
    }
    return status;
}

static int
cast_scalar(const struct kernel_unary_binding* binding, const union cast_value* value, union cast_value* out_value) {
    const struct dora_logical_type* const source = binding->request.input_type;
    const struct dora_logical_type* const target = binding->request.result_type;
    if (source->id >= DORA_TYPE_INTERVAL_YEAR_MONTH) {
        *out_value = *value;
        return DORA_OK;
    }
    const bool floating = cast_is_float(binding->input_physical);
    struct dora_i256 integer = {};
    if (source->id == DORA_TYPE_BOOLEAN) {
        integer = numeric_i256_from_u64(value->boolean ? 1 : 0);
    } else if (!floating) {
        const int status = numeric_load_integer(binding->input_physical, value, &integer);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (target->id == DORA_TYPE_BOOLEAN) {
        if (floating) {
            const double number = cast_double(binding->input_physical, value);
            if (!isfinite(number)) {
                return ERANGE;
            }
            out_value->boolean = number != 0;
        } else {
            out_value->boolean = !dora_limbs_is_zero(integer.limbs, 4);
        }
        return DORA_OK;
    }
    return cast_is_float(binding->result_physical) ? cast_to_float(binding, value, integer, out_value)
                                                   : cast_to_integer(binding, value, integer, out_value);
}

//------------------------------------------------------------------------------
//
// Leaves
//
//------------------------------------------------------------------------------

#define LEAF_LOCALS                                                                                                    \
    const uint32_t count = source->count;                                                                              \
    const uint8_t* const values = source->values;                                                                      \
    const uint64_t offset = source->value_offset;                                                                      \
    const uint32_t* const rows = source->rows;                                                                         \
    uint8_t* const results = out_result->values;                                                                       \
    uint32_t null_count = 0

#define FLAT_ELEMENT(ROW) (offset + (ROW))
#define CONSTANT_ELEMENT(ROW) (offset)
#define MAP_ELEMENT(ROW) (offset + rows[ROW])

#define CAST_LANES(ELEMENT, LANE)                                                                                      \
    for (uint32_t word = 0; word * 64 < count; ++word) {                                                               \
        const uint64_t tail = bitmap_word_tail_mask(count, word);                                                      \
        uint64_t active = source->validity == nullptr ? tail : source->validity[word] & tail;                          \
        uint64_t valid = active;                                                                                       \
        if (active == tail) {                                                                                          \
            const uint32_t end = count - word * 64 < 64 ? count : word * 64 + 64;                                      \
            for (uint32_t row = word * 64; row < end; ++row) {                                                         \
                LANE(row, ELEMENT(row));                                                                               \
            }                                                                                                          \
        } else {                                                                                                       \
            while (active != 0) {                                                                                      \
                const uint32_t row = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                             \
                LANE(row, ELEMENT(row));                                                                               \
            }                                                                                                          \
        }                                                                                                              \
        if (out_result->validity != nullptr) {                                                                         \
            out_result->validity[word] = valid;                                                                        \
        }                                                                                                              \
        null_count += dora_bitmap_word_count_ones(tail & ~valid);                                                      \
    }

#define CAST_SHAPES(LANE)                                                                                              \
    switch (source->shape) {                                                                                           \
    case KERNEL_SHAPE_FLAT:                                                                                            \
        CAST_LANES(FLAT_ELEMENT, LANE)                                                                                 \
        break;                                                                                                         \
    case KERNEL_SHAPE_CONSTANT:                                                                                        \
        CAST_LANES(CONSTANT_ELEMENT, LANE)                                                                             \
        break;                                                                                                         \
    case KERNEL_SHAPE_GATHERED:                                                                                        \
        CAST_LANES(MAP_ELEMENT, LANE)                                                                                  \
        break;                                                                                                         \
    }

// Returns from the enclosing function unless policy nulls the failure; evaluates STATUS twice on return.
#define CAST_FAILED(ROW, STATUS)                                                                                       \
    do {                                                                                                               \
        if (!cast_should_null_failure(binding, STATUS)) {                                                              \
            return STATUS;                                                                                             \
        }                                                                                                              \
        bitmap_clear(&valid, (ROW) - word * 64);                                                                       \
    } while (0)

// Returns DORA_OK from the enclosing function.
#define LEAF_FINISH                                                                                                    \
    out_result->null_count = null_count;                                                                               \
    out_result->failure_bits = 0;                                                                                      \
    return DORA_OK

//--------------------------------------
// Numeric Carriers
//--------------------------------------

#define PHYSICAL_i8 DORA_PHYSICAL_TYPE_I8
#define PHYSICAL_i16 DORA_PHYSICAL_TYPE_I16
#define PHYSICAL_i32 DORA_PHYSICAL_TYPE_I32
#define PHYSICAL_i64 DORA_PHYSICAL_TYPE_I64
#define PHYSICAL_i128 DORA_PHYSICAL_TYPE_I128
#define PHYSICAL_i256 DORA_PHYSICAL_TYPE_I256
#define PHYSICAL_u8 DORA_PHYSICAL_TYPE_U8
#define PHYSICAL_u16 DORA_PHYSICAL_TYPE_U16
#define PHYSICAL_u32 DORA_PHYSICAL_TYPE_U32
#define PHYSICAL_u64 DORA_PHYSICAL_TYPE_U64
#define PHYSICAL_u128 DORA_PHYSICAL_TYPE_U128
#define PHYSICAL_f16 DORA_PHYSICAL_TYPE_F16
#define PHYSICAL_f32 DORA_PHYSICAL_TYPE_F32
#define PHYSICAL_f64 DORA_PHYSICAL_TYPE_F64

#define SOURCE_SIGNED(VALUE) ((int64_t)(VALUE))
#define SOURCE_UNSIGNED(VALUE) ((uint64_t)(VALUE))
#define SOURCE_WIDE_i256(VALUE) (VALUE)
#define SOURCE_WIDE_u128(VALUE) numeric_i256_from_u128(VALUE)
#define SOURCE_FLOAT_f16(VALUE) ((double)numeric_half_widen(VALUE))
#define SOURCE_FLOAT_f32(VALUE) ((double)(VALUE))
#define SOURCE_FLOAT_f64(VALUE) (VALUE)

#define SOURCE_INTERMEDIATE_SIGNED(S, VALUE) SOURCE_SIGNED(VALUE)
#define SOURCE_INTERMEDIATE_UNSIGNED(S, VALUE) SOURCE_UNSIGNED(VALUE)
#define SOURCE_INTERMEDIATE_WIDE(S, VALUE) SOURCE_WIDE_##S(VALUE)
#define SOURCE_INTERMEDIATE_I128(S, VALUE) type_i128_load(VALUE)
#define SOURCE_INTERMEDIATE_FLOAT(S, VALUE) SOURCE_FLOAT_##S(VALUE)

struct float_bounds {
    double lower;
    double upper;
};

static inline bool cast_float_conversion_has_failed(double number, double restored, bool exact) {
    return (isfinite(number) && !isfinite(restored)) || (exact && !isnan(number) && restored != number);
}

static inline struct dora_f16_bits narrow_f16(double number, double* out_restored) {
    const struct dora_f16_bits result = numeric_half_narrow((float)number);
    *out_restored = numeric_half_widen(result);
    return result;
}

static inline float narrow_f32(double number, double* out_restored) {
    const float result = (float)number;
    *out_restored = result;
    return result;
}

static inline double narrow_f64(double number, double* out_restored) {
    *out_restored = number;
    return number;
}

static inline bool roundtrip_signed_matches(double restored, int64_t value) {
    return restored >= -0x1p63 && restored < 0x1p63 && (int64_t)restored == value;
}

static inline bool roundtrip_unsigned_matches(double restored, uint64_t value) {
    return restored >= 0 && restored < 0x1p64 && (uint64_t)restored == value;
}

static inline bool roundtrip_wide_matches(double restored, struct dora_i256 value) {
    struct dora_i256 check = {};
    return cast_double_integer(restored, &check, DORA_ROUNDING_MODE_EXACT) == DORA_OK &&
           memcmp(&check, &value, sizeof(check)) == 0;
}

static inline bool cast_value_is_within_bounds(double integral, struct float_bounds bounds) {
    return integral >= bounds.lower && integral < bounds.upper;
}

static constexpr struct float_bounds BOUNDS_i8 = {-0x1p7, 0x1p7};
static constexpr struct float_bounds BOUNDS_i16 = {-0x1p15, 0x1p15};
static constexpr struct float_bounds BOUNDS_i32 = {-0x1p31, 0x1p31};
static constexpr struct float_bounds BOUNDS_i64 = {-0x1p63, 0x1p63};
static constexpr struct float_bounds BOUNDS_u8 = {0, 0x1p8};
static constexpr struct float_bounds BOUNDS_u16 = {0, 0x1p16};
static constexpr struct float_bounds BOUNDS_u32 = {0, 0x1p32};
static constexpr struct float_bounds BOUNDS_u64 = {0, 0x1p64};

#define DEFINE_CONVERT_NATIVE(D)                                                                                       \
    static inline bool convert_float_##D(double value,                                                                 \
                                         typeof(type_load_##D(nullptr, 0))* out_value,                                 \
                                         enum dora_rounding_mode mode) {                                               \
        const double integral = mode == DORA_ROUNDING_MODE_HALF_EVEN   ? nearbyint(value)                              \
                                : mode == DORA_ROUNDING_MODE_HALF_AWAY ? round(value)                                  \
                                                                       : trunc(value);                                 \
        const bool admitted = cast_value_is_within_bounds(integral, BOUNDS_##D) &&                                     \
                              (mode != DORA_ROUNDING_MODE_EXACT || integral == value);                                 \
        *out_value = (typeof(type_load_##D(nullptr, 0)))(admitted ? integral : 0.0);                                   \
        return !admitted;                                                                                              \
    }
DEFINE_CONVERT_NATIVE(i8)
DEFINE_CONVERT_NATIVE(i16)
DEFINE_CONVERT_NATIVE(i32)
DEFINE_CONVERT_NATIVE(i64)
DEFINE_CONVERT_NATIVE(u8)
DEFINE_CONVERT_NATIVE(u16)
DEFINE_CONVERT_NATIVE(u32)
DEFINE_CONVERT_NATIVE(u64)
#undef DEFINE_CONVERT_NATIVE

static inline bool convert_float_i128(double value, struct dora_i128* out_value, enum dora_rounding_mode mode) {
    if (!isfinite(value)) {
        return true;
    }
    double rounded = mode == DORA_ROUNDING_MODE_HALF_EVEN   ? nearbyint(value)
                     : mode == DORA_ROUNDING_MODE_HALF_AWAY ? round(value)
                                                            : trunc(value);
    if ((mode == DORA_ROUNDING_MODE_EXACT && rounded != value) || rounded <= -0x1p127 || rounded >= 0x1p127) {
        return true;
    }
#if defined(__GNUC__) && !defined(__clang__)
    const _BitInt(128) signed_value = (_BitInt(128))rounded;
    *out_value = type_i128_store(signed_value);
#else
    const bool negative = rounded < 0;
    if (negative) {
        rounded = -rounded;
    }
    uint64_t lower = (uint64_t)fmod(rounded, 0x1p64);
    int64_t upper = (int64_t)(rounded / 0x1p64);
    if (negative) {
        lower = UINT64_MAX - lower + 1;
        upper = -1 - upper + (int64_t)(lower == 0);
    }
    *out_value = (struct dora_i128){.limbs = {lower, (uint64_t)upper}};
#endif
    return false;
}

static inline bool convert_float_u128(double value, struct dora_u128* out_value, enum dora_rounding_mode mode) {
    if (!isfinite(value)) {
        return true;
    }
    double rounded = mode == DORA_ROUNDING_MODE_HALF_EVEN   ? nearbyint(value)
                     : mode == DORA_ROUNDING_MODE_HALF_AWAY ? round(value)
                                                            : trunc(value);
    if ((mode == DORA_ROUNDING_MODE_EXACT && rounded != value) || rounded < 0.0 || rounded >= 0x1p128) {
        return true;
    }
    const unsigned _BitInt(128) unsigned_value = (unsigned _BitInt(128))rounded;
    *out_value = type_u128_store(unsigned_value);
    return false;
}

#define DEFINE_CONVERT_WIDE(D)                                                                                         \
    static inline bool convert_float_##D(double value,                                                                 \
                                         typeof(type_load_##D(nullptr, 0))* out_value,                                 \
                                         enum dora_rounding_mode mode) {                                               \
        struct dora_i256 integer = {};                                                                                 \
        return cast_double_integer(value, &integer, mode) != DORA_OK ||                                                \
               dora_numeric_store_integer(integer, PHYSICAL_##D, out_value) != DORA_OK;                                \
    }
DEFINE_CONVERT_WIDE(i256)
#undef DEFINE_CONVERT_WIDE

#define DEFINE_CONVERT_FLOAT(D)                                                                                        \
    static inline bool convert_signed_##D(int64_t value,                                                               \
                                          typeof(type_load_##D(nullptr, 0))* out_value,                                \
                                          enum dora_rounding_mode mode) {                                              \
        const bool exact = mode == DORA_ROUNDING_MODE_EXACT;                                                           \
        const double number = (double)value;                                                                           \
        double restored = 0;                                                                                           \
        *out_value = narrow_##D(number, &restored);                                                                    \
        return exact ? !roundtrip_signed_matches(restored, value) : !isfinite(restored);                               \
    }                                                                                                                  \
    static inline bool convert_unsigned_##D(uint64_t value,                                                            \
                                            typeof(type_load_##D(nullptr, 0))* out_value,                              \
                                            enum dora_rounding_mode mode) {                                            \
        const bool exact = mode == DORA_ROUNDING_MODE_EXACT;                                                           \
        const double number = (double)value;                                                                           \
        double restored = 0;                                                                                           \
        *out_value = narrow_##D(number, &restored);                                                                    \
        return exact ? !roundtrip_unsigned_matches(restored, value) : !isfinite(restored);                             \
    }                                                                                                                  \
    static inline bool convert_wide_##D(struct dora_i256 value,                                                        \
                                        typeof(type_load_##D(nullptr, 0))* out_value,                                  \
                                        enum dora_rounding_mode mode) {                                                \
        const bool exact = mode == DORA_ROUNDING_MODE_EXACT;                                                           \
        const double number = numeric_i256_to_double(value);                                                           \
        double restored = 0;                                                                                           \
        *out_value = narrow_##D(number, &restored);                                                                    \
        return cast_float_conversion_has_failed(number, restored, exact) ||                                            \
               (exact && !roundtrip_wide_matches(restored, value));                                                    \
    }                                                                                                                  \
    static inline bool convert_float_##D(double number,                                                                \
                                         typeof(type_load_##D(nullptr, 0))* out_value,                                 \
                                         enum dora_rounding_mode mode) {                                               \
        double restored = 0;                                                                                           \
        *out_value = narrow_##D(number, &restored);                                                                    \
        return cast_float_conversion_has_failed(number, restored, mode == DORA_ROUNDING_MODE_EXACT);                   \
    }
DEFINE_CONVERT_FLOAT(f16)
DEFINE_CONVERT_FLOAT(f32)
DEFINE_CONVERT_FLOAT(f64)
#undef DEFINE_CONVERT_FLOAT

#define DEFINE_DECIMAL_INTEGER_FLOAT(D)                                                                                \
    static inline bool convert_decimal_integer_##D(int64_t value,                                                      \
                                                   typeof(type_load_##D(nullptr, 0))* out_value,                       \
                                                   enum dora_rounding_mode mode) {                                     \
        *out_value = (typeof(*out_value))value;                                                                        \
        return mode == DORA_ROUNDING_MODE_EXACT && !roundtrip_signed_matches((double)*out_value, value);               \
    }
DEFINE_DECIMAL_INTEGER_FLOAT(f32)
DEFINE_DECIMAL_INTEGER_FLOAT(f64)
#undef DEFINE_DECIMAL_INTEGER_FLOAT

#define DEFINE_NARROW_UNSIGNED(D, MAXIMUM)                                                                             \
    static inline bool narrow_unsigned_##D(uint64_t value, typeof(type_load_##D(nullptr, 0))* out_value) {             \
        if (value <= (uint64_t)(MAXIMUM)) {                                                                            \
            *out_value = (typeof(type_load_##D(nullptr, 0)))value;                                                     \
            return false;                                                                                              \
        }                                                                                                              \
        return true;                                                                                                   \
    }
DEFINE_NARROW_UNSIGNED(i8, INT8_MAX)
DEFINE_NARROW_UNSIGNED(i16, INT16_MAX)
DEFINE_NARROW_UNSIGNED(i32, INT32_MAX)
DEFINE_NARROW_UNSIGNED(i64, INT64_MAX)
#undef DEFINE_NARROW_UNSIGNED

#define NARROW_UNSIGNED_i8(VALUE, OUT) narrow_unsigned_i8(VALUE, OUT)
#define NARROW_UNSIGNED_i16(VALUE, OUT) narrow_unsigned_i16(VALUE, OUT)
#define NARROW_UNSIGNED_i32(VALUE, OUT) narrow_unsigned_i32(VALUE, OUT)
#define NARROW_UNSIGNED_i64(VALUE, OUT) narrow_unsigned_i64(VALUE, OUT)
#define NARROW_UNSIGNED_u8(VALUE, OUT) ckd_add(OUT, VALUE, (uint8_t)0)
#define NARROW_UNSIGNED_u16(VALUE, OUT) ckd_add(OUT, VALUE, (uint16_t)0)
#define NARROW_UNSIGNED_u32(VALUE, OUT) ckd_add(OUT, VALUE, (uint32_t)0)
#define NARROW_UNSIGNED_u64(VALUE, OUT) ckd_add(OUT, VALUE, (uint64_t)0)

#define DEFINE_NARROW_I128(D, MINIMUM, MAXIMUM)                                                                        \
    static inline bool narrow_i128_##D(_BitInt(128) value, typeof(type_load_##D(nullptr, 0))* out_value) {             \
        if (value < (MINIMUM) || value > (MAXIMUM)) {                                                                  \
            return true;                                                                                               \
        }                                                                                                              \
        *out_value = (typeof(type_load_##D(nullptr, 0)))value;                                                         \
        return false;                                                                                                  \
    }
DEFINE_NARROW_I128(i8, INT8_MIN, INT8_MAX)
DEFINE_NARROW_I128(i16, INT16_MIN, INT16_MAX)
DEFINE_NARROW_I128(i32, INT32_MIN, INT32_MAX)
DEFINE_NARROW_I128(i64, INT64_MIN, INT64_MAX)
DEFINE_NARROW_I128(u8, 0, UINT8_MAX)
DEFINE_NARROW_I128(u16, 0, UINT16_MAX)
DEFINE_NARROW_I128(u32, 0, UINT32_MAX)
DEFINE_NARROW_I128(u64, 0, UINT64_MAX)
#undef DEFINE_NARROW_I128

#define CONVERT_SIGNED_NATIVE(D, VALUE, OUT) ckd_add(OUT, VALUE, (typeof(type_load_##D(nullptr, 0)))0)
#define CONVERT_UNSIGNED_NATIVE(D, VALUE, OUT) NARROW_UNSIGNED_##D(VALUE, OUT)
#define CONVERT_WIDE_NATIVE(D, VALUE, OUT) (dora_numeric_store_integer(VALUE, PHYSICAL_##D, OUT) != DORA_OK)
#define CONVERT_SIGNED_WIDE(D, VALUE, OUT)                                                                             \
    (dora_numeric_store_integer(numeric_i256_from_i64(VALUE), PHYSICAL_##D, OUT) != DORA_OK)
#define CONVERT_UNSIGNED_WIDE(D, VALUE, OUT)                                                                           \
    (dora_numeric_store_integer(numeric_i256_from_u64(VALUE), PHYSICAL_##D, OUT) != DORA_OK)
#define CONVERT_WIDE_WIDE(D, VALUE, OUT) (dora_numeric_store_integer(VALUE, PHYSICAL_##D, OUT) != DORA_OK)
#define CONVERT_SIGNED_I128(D, VALUE, OUT) (*(OUT) = type_i128_store(VALUE), false)
#define CONVERT_UNSIGNED_I128(D, VALUE, OUT) (*(OUT) = type_i128_store(VALUE), false)
#define CONVERT_WIDE_I128(D, VALUE, OUT) (dora_numeric_store_integer(VALUE, PHYSICAL_##D, OUT) != DORA_OK)
#define CONVERT_I128_I128(D, VALUE, OUT) (*(OUT) = type_i128_store(VALUE), false)
#define CONVERT_I128_NATIVE(D, VALUE, OUT) narrow_i128_##D(VALUE, OUT)
#define CONVERT_I128_WIDE(D, VALUE, OUT)                                                                               \
    (dora_numeric_store_integer(numeric_i256_from_i128(type_i128_store(VALUE)), PHYSICAL_##D, OUT) != DORA_OK)

#define CONVERT_I128_FLOAT(D, VALUE, OUT, MODE)                                                                        \
    convert_wide_##D(numeric_i256_from_i128(type_i128_store(VALUE)), OUT, MODE)
#define CONVERT_FLOAT_NATIVE(D, VALUE, OUT, MODE) convert_float_##D(VALUE, OUT, MODE)
#define CONVERT_FLOAT_I128(D, VALUE, OUT, MODE) convert_float_##D(VALUE, OUT, MODE)
#define CONVERT_FLOAT_WIDE(D, VALUE, OUT, MODE) convert_float_##D(VALUE, OUT, MODE)
#define CONVERT_SIGNED_FLOAT(D, VALUE, OUT, MODE) convert_signed_##D(VALUE, OUT, MODE)
#define CONVERT_UNSIGNED_FLOAT(D, VALUE, OUT, MODE) convert_unsigned_##D(VALUE, OUT, MODE)
#define CONVERT_WIDE_FLOAT(D, VALUE, OUT, MODE) convert_wide_##D(VALUE, OUT, MODE)
#define CONVERT_FLOAT_FLOAT(D, VALUE, OUT, MODE) convert_float_##D(VALUE, OUT, MODE)

// Inexact integer-to-float modes share nearest rounding.
#define MODE_EXACT_OR_NEAREST(MODE, BODY)                                                                              \
    if ((MODE) == DORA_ROUNDING_MODE_EXACT) {                                                                          \
        BODY(DORA_ROUNDING_MODE_EXACT)                                                                                 \
    } else {                                                                                                           \
        BODY(DORA_ROUNDING_MODE_HALF_EVEN)                                                                             \
    }
// Returns ERANGE from the enclosing function unless policy nulls failed conversions.
#define NUMERIC_LANES(ELEMENT, ...)                                                                                    \
    for (uint32_t word = 0; word * 64 < count; ++word) {                                                               \
        const uint64_t tail = bitmap_word_tail_mask(count, word);                                                      \
        const uint64_t base = (uint64_t)word * 64;                                                                     \
        const uint64_t lanes = count - base < 64 ? count - base : 64;                                                  \
        uint64_t valid = source->validity == nullptr ? tail : source->validity[word] & tail;                           \
        if (valid != 0) {                                                                                              \
            uint64_t failed = 0;                                                                                       \
            if (valid == tail && source->shape == KERNEL_SHAPE_FLAT) {                                                 \
                failed = word_convert(values + (offset + base) * source_width,                                         \
                                      lanes,                                                                           \
                                      results + base __VA_OPT__(, ) __VA_ARGS__);                                      \
            } else if (valid == tail) {                                                                                \
                for (uint64_t lane = 0; lane < lanes; ++lane) {                                                        \
                    failed |= (uint64_t)lane_convert(values,                                                           \
                                                     ELEMENT(base + lane),                                             \
                                                     results + base + lane __VA_OPT__(, ) __VA_ARGS__)                 \
                              << lane;                                                                                 \
                }                                                                                                      \
            } else {                                                                                                   \
                for (uint64_t pending = valid; pending != 0;) {                                                        \
                    const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);                                   \
                    failed |= (uint64_t)lane_convert(values,                                                           \
                                                     ELEMENT(base + lane),                                             \
                                                     results + base + lane __VA_OPT__(, ) __VA_ARGS__)                 \
                              << lane;                                                                                 \
                }                                                                                                      \
            }                                                                                                          \
            if (failed != 0 && !cast_should_null_failure(binding, ERANGE)) {                                           \
                return ERANGE;                                                                                         \
            }                                                                                                          \
            valid &= ~failed;                                                                                          \
        }                                                                                                              \
        if (out_result->validity != nullptr) {                                                                         \
            out_result->validity[word] = valid;                                                                        \
        }                                                                                                              \
        null_count += dora_bitmap_word_count_ones(tail & ~valid);                                                      \
    }

#define NUMERIC_BODY(...)                                                                                              \
    {                                                                                                                  \
        __VA_OPT__(const enum dora_rounding_mode lane_mode = __VA_ARGS__;)                                             \
        switch (source->shape) {                                                                                       \
        case KERNEL_SHAPE_FLAT:                                                                                        \
            NUMERIC_LANES(FLAT_ELEMENT __VA_OPT__(, lane_mode))                                                        \
            break;                                                                                                     \
        case KERNEL_SHAPE_CONSTANT:                                                                                    \
            NUMERIC_LANES(CONSTANT_ELEMENT __VA_OPT__(, lane_mode))                                                    \
            break;                                                                                                     \
        case KERNEL_SHAPE_GATHERED:                                                                                    \
            NUMERIC_LANES(MAP_ELEMENT __VA_OPT__(, lane_mode))                                                         \
            break;                                                                                                     \
        }                                                                                                              \
    }

#define NUMERIC_LEAF(NAME, S, D, LANE, WORD, TRAVERSAL)                                                                \
    static int NAME(const struct kernel_packet* source,                                                                \
                    const struct kernel_unary_binding* binding,                                                        \
                    struct kernel_result* out_result) {                                                                \
        const uint32_t count = source->count;                                                                          \
        const uint8_t* const values = source->values;                                                                  \
        const uint64_t offset = source->value_offset;                                                                  \
        const uint32_t* const rows = source->rows;                                                                     \
        typeof(type_load_##D(nullptr, 0))* const results = out_result->values;                                         \
        uint32_t null_count = 0;                                                                                       \
        const size_t source_width = sizeof(typeof(type_load_##S(nullptr, 0)));                                         \
        auto const lane_convert = LANE;                                                                                \
        auto const word_convert = WORD;                                                                                \
        TRAVERSAL                                                                                                      \
        LEAF_FINISH;                                                                                                   \
    }

// Signed integers, unsigned integers; pairwise
#define DEFINE_CAST_INTEGER(S, KS, D, KD)                                                                              \
    static inline bool lane_##S##_##D(const uint8_t* values,                                                           \
                                      uint64_t element,                                                                \
                                      typeof(type_load_##D(nullptr, 0))* out_slot) {                                   \
        const typeof(type_load_##S(nullptr, 0)) value = type_load_##S(values, element);                                \
        return CONVERT_##KS##_##KD(D, SOURCE_INTERMEDIATE_##KS(S, value), out_slot);                                   \
    }                                                                                                                  \
    static inline uint64_t word_##S##_##D(const uint8_t* restrict values,                                              \
                                          uint64_t lanes,                                                              \
                                          typeof(type_load_##D(nullptr, 0))* restrict results) {                       \
        uint64_t failing = 0;                                                                                          \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            failing |= (uint64_t)lane_##S##_##D(values, lane, results + lane);                                         \
        }                                                                                                              \
        if (failing == 0) {                                                                                            \
            return 0;                                                                                                  \
        }                                                                                                              \
        uint64_t failed = 0;                                                                                           \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            failed |= (uint64_t)lane_##S##_##D(values, lane, results + lane) << lane;                                  \
        }                                                                                                              \
        return failed;                                                                                                 \
    }                                                                                                                  \
    NUMERIC_LEAF(cast_##S##_##D, S, D, lane_##S##_##D, word_##S##_##D, NUMERIC_BODY())

#define DEFINE_CAST_ROUNDED(NAME, S, KS, D, KD)                                                                        \
    static inline bool lane_##NAME(const uint8_t* values,                                                              \
                                   uint64_t element,                                                                   \
                                   typeof(type_load_##D(nullptr, 0))* out_slot,                                        \
                                   enum dora_rounding_mode mode) {                                                     \
        const typeof(type_load_##S(nullptr, 0)) value = type_load_##S(values, element);                                \
        return CONVERT_##KS##_##KD(D, SOURCE_INTERMEDIATE_##KS(S, value), out_slot, mode);                             \
    }                                                                                                                  \
    static inline uint64_t word_##NAME(const uint8_t* restrict values,                                                 \
                                       uint64_t lanes,                                                                 \
                                       typeof(type_load_##D(nullptr, 0))* restrict results,                            \
                                       enum dora_rounding_mode mode) {                                                 \
        uint64_t failing = 0;                                                                                          \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            failing |= (uint64_t)lane_##NAME(values, lane, results + lane, mode);                                      \
        }                                                                                                              \
        if (failing == 0) {                                                                                            \
            return 0;                                                                                                  \
        }                                                                                                              \
        uint64_t failed = 0;                                                                                           \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            failed |= (uint64_t)lane_##NAME(values, lane, results + lane, mode) << lane;                               \
        }                                                                                                              \
        return failed;                                                                                                 \
    }

#define DEFINE_CAST_REQUESTED_MODE(S, KS, D, KD)                                                                       \
    DEFINE_CAST_ROUNDED(S##_##D, S, KS, D, KD)                                                                         \
    NUMERIC_LEAF(cast_##S##_##D, S, D, lane_##S##_##D, word_##S##_##D, NUMERIC_BODY(binding->request.rounding))

#define DEFINE_CAST_EXACT_OR_NEAREST(S, KS, D, KD)                                                                     \
    DEFINE_CAST_ROUNDED(S##_##D, S, KS, D, KD)                                                                         \
    NUMERIC_LEAF(cast_##S##_##D,                                                                                       \
                 S,                                                                                                    \
                 D,                                                                                                    \
                 lane_##S##_##D,                                                                                       \
                 word_##S##_##D,                                                                                       \
                 MODE_EXACT_OR_NEAREST(binding->request.rounding, NUMERIC_BODY))

#define DEFINE_CAST_EACH_MODE(S, KS, D, KD)                                                                            \
    DEFINE_CAST_ROUNDED(S##_##D, S, KS, D, KD)                                                                         \
    NUMERIC_LEAF(cast_exact_##S##_##D, S, D, lane_##S##_##D, word_##S##_##D, NUMERIC_BODY(DORA_ROUNDING_MODE_EXACT))   \
    NUMERIC_LEAF(cast_zero_##S##_##D,                                                                                  \
                 S,                                                                                                    \
                 D,                                                                                                    \
                 lane_##S##_##D,                                                                                       \
                 word_##S##_##D,                                                                                       \
                 NUMERIC_BODY(DORA_ROUNDING_MODE_TOWARD_ZERO))                                                         \
    NUMERIC_LEAF(cast_##S##_##D, S, D, lane_##S##_##D, word_##S##_##D, NUMERIC_BODY(DORA_ROUNDING_MODE_HALF_EVEN))     \
    NUMERIC_LEAF(cast_away_##S##_##D, S, D, lane_##S##_##D, word_##S##_##D, NUMERIC_BODY(DORA_ROUNDING_MODE_HALF_AWAY))

#define CAST_PAIR_SIGNED_NATIVE DEFINE_CAST_INTEGER
#define CAST_PAIR_UNSIGNED_NATIVE DEFINE_CAST_INTEGER
#define CAST_PAIR_WIDE_NATIVE DEFINE_CAST_INTEGER
#define CAST_PAIR_I128_NATIVE DEFINE_CAST_INTEGER
#define CAST_PAIR_FLOAT_NATIVE DEFINE_CAST_EACH_MODE
#define CAST_PAIR_SIGNED_WIDE DEFINE_CAST_INTEGER
#define CAST_PAIR_UNSIGNED_WIDE DEFINE_CAST_INTEGER
#define CAST_PAIR_WIDE_WIDE DEFINE_CAST_INTEGER
#define CAST_PAIR_I128_WIDE DEFINE_CAST_INTEGER
#define CAST_PAIR_FLOAT_WIDE DEFINE_CAST_REQUESTED_MODE
#define CAST_PAIR_SIGNED_I128 DEFINE_CAST_INTEGER
#define CAST_PAIR_UNSIGNED_I128 DEFINE_CAST_INTEGER
#define CAST_PAIR_WIDE_I128 DEFINE_CAST_INTEGER
#define CAST_PAIR_I128_I128 DEFINE_CAST_INTEGER
#define CAST_PAIR_FLOAT_I128 DEFINE_CAST_REQUESTED_MODE
#define CAST_PAIR_SIGNED_FLOAT DEFINE_CAST_EXACT_OR_NEAREST
#define CAST_PAIR_UNSIGNED_FLOAT DEFINE_CAST_EXACT_OR_NEAREST
#define CAST_PAIR_WIDE_FLOAT DEFINE_CAST_EXACT_OR_NEAREST
#define CAST_PAIR_I128_FLOAT DEFINE_CAST_EXACT_OR_NEAREST
#define CAST_PAIR_FLOAT_FLOAT DEFINE_CAST_EXACT_OR_NEAREST

#define CAST_TARGETS(X, S, KS)                                                                                         \
    X(S, KS, i8, NATIVE)                                                                                               \
    X(S, KS, i16, NATIVE)                                                                                              \
    X(S, KS, i32, NATIVE)                                                                                              \
    X(S, KS, i64, NATIVE)                                                                                              \
    X(S, KS, i128, I128)                                                                                               \
    X(S, KS, i256, WIDE)                                                                                               \
    X(S, KS, u8, NATIVE)                                                                                               \
    X(S, KS, u16, NATIVE)                                                                                              \
    X(S, KS, u32, NATIVE)                                                                                              \
    X(S, KS, u64, NATIVE)                                                                                              \
    X(S, KS, u128, WIDE)                                                                                               \
    X(S, KS, f16, FLOAT)                                                                                               \
    X(S, KS, f32, FLOAT)                                                                                               \
    X(S, KS, f64, FLOAT)

#define CAST_SOURCES(X)                                                                                                \
    X(i8, SIGNED)                                                                                                      \
    X(i16, SIGNED)                                                                                                     \
    X(i32, SIGNED)                                                                                                     \
    X(i64, SIGNED)                                                                                                     \
    X(i128, I128)                                                                                                      \
    X(i256, WIDE)                                                                                                      \
    X(u8, UNSIGNED)                                                                                                    \
    X(u16, UNSIGNED)                                                                                                   \
    X(u32, UNSIGNED)                                                                                                   \
    X(u64, UNSIGNED)                                                                                                   \
    X(u128, WIDE)                                                                                                      \
    X(f16, FLOAT)                                                                                                      \
    X(f32, FLOAT)                                                                                                      \
    X(f64, FLOAT)

// Signed integers, unsigned integers, floating-point types; pairwise
#define DEFINE_CAST_NUMERIC(S, KS, D, KD) CAST_PAIR_##KS##_##KD(S, KS, D, KD)
#define DEFINE_CAST_ROW(S, KS) CAST_TARGETS(DEFINE_CAST_NUMERIC, S, KS)
CAST_SOURCES(DEFINE_CAST_ROW)
#undef DEFINE_CAST_ROW
#undef DEFINE_CAST_NUMERIC

#define SOURCE_INTERMEDIATE_DECIMAL_INTEGER(S, VALUE) ((int64_t)type_i128_load(VALUE))
#define CONVERT_DECIMAL_INTEGER_FLOAT(D, VALUE, OUT, MODE) convert_decimal_integer_##D(VALUE, OUT, MODE)
// Decimal (128-bit, scale zero, precision <= 18) to Float32, Float64
#define DEFINE_CAST_DECIMAL_INTEGER(D)                                                                                 \
    DEFINE_CAST_ROUNDED(decimal_integer_##D, i128, DECIMAL_INTEGER, D, FLOAT)                                          \
    NUMERIC_LEAF(cast_decimal_integer_##D,                                                                             \
                 i128,                                                                                                 \
                 D,                                                                                                    \
                 lane_decimal_integer_##D,                                                                             \
                 word_decimal_integer_##D,                                                                             \
                 MODE_EXACT_OR_NEAREST(binding->request.rounding, NUMERIC_BODY))
DEFINE_CAST_DECIMAL_INTEGER(f32)
DEFINE_CAST_DECIMAL_INTEGER(f64)
#undef DEFINE_CAST_DECIMAL_INTEGER
#undef CONVERT_DECIMAL_INTEGER_FLOAT
#undef SOURCE_INTERMEDIATE_DECIMAL_INTEGER

#undef CAST_PAIR_FLOAT_FLOAT
#undef CAST_PAIR_I128_FLOAT
#undef CAST_PAIR_WIDE_FLOAT
#undef CAST_PAIR_UNSIGNED_FLOAT
#undef CAST_PAIR_SIGNED_FLOAT
#undef CAST_PAIR_FLOAT_I128
#undef CAST_PAIR_I128_I128
#undef CAST_PAIR_WIDE_I128
#undef CAST_PAIR_UNSIGNED_I128
#undef CAST_PAIR_SIGNED_I128
#undef CAST_PAIR_FLOAT_WIDE
#undef CAST_PAIR_I128_WIDE
#undef CAST_PAIR_WIDE_WIDE
#undef CAST_PAIR_UNSIGNED_WIDE
#undef CAST_PAIR_SIGNED_WIDE
#undef CAST_PAIR_FLOAT_NATIVE
#undef CAST_PAIR_I128_NATIVE
#undef CAST_PAIR_WIDE_NATIVE
#undef CAST_PAIR_UNSIGNED_NATIVE
#undef CAST_PAIR_SIGNED_NATIVE
#undef DEFINE_CAST_EACH_MODE
#undef DEFINE_CAST_EXACT_OR_NEAREST
#undef DEFINE_CAST_REQUESTED_MODE
#undef DEFINE_CAST_ROUNDED
#undef DEFINE_CAST_INTEGER
#undef MODE_EXACT_OR_NEAREST
#undef CONVERT_FLOAT_FLOAT
#undef CONVERT_WIDE_FLOAT
#undef CONVERT_UNSIGNED_FLOAT
#undef CONVERT_SIGNED_FLOAT
#undef CONVERT_FLOAT_WIDE
#undef CONVERT_FLOAT_I128
#undef CONVERT_FLOAT_NATIVE
#undef CONVERT_I128_FLOAT
#undef CONVERT_I128_WIDE
#undef CONVERT_I128_NATIVE
#undef CONVERT_I128_I128
#undef CONVERT_WIDE_I128
#undef CONVERT_UNSIGNED_I128
#undef CONVERT_SIGNED_I128
#undef CONVERT_WIDE_WIDE
#undef CONVERT_UNSIGNED_WIDE
#undef CONVERT_SIGNED_WIDE
#undef CONVERT_WIDE_NATIVE
#undef CONVERT_UNSIGNED_NATIVE
#undef CONVERT_SIGNED_NATIVE
#undef NARROW_UNSIGNED_u64
#undef NARROW_UNSIGNED_u32
#undef NARROW_UNSIGNED_u16
#undef NARROW_UNSIGNED_u8
#undef NARROW_UNSIGNED_i64
#undef NARROW_UNSIGNED_i32
#undef NARROW_UNSIGNED_i16
#undef NARROW_UNSIGNED_i8
#undef SOURCE_INTERMEDIATE_FLOAT
#undef SOURCE_INTERMEDIATE_I128
#undef SOURCE_INTERMEDIATE_WIDE
#undef SOURCE_INTERMEDIATE_UNSIGNED
#undef SOURCE_INTERMEDIATE_SIGNED
#undef SOURCE_FLOAT_f64
#undef SOURCE_FLOAT_f32
#undef SOURCE_FLOAT_f16
#undef SOURCE_WIDE_u128
#undef SOURCE_WIDE_i256
#undef SOURCE_UNSIGNED
#undef SOURCE_SIGNED

// Requires statistics-proven bounds.
// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64; pairwise
#define DEFINE_CAST_UNCHECKED(S, D)                                                                                    \
    static inline bool unchecked_lane_##S##_##D(const uint8_t* values,                                                 \
                                                uint64_t element,                                                      \
                                                typeof(type_load_##D(nullptr, 0))* out_slot) {                         \
        const typeof(type_load_##S(nullptr, 0)) value = type_load_##S(values, element);                                \
        *out_slot = (typeof(type_load_##D(nullptr, 0)))value;                                                          \
        return false;                                                                                                  \
    }                                                                                                                  \
    static inline uint64_t unchecked_word_##S##_##D(const uint8_t* restrict values,                                    \
                                                    uint64_t lanes,                                                    \
                                                    typeof(type_load_##D(nullptr, 0))* restrict results) {             \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            const typeof(type_load_##S(nullptr, 0)) value = type_load_##S(values, lane);                               \
            results[lane] = (typeof(type_load_##D(nullptr, 0)))value;                                                  \
        }                                                                                                              \
        return 0;                                                                                                      \
    }                                                                                                                  \
    NUMERIC_LEAF(cast_unchecked_##S##_##D, S, D, unchecked_lane_##S##_##D, unchecked_word_##S##_##D, NUMERIC_BODY())

#define UNCHECKED_TARGETS(X, S)                                                                                        \
    X(S, i8)                                                                                                           \
    X(S, i16)                                                                                                          \
    X(S, i32)                                                                                                          \
    X(S, i64)                                                                                                          \
    X(S, u8)                                                                                                           \
    X(S, u16)                                                                                                          \
    X(S, u32)                                                                                                          \
    X(S, u64)

#define UNCHECKED_SOURCES(X)                                                                                           \
    X(i8)                                                                                                              \
    X(i16)                                                                                                             \
    X(i32)                                                                                                             \
    X(i64)                                                                                                             \
    X(u8)                                                                                                              \
    X(u16)                                                                                                             \
    X(u32)                                                                                                             \
    X(u64)

#define DEFINE_UNCHECKED_ROW(S) UNCHECKED_TARGETS(DEFINE_CAST_UNCHECKED, S)
UNCHECKED_SOURCES(DEFINE_UNCHECKED_ROW)
#undef DEFINE_UNCHECKED_ROW
#undef DEFINE_CAST_UNCHECKED
#undef NUMERIC_LEAF
#undef NUMERIC_BODY
#undef NUMERIC_LANES

//--------------------------------------
// Decimal Rescale
//--------------------------------------

#define DECIMAL_i32 int32_t
#define DECIMAL_i64 int64_t
#define DECIMAL_i128 _BitInt(128)
#define DECIMAL_UNSIGNED_i32 uint32_t
#define DECIMAL_UNSIGNED_i64 uint64_t
#define DECIMAL_UNSIGNED_i128 typeof((unsigned _BitInt(128))0)
#define RESCALE_LOAD_i32 type_load_i32
#define RESCALE_LOAD_i64 type_load_i64
#define RESCALE_LOAD_i128 rescale_load_i128

static inline _BitInt(128) rescale_load_i128(const uint8_t* values, uint64_t element) {
    return type_i128_load(type_load_i128(values, element));
}

#define DEFINE_RESCALE_STORE(S, T, P)                                                                                  \
    static inline void rescale_store_##S(uint8_t* results, uint32_t row, DECIMAL_UNSIGNED_##S value) {                 \
        (memcpy)(results + (size_t)row * sizeof(value), &value, sizeof(value));                                        \
    }

DECIMAL_RESCALE_CARRIERS(DEFINE_RESCALE_STORE)
#undef DEFINE_RESCALE_STORE

#define RESCALE_UP_LANE(ROW, ELEMENT)                                                                                  \
    do {                                                                                                               \
        const typeof(limit) value = load(values, ELEMENT);                                                             \
        if (CHECK && (value >= limit || value <= -limit)) {                                                            \
            CAST_FAILED(ROW, EOVERFLOW);                                                                               \
        } else {                                                                                                       \
            store(results, ROW, (typeof(factor))value * factor);                                                       \
        }                                                                                                              \
    } while (0)

#define RESCALE_DOWN_LANE(ROW, ELEMENT)                                                                                \
    do {                                                                                                               \
        const typeof(factor) value = load(values, ELEMENT);                                                            \
        const typeof(factor) remainder = value % factor;                                                               \
        const typeof(factor) excess = remainder < 0 ? -remainder : remainder;                                          \
        const typeof(factor) quotient = value / factor;                                                                \
        const bool up =                                                                                                \
            (MODE == DORA_ROUNDING_MODE_HALF_AWAY && excess >= half) ||                                                \
            (MODE == DORA_ROUNDING_MODE_HALF_EVEN && (excess > half || (excess == half && quotient % 2 != 0)));        \
        const bool inexact = MODE == DORA_ROUNDING_MODE_EXACT && excess != 0;                                          \
        const typeof(limit) magnitude = value < 0 ? (typeof(limit))0 - (typeof(limit))value : (typeof(limit))value;    \
        int lane_status = DORA_OK;                                                                                     \
        if (inexact) {                                                                                                 \
            lane_status = ERANGE;                                                                                      \
        } else if (CHECK && magnitude + (up ? (typeof(limit))factor : 0) >= limit) {                                   \
            lane_status = EOVERFLOW;                                                                                   \
        }                                                                                                              \
        if (lane_status != DORA_OK) {                                                                                  \
            CAST_FAILED(ROW, lane_status);                                                                             \
        } else if (MODE == DORA_ROUNDING_MODE_HALF_AWAY) {                                                             \
            typeof_unqual(factor) scaled = value / half;                                                               \
            scaled += scaled < 0 ? -1 : 1;                                                                             \
            store(results, ROW, scaled / 2);                                                                           \
        } else {                                                                                                       \
            store(results, ROW, quotient + (up ? (value < 0 ? -1 : 1) : 0));                                           \
        }                                                                                                              \
    } while (0)

#define RESCALE_DOWN_MODE(MODE_VALUE)                                                                                  \
    case MODE_VALUE: {                                                                                                 \
        constexpr enum dora_rounding_mode MODE = MODE_VALUE;                                                           \
        CAST_SHAPES(RESCALE_DOWN_LANE)                                                                                 \
        break;                                                                                                         \
    }

// Decimal (32/64/128-bit); pairwise
#define DEFINE_RESCALE_PAIR(S, D, NAME, CHECK_VALUE)                                                                   \
    static int rescale_up_##NAME##_##S##_##D(const struct kernel_packet* source,                                       \
                                             const struct kernel_unary_binding* binding,                               \
                                             struct kernel_result* out_result) {                                       \
        LEAF_LOCALS;                                                                                                   \
        constexpr bool CHECK = CHECK_VALUE;                                                                            \
        const DECIMAL_##S limit = (DECIMAL_##S)type_i128_load(binding->cast.rescale.limit);                            \
        const DECIMAL_UNSIGNED_##D factor = (DECIMAL_UNSIGNED_##D)type_i128_load(binding->cast.rescale.factor);        \
        typeof(&RESCALE_LOAD_##S) const load = RESCALE_LOAD_##S;                                                       \
        typeof(&rescale_store_##D) const store = rescale_store_##D;                                                    \
        CAST_SHAPES(RESCALE_UP_LANE)                                                                                   \
        LEAF_FINISH;                                                                                                   \
    }                                                                                                                  \
    static int rescale_down_##NAME##_##S##_##D(const struct kernel_packet* source,                                     \
                                               const struct kernel_unary_binding* binding,                             \
                                               struct kernel_result* out_result) {                                     \
        LEAF_LOCALS;                                                                                                   \
        constexpr bool CHECK = CHECK_VALUE;                                                                            \
        const DECIMAL_UNSIGNED_##S limit = (DECIMAL_UNSIGNED_##S)type_i128_load(binding->cast.rescale.limit);          \
        const DECIMAL_##S factor = (DECIMAL_##S)type_i128_load(binding->cast.rescale.factor);                          \
        const DECIMAL_##S half = factor / 2;                                                                           \
        typeof(&RESCALE_LOAD_##S) const load = RESCALE_LOAD_##S;                                                       \
        typeof(&rescale_store_##D) const store = rescale_store_##D;                                                    \
        switch (binding->request.rounding) {                                                                           \
            RESCALE_DOWN_MODE(DORA_ROUNDING_MODE_EXACT)                                                                \
            RESCALE_DOWN_MODE(DORA_ROUNDING_MODE_TOWARD_ZERO)                                                          \
            RESCALE_DOWN_MODE(DORA_ROUNDING_MODE_HALF_EVEN)                                                            \
            RESCALE_DOWN_MODE(DORA_ROUNDING_MODE_HALF_AWAY)                                                            \
        }                                                                                                              \
        LEAF_FINISH;                                                                                                   \
    }

#define DEFINE_RESCALE_SOURCE(S, T, P)                                                                                 \
    DEFINE_RESCALE_PAIR(S, i32, checked, true)                                                                         \
    DEFINE_RESCALE_PAIR(S, i32, unchecked, false)                                                                      \
    DEFINE_RESCALE_PAIR(S, i64, checked, true)                                                                         \
    DEFINE_RESCALE_PAIR(S, i64, unchecked, false)                                                                      \
    DEFINE_RESCALE_PAIR(S, i128, checked, true)                                                                        \
    DEFINE_RESCALE_PAIR(S, i128, unchecked, false)

DECIMAL_RESCALE_CARRIERS(DEFINE_RESCALE_SOURCE)
#undef DEFINE_RESCALE_SOURCE
#undef DEFINE_RESCALE_PAIR
#undef RESCALE_DOWN_MODE
#undef RESCALE_DOWN_LANE
#undef RESCALE_UP_LANE
#undef RESCALE_LOAD_i128
#undef RESCALE_LOAD_i64
#undef RESCALE_LOAD_i32
#undef DECIMAL_UNSIGNED_i128
#undef DECIMAL_UNSIGNED_i64
#undef DECIMAL_UNSIGNED_i32
#undef DECIMAL_i128
#undef DECIMAL_i64
#undef DECIMAL_i32

//--------------------------------------
// Boolean
//--------------------------------------

static inline uint64_t
boolean_source_word(const struct kernel_packet* source, const uint64_t* gathered, uint32_t word, uint32_t lanes) {
    switch (source->shape) {
    case KERNEL_SHAPE_FLAT:
        return dora_bitmap_load_window(source->values, source->value_offset + (uint64_t)word * 64, lanes);
    case KERNEL_SHAPE_CONSTANT:
        return dora_bitmap_test(source->values, source->value_offset) ? UINT64_MAX : 0;
    case KERNEL_SHAPE_GATHERED:
        return gathered[word];
    }
    return 0;
}

#define BOOLEAN_WORDS(ACTIVE_BODY)                                                                                     \
    for (uint32_t word = 0; word * 64 < count; ++word) {                                                               \
        const uint32_t lanes = count - word * 64 < 64 ? count - word * 64 : 64;                                        \
        const uint64_t full = bitmap_word_tail_mask(count, word);                                                      \
        const uint64_t active = source->validity == nullptr ? full : source->validity[word] & full;                    \
        ACTIVE_BODY                                                                                                    \
        if (out_result->validity != nullptr) {                                                                         \
            out_result->validity[word] = active;                                                                       \
        }                                                                                                              \
        null_count += lanes - dora_bitmap_word_count_ones(active);                                                     \
    }

#define BOOLEAN_LANES(LANE)                                                                                            \
    if (active == full) {                                                                                              \
        for (uint32_t lane = 0; lane < lanes; ++lane) {                                                                \
            LANE(lane);                                                                                                \
        }                                                                                                              \
    } else if (active != 0 && dora_bitmap_word_count_ones(active) * 4U >= lanes * 3U) {                                \
        for (uint32_t lane = 0; lane < lanes; ++lane) {                                                                \
            if (bitmap_word_bit(active, lane) != 0) {                                                                  \
                LANE(lane);                                                                                            \
            }                                                                                                          \
        }                                                                                                              \
    } else {                                                                                                           \
        for (uint64_t remaining = active; remaining != 0;) {                                                           \
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&remaining);                                         \
            LANE(lane);                                                                                                \
        }                                                                                                              \
    }

// Boolean to Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64
#define DEFINE_BOOLEAN_TO(D)                                                                                           \
    static int cast_boolean_##D(const struct kernel_packet* source,                                                    \
                                const struct kernel_unary_binding*,                                                    \
                                struct kernel_result* out_result) {                                                    \
        const uint32_t count = source->count;                                                                          \
        typeof(type_load_##D(nullptr, 0))* const results = out_result->values;                                         \
        uint32_t null_count = 0;                                                                                       \
        uint64_t gathered[KERNEL_CHUNK_WORD_COUNT];                                                                    \
        if (source->shape == KERNEL_SHAPE_GATHERED) {                                                                  \
            const uint64_t* active_words = source->validity;                                                           \
            if (active_words == nullptr) {                                                                             \
                bitmap_fill_bytes((uint8_t*)gathered, sizeof gathered, true);                                          \
                active_words = gathered;                                                                               \
            }                                                                                                          \
            dora_bitmap_gather_words_u32(gathered,                                                                     \
                                         source->values,                                                               \
                                         source->value_offset,                                                         \
                                         source->rows,                                                                 \
                                         active_words,                                                                 \
                                         count);                                                                       \
        }                                                                                                              \
        BOOLEAN_WORDS({                                                                                                \
            const uint64_t bits = boolean_source_word(source, gathered, word, lanes);                                  \
            typeof(type_load_##D(nullptr, 0))* const lane_out = results + (size_t)word * 64;                           \
            BOOLEAN_LANES(BOOLEAN_UNPACK_LANE)                                                                         \
        })                                                                                                             \
        LEAF_FINISH;                                                                                                   \
    }
#define BOOLEAN_UNPACK_LANE(LANE) (lane_out[LANE] = (typeof(*lane_out))bitmap_word_bit(bits, (LANE)))
DEFINE_BOOLEAN_TO(i8)
DEFINE_BOOLEAN_TO(i16)
DEFINE_BOOLEAN_TO(i32)
DEFINE_BOOLEAN_TO(i64)
DEFINE_BOOLEAN_TO(u8)
DEFINE_BOOLEAN_TO(u16)
DEFINE_BOOLEAN_TO(u32)
DEFINE_BOOLEAN_TO(u64)
DEFINE_BOOLEAN_TO(f32)
DEFINE_BOOLEAN_TO(f64)
#undef BOOLEAN_UNPACK_LANE
#undef DEFINE_BOOLEAN_TO

// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64 to Boolean
#define DEFINE_TO_BOOLEAN(S)                                                                                           \
    static int cast_##S##_boolean(const struct kernel_packet* source,                                                  \
                                  const struct kernel_unary_binding*,                                                  \
                                  struct kernel_result* out_result) {                                                  \
        const uint32_t count = source->count;                                                                          \
        const uint8_t* const values =                                                                                  \
            (const uint8_t*)source->values + source->value_offset * sizeof(typeof(type_load_##S(nullptr, 0)));         \
        auto const load = type_load_##S;                                                                               \
        const uint32_t* const rows = source->rows;                                                                     \
        uint64_t* const results = out_result->values;                                                                  \
        uint32_t null_count = 0;                                                                                       \
        BOOLEAN_WORDS({                                                                                                \
            uint64_t bits = 0;                                                                                         \
            switch (source->shape) {                                                                                   \
            case KERNEL_SHAPE_FLAT:                                                                                    \
                BOOLEAN_LANES(BOOLEAN_FLAT_LANE)                                                                       \
                break;                                                                                                 \
            case KERNEL_SHAPE_CONSTANT:                                                                                \
                BOOLEAN_LANES(BOOLEAN_CONSTANT_LANE)                                                                   \
                break;                                                                                                 \
            case KERNEL_SHAPE_GATHERED:                                                                                \
                BOOLEAN_LANES(BOOLEAN_GATHERED_LANE)                                                                   \
                break;                                                                                                 \
            }                                                                                                          \
            results[word] = (results[word] & ~full) | bits;                                                            \
        })                                                                                                             \
        LEAF_FINISH;                                                                                                   \
    }
#define BOOLEAN_FLAT_LANE(LANE) (bits |= (uint64_t)(load(values, word * 64 + (LANE)) != 0) << (LANE))
#define BOOLEAN_CONSTANT_LANE(LANE) (bits |= (uint64_t)(load(values, 0) != 0) << (LANE))
#define BOOLEAN_GATHERED_LANE(LANE) (bits |= (uint64_t)(load(values, rows[word * 64 + (LANE)]) != 0) << (LANE))
DEFINE_TO_BOOLEAN(i8)
DEFINE_TO_BOOLEAN(i16)
DEFINE_TO_BOOLEAN(i32)
DEFINE_TO_BOOLEAN(i64)
DEFINE_TO_BOOLEAN(u8)
DEFINE_TO_BOOLEAN(u16)
DEFINE_TO_BOOLEAN(u32)
DEFINE_TO_BOOLEAN(u64)
#undef BOOLEAN_GATHERED_LANE
#undef BOOLEAN_CONSTANT_LANE
#undef BOOLEAN_FLAT_LANE
#undef DEFINE_TO_BOOLEAN
#undef BOOLEAN_LANES
#undef BOOLEAN_WORDS

//--------------------------------------
// Temporal
//--------------------------------------

static constexpr int64_t CAST_MICROS_PER_DAY = INT64_C(86'400'000'000);

#define TIMESTAMP_SCALE_LANE(ROW, ELEMENT)                                                                             \
    do {                                                                                                               \
        const int64_t value = type_load_i64(values, ELEMENT);                                                          \
        int64_t result = 0;                                                                                            \
        if (ckd_mul(&result, value, factor)) {                                                                         \
            CAST_FAILED(ROW, ERANGE);                                                                                  \
        } else {                                                                                                       \
            (memcpy)(results + (size_t)(ROW) * sizeof(result), &result, sizeof(result));                               \
        }                                                                                                              \
    } while (0)

// Timestamp to Timestamp (finer unit)
static int cast_timestamp_scale(const struct kernel_packet* source,
                                const struct kernel_unary_binding* binding,
                                struct kernel_result* out_result) {
    LEAF_LOCALS;
    const int64_t factor = (int64_t)type_i128_load(binding->cast.rescale.factor);
    CAST_SHAPES(TIMESTAMP_SCALE_LANE)
    LEAF_FINISH;
}

#undef TIMESTAMP_SCALE_LANE

#define MILLIS_DATE_LANE(ROW, ELEMENT)                                                                                 \
    do {                                                                                                               \
        const int64_t value = type_load_i64(values, ELEMENT);                                                          \
        int64_t micros = 0;                                                                                            \
        if (ckd_mul(&micros, value, INT64_C(1000))) {                                                                  \
            CAST_FAILED(ROW, ERANGE);                                                                                  \
        } else {                                                                                                       \
            const int64_t before = micros < 0;                                                                         \
            const int32_t days = (int32_t)((micros + before) / CAST_MICROS_PER_DAY - before);                          \
            (memcpy)(results + (size_t)(ROW) * sizeof(days), &days, sizeof(days));                                     \
        }                                                                                                              \
    } while (0)

// Timestamp (ms) to Date32
static int cast_millis_date(const struct kernel_packet* source,
                            const struct kernel_unary_binding* binding,
                            struct kernel_result* out_result) {
    LEAF_LOCALS;
    CAST_SHAPES(MILLIS_DATE_LANE)
    LEAF_FINISH;
}

#undef MILLIS_DATE_LANE

//--------------------------------------
// Scalar Values
//--------------------------------------

static inline union cast_value
load_value(const struct kernel_unary_binding* binding, const uint8_t* values, uint64_t element) {
    union cast_value value = {};
    if (binding->input_byte_width == 0) {
        value.boolean = dora_bitmap_test(values, element);
    } else {
        (memcpy)(&value, values + element * binding->input_byte_width, binding->input_byte_width);
    }
    return value;
}

static inline void
store_value(const struct kernel_unary_binding* binding, const union cast_value* value, uint8_t* results, uint32_t row) {
    if (binding->result_byte_width == 0) {
        bitmap_assign((uint64_t*)results, row, value->boolean);
    } else {
        (memcpy)(results + (size_t)row * binding->result_byte_width, value, binding->result_byte_width);
    }
}

#define VALUE_LANE(ROW, ELEMENT)                                                                                       \
    do {                                                                                                               \
        const union cast_value value = load_value(binding, values, ELEMENT);                                           \
        union cast_value result = {};                                                                                  \
        const int lane_status = cast_scalar(binding, &value, &result);                                                 \
        if (lane_status == DORA_OK) {                                                                                  \
            store_value(binding, &result, results, ROW);                                                               \
        } else {                                                                                                       \
            CAST_FAILED(ROW, lane_status);                                                                             \
        }                                                                                                              \
    } while (0)

// Boolean, signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals
static int cast_values(const struct kernel_packet* source,
                       const struct kernel_unary_binding* binding,
                       struct kernel_result* out_result) {
    LEAF_LOCALS;
    CAST_SHAPES(VALUE_LANE)
    LEAF_FINISH;
}

#undef VALUE_LANE

//--------------------------------------
// Text
//--------------------------------------

#define PARSE_LANE(ROW, ELEMENT)                                                                                       \
    do {                                                                                                               \
        const struct dora_bytes text = type_load_bytes(values, ELEMENT);                                               \
        union cast_value result = {};                                                                                  \
        const int lane_status =                                                                                        \
            parse(target, physical, (const char*)text.data, text.byte_count, rounding, bound, &result);                \
        if (lane_status == DORA_OK) {                                                                                  \
            store_value(binding, &result, results, ROW);                                                               \
        } else {                                                                                                       \
            CAST_FAILED(ROW, lane_status);                                                                             \
        }                                                                                                              \
    } while (0)

// Utf8, Binary, FixedBinary to Boolean, integers, floating-point types, Decimal, temporal types, intervals
static int cast_parse(const struct kernel_packet* source,
                      const struct kernel_unary_binding* binding,
                      struct kernel_result* out_result) {
    LEAF_LOCALS;
    auto const parse = binding->cast.parse;
    const struct dora_logical_type* const target = binding->request.result_type;
    const enum dora_physical_type physical = binding->result_physical;
    const enum dora_rounding_mode rounding = binding->request.rounding;
    const struct dora_i256* const bound =
        binding->result_decimal.type.precision != 0 ? &binding->result_decimal.bound : nullptr;
    CAST_SHAPES(PARSE_LANE)
    LEAF_FINISH;
}

#undef PARSE_LANE

#define MEASURE_LANE(ROW, ELEMENT)                                                                                     \
    do {                                                                                                               \
        const struct dora_bytes text = type_load_bytes(values, ELEMENT);                                               \
        const int64_t length = (int64_t)text.byte_count;                                                               \
        (memcpy)(results + (size_t)(ROW) * sizeof(length), &length, sizeof(length));                                   \
    } while (0)

static int
cast_measure(const struct kernel_packet* source, const struct kernel_unary_binding*, struct kernel_result* out_result) {
    LEAF_LOCALS;
    memset(results, 0, (size_t)count * sizeof(int64_t));
    CAST_SHAPES(MEASURE_LANE)
    LEAF_FINISH;
}

#undef MEASURE_LANE

#define TEXT_ROWS(ELEMENT, LANE, ...)                                                                                  \
    for (uint32_t word = 0; word * 64 < count; ++word) {                                                               \
        const uint64_t tail = bitmap_word_tail_mask(count, word);                                                      \
        const uint64_t active = source->validity == nullptr ? tail : source->validity[word] & tail;                    \
        const uint32_t end = count - word * 64 < 64 ? count : word * 64 + 64;                                          \
        uint64_t valid = active;                                                                                       \
        if (active == tail) {                                                                                          \
            for (uint32_t row = word * 64; row < end; ++row) {                                                         \
                LANE(row, ELEMENT(row), __VA_ARGS__);                                                                  \
                TEXT_END(row);                                                                                         \
            }                                                                                                          \
        } else if (active == 0) {                                                                                      \
            for (uint32_t row = word * 64; row < end; ++row) {                                                         \
                TEXT_END(row);                                                                                         \
            }                                                                                                          \
        } else {                                                                                                       \
            for (uint32_t row = word * 64; row < end; ++row) {                                                         \
                if (bitmap_word_bit(active, row - word * 64) != 0) {                                                   \
                    LANE(row, ELEMENT(row), __VA_ARGS__);                                                              \
                }                                                                                                      \
                TEXT_END(row);                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        if (out_result->validity != nullptr) {                                                                         \
            out_result->validity[word] = valid;                                                                        \
        }                                                                                                              \
        null_count += dora_bitmap_word_count_ones(tail & ~valid);                                                      \
    }

#define TEXT_SHAPES(LANE, ...)                                                                                         \
    switch (source->shape) {                                                                                           \
    case KERNEL_SHAPE_FLAT:                                                                                            \
        TEXT_ROWS(FLAT_ELEMENT, LANE, __VA_ARGS__)                                                                     \
        break;                                                                                                         \
    case KERNEL_SHAPE_CONSTANT:                                                                                        \
        TEXT_ROWS(CONSTANT_ELEMENT, LANE, __VA_ARGS__)                                                                 \
        break;                                                                                                         \
    case KERNEL_SHAPE_GATHERED:                                                                                        \
        TEXT_ROWS(MAP_ELEMENT, LANE, __VA_ARGS__)                                                                      \
        break;                                                                                                         \
    }

#define TEXT_LOCALS                                                                                                    \
    const uint32_t count = source->count;                                                                              \
    const uint8_t* const values = source->values;                                                                      \
    const uint64_t offset = source->value_offset;                                                                      \
    const uint32_t* const rows = source->rows;                                                                         \
    struct kernel_text_output* const output = out_result->values;                                                      \
    int64_t* const ends = output->end_offsets;                                                                         \
    uint8_t* const data = output->data;                                                                                \
    const size_t width = binding->result_byte_width;                                                                   \
    int64_t cursor = output->written_byte_count;                                                                       \
    uint32_t null_count = 0

#define TEXT_END(ROW)                                                                                                  \
    do {                                                                                                               \
        if (ends != nullptr) {                                                                                         \
            ends[ROW] = cursor;                                                                                        \
        }                                                                                                              \
    } while (0)

#define TEXT_PLACE(ROW, SIZE, WRITE)                                                                                   \
    do {                                                                                                               \
        if (width != 0 && (SIZE) != width) {                                                                           \
            CAST_FAILED(ROW, EINVAL);                                                                                  \
        } else {                                                                                                       \
            char* const text = (char*)data + (width != 0 ? (size_t)(ROW) * width : (size_t)cursor);                    \
            WRITE;                                                                                                     \
            cursor += width != 0 ? 0 : (int64_t)(SIZE);                                                                \
        }                                                                                                              \
    } while (0)

#define TEXT_FINISH                                                                                                    \
    output->written_byte_count = cursor;                                                                               \
    LEAF_FINISH

#define TEXT_BYTES_LANE(ROW, ELEMENT, ...)                                                                             \
    do {                                                                                                               \
        const struct dora_bytes bytes = type_load_bytes(values, ELEMENT);                                              \
        TEXT_PLACE(                                                                                                    \
            ROW,                                                                                                       \
            bytes.byte_count,                                                                                          \
            if (bytes.byte_count != 0) { (memcpy)(text, bytes.data, bytes.byte_count); });                             \
    } while (0)

// Utf8, Binary, FixedBinary; pairwise
static int cast_text_bytes(const struct kernel_packet* source,
                           const struct kernel_unary_binding* binding,
                           struct kernel_result* out_result) {
    TEXT_LOCALS;
    TEXT_SHAPES(TEXT_BYTES_LANE)
    TEXT_FINISH;
}

#undef TEXT_BYTES_LANE

#define TEXT_SIGNED_LANE(ROW, ELEMENT, ...)                                                                            \
    do {                                                                                                               \
        const int64_t number = (int64_t)load(values, ELEMENT);                                                         \
        const size_t size = dora_type_signed_length(number);                                                           \
        TEXT_PLACE(ROW, size, dora_type_format_signed(number, text + size));                                           \
    } while (0)

// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32 to Utf8, Binary, FixedBinary
#define DEFINE_TEXT_SIGNED(S)                                                                                          \
    static int cast_text_##S(const struct kernel_packet* source,                                                       \
                             const struct kernel_unary_binding* binding,                                               \
                             struct kernel_result* out_result) {                                                       \
        TEXT_LOCALS;                                                                                                   \
        auto const load = type_load_##S;                                                                               \
        TEXT_SHAPES(TEXT_SIGNED_LANE)                                                                                  \
        TEXT_FINISH;                                                                                                   \
    }
DEFINE_TEXT_SIGNED(i8)
DEFINE_TEXT_SIGNED(i16)
DEFINE_TEXT_SIGNED(i32)
DEFINE_TEXT_SIGNED(i64)
DEFINE_TEXT_SIGNED(u8)
DEFINE_TEXT_SIGNED(u16)
DEFINE_TEXT_SIGNED(u32)
#undef DEFINE_TEXT_SIGNED
#undef TEXT_SIGNED_LANE

#define TEXT_UNSIGNED_LANE(ROW, ELEMENT, ...)                                                                          \
    do {                                                                                                               \
        const uint64_t number = type_load_u64(values, ELEMENT);                                                        \
        const size_t size = dora_type_unsigned_length(number);                                                         \
        TEXT_PLACE(ROW, size, dora_type_format_unsigned(number, text + size));                                         \
    } while (0)

// UInt64 to Utf8, Binary, FixedBinary
static int cast_text_u64(const struct kernel_packet* source,
                         const struct kernel_unary_binding* binding,
                         struct kernel_result* out_result) {
    TEXT_LOCALS;
    TEXT_SHAPES(TEXT_UNSIGNED_LANE)
    TEXT_FINISH;
}

#undef TEXT_UNSIGNED_LANE

#define TEXT_INTEGER_LANE(ROW, ELEMENT, S, WIDEN)                                                                      \
    do {                                                                                                               \
        const struct dora_i256 coefficient = WIDEN(type_load_##S(values, ELEMENT));                                    \
        const int lane_status = dora_type_integer_text_prepare(coefficient, scale, &digits);                           \
        if (lane_status != DORA_OK) {                                                                                  \
            CAST_FAILED(ROW, lane_status);                                                                             \
        } else {                                                                                                       \
            TEXT_PLACE(ROW, digits.byte_count, dora_type_integer_text_write(&digits, text));                           \
        }                                                                                                              \
    } while (0)

// Int128, Int256, UInt128, Decimal (32/64/128/256-bit) to Utf8, Binary, FixedBinary
#define DEFINE_TEXT_INTEGER(S, T, P, WIDEN)                                                                            \
    static int cast_text_integer_##S(const struct kernel_packet* source,                                               \
                                     const struct kernel_unary_binding* binding,                                       \
                                     struct kernel_result* out_result) {                                               \
        TEXT_LOCALS;                                                                                                   \
        const struct dora_logical_type* const type = binding->request.input_type;                                      \
        const int32_t scale = type->id == DORA_TYPE_DECIMAL ? type->parameters.decimal.scale : 0;                      \
        struct type_integer_text digits;                                                                               \
        TEXT_SHAPES(TEXT_INTEGER_LANE, S, WIDEN)                                                                       \
        TEXT_FINISH;                                                                                                   \
    }
DEFINE_TEXT_INTEGER(i32, int32_t, DORA_PHYSICAL_TYPE_I32, numeric_i256_from_i64)
DEFINE_TEXT_INTEGER(i64, int64_t, DORA_PHYSICAL_TYPE_I64, numeric_i256_from_i64)
DEFINE_TEXT_INTEGER(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128, numeric_i256_from_i128)
DEFINE_TEXT_INTEGER(u128, struct dora_u128, DORA_PHYSICAL_TYPE_U128, numeric_i256_from_u128)
DEFINE_TEXT_INTEGER(i256, struct dora_i256, DORA_PHYSICAL_TYPE_I256, )
#undef DEFINE_TEXT_INTEGER

#undef TEXT_INTEGER_LANE

#define TEXT_FLOAT_LANE(ROW, ELEMENT, S, FORMAT, WIDEN)                                                                \
    do {                                                                                                               \
        char literal[TYPE_FLOAT_TEXT_CAPACITY];                                                                        \
        const size_t size = FORMAT(WIDEN(type_load_##S(values, ELEMENT)), literal);                                    \
        TEXT_PLACE(ROW, size, (memcpy)(text, literal, size));                                                          \
    } while (0)

// Float16, Float32, Float64 to Utf8, Binary, FixedBinary
#define DEFINE_TEXT_FLOAT(S, FORMAT, WIDEN)                                                                            \
    static int cast_text_##S(const struct kernel_packet* source,                                                       \
                             const struct kernel_unary_binding* binding,                                               \
                             struct kernel_result* out_result) {                                                       \
        TEXT_LOCALS;                                                                                                   \
        TEXT_SHAPES(TEXT_FLOAT_LANE, S, FORMAT, WIDEN)                                                                 \
        TEXT_FINISH;                                                                                                   \
    }
DEFINE_TEXT_FLOAT(f16, dora_type_format_f32, numeric_half_widen)
DEFINE_TEXT_FLOAT(f32, dora_type_format_f32, )
DEFINE_TEXT_FLOAT(f64, dora_type_format_f64, )
#undef DEFINE_TEXT_FLOAT
#undef TEXT_FLOAT_LANE

#define TEXT_SCALAR_LANE(ROW, ELEMENT, ...)                                                                            \
    do {                                                                                                               \
        const union cast_value value = load_value(binding, values, ELEMENT);                                           \
        char literal[TEMPORAL_TEXT_CAPACITY];                                                                          \
        size_t size = 0;                                                                                               \
        const int lane_status =                                                                                        \
            dora_type_format_scalar(type, physical, &value, false, literal, sizeof literal, &size);                    \
        if (lane_status != DORA_OK) {                                                                                  \
            CAST_FAILED(ROW, lane_status);                                                                             \
        } else {                                                                                                       \
            TEXT_PLACE(ROW, size, (memcpy)(text, literal, size));                                                      \
        }                                                                                                              \
    } while (0)

// Boolean, Date32, Date64, Time, Timestamp, Duration, intervals to Utf8, Binary, FixedBinary
static int cast_text_scalar(const struct kernel_packet* source,
                            const struct kernel_unary_binding* binding,
                            struct kernel_result* out_result) {
    TEXT_LOCALS;
    const enum dora_physical_type physical = binding->input_physical;
    const struct dora_logical_type* const type = binding->request.input_type;
    TEXT_SHAPES(TEXT_SCALAR_LANE)
    TEXT_FINISH;
}

#undef TEXT_SCALAR_LANE

// Null to Utf8, Binary, FixedBinary
static int cast_text_nulls(const struct kernel_packet* source,
                           const struct kernel_unary_binding*,
                           struct kernel_result* out_result) {
    const struct kernel_text_output* const output = out_result->values;
    if (output->end_offsets != nullptr) {
        for (uint32_t row = 0; row < source->count; ++row) {
            output->end_offsets[row] = output->written_byte_count;
        }
    }
    if (out_result->validity != nullptr) {
        bitmap_fill_bytes((uint8_t*)out_result->validity,
                          (size_t)bitmap_words_for_bits(source->count) * sizeof(uint64_t),
                          false);
    }
    out_result->null_count = source->count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

#undef TEXT_FINISH
#undef TEXT_PLACE
#undef TEXT_END
#undef TEXT_LOCALS
#undef TEXT_SHAPES
#undef TEXT_ROWS

//--------------------------------------
// Nulls
//--------------------------------------

// Scalar types to Null; Null to non-byte scalar types
static int
cast_nulls(const struct kernel_packet* source, const struct kernel_unary_binding*, struct kernel_result* out_result) {
    const uint64_t word_count = bitmap_words_for_bits(source->count);
    if (out_result->validity != nullptr) {
        bitmap_fill_bytes((uint8_t*)out_result->validity, (size_t)word_count * sizeof(uint64_t), false);
    }
    out_result->null_count = source->count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

#undef LEAF_FINISH
#undef CAST_FAILED
#undef CAST_SHAPES
#undef CAST_LANES
#undef MAP_ELEMENT
#undef CONSTANT_ELEMENT
#undef FLAT_ELEMENT
#undef LEAF_LOCALS

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

static constexpr uint32_t CAST_CARRIER_COUNT = DORA_PHYSICAL_TYPE_F64 + 1;

#define NUMERIC_ENTRY(S, KS, D, KD) [PHYSICAL_##S][PHYSICAL_##D] = cast_##S##_##D,
#define NUMERIC_ROW(S, KS) CAST_TARGETS(NUMERIC_ENTRY, S, KS)

static typeof(&cast_values) const numeric_leaves[CAST_CARRIER_COUNT][CAST_CARRIER_COUNT] = {CAST_SOURCES(NUMERIC_ROW)};

#define FLOAT_INTEGER_ENTRY(D, T, P, S)                                                                                \
    [DORA_ROUNDING_MODE_EXACT][PHYSICAL_##S - DORA_PHYSICAL_TYPE_F16]                                                  \
                              [P] = cast_exact_##S##_##D,                                                              \
       [DORA_ROUNDING_MODE_TOWARD_ZERO][PHYSICAL_##S - DORA_PHYSICAL_TYPE_F16][P] = cast_zero_##S##_##D,               \
       [DORA_ROUNDING_MODE_HALF_AWAY][PHYSICAL_##S - DORA_PHYSICAL_TYPE_F16][P] = cast_away_##S##_##D,                 \
       [DORA_ROUNDING_MODE_HALF_EVEN][PHYSICAL_##S - DORA_PHYSICAL_TYPE_F16][P] = cast_##S##_##D,

static typeof(&cast_values) const float_integer_leaves[4][3][CAST_CARRIER_COUNT] = {
    SIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f16) UNSIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f16)
        SIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f32) UNSIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f32)
            SIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f64) UNSIGNED_INDEX_CARRIERS(FLOAT_INTEGER_ENTRY, f64)
};

#undef FLOAT_INTEGER_ENTRY

#undef NUMERIC_ROW
#undef NUMERIC_ENTRY
#undef CAST_SOURCES
#undef CAST_TARGETS

#define UNCHECKED_ENTRY(S, D) [PHYSICAL_##S][PHYSICAL_##D] = cast_unchecked_##S##_##D,
#define UNCHECKED_ROW(S) UNCHECKED_TARGETS(UNCHECKED_ENTRY, S)

static typeof(&cast_values)
    const unchecked_leaves[CAST_CARRIER_COUNT][CAST_CARRIER_COUNT] = {UNCHECKED_SOURCES(UNCHECKED_ROW)};

#undef UNCHECKED_ROW
#undef UNCHECKED_ENTRY
#undef UNCHECKED_SOURCES
#undef UNCHECKED_TARGETS
#undef PHYSICAL_f64
#undef PHYSICAL_f32
#undef PHYSICAL_f16
#undef PHYSICAL_u128
#undef PHYSICAL_u64
#undef PHYSICAL_u32
#undef PHYSICAL_u16
#undef PHYSICAL_u8
#undef PHYSICAL_i256
#undef PHYSICAL_i128
#undef PHYSICAL_i64
#undef PHYSICAL_i32
#undef PHYSICAL_i16
#undef PHYSICAL_i8

static bool cast_is_numeric(enum dora_type type) {
    return type >= DORA_TYPE_INT8 && type <= DORA_TYPE_FLOAT64;
}

static typeof(&cast_values) const boolean_to_leaves[CAST_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = cast_boolean_i8,
    [DORA_PHYSICAL_TYPE_I16] = cast_boolean_i16,
    [DORA_PHYSICAL_TYPE_I32] = cast_boolean_i32,
    [DORA_PHYSICAL_TYPE_I64] = cast_boolean_i64,
    [DORA_PHYSICAL_TYPE_U8] = cast_boolean_u8,
    [DORA_PHYSICAL_TYPE_U16] = cast_boolean_u16,
    [DORA_PHYSICAL_TYPE_U32] = cast_boolean_u32,
    [DORA_PHYSICAL_TYPE_U64] = cast_boolean_u64,
    [DORA_PHYSICAL_TYPE_F32] = cast_boolean_f32,
    [DORA_PHYSICAL_TYPE_F64] = cast_boolean_f64,
};

static typeof(&cast_values) const boolean_from_leaves[CAST_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = cast_i8_boolean,
    [DORA_PHYSICAL_TYPE_I16] = cast_i16_boolean,
    [DORA_PHYSICAL_TYPE_I32] = cast_i32_boolean,
    [DORA_PHYSICAL_TYPE_I64] = cast_i64_boolean,
    [DORA_PHYSICAL_TYPE_U8] = cast_u8_boolean,
    [DORA_PHYSICAL_TYPE_U16] = cast_u16_boolean,
    [DORA_PHYSICAL_TYPE_U32] = cast_u32_boolean,
    [DORA_PHYSICAL_TYPE_U64] = cast_u64_boolean,
};

static int32_t rescale_carrier(enum dora_physical_type physical) {
    switch (physical) {
    case DORA_PHYSICAL_TYPE_I32:
        return 0;
    case DORA_PHYSICAL_TYPE_I64:
        return 1;
    case DORA_PHYSICAL_TYPE_I128:
        return 2;
    default:
        return -1;
    }
}

static struct dora_i128 rescale_power(int32_t exponent) {
    _BitInt(128) power = 1;
    for (int32_t step = 0; step < exponent; ++step) {
        power *= 10;
    }
    return type_i128_store(power);
}

#define RESCALE_ENTRY(D, T, P, S, SP)                                                                                  \
    [false][false][(SP) - DORA_PHYSICAL_TYPE_I32]                                                                      \
           [(P) - DORA_PHYSICAL_TYPE_I32] = rescale_down_unchecked_##S##_##D,                                          \
                  [false][true][(SP) - DORA_PHYSICAL_TYPE_I32][(P) - DORA_PHYSICAL_TYPE_I32] =                         \
                      rescale_down_checked_##S##_##D,                                                                  \
                  [true][false][(SP) - DORA_PHYSICAL_TYPE_I32][(P) - DORA_PHYSICAL_TYPE_I32] =                         \
                      rescale_up_unchecked_##S##_##D,                                                                  \
                  [true][true][(SP) - DORA_PHYSICAL_TYPE_I32][(P) - DORA_PHYSICAL_TYPE_I32] =                          \
                      rescale_up_checked_##S##_##D,
#define RESCALE_ROW(S, T, P) DECIMAL_RESCALE_CARRIERS(RESCALE_ENTRY, S, P)

static typeof(&cast_values)
    const rescale_leaves[2][2][3][3] = {RESCALE_ROW(i32, int32_t, DORA_PHYSICAL_TYPE_I32)
                                            RESCALE_ROW(i64, int64_t, DORA_PHYSICAL_TYPE_I64)
                                                RESCALE_ROW(i128, struct dora_i128, DORA_PHYSICAL_TYPE_I128)};

#undef RESCALE_ROW
#undef RESCALE_ENTRY

static bool cast_rescale_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* in_out_binding) {
    static constexpr int32_t carrier_digits[3] = {9, 18, 38};
    const int32_t source = rescale_carrier(in_out_binding->input_physical);
    const int32_t target = rescale_carrier(in_out_binding->result_physical);
    if (request->input_type->id != DORA_TYPE_DECIMAL || request->result_type->id != DORA_TYPE_DECIMAL || source < 0 ||
        target < 0) {
        return false;
    }
    const struct dora_decimal_type input = in_out_binding->input_decimal.type;
    const struct dora_decimal_type result = in_out_binding->result_decimal.type;
    const bool up = result.scale >= input.scale;
    const int64_t difference = up ? (int64_t)result.scale - input.scale : (int64_t)input.scale - result.scale;
    const int64_t target_width = up ? result.precision - difference : result.precision + difference;
    if (target_width < 0 || difference > carrier_digits[up ? target : source]) {
        return false;
    }
    const bool check = input.precision >= target_width;
    in_out_binding->cast.rescale = (struct kernel_decimal_rescale){
        .factor = rescale_power((int32_t)difference),
        .limit = check ? rescale_power((int32_t)target_width) : (struct dora_i128){},
    };
    in_out_binding->fill = rescale_leaves[up][check][source][target];
    if (!check && (up || request->rounding != DORA_ROUNDING_MODE_EXACT)) {
        in_out_binding->can_fail = false;
        in_out_binding->is_result_all_valid = true;
    }
    return true;
}

static typeof(&cast_values) const text_leaves[CAST_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = cast_text_i8,
    [DORA_PHYSICAL_TYPE_I16] = cast_text_i16,
    [DORA_PHYSICAL_TYPE_I32] = cast_text_i32,
    [DORA_PHYSICAL_TYPE_I64] = cast_text_i64,
    [DORA_PHYSICAL_TYPE_I128] = cast_text_integer_i128,
    [DORA_PHYSICAL_TYPE_I256] = cast_text_integer_i256,
    [DORA_PHYSICAL_TYPE_U8] = cast_text_u8,
    [DORA_PHYSICAL_TYPE_U16] = cast_text_u16,
    [DORA_PHYSICAL_TYPE_U32] = cast_text_u32,
    [DORA_PHYSICAL_TYPE_U64] = cast_text_u64,
    [DORA_PHYSICAL_TYPE_U128] = cast_text_integer_u128,
    [DORA_PHYSICAL_TYPE_F16] = cast_text_f16,
    [DORA_PHYSICAL_TYPE_F32] = cast_text_f32,
    [DORA_PHYSICAL_TYPE_F64] = cast_text_f64,
};

static typeof(&cast_values) const text_integer_leaves[CAST_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I32] = cast_text_integer_i32,
    [DORA_PHYSICAL_TYPE_I64] = cast_text_integer_i64,
    [DORA_PHYSICAL_TYPE_I128] = cast_text_integer_i128,
    [DORA_PHYSICAL_TYPE_I256] = cast_text_integer_i256,
};

static constexpr uint32_t text_digits[CAST_CARRIER_COUNT] = {
    [DORA_PHYSICAL_TYPE_I8] = 3,
    [DORA_PHYSICAL_TYPE_I16] = 5,
    [DORA_PHYSICAL_TYPE_I32] = 10,
    [DORA_PHYSICAL_TYPE_I64] = 19,
    [DORA_PHYSICAL_TYPE_I128] = 39,
    [DORA_PHYSICAL_TYPE_I256] = 77,
    [DORA_PHYSICAL_TYPE_U8] = 3,
    [DORA_PHYSICAL_TYPE_U16] = 5,
    [DORA_PHYSICAL_TYPE_U32] = 10,
    [DORA_PHYSICAL_TYPE_U64] = 20,
    [DORA_PHYSICAL_TYPE_U128] = 39,
};

static int cast_text_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* in_out_binding) {
    const struct dora_logical_type* const source = request->input_type;
    const enum dora_physical_type physical = in_out_binding->input_physical;
    if (type_is_bytes(source->id)) {
        in_out_binding->input_byte_width = sizeof(struct dora_bytes);
        in_out_binding->fill = cast_text_bytes;
        return DORA_OK;
    }
    if (source->id == DORA_TYPE_BOOLEAN) {
        in_out_binding->fill = cast_text_scalar;
        in_out_binding->cast.maximum_text_byte_count = sizeof("false") - 1;
        return DORA_OK;
    }
    if (type_is_temporal(source->id) || source->id >= DORA_TYPE_INTERVAL_YEAR_MONTH) {
        in_out_binding->fill = cast_text_scalar;
        in_out_binding->cast.maximum_text_byte_count = TEMPORAL_TEXT_CAPACITY - 1;
        return DORA_OK;
    }
    if (physical >= CAST_CARRIER_COUNT || text_leaves[physical] == nullptr) {
        return ENOTSUP;
    }
    in_out_binding->fill = source->id == DORA_TYPE_DECIMAL ? text_integer_leaves[physical] : text_leaves[physical];
    if (cast_is_float(physical)) {
        in_out_binding->cast.maximum_text_byte_count = TYPE_FLOAT_TEXT_CAPACITY - 1;
        return DORA_OK;
    }
    const int64_t scale = source->id == DORA_TYPE_DECIMAL ? source->parameters.decimal.scale : 0;
    const int64_t digits = text_digits[physical];
    const int64_t magnitude = scale < 0         ? digits - scale
                              : scale == 0      ? digits
                              : scale >= digits ? scale + 2
                                                : digits + 1;
    in_out_binding->cast.maximum_text_byte_count = (size_t)magnitude + 1;
    return DORA_OK;
}

static int cast_select(const struct kernel_bind_request* request, struct kernel_unary_binding* in_out_binding) {
    const struct dora_logical_type* const source = request->input_type;
    const struct dora_logical_type* const target = request->result_type;
    if (source->id == DORA_TYPE_NULL || target->id == DORA_TYPE_NULL) {
        in_out_binding->fill = type_is_bytes(target->id) ? cast_text_nulls : cast_nulls;
        return DORA_OK;
    }
    if (type_is_bytes(target->id)) {
        in_out_binding->result_byte_width = target->id == DORA_TYPE_FIXED_BINARY ? target->parameters.byte_width : 0;
        return cast_text_bind(request, in_out_binding);
    }
    if (type_is_bytes(source->id)) {
        in_out_binding->fill = cast_parse;
        in_out_binding->cast.parse = dora_type_parse_select(target, in_out_binding->result_physical);
        return in_out_binding->cast.parse == nullptr ? ENOTSUP : DORA_OK;
    }
    if (source->id >= DORA_TYPE_INTERVAL_YEAR_MONTH || target->id >= DORA_TYPE_INTERVAL_YEAR_MONTH) {
        in_out_binding->fill = cast_values;
        return source->id == target->id ? DORA_OK : ENOTSUP;
    }
    const bool source_float = source->id >= DORA_TYPE_FLOAT16 && source->id <= DORA_TYPE_FLOAT64;
    const bool target_float = target->id >= DORA_TYPE_FLOAT16 && target->id <= DORA_TYPE_FLOAT64;
    if ((type_is_temporal(target->id) &&
         (source_float || source->id == DORA_TYPE_DECIMAL || source->id == DORA_TYPE_BOOLEAN)) ||
        (type_is_temporal(source->id) &&
         (target_float || target->id == DORA_TYPE_BOOLEAN || target->id == DORA_TYPE_DECIMAL))) {
        return ENOTSUP;
    }
    if (type_is_temporal(source->id) && type_is_temporal(target->id)) {
        const int status = cast_temporal_admit(source, target);
        if (status != DORA_OK) {
            return status;
        }
        if (source->id == DORA_TYPE_TIMESTAMP && target->id == DORA_TYPE_TIMESTAMP &&
            target->parameters.timestamp.unit > source->parameters.timestamp.unit) {
            in_out_binding->cast.rescale.factor =
                rescale_power(3 * (int32_t)(target->parameters.timestamp.unit - source->parameters.timestamp.unit));
            in_out_binding->fill = cast_timestamp_scale;
            return DORA_OK;
        }
        if (source->id == DORA_TYPE_TIMESTAMP && source->parameters.timestamp.unit == DORA_TIME_UNIT_MILLISECOND &&
            target->id == DORA_TYPE_DATE32) {
            in_out_binding->fill = cast_millis_date;
            return DORA_OK;
        }
    }
    if (source->id == DORA_TYPE_DECIMAL && source->parameters.decimal.scale == 0 &&
        source->parameters.decimal.precision <= DECIMAL64_MAX_PRECISION &&
        in_out_binding->input_physical == DORA_PHYSICAL_TYPE_I128 &&
        (target->id == DORA_TYPE_FLOAT32 || target->id == DORA_TYPE_FLOAT64)) {
        in_out_binding->fill = target->id == DORA_TYPE_FLOAT32 ? cast_decimal_integer_f32 : cast_decimal_integer_f64;
        return DORA_OK;
    }
    if (cast_rescale_bind(request, in_out_binding)) {
        return DORA_OK;
    }
    if (source->id == DORA_TYPE_BOOLEAN && cast_is_numeric(target->id) &&
        in_out_binding->result_physical < CAST_CARRIER_COUNT &&
        boolean_to_leaves[in_out_binding->result_physical] != nullptr) {
        in_out_binding->fill = boolean_to_leaves[in_out_binding->result_physical];
        return DORA_OK;
    }
    if (target->id == DORA_TYPE_BOOLEAN && cast_is_numeric(source->id) &&
        in_out_binding->input_physical < CAST_CARRIER_COUNT &&
        boolean_from_leaves[in_out_binding->input_physical] != nullptr) {
        in_out_binding->fill = boolean_from_leaves[in_out_binding->input_physical];
        return DORA_OK;
    }
    const bool numeric = cast_is_numeric(source->id) && cast_is_numeric(target->id);
    if (numeric && source_float && type_is_native_integer(target->id)) {
        in_out_binding->fill =
            float_integer_leaves[request->rounding][in_out_binding->input_physical - DORA_PHYSICAL_TYPE_F16]
                                [in_out_binding->result_physical];
    } else {
        in_out_binding->fill =
            numeric ? numeric_leaves[in_out_binding->input_physical][in_out_binding->result_physical] : cast_values;
    }
    return in_out_binding->fill == nullptr ? EINVAL : DORA_OK;
}

static int cast_prepare(const struct kernel_bind_request* request, struct kernel_unary_binding* in_out_binding) {
    const struct dora_logical_type* const source = request->input_type;
    const struct dora_logical_type* const target = request->result_type;
    int status = DORA_OK;
    if (source->id == DORA_TYPE_DECIMAL) {
        status = dora_decimal_prepare(source->parameters.decimal,
                                      in_out_binding->input_physical,
                                      &in_out_binding->input_decimal);
    }
    if (status == DORA_OK && target->id == DORA_TYPE_DECIMAL) {
        status = dora_decimal_prepare(target->parameters.decimal,
                                      in_out_binding->result_physical,
                                      &in_out_binding->result_decimal);
    }
    const bool temporal = type_is_temporal(source->id) && type_is_temporal(target->id);
    in_out_binding->cast.scale = (struct decimal_scale_pair){
        .source = temporal ? cast_tick_scale(source) : in_out_binding->input_decimal.type.scale,
        .target = temporal ? cast_tick_scale(target) : in_out_binding->result_decimal.type.scale,
    };
    if (source->id == DORA_TYPE_DATE32 || source->id == DORA_TYPE_DATE64) {
        in_out_binding->cast.scale.source = 0;
    }
    if (target->id == DORA_TYPE_TIME) {
        in_out_binding->cast.ticks_per_day = cast_day_ticks(target);
    } else if (temporal && (target->id == DORA_TYPE_DATE32 || target->id == DORA_TYPE_DATE64) &&
               source->id != DORA_TYPE_DATE32 && source->id != DORA_TYPE_DATE64) {
        in_out_binding->cast.ticks_per_day = cast_day_ticks(source);
    }
    return status;
}

int dora_kernel_cast_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->input_type == nullptr ||
        request->result_type == nullptr || request->policy > DORA_FAILURE_POLICY_WRAP ||
        request->rounding > DORA_ROUNDING_MODE_HALF_AWAY) {
        return EINVAL;
    }
    struct kernel_unary_binding binding = {
        .request = *request,
        .result_type = *request->result_type,
        .is_result_all_valid = request->policy != DORA_FAILURE_POLICY_NULL,
        .can_fail = true,
    };
    int status = type_scalar_physical(request->input_type, &binding.input_physical);
    if (status == DORA_OK) {
        status = type_scalar_physical(request->result_type, &binding.result_physical);
    }
    if (status != DORA_OK) {
        return status;
    }
    binding.input_byte_width = physical_width(binding.input_physical);
    binding.result_byte_width = physical_width(binding.result_physical);
    if (request->policy == DORA_FAILURE_POLICY_WRAP) {
        if (!type_is_native_integer(request->input_type->id) || !type_is_native_integer(request->result_type->id) ||
            binding.input_physical >= CAST_CARRIER_COUNT || binding.result_physical >= CAST_CARRIER_COUNT) {
            return EINVAL;
        }
        binding.fill = unchecked_leaves[binding.input_physical][binding.result_physical];
        binding.can_fail = false;
        if (binding.fill == nullptr) {
            return EINVAL;
        }
        *out_binding = binding;
        return DORA_OK;
    }
    status = cast_prepare(request, &binding);
    if (status == DORA_OK) {
        status = cast_select(request, &binding);
    }
    if (status == DORA_OK) {
        *out_binding = binding;
    }
    return status;
}

int dora_kernel_cast_measure_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->input_type == nullptr ||
        request->result_type == nullptr || !type_is_bytes(request->input_type->id) ||
        !type_is_bytes(request->result_type->id)) {
        return EINVAL;
    }
    struct kernel_unary_binding binding = {};
    const int status = dora_kernel_cast_bind(request, &binding);
    if (status != DORA_OK) {
        return status;
    }
    binding.fill = cast_measure;
    binding.result_type = (struct dora_logical_type){.id = DORA_TYPE_INT64};
    binding.result_physical = DORA_PHYSICAL_TYPE_I64;
    binding.result_byte_width = sizeof(int64_t);
    *out_binding = binding;
    return DORA_OK;
}
