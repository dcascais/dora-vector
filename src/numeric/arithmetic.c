// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "numeric/arithmetic.h"

#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Integer Arithmetic
//
//------------------------------------------------------------------------------

int dora_numeric_store_integer(struct dora_i256 value, enum dora_physical_type physical, void* out_value) {
    if (out_value == nullptr) {
        return EINVAL;
    }
    const uint64_t fill = (value.limbs[0] >> 63U) != 0 ? UINT64_MAX : 0;
    const bool signed64 = value.limbs[1] == fill && value.limbs[2] == fill && value.limbs[3] == fill;
    const bool unsigned64 = value.limbs[1] == 0 && value.limbs[2] == 0 && value.limbs[3] == 0;
    int64_t signed_value = 0;
    memcpy(&signed_value, &value.limbs[0], sizeof(signed_value));
    switch (physical) {
// Switch clause; returns DORA_OK or ERANGE from the enclosing function.
#define PARSE_SIGNED(TAG, TYPE, MINIMUM, MAXIMUM)                                                                      \
    case TAG: {                                                                                                        \
        if (!signed64 || signed_value < (MINIMUM) || signed_value > (MAXIMUM)) {                                       \
            return ERANGE;                                                                                             \
        }                                                                                                              \
        const TYPE result = (TYPE)signed_value;                                                                        \
        memcpy(out_value, &result, sizeof(result));                                                                    \
        return DORA_OK;                                                                                                \
    }
        PARSE_SIGNED(DORA_PHYSICAL_TYPE_I8, int8_t, INT8_MIN, INT8_MAX)
        PARSE_SIGNED(DORA_PHYSICAL_TYPE_I16, int16_t, INT16_MIN, INT16_MAX)
        PARSE_SIGNED(DORA_PHYSICAL_TYPE_I32, int32_t, INT32_MIN, INT32_MAX)
        PARSE_SIGNED(DORA_PHYSICAL_TYPE_I64, int64_t, INT64_MIN, INT64_MAX)
#undef PARSE_SIGNED
// Switch clause; returns DORA_OK or ERANGE from the enclosing function.
#define PARSE_UNSIGNED(TAG, TYPE, MAXIMUM)                                                                             \
    case TAG: {                                                                                                        \
        if (!unsigned64 || value.limbs[0] > (MAXIMUM)) {                                                               \
            return ERANGE;                                                                                             \
        }                                                                                                              \
        const TYPE result = (TYPE)value.limbs[0];                                                                      \
        memcpy(out_value, &result, sizeof(result));                                                                    \
        return DORA_OK;                                                                                                \
    }
        PARSE_UNSIGNED(DORA_PHYSICAL_TYPE_U8, uint8_t, UINT8_MAX)
        PARSE_UNSIGNED(DORA_PHYSICAL_TYPE_U16, uint16_t, UINT16_MAX)
        PARSE_UNSIGNED(DORA_PHYSICAL_TYPE_U32, uint32_t, UINT32_MAX)
        PARSE_UNSIGNED(DORA_PHYSICAL_TYPE_U64, uint64_t, UINT64_MAX)
#undef PARSE_UNSIGNED
    case DORA_PHYSICAL_TYPE_I128: {
        const uint64_t upper = (value.limbs[1] >> 63U) != 0 ? UINT64_MAX : 0;
        if (value.limbs[2] != upper || value.limbs[3] != upper) {
            return ERANGE;
        }
        const struct dora_i128 result = {.limbs = {value.limbs[0], value.limbs[1]}};
        memcpy(out_value, &result, sizeof(result));
        return DORA_OK;
    }
    case DORA_PHYSICAL_TYPE_U128: {
        if (value.limbs[2] != 0 || value.limbs[3] != 0) {
            return ERANGE;
        }
        const struct dora_u128 result = {.limbs = {value.limbs[0], value.limbs[1]}};
        memcpy(out_value, &result, sizeof(result));
        return DORA_OK;
    }
    case DORA_PHYSICAL_TYPE_I256:
        memcpy(out_value, &value, sizeof(value));
        return DORA_OK;
    default:
        return EINVAL;
    }
}

int64_t dora_numeric_floor_div(int64_t value, int64_t divisor) {
    int64_t quotient = value / divisor;
    return quotient - (value % divisor < 0);
}
