// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "statistics/bounds.h"

#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "vector/vector_internal.h"

#include <stdint.h>

bool dora_statistics_try_get_exact_bounds(const struct dora_vector* vector,
                                          const struct dora_logical_type* type,
                                          struct statistics_bounds* out_bounds) {
    struct dora_vector statistics = {};
    struct dora_vector group = {};
    if (dora_vector_get_computed_statistics(vector, &statistics) != DORA_OK ||
        dora_vector_get_child(&statistics, 0, &group) != DORA_OK) {
        return false;
    }
    static const char* const names[2][2] = {{"min", "is_min_exact"}, {"max", "is_max_exact"}};
    union decoded_row_value* const bounds[2] = {&out_bounds->minimum, &out_bounds->maximum};
    for (uint32_t side = 0; side < 2; ++side) {
        struct dora_vector bound = {};
        struct dora_vector exact = {};
        struct decoded_row value = {};
        struct decoded_row flag = {};
        if (dora_vector_find_child(&group, names[side][0], &bound) != DORA_OK ||
            dora_vector_find_child(&group, names[side][1], &exact) != DORA_OK ||
            dora_vector_decode_row(&bound, 0, &value) != DORA_OK ||
            dora_vector_decode_row(&exact, 0, &flag) != DORA_OK || value.is_null || value.type != type->id ||
            flag.is_null || !flag.value.boolean) {
            return false;
        }
        *bounds[side] = value.value;
    }
    return true;
}
