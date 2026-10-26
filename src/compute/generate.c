// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/generate.h"

#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/packet.h"
#include "types/types_internal.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Generation
//
//------------------------------------------------------------------------------

// Int8, Int16, Int32, Int64
int dora_compute_sequence(struct dora_pool pool,
                          const struct dora_logical_type* type,
                          int64_t start,
                          int64_t step,
                          uint64_t row_count,
                          struct dora_vector* out_result) {
    if (type == nullptr || !dora_vector_is_empty(out_result)) {
        return EINVAL;
    }
    int64_t minimum = 0;
    int64_t maximum = 0;
    switch (type->id) {
    case DORA_TYPE_INT8:
        minimum = INT8_MIN;
        maximum = INT8_MAX;
        break;
    case DORA_TYPE_INT16:
        minimum = INT16_MIN;
        maximum = INT16_MAX;
        break;
    case DORA_TYPE_INT32:
        minimum = INT32_MIN;
        maximum = INT32_MAX;
        break;
    case DORA_TYPE_INT64:
        minimum = INT64_MIN;
        maximum = INT64_MAX;
        break;
    default:
        return EINVAL;
    }
    if (type->extension_name.byte_count != 0) {
        return ENOTSUP;
    }
    if (start < minimum || start > maximum) {
        return ERANGE;
    }
    if (row_count > 1) {
        const uint64_t distance = step < 0 ? (uint64_t)start - minimum : (uint64_t)maximum - start;
        uint64_t span = 0;
        if (ckd_mul(&span, step < 0 ? UINT64_C(0) - step : (uint64_t)step, row_count - 1) || span > distance) {
            return ERANGE;
        }
    }
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    int status = type_physical(type, &physical);
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, type, row_count, false, true, &output);
    }
    for (uint64_t first = 0; status == DORA_OK && first < row_count; first += KERNEL_CHUNK_ROW_COUNT) {
        const uint32_t count = vector_operand_chunk(first, row_count).length;
        const struct kernel_result result = dora_compute_output_result(&output, first, count);
        status = dora_kernel_sequence_fill(physical, start, step, first, count, result.values);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, out_result);
    }
    dora_compute_output_release(&output);
    return status;
}
