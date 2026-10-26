// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/generate.h"

#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Sequences
//
//------------------------------------------------------------------------------

struct sequence_span {
    uint64_t begin;
    uint64_t step;
    uint32_t count;
    enum dora_physical_type physical;
};

static inline int64_t sequence_value(uint64_t bits) {
    int64_t value = 0;
    (memcpy)(&value, &bits, sizeof(value));
    return value;
}

int dora_kernel_sequence_fill(enum dora_physical_type physical,
                              int64_t start,
                              int64_t step,
                              uint64_t first_index,
                              uint32_t count,
                              void* out_values) {
    const struct sequence_span span = {
        .begin = (uint64_t)start + (uint64_t)step * first_index,
        .step = (uint64_t)step,
        .count = count,
        .physical = physical,
    };
    if (out_values == nullptr && span.count != 0) {
        return EINVAL;
    }
    switch (span.physical) {
// Switch clause; returns DORA_OK from the enclosing function.
#define SEQUENCE_FILL_CLOSED(P, T)                                                                                     \
    case P: {                                                                                                          \
        T* const out = out_values;                                                                                     \
        for (uint32_t lane = 0; lane < span.count; ++lane) {                                                           \
            out[lane] = (T)sequence_value(span.begin + span.step * lane);                                              \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
// Switch clause; returns DORA_OK from the enclosing function.
#define SEQUENCE_FILL_RUNNING(P, T)                                                                                    \
    case P: {                                                                                                          \
        T* const out = out_values;                                                                                     \
        T value = (T)sequence_value(span.begin);                                                                       \
        for (uint32_t lane = 0; lane < span.count; ++lane) {                                                           \
            if (lane > 0) {                                                                                            \
                value = (T)sequence_value((uint64_t)(int64_t)value + span.step);                                       \
            }                                                                                                          \
            out[lane] = value;                                                                                         \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
        SEQUENCE_FILL_CLOSED(DORA_PHYSICAL_TYPE_I8, int8_t)
        SEQUENCE_FILL_CLOSED(DORA_PHYSICAL_TYPE_I16, int16_t)
        SEQUENCE_FILL_RUNNING(DORA_PHYSICAL_TYPE_I32, int32_t)
        SEQUENCE_FILL_CLOSED(DORA_PHYSICAL_TYPE_I64, int64_t)
#undef SEQUENCE_FILL_RUNNING
#undef SEQUENCE_FILL_CLOSED
    default:
        return EINVAL;
    }
}
