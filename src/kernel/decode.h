// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Validity gathering and run ends.

#ifndef DORA_KERNEL_DECODE_H
#define DORA_KERNEL_DECODE_H

#include "dora/vector/view.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Carrier-Independent Leaves
//
//------------------------------------------------------------------------------

void dora_kernel_gather_validity(const struct dora_vector_view* view,
                                 const uint64_t* rows,
                                 const uint64_t* active,
                                 uint32_t count,
                                 uint64_t* out_words);

void dora_kernel_gather_map_validity(const struct dora_vector_view* view,
                                     const uint32_t* map,
                                     const uint64_t* active,
                                     uint32_t count,
                                     uint64_t* out_words);

void dora_kernel_load_validity(const struct dora_vector_view* view,
                               uint64_t first_row,
                               uint32_t count,
                               uint64_t* out_words);

void dora_kernel_narrow_rows(const uint64_t* rows,
                             uint64_t base,
                             uint32_t* restrict out_map,
                             const uint64_t* validity,
                             uint32_t count);

// Invalid lanes receive null_value without reading values or map.
void dora_kernel_gather_u64(const uint64_t* values,
                            const uint32_t* map,
                            const uint64_t* validity,
                            uint32_t count,
                            uint64_t* out_values,
                            uint64_t null_value);

// Requires nonempty ends.
// Run ends: Int16, Int32, Int64
int dora_kernel_run_end_last(const struct dora_vector_view* ends, int64_t* out_end);

#endif
