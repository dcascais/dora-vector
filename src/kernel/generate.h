// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Integer sequence filling.

#ifndef DORA_KERNEL_GENERATE_H
#define DORA_KERNEL_GENERATE_H

#include "dora/vector/types.h"

#include <stdint.h>

// Int8, Int16, Int32, Int64
int dora_kernel_sequence_fill(enum dora_physical_type physical,
                              int64_t start,
                              int64_t step,
                              uint64_t first_index,
                              uint32_t count,
                              void* out_values);

#endif
