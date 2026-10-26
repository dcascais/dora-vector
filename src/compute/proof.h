// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Overflow proofs from exact statistics.

#ifndef DORA_COMPUTE_PROOF_H
#define DORA_COMPUTE_PROOF_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Element-Wise
//
//------------------------------------------------------------------------------

//--------------------------------------
// Arithmetic
//--------------------------------------

// Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128
bool dora_compute_arithmetic_is_proven(enum kernel_operation operation,
                                       const struct dora_logical_type* type,
                                       const struct dora_vector* left,
                                       const struct dora_vector* right);

//--------------------------------------
// Comparison and Conversion
//--------------------------------------

// Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64; source and target
bool dora_compute_cast_is_proven(const struct dora_vector* source,
                                 const struct dora_logical_type* type,
                                 const struct dora_logical_type* target);

//------------------------------------------------------------------------------
//
// Aggregates
//
//------------------------------------------------------------------------------

//--------------------------------------
// Numeric
//--------------------------------------

// Int32, Int64
bool dora_compute_sum_is_proven(const struct dora_vector* source,
                                const struct dora_logical_type* type,
                                uint64_t row_count);

#endif
