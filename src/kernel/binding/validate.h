// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector validation bindings.

#ifndef DORA_KERNEL_BINDING_VALIDATE_H
#define DORA_KERNEL_BINDING_VALIDATE_H

#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "numeric/decimal.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

enum : uint32_t {
    KERNEL_UNION_TYPE_ID_COUNT = 128
};

struct kernel_validate_binding {
    int (*check)(struct kernel_validate_binding* in_out_binding,
                 uint64_t first_row,
                 uint32_t count,
                 uint64_t* out_first_bad_row);
    const struct dora_vector_view* view;
    const struct dora_vector_view* related;
    struct decimal_prepared decimal;
    int64_t limit;
    uint64_t union_row;
    // Caller supplies KERNEL_UNION_TYPE_ID_COUNT zeroed entries for dense unions.
    int32_t* union_offsets;
};

//------------------------------------------------------------------------------
//
// Bind Functions
//
//------------------------------------------------------------------------------

// related holds dictionary values, a list/map child, run ends or union members; otherwise nullptr.
// Utf8, Binary, List, Map, Union, Decimal; all logical types in Dictionary or RunEnd layouts
int dora_kernel_validate_bind(const struct dora_vector_view* view,
                              const struct dora_logical_type* type,
                              const struct dora_vector_view* related,
                              struct kernel_validate_binding* out_binding);

// Utf8, Binary: BinaryView layout; Date64, Time: Fixed layout
int dora_kernel_validate_content_bind(const struct dora_vector_view* view,
                                      const struct dora_logical_type* type,
                                      struct kernel_validate_binding* out_binding);

static inline bool kernel_validate_has_content(const struct dora_vector_view* view,
                                               const struct dora_logical_type* type) {
    return view->layout == DORA_LAYOUT_BINARY_VIEW || type->id == DORA_TYPE_DATE64 || type->id == DORA_TYPE_TIME;
}

#endif
