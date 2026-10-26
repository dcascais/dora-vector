// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Decode and copy bindings.

#ifndef DORA_KERNEL_BINDING_READ_H
#define DORA_KERNEL_BINDING_READ_H

#include "dora/vector/types.h"
#include "dora/vector/view.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Requests
//
//------------------------------------------------------------------------------

enum kernel_decode_kind : uint8_t {
    KERNEL_DECODE_DICTIONARY,
    KERNEL_DECODE_RUN_END,
    KERNEL_DECODE_SELECTION
};

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

struct kernel_decode_binding {
    void (*resolve)(struct kernel_decode_binding* in_out_binding,
                    uint64_t first_row,
                    uint32_t count,
                    uint64_t* in_out_rows,
                    uint64_t* in_out_validity);
    void (*narrow)(const struct kernel_decode_binding* binding,
                   uint64_t first_row,
                   uint32_t count,
                   uint64_t* in_out_validity,
                   uint32_t* out_map);
    const struct dora_vector_view* view;
    uint64_t run_cursor;
    uint64_t run_end_cache;
    enum kernel_decode_kind kind;
    enum dora_physical_type index_physical;
};

struct kernel_gather_binding {
    void (*gather)(const struct kernel_gather_binding* binding,
                   const uint64_t* rows,
                   void* out_values,
                   const uint64_t* validity,
                   uint32_t count);
    const struct dora_vector_view* view;
    uint32_t width;
    bool can_use_u32_map;
};

struct kernel_bytes_binding {
    int (*fill)(const struct kernel_bytes_binding* binding,
                uint64_t first_row,
                const uint64_t* rows,
                const uint64_t* validity,
                uint32_t count,
                struct dora_bytes* out_values);
    const struct dora_vector_view* view;
};

//------------------------------------------------------------------------------
//
// Bind Functions
//
//------------------------------------------------------------------------------

// All logical types; layouts: Dictionary, RunEnd; selection: UInt32/UInt64 indices in any layout
int dora_kernel_decode_bind(const struct dora_vector_view* view, struct kernel_decode_binding* out_binding);

// Fixed widths outside 1/2/4/8/16/32 bytes leave gather nullptr.
// Layouts: Bits, Fixed
int dora_kernel_gather_bind(const struct dora_vector_view* view, struct kernel_gather_binding* out_binding);

// Utf8, Binary, FixedBinary; layouts: Binary, LargeBinary, BinaryView, Fixed
int dora_kernel_bytes_bind(const struct dora_vector_view* view, struct kernel_bytes_binding* out_binding);

#endif
