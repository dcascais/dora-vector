// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Boolean selection output.

#ifndef DORA_KERNEL_SELECTION_H
#define DORA_KERNEL_SELECTION_H

#include "kernel/packet.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

struct kernel_select_output {
    void* out_true;
    void* out_false;
    void* out_null;
    uint64_t true_count;
    uint64_t false_count;
    uint64_t null_count;
    bool is_wide;
};

//------------------------------------------------------------------------------
//
// Carrier-Independent Leaves
//
//------------------------------------------------------------------------------

// Tail bits beyond mask->count are unspecified.
void dora_kernel_select_mask_words(const struct kernel_packet* mask, uint64_t* out_matches);

uint32_t dora_kernel_select_padding(bool is_wide);

void dora_kernel_select_emit(const uint64_t* matches,
                             const uint64_t* validity,
                             uint32_t count,
                             uint64_t logical_first,
                             struct kernel_select_output* in_out_output);

#endif
