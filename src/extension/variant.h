// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Variant metadata and values.

#ifndef DORA_EXTENSION_VARIANT_H
#define DORA_EXTENSION_VARIANT_H

#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

struct variant_frame {
    uint64_t begin_byte_offset;
    uint64_t end_byte_offset;
    uint64_t next_child_index;
};

int dora_variant_metadata_check(struct dora_bytes metadata);
int dora_variant_value_check(struct dora_bytes metadata,
                             struct dora_bytes value,
                             void* scratch,
                             size_t scratch_capacity_bytes,
                             const struct dora_i256 decimal_bounds[3]);
// Requires validated metadata and object value, and a well-formed key.
int dora_variant_object_lookup(struct dora_bytes metadata,
                               const struct dora_bytes* value,
                               struct dora_bytes key,
                               bool* out_found);

#endif
