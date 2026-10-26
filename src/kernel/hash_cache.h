// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Packet hash caching.

#ifndef DORA_KERNEL_HASH_CACHE_H
#define DORA_KERNEL_HASH_CACHE_H

#include "kernel/packet.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Bindings
//
//------------------------------------------------------------------------------

struct kernel_hash_cache {
    const uint64_t* base_rows;
    uint64_t* entries;
    uint32_t base_row_count;
};

void dora_kernel_hash_bytes_cached(const struct kernel_packet* source,
                                   const struct kernel_hash_cache* cache,
                                   uint64_t seed,
                                   const struct kernel_packet* prior,
                                   uint64_t* out_hashes);

#endif
