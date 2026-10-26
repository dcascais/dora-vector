// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
//
// Backends
//
//------------------------------------------------------------------------------

//--------------------------------------
// Host
//--------------------------------------

static int host_allocate(struct dora_allocator*, size_t capacity_bytes, size_t alignment_bytes, void** out_address) {
    void* const address = aligned_alloc(alignment_bytes, capacity_bytes);
    if (address == nullptr) {
        return ENOMEM;
    }
    *out_address = address;
    return DORA_OK;
}

static void host_deallocate(struct dora_allocator*, void* address, size_t, size_t) {
    free(address);
}

static struct dora_allocator host_allocator = {.allocate = host_allocate, .deallocate = host_deallocate};

struct dora_allocator* dora_memory_host_allocator() {
    return &host_allocator;
}

//--------------------------------------
// Fixed Region
//--------------------------------------

static int
region_allocate(struct dora_allocator* allocator, size_t capacity_bytes, size_t alignment_bytes, void** out_address) {
    struct fixed_region_allocator* const region = (struct fixed_region_allocator*)allocator;
    const uint64_t block_count = capacity_bytes / alignment_bytes;
    uint64_t first_block = 0;
    if (!dora_bitmap_find_clear_run(region->bitmap,
                                    region->search_block_index,
                                    region->block_count,
                                    block_count,
                                    &first_block) &&
        !dora_bitmap_find_clear_run(region->bitmap, 0, region->block_count, block_count, &first_block)) {
        return ENOMEM;
    }
    dora_bitmap_set_range(region->bitmap, first_block, first_block + block_count, true);
    region->search_block_index = (size_t)(first_block + block_count);
    if (region->search_block_index == region->block_count) {
        region->search_block_index = 0;
    }
    *out_address = (unsigned char*)region + region->payload_byte_offset + (size_t)first_block * alignment_bytes;
    return DORA_OK;
}

static void
region_deallocate(struct dora_allocator* allocator, void* address, size_t capacity_bytes, size_t alignment_bytes) {
    struct fixed_region_allocator* const region = (struct fixed_region_allocator*)allocator;
    const size_t first_block =
        ((size_t)((unsigned char*)address - (unsigned char*)region) - region->payload_byte_offset) / alignment_bytes;
    dora_bitmap_set_range(region->bitmap, first_block, first_block + capacity_bytes / alignment_bytes, false);
    if (first_block < region->search_block_index) {
        region->search_block_index = first_block;
    }
}

int dora_memory_fixed_region_init(void* region,
                                  size_t region_byte_count,
                                  size_t alignment_bytes,
                                  struct dora_allocator** out_allocator,
                                  size_t* out_limit_bytes) {
    if (region == nullptr || out_allocator == nullptr || out_limit_bytes == nullptr ||
        !memory_alignment_is_valid(alignment_bytes)) {
        return EINVAL;
    }
    const uintptr_t region_address = (uintptr_t)region;
    uintptr_t region_end;
    uintptr_t header_address;
    if (region_byte_count < alignment_bytes || ckd_add(&region_end, region_address, region_byte_count) ||
        ckd_add(&header_address, region_address, alignof(struct fixed_region_allocator) - 1)) {
        return ENOMEM;
    }
    header_address &= ~(uintptr_t)(alignof(struct fixed_region_allocator) - 1);
    const size_t head_byte_count = offsetof(struct fixed_region_allocator, bitmap);
    if (header_address > region_end || head_byte_count > region_end - header_address) {
        return ENOMEM;
    }
    const uint64_t upper_block_count = (region_end - header_address - head_byte_count) / alignment_bytes;
    const uint64_t bitmap_word_count = bitmap_words_for_bits(upper_block_count);
    size_t bitmap_byte_count;
    uintptr_t bitmap_end;
    uintptr_t payload_address;
    if (ckd_mul(&bitmap_byte_count, bitmap_word_count, sizeof(uint64_t)) ||
        ckd_add(&bitmap_end, header_address, head_byte_count) || ckd_add(&bitmap_end, bitmap_end, bitmap_byte_count) ||
        ckd_add(&payload_address, bitmap_end, alignment_bytes - 1)) {
        return ENOMEM;
    }
    payload_address &= ~(uintptr_t)(alignment_bytes - 1);
    if (payload_address > region_end) {
        return ENOMEM;
    }
    const size_t block_count = (region_end - payload_address) / alignment_bytes;
    if (block_count == 0) {
        return ENOMEM;
    }
    if (block_count > DORA_POOL_FIXED_REGION_MAX_BLOCKS) {
        return ERANGE;
    }
    struct fixed_region_allocator* const allocator =
        (struct fixed_region_allocator*)((unsigned char*)region + (header_address - region_address));
    allocator->base = (struct dora_allocator){.allocate = region_allocate, .deallocate = region_deallocate};
    allocator->block_count = block_count;
    allocator->payload_byte_offset = payload_address - header_address;
    allocator->search_block_index = 0;
    dora_bitmap_set_range(allocator->bitmap, 0, block_count, false);
    *out_allocator = &allocator->base;
    *out_limit_bytes = block_count * alignment_bytes;
    return DORA_OK;
}
