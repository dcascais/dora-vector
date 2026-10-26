// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Packed bitmap access.

#ifndef DORA_VECTOR_BITMAP_H
#define DORA_VECTOR_BITMAP_H

#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Inspection
//
//------------------------------------------------------------------------------

static inline bool dora_bitmap_test(const uint8_t* bitmap, uint64_t bit) {
    return (((uint32_t)bitmap[bit >> 3U] >> (bit & 7U)) & 1U) != 0;
}

/// Requires bit_count <= 64 and ceil((bit_offset + bit_count) / 8) readable bytes.
static inline uint64_t dora_bitmap_load_window(const uint8_t* bitmap, uint64_t bit_offset, uint32_t bit_count) {
    const uint8_t* bytes = bitmap + (bit_offset >> 3U);
    const uint32_t shift = bit_offset & 7U;
    const uint32_t byte_count = ((bit_offset & 7U) + bit_count + 7U) >> 3U;
    uint64_t word = 0;
    if (byte_count >= 8) {
        (memcpy)(&word, bytes, 8);
    } else {
        for (uint32_t byte = 0; byte < byte_count; ++byte) {
            word |= (uint64_t)bytes[byte] << (byte * 8U);
        }
    }
    word >>= shift;
    if (shift != 0 && byte_count > 8) {
        word |= (uint64_t)bytes[8] << (64U - shift);
    }
    return bit_count == 64 ? word : word & ((UINT64_C(1) << bit_count) - 1);
}

static inline uint64_t dora_bitmap_get_word(const uint8_t* bitmap, uint64_t first_bit, uint64_t bit_count) {
    if (first_bit >= bit_count) {
        return 0;
    }
    const uint64_t available = bit_count - first_bit;
    return dora_bitmap_load_window(bitmap, first_bit, available < 64 ? available : 64);
}

static inline uint32_t dora_bitmap_word_count_ones(uint64_t word) {
    return __builtin_popcountll(word);
}

/// Requires *in_out_word != 0.
static inline uint32_t dora_bitmap_word_pop_lowest_set(uint64_t* in_out_word) {
    const uint32_t bit = __builtin_ctzll(*in_out_word);
    *in_out_word &= *in_out_word - 1;
    return bit;
}

#endif
