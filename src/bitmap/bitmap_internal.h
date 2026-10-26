// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Bitmap geometry and ranges.

#ifndef DORA_BITMAP_BITMAP_INTERNAL_H
#define DORA_BITMAP_BITMAP_INTERNAL_H

#include "dora/vector/bitmap.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if DORA_KERNEL_USE_NEON && defined(__aarch64__) && defined(__ARM_NEON)
#define BITMAP_USE_NEON 1
#include <arm_neon.h>

extern const uint32_t dora_bitmap_byte_set_bits[256][8];
#endif

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Dora bitmap words are little-endian byte sequences; big-endian hosts are unsupported"
#endif

// Null pointers require byte_count == 0.
static inline void bitmap_fill_bytes(uint8_t* bitmap, size_t byte_count, bool value) {
    if (byte_count != 0) {
        memset(bitmap, value ? UINT8_MAX : 0, byte_count);
    }
}

// Source and destination must be disjoint.
static inline void bitmap_copy_bytes(uint8_t* restrict destination, const uint8_t* restrict source, size_t byte_count) {
    if (byte_count != 0) {
        memcpy(destination, source, byte_count);
    }
}

static inline bool bitmap_words_are_equal(const uint64_t* left, const uint64_t* right, size_t word_count) {
    return word_count == 0 || memcmp(left, right, word_count * sizeof *left) == 0;
}

// Exact alias is allowed; partial overlap is forbidden.
static inline void bitmap_or_aligned_words(uint64_t* destination, const uint64_t* source, size_t word_count) {
    for (size_t i = 0; i < word_count; ++i) {
        destination[i] |= source[i];
    }
}

static inline void bitmap_or_not_aligned_words(uint64_t* destination, const uint64_t* source, size_t word_count) {
    for (size_t i = 0; i < word_count; ++i) {
        destination[i] |= ~source[i];
    }
}

//------------------------------------------------------------------------------
//
// Bitmap Geometry
//
//------------------------------------------------------------------------------

enum bitmap_word_geometry : uint32_t {
    DORA_BITMAP_WORD_BIT_COUNT = 64,
    DORA_BITMAP_WORD_BIT_INDEX_MASK = 63,
    DORA_BITMAP_WORD_INDEX_SHIFT = 6
};

enum bitmap_byte_geometry : uint32_t {
    DORA_BITMAP_BYTE_BIT_COUNT = 8,
    DORA_BITMAP_BYTE_BIT_INDEX_MASK = 7,
    DORA_BITMAP_BYTE_INDEX_SHIFT = 3
};

//------------------------------------------------------------------------------
//
// Inspection
//
//------------------------------------------------------------------------------

//--------------------------------------
// Word Geometry
//--------------------------------------

static inline uint32_t bitmap_bit_of(uint64_t bit) {
    return bit & DORA_BITMAP_WORD_BIT_INDEX_MASK;
}

static inline uint64_t bitmap_word_of(uint64_t bit) {
    return bit >> DORA_BITMAP_WORD_INDEX_SHIFT;
}

// Requires aligned words and an in-range bit.
static inline bool bitmap_test_aligned(const uint64_t* words, uint64_t bit) {
    return (words[bitmap_word_of(bit)] & (UINT64_C(1) << bitmap_bit_of(bit))) != 0;
}

static inline uint64_t bitmap_byte_of(uint64_t bit) {
    return bit >> DORA_BITMAP_BYTE_INDEX_SHIFT;
}

static inline uint32_t bitmap_bit_in_byte(uint64_t bit) {
    return bit & DORA_BITMAP_BYTE_BIT_INDEX_MASK;
}

// Requires the bit capacity to fit uint64_t.
static inline uint64_t bitmap_whole_word_capacity(size_t byte_count) {
    return (byte_count / sizeof(uint64_t)) * DORA_BITMAP_WORD_BIT_COUNT;
}

// Requires bit < 64.
static inline uint64_t bitmap_word_bit(uint64_t word, uint32_t bit) {
    return (word >> bit) & 1U;
}

// Requires bit < 64.
static inline uint64_t bitmap_word_bit_mask(uint64_t word, uint32_t bit) {
    return (word & (UINT64_C(1) << bit)) != 0 ? UINT64_MAX : 0;
}

static inline uint64_t bitmap_words_for_bits(uint64_t bits) {
    return bits == 0U ? 0U : (bits - 1U) / DORA_BITMAP_WORD_BIT_COUNT + 1U;
}

static inline uint64_t bitmap_bytes_for_bits(uint64_t bits) {
    return bits == 0U ? 0U : (bits - 1U) / DORA_BITMAP_BYTE_BIT_COUNT + 1U;
}

static inline uint64_t bitmap_mask_low(uint32_t bits) {
    if (bits == 0U) {
        return 0ULL;
    }
    if (bits >= DORA_BITMAP_WORD_BIT_COUNT) {
        return ~0ULL;
    }
    return (1ULL << bits) - 1ULL;
}

static inline void bitmap_prefix_words(uint64_t bits, uint64_t* words, size_t word_count) {
    for (size_t index = 0; index < word_count; ++index) {
        const uint64_t first = (uint64_t)index * DORA_BITMAP_WORD_BIT_COUNT;
        const uint64_t remaining = bits > first ? bits - first : 0U;
        words[index] = bitmap_mask_low(
            (uint32_t)(remaining < DORA_BITMAP_WORD_BIT_COUNT ? remaining : DORA_BITMAP_WORD_BIT_COUNT));
    }
}

// Requires first < 64 and count <= 64 - first.
static inline uint64_t bitmap_word_slice(uint64_t word, uint32_t first, uint32_t count) {
    return (word >> first) & bitmap_mask_low(count);
}

static inline uint64_t bitmap_mask_range(uint32_t begin, uint32_t end) {
    if (begin >= end) {
        return 0ULL;
    }
    return bitmap_mask_low(end) & ~bitmap_mask_low(begin);
}

static inline uint64_t bitmap_tail_mask(uint64_t bits) {
    const uint32_t tail_bits = bitmap_bit_of(bits);
    return tail_bits == 0U ? ~0ULL : bitmap_mask_low(tail_bits);
}

static inline uint64_t bitmap_word_tail_mask(uint64_t bits, uint64_t word) {
    return word + 1 == bitmap_words_for_bits(bits) ? bitmap_tail_mask(bits) : UINT64_MAX;
}

//--------------------------------------
// Word Counting
//--------------------------------------

static inline uint32_t bitmap_word_leading_zeros(uint64_t word) {
    return word == 0ULL ? DORA_BITMAP_WORD_BIT_COUNT : __builtin_clzll(word);
}

static inline uint32_t bitmap_word_trailing_zeros(uint64_t word) {
    return word == 0ULL ? DORA_BITMAP_WORD_BIT_COUNT : __builtin_ctzll(word);
}

// Returns end if no bit matches.
static inline uint64_t bitmap_next_aligned(const uint64_t* words, uint64_t begin, uint64_t end, bool value) {
    while (begin < end) {
        const uint64_t index = bitmap_word_of(begin);
        const uint64_t word = (value ? words[index] : ~words[index]) & ~bitmap_mask_low(bitmap_bit_of(begin));
        if (word != 0) {
            const uint64_t bit = index * DORA_BITMAP_WORD_BIT_COUNT + bitmap_word_trailing_zeros(word);
            return bit < end ? bit : end;
        }
        begin = (index + 1) * DORA_BITMAP_WORD_BIT_COUNT;
    }
    return end;
}

// Each byte must be 0 or 1.
static inline uint64_t bitmap_pack_bool_bytes_word(const uint8_t values[static 64]) {
    uint64_t match = 0;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        match |= (uint64_t)values[lane] << lane;
    }
    return match;
}

static inline uint64_t bitmap_pack_nonzero_bytes_word(const uint8_t values[static 64]) {
    uint64_t match = 0;
#ifdef BITMAP_USE_NEON
    for (uint32_t part = 0; part < 4; ++part) {
        const uint8x16_t batch = vld1q_u8(values + (size_t)part * 16);
        const uint8x16_t mask = vmvnq_u8(vceqq_u8(batch, vdupq_n_u8(0)));
        const uint8x16_t lane_bits =
            {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        const uint8x16_t bits = vandq_u8(mask, lane_bits);
        const uint32_t packed = ((uint32_t)vaddv_u8(vget_high_u8(bits)) << 8U) | (uint32_t)vaddv_u8(vget_low_u8(bits));
        match |= (uint64_t)packed << (part * 16U);
    }
#else
    for (uint32_t lane = 0; lane < 64; ++lane) {
        match |= (uint64_t)(values[lane] != 0) << lane;
    }
#endif
    return match;
}

// Requires base + 63 <= UINT32_MAX; dense NEON output needs popcount + 3 writable entries.
static inline uint32_t bitmap_word_indices(uint64_t word,
                                           uint32_t base,
#ifdef BITMAP_USE_NEON
                                           bool dense,
#else
                                           bool,
#endif
                                           uint32_t* out_indices) {
    uint32_t* result = out_indices;
    if (word == 0) {
        return 0;
    }
#ifdef BITMAP_USE_NEON
    if (dense) {
        for (uint32_t byte_count = 0; byte_count < 8; ++byte_count) {
            const uint32_t byte = (uint32_t)(word & 0xFFU);
            word >>= 8U;
            if (byte != 0) {
                const uint32_t* const indices = dora_bitmap_byte_set_bits[byte];
                const uint32x4_t row_batch = vdupq_n_u32(base);
                const uint32_t low = byte & 0xFU;
                const uint32_t high = byte >> 4U;
                uint32_t pop = 0;
                if (low != 0) {
                    vst1q_u32(result, vaddq_u32(vld1q_u32(indices), row_batch));
                    pop = dora_bitmap_word_count_ones(low);
                    result += pop;
                }
                if (high != 0) {
                    vst1q_u32(result, vaddq_u32(vld1q_u32(indices + pop), row_batch));
                    result += dora_bitmap_word_count_ones(high);
                }
            }
            base += 8;
        }
        return (uint32_t)(result - out_indices);
    }
#endif
    do {
        *result++ = dora_bitmap_word_pop_lowest_set(&word) + base;
    } while (word != 0);
    return (uint32_t)(result - out_indices);
}

//--------------------------------------
// Ranges
//--------------------------------------

uint64_t dora_bitmap_count_set_range(const uint64_t* words, uint64_t begin, uint64_t end);
uint64_t dora_bitmap_count_set_words(const uint64_t* words, uint64_t bit_count);
uint64_t dora_bitmap_count_set_bytes(const uint8_t* bitmap, uint64_t first, uint64_t bit_count);
uint64_t dora_bitmap_count_and_words(const uint64_t* value_words, const uint64_t* mask_words, uint64_t bit_count);
uint64_t dora_bitmap_count_and_not_words(const uint64_t* value_words, const uint64_t* mask_words, uint64_t bit_count);
bool dora_bitmap_find_clear_run(const uint64_t* words,
                                uint64_t begin,
                                uint64_t end,
                                uint64_t run_bit_count,
                                uint64_t* out_run_begin);

//------------------------------------------------------------------------------
//
// Mutation
//
//------------------------------------------------------------------------------

//--------------------------------------
// Bits
//--------------------------------------

static inline void bitmap_assign(uint64_t* words, uint64_t bit, bool value) {
    const uint64_t mask = 1ULL << bitmap_bit_of(bit);
    const size_t word = bitmap_word_of(bit);
    words[word] = (words[word] & ~mask) | (value ? mask : 0ULL);
}

static inline void bitmap_clear(uint64_t* words, uint64_t bit) {
    words[bitmap_word_of(bit)] &= ~(1ULL << bitmap_bit_of(bit));
}

static inline void bitmap_set(uint64_t* words, uint64_t bit) {
    words[bitmap_word_of(bit)] |= 1ULL << bitmap_bit_of(bit);
}

//--------------------------------------
// Ranges and Words
//--------------------------------------

void dora_bitmap_set_range(uint64_t* words, uint64_t begin, uint64_t end, bool value);
// Requires 1 <= count <= 64; preserves bits outside the range.
void dora_bitmap_store_bits(uint64_t* words, uint64_t bit, uint64_t value, uint32_t count);
void dora_bitmap_and_aligned_words(uint64_t* restrict dst_words, const uint64_t* restrict src_words, size_t word_count);
void dora_bitmap_not_words(uint64_t* words, uint64_t bit_count);

//------------------------------------------------------------------------------
//
// Copy
//
//------------------------------------------------------------------------------

void dora_bitmap_copy_bool8_to_words(uint64_t* restrict dst_words,
                                     const bool* restrict src_values,
                                     uint64_t src_value_offset,
                                     uint64_t value_count);
void dora_bitmap_copy_aligned_words(uint64_t* restrict dst_words,
                                    const uint64_t* restrict src_words,
                                    size_t word_count);

// Requires disjoint spans; preserves destination bits outside the range.
uint64_t dora_bitmap_copy_count(uint64_t count,
                                uint64_t* destination,
                                uint64_t destination_bit,
                                const uint8_t* source,
                                uint64_t source_bit);
void dora_bitmap_load_words(const uint8_t* bitmap, uint64_t bit_offset, uint64_t bit_count, uint64_t* out_words);

//------------------------------------------------------------------------------
//
// Gather
//
//------------------------------------------------------------------------------

void dora_bitmap_gather_words_u32(uint64_t* out_words,
                                  const uint8_t* source,
                                  uint64_t source_offset,
                                  const uint32_t* rows,
                                  const uint64_t* active_words,
                                  uint32_t row_count);
void dora_bitmap_gather_words_u64(uint64_t* out_words,
                                  const uint8_t* source,
                                  uint64_t source_offset,
                                  const uint64_t* rows,
                                  const uint64_t* active_words,
                                  uint32_t row_count);

#endif
