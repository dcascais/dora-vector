// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/bitmap.h"

#include "bitmap/bitmap_internal.h"

#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__clang__)
static constexpr size_t BITMAP_NEON_COPY_THRESHOLD_WORDS = 128;
#endif

#ifdef BITMAP_USE_NEON

static constexpr size_t BITMAP_NEON_WORDS_PER_ITERATION = 8;
static constexpr size_t BITMAP_NEON_WORDS_PER_BLOCK = 1024;
#endif

#ifdef BITMAP_USE_NEON
alignas(16) const uint32_t dora_bitmap_byte_set_bits[256][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {1, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {2, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 2, 3, 4, 5, 6, 7}, {1, 2, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {3, 1, 2, 3, 4, 5, 6, 7}, {0, 3, 2, 3, 4, 5, 6, 7}, {1, 3, 2, 3, 4, 5, 6, 7}, {0, 1, 3, 3, 4, 5, 6, 7},
    {2, 3, 2, 3, 4, 5, 6, 7}, {0, 2, 3, 3, 4, 5, 6, 7}, {1, 2, 3, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {4, 1, 2, 3, 4, 5, 6, 7}, {0, 4, 2, 3, 4, 5, 6, 7}, {1, 4, 2, 3, 4, 5, 6, 7}, {0, 1, 4, 3, 4, 5, 6, 7},
    {2, 4, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 3, 4, 5, 6, 7}, {1, 2, 4, 3, 4, 5, 6, 7}, {0, 1, 2, 4, 4, 5, 6, 7},
    {3, 4, 2, 3, 4, 5, 6, 7}, {0, 3, 4, 3, 4, 5, 6, 7}, {1, 3, 4, 3, 4, 5, 6, 7}, {0, 1, 3, 4, 4, 5, 6, 7},
    {2, 3, 4, 3, 4, 5, 6, 7}, {0, 2, 3, 4, 4, 5, 6, 7}, {1, 2, 3, 4, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {5, 1, 2, 3, 4, 5, 6, 7}, {0, 5, 2, 3, 4, 5, 6, 7}, {1, 5, 2, 3, 4, 5, 6, 7}, {0, 1, 5, 3, 4, 5, 6, 7},
    {2, 5, 2, 3, 4, 5, 6, 7}, {0, 2, 5, 3, 4, 5, 6, 7}, {1, 2, 5, 3, 4, 5, 6, 7}, {0, 1, 2, 5, 4, 5, 6, 7},
    {3, 5, 2, 3, 4, 5, 6, 7}, {0, 3, 5, 3, 4, 5, 6, 7}, {1, 3, 5, 3, 4, 5, 6, 7}, {0, 1, 3, 5, 4, 5, 6, 7},
    {2, 3, 5, 3, 4, 5, 6, 7}, {0, 2, 3, 5, 4, 5, 6, 7}, {1, 2, 3, 5, 4, 5, 6, 7}, {0, 1, 2, 3, 5, 5, 6, 7},
    {4, 5, 2, 3, 4, 5, 6, 7}, {0, 4, 5, 3, 4, 5, 6, 7}, {1, 4, 5, 3, 4, 5, 6, 7}, {0, 1, 4, 5, 4, 5, 6, 7},
    {2, 4, 5, 3, 4, 5, 6, 7}, {0, 2, 4, 5, 4, 5, 6, 7}, {1, 2, 4, 5, 4, 5, 6, 7}, {0, 1, 2, 4, 5, 5, 6, 7},
    {3, 4, 5, 3, 4, 5, 6, 7}, {0, 3, 4, 5, 4, 5, 6, 7}, {1, 3, 4, 5, 4, 5, 6, 7}, {0, 1, 3, 4, 5, 5, 6, 7},
    {2, 3, 4, 5, 4, 5, 6, 7}, {0, 2, 3, 4, 5, 5, 6, 7}, {1, 2, 3, 4, 5, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {6, 1, 2, 3, 4, 5, 6, 7}, {0, 6, 2, 3, 4, 5, 6, 7}, {1, 6, 2, 3, 4, 5, 6, 7}, {0, 1, 6, 3, 4, 5, 6, 7},
    {2, 6, 2, 3, 4, 5, 6, 7}, {0, 2, 6, 3, 4, 5, 6, 7}, {1, 2, 6, 3, 4, 5, 6, 7}, {0, 1, 2, 6, 4, 5, 6, 7},
    {3, 6, 2, 3, 4, 5, 6, 7}, {0, 3, 6, 3, 4, 5, 6, 7}, {1, 3, 6, 3, 4, 5, 6, 7}, {0, 1, 3, 6, 4, 5, 6, 7},
    {2, 3, 6, 3, 4, 5, 6, 7}, {0, 2, 3, 6, 4, 5, 6, 7}, {1, 2, 3, 6, 4, 5, 6, 7}, {0, 1, 2, 3, 6, 5, 6, 7},
    {4, 6, 2, 3, 4, 5, 6, 7}, {0, 4, 6, 3, 4, 5, 6, 7}, {1, 4, 6, 3, 4, 5, 6, 7}, {0, 1, 4, 6, 4, 5, 6, 7},
    {2, 4, 6, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 4, 5, 6, 7}, {1, 2, 4, 6, 4, 5, 6, 7}, {0, 1, 2, 4, 6, 5, 6, 7},
    {3, 4, 6, 3, 4, 5, 6, 7}, {0, 3, 4, 6, 4, 5, 6, 7}, {1, 3, 4, 6, 4, 5, 6, 7}, {0, 1, 3, 4, 6, 5, 6, 7},
    {2, 3, 4, 6, 4, 5, 6, 7}, {0, 2, 3, 4, 6, 5, 6, 7}, {1, 2, 3, 4, 6, 5, 6, 7}, {0, 1, 2, 3, 4, 6, 6, 7},
    {5, 6, 2, 3, 4, 5, 6, 7}, {0, 5, 6, 3, 4, 5, 6, 7}, {1, 5, 6, 3, 4, 5, 6, 7}, {0, 1, 5, 6, 4, 5, 6, 7},
    {2, 5, 6, 3, 4, 5, 6, 7}, {0, 2, 5, 6, 4, 5, 6, 7}, {1, 2, 5, 6, 4, 5, 6, 7}, {0, 1, 2, 5, 6, 5, 6, 7},
    {3, 5, 6, 3, 4, 5, 6, 7}, {0, 3, 5, 6, 4, 5, 6, 7}, {1, 3, 5, 6, 4, 5, 6, 7}, {0, 1, 3, 5, 6, 5, 6, 7},
    {2, 3, 5, 6, 4, 5, 6, 7}, {0, 2, 3, 5, 6, 5, 6, 7}, {1, 2, 3, 5, 6, 5, 6, 7}, {0, 1, 2, 3, 5, 6, 6, 7},
    {4, 5, 6, 3, 4, 5, 6, 7}, {0, 4, 5, 6, 4, 5, 6, 7}, {1, 4, 5, 6, 4, 5, 6, 7}, {0, 1, 4, 5, 6, 5, 6, 7},
    {2, 4, 5, 6, 4, 5, 6, 7}, {0, 2, 4, 5, 6, 5, 6, 7}, {1, 2, 4, 5, 6, 5, 6, 7}, {0, 1, 2, 4, 5, 6, 6, 7},
    {3, 4, 5, 6, 4, 5, 6, 7}, {0, 3, 4, 5, 6, 5, 6, 7}, {1, 3, 4, 5, 6, 5, 6, 7}, {0, 1, 3, 4, 5, 6, 6, 7},
    {2, 3, 4, 5, 6, 5, 6, 7}, {0, 2, 3, 4, 5, 6, 6, 7}, {1, 2, 3, 4, 5, 6, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {7, 1, 2, 3, 4, 5, 6, 7}, {0, 7, 2, 3, 4, 5, 6, 7}, {1, 7, 2, 3, 4, 5, 6, 7}, {0, 1, 7, 3, 4, 5, 6, 7},
    {2, 7, 2, 3, 4, 5, 6, 7}, {0, 2, 7, 3, 4, 5, 6, 7}, {1, 2, 7, 3, 4, 5, 6, 7}, {0, 1, 2, 7, 4, 5, 6, 7},
    {3, 7, 2, 3, 4, 5, 6, 7}, {0, 3, 7, 3, 4, 5, 6, 7}, {1, 3, 7, 3, 4, 5, 6, 7}, {0, 1, 3, 7, 4, 5, 6, 7},
    {2, 3, 7, 3, 4, 5, 6, 7}, {0, 2, 3, 7, 4, 5, 6, 7}, {1, 2, 3, 7, 4, 5, 6, 7}, {0, 1, 2, 3, 7, 5, 6, 7},
    {4, 7, 2, 3, 4, 5, 6, 7}, {0, 4, 7, 3, 4, 5, 6, 7}, {1, 4, 7, 3, 4, 5, 6, 7}, {0, 1, 4, 7, 4, 5, 6, 7},
    {2, 4, 7, 3, 4, 5, 6, 7}, {0, 2, 4, 7, 4, 5, 6, 7}, {1, 2, 4, 7, 4, 5, 6, 7}, {0, 1, 2, 4, 7, 5, 6, 7},
    {3, 4, 7, 3, 4, 5, 6, 7}, {0, 3, 4, 7, 4, 5, 6, 7}, {1, 3, 4, 7, 4, 5, 6, 7}, {0, 1, 3, 4, 7, 5, 6, 7},
    {2, 3, 4, 7, 4, 5, 6, 7}, {0, 2, 3, 4, 7, 5, 6, 7}, {1, 2, 3, 4, 7, 5, 6, 7}, {0, 1, 2, 3, 4, 7, 6, 7},
    {5, 7, 2, 3, 4, 5, 6, 7}, {0, 5, 7, 3, 4, 5, 6, 7}, {1, 5, 7, 3, 4, 5, 6, 7}, {0, 1, 5, 7, 4, 5, 6, 7},
    {2, 5, 7, 3, 4, 5, 6, 7}, {0, 2, 5, 7, 4, 5, 6, 7}, {1, 2, 5, 7, 4, 5, 6, 7}, {0, 1, 2, 5, 7, 5, 6, 7},
    {3, 5, 7, 3, 4, 5, 6, 7}, {0, 3, 5, 7, 4, 5, 6, 7}, {1, 3, 5, 7, 4, 5, 6, 7}, {0, 1, 3, 5, 7, 5, 6, 7},
    {2, 3, 5, 7, 4, 5, 6, 7}, {0, 2, 3, 5, 7, 5, 6, 7}, {1, 2, 3, 5, 7, 5, 6, 7}, {0, 1, 2, 3, 5, 7, 6, 7},
    {4, 5, 7, 3, 4, 5, 6, 7}, {0, 4, 5, 7, 4, 5, 6, 7}, {1, 4, 5, 7, 4, 5, 6, 7}, {0, 1, 4, 5, 7, 5, 6, 7},
    {2, 4, 5, 7, 4, 5, 6, 7}, {0, 2, 4, 5, 7, 5, 6, 7}, {1, 2, 4, 5, 7, 5, 6, 7}, {0, 1, 2, 4, 5, 7, 6, 7},
    {3, 4, 5, 7, 4, 5, 6, 7}, {0, 3, 4, 5, 7, 5, 6, 7}, {1, 3, 4, 5, 7, 5, 6, 7}, {0, 1, 3, 4, 5, 7, 6, 7},
    {2, 3, 4, 5, 7, 5, 6, 7}, {0, 2, 3, 4, 5, 7, 6, 7}, {1, 2, 3, 4, 5, 7, 6, 7}, {0, 1, 2, 3, 4, 5, 7, 7},
    {6, 7, 2, 3, 4, 5, 6, 7}, {0, 6, 7, 3, 4, 5, 6, 7}, {1, 6, 7, 3, 4, 5, 6, 7}, {0, 1, 6, 7, 4, 5, 6, 7},
    {2, 6, 7, 3, 4, 5, 6, 7}, {0, 2, 6, 7, 4, 5, 6, 7}, {1, 2, 6, 7, 4, 5, 6, 7}, {0, 1, 2, 6, 7, 5, 6, 7},
    {3, 6, 7, 3, 4, 5, 6, 7}, {0, 3, 6, 7, 4, 5, 6, 7}, {1, 3, 6, 7, 4, 5, 6, 7}, {0, 1, 3, 6, 7, 5, 6, 7},
    {2, 3, 6, 7, 4, 5, 6, 7}, {0, 2, 3, 6, 7, 5, 6, 7}, {1, 2, 3, 6, 7, 5, 6, 7}, {0, 1, 2, 3, 6, 7, 6, 7},
    {4, 6, 7, 3, 4, 5, 6, 7}, {0, 4, 6, 7, 4, 5, 6, 7}, {1, 4, 6, 7, 4, 5, 6, 7}, {0, 1, 4, 6, 7, 5, 6, 7},
    {2, 4, 6, 7, 4, 5, 6, 7}, {0, 2, 4, 6, 7, 5, 6, 7}, {1, 2, 4, 6, 7, 5, 6, 7}, {0, 1, 2, 4, 6, 7, 6, 7},
    {3, 4, 6, 7, 4, 5, 6, 7}, {0, 3, 4, 6, 7, 5, 6, 7}, {1, 3, 4, 6, 7, 5, 6, 7}, {0, 1, 3, 4, 6, 7, 6, 7},
    {2, 3, 4, 6, 7, 5, 6, 7}, {0, 2, 3, 4, 6, 7, 6, 7}, {1, 2, 3, 4, 6, 7, 6, 7}, {0, 1, 2, 3, 4, 6, 7, 7},
    {5, 6, 7, 3, 4, 5, 6, 7}, {0, 5, 6, 7, 4, 5, 6, 7}, {1, 5, 6, 7, 4, 5, 6, 7}, {0, 1, 5, 6, 7, 5, 6, 7},
    {2, 5, 6, 7, 4, 5, 6, 7}, {0, 2, 5, 6, 7, 5, 6, 7}, {1, 2, 5, 6, 7, 5, 6, 7}, {0, 1, 2, 5, 6, 7, 6, 7},
    {3, 5, 6, 7, 4, 5, 6, 7}, {0, 3, 5, 6, 7, 5, 6, 7}, {1, 3, 5, 6, 7, 5, 6, 7}, {0, 1, 3, 5, 6, 7, 6, 7},
    {2, 3, 5, 6, 7, 5, 6, 7}, {0, 2, 3, 5, 6, 7, 6, 7}, {1, 2, 3, 5, 6, 7, 6, 7}, {0, 1, 2, 3, 5, 6, 7, 7},
    {4, 5, 6, 7, 4, 5, 6, 7}, {0, 4, 5, 6, 7, 5, 6, 7}, {1, 4, 5, 6, 7, 5, 6, 7}, {0, 1, 4, 5, 6, 7, 6, 7},
    {2, 4, 5, 6, 7, 5, 6, 7}, {0, 2, 4, 5, 6, 7, 6, 7}, {1, 2, 4, 5, 6, 7, 6, 7}, {0, 1, 2, 4, 5, 6, 7, 7},
    {3, 4, 5, 6, 7, 5, 6, 7}, {0, 3, 4, 5, 6, 7, 6, 7}, {1, 3, 4, 5, 6, 7, 6, 7}, {0, 1, 3, 4, 5, 6, 7, 7},
    {2, 3, 4, 5, 6, 7, 6, 7}, {0, 2, 3, 4, 5, 6, 7, 7}, {1, 2, 3, 4, 5, 6, 7, 7}, {0, 1, 2, 3, 4, 5, 6, 7}
};
#endif

//------------------------------------------------------------------------------
//
// Count Kernels
//
//------------------------------------------------------------------------------

// Re-evaluates arguments per word; WORD and TOTAL require lvalues.
#define BITMAP_COUNT_SCALAR_WORDS(BYTES, WORD, END, TOTAL)                                                             \
    for (; (WORD) < (END); ++(WORD)) {                                                                                 \
        uint64_t value = 0;                                                                                            \
        memcpy(&value, (BYTES) + (WORD) * sizeof(uint64_t), sizeof value);                                             \
        (TOTAL) += dora_bitmap_word_count_ones(value);                                                                 \
    }

static inline uint64_t bitmap_count_whole_bytes(const uint8_t* bytes, size_t word_count) {
    uint64_t total = 0;
    size_t word = 0;
#ifdef BITMAP_USE_NEON
    while (word + BITMAP_NEON_WORDS_PER_ITERATION <= word_count) {
        const size_t remaining_words = word_count - word;
        size_t block_words =
            remaining_words < BITMAP_NEON_WORDS_PER_BLOCK ? remaining_words : BITMAP_NEON_WORDS_PER_BLOCK;
        block_words -= block_words % BITMAP_NEON_WORDS_PER_ITERATION;
        const size_t block_end = word + block_words;
        uint16x8_t count_0 = vdupq_n_u16(0);
        uint16x8_t count_1 = vdupq_n_u16(0);
        uint16x8_t count_2 = vdupq_n_u16(0);
        uint16x8_t count_3 = vdupq_n_u16(0);
        for (; word < block_end; word += BITMAP_NEON_WORDS_PER_ITERATION) {
            const uint8_t* block = bytes + word * sizeof(uint64_t);
            count_0 = vpadalq_u8(count_0, vcntq_u8(vld1q_u8(block)));
            count_1 = vpadalq_u8(count_1, vcntq_u8(vld1q_u8(block + 16)));
            count_2 = vpadalq_u8(count_2, vcntq_u8(vld1q_u8(block + 32)));
            count_3 = vpadalq_u8(count_3, vcntq_u8(vld1q_u8(block + 48)));
        }
        const uint16x8_t counts_16 = vaddq_u16(vaddq_u16(count_0, count_1), vaddq_u16(count_2, count_3));
        const uint64x2_t counts_64 = vpaddlq_u32(vpaddlq_u16(counts_16));
        total += vgetq_lane_u64(counts_64, 0) + vgetq_lane_u64(counts_64, 1);
    }
#endif
    BITMAP_COUNT_SCALAR_WORDS(bytes, word, word_count, total)
    return total;
}

#ifdef BITMAP_USE_NEON
#define DEFINE_BITMAP_NEON_LOGICAL_COUNT(NAME, LOGICAL)                                                                \
    static uint64_t NAME(const uint64_t* lhs, const uint64_t* rhs, size_t word_count) {                                \
        uint64_t total = 0;                                                                                            \
        size_t word = 0;                                                                                               \
        while (word < word_count) {                                                                                    \
            const size_t remaining_words = word_count - word;                                                          \
            const size_t block_words =                                                                                 \
                remaining_words < BITMAP_NEON_WORDS_PER_BLOCK ? remaining_words : BITMAP_NEON_WORDS_PER_BLOCK;         \
            const size_t block_end = word + block_words;                                                               \
            uint16x8_t count_0 = vdupq_n_u16(0);                                                                       \
            uint16x8_t count_1 = vdupq_n_u16(0);                                                                       \
            uint16x8_t count_2 = vdupq_n_u16(0);                                                                       \
            uint16x8_t count_3 = vdupq_n_u16(0);                                                                       \
            for (; word < block_end; word += BITMAP_NEON_WORDS_PER_ITERATION) {                                        \
                const uint64x2_t bits_0 = LOGICAL(vld1q_u64(lhs + word), vld1q_u64(rhs + word));                       \
                const uint64x2_t bits_1 = LOGICAL(vld1q_u64(lhs + word + 2), vld1q_u64(rhs + word + 2));               \
                const uint64x2_t bits_2 = LOGICAL(vld1q_u64(lhs + word + 4), vld1q_u64(rhs + word + 4));               \
                const uint64x2_t bits_3 = LOGICAL(vld1q_u64(lhs + word + 6), vld1q_u64(rhs + word + 6));               \
                count_0 = vpadalq_u8(count_0, vcntq_u8(vreinterpretq_u8_u64(bits_0)));                                 \
                count_1 = vpadalq_u8(count_1, vcntq_u8(vreinterpretq_u8_u64(bits_1)));                                 \
                count_2 = vpadalq_u8(count_2, vcntq_u8(vreinterpretq_u8_u64(bits_2)));                                 \
                count_3 = vpadalq_u8(count_3, vcntq_u8(vreinterpretq_u8_u64(bits_3)));                                 \
            }                                                                                                          \
            const uint16x8_t counts_16 = vaddq_u16(vaddq_u16(count_0, count_1), vaddq_u16(count_2, count_3));          \
            const uint64x2_t counts_64 = vpaddlq_u32(vpaddlq_u16(counts_16));                                          \
            total += vgetq_lane_u64(counts_64, 0) + vgetq_lane_u64(counts_64, 1);                                      \
        }                                                                                                              \
        return total;                                                                                                  \
    }

DEFINE_BITMAP_NEON_LOGICAL_COUNT(bitmap_count_and_words_neon, vandq_u64)
DEFINE_BITMAP_NEON_LOGICAL_COUNT(bitmap_count_and_not_words_neon, vbicq_u64)

#undef DEFINE_BITMAP_NEON_LOGICAL_COUNT
#define BITMAP_COUNT_LOGICAL_NEON(NAME)                                                                                \
    do {                                                                                                               \
        word = full_words - full_words % BITMAP_NEON_WORDS_PER_ITERATION;                                              \
        total = NAME(value_words, mask_words, word);                                                                   \
    } while (0)
#else
// NAME is not expanded or evaluated.
#define BITMAP_COUNT_LOGICAL_NEON(NAME)                                                                                \
    do {                                                                                                               \
    } while (0)
#endif

//------------------------------------------------------------------------------
//
// Inspection
//
//------------------------------------------------------------------------------

uint64_t dora_bitmap_count_set_range(const uint64_t* words, uint64_t begin, uint64_t end) {
    if (words == nullptr || begin >= end || bitmap_words_for_bits(end) > SIZE_MAX / sizeof(*words)) {
        return 0;
    }

    const size_t first = bitmap_word_of(begin);
    const size_t last = bitmap_word_of(end - 1U);
    const uint32_t end_bit = bitmap_bit_of(end - 1U) + 1U;
    if (first == last) {
        return dora_bitmap_word_count_ones(words[first] & bitmap_mask_range(bitmap_bit_of(begin), end_bit));
    }

    uint64_t total = dora_bitmap_word_count_ones(words[first] & ~bitmap_mask_low(bitmap_bit_of(begin)));
    size_t word = first + 1U;
    BITMAP_COUNT_SCALAR_WORDS((const uint8_t*)words, word, last, total)
    return total + dora_bitmap_word_count_ones(words[last] & bitmap_mask_low(end_bit));
}

#undef BITMAP_COUNT_SCALAR_WORDS

uint64_t dora_bitmap_count_set_words(const uint64_t* words, uint64_t bit_count) {
    const size_t full_words = bitmap_word_of(bit_count);
    const uint64_t total = bitmap_count_whole_bytes((const uint8_t*)words, full_words);
    const uint32_t tail_bits = bitmap_bit_of(bit_count);
    if (tail_bits == 0) {
        return total;
    }
    return total + dora_bitmap_word_count_ones(words[full_words] & bitmap_mask_low(tail_bits));
}

uint64_t dora_bitmap_count_set_bytes(const uint8_t* bitmap, uint64_t first, uint64_t bit_count) {
    if (bit_count == 0) {
        return 0;
    }
    uint64_t total = 0;
    uint64_t bit = first;
    uint64_t remaining = bit_count;
    const uint32_t head_offset = bitmap_bit_in_byte(bit);
    if (head_offset != 0) {
        const uint32_t head_room = DORA_BITMAP_BYTE_BIT_COUNT - head_offset;
        const uint32_t head_bits = remaining < head_room ? (uint32_t)remaining : head_room;
        total += dora_bitmap_word_count_ones(dora_bitmap_load_window(bitmap, bit, head_bits));
        bit += head_bits;
        remaining -= head_bits;
    }
    const size_t full_words = bitmap_word_of(remaining);
    total += bitmap_count_whole_bytes(bitmap + (bit >> DORA_BITMAP_BYTE_INDEX_SHIFT), full_words);
    const uint32_t tail_bits = bitmap_bit_of(remaining);
    if (tail_bits != 0) {
        bit += (uint64_t)full_words * DORA_BITMAP_WORD_BIT_COUNT;
        total += dora_bitmap_word_count_ones(dora_bitmap_load_window(bitmap, bit, tail_bits));
    }
    return total;
}

#define DEFINE_BITMAP_LOGICAL_COUNT(NAME, LOGICAL, INVERT)                                                             \
    uint64_t NAME(const uint64_t* value_words, const uint64_t* mask_words, uint64_t bit_count) {                       \
        const size_t full_words = bitmap_word_of(bit_count);                                                           \
        uint64_t total = 0;                                                                                            \
        size_t word = 0;                                                                                               \
        BITMAP_COUNT_LOGICAL_NEON(LOGICAL);                                                                            \
        for (; word < full_words; ++word) {                                                                            \
            total += dora_bitmap_word_count_ones(value_words[word] & INVERT mask_words[word]);                         \
        }                                                                                                              \
        const uint32_t tail_bits = bitmap_bit_of(bit_count);                                                           \
        if (tail_bits == 0) {                                                                                          \
            return total;                                                                                              \
        }                                                                                                              \
        return total + dora_bitmap_word_count_ones(value_words[full_words] & INVERT mask_words[full_words] &           \
                                                   bitmap_mask_low(tail_bits));                                        \
    }
DEFINE_BITMAP_LOGICAL_COUNT(dora_bitmap_count_and_words, bitmap_count_and_words_neon, )
DEFINE_BITMAP_LOGICAL_COUNT(dora_bitmap_count_and_not_words, bitmap_count_and_not_words_neon, ~)
#undef DEFINE_BITMAP_LOGICAL_COUNT
#undef BITMAP_COUNT_LOGICAL_NEON

bool dora_bitmap_find_clear_run(const uint64_t* words,
                                uint64_t begin,
                                uint64_t end,
                                uint64_t run_bit_count,
                                uint64_t* out_run_begin) {
    if (words == nullptr || out_run_begin == nullptr || begin >= end || run_bit_count == 0U ||
        run_bit_count > end - begin || bitmap_words_for_bits(end) > SIZE_MAX / sizeof(*words)) {
        return false;
    }

    uint64_t clear_run = 0;
    uint64_t index = begin;
    while (index < end) {
        const size_t word_index = bitmap_word_of(index);
        const uint32_t bit = bitmap_bit_of(index);
        const uint32_t in_word = DORA_BITMAP_WORD_BIT_COUNT - bit;
        const uint64_t remaining = end - index;
        const uint32_t span = remaining < in_word ? (uint32_t)remaining : in_word;
        const uint64_t span_mask = bitmap_mask_low(span);
        const uint64_t occupied = (words[word_index] >> bit) & span_mask;

        if (occupied == 0ULL) {
            clear_run += span;
            index += span;
            if (clear_run >= run_bit_count) {
                *out_run_begin = index - clear_run;
                return true;
            }
            continue;
        }

        const uint32_t leading_clear = bitmap_word_trailing_zeros(occupied);
        clear_run += leading_clear;
        if (clear_run >= run_bit_count) {
            *out_run_begin = index + leading_clear - clear_run;
            return true;
        }

        const uint64_t after_clear = occupied >> leading_clear;
        const uint64_t clear_again = ~after_clear & (span_mask >> leading_clear);
        const uint32_t occupied_length =
            clear_again != 0ULL ? bitmap_word_trailing_zeros(clear_again) : span - leading_clear;
        index += leading_clear + occupied_length;
        clear_run = 0;
    }
    return false;
}

//------------------------------------------------------------------------------
//
// Mutation
//
//------------------------------------------------------------------------------

void dora_bitmap_set_range(uint64_t* words, uint64_t begin, uint64_t end, bool value) {
    if (words == nullptr || begin >= end || bitmap_words_for_bits(end) > SIZE_MAX / sizeof(*words)) {
        return;
    }

    const size_t first = bitmap_word_of(begin);
    const size_t last = bitmap_word_of(end - 1U);
    const uint32_t end_bit = bitmap_bit_of(end - 1U) + 1U;
    const uint64_t set_mask = value ? ~0ULL : 0ULL;

    if (first == last) {
        const uint64_t mask = bitmap_mask_range(bitmap_bit_of(begin), end_bit);
        words[first] = (words[first] & ~mask) | (mask & set_mask);
        return;
    }

    const uint64_t head_keep = bitmap_mask_low(bitmap_bit_of(begin));
    words[first] = (words[first] & head_keep) | (~head_keep & set_mask);
    for (size_t word = first + 1U; word < last; ++word) {
        words[word] = set_mask;
    }
    const uint64_t tail = bitmap_mask_low(end_bit);
    words[last] = (words[last] & ~tail) | (tail & set_mask);
}

void dora_bitmap_store_bits(uint64_t* words, const uint64_t bit, uint64_t value, const uint32_t count) {
    if (words == nullptr || count == 0U || count > DORA_BITMAP_WORD_BIT_COUNT || bit > UINT64_MAX - count) {
        return;
    }
    const uint64_t mask = bitmap_mask_low(count);
    const size_t index = bitmap_word_of(bit);
    const uint32_t shift = bitmap_bit_of(bit);
    value &= mask;
    words[index] = (words[index] & ~(mask << shift)) | (value << shift);
    if (shift != 0U && shift + count > DORA_BITMAP_WORD_BIT_COUNT) {
        const uint64_t spill = bitmap_mask_low(shift + count - DORA_BITMAP_WORD_BIT_COUNT);
        words[index + 1U] = (words[index + 1U] & ~spill) | (value >> (DORA_BITMAP_WORD_BIT_COUNT - shift));
    }
}

void dora_bitmap_and_aligned_words(uint64_t* restrict dst_words,
                                   const uint64_t* restrict src_words,
                                   size_t word_count) {
    if (dst_words == nullptr || src_words == nullptr) {
        return;
    }
    for (size_t word = 0; word < word_count; ++word) {
        dst_words[word] &= src_words[word];
    }
}

void dora_bitmap_not_words(uint64_t* words, uint64_t bit_count) {
    const size_t word_count = bitmap_words_for_bits(bit_count);
    if (word_count == 0) {
        return;
    }
    for (size_t word = 0; word < word_count; ++word) {
        words[word] = ~words[word];
    }
    words[word_count - 1] &= bitmap_tail_mask(bit_count);
}

//------------------------------------------------------------------------------
//
// Copy
//
//------------------------------------------------------------------------------

void dora_bitmap_copy_bool8_to_words(uint64_t* restrict dst_words,
                                     const bool* restrict src_values,
                                     uint64_t src_value_offset,
                                     uint64_t value_count) {
    if (dst_words == nullptr || src_values == nullptr || value_count == 0U ||
        value_count > UINT64_MAX - src_value_offset) {
        return;
    }

    const uint64_t source_end = src_value_offset + value_count;
    if (source_end > SIZE_MAX) {
        return;
    }
    const size_t source_offset = src_value_offset;
    const size_t word_count = bitmap_words_for_bits(value_count);
    size_t value = 0;
    for (size_t word = 0; word < word_count; ++word) {
        const size_t remaining = (size_t)value_count - value;
        const size_t lanes = remaining < DORA_BITMAP_WORD_BIT_COUNT ? remaining : DORA_BITMAP_WORD_BIT_COUNT;
        uint64_t packed = 0ULL;
        for (size_t lane = 0; lane < lanes; ++lane) {
            packed |= (uint64_t)src_values[source_offset + value + lane] << lane;
        }
        dst_words[word] = packed;
        value += lanes;
    }
}

void dora_bitmap_copy_aligned_words(uint64_t* restrict dst_words,
                                    const uint64_t* restrict src_words,
                                    size_t word_count) {
    if (dst_words == nullptr || src_words == nullptr || word_count == 0 || word_count > SIZE_MAX / sizeof(*dst_words)) {
        return;
    }
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__clang__)
    if (word_count <= BITMAP_NEON_COPY_THRESHOLD_WORDS) {
        for (size_t word = 0; word < word_count; ++word) {
            dst_words[word] = src_words[word];
        }
        return;
    }
#endif
    memcpy(dst_words, src_words, word_count * sizeof(uint64_t));
}

uint64_t dora_bitmap_copy_count(uint64_t count,
                                uint64_t* destination,
                                uint64_t destination_bit,
                                const uint8_t* source,
                                uint64_t source_bit) {
    uint64_t total = 0;
    for (uint64_t at = 0; at < count;) {
        const uint32_t lanes = count - at < 64 ? (uint32_t)(count - at) : 64;
        const uint64_t word = dora_bitmap_load_window(source, source_bit + at, lanes);
        dora_bitmap_store_bits(destination, destination_bit + at, word, lanes);
        total += dora_bitmap_word_count_ones(word);
        at += lanes;
    }
    return total;
}

void dora_bitmap_load_words(const uint8_t* bitmap, uint64_t bit_offset, uint64_t bit_count, uint64_t* out_words) {
    const size_t full_words = bitmap_word_of(bit_count);
    if (bitmap_bit_of(bit_offset) == 0) {
        if (full_words != 0) {
            (memcpy)(out_words, bitmap + (bit_offset >> DORA_BITMAP_BYTE_INDEX_SHIFT), full_words * sizeof(uint64_t));
        }
    } else {
        for (size_t word = 0; word < full_words; ++word) {
            out_words[word] = dora_bitmap_load_window(bitmap,
                                                      bit_offset + (uint64_t)word * DORA_BITMAP_WORD_BIT_COUNT,
                                                      DORA_BITMAP_WORD_BIT_COUNT);
        }
    }
    const uint32_t tail_bits = bitmap_bit_of(bit_count);
    if (tail_bits != 0) {
        out_words[full_words] = dora_bitmap_load_window(bitmap, bit_offset + bit_count - tail_bits, tail_bits);
    }
}

//------------------------------------------------------------------------------
//
// Gather
//
//------------------------------------------------------------------------------

// Row-index carriers: UInt32, UInt64
#define DEFINE_BITMAP_GATHER_WORDS(NAME, ROW_T)                                                                        \
    void NAME(uint64_t* out_words,                                                                                     \
              const uint8_t* source,                                                                                   \
              uint64_t source_offset,                                                                                  \
              const ROW_T* const rows,                                                                                 \
              const uint64_t* active_words,                                                                            \
              uint32_t row_count) {                                                                                    \
        const size_t word_count = bitmap_words_for_bits(row_count);                                                    \
        for (size_t word = 0; word < word_count; ++word) {                                                             \
            const uint32_t first_lane = (uint32_t)word * DORA_BITMAP_WORD_BIT_COUNT;                                   \
            const uint32_t remaining = row_count - first_lane;                                                         \
            const uint32_t lane_count =                                                                                \
                remaining < DORA_BITMAP_WORD_BIT_COUNT ? remaining : DORA_BITMAP_WORD_BIT_COUNT;                       \
            const uint64_t lane_mask = bitmap_mask_low(lane_count);                                                    \
            const uint64_t active = active_words[word] & lane_mask;                                                    \
            const ROW_T* lane_rows = rows + first_lane;                                                                \
            uint64_t gathered = 0;                                                                                     \
            if (active == lane_mask) {                                                                                 \
                for (uint32_t lane = 0; lane < lane_count; ++lane) {                                                   \
                    if (dora_bitmap_test(source, source_offset + lane_rows[lane])) {                                   \
                        gathered |= 1ULL << lane;                                                                      \
                    }                                                                                                  \
                }                                                                                                      \
            } else if (active != 0 && dora_bitmap_word_count_ones(active) * 4U >= lane_count * 3U) {                   \
                for (uint32_t lane = 0; lane < lane_count; ++lane) {                                                   \
                    if (((active >> lane) & 1U) != 0 && dora_bitmap_test(source, source_offset + lane_rows[lane])) {   \
                        gathered |= 1ULL << lane;                                                                      \
                    }                                                                                                  \
                }                                                                                                      \
            } else {                                                                                                   \
                uint64_t pending = active;                                                                             \
                while (pending != 0) {                                                                                 \
                    const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);                                   \
                    if (dora_bitmap_test(source, source_offset + lane_rows[lane])) {                                   \
                        gathered |= 1ULL << lane;                                                                      \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            out_words[word] = gathered;                                                                                \
        }                                                                                                              \
    }
DEFINE_BITMAP_GATHER_WORDS(dora_bitmap_gather_words_u32, uint32_t)
DEFINE_BITMAP_GATHER_WORDS(dora_bitmap_gather_words_u64, uint64_t)
#undef DEFINE_BITMAP_GATHER_WORDS

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

#ifdef BITMAP_USE_NEON
static_assert(BITMAP_NEON_WORDS_PER_BLOCK % BITMAP_NEON_WORDS_PER_ITERATION == 0,
              "NEON popcount blocks must contain complete iterations");
static_assert((BITMAP_NEON_WORDS_PER_BLOCK / BITMAP_NEON_WORDS_PER_ITERATION) * 16U * 4U <= UINT16_MAX,
              "NEON popcount blocks must not overflow 16-bit lanes");
#endif
