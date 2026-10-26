// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "kernel/packet.h"
#include "kernel/selection.h"

#include <stddef.h>
#include <stdint.h>

#if DORA_KERNEL_USE_NEON && defined(__aarch64__) && defined(__ARM_NEON)
#define SELECT_USE_NEON 1
#endif

//------------------------------------------------------------------------------
//
// Selection
//
//------------------------------------------------------------------------------

enum : uint32_t {
    SELECT_TABLE_PADDING = 3,
    SELECT_DUAL_WRITE_PADDING = 1
};

struct select_source {
    const uint64_t* matches;
    const uint64_t* validity;
};

struct select_plane {
    void* positions;
    uint64_t* count;
    uint64_t chunk_first;
    bool is_wide;
};

struct select_word {
    uint64_t lanes;
    uint64_t base;
    uint32_t row;
    uint32_t lane_count;
};

//--------------------------------------
// Whole Words
//--------------------------------------

static inline void select_run(const struct select_plane* plane, uint64_t base) {
    if (plane->is_wide) {
        uint64_t* const positions = (uint64_t*)plane->positions + *plane->count;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            positions[lane] = base + lane;
        }
    } else {
        uint32_t* const positions = (uint32_t*)plane->positions + *plane->count;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            positions[lane] = (uint32_t)(base + lane);
        }
    }
    *plane->count += 64;
}

//--------------------------------------
// Partial Words
//--------------------------------------

#define SELECT_DUAL_WRITE(T)                                                                                           \
    do {                                                                                                               \
        T* const positions = plane->positions;                                                                         \
        uint64_t count = *plane->count;                                                                                \
        for (uint32_t lane = 0; lane < word.lane_count; ++lane) {                                                      \
            positions[count] = (T)(word.base + lane);                                                                  \
            count += bitmap_word_bit(word.lanes, lane);                                                                \
        }                                                                                                              \
        *plane->count = count;                                                                                         \
    } while (0)

static inline void select_dual_write(const struct select_plane* plane, struct select_word word) {
    if (plane->is_wide) {
        SELECT_DUAL_WRITE(uint64_t);
    } else {
        SELECT_DUAL_WRITE(uint32_t);
    }
}

#undef SELECT_DUAL_WRITE

static inline void select_partial(const struct select_plane* plane, struct select_word word) {
#ifdef SELECT_USE_NEON
    if (!plane->is_wide) {
        const uint32_t emitted = (uint32_t)(*plane->count - plane->chunk_first);
        *plane->count += bitmap_word_indices(word.lanes,
                                             (uint32_t)word.base,
                                             emitted >= word.row / 4U,
                                             (uint32_t*)plane->positions + *plane->count);
        return;
    }
#endif
    select_dual_write(plane, word);
}

//--------------------------------------
// Planes
//--------------------------------------

static inline void select_append(const struct select_plane* plane, struct select_word word) {
    if (plane->positions == nullptr) {
        *plane->count += dora_bitmap_word_count_ones(word.lanes);
    } else if (word.lanes == UINT64_MAX) {
        select_run(plane, word.base);
    } else if (word.lanes != 0) {
        select_partial(plane, word);
    }
}

#ifdef SELECT_USE_NEON
uint32_t dora_kernel_select_padding(bool is_wide) {
    return is_wide ? SELECT_DUAL_WRITE_PADDING : SELECT_TABLE_PADDING;
}
#else
uint32_t dora_kernel_select_padding(bool) {
    return SELECT_DUAL_WRITE_PADDING;
}
#endif

void dora_kernel_select_emit(const uint64_t* matches,
                             const uint64_t* validity,
                             uint32_t count,
                             uint64_t logical_first,
                             struct kernel_select_output* in_out_output) {
    const struct select_plane planes[3] = {
        {
            .positions = in_out_output->out_true,
            .count = &in_out_output->true_count,
            .chunk_first = in_out_output->true_count,
            .is_wide = in_out_output->is_wide,
        },
        {
            .positions = in_out_output->out_false,
            .count = &in_out_output->false_count,
            .chunk_first = in_out_output->false_count,
            .is_wide = in_out_output->is_wide,
        },
        {
            .positions = in_out_output->out_null,
            .count = &in_out_output->null_count,
            .chunk_first = in_out_output->null_count,
            .is_wide = in_out_output->is_wide,
        },
    };
    const struct select_source source = {.matches = matches, .validity = validity};
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    const uint64_t last_tail = bitmap_tail_mask(count);
    for (uint32_t word = 0; word < word_count; ++word) {
        const uint64_t tail = word + 1 == word_count ? last_tail : UINT64_MAX;
        const uint64_t match = source.matches[word] & tail;
        const uint64_t valid = source.validity == nullptr ? tail : source.validity[word] & tail;
        const struct select_word shape = {
            .base = logical_first + (uint64_t)word * 64,
            .row = word * 64,
            .lane_count = word + 1 == word_count ? count - word * 64 : 64,
        };
        const uint64_t lanes[3] = {match & valid, ~match & valid, ~valid & tail};
        for (uint32_t plane = 0; plane < 3; ++plane) {
            struct select_word plane_word = shape;
            plane_word.lanes = lanes[plane];
            select_append(&planes[plane], plane_word);
        }
    }
}

//--------------------------------------
// Mask Words
//--------------------------------------

void dora_kernel_select_mask_words(const struct kernel_packet* mask, uint64_t* out_matches) {
    const size_t word_count = (size_t)bitmap_words_for_bits(mask->count);
    switch (mask->shape) {
    case KERNEL_SHAPE_FLAT:
        dora_bitmap_load_words(mask->values, mask->value_offset, mask->count, out_matches);
        break;
    case KERNEL_SHAPE_CONSTANT:
        dora_bitmap_set_range(out_matches,
                              0,
                              mask->count,
                              (mask->validity == nullptr || bitmap_word_bit(mask->validity[0], 0) != 0) &&
                                  dora_bitmap_test(mask->values, mask->value_offset));
        break;
    case KERNEL_SHAPE_GATHERED: {
        const uint64_t* active = mask->validity;
        if (active == nullptr) {
            dora_bitmap_set_range(out_matches, 0, (uint64_t)word_count * 64, true);
            active = out_matches;
        }
        dora_bitmap_gather_words_u32(out_matches, mask->values, mask->value_offset, mask->rows, active, mask->count);
        break;
    }
    }
    if (mask->validity != nullptr) {
        dora_bitmap_and_aligned_words(out_matches, mask->validity, word_count);
    }
}
