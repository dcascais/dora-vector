// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Packet row and bitmap access.

#ifndef DORA_KERNEL_CPU_PACKET_BITS_H
#define DORA_KERNEL_CPU_PACKET_BITS_H

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "kernel/packet.h"

#include <stdint.h>

enum kernel_shape_pair : uint8_t {
    KERNEL_PAIR_FLAT_FLAT,
    KERNEL_PAIR_FLAT_CONSTANT,
    KERNEL_PAIR_CONSTANT_FLAT,
    KERNEL_PAIR_CONSTANT_CONSTANT,
    KERNEL_PAIR_GENERIC
};

static inline enum kernel_shape_pair kernel_shape_pair_of(enum kernel_shape left, enum kernel_shape right) {
    const uint32_t combined = ((uint32_t)left << 8U) | (uint32_t)right;
    switch (combined) {
    case ((uint32_t)KERNEL_SHAPE_FLAT << 8U) | (uint32_t)KERNEL_SHAPE_FLAT:
        return KERNEL_PAIR_FLAT_FLAT;
    case ((uint32_t)KERNEL_SHAPE_FLAT << 8U) | (uint32_t)KERNEL_SHAPE_CONSTANT:
        return KERNEL_PAIR_FLAT_CONSTANT;
    case ((uint32_t)KERNEL_SHAPE_CONSTANT << 8U) | (uint32_t)KERNEL_SHAPE_FLAT:
        return KERNEL_PAIR_CONSTANT_FLAT;
    case ((uint32_t)KERNEL_SHAPE_CONSTANT << 8U) | (uint32_t)KERNEL_SHAPE_CONSTANT:
        return KERNEL_PAIR_CONSTANT_CONSTANT;
    default:
        return KERNEL_PAIR_GENERIC;
    }
}

// Requires rows for gathered packets; otherwise mask is UINT32_MAX for flat packets, zero for constants.
static inline uint64_t kernel_row_index(uint64_t offset, const uint32_t* rows, uint32_t mask, uint32_t row) {
    return offset + (rows != nullptr ? rows[row] : row & mask);
}

static inline bool kernel_packet_is_valid_constant(const struct kernel_packet* packet) {
    return packet->shape == KERNEL_SHAPE_CONSTANT && packet->count != 0 &&
           (packet->validity == nullptr || bitmap_word_bit(packet->validity[0], 0) != 0);
}

static inline uint64_t kernel_packet_bits_word(const struct kernel_packet* packet, uint32_t word, uint64_t active) {
    if (active == 0) {
        return 0;
    }
    switch (packet->shape) {
    case KERNEL_SHAPE_FLAT: {
        const uint32_t lanes = packet->count - word * 64 < 64 ? packet->count - word * 64 : 64;
        return active & dora_bitmap_load_window(packet->values, packet->value_offset + (uint64_t)word * 64, lanes);
    }
    case KERNEL_SHAPE_CONSTANT:
        return dora_bitmap_test(packet->values, packet->value_offset) ? active : 0;
    case KERNEL_SHAPE_GATHERED: {
        uint64_t value = 0;
        while (active != 0) {
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&active);
            if (dora_bitmap_test(packet->values, packet->value_offset + packet->rows[word * 64 + lane])) {
                bitmap_set(&value, lane);
            }
        }
        return value;
    }
    }
    return 0;
}

#endif
