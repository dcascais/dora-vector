// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Kernel packets and results.

#ifndef DORA_KERNEL_PACKET_H
#define DORA_KERNEL_PACKET_H

#include "dora/vector/device.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Chunk Geometry
//
//------------------------------------------------------------------------------

enum : uint32_t {
    KERNEL_CHUNK_ROW_COUNT = 2048,
    KERNEL_CHUNK_WORD_COUNT = 32
};

//------------------------------------------------------------------------------
//
// Packets
//
//------------------------------------------------------------------------------

enum kernel_shape : uint8_t {
    KERNEL_SHAPE_FLAT,
    KERNEL_SHAPE_CONSTANT,
    KERNEL_SHAPE_GATHERED
};

struct kernel_packet {
    const void* values;
    const uint64_t* validity;
    const uint32_t* rows;
    uint64_t value_offset;
    uint32_t count;
    enum kernel_shape shape;
    struct dora_placement placement;
};

//------------------------------------------------------------------------------
//
// Results
//
//------------------------------------------------------------------------------

struct kernel_result {
    void* values;
    uint64_t* validity;
    uint32_t row_capacity;
    uint32_t null_count;
    uint8_t failure_bits;
};

struct kernel_text_output {
    int64_t* end_offsets;
    uint8_t* data;
    int64_t written_byte_count;
};

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(KERNEL_CHUNK_ROW_COUNT == KERNEL_CHUNK_WORD_COUNT * 64);

#endif
