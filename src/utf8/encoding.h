// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 scalar encoding and decoding.

#ifndef DORA_UTF8_ENCODING_H
#define DORA_UTF8_ENCODING_H

#include "dora/vector/types.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

extern const uint16_t dora_utf8_scalar_transitions[];

static inline uint16_t utf8_scalar_next_state(uint8_t byte, uint16_t state) {
    return dora_utf8_scalar_transitions[state + byte];
}

static inline int utf8_read_scalar(struct dora_bytes bytes, uint32_t* out_scalar, size_t* out_byte_count) {
    if (bytes.byte_count == 0) {
        return EILSEQ;
    }
    uint16_t state = utf8_scalar_next_state(bytes.data[0], 0);
    if (state == 256) {
        return EILSEQ;
    }
    uint32_t scalar = bytes.data[0];
    if (state != 0) {
        scalar &= state == 512 ? 0b0001'1111U : state <= 1280 ? 0b0000'1111U : 0b0000'0111U;
    }
    size_t position = 1;
    while (state != 0) {
        if (position == bytes.byte_count) {
            return EILSEQ;
        }
        const uint8_t byte = bytes.data[position++];
        state = utf8_scalar_next_state(byte, state);
        if (state == 256) {
            return EILSEQ;
        }
        scalar = (scalar << 6U) | (byte & 0b0011'1111U);
    }
    *out_scalar = scalar;
    *out_byte_count = position;
    return 0;
}

#endif
