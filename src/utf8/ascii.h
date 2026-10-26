// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// ASCII classification and scanning.

#ifndef DORA_UTF8_ASCII_H
#define DORA_UTF8_ASCII_H

#include "bitmap/bitmap_internal.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr uint64_t UTF8_ASCII_MASK = UINT64_C(0x8080'8080'8080'8080);
static constexpr uint32_t UTF8_ASCII_MASK_32 = UINT32_C(0x8080'8080);
static constexpr uint16_t UTF8_ASCII_MASK_16 = UINT16_C(0x8080);

static inline size_t utf8_ascii_prefix(struct dora_bytes bytes) {
    const size_t word_bytes = bytes.byte_count - bytes.byte_count % sizeof(uint64_t);
    size_t position = 0;
    while (position < word_bytes) {
        uint64_t word;
        memcpy(&word, bytes.data + position, 8);
        if ((word & UTF8_ASCII_MASK) != 0) {
            return position + bitmap_word_trailing_zeros(word & UTF8_ASCII_MASK) / 8;
        }
        position += 8;
    }
    while (position < bytes.byte_count && bytes.data[position] < 0b1000'0000) {
        ++position;
    }
    return position;
}

static inline bool utf8_short_span_is_ascii(const uint8_t* bytes, size_t byte_count) {
    if (byte_count >= sizeof(uint64_t)) {
        uint64_t first = 0;
        uint64_t last = 0;
        memcpy(&first, bytes, sizeof first);
        memcpy(&last, bytes + byte_count - sizeof last, sizeof last);
        return ((first | last) & UTF8_ASCII_MASK) == 0;
    }
    if (byte_count >= sizeof(uint32_t)) {
        uint32_t first = 0;
        uint32_t last = 0;
        memcpy(&first, bytes, sizeof first);
        memcpy(&last, bytes + byte_count - sizeof last, sizeof last);
        return ((first | last) & UTF8_ASCII_MASK_32) == 0;
    }
    if (byte_count >= sizeof(uint16_t)) {
        uint16_t first = 0;
        uint16_t last = 0;
        memcpy(&first, bytes, sizeof first);
        memcpy(&last, bytes + byte_count - sizeof last, sizeof last);
        return (((uint32_t)first | (uint32_t)last) & (uint32_t)UTF8_ASCII_MASK_16) == 0;
    }
    return byte_count == 0 || (bytes[0] & 0b1000'0000U) == 0;
}

static inline bool utf8_span_is_ascii(const uint8_t* bytes, size_t byte_count) {
    if (byte_count < 16) {
        return utf8_short_span_is_ascii(bytes, byte_count);
    }
    size_t position = 0;
    while (byte_count - position >= 16) {
        uint64_t first = 0;
        uint64_t second = 0;
        memcpy(&first, bytes + position, sizeof first);
        memcpy(&second, bytes + position + sizeof first, sizeof second);
        if (((first | second) & UTF8_ASCII_MASK) != 0) {
            return false;
        }
        position += 16;
    }
    if (byte_count - position >= sizeof(uint64_t)) {
        uint64_t tail = 0;
        memcpy(&tail, bytes + position, sizeof tail);
        if ((tail & UTF8_ASCII_MASK) != 0) {
            return false;
        }
        position += sizeof tail;
    }
    while (position < byte_count) {
        if ((bytes[position] & 0b1000'0000U) != 0) {
            return false;
        }
        position++;
    }
    return true;
}

bool dora_utf8_ascii_is_text_space(char value);

#endif
