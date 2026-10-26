// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 span search.

#ifndef DORA_UTF8_SPAN_H
#define DORA_UTF8_SPAN_H

#include "dora/vector/bitmap.h"
#include "dora/vector/types.h"
#include "utf8/ascii.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline int utf8_prove_ascii(struct dora_utf8_input* in_out_input) {
    if (in_out_input->proof != DORA_UTF8_CHECKED) {
        return 0;
    }
    bool ascii;
    const int status = dora_utf8_validate_span(in_out_input->bytes, &ascii);
    if (status == 0) {
        in_out_input->proof = ascii ? DORA_UTF8_VALIDATED_ASCII : DORA_UTF8_VALIDATED;
    }
    return status;
}

static inline size_t utf8_count_word(uint64_t word) {
    return 8 - dora_bitmap_word_count_ones((word & ~(word << 1U)) & UTF8_ASCII_MASK);
}

static inline size_t utf8_count_valid(struct dora_utf8_input input) {
    if (input.proof == DORA_UTF8_VALIDATED_ASCII ||
        (input.bytes.byte_count < 16 && utf8_short_span_is_ascii(input.bytes.data, input.bytes.byte_count))) {
        return input.bytes.byte_count;
    }
    size_t count = utf8_ascii_prefix(input.bytes), position = count;
    if (input.bytes.byte_count - position >= 128) {
        while (input.bytes.byte_count - position >= 8) {
            uint64_t word;
            memcpy(&word, input.bytes.data + position, 8);
            count += utf8_count_word(word);
            position += 8;
        }
    }
    for (; position < input.bytes.byte_count; ++position) {
        count += !utf8_byte_is_continuation(input.bytes.data[position]);
    }
    return count;
}

static inline size_t utf8_scalar_position(size_t byte_offset, struct dora_bytes bytes, size_t scalar_count) {
    while (byte_offset <= bytes.byte_count && bytes.byte_count - byte_offset >= 16) {
        uint64_t first, second;
        memcpy(&first, bytes.data + byte_offset, 8);
        memcpy(&second, bytes.data + byte_offset + 8, 8);
        if (((first | second) & UTF8_ASCII_MASK) == 0) {
            if (scalar_count < 16) {
                return byte_offset + scalar_count;
            }
            scalar_count -= 16;
        } else {
            const size_t scalars = utf8_count_word(first) + utf8_count_word(second);
            if (scalar_count < scalars) {
                break;
            }
            scalar_count -= scalars;
        }
        byte_offset += 16;
    }
    if (bytes.byte_count - byte_offset >= 8) {
        uint64_t word;
        memcpy(&word, bytes.data + byte_offset, 8);
        if ((word & UTF8_ASCII_MASK) == 0) {
            if (scalar_count < 8) {
                return byte_offset + scalar_count;
            }
            scalar_count -= 8;
            byte_offset += 8;
        } else {
            const size_t scalars = utf8_count_word(word);
            if (scalar_count >= scalars) {
                scalar_count -= scalars;
                byte_offset += 8;
            }
        }
    }
    for (; byte_offset < bytes.byte_count; ++byte_offset) {
        if (!utf8_byte_is_continuation(bytes.data[byte_offset])) {
            if (scalar_count == 0) {
                return byte_offset;
            }
            --scalar_count;
        }
    }
    return bytes.byte_count;
}

static inline struct dora_range
utf8_locate_valid(enum dora_utf8_proof proof, struct dora_bytes bytes, size_t start_scalar_index, size_t scalar_count) {
    if (proof == DORA_UTF8_VALIDATED_ASCII ||
        (bytes.byte_count < 16 && utf8_span_is_ascii(bytes.data, bytes.byte_count))) {
        const size_t begin = start_scalar_index < bytes.byte_count ? start_scalar_index : bytes.byte_count;
        return (struct dora_range){
            begin,
            scalar_count < bytes.byte_count - begin ? scalar_count : bytes.byte_count - begin
        };
    }
    const size_t begin = utf8_scalar_position(0, bytes, start_scalar_index);
    const size_t end = utf8_scalar_position(begin, bytes, scalar_count);
    return (struct dora_range){begin, end - begin};
}

#endif
