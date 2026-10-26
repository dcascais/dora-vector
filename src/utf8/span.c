// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "utf8/span.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "types/types_internal.h"
#include "utf8/ascii.h"
#include "utf8/encoding.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Span Validation
//
//------------------------------------------------------------------------------

//--------------------------------------
// Scalar
//--------------------------------------

static inline bool utf8_span_is_valid_scalar_core(const uint8_t* bytes, size_t byte_count) {
    if (bytes == nullptr) {
        return byte_count == 0;
    }
    while (byte_count >= sizeof(uint64_t)) {
        uint64_t word = 0;
        memcpy(&word, bytes, sizeof word);
        if ((word & UTF8_ASCII_MASK) == 0) {
            bytes += sizeof word;
            byte_count -= sizeof word;
            continue;
        }
        uint16_t state = 0;
        state = utf8_scalar_next_state((uint8_t)(word >> 0U), state);
        ++bytes;
        byte_count--;
        state = utf8_scalar_next_state((uint8_t)(word >> 8U), state);
        ++bytes;
        byte_count--;
        state = utf8_scalar_next_state((uint8_t)(word >> 16U), state);
        ++bytes;
        byte_count--;
        state = utf8_scalar_next_state((uint8_t)(word >> 24U), state);
        ++bytes;
        byte_count--;
        state = utf8_scalar_next_state((uint8_t)(word >> 32U), state);
        ++bytes;
        byte_count--;
        if (state == 0) {
            continue;
        }
        state = utf8_scalar_next_state((uint8_t)(word >> 40U), state);
        ++bytes;
        byte_count--;
        if (state == 0) {
            continue;
        }
        state = utf8_scalar_next_state((uint8_t)(word >> 48U), state);
        ++bytes;
        byte_count--;
        if (state == 0) {
            continue;
        }
        state = utf8_scalar_next_state((uint8_t)(word >> 56U), state);
        ++bytes;
        byte_count--;
        if (state == 0) {
            continue;
        }
        return false;
    }
    if (utf8_short_span_is_ascii(bytes, byte_count)) {
        return true;
    }
    uint16_t state = 0;
    switch (byte_count) {
    case 7:
        state = utf8_scalar_next_state(bytes[byte_count - 7], state);
        [[fallthrough]];
    case 6:
        state = utf8_scalar_next_state(bytes[byte_count - 6], state);
        [[fallthrough]];
    case 5:
        state = utf8_scalar_next_state(bytes[byte_count - 5], state);
        [[fallthrough]];
    case 4:
        state = utf8_scalar_next_state(bytes[byte_count - 4], state);
        [[fallthrough]];
    case 3:
        state = utf8_scalar_next_state(bytes[byte_count - 3], state);
        [[fallthrough]];
    case 2:
        state = utf8_scalar_next_state(bytes[byte_count - 2], state);
        [[fallthrough]];
    case 1:
        state = utf8_scalar_next_state(bytes[byte_count - 1], state);
        [[fallthrough]];
    default:
        break;
    }
    return state == 0;
}

static inline bool utf8_span_is_valid_scalar(const uint8_t* bytes, size_t byte_count, bool* out_all_ascii) {
    if (out_all_ascii == nullptr || bytes == nullptr) {
        return utf8_span_is_valid_scalar_core(bytes, byte_count);
    }
    const size_t prefix = utf8_ascii_prefix((struct dora_bytes){bytes, byte_count});
    if (prefix == byte_count) {
        return true;
    }
    const bool valid = utf8_span_is_valid_scalar_core(bytes + prefix, byte_count - prefix);
    *out_all_ascii = false;
    return valid;
}

//--------------------------------------
// Dispatch
//--------------------------------------

int dora_utf8_validate_span(struct dora_bytes bytes, bool* out_all_ascii) {
    if (bytes.byte_count != 0 && bytes.data == nullptr) {
        return EINVAL;
    }
    if (bytes.byte_count < 16) {
        const bool ascii = utf8_short_span_is_ascii(bytes.data, bytes.byte_count);
        const bool valid = ascii || utf8_span_is_valid_scalar_core(bytes.data, bytes.byte_count);
        if (out_all_ascii != nullptr) {
            *out_all_ascii = ascii;
        }
        return valid ? 0 : EILSEQ;
    }
    bool ascii = true;
    const bool valid =
        utf8_span_is_valid_scalar(bytes.data, bytes.byte_count, out_all_ascii != nullptr ? &ascii : nullptr);
    if (out_all_ascii != nullptr) {
        *out_all_ascii = valid && ascii;
    }
    return valid ? 0 : EILSEQ;
}

static void
utf8_prefix_scan(struct dora_bytes bytes, bool complete, size_t* out_count, enum dora_utf8_prefix_stop* out_stop) {
    uint16_t state = 0;
    size_t prefix = 0;
    for (size_t index = 0; index < bytes.byte_count; ++index) {
        state = utf8_scalar_next_state(bytes.data[index], state);
        if (state == 256) {
            *out_count = prefix;
            *out_stop = DORA_UTF8_PREFIX_INVALID;
            return;
        }
        if (state == 0) {
            prefix = index + 1;
        }
    }
    *out_count = prefix;
    *out_stop = !complete ? DORA_UTF8_PREFIX_LIMIT : state != 0 ? DORA_UTF8_PREFIX_INVALID : DORA_UTF8_PREFIX_END;
}

int dora_utf8_copy_cstr_prefix(const char* source,
                               struct dora_mutable_bytes output,
                               size_t* out_byte_count,
                               enum dora_utf8_prefix_stop* out_stop) {
    if (!utf8_output_is_valid(output) || out_byte_count == nullptr || out_stop == nullptr) {
        return EINVAL;
    }
    if (output.byte_count == 0) {
        return ENOBUFS;
    }
    if (source == nullptr) {
        output.data[0] = 0;
        *out_byte_count = 0;
        *out_stop = DORA_UTF8_PREFIX_END;
        return 0;
    }
    const size_t limit = output.byte_count - 1;
    size_t examined = 0;
    bool complete = false;
    while (examined < limit) {
        if (source[examined] == '\0') {
            complete = true;
            break;
        }
        ++examined;
    }
    size_t size;
    enum dora_utf8_prefix_stop stop;
    utf8_prefix_scan((struct dora_bytes){(const uint8_t*)source, examined}, complete, &size, &stop);
    if (size != 0) {
        memmove(output.data, source, size);
    }
    output.data[size] = 0;
    *out_byte_count = size;
    *out_stop = stop;
    return 0;
}

int dora_utf8_bom_prefix(struct dora_bytes bytes, bool input_complete, size_t* out_byte_count) {
    if (!bytes_are_well_formed(bytes) || out_byte_count == nullptr) {
        return EINVAL;
    }
    static constexpr uint8_t bom[] = {0xEF, 0xBB, 0xBF};
    size_t i = 0;
    while (i < sizeof bom && i < bytes.byte_count) {
        if (bytes.data[i] != bom[i]) {
            *out_byte_count = 0;
            return 0;
        }
        ++i;
    }
    if (i < sizeof bom && !input_complete) {
        return EAGAIN;
    }
    *out_byte_count = i == sizeof bom ? sizeof bom : 0;
    return 0;
}

static int utf8_bound_decide(enum dora_utf8_bound bound,
                             struct dora_utf8_input input,
                             size_t budget,
                             size_t* out_size,
                             bool* out_available,
                             size_t* out_prefix,
                             bool* out_exact,
                             uint8_t tail[static 4],
                             size_t* out_tail_size) {
    if (!utf8_input_is_valid(input) || (bound != DORA_UTF8_BOUND_LOWER && bound != DORA_UTF8_BOUND_UPPER)) {
        return EINVAL;
    }
    if (input.proof == DORA_UTF8_CHECKED) {
        const int status = dora_utf8_validate_span(input.bytes, nullptr);
        if (status != 0) {
            return status;
        }
    }
    *out_tail_size = 0;
    if (input.bytes.byte_count <= budget) {
        *out_size = input.bytes.byte_count;
        *out_prefix = input.bytes.byte_count;
        *out_available = true;
        *out_exact = true;
        return 0;
    }
    *out_exact = false;
    size_t prefix = 0;
    enum dora_utf8_prefix_stop stop;
    utf8_prefix_scan((struct dora_bytes){input.bytes.data, budget}, false, &prefix, &stop);
    if (bound == DORA_UTF8_BOUND_LOWER) {
        *out_prefix = prefix;
        if (prefix < budget) {
            tail[0] = 0;
            *out_tail_size = 1;
        }
        *out_size = prefix + *out_tail_size;
        *out_available = true;
        return 0;
    }
    while (prefix != 0) {
        size_t begin = prefix - 1;
        while (begin != 0 && utf8_byte_is_continuation(input.bytes.data[begin])) {
            --begin;
        }
        uint32_t scalar = 0;
        size_t size = 0;
        const int status =
            utf8_read_scalar((struct dora_bytes){input.bytes.data + begin, prefix - begin}, &scalar, &size);
        if (status == 0 && scalar < 0x10'FFFF) {
            scalar = scalar == 0xD7FF ? 0xE000 : scalar + 1;
            const int encode_status = utf8_scalar_bytes(scalar, tail, &size);
            if (encode_status != 0) {
                return encode_status;
            }
            if (size <= budget - begin) {
                *out_prefix = begin;
                *out_tail_size = size;
                *out_size = begin + size;
                *out_available = true;
                return 0;
            }
        }
        prefix = begin;
    }
    *out_prefix = 0;
    *out_size = 0;
    *out_available = false;
    return 0;
}

int dora_utf8_measure_bound(struct dora_utf8_input input,
                            enum dora_utf8_bound bound,
                            size_t budget,
                            size_t* out_byte_count,
                            bool* out_available,
                            bool* out_exact) {
    if (out_byte_count == nullptr || out_available == nullptr || out_exact == nullptr) {
        return EINVAL;
    }
    uint8_t tail[4];
    size_t size, prefix, tail_size;
    bool available, exact;
    const int status = utf8_bound_decide(bound, input, budget, &size, &available, &prefix, &exact, tail, &tail_size);
    if (status != 0) {
        return status;
    }
    *out_byte_count = size;
    *out_available = available;
    *out_exact = exact;
    return 0;
}

int dora_utf8_make_bound(struct dora_utf8_input input,
                         enum dora_utf8_bound bound,
                         struct dora_mutable_bytes output,
                         size_t* out_byte_count,
                         bool* out_available,
                         bool* out_exact) {
    if (!utf8_output_is_valid(output) || out_byte_count == nullptr || out_available == nullptr ||
        out_exact == nullptr) {
        return EINVAL;
    }
    uint8_t tail[4];
    size_t size, prefix, tail_size;
    bool available, exact;
    const int status =
        utf8_bound_decide(bound, input, output.byte_count, &size, &available, &prefix, &exact, tail, &tail_size);
    if (status != 0) {
        return status;
    }
    if (available) {
        if (prefix != 0) {
            memcpy(output.data, input.bytes.data, prefix);
        }
        if (tail_size != 0) {
            memcpy(output.data + prefix, tail, tail_size);
        }
    }
    *out_byte_count = size;
    *out_available = available;
    *out_exact = exact;
    return 0;
}

int dora_utf8_count(struct dora_utf8_input input, size_t* out_scalar_count) {
    if (!utf8_input_is_valid(input) || out_scalar_count == nullptr) {
        return EINVAL;
    }
    const int status = utf8_prove_ascii(&input);
    if (status != 0) {
        return status;
    }
    *out_scalar_count = utf8_count_valid(input);
    return 0;
}

int dora_utf8_locate(struct dora_utf8_input input,
                     size_t start_scalar_index,
                     size_t scalar_count,
                     struct dora_range* out_byte_range) {
    if (!utf8_input_is_valid(input) || out_byte_range == nullptr) {
        return EINVAL;
    }
    const int status = utf8_prove_ascii(&input);
    if (status != 0) {
        return status;
    }
    *out_byte_range = utf8_locate_valid(input.proof, input.bytes, start_scalar_index, scalar_count);
    return 0;
}

int dora_utf8_locate_tail(uint64_t scalar_distance,
                          struct dora_utf8_input input,
                          size_t scalar_count,
                          struct dora_range* out_byte_range) {
    if (!utf8_input_is_valid(input) || out_byte_range == nullptr) {
        return EINVAL;
    }
    const int status = utf8_prove_ascii(&input);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_bytes bytes = input.bytes;
    if (input.proof == DORA_UTF8_VALIDATED_ASCII) {
        const size_t start = scalar_distance >= bytes.byte_count ? 0 : bytes.byte_count - (size_t)scalar_distance;
        *out_byte_range = utf8_locate_valid(input.proof, bytes, start, scalar_count);
        return DORA_OK;
    }
    size_t start = bytes.byte_count;
    while (scalar_distance != 0 && start != 0) {
        --start;
        while (start != 0 && utf8_byte_is_continuation(bytes.data[start])) {
            --start;
        }
        --scalar_distance;
    }
    const size_t end = utf8_scalar_position(start, bytes, scalar_count);
    *out_byte_range = (struct dora_range){start, end - start};
    return DORA_OK;
}

static bool utf8_candidate_matches(const uint8_t* bytes, struct dora_bytes needle) {
    return bytes[0] == needle.data[0] && bytes[needle.byte_count - 1] == needle.data[needle.byte_count - 1] &&
           (needle.byte_count == 2 || memcmp(bytes + 1, needle.data + 1, needle.byte_count - 2) == 0);
}

// Overlapping endpoint words cover needles of 2-16 bytes.
#define DEFINE_UTF8_FIND_WORDS(NAME, TYPE)                                                                             \
    static inline bool utf8_candidate_words_match_##NAME(const uint8_t* value,                                         \
                                                         size_t needle_size,                                           \
                                                         TYPE first,                                                   \
                                                         TYPE last) {                                                  \
        TYPE first_value;                                                                                              \
        TYPE last_value;                                                                                               \
        memcpy(&first_value, value, sizeof first_value);                                                               \
        memcpy(&last_value, value + needle_size - sizeof last_value, sizeof last_value);                               \
        return first_value == first && last_value == last;                                                             \
    }                                                                                                                  \
    static size_t utf8_find_##NAME(struct dora_bytes input, struct dora_bytes needle, size_t start) {                  \
        TYPE first;                                                                                                    \
        TYPE last;                                                                                                     \
        memcpy(&first, needle.data, sizeof first);                                                                     \
        memcpy(&last, needle.data + needle.byte_count - sizeof last, sizeof last);                                     \
        const size_t last_position = input.byte_count - needle.byte_count;                                             \
        size_t position = start;                                                                                       \
        while (position <= last_position) {                                                                            \
            const uint8_t* const match = memchr(input.data + position, needle.data[0], last_position - position + 1);  \
            if (match == nullptr) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            position = (size_t)(match - input.data);                                                                   \
            if (utf8_candidate_words_match_##NAME(match, needle.byte_count, first, last)) {                            \
                return position;                                                                                       \
            }                                                                                                          \
            ++position;                                                                                                \
        }                                                                                                              \
        return input.byte_count;                                                                                       \
    }
DEFINE_UTF8_FIND_WORDS(u16, uint16_t)
DEFINE_UTF8_FIND_WORDS(u32, uint32_t)
DEFINE_UTF8_FIND_WORDS(u64, uint64_t)
#undef DEFINE_UTF8_FIND_WORDS

static size_t utf8_find_generic(struct dora_bytes input, struct dora_bytes needle, size_t start) {
    const size_t last_position = input.byte_count - needle.byte_count;
    size_t position = start;
    while (position <= last_position) {
        const uint8_t* const match = memchr(input.data + position, needle.data[0], last_position - position + 1);
        if (match == nullptr) {
            break;
        }
        position = (size_t)(match - input.data);
        if (utf8_candidate_matches(match, needle)) {
            return position;
        }
        ++position;
    }
    return input.byte_count;
}

size_t dora_utf8_find_valid(struct dora_bytes input, struct dora_bytes needle, size_t start_byte_offset) {
    if (needle.byte_count == 0) {
        return start_byte_offset;
    }
    if (needle.byte_count > input.byte_count - start_byte_offset) {
        return input.byte_count;
    }
    switch (needle.byte_count) {
    case 1: {
        const uint8_t* const match =
            memchr(input.data + start_byte_offset, needle.data[0], input.byte_count - start_byte_offset);
        return match == nullptr ? input.byte_count : (size_t)(match - input.data);
    }
    case 2:
    case 3:
        return utf8_find_u16(input, needle, start_byte_offset);
    case 4:
    case 5:
    case 6:
    case 7:
        return utf8_find_u32(input, needle, start_byte_offset);
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
        return utf8_find_u64(input, needle, start_byte_offset);
    default:
        return utf8_find_generic(input, needle, start_byte_offset);
    }
}

int dora_utf8_find(struct dora_utf8_input input,
                   struct dora_utf8_input needle,
                   size_t start_byte_offset,
                   size_t* out_byte_offset,
                   bool* out_found) {
    if (!utf8_input_is_valid(input) || !utf8_input_is_valid(needle) || start_byte_offset > input.bytes.byte_count ||
        out_byte_offset == nullptr || out_found == nullptr) {
        return EINVAL;
    }
    int status = utf8_check_input(input);
    if (status == 0) {
        status = utf8_check_input(needle);
    }
    if (status != 0) {
        return status;
    }
    if (start_byte_offset < input.bytes.byte_count && utf8_byte_is_continuation(input.bytes.data[start_byte_offset])) {
        return EINVAL;
    }
    *out_byte_offset = dora_utf8_find_valid(input.bytes, needle.bytes, start_byte_offset);
    *out_found = needle.bytes.byte_count == 0 || *out_byte_offset < input.bytes.byte_count;
    return 0;
}

int dora_utf8_byte_matcher_init(const uint64_t set[4], struct dora_utf8_byte_matcher* out_matcher) {
    if (set == nullptr || out_matcher == nullptr) {
        return EINVAL;
    }
    struct dora_utf8_byte_matcher matcher = {};
    bitmap_copy_bytes((uint8_t*)matcher.words, (const uint8_t*)set, sizeof matcher.words);
    uint32_t count = 0;
    for (size_t w = 0; w < 4; ++w) {
        count += dora_bitmap_word_count_ones(set[w]);
    }
    if (count == 0) {
        matcher.kind = DORA_UTF8_MATCHER_EMPTY;
    } else if (count == 256) {
        matcher.kind = DORA_UTF8_MATCHER_FULL;
    } else if (count <= 5) {
        matcher.kind = DORA_UTF8_MATCHER_SMALL;
        uint8_t size = 0;
        for (uint32_t index = 0; index < 4; ++index) {
            uint64_t word = matcher.words[index];
            while (word != 0) {
                const uint32_t bit = dora_bitmap_word_pop_lowest_set(&word);
                matcher.small_bytes[size++] = (uint8_t)(index * 64U + bit);
            }
        }
        matcher.small_byte_count = size;
    } else {
        uint64_t json[4];
        bitmap_fill_bytes((uint8_t*)json, sizeof json, false);
        dora_bitmap_set_range(json, 0, 32, true);
        bitmap_set(json, '"');
        bitmap_set(json, '\\');
        matcher.kind = bitmap_words_are_equal(json, set, 4) ? DORA_UTF8_MATCHER_JSON : DORA_UTF8_MATCHER_BITS;
    }
    *out_matcher = matcher;
    return 0;
}

static inline uint64_t utf8_event_word_matches(uint64_t word, uint64_t repeated) {
    const uint64_t value = word ^ repeated;
    const uint64_t low = UINT64_C(0x7F7F7F7F7F7F7F7F);
    return ~(((value & low) + low) | value) & UTF8_ASCII_MASK;
}

static inline bool utf8_byte_matches_3(uint8_t byte, const struct dora_utf8_byte_matcher* matcher) {
    return byte == matcher->small_bytes[0] || byte == matcher->small_bytes[1] || byte == matcher->small_bytes[2];
}

static inline bool utf8_byte_matches_5(uint8_t byte, const struct dora_utf8_byte_matcher* matcher) {
    return byte == matcher->small_bytes[0] || byte == matcher->small_bytes[1] || byte == matcher->small_bytes[2] ||
           byte == matcher->small_bytes[3] || byte == matcher->small_bytes[4];
}

#define DEFINE_UTF8_EVENT_SCAN(SIZE)                                                                                   \
    static size_t utf8_find_event_##SIZE(struct dora_bytes bytes,                                                      \
                                         const struct dora_utf8_byte_matcher* matcher,                                 \
                                         size_t position) {                                                            \
        if (position == bytes.byte_count) {                                                                            \
            return bytes.byte_count;                                                                                   \
        }                                                                                                              \
        if (utf8_byte_matches_##SIZE(bytes.data[position], matcher)) {                                                 \
            return position;                                                                                           \
        }                                                                                                              \
        ++position;                                                                                                    \
        if (bytes.byte_count - position >= sizeof(uint64_t)) {                                                         \
            uint64_t repeated[(SIZE)];                                                                                 \
            for (size_t pattern = 0; pattern < (SIZE); ++pattern) {                                                    \
                repeated[pattern] = UINT64_C(0x0101010101010101) * matcher->small_bytes[pattern];                      \
            }                                                                                                          \
            do {                                                                                                       \
                uint64_t word;                                                                                         \
                memcpy(&word, bytes.data + position, sizeof word);                                                     \
                uint64_t flags = 0;                                                                                    \
                for (size_t pattern = 0; pattern < (SIZE); ++pattern) {                                                \
                    flags |= utf8_event_word_matches(word, repeated[pattern]);                                         \
                }                                                                                                      \
                if (flags != 0) {                                                                                      \
                    return position + dora_bitmap_word_pop_lowest_set(&flags) / 8U;                                    \
                }                                                                                                      \
                position += sizeof word;                                                                               \
            } while (bytes.byte_count - position >= sizeof(uint64_t));                                                 \
        }                                                                                                              \
        for (; position < bytes.byte_count; ++position) {                                                              \
            if (utf8_byte_matches_##SIZE(bytes.data[position], matcher)) {                                             \
                return position;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
        return bytes.byte_count;                                                                                       \
    }

DEFINE_UTF8_EVENT_SCAN(3)
DEFINE_UTF8_EVENT_SCAN(5)
#undef DEFINE_UTF8_EVENT_SCAN

static size_t utf8_find_event(struct dora_bytes bytes, const struct dora_utf8_byte_matcher* matcher, size_t start) {
    switch (matcher->kind) {
    case DORA_UTF8_MATCHER_EMPTY:
        return bytes.byte_count;
    case DORA_UTF8_MATCHER_FULL:
        return start;
    case DORA_UTF8_MATCHER_JSON:
        for (size_t i = start; i < bytes.byte_count; ++i) {
            if (bytes.data[i] < 32 || bytes.data[i] == '"' || bytes.data[i] == '\\') {
                return i;
            }
        }
        break;
    case DORA_UTF8_MATCHER_SMALL:
        if (matcher->small_byte_count == 3) {
            return utf8_find_event_3(bytes, matcher, start);
        }
        if (matcher->small_byte_count == 5) {
            return utf8_find_event_5(bytes, matcher, start);
        }
        for (size_t i = start; i < bytes.byte_count; ++i) {
            for (uint8_t n = 0; n < matcher->small_byte_count; ++n) {
                if (bytes.data[i] == matcher->small_bytes[n]) {
                    return i;
                }
            }
        }
        break;
    case DORA_UTF8_MATCHER_BITS:
        for (size_t i = start; i < bytes.byte_count; ++i) {
            if (dora_bitmap_test((const uint8_t*)matcher->words, bytes.data[i])) {
                return i;
            }
        }
        break;
    }
    return bytes.byte_count;
}

int dora_utf8_find_byte_set_general(struct dora_bytes bytes,
                                    const struct dora_utf8_byte_matcher* matcher,
                                    size_t start_byte_offset,
                                    size_t* out_byte_offset) {
    if (!bytes_are_well_formed(bytes) || matcher == nullptr || out_byte_offset == nullptr ||
        start_byte_offset > bytes.byte_count) {
        return EINVAL;
    }
    *out_byte_offset = utf8_find_event(bytes, matcher, start_byte_offset);
    return 0;
}

int dora_utf8_move(struct dora_utf8_input input, struct dora_mutable_bytes output, size_t* out_written) {
    if (!utf8_input_is_valid(input) || !utf8_output_is_valid(output) || out_written == nullptr) {
        return EINVAL;
    }
    int status = utf8_check_input(input);
    return status != 0 ? status : utf8_move_bytes(input.bytes, output, out_written);
}

static bool utf8_scalar_is_trim_space(uint32_t scalar) {
    return scalar == 0x20 || scalar == 0xA0 || scalar == 0x1680 || (scalar >= 0x2000 && scalar <= 0x200A) ||
           scalar == 0x202F || scalar == 0x205F || scalar == 0x3000;
}

static int utf8_trim_member(struct dora_bytes piece,
                            const struct dora_utf8_trim_set* characters,
                            size_t* out_width,
                            bool* out_member) {
    uint32_t scalar = piece.data[0];
    *out_width = 1;
    if (scalar >= 0b1000'0000) {
        const int status = utf8_read_scalar(piece, &scalar, out_width);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (characters == nullptr) {
        *out_member = utf8_scalar_is_trim_space(scalar);
    } else {
        *out_member = false;
        for (size_t index = 0; index < characters->scalar_count; ++index) {
            if (characters->scalars[index] == scalar) {
                *out_member = true;
                break;
            }
        }
    }
    return DORA_OK;
}

static int utf8_trim_short(struct dora_bytes source, struct dora_range* out_range) {
    size_t cursor = 0, begin = 0, end = 0;
    bool seen = false;
    while (cursor < source.byte_count) {
        uint32_t scalar = source.data[cursor];
        size_t width = 1;
        if (scalar >= 0b1000'0000) {
            const int status = utf8_read_scalar((struct dora_bytes){source.data + cursor, source.byte_count - cursor},
                                                &scalar,
                                                &width);
            if (status != DORA_OK) {
                return status;
            }
        }
        if (!utf8_scalar_is_trim_space(scalar)) {
            if (!seen) {
                begin = cursor;
                seen = true;
            }
            end = cursor + width;
        }
        cursor += width;
    }
    *out_range = (struct dora_range){begin, end - begin};
    return DORA_OK;
}

int dora_utf8_trim_range(struct dora_utf8_input input,
                         const struct dora_utf8_trim_set* character_set,
                         struct dora_range* out_range) {
    if (!utf8_input_is_valid(input) || out_range == nullptr ||
        (character_set != nullptr && character_set->scalar_count != 0 && character_set->scalars == nullptr)) {
        return EINVAL;
    }
    if (character_set == nullptr && input.proof == DORA_UTF8_CHECKED && input.bytes.byte_count != 0 &&
        input.bytes.byte_count < 32 && input.bytes.data[0] >= 0b1000'0000) {
        return utf8_trim_short(input.bytes, out_range);
    }
    int status = utf8_check_input(input);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_bytes source = input.bytes;
    const struct dora_utf8_trim_set* const characters = character_set;
    size_t begin = 0, end = source.byte_count;
    while (begin < end) {
        size_t width = 0;
        bool member = false;
        status = utf8_trim_member((struct dora_bytes){source.data + begin, end - begin}, characters, &width, &member);
        if (status != DORA_OK) {
            return status;
        }
        if (!member) {
            break;
        }
        begin += width;
    }
    while (end > begin) {
        size_t start = end - 1;
        while (start > begin && utf8_byte_is_continuation(source.data[start])) {
            --start;
        }
        size_t width = 0;
        bool member = false;
        status = utf8_trim_member((struct dora_bytes){source.data + start, end - start}, characters, &width, &member);
        if (status != DORA_OK) {
            return status;
        }
        if (!member) {
            break;
        }
        end = start;
    }
    *out_range = begin == end ? (struct dora_range){} : (struct dora_range){begin, end - begin};
    return DORA_OK;
}

int dora_utf8_first_scalar(struct dora_utf8_input input, int32_t* out_scalar) {
    if (!utf8_input_is_valid(input) || out_scalar == nullptr) {
        return EINVAL;
    }
    const struct dora_bytes piece = input.bytes;
    const int checked = utf8_check_input(input);
    if (checked != DORA_OK) {
        return checked;
    }
    if (piece.byte_count == 0) {
        *out_scalar = -1;
        return DORA_OK;
    }
    if (input.proof == DORA_UTF8_VALIDATED_ASCII) {
        *out_scalar = piece.data[0];
        return DORA_OK;
    }
    uint32_t scalar = 0;
    size_t consumed = 0;
    const int status = utf8_read_scalar(piece, &scalar, &consumed);
    if (status != DORA_OK) {
        return status;
    }
    *out_scalar = scalar;
    return DORA_OK;
}
