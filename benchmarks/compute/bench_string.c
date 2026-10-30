// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "types/array_count.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_FLAT,
    VARIANT_UTF8_VIEW
};

enum operation : uint8_t {
    OPERATION_LIKE,
    OPERATION_CONTAINS,
    OPERATION_STARTS_WITH,
    OPERATION_LOWER,
    OPERATION_UPPER,
    OPERATION_SUBSTRING,
    OPERATION_LENGTH
};

static constexpr char MATCH_PATTERN[] = "%ab%";
static constexpr size_t MATCH_TEXT_OFFSET = 1;
static constexpr size_t MATCH_TEXT_BYTES = sizeof(MATCH_PATTERN) - 3;

struct inputs {
    struct dora_vector source;
    struct dora_vector needle;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args, enum operation operation) {
    struct inputs inputs = {};
    const enum dora_layout layout = args->variant == VARIANT_UTF8_VIEW ? DORA_LAYOUT_BINARY_VIEW : DORA_LAYOUT_BINARY;
    bench_check(bench, bench_utf8(bench, args, 1, layout, &inputs.source));
    if (operation == OPERATION_LIKE || operation == OPERATION_CONTAINS || operation == OPERATION_STARTS_WITH) {
        const bool is_pattern = operation == OPERATION_LIKE;
        const struct dora_bytes value = {
            .data = (const uint8_t*)(is_pattern ? MATCH_PATTERN : MATCH_PATTERN + MATCH_TEXT_OFFSET),
            .byte_count = is_pattern ? sizeof(MATCH_PATTERN) - 1 : MATCH_TEXT_BYTES
        };
        bench_check(bench,
                    dora_vector_init_copy_bytes(&inputs.needle,
                                                bench->pool,
                                                &args->type,
                                                DORA_LAYOUT_BINARY,
                                                &value,
                                                nullptr,
                                                1,
                                                DORA_FAILURE_POLICY_ERROR));
    }
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static size_t scalar_width(uint8_t first) {
    return first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
}

static bool contains_ab(struct dora_bytes text) {
    for (size_t offset = 0; offset + 1 < text.byte_count; ++offset) {
        if (text.data[offset] == MATCH_PATTERN[MATCH_TEXT_OFFSET] &&
            text.data[offset + 1] == MATCH_PATTERN[MATCH_TEXT_OFFSET + 1]) {
            return true;
        }
    }
    return false;
}

static constexpr uint64_t SUBSTRING_OFFSET = 1;
static constexpr uint64_t SUBSTRING_LENGTH = 8;
static constexpr uint8_t E_ACUTE_LEAD = 0xc3;
static constexpr uint8_t E_ACUTE_SMALL_TRAIL = 0xa9;
static constexpr uint8_t E_ACUTE_CAPITAL_TRAIL = 0x89;

static size_t reference_text(struct dora_bytes text, enum operation operation, uint8_t* out_text) {
    size_t count = 0;
    size_t scalar = 0;
    for (size_t offset = 0; offset < text.byte_count; ++scalar) {
        const size_t width = scalar_width(text.data[offset]);
        if (operation != OPERATION_SUBSTRING ||
            (scalar >= SUBSTRING_OFFSET && scalar < SUBSTRING_OFFSET + SUBSTRING_LENGTH)) {
            memcpy(out_text + count, text.data + offset, width);
            if (operation == OPERATION_LOWER) {
                if (width == 1 && out_text[count] >= 'A' && out_text[count] <= 'Z') {
                    out_text[count] += 'a' - 'A';
                } else if (width == 2 && out_text[count] == E_ACUTE_LEAD &&
                           out_text[count + 1] == E_ACUTE_CAPITAL_TRAIL) {
                    out_text[count + 1] = E_ACUTE_SMALL_TRAIL;
                }
            } else if (operation == OPERATION_UPPER) {
                if (width == 1 && out_text[count] >= 'a' && out_text[count] <= 'z') {
                    out_text[count] -= 'a' - 'A';
                } else if (width == 2 && out_text[count] == E_ACUTE_LEAD &&
                           out_text[count + 1] == E_ACUTE_SMALL_TRAIL) {
                    out_text[count + 1] = E_ACUTE_CAPITAL_TRAIL;
                }
            }
            count += width;
        }
        offset += width;
    }
    return count;
}

static int check_result(const struct bench_args* args,
                        const struct inputs* inputs,
                        const struct dora_vector* result,
                        enum operation operation) {
    uint64_t length = 0;
    int status = dora_vector_get_length(result, &length);
    if (status != 0 || length != args->rows) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t row = 0; row < length; ++row) {
        struct dora_bytes text = {};
        bool is_null = false;
        status = dora_vector_read_bytes(&inputs->source, row, &text, &is_null);
        if (status != 0 || is_null) {
            return status != 0 ? status : EDOM;
        }
        if (operation == OPERATION_LENGTH) {
            uint64_t actual = 0;
            uint64_t expected = 0;
            for (size_t offset = 0; offset < text.byte_count; offset += scalar_width(text.data[offset])) {
                ++expected;
            }
            status = dora_vector_read_u64(result, row, &actual, &is_null);
            if (status != 0) {
                return status;
            }
            if (actual != expected) {
                return EDOM;
            }
        } else if (operation == OPERATION_LIKE || operation == OPERATION_CONTAINS ||
                   operation == OPERATION_STARTS_WITH) {
            bool actual = false;
            bool expected = contains_ab(text);
            if (operation == OPERATION_STARTS_WITH) {
                expected = text.byte_count >= MATCH_TEXT_BYTES &&
                           memcmp(text.data, MATCH_PATTERN + MATCH_TEXT_OFFSET, MATCH_TEXT_BYTES) == 0;
            }
            status = dora_vector_read_bool(result, row, &actual, &is_null);
            if (status != 0) {
                return status;
            }
            if (actual != expected) {
                return EDOM;
            }
        } else {
            uint8_t expected[BENCH_UTF8_SLOT_BYTES];
            const size_t count = reference_text(text, operation, expected);
            struct dora_bytes actual = {};
            status = dora_vector_read_bytes(result, row, &actual, &is_null);
            if (status != 0) {
                return status;
            }
            if (!bench_bytes_equal((struct dora_bytes){.data = expected, .byte_count = count}, actual)) {
                return EDOM;
            }
        }
        if (is_null) {
            return EDOM;
        }
    }
    return 0;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    uint64_t bytes = 0;
    uint64_t needle_bytes = 0;
    bench_check(bench, bench_input_bytes(&inputs->source, &bytes));
    if (dora_vector_is_valid(&inputs->needle)) {
        bench_check(bench, bench_input_bytes(&inputs->needle, &needle_bytes));
    }
    bench_set_items(bench, args->rows);
    bench_set_bytes(bench, bytes + needle_bytes);
    dora_vector_release(&inputs->needle);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

#define BENCH_STRING_DEFINE_CASE(name, operation)                                                                      \
    BENCH_DEFINE_OPERATION(name##_operation, struct dora_vector, compute_##name, bench_release_vectors)                \
    static void name(struct bench* bench, const struct bench_args* args) {                                             \
        struct inputs inputs = prepare(bench, args, operation);                                                        \
        if (bench->status == 0) {                                                                                      \
            struct dora_vector result = {};                                                                            \
            bench_check(bench, compute_##name(&inputs, &result));                                                      \
            bench_check(bench, check_result(args, &inputs, &result, operation));                                       \
            dora_vector_release(&result);                                                                              \
        }                                                                                                              \
        bench_measure(bench, &name##_operation, &inputs);                                                              \
        finish(bench, args, &inputs);                                                                                  \
    }

//--------------------------------------
// Matching
//--------------------------------------

static int compute_like(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_like(&inputs->source, &inputs->needle, nullptr, out_slot);
}

BENCH_STRING_DEFINE_CASE(like, OPERATION_LIKE)

static int compute_contains(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_contains(&inputs->source, &inputs->needle, out_slot);
}

BENCH_STRING_DEFINE_CASE(contains, OPERATION_CONTAINS)

static int compute_starts_with(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_starts_with(&inputs->source, &inputs->needle, out_slot);
}

BENCH_STRING_DEFINE_CASE(starts_with, OPERATION_STARTS_WITH)

//--------------------------------------
// Transformations and Lengths
//--------------------------------------

static int compute_lower(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_lower(&inputs->source, out_slot);
}

BENCH_STRING_DEFINE_CASE(lower, OPERATION_LOWER)

static int compute_upper(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_upper(&inputs->source, out_slot);
}

BENCH_STRING_DEFINE_CASE(upper, OPERATION_UPPER)

static int compute_substring(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_substring(&inputs->source, SUBSTRING_OFFSET, SUBSTRING_LENGTH, out_slot);
}

BENCH_STRING_DEFINE_CASE(substring, OPERATION_SUBSTRING)

static int compute_length(const struct inputs* inputs, struct dora_vector* out_slot) {
    return dora_compute_utf8_scalar_count(&inputs->source, out_slot);
}

BENCH_STRING_DEFINE_CASE(length, OPERATION_LENGTH)

#undef BENCH_STRING_DEFINE_CASE

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"like/utf8/ascii", like, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"like/utf8_view/ascii", like, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .variant = VARIANT_UTF8_VIEW}},
        {"contains/utf8/ascii", contains, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"starts_with/utf8/ascii", starts_with, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"lower/utf8/ascii", lower, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"upper/utf8/ascii", upper, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"substring/utf8/ascii", substring, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"length/utf8/ascii", length, {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8}},
        {"lower/utf8/unicode",
         lower,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_UNICODE}},
        {"upper/utf8/unicode",
         upper,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_UNICODE}},
        {"substring/utf8/unicode",
         substring,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_UNICODE}},
        {"length/utf8/unicode",
         length,
         {.rows = BENCH_ROWS, .type.id = DORA_TYPE_UTF8, .alphabet = BENCH_ALPHABET_UNICODE}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
