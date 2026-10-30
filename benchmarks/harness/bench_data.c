// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench_data.h"

#include "bench.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "types/array_count.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//------------------------------------------------------------------------------
//
// Random Source
//
//------------------------------------------------------------------------------

struct random_state {
    uint64_t state;
    uint64_t increment;
};

static void advance(struct random_state* random) {
    random->state = random->state * UINT64_C(6'364'136'223'846'793'005) + random->increment;
}

static uint32_t pcg32(struct random_state* random) {
    const uint64_t previous = random->state;
    advance(random);
    const uint32_t bits = (uint32_t)(((previous >> 18U) ^ previous) >> 27U);
    const uint32_t rotation = (uint32_t)(previous >> 59U);
    return (bits >> rotation) | (bits << ((0U - rotation) & 31U));
}

static struct random_state seeded(uint64_t seed) {
    struct random_state random = {.increment = UINT64_C(1'442'695'040'888'963'407)};
    advance(&random);
    random.state += seed ^ UINT64_C(0x9437'8165);
    advance(&random);
    return random;
}

static uint32_t bounded(struct random_state* random, uint32_t bound) {
    const uint32_t threshold = (0U - bound) % bound;
    for (;;) {
        const uint32_t value = pcg32(random);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

static bool fraction(struct random_state* random, double proportion) {
    return (double)pcg32(random) / 4'294'967'296.0 < proportion;
}

//------------------------------------------------------------------------------
//
// Columns
//
//------------------------------------------------------------------------------

//--------------------------------------
// Storage
//--------------------------------------

static int allocate_array(uint64_t count, size_t width, void** out_values) {
    size_t bytes;
    if (ckd_mul(&bytes, count, width)) {
        return EOVERFLOW;
    }
    if (count == 0) {
        *out_values = nullptr;
        return 0;
    }
    *out_values = malloc(bytes);
    return *out_values == nullptr ? ENOMEM : 0;
}

static int validity_array(const struct bench_args* args, struct random_state* random, bool** out_validity) {
    if (!isfinite(args->null_fraction) || args->null_fraction < 0 || args->null_fraction > 1) {
        return EINVAL;
    }
    if (args->null_fraction == 0) {
        *out_validity = nullptr;
        return 0;
    }
    void* allocation = nullptr;
    const int status = allocate_array(args->rows, sizeof(bool), &allocation);
    if (status != 0) {
        return status;
    }
    *out_validity = allocation;
    for (uint64_t row = 0; row < args->rows; ++row) {
        (*out_validity)[row] = !fraction(random, args->null_fraction);
    }
    return 0;
}

//--------------------------------------
// Numeric Columns
//--------------------------------------

static uint32_t random_integer(struct random_state* random, uint64_t maximum) {
    return bounded(random, (uint32_t)(maximum < BENCH_INTEGER_LIMIT ? maximum : BENCH_INTEGER_LIMIT)) + 1;
}

static double random_float(struct random_state* random) {
    return (double)(random_integer(random, BENCH_INTEGER_LIMIT) * BENCH_FLOAT_NUMERATOR) / BENCH_FLOAT_DENOMINATOR;
}

static int16_t random_value_i16(const struct bench_args*, struct random_state* random) {
    return (int16_t)random_integer(random, INT16_MAX);
}

static constexpr int32_t RANDOM_DATE_BASE_DAYS = 8'035;
static constexpr uint32_t RANDOM_DATE_SPAN_DAYS = 10'000;

static int32_t random_value_i32(const struct bench_args* args, struct random_state* random) {
    return args->type.id == DORA_TYPE_DATE32 ? RANDOM_DATE_BASE_DAYS + (int32_t)bounded(random, RANDOM_DATE_SPAN_DAYS)
                                             : (int32_t)random_integer(random, INT32_MAX);
}

static constexpr int64_t SECONDS_PER_DAY = 86'400;
static constexpr int64_t RANDOM_TIMESTAMP_BASE_SECONDS = RANDOM_DATE_BASE_DAYS * SECONDS_PER_DAY + SECONDS_PER_DAY / 2;
static constexpr int64_t TICKS_PER_SECOND[] = {1, 1000, 1'000'000, 1'000'000'000};

static int64_t random_value_i64(const struct bench_args* args, struct random_state* random) {
    if (args->type.id != DORA_TYPE_TIMESTAMP) {
        return random_integer(random, INT64_MAX);
    }
    const int64_t seconds = RANDOM_TIMESTAMP_BASE_SECONDS + bounded(random, RANDOM_DATE_SPAN_DAYS) * SECONDS_PER_DAY;
    return seconds * TICKS_PER_SECOND[args->type.parameters.timestamp.unit];
}

static uint32_t random_value_u32(const struct bench_args*, struct random_state* random) {
    return random_integer(random, UINT32_MAX);
}

static uint64_t random_value_u64(const struct bench_args*, struct random_state* random) {
    return random_integer(random, UINT64_MAX);
}

static float random_value_f32(const struct bench_args*, struct random_state* random) {
    return (float)random_float(random);
}

static double random_value_f64(const struct bench_args*, struct random_state* random) {
    return random_float(random);
}

static struct dora_i128 random_value_i128(const struct bench_args* args, struct random_state* random) {
    unsigned _BitInt(128) limit = 1;
    for (int32_t digit = 0; digit < args->type.parameters.decimal.precision; ++digit) {
        limit *= 10U;
    }
    unsigned _BitInt(128) mask = limit - 1;
    for (uint32_t shift = 1; shift < 128U; shift *= 2) {
        mask |= mask >> shift;
    }
    unsigned _BitInt(128) value;
    do {
        value = 0;
        for (uint32_t shift = 0; shift < 128U; shift += 32U) {
            value |= (unsigned _BitInt(128))pcg32(random) << shift;
        }
        value &= mask;
    } while (value >= limit);
    return (struct dora_i128){.limbs = {(uint64_t)value, (uint64_t)(value >> 64U)}};
}

#define BENCH_DATA_DEFINE_RANDOM_NUMERIC(value_type, type_tag, convert_function)                                       \
    static int random_##type_tag(struct bench* bench,                                                                  \
                                 const struct bench_args* args,                                                        \
                                 struct random_state* random,                                                          \
                                 const bool* validity,                                                                 \
                                 struct dora_vector* out_vector) {                                                     \
        void* allocation = nullptr;                                                                                    \
        int status = allocate_array(args->rows, sizeof(value_type), &allocation);                                      \
        if (status != 0) {                                                                                             \
            return status;                                                                                             \
        }                                                                                                              \
        value_type* values = allocation;                                                                               \
        for (uint64_t row = 0; row < args->rows; ++row) {                                                              \
            values[row] = convert_function(args, random);                                                              \
        }                                                                                                              \
        status = dora_vector_init_copy_##type_tag(out_vector, bench->pool, &args->type, values, validity, args->rows); \
        free(values);                                                                                                  \
        return status;                                                                                                 \
    }

BENCH_DATA_DEFINE_RANDOM_NUMERIC(int16_t, i16, random_value_i16)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(int32_t, i32, random_value_i32)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(int64_t, i64, random_value_i64)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(uint32_t, u32, random_value_u32)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(uint64_t, u64, random_value_u64)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(float, f32, random_value_f32)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(double, f64, random_value_f64)
BENCH_DATA_DEFINE_RANDOM_NUMERIC(struct dora_i128, i128, random_value_i128)

#undef BENCH_DATA_DEFINE_RANDOM_NUMERIC

static int random_boolean(struct bench* bench,
                          const struct bench_args* args,
                          struct random_state* random,
                          const bool* validity,
                          struct dora_vector* out_vector) {
    if (!isfinite(args->selectivity) || args->selectivity < 0 || args->selectivity > 1) {
        return EINVAL;
    }
    void* allocation = nullptr;
    int status = allocate_array(args->rows, sizeof(bool), &allocation);
    if (status != 0) {
        return status;
    }
    bool* values = allocation;
    for (uint64_t row = 0; row < args->rows; ++row) {
        values[row] = fraction(random, args->selectivity);
    }
    status = dora_vector_init_copy_bool(out_vector, bench->pool, &args->type, values, validity, args->rows);
    free(values);
    return status;
}

//--------------------------------------
// Text Columns
//--------------------------------------

static constexpr uint32_t UTF8_ASCII_ALPHABET_SIZE = 'z' - 'A' + 1;

static size_t random_ascii(struct random_state* random, uint8_t* out_text) {
    const size_t length = bounded(random, BENCH_UTF8_SLOT_BYTES + 1);
    for (size_t index = 0; index < length; ++index) {
        out_text[index] = (uint8_t)('A' + bounded(random, UTF8_ASCII_ALPHABET_SIZE));
    }
    return length;
}

static constexpr uint32_t UTF8_UNICODE_SCALAR_LIMIT = 9;

static size_t random_unicode(struct random_state* random, uint8_t* out_text) {
    const uint8_t scalars[][4] = {{'A'}, {0xc3, 0xa9}, {0xc3, 0x89}, {0xe4, 0xb8, 0xad}, {0xf0, 0x9f, 0x98, 0x80}};
    const size_t widths[] = {1, 2, 2, 3, 4};
    const uint32_t count = bounded(random, UTF8_UNICODE_SCALAR_LIMIT);
    size_t length = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t scalar = bounded(random, ARRAY_COUNT(scalars));
        memcpy(out_text + length, scalars[scalar], widths[scalar]);
        length += widths[scalar];
    }
    return length;
}

static int random_utf8(struct bench* bench,
                       const struct bench_args* args,
                       struct random_state random,
                       enum bench_alphabet alphabet,
                       enum dora_layout layout,
                       enum dora_failure_policy content_policy,
                       bool invalid,
                       struct dora_vector* out_vector) {
    bool* validity = nullptr;
    int status = validity_array(args, &random, &validity);
    void* descriptors = nullptr;
    void* payload = nullptr;
    if (status == 0) {
        status = allocate_array(args->rows, sizeof(struct dora_bytes), &descriptors);
    }
    if (status == 0) {
        status = allocate_array(args->rows, BENCH_UTF8_SLOT_BYTES, &payload);
    }
    if (status == 0) {
        struct dora_bytes* strings = descriptors;
        uint8_t* bytes = payload;
        for (uint64_t row = 0; row < args->rows; ++row) {
            uint8_t* start = bytes + row * BENCH_UTF8_SLOT_BYTES;
            size_t length =
                alphabet == BENCH_ALPHABET_UNICODE ? random_unicode(&random, start) : random_ascii(&random, start);
            if (invalid && row + 1 == args->rows) {
                start[0] = 0xff;
                length = length == 0 ? 1 : length;
            }
            strings[row] = (struct dora_bytes){.data = start, .byte_count = length};
        }
        const struct dora_logical_type type = {.id = DORA_TYPE_UTF8};
        status = dora_vector_init_copy_bytes(out_vector,
                                             bench->pool,
                                             &type,
                                             layout,
                                             strings,
                                             validity,
                                             args->rows,
                                             content_policy);
    }
    free(payload);
    free(descriptors);
    free(validity);
    return status;
}

int bench_utf8(struct bench* bench,
               const struct bench_args* args,
               uint64_t seed,
               enum dora_layout layout,
               struct dora_vector* out_vector) {
    struct dora_vector_view view;
    int status =
        random_utf8(bench, args, seeded(seed), args->alphabet, layout, DORA_FAILURE_POLICY_ERROR, false, out_vector);
    if (status == 0) {
        status = dora_vector_get_view(out_vector, &view);
    }
    return status != 0 ? status : view.layout == layout ? 0 : EDOM;
}

//--------------------------------------
// Nested Columns
//--------------------------------------

static constexpr uint32_t LIST_LENGTH_LIMIT = 9;

static int random_list(struct bench* bench,
                       const struct bench_args* args,
                       struct random_state* random,
                       const bool* validity,
                       enum dora_type element_type,
                       enum dora_failure_policy content_policy,
                       bool invalid,
                       struct dora_vector* out_vector) {
    void* allocation = nullptr;
    int status = allocate_array(args->rows, sizeof(uint64_t), &allocation);
    if (status != 0) {
        return status;
    }
    uint64_t* lengths = allocation;
    uint64_t elements = 0;
    for (uint64_t row = 0; row < args->rows; ++row) {
        lengths[row] = bounded(random, LIST_LENGTH_LIMIT);
        if (ckd_add(&elements, elements, lengths[row])) {
            free(lengths);
            return EOVERFLOW;
        }
    }
    struct bench_args child_args = {.rows = elements, .type.id = element_type};
    struct dora_vector child = {};
    status = element_type == DORA_TYPE_UTF8 ? random_utf8(bench,
                                                          &child_args,
                                                          seeded(pcg32(random)),
                                                          BENCH_ALPHABET_ASCII,
                                                          DORA_LAYOUT_BINARY,
                                                          content_policy,
                                                          invalid,
                                                          &child)
                                            : random_i32(bench, &child_args, random, nullptr, &child);
    if (status == 0) {
        status = dora_vector_init_retain_list(out_vector, DORA_LAYOUT_LIST, &child, lengths, validity, args->rows);
    }
    dora_vector_release(&child);
    free(lengths);
    return status;
}

static int random_struct(struct bench* bench,
                         const struct bench_args* args,
                         struct random_state* random,
                         const bool* validity,
                         enum dora_failure_policy content_policy,
                         bool invalid,
                         struct dora_vector* out_vector) {
    struct dora_vector children[2] = {};
    const char* names[] = {"number", "text"};
    struct bench_args child_args = {.rows = args->rows, .type.id = DORA_TYPE_INT64};
    int status = random_i64(bench, &child_args, random, nullptr, &children[0]);
    if (status == 0) {
        status = random_utf8(bench,
                             &child_args,
                             seeded(pcg32(random)),
                             BENCH_ALPHABET_ASCII,
                             DORA_LAYOUT_BINARY,
                             content_policy,
                             invalid,
                             &children[1]);
    }
    if (status == 0) {
        status =
            dora_vector_init_retain_struct(out_vector, children, names, ARRAY_COUNT(children), validity, args->rows);
    }
    for (size_t index = 0; index < ARRAY_COUNT(children); ++index) {
        dora_vector_release(&children[index]);
    }
    return status;
}

//--------------------------------------
// Binary Columns
//--------------------------------------

static int random_fixed_binary(struct bench* bench,
                               const struct bench_args* args,
                               struct random_state* random,
                               const bool* validity,
                               struct dora_vector* out_vector) {
    const size_t width = args->type.parameters.byte_width;
    void* descriptors = nullptr;
    void* payload = nullptr;
    int status = allocate_array(args->rows, sizeof(struct dora_bytes), &descriptors);
    if (status == 0) {
        status = allocate_array(args->rows, width, &payload);
    }
    if (status == 0) {
        struct dora_bytes* values = descriptors;
        uint8_t* bytes = payload;
        for (uint64_t row = 0; row < args->rows; ++row) {
            uint8_t* start = bytes + row * width;
            for (size_t index = 0; index < width; ++index) {
                start[index] = (uint8_t)pcg32(random);
            }
            values[row] = (struct dora_bytes){.data = start, .byte_count = width};
        }
        status = dora_vector_init_copy_bytes(out_vector,
                                             bench->pool,
                                             &args->type,
                                             DORA_LAYOUT_FIXED,
                                             values,
                                             validity,
                                             args->rows,
                                             DORA_FAILURE_POLICY_ERROR);
    }
    free(payload);
    free(descriptors);
    return status;
}

//--------------------------------------
// Generated Columns
//--------------------------------------

int bench_random(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector) {
    if (args->type.id == DORA_TYPE_UTF8) {
        return bench_utf8(bench, args, seed, DORA_LAYOUT_BINARY, out_vector);
    }
    struct random_state random = seeded(seed);
    bool* validity = nullptr;
    int status = validity_array(args, &random, &validity);
    if (status != 0) {
        return status;
    }
    switch (args->type.id) {
    case DORA_TYPE_INT16:
        status = random_i16(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_INT32:
    case DORA_TYPE_DATE32:
        status = random_i32(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_INT64:
    case DORA_TYPE_TIMESTAMP:
        status = random_i64(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_UINT32:
        status = random_u32(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_UINT64:
        status = random_u64(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_FLOAT32:
        status = random_f32(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_FLOAT64:
        status = random_f64(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_BOOLEAN:
        status = random_boolean(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_DECIMAL:
        status = random_i128(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_FIXED_BINARY:
        status = random_fixed_binary(bench, args, &random, validity, out_vector);
        break;
    case DORA_TYPE_LIST:
        status =
            random_list(bench, args, &random, validity, DORA_TYPE_INT32, DORA_FAILURE_POLICY_ERROR, false, out_vector);
        break;
    case DORA_TYPE_STRUCT:
        status = random_struct(bench, args, &random, validity, DORA_FAILURE_POLICY_ERROR, false, out_vector);
        break;
    default:
        status = ENOTSUP;
        break;
    }
    free(validity);
    return status;
}

static int trusted_utf8(struct bench* bench,
                        const struct bench_args* args,
                        uint64_t seed,
                        bool invalid,
                        struct dora_vector* out_vector) {
    if (args->rows == 0 || args->null_fraction != 0) {
        return EINVAL;
    }
    if (args->type.id == DORA_TYPE_UTF8) {
        return random_utf8(bench,
                           args,
                           seeded(seed),
                           args->alphabet,
                           DORA_LAYOUT_BINARY,
                           DORA_FAILURE_POLICY_TRUST,
                           invalid,
                           out_vector);
    }
    struct random_state random = seeded(seed);
    if (args->type.id == DORA_TYPE_LIST) {
        return random_list(bench,
                           args,
                           &random,
                           nullptr,
                           DORA_TYPE_UTF8,
                           DORA_FAILURE_POLICY_TRUST,
                           invalid,
                           out_vector);
    }
    if (args->type.id == DORA_TYPE_STRUCT) {
        return random_struct(bench, args, &random, nullptr, DORA_FAILURE_POLICY_TRUST, invalid, out_vector);
    }
    return ENOTSUP;
}

int bench_unchecked_utf8(struct bench* bench,
                         const struct bench_args* args,
                         uint64_t seed,
                         struct dora_vector* out_vector) {
    return trusted_utf8(bench, args, seed, false, out_vector);
}

int bench_invalid_utf8(struct bench* bench,
                       const struct bench_args* args,
                       uint64_t seed,
                       struct dora_vector* out_vector) {
    return trusted_utf8(bench, args, seed, true, out_vector);
}

static int random_with_validity(struct bench* bench,
                                const struct bench_args* args,
                                uint64_t seed,
                                struct dora_vector* out_source,
                                struct dora_vector_view* out_view,
                                bool** out_validity) {
    void* validity = nullptr;
    int status = bench_random(bench, args, seed, out_source);
    if (status == 0) {
        status = dora_vector_get_view(out_source, out_view);
    }
    if (status == 0 && out_view->validity.data != nullptr) {
        status = allocate_array(args->rows, sizeof(bool), &validity);
    }
    if (validity != nullptr) {
        dora_vector_view_copy_validity(out_view, (struct dora_range){.length = args->rows}, validity);
    }
    *out_validity = validity;
    return status;
}

int bench_formatted(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector) {
    struct dora_vector source = {};
    struct dora_vector_view view;
    bool* validity = nullptr;
    int status = random_with_validity(bench, args, seed, &source, &view, &validity);
    void* descriptors = nullptr;
    void* payload = nullptr;
    if (status == 0) {
        status = allocate_array(args->rows, sizeof(struct dora_bytes), &descriptors);
    }
    if (status == 0) {
        status = allocate_array(args->rows, BENCH_FORMATTED_VALUE_BYTES, &payload);
    }
    struct dora_bytes* strings = descriptors;
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        char* text = (char*)payload + row * BENCH_FORMATTED_VALUE_BYTES;
        size_t length;
        bool is_null;
        status = dora_vector_format_value(&source, row, text, BENCH_FORMATTED_VALUE_BYTES, &length, &is_null);
        if (status == 0) {
            strings[row] = (struct dora_bytes){.data = (const uint8_t*)text, .byte_count = length};
        }
    }
    if (status == 0) {
        status = dora_vector_init_copy_bytes(out_vector,
                                             bench->pool,
                                             &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                             DORA_LAYOUT_BINARY,
                                             strings,
                                             validity,
                                             args->rows,
                                             DORA_FAILURE_POLICY_ERROR);
    }
    free(payload);
    free(descriptors);
    free(validity);
    dora_vector_release(&source);
    return status;
}

//------------------------------------------------------------------------------
//
// Encoded Columns
//
//------------------------------------------------------------------------------

static bool is_drawn(const struct dora_bytes* strings, size_t count, struct dora_bytes value) {
    for (size_t index = 0; index < count; ++index) {
        if (bench_bytes_equal(strings[index], value)) {
            return true;
        }
    }
    return false;
}

int bench_dictionary(struct bench* bench,
                     const struct bench_args* args,
                     uint64_t seed,
                     struct dora_vector* out_vector) {
    struct random_state random = seeded(seed);
    bool* validity = nullptr;
    int status = validity_array(args, &random, &validity);
    void* allocation = nullptr;
    if (status == 0) {
        status = allocate_array(args->rows, sizeof(uint32_t), &allocation);
    }
    struct dora_vector indices = {};
    struct dora_vector dictionary = {};
    if (status == 0) {
        uint32_t* values = allocation;
        for (uint64_t row = 0; row < args->rows; ++row) {
            values[row] = bounded(&random, BENCH_DICTIONARY_CARDINALITY);
        }
        status = dora_vector_init_copy_u32(&indices,
                                           bench->pool,
                                           &(const struct dora_logical_type){.id = DORA_TYPE_UINT32},
                                           values,
                                           validity,
                                           args->rows);
    }
    if (status == 0) {
        uint8_t words[BENCH_DICTIONARY_CARDINALITY][BENCH_UTF8_SLOT_BYTES];
        struct dora_bytes strings[BENCH_DICTIONARY_CARDINALITY];
        for (size_t index = 0; index < ARRAY_COUNT(words); ++index) {
            strings[index] = (struct dora_bytes){.data = words[index]};
            do {
                strings[index].byte_count = random_ascii(&random, words[index]);
            } while (is_drawn(strings, index, strings[index]));
        }
        status = dora_vector_init_copy_bytes(&dictionary,
                                             bench->pool,
                                             &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                             DORA_LAYOUT_BINARY,
                                             strings,
                                             nullptr,
                                             ARRAY_COUNT(strings),
                                             DORA_FAILURE_POLICY_ERROR);
    }
    if (status == 0) {
        status = dora_vector_init_retain_dictionary(out_vector, &indices, &dictionary);
    }
    dora_vector_release(&dictionary);
    dora_vector_release(&indices);
    free(allocation);
    free(validity);
    return status;
}

int bench_decoded_dictionary(struct bench* bench,
                             const struct bench_args* args,
                             uint64_t seed,
                             struct dora_vector* out_vector) {
    struct dora_vector dictionary = {};
    int status = bench_dictionary(bench, args, seed, &dictionary);
    if (status == 0) {
        status = dora_vector_decode(&dictionary, out_vector);
    }
    dora_vector_release(&dictionary);
    return status;
}

int bench_constant(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector) {
    struct bench_args scalar_args = *args;
    scalar_args.rows = 1;
    struct dora_vector scalar = {};
    int status = bench_random(bench, &scalar_args, seed, &scalar);
    if (status == 0) {
        status = dora_vector_init_retain_constant(out_vector, &scalar, args->rows);
    }
    dora_vector_release(&scalar);
    return status;
}

int bench_take_indices(struct bench* bench,
                       const struct bench_args* args,
                       uint64_t seed,
                       struct dora_vector* out_vector) {
    if (args->rows == 0 || args->rows > UINT32_MAX) {
        return EINVAL;
    }
    struct random_state random = seeded(seed);
    void* allocation = nullptr;
    int status = allocate_array(args->rows, sizeof(uint32_t), &allocation);
    if (status == 0) {
        uint32_t* indices = allocation;
        for (uint64_t row = 0; row < args->rows; ++row) {
            indices[row] = bounded(&random, (uint32_t)args->rows);
        }
        status = dora_vector_init_copy_u32(out_vector,
                                           bench->pool,
                                           &(const struct dora_logical_type){.id = DORA_TYPE_UINT32},
                                           indices,
                                           nullptr,
                                           args->rows);
    }
    free(allocation);
    return status;
}

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

//--------------------------------------
// Fixtures
//--------------------------------------

int bench_integer_range(enum dora_type type, struct bench_range* out_range) {
    switch (type) {
    case DORA_TYPE_INT32:
        *out_range = (struct bench_range){.minimum = INT32_MIN, .maximum = INT32_MAX};
        return 0;
    case DORA_TYPE_INT64:
        *out_range = (struct bench_range){.minimum = INT64_MIN, .maximum = INT64_MAX};
        return 0;
    case DORA_TYPE_UINT32:
        *out_range = (struct bench_range){.maximum = UINT32_MAX};
        return 0;
    case DORA_TYPE_UINT64:
        *out_range = (struct bench_range){.maximum = UINT64_MAX};
        return 0;
    default:
        return ENOTSUP;
    }
}

static int
outlier_row(const bool* validity, uint64_t row_count, const struct bench_outlier* outlier, uint64_t* out_row) {
    uint64_t row = outlier->from_row;
    while (row < row_count && validity != nullptr && !validity[row]) {
        ++row;
    }
    *out_row = row;
    return row < row_count ? 0 : EDOM;
}

#define BENCH_DATA_DEFINE_WITH_BOUNDS(value_type, type_tag)                                                            \
    static int with_bounds_##type_tag(struct dora_builder* builder,                                                    \
                                      const struct dora_vector_view* view,                                             \
                                      const bool* validity,                                                            \
                                      const struct bench_outlier* outlier) {                                           \
        const value_type* source = dora_vector_view_get_values_##type_tag(view);                                       \
        if (source == nullptr || view->length == 0) {                                                                  \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        void* allocation = nullptr;                                                                                    \
        int status = allocate_array(view->length, sizeof(value_type), &allocation);                                    \
        uint64_t row = 0;                                                                                              \
        if (status == 0 && outlier != nullptr) {                                                                       \
            status = outlier_row(validity, view->length, outlier, &row);                                               \
        }                                                                                                              \
        if (status == 0) {                                                                                             \
            value_type* values = allocation;                                                                           \
            memcpy(values, source, view->length * sizeof(value_type));                                                 \
            if (outlier != nullptr) {                                                                                  \
                values[row] = (value_type)outlier->value;                                                              \
            }                                                                                                          \
            status = dora_builder_append_values_##type_tag(builder, values, validity, view->length);                   \
        }                                                                                                              \
        free(allocation);                                                                                              \
        return status;                                                                                                 \
    }

BENCH_DATA_DEFINE_WITH_BOUNDS(int32_t, i32)
BENCH_DATA_DEFINE_WITH_BOUNDS(int64_t, i64)
BENCH_DATA_DEFINE_WITH_BOUNDS(uint32_t, u32)
BENCH_DATA_DEFINE_WITH_BOUNDS(uint64_t, u64)

#undef BENCH_DATA_DEFINE_WITH_BOUNDS

int bench_with_bounds(struct bench* bench,
                      const struct bench_args* args,
                      uint64_t seed,
                      const struct bench_outlier* outlier,
                      struct dora_vector* out_vector) {
    struct bench_range range = {};
    int status = bench_integer_range(args->type.id, &range);
    if (status == 0 && outlier != nullptr && (outlier->value < range.minimum || outlier->value > range.maximum)) {
        status = ERANGE;
    }
    struct dora_vector source = {};
    struct dora_vector_view view;
    bool* validity = nullptr;
    if (status == 0) {
        status = random_with_validity(bench, args, seed, &source, &view, &validity);
    }
    const struct dora_statistics_request request = {.kinds = DORA_STATISTICS_KIND_MIN | DORA_STATISTICS_KIND_MAX};
    const struct dora_builder_options options = {.statistics_requests = &request, .statistics_request_count = 1};
    struct dora_builder builder = {};
    if (status == 0) {
        status = dora_builder_init_type(&builder, bench->pool, &args->type, &options);
    }
    if (status == 0) {
        switch (args->type.id) {
        case DORA_TYPE_INT32:
            status = with_bounds_i32(&builder, &view, validity, outlier);
            break;
        case DORA_TYPE_INT64:
            status = with_bounds_i64(&builder, &view, validity, outlier);
            break;
        case DORA_TYPE_UINT32:
            status = with_bounds_u32(&builder, &view, validity, outlier);
            break;
        case DORA_TYPE_UINT64:
            status = with_bounds_u64(&builder, &view, validity, outlier);
            break;
        default:
            status = ENOTSUP;
            break;
        }
    }
    if (status == 0) {
        status = dora_builder_finish(&builder, out_vector);
    }
    dora_builder_release(&builder);
    free(validity);
    dora_vector_release(&source);
    return status;
}

//--------------------------------------
// Bounds
//--------------------------------------

static int read_bound(const struct dora_vector* group,
                      enum dora_type type,
                      const char* bound_name,
                      const char* exact_name,
                      struct dora_i128* out_bound) {
    struct dora_vector bound = {};
    struct dora_vector exact = {};
    struct dora_logical_type bound_type = {};
    bool is_null = true;
    bool is_exact = false;
    bool is_exact_null = true;
    int status = dora_vector_find_child(group, bound_name, &bound);
    if (status == 0) {
        status = dora_vector_find_child(group, exact_name, &exact);
    }
    if (status == 0) {
        status = dora_vector_get_type(&bound, &bound_type);
    }
    if (status == 0 && bound_type.id != type) {
        status = EDOM;
    }
    if (status == 0) {
        status = bench_read_integer(&bound, 0, out_bound, &is_null);
    }
    if (status == 0) {
        status = dora_vector_read_bool(&exact, 0, &is_exact, &is_exact_null);
    }
    dora_vector_release(&exact);
    dora_vector_release(&bound);
    return status != 0 ? status : !is_null && !is_exact_null && is_exact ? 0 : EDOM;
}

int bench_exact_bounds(const struct dora_vector* vector, struct bench_range* out_bounds) {
    struct dora_logical_type type = {};
    struct dora_vector statistics = {};
    struct dora_vector group = {};
    struct dora_i128 minimum = {};
    struct dora_i128 maximum = {};
    int status = dora_vector_get_type(vector, &type);
    if (status == 0) {
        status = dora_vector_get_statistics(vector, &statistics);
    }
    if (status == 0) {
        status = dora_vector_get_child(&statistics, 0, &group);
    }
    if (status == 0) {
        status = read_bound(&group, type.id, "min", "is_min_exact", &minimum);
    }
    if (status == 0) {
        status = read_bound(&group, type.id, "max", "is_max_exact", &maximum);
    }
    dora_vector_release(&group);
    dora_vector_release(&statistics);
    if (status == 0) {
        *out_bounds = (struct bench_range){.minimum = bench_i128_value(minimum), .maximum = bench_i128_value(maximum)};
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Batches
//
//------------------------------------------------------------------------------

static int next_batch(struct dora_vector_source* source,
                      struct dora_pool,
                      struct dora_vector* out_vector,
                      struct dora_stream_result* out_result,
                      struct dora_error*) {
    struct bench_batch_source* batches = (struct bench_batch_source*)source;
    if (batches->next == batches->count) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
        return 0;
    }
    const int status = dora_vector_retain(&batches->batches[batches->next], out_vector);
    if (status == 0) {
        ++batches->next;
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY};
    }
    return status;
}

static void release_source(struct dora_vector_source* source) {
    *source = (struct dora_vector_source){};
}

int bench_batches(struct bench* bench, size_t count, struct dora_vector* out_batches) {
    const struct bench_args args = {.rows = BENCH_BATCH_ROWS, .type.id = DORA_TYPE_INT64};
    for (size_t index = 0; index < count; ++index) {
        const int status = bench_random(bench, &args, index + 1, &out_batches[index]);
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

int bench_collect(struct dora_reader* reader, size_t capacity, struct dora_vector* out_batches, size_t* out_count) {
    size_t count = 0;
    int status = 0;
    struct dora_vector overflow = {};
    while (status == 0) {
        struct dora_vector* batch = count < capacity ? &out_batches[count] : &overflow;
        struct dora_stream_result flow = {};
        status = dora_reader_next(reader, batch, &flow, nullptr);
        if (status != 0 || flow.status == DORA_STREAM_STATUS_END) {
            break;
        }
        if (flow.status != DORA_STREAM_STATUS_READY) {
            status = EDOM;
        } else if (batch == &overflow) {
            dora_vector_release(&overflow);
            status = EOVERFLOW;
        } else {
            ++count;
        }
    }
    *out_count = count;
    return status;
}

void bench_batch_source_init(struct bench_batch_source* source, const struct dora_vector* batches, size_t count) {
    *source = (struct bench_batch_source){
        .source = {.next = next_batch, .release = release_source},
        .batches = batches,
        .count = count
    };
}

//------------------------------------------------------------------------------
//
// Documents
//
//------------------------------------------------------------------------------

size_t bench_line_end(const char* text, size_t byte_count, uint64_t lines) {
    size_t offset = 0;
    for (uint64_t line = 0; line < lines && offset < byte_count; ++line) {
        const char* newline = memchr(text + offset, '\n', byte_count - offset);
        offset = newline != nullptr ? (size_t)(newline - text) + 1 : byte_count;
    }
    return offset;
}

static constexpr uint64_t TEXT_SEED = 42;
static constexpr uint32_t TEXT_ALPHABET_SIZE = 'z' - 'a' + 1;
static constexpr int64_t TEXT_TIMESTAMP_BASE_MICROSECONDS = INT64_C(1'577'836'800'000'000);
static constexpr uint64_t TEXT_TIMESTAMP_FRACTION_LIMIT = 1'000'000;

void bench_text_rows(uint64_t count, struct bench_text_row* out_rows) {
    struct random_state random = seeded(TEXT_SEED);
    for (uint64_t row = 0; row < count; ++row) {
        struct bench_text_row* record = &out_rows[row];
        record->id = random_integer(&random, INT64_MAX);
        record->value = random_float(&random);
        const size_t length = bounded(&random, BENCH_UTF8_SLOT_BYTES + 1);
        for (size_t index = 0; index < length; ++index) {
            record->text[index] = (char)('a' + bounded(&random, TEXT_ALPHABET_SIZE));
        }
        record->text[length] = 0;
        record->flag = fraction(&random, 0.5);
        record->ts_microseconds = TEXT_TIMESTAMP_BASE_MICROSECONDS + (int64_t)(row % TEXT_TIMESTAMP_FRACTION_LIMIT);
    }
}

static int format_text_row(const struct bench_text_args* args,
                           const struct bench_text_row* row,
                           size_t capacity,
                           char* out_text,
                           size_t* out_byte_count) {
    const char* flag = row->flag ? "true" : "false";
    const int64_t fraction = row->ts_microseconds - TEXT_TIMESTAMP_BASE_MICROSECONDS;
    int count;
    if (args->format == BENCH_TEXT_FORMAT_NDJSON) {
        count = args->columns == BENCH_TEXT_COLUMNS_NUMERIC
                    ? snprintf(out_text, capacity, "{\"id\":%" PRIu32 ",\"value\":%.3f}\n", row->id, row->value)
                    : snprintf(out_text,
                               capacity,
                               "{\"id\":%" PRIu32
                               ",\"value\":%.3f,\"text\":\"%s\",\"flag\":%s,\"ts\":\"2020-01-01T00:00:00.%06" PRId64
                               "\"}\n",
                               row->id,
                               row->value,
                               row->text,
                               flag,
                               fraction);
    } else {
        count = args->columns == BENCH_TEXT_COLUMNS_NUMERIC
                    ? snprintf(out_text, capacity, "%" PRIu32 ",%.3f\n", row->id, row->value)
                    : snprintf(out_text,
                               capacity,
                               "%" PRIu32 ",%.3f,\"%s\",%s,2020-01-01T00:00:00.%06" PRId64 "\n",
                               row->id,
                               row->value,
                               row->text,
                               flag,
                               fraction);
    }
    const int status = bench_format_status(count, capacity);
    if (status == 0) {
        *out_byte_count = (size_t)count;
    }
    return status;
}

static constexpr size_t TEXT_ROW_CAPACITY_BYTES = 256;
static constexpr size_t TEXT_HEADER_CAPACITY_BYTES = 128;

int bench_text(struct bench* bench,
               const struct bench_text_args* args,
               struct dora_allocation* out_text,
               size_t* out_byte_count) {
    size_t capacity;
    if (ckd_mul(&capacity, args->rows, TEXT_ROW_CAPACITY_BYTES) ||
        ckd_add(&capacity, capacity, TEXT_HEADER_CAPACITY_BYTES)) {
        return EOVERFLOW;
    }
    void* allocation = nullptr;
    int status = allocate_array(args->rows, sizeof(struct bench_text_row), &allocation);
    if (status == 0) {
        bench_text_rows(args->rows, allocation);
        status = dora_pool_allocate(bench->pool, capacity, out_text);
    }
    if (status != 0) {
        free(allocation);
        return status;
    }
    const struct bench_text_row* rows = allocation;
    char* text = out_text->address;
    size_t used = 0;
    if (args->format == BENCH_TEXT_FORMAT_CSV) {
        const char* header = args->columns == BENCH_TEXT_COLUMNS_NUMERIC ? "id,value\n" : "id,value,text,flag,ts\n";
        used = strlen(header);
        memcpy(text, header, used);
    }
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        size_t count = 0;
        status = format_text_row(args, &rows[row], capacity - used, text + used, &count);
        used += count;
    }
    free(allocation);
    if (status != 0) {
        dora_allocation_release(out_text);
    } else {
        *out_byte_count = used;
    }
    return status;
}

static constexpr uint32_t NUMERIC_FIELD_COUNT = 2;

int bench_text_schema(struct bench* bench, enum bench_text_columns columns, struct dora_schema* out_schema) {
    const struct dora_schema_field fields[] = {
        {.name = "id", .type.id = DORA_TYPE_INT64},
        {.name = "value", .type.id = DORA_TYPE_FLOAT64},
        {.name = "text", .type.id = DORA_TYPE_UTF8},
        {.name = "flag", .type.id = DORA_TYPE_BOOLEAN},
        {.name = "ts",
         .type = {
             .id = DORA_TYPE_TIMESTAMP,
             .parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr}
         }}
    };
    const struct dora_schema_input input = {
        .fields = fields,
        .field_count = columns == BENCH_TEXT_COLUMNS_NUMERIC ? NUMERIC_FIELD_COUNT : ARRAY_COUNT(fields)
    };
    return dora_schema_init_copy(out_schema, bench->pool, &input);
}

void bench_release_byte_source(struct dora_byte_source* source) {
    if (source->release != nullptr) {
        source->release(source);
    }
}

//------------------------------------------------------------------------------
//
// Views
//
//------------------------------------------------------------------------------

const void* bench_view_values(const struct dora_vector_view* view, size_t* out_width) {
    const void* values = nullptr;
    size_t width = 0;
    switch (view->physical_type) {
    case DORA_PHYSICAL_TYPE_I16:
        values = dora_vector_view_get_values_i16(view);
        width = sizeof(int16_t);
        break;
    case DORA_PHYSICAL_TYPE_I32:
        values = dora_vector_view_get_values_i32(view);
        width = sizeof(int32_t);
        break;
    case DORA_PHYSICAL_TYPE_I64:
        values = dora_vector_view_get_values_i64(view);
        width = sizeof(int64_t);
        break;
    case DORA_PHYSICAL_TYPE_U32:
        values = dora_vector_view_get_values_u32(view);
        width = sizeof(uint32_t);
        break;
    case DORA_PHYSICAL_TYPE_F32:
        values = dora_vector_view_get_values_f32(view);
        width = sizeof(float);
        break;
    case DORA_PHYSICAL_TYPE_F64:
        values = dora_vector_view_get_values_f64(view);
        width = sizeof(double);
        break;
    case DORA_PHYSICAL_TYPE_I128:
        values = dora_vector_view_get_values_i128(view);
        width = sizeof(struct dora_i128);
        break;
    default:
        break;
    }
    if (out_width != nullptr) {
        *out_width = width;
    }
    return values;
}

//------------------------------------------------------------------------------
//
// Accounting
//
//------------------------------------------------------------------------------

int bench_input_bytes(const struct dora_vector* vector, uint64_t* out_bytes) {
    struct dora_vector_view view;
    int status = dora_vector_get_view(vector, &view);
    if (status != 0) {
        return status;
    }
    uint64_t bytes = 0;
    const size_t extents[] = {
        view.validity.byte_count,
        view.values.byte_count,
        view.offsets.byte_count,
        view.sizes.byte_count,
        view.type_ids.byte_count
    };
    for (size_t index = 0; index < ARRAY_COUNT(extents); ++index) {
        if (ckd_add(&bytes, bytes, extents[index])) {
            return EOVERFLOW;
        }
    }
    for (uint32_t index = 0; index < view.data_buffer_count; ++index) {
        if (ckd_add(&bytes, bytes, view.data_buffers[index].byte_count)) {
            return EOVERFLOW;
        }
    }
    uint32_t child_count;
    status = dora_vector_get_child_count(vector, &child_count);
    for (uint32_t index = 0; status == 0 && index < child_count; ++index) {
        struct dora_vector child = {};
        status = dora_vector_get_child(vector, index, &child);
        uint64_t child_bytes = 0;
        if (status == 0) {
            status = bench_input_bytes(&child, &child_bytes);
        }
        dora_vector_release(&child);
        if (status == 0 && ckd_add(&bytes, bytes, child_bytes)) {
            status = EOVERFLOW;
        }
    }
    if (status == 0) {
        *out_bytes = bytes;
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Oracles
//
//------------------------------------------------------------------------------

int bench_read_number(const struct dora_vector* source, uint64_t row, double* out_value) {
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(source, &type);
    if (status != 0) {
        return status;
    }
    bool is_null = false;
    switch (type.id) {
    case DORA_TYPE_INT32:
    case DORA_TYPE_DATE32: {
        int32_t value = 0;
        status = dora_vector_read_i32(source, row, &value, &is_null);
        *out_value = value;
        break;
    }
    case DORA_TYPE_UINT32: {
        uint32_t value = 0;
        status = dora_vector_read_u32(source, row, &value, &is_null);
        *out_value = value;
        break;
    }
    case DORA_TYPE_INT64:
    case DORA_TYPE_TIMESTAMP: {
        int64_t value = 0;
        status = dora_vector_read_i64(source, row, &value, &is_null);
        *out_value = (double)value;
        break;
    }
    case DORA_TYPE_FLOAT64:
        status = dora_vector_read_f64(source, row, out_value, &is_null);
        break;
    case DORA_TYPE_DECIMAL: {
        char text[BENCH_FORMATTED_VALUE_BYTES];
        size_t length = 0;
        status = dora_vector_format_value(source, row, text, sizeof(text), &length, &is_null);
        if (status == 0 && !is_null) {
            char* end = nullptr;
            *out_value = strtod(text, &end);
            status = end == text + length ? 0 : EDOM;
        }
        break;
    }
    default:
        return ENOTSUP;
    }
    return status != 0 ? status : is_null ? EDOM : 0;
}

int bench_read_integer(const struct dora_vector* vector, uint64_t row, struct dora_i128* out_value, bool* out_is_null) {
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(vector, &type);
    if (status != 0) {
        return status;
    }
    switch (type.id) {
    case DORA_TYPE_INT32: {
        int32_t value = 0;
        status = dora_vector_read_i32(vector, row, &value, out_is_null);
        *out_value = (struct dora_i128){.limbs = {(uint64_t)value, value < 0 ? UINT64_MAX : 0}};
        return status;
    }
    case DORA_TYPE_INT64: {
        int64_t value = 0;
        status = dora_vector_read_i64(vector, row, &value, out_is_null);
        *out_value = (struct dora_i128){.limbs = {(uint64_t)value, value < 0 ? UINT64_MAX : 0}};
        return status;
    }
    case DORA_TYPE_UINT32: {
        uint32_t value = 0;
        status = dora_vector_read_u32(vector, row, &value, out_is_null);
        *out_value = (struct dora_i128){.limbs = {value}};
        return status;
    }
    case DORA_TYPE_UINT64: {
        uint64_t value = 0;
        status = dora_vector_read_u64(vector, row, &value, out_is_null);
        *out_value = (struct dora_i128){.limbs = {value}};
        return status;
    }
    default:
        return ENOTSUP;
    }
}

_BitInt(128) bench_i128_value(struct dora_i128 value) {
    return (_BitInt(128))(((unsigned _BitInt(128))value.limbs[1] << 64U) | value.limbs[0]);
}

static constexpr int FRACTION_DIGITS[] = {0, 3, 6, 9};
static constexpr int TM_YEAR_BASE = 1900;
static constexpr int FOUR_DIGIT_YEAR_MIN = 1000;
static constexpr int FOUR_DIGIT_YEAR_MAX = 9999;

int bench_temporal_text(const struct dora_logical_type* type,
                        int64_t ticks,
                        char out_text[BENCH_FORMATTED_VALUE_BYTES],
                        size_t* out_byte_count) {
    const size_t capacity = BENCH_FORMATTED_VALUE_BYTES;
    const bool is_date = type->id == DORA_TYPE_DATE32;
    if (!is_date && (type->id != DORA_TYPE_TIMESTAMP || type->parameters.timestamp.unit > DORA_TIME_UNIT_NANOSECOND ||
                     (type->parameters.timestamp.timezone != nullptr && type->parameters.timestamp.timezone[0] != 0))) {
        return ENOTSUP;
    }
    const enum dora_time_unit unit = is_date ? DORA_TIME_UNIT_SECOND : type->parameters.timestamp.unit;
    int64_t seconds = is_date ? ticks * SECONDS_PER_DAY : ticks / TICKS_PER_SECOND[unit];
    int64_t fraction = is_date ? 0 : ticks % TICKS_PER_SECOND[unit];
    if (fraction < 0) {
        fraction += TICKS_PER_SECOND[unit];
        --seconds;
    }
    const time_t time = (time_t)seconds;
    struct tm calendar = {};
    if (gmtime_r(&time, &calendar) == nullptr) {
        return EOVERFLOW;
    }
    const int year = calendar.tm_year + TM_YEAR_BASE;
    if (year < FOUR_DIGIT_YEAR_MIN || year > FOUR_DIGIT_YEAR_MAX) {
        return ERANGE;
    }
    size_t count = strftime(out_text, capacity, is_date ? "%Y-%m-%d" : "%Y-%m-%dT%H:%M:%S", &calendar);
    if (count == 0) {
        return ENOSPC;
    }
    if (FRACTION_DIGITS[unit] != 0) {
        const int length = snprintf(out_text + count, capacity - count, ".%0*" PRId64, FRACTION_DIGITS[unit], fraction);
        const int status = bench_format_status(length, capacity - count);
        if (status != 0) {
            return status;
        }
        count += (size_t)length;
    }
    *out_byte_count = count;
    return 0;
}

bool bench_bytes_equal(struct dora_bytes left, struct dora_bytes right) {
    return left.byte_count == right.byte_count &&
           (left.byte_count == 0 || memcmp(left.data, right.data, left.byte_count) == 0);
}

int bench_vectors_equal(const struct dora_vector* expected, const struct dora_vector* actual) {
    bool equal = false;
    const int status = dora_vector_is_equal(expected, actual, &equal);
    return status != 0 ? status : equal ? 0 : EDOM;
}

int bench_batches_equal(const struct dora_vector* expected,
                        size_t expected_count,
                        const struct dora_vector* actual,
                        size_t actual_count) {
    int status = actual_count == expected_count ? 0 : EDOM;
    for (size_t index = 0; status == 0 && index < actual_count; ++index) {
        status = bench_vectors_equal(&expected[index], &actual[index]);
    }
    return status;
}
