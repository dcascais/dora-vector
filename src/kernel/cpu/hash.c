// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "hash/hash.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/cpu/packet_bits.h"
#include "kernel/hash_cache.h"
#include "kernel/packet.h"
#include "numeric/decimal.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Words
//
//------------------------------------------------------------------------------

struct hash_words {
    const uint64_t* validity;
    uint64_t last_tail;
    uint32_t count;
    uint32_t word_count;
};

static inline struct hash_words hash_words_of(const struct kernel_packet* packet) {
    return (struct hash_words){
        .validity = packet->validity,
        .last_tail = bitmap_tail_mask(packet->count),
        .count = packet->count,
        .word_count = (uint32_t)bitmap_words_for_bits(packet->count),
    };
}

static inline uint64_t hash_word_tail(const struct hash_words* words, uint32_t word) {
    return word + 1 == words->word_count ? words->last_tail : UINT64_MAX;
}

static inline uint64_t hash_word_valid(const struct hash_words* words, uint32_t word) {
    const uint64_t tail = hash_word_tail(words, word);
    return words->validity == nullptr ? tail : words->validity[word] & tail;
}

static inline uint32_t hash_word_lanes(const struct hash_words* words, uint32_t word) {
    return word + 1 == words->word_count ? words->count - word * 64 : 64;
}

//------------------------------------------------------------------------------
//
// Expansions
//
//------------------------------------------------------------------------------

#define AT_FLAT(ROW) hash_load(values, offset + (ROW))
#define AT_MAP(ROW) hash_load(values, offset + source->rows[ROW])

#define STORE_PLAIN(ROW, HASH) out_hashes[ROW] = (HASH)
#define STORE_FLAT(ROW, HASH) out_hashes[ROW] = hash_value_combine(type_load_u64(prior_values, ROW), (HASH))
#define STORE_CONSTANT(ROW, HASH) out_hashes[ROW] = hash_value_combine(constant_hash, (HASH))
#define STORE_MAP(ROW, HASH) out_hashes[ROW] = hash_value_combine(type_load_u64(prior_values, prior_rows[ROW]), (HASH))

#define HASH_MIXED_WALK(AT, HASH, STORE)                                                                               \
    {                                                                                                                  \
        for (uint32_t lane = 0; lane < lanes; ++lane) {                                                                \
            STORE(word * 64 + lane, HASH_NULL_VALUE);                                                                  \
        }                                                                                                              \
        for (uint64_t pending = valid; pending != 0;) {                                                                \
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);                                           \
            STORE(word * 64 + lane, HASH(AT(word * 64 + lane), seed));                                                 \
        }                                                                                                              \
    }

#define HASH_WORDS(AT, HASH, STORE)                                                                                    \
    for (uint32_t word = 0; word < words.word_count; ++word) {                                                         \
        const uint64_t valid = hash_word_valid(&words, word);                                                          \
        const uint32_t lanes = hash_word_lanes(&words, word);                                                          \
        if (valid == hash_word_tail(&words, word)) {                                                                   \
            for (uint32_t lane = 0; lane < lanes; ++lane) {                                                            \
                STORE(word * 64 + lane, HASH(AT(word * 64 + lane), seed));                                             \
            }                                                                                                          \
        } else if (valid == 0) {                                                                                       \
            for (uint32_t lane = 0; lane < lanes; ++lane) {                                                            \
                STORE(word * 64 + lane, HASH_NULL_VALUE);                                                              \
            }                                                                                                          \
        } else {                                                                                                       \
            HASH_MIXED_WALK(AT, HASH, STORE)                                                                           \
        }                                                                                                              \
    }

#define HASH_SHAPES(HASH, STORE)                                                                                       \
    switch (source->shape) {                                                                                           \
    case KERNEL_SHAPE_FLAT:                                                                                            \
        if (words.validity == nullptr) {                                                                               \
            for (uint32_t row = 0; row < words.count; ++row) {                                                         \
                STORE(row, HASH(AT_FLAT(row), seed));                                                                  \
            }                                                                                                          \
        } else {                                                                                                       \
            HASH_WORDS(AT_FLAT, HASH, STORE)                                                                           \
        }                                                                                                              \
        break;                                                                                                         \
    case KERNEL_SHAPE_CONSTANT: {                                                                                      \
        const uint64_t source_hash =                                                                                   \
            kernel_packet_is_valid_constant(source) ? HASH(hash_load(values, offset), seed) : HASH_NULL_VALUE;         \
        for (uint32_t row = 0; row < words.count; ++row) {                                                             \
            STORE(row, source_hash);                                                                                   \
        }                                                                                                              \
        break;                                                                                                         \
    }                                                                                                                  \
    case KERNEL_SHAPE_GATHERED:                                                                                        \
        HASH_WORDS(AT_MAP, HASH, STORE)                                                                                \
        break;                                                                                                         \
    }

#define HASH_PRIORS(BODY, HASH)                                                                                        \
    {                                                                                                                  \
        if (prior == nullptr) {                                                                                        \
            BODY(HASH, STORE_PLAIN)                                                                                    \
            return;                                                                                                    \
        }                                                                                                              \
        const uint8_t* const prior_values = (const uint8_t*)prior->values + prior->value_offset * sizeof(uint64_t);    \
        switch (prior->shape) {                                                                                        \
        case KERNEL_SHAPE_FLAT:                                                                                        \
            BODY(HASH, STORE_FLAT)                                                                                     \
            break;                                                                                                     \
        case KERNEL_SHAPE_CONSTANT: {                                                                                  \
            const uint64_t constant_hash = type_load_u64(prior_values, 0);                                             \
            BODY(HASH, STORE_CONSTANT)                                                                                 \
            break;                                                                                                     \
        }                                                                                                              \
        case KERNEL_SHAPE_GATHERED: {                                                                                  \
            const uint32_t* const prior_rows = prior->rows;                                                            \
            BODY(HASH, STORE_MAP)                                                                                      \
            break;                                                                                                     \
        }                                                                                                              \
        }                                                                                                              \
    }

#define DEFINE_HASH_VALUES(NAME, T, HASH)                                                                              \
    static void hash_values_##NAME(const struct kernel_packet* source,                                                 \
                                   uint64_t seed,                                                                      \
                                   const struct kernel_packet* prior,                                                  \
                                   uint64_t* out_hashes) {                                                             \
        auto const hash_load = TYPE_LOADER(T);                                                                         \
        const uint8_t* const values = source->values;                                                                  \
        const uint64_t offset = source->value_offset;                                                                  \
        const struct hash_words words = hash_words_of(source);                                                         \
        HASH_PRIORS(HASH_SHAPES, HASH)                                                                                 \
    }

#define DEFINE_HASH_LEAVES(NAME)                                                                                       \
    static int hash_##NAME(const struct kernel_packet* source,                                                         \
                           const struct kernel_unary_binding* binding,                                                 \
                           struct kernel_result* out_result) {                                                         \
        const uint64_t seed = binding->request.seed;                                                                   \
        hash_values_##NAME(source, seed, nullptr, out_result->values);                                                 \
        out_result->null_count = 0;                                                                                    \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }                                                                                                                  \
    static int hash_combine_##NAME(const struct kernel_packet* left,                                                   \
                                   const struct kernel_packet* right,                                                  \
                                   const struct kernel_binary_binding* binding,                                        \
                                   struct kernel_result* out_result) {                                                 \
        if (left->count != right->count) {                                                                             \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        const uint64_t seed = binding->request.seed;                                                                   \
        hash_values_##NAME(right, seed, left, out_result->values);                                                     \
        out_result->null_count = 0;                                                                                    \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }

//------------------------------------------------------------------------------
//
// Leaves
//
//------------------------------------------------------------------------------

//--------------------------------------
// Fixed Carriers
//--------------------------------------

// Signed integers, unsigned integers, floating-point types, Decimal, temporal types, intervals
#define DEFINE_HASH_FIXED(S, T, P)                                                                                     \
    DEFINE_HASH_VALUES(S, T, hash_value_##S)                                                                           \
    DEFINE_HASH_LEAVES(S)
FIXED_CARRIERS(DEFINE_HASH_FIXED)
DEFINE_HASH_FIXED(interval_year_month, int32_t, DORA_PHYSICAL_TYPE_I32)
#undef DEFINE_HASH_FIXED

//--------------------------------------
// Bytes, Boolean and Null
//--------------------------------------

// Utf8, Binary, FixedBinary
DEFINE_HASH_VALUES(bytes, struct dora_bytes, hash_value_bytes)
DEFINE_HASH_LEAVES(bytes)

#define HASH_BOOL_WORDS(HASH, STORE)                                                                                   \
    if (source->shape == KERNEL_SHAPE_CONSTANT) {                                                                      \
        const uint64_t h = kernel_packet_is_valid_constant(source)                                                     \
                               ? (HASH)[dora_bitmap_test(source->values, source->value_offset)]                        \
                               : HASH_NULL_VALUE;                                                                      \
        for (uint32_t row = 0; row < words.count; ++row) {                                                             \
            STORE(row, h);                                                                                             \
        }                                                                                                              \
    } else if (words.validity == nullptr && source->shape == KERNEL_SHAPE_FLAT) {                                      \
        for (uint32_t word = 0; word < words.word_count; ++word) {                                                     \
            const uint64_t bits = kernel_packet_bits_word(source, word, UINT64_MAX);                                   \
            const uint32_t lanes = hash_word_lanes(&words, word);                                                      \
            for (uint32_t lane = 0; lane < lanes; ++lane) {                                                            \
                STORE(word * 64 + lane, (HASH)[bitmap_word_bit(bits, lane)]);                                          \
            }                                                                                                          \
        }                                                                                                              \
    } else {                                                                                                           \
        for (uint32_t word = 0; word < words.word_count; ++word) {                                                     \
            const uint64_t valid = hash_word_valid(&words, word);                                                      \
            const uint64_t bits = kernel_packet_bits_word(source, word, valid);                                        \
            const uint32_t lanes = hash_word_lanes(&words, word);                                                      \
            for (uint32_t lane = 0; lane < lanes; ++lane) {                                                            \
                STORE(word * 64 + lane,                                                                                \
                      bitmap_word_bit(valid, lane) != 0 ? (HASH)[bitmap_word_bit(bits, lane)] : HASH_NULL_VALUE);      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define HASH_EVERY_ROW(HASH, STORE)                                                                                    \
    for (uint32_t row = 0; row < words.count; ++row) {                                                                 \
        STORE(row, HASH);                                                                                              \
    }

static void hash_values_bool(const struct kernel_packet* source,
                             uint64_t seed,
                             const struct kernel_packet* prior,
                             uint64_t* restrict out_hashes) {
    const uint64_t hashes[2] = {hash_value_bool(false, seed), hash_value_bool(true, seed)};
    const struct hash_words words = hash_words_of(source);
    HASH_PRIORS(HASH_BOOL_WORDS, hashes)
}

DEFINE_HASH_LEAVES(bool)

// Null hashes ignore the seed.
static void hash_values_null(const struct kernel_packet* source,
                             uint64_t,
                             const struct kernel_packet* prior,
                             uint64_t* out_hashes) {
    const struct hash_words words = hash_words_of(source);
    HASH_PRIORS(HASH_EVERY_ROW, HASH_NULL_VALUE)
}

DEFINE_HASH_LEAVES(null)

//--------------------------------------
// Repeated Gathered Bytes
//--------------------------------------

#define AT_LANE(ROW) (ROW)
#define HASH_CACHED(LANE, SEED) hash_cached(values, cache, LANE, SEED)
#define HASH_CACHED_WORDS(HASH, STORE) HASH_WORDS(AT_LANE, HASH, STORE)

static inline uint64_t
hash_cached(const struct dora_bytes* values, const struct kernel_hash_cache* cache, uint32_t lane, uint64_t seed) {
    uint64_t hash = cache->entries[cache->base_rows[lane]];
    if (hash == HASH_NULL_VALUE) {
        hash = hash_value_bytes(values[lane], seed);
        cache->entries[cache->base_rows[lane]] = hash;
    }
    return hash;
}

// Utf8, Binary, FixedBinary
void dora_kernel_hash_bytes_cached(const struct kernel_packet* source,
                                   const struct kernel_hash_cache* cache,
                                   uint64_t seed,
                                   const struct kernel_packet* prior,
                                   uint64_t* out_hashes) {
    const struct dora_bytes* const values = (const struct dora_bytes*)source->values + source->value_offset;
    const struct hash_words words = hash_words_of(source);
    HASH_PRIORS(HASH_CACHED_WORDS, HASH_CACHED)
}

#undef HASH_CACHED_WORDS
#undef HASH_CACHED
#undef AT_LANE
#undef HASH_EVERY_ROW
#undef HASH_BOOL_WORDS
#undef DEFINE_HASH_LEAVES
#undef DEFINE_HASH_VALUES
#undef HASH_PRIORS
#undef HASH_SHAPES
#undef HASH_WORDS
#undef HASH_MIXED_WALK
#undef STORE_MAP
#undef STORE_CONSTANT
#undef STORE_FLAT
#undef STORE_PLAIN
#undef AT_MAP
#undef AT_FLAT

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

enum hash_tag : uint8_t {
    HASH_TAG_I8,
    HASH_TAG_I16,
    HASH_TAG_I32,
    HASH_TAG_I64,
    HASH_TAG_I128,
    HASH_TAG_I256,
    HASH_TAG_U8,
    HASH_TAG_U16,
    HASH_TAG_U32,
    HASH_TAG_U64,
    HASH_TAG_U128,
    HASH_TAG_F16,
    HASH_TAG_F32,
    HASH_TAG_F64,
    HASH_TAG_INTERVAL_YEAR_MONTH,
    HASH_TAG_INTERVAL_DAY_TIME,
    HASH_TAG_INTERVAL_MONTH_DAY_NANO,
    HASH_TAG_BYTES,
    HASH_TAG_BOOL,
    HASH_TAG_NULL,
    HASH_TAG_COUNT
};

struct hash_bound {
    enum hash_tag tag;
    uint32_t width;
};

#define HASH_ROW(PREFIX)                                                                                               \
    {                                                                                                                  \
        [HASH_TAG_I8] = PREFIX##_i8,                                                                                   \
        [HASH_TAG_I16] = PREFIX##_i16,                                                                                 \
        [HASH_TAG_I32] = PREFIX##_i32,                                                                                 \
        [HASH_TAG_I64] = PREFIX##_i64,                                                                                 \
        [HASH_TAG_I128] = PREFIX##_i128,                                                                               \
        [HASH_TAG_I256] = PREFIX##_i256,                                                                               \
        [HASH_TAG_U8] = PREFIX##_u8,                                                                                   \
        [HASH_TAG_U16] = PREFIX##_u16,                                                                                 \
        [HASH_TAG_U32] = PREFIX##_u32,                                                                                 \
        [HASH_TAG_U64] = PREFIX##_u64,                                                                                 \
        [HASH_TAG_U128] = PREFIX##_u128,                                                                               \
        [HASH_TAG_F16] = PREFIX##_f16,                                                                                 \
        [HASH_TAG_F32] = PREFIX##_f32,                                                                                 \
        [HASH_TAG_F64] = PREFIX##_f64,                                                                                 \
        [HASH_TAG_INTERVAL_YEAR_MONTH] = PREFIX##_interval_year_month,                                                 \
        [HASH_TAG_INTERVAL_DAY_TIME] = PREFIX##_interval_day_time,                                                     \
        [HASH_TAG_INTERVAL_MONTH_DAY_NANO] = PREFIX##_interval_month_day_nano,                                         \
        [HASH_TAG_BYTES] = PREFIX##_bytes,                                                                             \
        [HASH_TAG_BOOL] = PREFIX##_bool,                                                                               \
        [HASH_TAG_NULL] = PREFIX##_null,                                                                               \
    }

static typeof(&hash_i8) const hash_leaves[HASH_TAG_COUNT] = HASH_ROW(hash);
static typeof(&hash_combine_i8) const hash_combine_leaves[HASH_TAG_COUNT] = HASH_ROW(hash_combine);

#undef HASH_ROW

static constexpr struct hash_bound hash_carriers[DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO + 1] = {
    [DORA_PHYSICAL_TYPE_I8] = {.tag = HASH_TAG_I8, .width = sizeof(int8_t)},
    [DORA_PHYSICAL_TYPE_I16] = {.tag = HASH_TAG_I16, .width = sizeof(int16_t)},
    [DORA_PHYSICAL_TYPE_I32] = {.tag = HASH_TAG_I32, .width = sizeof(int32_t)},
    [DORA_PHYSICAL_TYPE_I64] = {.tag = HASH_TAG_I64, .width = sizeof(int64_t)},
    [DORA_PHYSICAL_TYPE_I128] = {.tag = HASH_TAG_I128, .width = sizeof(struct dora_i128)},
    [DORA_PHYSICAL_TYPE_I256] = {.tag = HASH_TAG_I256, .width = sizeof(struct dora_i256)},
    [DORA_PHYSICAL_TYPE_U8] = {.tag = HASH_TAG_U8, .width = sizeof(uint8_t)},
    [DORA_PHYSICAL_TYPE_U16] = {.tag = HASH_TAG_U16, .width = sizeof(uint16_t)},
    [DORA_PHYSICAL_TYPE_U32] = {.tag = HASH_TAG_U32, .width = sizeof(uint32_t)},
    [DORA_PHYSICAL_TYPE_U64] = {.tag = HASH_TAG_U64, .width = sizeof(uint64_t)},
    [DORA_PHYSICAL_TYPE_U128] = {.tag = HASH_TAG_U128, .width = sizeof(struct dora_u128)},
    [DORA_PHYSICAL_TYPE_F16] = {.tag = HASH_TAG_F16, .width = sizeof(struct dora_f16_bits)},
    [DORA_PHYSICAL_TYPE_F32] = {.tag = HASH_TAG_F32, .width = sizeof(float)},
    [DORA_PHYSICAL_TYPE_F64] = {.tag = HASH_TAG_F64, .width = sizeof(double)},
    [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] =
        {.tag = HASH_TAG_INTERVAL_DAY_TIME, .width = sizeof(struct dora_interval_day_time)},
    [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] =
        {.tag = HASH_TAG_INTERVAL_MONTH_DAY_NANO, .width = sizeof(struct dora_interval_month_day_nano)},
};

static int hash_bind_decimal(const struct dora_logical_type* type,
                             enum dora_physical_type physical,
                             struct hash_bound* out_bound) {
    struct decimal_prepared decimal = {};
    const int status = dora_decimal_prepare(type->parameters.decimal, physical, &decimal);
    if (status == DORA_OK) {
        *out_bound = hash_carriers[physical];
    }
    return status;
}

static int
hash_bind_type(const struct dora_logical_type* type, enum dora_physical_type physical, struct hash_bound* out_bound) {
    if (type == nullptr) {
        return EINVAL;
    }
    switch (type->id) {
    case DORA_TYPE_NULL:
        *out_bound = (struct hash_bound){.tag = HASH_TAG_NULL};
        return DORA_OK;
    case DORA_TYPE_BOOLEAN:
        *out_bound = (struct hash_bound){.tag = HASH_TAG_BOOL};
        return DORA_OK;
    case DORA_TYPE_UTF8:
    case DORA_TYPE_BINARY:
    case DORA_TYPE_FIXED_BINARY:
        *out_bound = (struct hash_bound){.tag = HASH_TAG_BYTES, .width = sizeof(struct dora_bytes)};
        return DORA_OK;
    case DORA_TYPE_DECIMAL:
        return hash_bind_decimal(type, physical, out_bound);
    default:
        break;
    }
    *out_bound = hash_carriers[physical];
    if (type->id == DORA_TYPE_INTERVAL_YEAR_MONTH) {
        out_bound->tag = HASH_TAG_INTERVAL_YEAR_MONTH;
    }
    return DORA_OK;
}

int dora_kernel_hash_bind(const struct kernel_bind_request* request, struct kernel_unary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    struct hash_bound bound = {};
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    int status = type_scalar_physical(request->input_type, &physical);
    if (status == DORA_OK) {
        status = hash_bind_type(request->input_type, physical, &bound);
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_binding = (struct kernel_unary_binding){
        .fill = hash_leaves[bound.tag],
        .request = *request,
        .result_type = {.id = DORA_TYPE_UINT64},
        .input_physical = physical,
        .input_byte_width = bound.width,
        .result_byte_width = sizeof(uint64_t),
        .is_result_all_valid = true,
    };
    out_binding->result_physical = DORA_PHYSICAL_TYPE_U64;
    return DORA_OK;
}

int dora_kernel_hash_combine_bind(const struct kernel_bind_request* request,
                                  struct kernel_binary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->input_type == nullptr ||
        request->input_type->id != DORA_TYPE_UINT64) {
        return EINVAL;
    }
    struct hash_bound bound = {};
    enum dora_physical_type left_physical = DORA_PHYSICAL_TYPE_NONE;
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    int status = type_scalar_physical(request->input_type, &left_physical);
    if (status == DORA_OK) {
        status = type_scalar_physical(request->right_type, &physical);
    }
    if (status == DORA_OK) {
        status = hash_bind_type(request->right_type, physical, &bound);
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_binding = (struct kernel_binary_binding){
        .fill = hash_combine_leaves[bound.tag],
        .request = *request,
        .result_type = {.id = DORA_TYPE_UINT64},
        .input_physical = left_physical,
        .right_physical = physical,
        .left_byte_width = sizeof(uint64_t),
        .right_byte_width = bound.width,
        .result_byte_width = sizeof(uint64_t),
        .is_result_all_valid = true,
    };
    out_binding->result_physical = DORA_PHYSICAL_TYPE_U64;
    return DORA_OK;
}
