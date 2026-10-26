// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/cpu/packet_bits.h"
#include "kernel/packet.h"
#include "numeric/arithmetic.h"
#include "numeric/decimal.h"
#include "types/load.h"
#include "types/order.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if DORA_KERNEL_USE_NEON && defined(__aarch64__) && defined(__ARM_NEON)
#define COMPARE_USE_NEON 1
#endif

//------------------------------------------------------------------------------
//
// Words
//
//------------------------------------------------------------------------------

struct compare_relation {
    bool can_match_less;
    bool can_match_equal;
    bool can_match_greater;
};

struct compare_leaf {
    uint64_t* matches;
    int8_t* orders;
    uint64_t* validity;
    uint64_t both_null;
    uint64_t left_null;
    uint64_t right_null;
    uint64_t null_valid;
    uint64_t last_tail;
    uint32_t count;
    uint32_t word_count;
    uint32_t null_count;
    int8_t left_null_order;
};

struct compare_word {
    uint64_t match;
    uint64_t left_valid;
    uint64_t right_valid;
    uint64_t tail;
};

static inline bool compare_values_are_live(const struct kernel_packet* packet) {
    return packet->shape != KERNEL_SHAPE_CONSTANT || kernel_packet_is_valid_constant(packet);
}

static inline struct compare_leaf compare_leaf_init(uint32_t count,
                                                    const struct kernel_binary_binding* binding,
                                                    struct kernel_result* out_result,
                                                    struct compare_relation relation) {
    const bool distinct = binding->request.null_order != DORA_NULL_ORDER_PROPAGATE;
    const bool last = binding->request.null_order == DORA_NULL_ORDER_LAST;
    return (struct compare_leaf){
        .matches = out_result->values,
        .orders = out_result->values,
        .validity = out_result->validity,
        .both_null = distinct && relation.can_match_equal ? UINT64_MAX : 0,
        .left_null =
            distinct && ((last && relation.can_match_greater) || (!last && relation.can_match_less)) ? UINT64_MAX : 0,
        .right_null =
            distinct && ((last && relation.can_match_less) || (!last && relation.can_match_greater)) ? UINT64_MAX : 0,
        .null_valid = distinct ? UINT64_MAX : 0,
        .last_tail = bitmap_tail_mask(count),
        .count = count,
        .word_count = (uint32_t)bitmap_words_for_bits(count),
        .left_null_order = last ? (int8_t)1 : (int8_t)-1,
    };
}

static inline struct compare_word compare_word_at(const struct compare_leaf* leaf,
                                                  const struct kernel_packet* left,
                                                  const struct kernel_packet* right,
                                                  uint32_t word) {
    const uint64_t tail = (word + 1 == leaf->word_count ? leaf->last_tail : UINT64_MAX);
    return (struct compare_word){
        .left_valid = left->validity == nullptr ? tail : left->validity[word] & tail,
        .right_valid = right->validity == nullptr ? tail : right->validity[word] & tail,
        .tail = tail,
    };
}

static inline void compare_store(struct compare_leaf* leaf, uint32_t word, struct compare_word value) {
    const uint64_t left_null = value.tail & ~value.left_valid;
    const uint64_t right_null = value.tail & ~value.right_valid;
    const uint64_t active = value.left_valid & value.right_valid;
    const uint64_t nulls = (left_null & right_null & leaf->both_null) | (left_null & ~right_null & leaf->left_null) |
                           (right_null & ~left_null & leaf->right_null);
    const uint64_t valid = active | ((left_null | right_null) & leaf->null_valid);
    leaf->matches[word] = (value.match & active) | nulls;
    if (leaf->validity != nullptr) {
        leaf->validity[word] = valid;
    }
    leaf->null_count += dora_bitmap_word_count_ones(value.tail & ~valid);
}

static inline void order_fill_lanes(int8_t* orders, uint64_t lanes, int8_t order) {
    while (lanes != 0) {
        orders[dora_bitmap_word_pop_lowest_set(&lanes)] = order;
    }
}

static inline void order_store(struct compare_leaf* leaf, uint32_t word, struct compare_word value) {
    const uint64_t left_null = value.tail & ~value.left_valid;
    const uint64_t right_null = value.tail & ~value.right_valid;
    const uint64_t valid = (value.left_valid & value.right_valid) | ((left_null | right_null) & leaf->null_valid);
    if (leaf->null_valid != 0 && (left_null | right_null) != 0) {
        int8_t* const orders = leaf->orders + (size_t)word * 64;
        order_fill_lanes(orders, left_null & right_null, 0);
        order_fill_lanes(orders, left_null & ~right_null, leaf->left_null_order);
        order_fill_lanes(orders, right_null & ~left_null, (int8_t)-leaf->left_null_order);
    }
    if (leaf->validity != nullptr) {
        leaf->validity[word] = valid;
    }
    leaf->null_count += dora_bitmap_word_count_ones(value.tail & ~valid);
}

static int compare_null(const struct kernel_packet* left,
                        const struct kernel_packet* right,
                        const struct kernel_binary_binding* binding,
                        struct kernel_result* out_result) {
    const enum dora_compare_op comparison = binding->request.comparison;
    const struct compare_relation relation = {
        .can_match_less = type_order_matches(-1, comparison),
        .can_match_equal = type_order_matches(0, comparison),
        .can_match_greater = type_order_matches(1, comparison),
    };
    struct compare_leaf leaf = compare_leaf_init(left->count, binding, out_result, relation);
    for (uint32_t word = 0; word < leaf.word_count; ++word) {
        struct compare_word value = compare_word_at(&leaf, left, right, word);
        value.match = relation.can_match_equal ? UINT64_MAX : 0;
        compare_store(&leaf, word, value);
    }
    out_result->null_count = leaf.null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

static int order_null(const struct kernel_packet* left,
                      const struct kernel_packet* right,
                      const struct kernel_binary_binding* binding,
                      struct kernel_result* out_result) {
    struct compare_leaf leaf = compare_leaf_init(left->count, binding, out_result, (struct compare_relation){});
    for (uint32_t word = 0; word < leaf.word_count; ++word) {
        const struct compare_word value = compare_word_at(&leaf, left, right, word);
        order_fill_lanes(leaf.orders + (size_t)word * 64, value.left_valid & value.right_valid, 0);
        order_store(&leaf, word, value);
    }
    out_result->null_count = leaf.null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Expansions
//
//------------------------------------------------------------------------------

#define LOAD_VALUE(SIDE, ELEMENT) compare_load(SIDE##_values, ELEMENT)
#define LOAD_I128(SIDE, ELEMENT) type_i128_load(LOAD_VALUE(SIDE, ELEMENT))
#define LOAD_U128(SIDE, ELEMENT) type_u128_load(LOAD_VALUE(SIDE, ELEMENT))
#define LOAD_BIT(SIDE, ELEMENT) dora_bitmap_test(SIDE##_values, ELEMENT)
#define LOAD_WIDE(SIDE, ELEMENT) compare_decimal_load(SIDE##_decimal->bit_width, SIDE##_values, ELEMENT)
#define LOAD_DIGITS(SIDE, ELEMENT) compare_decimal_digits(SIDE##_values, ELEMENT, SIDE##_decimal)
#define LOAD_COMMON(SIDE, ELEMENT) compare_common_load(SIDE##_values, ELEMENT, &SIDE##_common)

#define AT_FLAT(LOAD, SIDE, ROW) LOAD(SIDE, SIDE##_offset + (ROW))
#define AT_CONSTANT(LOAD, SIDE, ROW) LOAD(SIDE, SIDE##_offset)
#define AT_CACHED(LOAD, SIDE, ROW) SIDE##_cached
#define AT_GENERIC(LOAD, SIDE, ROW) LOAD(SIDE, kernel_row_index(SIDE##_offset, SIDE##_rows, SIDE##_mask, (ROW)))

#define TRUTH_LANE_HOLDS(ORDER, REL, LEFT, RIGHT) (ORDER(LEFT, RIGHT) REL 0)
#define NATIVE_LANE_HOLDS(ORDER, REL, LEFT, RIGHT) ((LEFT)REL(RIGHT))
#define RANGE_LANE_HOLDS(ORDER, REL, LEFT, RIGHT) ORDER(LEFT, lower_value, upper_value)
#define ORDER_LANE(ORDER, REL, LANE, ROW, LEFT, RIGHT) orders[ROW] = (int8_t)ORDER(LEFT, RIGHT)

#define ORDER_FINISH                                                                                                   \
    order_store(&leaf, word, (struct compare_word){.left_valid = left_valid, .right_valid = right_valid, .tail = tail})

#define COMPARE_RELATION(REL)                                                                                          \
    ((struct compare_relation){.can_match_less = -1 REL 0, .can_match_equal = 0 REL 0, .can_match_greater = 1 REL 0})

#define FUSED_MATCH(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, MATCH)                                                  \
    for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                    \
        const uint64_t holds = (uint64_t)(LANE##_HOLDS(ORDER,                                                          \
                                                       REL,                                                            \
                                                       LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),                \
                                                       RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane))              \
                                              ? 1                                                                      \
                                              : 0);                                                                    \
        (MATCH) |= holds << lane;                                                                                      \
    }

#ifdef COMPARE_USE_NEON
#define SPLIT_ALL_VALID(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, MATCH)                                              \
    {                                                                                                                  \
        if (lanes == 64) {                                                                                             \
            alignas(16) int8_t result_data[64];                                                                        \
            for (uint32_t lane = 0; lane < 64; ++lane) {                                                               \
                result_data[lane] = (int8_t)(LANE##_HOLDS(ORDER,                                                       \
                                                          REL,                                                         \
                                                          LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),             \
                                                          RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane))           \
                                                 ? -1                                                                  \
                                                 : 0);                                                                 \
            }                                                                                                          \
            (MATCH) = bitmap_pack_nonzero_bytes_word((const uint8_t*)result_data);                                     \
        } else {                                                                                                       \
            FUSED_MATCH(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, MATCH)                                              \
        }                                                                                                              \
    }

#else
#define SPLIT_ALL_VALID(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, MATCH)                                              \
    {                                                                                                                  \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            truth[lane] = (uint8_t)LANE##_HOLDS(ORDER,                                                                 \
                                                REL,                                                                   \
                                                LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),                       \
                                                RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane));                    \
        }                                                                                                              \
        (MATCH) = bitmap_pack_bool_bytes_word(truth);                                                                  \
    }

#endif

#define COMPARE_SPLIT_LANES(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL)                                                 \
    if (active == tail) {                                                                                              \
        SPLIT_ALL_VALID(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, match)                                              \
    } else if (active != 0) {                                                                                          \
        FUSED_MATCH(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, match)                                                  \
    }

#define COMPARE_MASKED_LANES(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL)                                                \
    if (active == tail) {                                                                                              \
        SPLIT_ALL_VALID(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, match)                                              \
    } else if (active != 0) {                                                                                          \
        uint64_t pending = active;                                                                                     \
        while (pending != 0) {                                                                                         \
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);                                           \
            const uint64_t holds = LANE##_HOLDS(ORDER,                                                                 \
                                                REL,                                                                   \
                                                LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),                       \
                                                RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane))                     \
                                       ? 1                                                                             \
                                       : 0;                                                                            \
            match |= holds << lane;                                                                                    \
        }                                                                                                              \
    }

#define ORDER_LANES(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL)                                                         \
    if (active == tail) {                                                                                              \
        for (uint64_t lane = 0; lane < lanes; ++lane) {                                                                \
            LANE(ORDER,                                                                                                \
                 REL,                                                                                                  \
                 lane,                                                                                                 \
                 (uint64_t)word * 64 + lane,                                                                           \
                 LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),                                                      \
                 RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane));                                                   \
        }                                                                                                              \
    } else if (active != 0) {                                                                                          \
        uint64_t pending = active;                                                                                     \
        while (pending != 0) {                                                                                         \
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);                                           \
            LANE(ORDER,                                                                                                \
                 REL,                                                                                                  \
                 lane,                                                                                                 \
                 (uint64_t)word * 64 + lane,                                                                           \
                 LEFT_AT(LOAD, left, (uint64_t)word * 64 + lane),                                                      \
                 RIGHT_AT(LOAD, right, (uint64_t)word * 64 + lane));                                                   \
        }                                                                                                              \
    }

#define COMPARE_CONSTANT(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL)                                                    \
    if (active != 0) {                                                                                                 \
        match = LANE##_HOLDS(ORDER, REL, LEFT_AT(LOAD, left, 0), RIGHT_AT(LOAD, right, 0)) ? tail : 0;                 \
    }

#define TRUTH_LANE_BEGIN uint64_t match = 0
#define NATIVE_LANE_BEGIN uint64_t match = 0
#define RANGE_LANE_BEGIN uint64_t match = 0
#define ORDER_LANE_BEGIN
#define COMPARE_FINISH                                                                                                 \
    compare_store(                                                                                                     \
        &leaf,                                                                                                         \
        word,                                                                                                          \
        (struct compare_word){.match = match, .left_valid = left_valid, .right_valid = right_valid, .tail = tail})

#ifdef COMPARE_USE_NEON
#define PROLOGUE_TRUTH_LANE
#define PROLOGUE_NATIVE_LANE
#define PROLOGUE_RANGE_LANE
#else
#define PROLOGUE_TRUTH_LANE uint8_t truth[64] = {};
#define PROLOGUE_NATIVE_LANE uint8_t truth[64] = {};
#define PROLOGUE_RANGE_LANE uint8_t truth[64] = {};
#endif
#define PROLOGUE_ORDER_LANE

#define COMPARE_SPLIT_LANES_BEGIN const uint32_t lanes = word + 1 == leaf.word_count ? leaf.count - word * 64 : 64
#define COMPARE_MASKED_LANES_BEGIN COMPARE_SPLIT_LANES_BEGIN
#define ORDER_LANES_BEGIN COMPARE_SPLIT_LANES_BEGIN
#define COMPARE_CONSTANT_BEGIN

#define COMPARE_WORD_LOOP(WORDS, LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL, FINISH)                                    \
    for (uint32_t word = 0; word < leaf.word_count; ++word) {                                                          \
        const uint64_t tail = word + 1 == leaf.word_count ? leaf.last_tail : UINT64_MAX;                               \
        const uint64_t left_valid = left_validity == nullptr ? tail : left_validity[word] & tail;                      \
        const uint64_t right_valid = right_validity == nullptr ? tail : right_validity[word] & tail;                   \
        const uint64_t active = left_valid & right_valid;                                                              \
        WORDS##_BEGIN;                                                                                                 \
        LANE##_BEGIN;                                                                                                  \
        WORDS(LEFT_AT, RIGHT_AT, LOAD, LANE, ORDER, REL)                                                               \
        FINISH;                                                                                                        \
    }

#define COMPARE_EXPANSIONS(WORDS, MASKED_WORDS, CONSTANT_WORD, LOAD, CONSTANT_AT, LANE, ORDER, REL, FINISH)            \
    do {                                                                                                               \
        PROLOGUE_##LANE const enum kernel_shape_pair shapes = kernel_shape_pair_of(left->shape, right->shape);         \
        switch (shapes) {                                                                                              \
        case KERNEL_PAIR_FLAT_FLAT:                                                                                    \
            COMPARE_WORD_LOOP(WORDS, AT_FLAT, AT_FLAT, LOAD, LANE, ORDER, REL, FINISH)                                 \
            break;                                                                                                     \
        case KERNEL_PAIR_FLAT_CONSTANT:                                                                                \
            COMPARE_WORD_LOOP(WORDS, AT_FLAT, CONSTANT_AT, LOAD, LANE, ORDER, REL, FINISH)                             \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_FLAT:                                                                                \
            COMPARE_WORD_LOOP(WORDS, CONSTANT_AT, AT_FLAT, LOAD, LANE, ORDER, REL, FINISH)                             \
            break;                                                                                                     \
        case KERNEL_PAIR_CONSTANT_CONSTANT:                                                                            \
            COMPARE_WORD_LOOP(CONSTANT_WORD, CONSTANT_AT, CONSTANT_AT, LOAD, LANE, ORDER, REL, FINISH)                 \
            break;                                                                                                     \
        case KERNEL_PAIR_GENERIC:                                                                                      \
            COMPARE_WORD_LOOP(MASKED_WORDS, AT_GENERIC, AT_GENERIC, LOAD, LANE, ORDER, REL, FINISH)                    \
            break;                                                                                                     \
        }                                                                                                              \
    } while (0)

#define COMPARE_VIEWS                                                                                                  \
    const uint8_t* const left_values = left->values;                                                                   \
    const uint8_t* const right_values = right->values;                                                                 \
    const uint64_t left_offset = left->value_offset;                                                                   \
    const uint64_t right_offset = right->value_offset;                                                                 \
    const uint32_t* const left_rows = left->shape == KERNEL_SHAPE_GATHERED ? left->rows : nullptr;                     \
    const uint32_t* const right_rows = right->shape == KERNEL_SHAPE_GATHERED ? right->rows : nullptr;                  \
    const uint32_t left_mask = left->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                      \
    const uint32_t right_mask = right->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                    \
    const uint64_t* const left_validity = left->validity;                                                              \
    const uint64_t* const right_validity = right->validity

#define COMPARE_SIDES                                                                                                  \
    COMPARE_VIEWS;                                                                                                     \
    constexpr int sides_status = DORA_OK

#define COMPARE_LOADED_SIDES(T)                                                                                        \
    auto const compare_load = TYPE_LOADER(T);                                                                          \
    const uint8_t* const left_values =                                                                                 \
        left->values == nullptr ? nullptr : (const uint8_t*)left->values + left->value_offset * sizeof(T);             \
    const uint8_t* const right_values =                                                                                \
        right->values == nullptr ? nullptr : (const uint8_t*)right->values + right->value_offset * sizeof(T);          \
    constexpr uint64_t left_offset = 0;                                                                                \
    constexpr uint64_t right_offset = 0;                                                                               \
    const uint32_t* const left_rows = left->shape == KERNEL_SHAPE_GATHERED ? left->rows : nullptr;                     \
    const uint32_t* const right_rows = right->shape == KERNEL_SHAPE_GATHERED ? right->rows : nullptr;                  \
    const uint32_t left_mask = left->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                      \
    const uint32_t right_mask = right->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                    \
    const uint64_t* const left_validity = left->validity;                                                              \
    const uint64_t* const right_validity = right->validity;                                                            \
    constexpr int sides_status = DORA_OK

// Byte outputs may alias constant storage.
#define COMPARE_CACHED_SIDES(T)                                                                                        \
    auto const no_constant = (T){};                                                                                    \
    COMPARE_LOADED_SIDES(T);                                                                                           \
    const T left_constant = left->shape == KERNEL_SHAPE_CONSTANT && compare_values_are_live(left)                      \
                                ? compare_load(left_values, 0)                                                         \
                                : no_constant;                                                                         \
    const T right_constant = right->shape == KERNEL_SHAPE_CONSTANT && compare_values_are_live(right)                   \
                                 ? compare_load(right_values, 0)                                                       \
                                 : no_constant
#define CACHED_LOAD_VALUE(SIDE) SIDE##_constant
#define CACHED_LOAD_I128(SIDE) type_i128_load(SIDE##_constant)
#define AT_TYPED_CONSTANT(LOAD, SIDE, ROW) CACHED_##LOAD(SIDE)

#define COMPARE_DECIMAL_VIEWS                                                                                          \
    COMPARE_VIEWS;                                                                                                     \
    const struct dora_decimal_type* const left_decimal = &binding->left_decimal.type;                                  \
    const struct dora_decimal_type* const right_decimal = &binding->right_decimal.type

#define COMPARE_DECIMAL_SIDES                                                                                          \
    COMPARE_DECIMAL_VIEWS;                                                                                             \
    constexpr int sides_status = DORA_OK

#define COMPARE_DIGIT_SIDES                                                                                            \
    COMPARE_DECIMAL_VIEWS;                                                                                             \
    const struct type_decimal_digits left_cached = compare_cached_digits(left, left_decimal);                          \
    const struct type_decimal_digits right_cached = compare_cached_digits(right, right_decimal);                       \
    constexpr int sides_status = DORA_OK

#define COMPARE_COMMON_SIDES                                                                                           \
    COMPARE_VIEWS;                                                                                                     \
    const struct compare_common left_common =                                                                          \
        compare_common_side(&binding->left_decimal, binding->result_decimal.type);                                     \
    const struct compare_common right_common =                                                                         \
        compare_common_side(&binding->right_decimal, binding->result_decimal.type);                                    \
    const bool common_live = compare_values_are_live(left) && compare_values_are_live(right);                          \
    const int sides_status = compare_common_check(left, &left_common, common_live) != DORA_OK ||                       \
                                     compare_common_check(right, &right_common, common_live) != DORA_OK                \
                                 ? ERANGE                                                                              \
                                 : DORA_OK

#define DEFINE_COMPARE_LEAF(NAME, WORDS, LANE, SIDES, LOAD, CONSTANT_AT, ORDER, REL)                                   \
    static int compare_##NAME(const struct kernel_packet* left,                                                        \
                              const struct kernel_packet* right,                                                       \
                              const struct kernel_binary_binding* binding,                                             \
                              struct kernel_result* out_result) {                                                      \
        struct compare_leaf leaf = compare_leaf_init(left->count, binding, out_result, COMPARE_RELATION(REL));         \
        if (!compare_values_are_live(left) || !compare_values_are_live(right)) {                                       \
            return compare_null(left, right, binding, out_result);                                                     \
        }                                                                                                              \
        SIDES;                                                                                                         \
        if (sides_status != DORA_OK) {                                                                                 \
            return sides_status;                                                                                       \
        }                                                                                                              \
        COMPARE_EXPANSIONS(WORDS,                                                                                      \
                           COMPARE_MASKED_LANES,                                                                       \
                           COMPARE_CONSTANT,                                                                           \
                           LOAD,                                                                                       \
                           CONSTANT_AT,                                                                                \
                           LANE,                                                                                       \
                           ORDER,                                                                                      \
                           REL,                                                                                        \
                           COMPARE_FINISH);                                                                            \
        out_result->null_count = leaf.null_count;                                                                      \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }

#define DEFINE_ORDER_LEAF(NAME, SIDES, LOAD, CONSTANT_AT, ORDER)                                                       \
    static int order_##NAME(const struct kernel_packet* left,                                                          \
                            const struct kernel_packet* right,                                                         \
                            const struct kernel_binary_binding* binding,                                               \
                            struct kernel_result* out_result) {                                                        \
        struct compare_leaf leaf = compare_leaf_init(left->count, binding, out_result, (struct compare_relation){});   \
        if (!compare_values_are_live(left) || !compare_values_are_live(right)) {                                       \
            return order_null(left, right, binding, out_result);                                                       \
        }                                                                                                              \
        SIDES;                                                                                                         \
        if (sides_status != DORA_OK) {                                                                                 \
            return sides_status;                                                                                       \
        }                                                                                                              \
        int8_t* const orders = leaf.orders;                                                                            \
        COMPARE_EXPANSIONS(ORDER_LANES,                                                                                \
                           ORDER_LANES,                                                                                \
                           ORDER_LANES,                                                                                \
                           LOAD,                                                                                       \
                           CONSTANT_AT,                                                                                \
                           ORDER_LANE,                                                                                 \
                           ORDER,                                                                                      \
                           ==,                                                                                         \
                           ORDER_FINISH);                                                                              \
        out_result->null_count = leaf.null_count;                                                                      \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }

#define DEFINE_COMPARE_FAMILY(NAME, WORDS, SIDES, LOAD, CONSTANT_AT, EQUALITY, ORDER)                                  \
    DEFINE_COMPARE_LEAF(eq_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, EQUALITY, ==)                          \
    DEFINE_COMPARE_LEAF(ne_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, EQUALITY, !=)                          \
    DEFINE_COMPARE_LEAF(lt_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, ORDER, <)                              \
    DEFINE_COMPARE_LEAF(le_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, ORDER, <=)                             \
    DEFINE_COMPARE_LEAF(gt_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, ORDER, >)                              \
    DEFINE_COMPARE_LEAF(ge_##NAME, WORDS, TRUTH_LANE, SIDES, LOAD, CONSTANT_AT, ORDER, >=)                             \
    DEFINE_ORDER_LEAF(NAME, SIDES, LOAD, CONSTANT_AT, ORDER)

//------------------------------------------------------------------------------
//
// Leaves
//
//------------------------------------------------------------------------------

//--------------------------------------
// Fixed Carriers
//--------------------------------------

#define COMPARE_FIXED_TAGS(X)                                                                                          \
    X(i8, int8_t, COMPARE_SPLIT_LANES)                                                                                 \
    X(i16, int16_t, COMPARE_SPLIT_LANES)                                                                               \
    X(i32, int32_t, COMPARE_SPLIT_LANES)                                                                               \
    X(i64, int64_t, COMPARE_SPLIT_LANES)                                                                               \
    X(i256, struct dora_i256, COMPARE_SPLIT_LANES)                                                                     \
    X(u8, uint8_t, COMPARE_SPLIT_LANES)                                                                                \
    X(u16, uint16_t, COMPARE_SPLIT_LANES)                                                                              \
    X(u32, uint32_t, COMPARE_SPLIT_LANES)                                                                              \
    X(u64, uint64_t, COMPARE_SPLIT_LANES)                                                                              \
    X(f16, struct dora_f16_bits, COMPARE_SPLIT_LANES)                                                                  \
    X(f32, float, COMPARE_SPLIT_LANES)                                                                                 \
    X(f64, double, COMPARE_SPLIT_LANES)                                                                                \
    X(interval_day_time, struct dora_interval_day_time, COMPARE_SPLIT_LANES)                                           \
    X(interval_month_day_nano, struct dora_interval_month_day_nano, COMPARE_SPLIT_LANES)

// Integers except Int128, floating-point types, Decimal (32/64/256-bit), temporal types, intervals
#define DEFINE_COMPARE_FIXED(S, T, WORDS)                                                                              \
    DEFINE_COMPARE_FAMILY(S,                                                                                           \
                          WORDS,                                                                                       \
                          COMPARE_CACHED_SIDES(T),                                                                     \
                          LOAD_VALUE,                                                                                  \
                          AT_TYPED_CONSTANT,                                                                           \
                          type_order_##S,                                                                              \
                          type_order_##S)
COMPARE_FIXED_TAGS(DEFINE_COMPARE_FIXED)
DEFINE_COMPARE_FIXED(u128, struct dora_u128, COMPARE_SPLIT_LANES)
#undef DEFINE_COMPARE_FIXED

// Int128, Decimal (128-bit)
#define DEFINE_COMPARE_NATIVE(NAME, OP, REL, T, LOAD, ORDER)                                                           \
    DEFINE_COMPARE_LEAF(OP##_##NAME,                                                                                   \
                        COMPARE_SPLIT_LANES,                                                                           \
                        NATIVE_LANE,                                                                                   \
                        COMPARE_CACHED_SIDES(T),                                                                       \
                        LOAD,                                                                                          \
                        AT_TYPED_CONSTANT,                                                                             \
                        ORDER,                                                                                         \
                        REL)

#define DEFINE_COMPARE_NATIVES(NAME, T, LOAD, ORDER)                                                                   \
    DEFINE_COMPARE_NATIVE(NAME, eq, ==, T, LOAD, ORDER)                                                                \
    DEFINE_COMPARE_NATIVE(NAME, ne, !=, T, LOAD, ORDER)                                                                \
    DEFINE_COMPARE_NATIVE(NAME, lt, <, T, LOAD, ORDER)                                                                 \
    DEFINE_COMPARE_NATIVE(NAME, le, <=, T, LOAD, ORDER)                                                                \
    DEFINE_COMPARE_NATIVE(NAME, gt, >, T, LOAD, ORDER)                                                                 \
    DEFINE_COMPARE_NATIVE(NAME, ge, >=, T, LOAD, ORDER)                                                                \
    DEFINE_ORDER_LEAF(NAME, COMPARE_CACHED_SIDES(T), LOAD_VALUE, AT_TYPED_CONSTANT, ORDER)

DEFINE_COMPARE_NATIVES(i128, struct dora_i128, LOAD_I128, type_order_i128)

#undef DEFINE_COMPARE_NATIVES
#undef DEFINE_COMPARE_NATIVE

//--------------------------------------
// Boolean
//--------------------------------------

#define DEFINE_BOOLEAN_COMPARE(OP, REL, FORMULA)                                                                       \
    static int compare_##OP##_bool(const struct kernel_packet* left,                                                   \
                                   const struct kernel_packet* right,                                                  \
                                   const struct kernel_binary_binding* binding,                                        \
                                   struct kernel_result* out_result) {                                                 \
        struct compare_leaf leaf = compare_leaf_init(left->count, binding, out_result, COMPARE_RELATION(REL));         \
        for (uint32_t word = 0; word < leaf.word_count; ++word) {                                                      \
            struct compare_word value = compare_word_at(&leaf, left, right, word);                                     \
            const uint64_t active = value.left_valid & value.right_valid;                                              \
            const uint64_t a = kernel_packet_bits_word(left, word, active);                                            \
            const uint64_t b = kernel_packet_bits_word(right, word, active);                                           \
            value.match = FORMULA;                                                                                     \
            compare_store(&leaf, word, value);                                                                         \
        }                                                                                                              \
        out_result->null_count = leaf.null_count;                                                                      \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }
DEFINE_BOOLEAN_COMPARE(eq, ==, ~(a ^ b))
DEFINE_BOOLEAN_COMPARE(ne, !=, a ^ b)
DEFINE_BOOLEAN_COMPARE(lt, <, ~a& b)
DEFINE_BOOLEAN_COMPARE(le, <=, ~a | b)
DEFINE_BOOLEAN_COMPARE(gt, >, a & ~b)
DEFINE_BOOLEAN_COMPARE(ge, >=, a | ~b)
#undef DEFINE_BOOLEAN_COMPARE

DEFINE_ORDER_LEAF(bool, COMPARE_SIDES, LOAD_BIT, AT_CONSTANT, type_order_bool)

//--------------------------------------
// Bytes and Null
//--------------------------------------

static inline int compare_bytes_equality(struct dora_bytes left, struct dora_bytes right) {
    return bytes_are_equal(left, right) ? 0 : 1;
}

// Utf8, Binary, FixedBinary
DEFINE_COMPARE_FAMILY(bytes,
                      COMPARE_SPLIT_LANES,
                      COMPARE_CACHED_SIDES(struct dora_bytes),
                      LOAD_VALUE,
                      AT_TYPED_CONSTANT,
                      compare_bytes_equality,
                      type_order_bytes)

//--------------------------------------
// Decimals
//--------------------------------------

static inline struct dora_i256 compare_decimal_load(uint32_t bit_width, const uint8_t* values, uint64_t element) {
    switch (bit_width) {
    case 32:
        return numeric_i256_from_i64(type_load_i32(values, element));
    case 64:
        return numeric_i256_from_i64(type_load_i64(values, element));
    case 128:
        return numeric_i256_from_i128(type_load_i128(values, element));
    default:
        return type_load_i256(values, element);
    }
}

static inline struct type_decimal_digits
compare_decimal_digits(const uint8_t* values, uint64_t element, const struct dora_decimal_type* decimal) {
    return type_decimal_digits((struct type_decimal_operand){
        .coefficient = compare_decimal_load(decimal->bit_width, values, element),
        .scale = decimal->scale,
    });
}

static inline struct type_decimal_digits compare_cached_digits(const struct kernel_packet* packet,
                                                               const struct dora_decimal_type* decimal) {
    return kernel_packet_is_valid_constant(packet)
               ? compare_decimal_digits(packet->values, packet->value_offset, decimal)
               : (struct type_decimal_digits){};
}

static inline int compare_order_digits(struct type_decimal_digits left, struct type_decimal_digits right) {
    return type_order_decimal_digits(&left, &right);
}

struct compare_common {
    struct dora_i256 bound;
    _BitInt(128) factor;
    uint32_t bit_width;
    bool is_scaled;
    bool should_check_bounds;
};

static inline struct compare_common compare_common_side(const struct decimal_prepared* side,
                                                        struct dora_decimal_type common) {
    const int32_t difference = common.scale - side->type.scale;
    return (struct compare_common){
        .bound = side->bound,
        .factor = type_i128_load(side->limit128),
        .bit_width = side->type.bit_width,
        .is_scaled = difference != 0,
        .should_check_bounds = difference != 0 && side->type.precision > common.precision - difference,
    };
}

static inline bool
compare_common_fits(const struct kernel_packet* packet, uint64_t element, const struct compare_common* side) {
    return decimal_bound_admits_i256(compare_decimal_load(side->bit_width, packet->values, element), side->bound);
}

static int compare_common_check(const struct kernel_packet* packet, const struct compare_common* side, bool live) {
    if (!side->should_check_bounds || !live) {
        return DORA_OK;
    }
    if (packet->shape == KERNEL_SHAPE_CONSTANT) {
        return compare_common_fits(packet, packet->value_offset, side) ? DORA_OK : ERANGE;
    }
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(packet->count);
    const uint32_t* const rows = packet->shape == KERNEL_SHAPE_GATHERED ? packet->rows : nullptr;
    const uint64_t offset = packet->value_offset;
    for (uint32_t word = 0; word < word_count; ++word) {
        const uint64_t tail = bitmap_word_tail_mask(packet->count, word);
        uint64_t pending = packet->validity == nullptr ? tail : packet->validity[word] & tail;
        while (pending != 0) {
            const uint32_t row = word * 64 + dora_bitmap_word_pop_lowest_set(&pending);
            if (!compare_common_fits(packet, kernel_row_index(offset, rows, UINT32_MAX, row), side)) {
                return ERANGE;
            }
        }
    }
    return DORA_OK;
}

static inline _BitInt(128)
    compare_common_load(const uint8_t* values, uint64_t element, const struct compare_common* side) {
    const struct dora_i256 wide = compare_decimal_load(side->bit_width, values, element);
    const unsigned _BitInt(128) bits = ((unsigned _BitInt(128))wide.limbs[1] << 64U) | wide.limbs[0];
    return side->is_scaled ? (_BitInt(128))(bits * (unsigned _BitInt(128))side->factor) : (_BitInt(128))bits;
}

static inline int compare_order_common(_BitInt(128) left, _BitInt(128) right) {
    return (left > right) - (left < right);
}

static inline int compare_order_u128(unsigned _BitInt(128) left, unsigned _BitInt(128) right) {
    return (left > right) - (left < right);
}

DEFINE_COMPARE_FAMILY(decimal_wide,
                      COMPARE_SPLIT_LANES,
                      COMPARE_DECIMAL_SIDES,
                      LOAD_WIDE,
                      AT_CONSTANT,
                      type_order_i256,
                      type_order_i256)
DEFINE_COMPARE_FAMILY(decimal_scaled,
                      COMPARE_MASKED_LANES,
                      COMPARE_DIGIT_SIDES,
                      LOAD_DIGITS,
                      AT_CACHED,
                      compare_order_digits,
                      compare_order_digits)
DEFINE_COMPARE_FAMILY(decimal_common,
                      COMPARE_SPLIT_LANES,
                      COMPARE_COMMON_SIDES,
                      LOAD_COMMON,
                      AT_CONSTANT,
                      compare_order_common,
                      compare_order_common)

//--------------------------------------
// Ranges
//--------------------------------------

static const uint64_t range_no_rows[KERNEL_CHUNK_WORD_COUNT] = {};

static inline bool range_operands_are_admitted(const struct kernel_packet* source,
                                               const struct kernel_packet* lower,
                                               const struct kernel_packet* upper) {
    return lower->shape == KERNEL_SHAPE_CONSTANT && upper->shape == KERNEL_SHAPE_CONSTANT &&
           source->count == lower->count && lower->count == upper->count;
}

static int range_null(const struct kernel_packet* left,
                      const struct kernel_packet* lower,
                      const struct kernel_packet* upper,
                      const struct kernel_range_binding* binding,
                      struct kernel_result* out_result) {
    if (!range_operands_are_admitted(left, lower, upper)) {
        return EINVAL;
    }
    const bool holds = type_order_matches(0, binding->lower.request.comparison) &&
                       type_order_matches(0, binding->upper.request.comparison);
    struct compare_leaf leaf = compare_leaf_init(left->count, &binding->lower, out_result, (struct compare_relation){});
    const bool bounds_live = kernel_packet_is_valid_constant(lower) && kernel_packet_is_valid_constant(upper);
    for (uint32_t word = 0; word < leaf.word_count; ++word) {
        const uint64_t tail = (word + 1 == leaf.word_count ? leaf.last_tail : UINT64_MAX);
        compare_store(&leaf,
                      word,
                      (struct compare_word){
                          .match = holds ? UINT64_MAX : 0,
                          .left_valid = left->validity == nullptr ? tail : left->validity[word] & tail,
                          .right_valid = bounds_live ? tail : 0,
                          .tail = tail,
                      });
    }
    out_result->null_count = leaf.null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

#define RANGE_VIEWS                                                                                                    \
    const uint8_t* const left_values = left->values;                                                                   \
    const uint8_t* const lower_values = lower->values;                                                                 \
    const uint8_t* const upper_values = upper->values;                                                                 \
    const uint64_t left_offset = left->value_offset;                                                                   \
    const uint64_t lower_offset = lower->value_offset;                                                                 \
    const uint64_t upper_offset = upper->value_offset;                                                                 \
    const uint32_t* const left_rows = left->shape == KERNEL_SHAPE_GATHERED ? left->rows : nullptr;                     \
    const uint32_t left_mask = left->shape == KERNEL_SHAPE_FLAT ? UINT32_MAX : 0;                                      \
    const uint64_t* const left_validity = left->validity;                                                              \
    const uint64_t* const right_validity = bounds_live ? nullptr : range_no_rows

#define RANGE_SIDES                                                                                                    \
    RANGE_VIEWS;                                                                                                       \
    constexpr int sides_status = DORA_OK

#define RANGE_LOADED_SIDES(T)                                                                                          \
    auto const compare_load = TYPE_LOADER(T);                                                                          \
    RANGE_SIDES

#define RANGE_DECIMAL_SIDES                                                                                            \
    RANGE_VIEWS;                                                                                                       \
    const struct dora_decimal_type* const left_decimal = &binding->lower.left_decimal.type;                            \
    const struct dora_decimal_type* const lower_decimal = &binding->lower.right_decimal.type;                          \
    const struct dora_decimal_type* const upper_decimal = &binding->upper.right_decimal.type;                          \
    constexpr int sides_status = DORA_OK

#define RANGE_COMMON_SIDES                                                                                             \
    RANGE_VIEWS;                                                                                                       \
    const struct compare_common left_common =                                                                          \
        compare_common_side(&binding->lower.left_decimal, binding->lower.result_decimal.type);                         \
    const struct compare_common lower_common =                                                                         \
        compare_common_side(&binding->lower.right_decimal, binding->lower.result_decimal.type);                        \
    const struct compare_common upper_common =                                                                         \
        compare_common_side(&binding->upper.right_decimal, binding->upper.result_decimal.type);                        \
    const bool common_live = bounds_live && compare_values_are_live(left);                                             \
    const int sides_status = compare_common_check(left, &left_common, common_live) != DORA_OK ||                       \
                                     compare_common_check(lower, &lower_common, common_live) != DORA_OK ||             \
                                     compare_common_check(upper, &upper_common, common_live) != DORA_OK                \
                                 ? ERANGE                                                                              \
                                 : DORA_OK

#define DEFINE_RANGE_LEAF(NAME, FLAGS, WORDS, SIDES, LOAD, VALUE_T, CORE, LOWER_REL, UPPER_REL)                        \
    static inline bool range_holds_##NAME##_##FLAGS(VALUE_T value, VALUE_T lower, VALUE_T upper) {                     \
        return CORE(value, lower) LOWER_REL 0 && CORE(value, upper) UPPER_REL 0;                                       \
    }                                                                                                                  \
    static int range_##NAME##_##FLAGS(const struct kernel_packet* left,                                                \
                                      const struct kernel_packet* lower,                                               \
                                      const struct kernel_packet* upper,                                               \
                                      const struct kernel_range_binding* binding,                                      \
                                      struct kernel_result* out_result) {                                              \
        if (!range_operands_are_admitted(left, lower, upper) ||                                                        \
            binding->lower.request.input_type != binding->upper.request.input_type) {                                  \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        const struct kernel_packet* const right = lower;                                                               \
        struct compare_leaf leaf =                                                                                     \
            compare_leaf_init(left->count, &binding->lower, out_result, (struct compare_relation){});                  \
        const bool bounds_live = kernel_packet_is_valid_constant(lower) && kernel_packet_is_valid_constant(upper);     \
        if (!compare_values_are_live(left) || !bounds_live) {                                                          \
            return range_null(left, lower, upper, binding, out_result);                                                \
        }                                                                                                              \
        SIDES;                                                                                                         \
        if (sides_status != DORA_OK) {                                                                                 \
            return sides_status;                                                                                       \
        }                                                                                                              \
        VALUE_T lower_value = {};                                                                                      \
        VALUE_T upper_value = {};                                                                                      \
        if (bounds_live) {                                                                                             \
            lower_value = LOAD(lower, lower_offset);                                                                   \
            upper_value = LOAD(upper, upper_offset);                                                                   \
        }                                                                                                              \
        COMPARE_EXPANSIONS(WORDS,                                                                                      \
                           COMPARE_MASKED_LANES,                                                                       \
                           COMPARE_CONSTANT,                                                                           \
                           LOAD,                                                                                       \
                           AT_CONSTANT,                                                                                \
                           RANGE_LANE,                                                                                 \
                           range_holds_##NAME##_##FLAGS,                                                               \
                           ,                                                                                           \
                           COMPARE_FINISH);                                                                            \
        out_result->null_count = leaf.null_count;                                                                      \
        out_result->failure_bits = 0;                                                                                  \
        return DORA_OK;                                                                                                \
    }

#define DEFINE_RANGE_FAMILY(NAME, WORDS, SIDES, LOAD, VALUE_T, CORE)                                                   \
    DEFINE_RANGE_LEAF(NAME, 0, WORDS, SIDES, LOAD, VALUE_T, CORE, >=, <=)                                              \
    DEFINE_RANGE_LEAF(NAME, 1, WORDS, SIDES, LOAD, VALUE_T, CORE, >, <=)                                               \
    DEFINE_RANGE_LEAF(NAME, 2, WORDS, SIDES, LOAD, VALUE_T, CORE, >=, <)                                               \
    DEFINE_RANGE_LEAF(NAME, 3, WORDS, SIDES, LOAD, VALUE_T, CORE, >, <)

// Integers except Int128 and UInt128, floating-point types, Decimal (32/64/256-bit), temporal types, intervals
#define DEFINE_RANGE_FIXED(S, T, WORDS)                                                                                \
    DEFINE_RANGE_FAMILY(S, WORDS, RANGE_LOADED_SIDES(T), LOAD_VALUE, T, type_order_##S)
COMPARE_FIXED_TAGS(DEFINE_RANGE_FIXED)
#undef DEFINE_RANGE_FIXED
#undef COMPARE_FIXED_TAGS

DEFINE_RANGE_FAMILY(i128,
                    COMPARE_SPLIT_LANES,
                    RANGE_LOADED_SIDES(struct dora_i128),
                    LOAD_I128,
                    _BitInt(128),
                    compare_order_common)
DEFINE_RANGE_FAMILY(u128,
                    COMPARE_SPLIT_LANES,
                    RANGE_LOADED_SIDES(struct dora_u128),
                    LOAD_U128,
                    unsigned _BitInt(128),
                    compare_order_u128)
DEFINE_RANGE_FAMILY(bool, COMPARE_SPLIT_LANES, RANGE_SIDES, LOAD_BIT, bool, type_order_bool)
DEFINE_RANGE_FAMILY(bytes,
                    COMPARE_SPLIT_LANES,
                    RANGE_LOADED_SIDES(struct dora_bytes),
                    LOAD_VALUE,
                    struct dora_bytes,
                    type_order_bytes)
DEFINE_RANGE_FAMILY(decimal_wide,
                    COMPARE_SPLIT_LANES,
                    RANGE_DECIMAL_SIDES,
                    LOAD_WIDE,
                    struct dora_i256,
                    type_order_i256)
DEFINE_RANGE_FAMILY(decimal_scaled,
                    COMPARE_MASKED_LANES,
                    RANGE_DECIMAL_SIDES,
                    LOAD_DIGITS,
                    struct type_decimal_digits,
                    compare_order_digits)
DEFINE_RANGE_FAMILY(decimal_common,
                    COMPARE_SPLIT_LANES,
                    RANGE_COMMON_SIDES,
                    LOAD_COMMON,
                    _BitInt(128),
                    compare_order_common)

#undef DEFINE_RANGE_FAMILY
#undef DEFINE_RANGE_LEAF
#undef RANGE_COMMON_SIDES
#undef RANGE_DECIMAL_SIDES
#undef RANGE_LOADED_SIDES
#undef RANGE_SIDES
#undef RANGE_VIEWS
#undef DEFINE_COMPARE_FAMILY
#undef DEFINE_ORDER_LEAF
#undef DEFINE_COMPARE_LEAF
#undef COMPARE_COMMON_SIDES
#undef COMPARE_DIGIT_SIDES
#undef COMPARE_DECIMAL_SIDES
#undef COMPARE_DECIMAL_VIEWS
#undef COMPARE_CACHED_SIDES
#undef CACHED_LOAD_VALUE
#undef CACHED_LOAD_I128
#undef AT_TYPED_CONSTANT
#undef COMPARE_LOADED_SIDES
#undef COMPARE_SIDES
#undef COMPARE_VIEWS
#undef COMPARE_CONSTANT
#undef COMPARE_FINISH
#undef ORDER_LANE_BEGIN
#undef RANGE_LANE_BEGIN
#undef NATIVE_LANE_BEGIN
#undef TRUTH_LANE_BEGIN
#undef COMPARE_EXPANSIONS
#undef COMPARE_WORD_LOOP
#undef COMPARE_SPLIT_LANES_BEGIN
#undef COMPARE_MASKED_LANES_BEGIN
#undef ORDER_LANES_BEGIN
#undef COMPARE_CONSTANT_BEGIN
#undef PROLOGUE_ORDER_LANE
#undef PROLOGUE_RANGE_LANE
#undef PROLOGUE_NATIVE_LANE
#undef PROLOGUE_TRUTH_LANE
#undef COMPARE_MASKED_LANES
#undef COMPARE_SPLIT_LANES
#undef SPLIT_ALL_VALID
#undef FUSED_MATCH
#undef ORDER_LANES
#undef COMPARE_RELATION
#undef ORDER_FINISH
#undef ORDER_LANE
#undef RANGE_LANE_HOLDS
#undef NATIVE_LANE_HOLDS
#undef TRUTH_LANE_HOLDS
#undef AT_GENERIC
#undef AT_CACHED
#undef AT_CONSTANT
#undef AT_FLAT
#undef LOAD_COMMON
#undef LOAD_DIGITS
#undef LOAD_WIDE
#undef LOAD_BIT
#undef LOAD_U128
#undef LOAD_I128
#undef LOAD_VALUE

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

enum compare_tag : uint8_t {
    COMPARE_TAG_I8,
    COMPARE_TAG_I16,
    COMPARE_TAG_I32,
    COMPARE_TAG_I64,
    COMPARE_TAG_I128,
    COMPARE_TAG_I256,
    COMPARE_TAG_U8,
    COMPARE_TAG_U16,
    COMPARE_TAG_U32,
    COMPARE_TAG_U64,
    COMPARE_TAG_U128,
    COMPARE_TAG_F16,
    COMPARE_TAG_F32,
    COMPARE_TAG_F64,
    COMPARE_TAG_INTERVAL_DAY_TIME,
    COMPARE_TAG_INTERVAL_MONTH_DAY_NANO,
    COMPARE_TAG_BOOL,
    COMPARE_TAG_BYTES,
    COMPARE_TAG_NULL,
    COMPARE_TAG_DECIMAL_WIDE,
    COMPARE_TAG_DECIMAL_SCALED,
    COMPARE_TAG_DECIMAL_COMMON,
    COMPARE_TAG_COUNT
};

static constexpr int32_t COMPARE_DECIMAL_MAX_WIDTH = 38;

#define COMPARE_ROW(PREFIX, NULL_LEAF)                                                                                 \
    {                                                                                                                  \
        [COMPARE_TAG_I8] = PREFIX##_i8,                                                                                \
        [COMPARE_TAG_I16] = PREFIX##_i16,                                                                              \
        [COMPARE_TAG_I32] = PREFIX##_i32,                                                                              \
        [COMPARE_TAG_I64] = PREFIX##_i64,                                                                              \
        [COMPARE_TAG_I128] = PREFIX##_i128,                                                                            \
        [COMPARE_TAG_I256] = PREFIX##_i256,                                                                            \
        [COMPARE_TAG_U8] = PREFIX##_u8,                                                                                \
        [COMPARE_TAG_U16] = PREFIX##_u16,                                                                              \
        [COMPARE_TAG_U32] = PREFIX##_u32,                                                                              \
        [COMPARE_TAG_U64] = PREFIX##_u64,                                                                              \
        [COMPARE_TAG_U128] = PREFIX##_u128,                                                                            \
        [COMPARE_TAG_F16] = PREFIX##_f16,                                                                              \
        [COMPARE_TAG_F32] = PREFIX##_f32,                                                                              \
        [COMPARE_TAG_F64] = PREFIX##_f64,                                                                              \
        [COMPARE_TAG_INTERVAL_DAY_TIME] = PREFIX##_interval_day_time,                                                  \
        [COMPARE_TAG_INTERVAL_MONTH_DAY_NANO] = PREFIX##_interval_month_day_nano,                                      \
        [COMPARE_TAG_BOOL] = PREFIX##_bool,                                                                            \
        [COMPARE_TAG_BYTES] = PREFIX##_bytes,                                                                          \
        [COMPARE_TAG_NULL] = NULL_LEAF,                                                                                \
        [COMPARE_TAG_DECIMAL_WIDE] = PREFIX##_decimal_wide,                                                            \
        [COMPARE_TAG_DECIMAL_SCALED] = PREFIX##_decimal_scaled,                                                        \
        [COMPARE_TAG_DECIMAL_COMMON] = PREFIX##_decimal_common,                                                        \
    }

static typeof(&compare_eq_i8) const compare_leaves[DORA_COMPARE_OP_GE + 1][COMPARE_TAG_COUNT] = {
    [DORA_COMPARE_OP_EQ] = COMPARE_ROW(compare_eq, compare_null),
    [DORA_COMPARE_OP_NE] = COMPARE_ROW(compare_ne, compare_null),
    [DORA_COMPARE_OP_LT] = COMPARE_ROW(compare_lt, compare_null),
    [DORA_COMPARE_OP_LE] = COMPARE_ROW(compare_le, compare_null),
    [DORA_COMPARE_OP_GT] = COMPARE_ROW(compare_gt, compare_null),
    [DORA_COMPARE_OP_GE] = COMPARE_ROW(compare_ge, compare_null),
};

static typeof(&order_i8) const order_leaves[COMPARE_TAG_COUNT] = COMPARE_ROW(order, order_null);

#define RANGE_ROW(S) {range_##S##_0, range_##S##_1, range_##S##_2, range_##S##_3}
static constexpr uint32_t RANGE_FLAG_COUNT = (DORA_RANGE_BOUND_LOWER_EXCLUSIVE | DORA_RANGE_BOUND_UPPER_EXCLUSIVE) + 1;
static typeof(&range_i8_0) const range_leaves[COMPARE_TAG_COUNT][RANGE_FLAG_COUNT] = {
    [COMPARE_TAG_I8] = RANGE_ROW(i8),
    [COMPARE_TAG_I16] = RANGE_ROW(i16),
    [COMPARE_TAG_I32] = RANGE_ROW(i32),
    [COMPARE_TAG_I64] = RANGE_ROW(i64),
    [COMPARE_TAG_I128] = RANGE_ROW(i128),
    [COMPARE_TAG_I256] = RANGE_ROW(i256),
    [COMPARE_TAG_U8] = RANGE_ROW(u8),
    [COMPARE_TAG_U16] = RANGE_ROW(u16),
    [COMPARE_TAG_U32] = RANGE_ROW(u32),
    [COMPARE_TAG_U64] = RANGE_ROW(u64),
    [COMPARE_TAG_U128] = RANGE_ROW(u128),
    [COMPARE_TAG_F16] = RANGE_ROW(f16),
    [COMPARE_TAG_F32] = RANGE_ROW(f32),
    [COMPARE_TAG_F64] = RANGE_ROW(f64),
    [COMPARE_TAG_INTERVAL_DAY_TIME] = RANGE_ROW(interval_day_time),
    [COMPARE_TAG_INTERVAL_MONTH_DAY_NANO] = RANGE_ROW(interval_month_day_nano),
    [COMPARE_TAG_BOOL] = RANGE_ROW(bool),
    [COMPARE_TAG_BYTES] = RANGE_ROW(bytes),
    [COMPARE_TAG_NULL] = {range_null, range_null, range_null, range_null},
    [COMPARE_TAG_DECIMAL_WIDE] = RANGE_ROW(decimal_wide),
    [COMPARE_TAG_DECIMAL_SCALED] = RANGE_ROW(decimal_scaled),
    [COMPARE_TAG_DECIMAL_COMMON] = RANGE_ROW(decimal_common),
};
#undef RANGE_ROW

#undef COMPARE_ROW

static constexpr enum compare_tag compare_carriers[DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO + 1] = {
    [DORA_PHYSICAL_TYPE_NONE] = COMPARE_TAG_COUNT,
    [DORA_PHYSICAL_TYPE_I8] = COMPARE_TAG_I8,
    [DORA_PHYSICAL_TYPE_I16] = COMPARE_TAG_I16,
    [DORA_PHYSICAL_TYPE_I32] = COMPARE_TAG_I32,
    [DORA_PHYSICAL_TYPE_I64] = COMPARE_TAG_I64,
    [DORA_PHYSICAL_TYPE_I128] = COMPARE_TAG_I128,
    [DORA_PHYSICAL_TYPE_I256] = COMPARE_TAG_I256,
    [DORA_PHYSICAL_TYPE_U8] = COMPARE_TAG_U8,
    [DORA_PHYSICAL_TYPE_U16] = COMPARE_TAG_U16,
    [DORA_PHYSICAL_TYPE_U32] = COMPARE_TAG_U32,
    [DORA_PHYSICAL_TYPE_U64] = COMPARE_TAG_U64,
    [DORA_PHYSICAL_TYPE_U128] = COMPARE_TAG_U128,
    [DORA_PHYSICAL_TYPE_F16] = COMPARE_TAG_F16,
    [DORA_PHYSICAL_TYPE_F32] = COMPARE_TAG_F32,
    [DORA_PHYSICAL_TYPE_F64] = COMPARE_TAG_F64,
    [DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME] = COMPARE_TAG_INTERVAL_DAY_TIME,
    [DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO] = COMPARE_TAG_INTERVAL_MONTH_DAY_NANO,
};

static int compare_carrier_of(enum dora_physical_type physical, enum compare_tag* out_tag) {
    if (physical > DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO || compare_carriers[physical] == COMPARE_TAG_COUNT) {
        return EINVAL;
    }
    *out_tag = compare_carriers[physical];
    return DORA_OK;
}

static inline bool compare_decimal_is_bindable(struct dora_decimal_type type) {
    return type.scale >= 0 && type.scale <= type.precision && type.precision <= COMPARE_DECIMAL_MAX_WIDTH;
}

static inline struct dora_decimal_type compare_decimal_common(struct dora_decimal_type left,
                                                              struct dora_decimal_type right) {
    const int32_t scale = left.scale > right.scale ? left.scale : right.scale;
    const int32_t left_integral = left.precision - left.scale;
    const int32_t right_integral = right.precision - right.scale;
    const int32_t max_integral = left_integral > right_integral ? left_integral : right_integral;
    const int32_t needed = scale + max_integral;
    int32_t width = left.precision > right.precision ? left.precision : right.precision;
    width = needed > width ? needed : width;
    return (struct dora_decimal_type){
        .precision = width < COMPARE_DECIMAL_MAX_WIDTH ? width : COMPARE_DECIMAL_MAX_WIDTH,
        .scale = scale,
    };
}

static int compare_common_prepare(struct decimal_prepared* side, struct dora_decimal_type common) {
    const int32_t difference = common.scale - side->type.scale;
    const int32_t target = common.precision - difference;
    unsigned _BitInt(128) factor = 1;
    for (int32_t power = 0; power < difference; ++power) {
        factor *= 10U;
    }
    side->limit128 = type_i128_store(factor);
    if (target == 0) {
        side->bound = (struct dora_i256){};
        return DORA_OK;
    }
    return dora_decimal_bound_prepare((struct dora_decimal_type){.precision = target},
                                      DECIMAL128_MAX_PRECISION,
                                      &side->bound);
}

static int compare_bind_decimal(struct kernel_binary_binding* in_out_binding, enum compare_tag* out_tag) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    int status = dora_decimal_prepare(request->input_type->parameters.decimal,
                                      in_out_binding->input_physical,
                                      &in_out_binding->left_decimal);
    if (status == DORA_OK) {
        status = dora_decimal_prepare(request->right_type->parameters.decimal,
                                      in_out_binding->right_physical,
                                      &in_out_binding->right_decimal);
    }
    if (status != DORA_OK) {
        return status;
    }
    in_out_binding->left_byte_width = in_out_binding->left_decimal.type.bit_width / 8;
    in_out_binding->right_byte_width = in_out_binding->right_decimal.type.bit_width / 8;
    enum compare_tag carrier_tag = COMPARE_TAG_COUNT;
    status = compare_carrier_of(in_out_binding->input_physical, &carrier_tag);
    const struct dora_decimal_type left = in_out_binding->left_decimal.type;
    const struct dora_decimal_type right = in_out_binding->right_decimal.type;
    struct dora_decimal_type common = {};
    if (left.scale == right.scale) {
        *out_tag = in_out_binding->left_byte_width != in_out_binding->right_byte_width ? COMPARE_TAG_DECIMAL_WIDE
                                                                                       : carrier_tag;
    } else if (compare_decimal_is_bindable(left) && compare_decimal_is_bindable(right)) {
        common = compare_decimal_common(left, right);
        *out_tag = COMPARE_TAG_DECIMAL_COMMON;
        in_out_binding->result_decimal.type = common;
        status = compare_common_prepare(&in_out_binding->left_decimal, common);
        if (status == DORA_OK) {
            status = compare_common_prepare(&in_out_binding->right_decimal, common);
        }
    } else {
        *out_tag = COMPARE_TAG_DECIMAL_SCALED;
    }
    return status;
}

static int compare_bind_tag(struct kernel_binary_binding* in_out_binding, enum compare_tag* out_tag) {
    const struct kernel_bind_request* const request = &in_out_binding->request;
    if (request->input_type == nullptr || request->right_type == nullptr ||
        request->input_type->id != request->right_type->id) {
        return EINVAL;
    }
    switch (request->input_type->id) {
    case DORA_TYPE_NULL:
        *out_tag = COMPARE_TAG_NULL;
        return DORA_OK;
    case DORA_TYPE_BOOLEAN:
        *out_tag = COMPARE_TAG_BOOL;
        return DORA_OK;
    case DORA_TYPE_UTF8:
    case DORA_TYPE_BINARY:
    case DORA_TYPE_FIXED_BINARY:
        *out_tag = COMPARE_TAG_BYTES;
        in_out_binding->left_byte_width = sizeof(struct dora_bytes);
        in_out_binding->right_byte_width = sizeof(struct dora_bytes);
        return DORA_OK;
    case DORA_TYPE_DECIMAL:
        return compare_bind_decimal(in_out_binding, out_tag);
    default:
        break;
    }
    enum compare_tag tag = COMPARE_TAG_COUNT;
    const int status = compare_carrier_of(in_out_binding->input_physical, &tag);
    if (status != DORA_OK || in_out_binding->right_physical != in_out_binding->input_physical) {
        return EINVAL;
    }
    *out_tag = tag;
    in_out_binding->left_byte_width = physical_width(in_out_binding->input_physical);
    in_out_binding->right_byte_width = in_out_binding->left_byte_width;
    return DORA_OK;
}

static int compare_bind(const struct kernel_bind_request* request,
                        enum dora_null_order null_order,
                        struct kernel_binary_binding* out_binding,
                        enum compare_tag* out_tag) {
    struct kernel_binary_binding binding = {.request = *request, .result_type = {.id = DORA_TYPE_BOOLEAN}};
    binding.request.null_order = null_order;
    int status = type_scalar_physical(request->input_type, &binding.input_physical);
    if (status == DORA_OK) {
        status = type_scalar_physical(request->right_type, &binding.right_physical);
    }
    if (status == DORA_OK) {
        status = compare_bind_tag(&binding, out_tag);
    }
    if (status == DORA_OK) {
        *out_binding = binding;
    }
    return status;
}

int dora_kernel_compare_bind(const struct kernel_bind_request* request, struct kernel_binary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->comparison > DORA_COMPARE_OP_GE) {
        return EINVAL;
    }
    struct kernel_binary_binding binding = {};
    enum compare_tag tag = COMPARE_TAG_NULL;
    const int status = compare_bind(request, DORA_NULL_ORDER_PROPAGATE, &binding, &tag);
    if (status == DORA_OK) {
        binding.fill = compare_leaves[request->comparison][tag];
        *out_binding = binding;
    }
    return status;
}

int dora_kernel_compare_distinct_bind(const struct kernel_bind_request* request,
                                      struct kernel_binary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->comparison > DORA_COMPARE_OP_GE ||
        request->null_order >= DORA_NULL_ORDER_PROPAGATE) {
        return EINVAL;
    }
    struct kernel_binary_binding binding = {};
    enum compare_tag tag = COMPARE_TAG_NULL;
    const int status = compare_bind(request, request->null_order, &binding, &tag);
    if (status == DORA_OK) {
        binding.fill = compare_leaves[request->comparison][tag];
        binding.is_result_all_valid = true;
        *out_binding = binding;
    }
    return status;
}

int dora_kernel_order_bind(const struct kernel_bind_request* request, struct kernel_binary_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->null_order > DORA_NULL_ORDER_PROPAGATE) {
        return EINVAL;
    }
    struct kernel_binary_binding binding = {};
    enum compare_tag tag = COMPARE_TAG_NULL;
    const int status = compare_bind(request, request->null_order, &binding, &tag);
    if (status == DORA_OK) {
        binding.fill = order_leaves[tag];
        binding.result_type = (struct dora_logical_type){.id = DORA_TYPE_INT8};
        binding.result_physical = DORA_PHYSICAL_TYPE_I8;
        binding.result_byte_width = sizeof(int8_t);
        binding.is_result_all_valid = request->null_order != DORA_NULL_ORDER_PROPAGATE;
        *out_binding = binding;
    }
    return status;
}

static int range_tag(struct kernel_range_binding* in_out_binding,
                     enum compare_tag lower_tag,
                     enum compare_tag upper_tag,
                     enum compare_tag* out_tag) {
    if (in_out_binding->lower.request.input_type->id != DORA_TYPE_DECIMAL) {
        *out_tag = lower_tag;
        return lower_tag == upper_tag ? DORA_OK : EINVAL;
    }
    const struct dora_decimal_type source = in_out_binding->lower.left_decimal.type;
    const struct dora_decimal_type lower = in_out_binding->lower.right_decimal.type;
    const struct dora_decimal_type upper = in_out_binding->upper.right_decimal.type;
    struct dora_decimal_type common = {};
    if (source.scale == lower.scale && source.scale == upper.scale) {
        *out_tag = lower_tag == upper_tag ? lower_tag : COMPARE_TAG_DECIMAL_WIDE;
    } else if (compare_decimal_is_bindable(source) && compare_decimal_is_bindable(lower) &&
               compare_decimal_is_bindable(upper)) {
        common = compare_decimal_common(compare_decimal_common(source, lower), upper);
        in_out_binding->lower.result_decimal.type = common;
        in_out_binding->upper.result_decimal.type = common;
        int status = compare_common_prepare(&in_out_binding->lower.left_decimal, common);
        if (status == DORA_OK) {
            status = compare_common_prepare(&in_out_binding->lower.right_decimal, common);
        }
        if (status == DORA_OK) {
            status = compare_common_prepare(&in_out_binding->upper.right_decimal, common);
        }
        if (status != DORA_OK) {
            return status;
        }
        *out_tag = COMPARE_TAG_DECIMAL_COMMON;
    } else {
        *out_tag = COMPARE_TAG_DECIMAL_SCALED;
    }
    return DORA_OK;
}

int dora_kernel_range_bind(const struct kernel_bind_request* request, struct kernel_range_binding* out_binding) {
    if (request == nullptr || out_binding == nullptr || request->upper_type == nullptr ||
        request->range_bounds > (DORA_RANGE_BOUND_LOWER_EXCLUSIVE | DORA_RANGE_BOUND_UPPER_EXCLUSIVE)) {
        return EINVAL;
    }
    struct kernel_bind_request lower = *request;
    lower.comparison =
        (request->range_bounds & DORA_RANGE_BOUND_LOWER_EXCLUSIVE) != 0 ? DORA_COMPARE_OP_GT : DORA_COMPARE_OP_GE;
    struct kernel_bind_request upper = *request;
    upper.right_type = request->upper_type;
    upper.comparison =
        (request->range_bounds & DORA_RANGE_BOUND_UPPER_EXCLUSIVE) != 0 ? DORA_COMPARE_OP_LT : DORA_COMPARE_OP_LE;
    struct kernel_range_binding binding = {};
    enum compare_tag lower_tag = COMPARE_TAG_NULL;
    enum compare_tag upper_tag = COMPARE_TAG_NULL;
    enum compare_tag tag = COMPARE_TAG_NULL;
    int status = compare_bind(&lower, DORA_NULL_ORDER_PROPAGATE, &binding.lower, &lower_tag);
    if (status == DORA_OK) {
        binding.upper.request = upper;
        binding.upper.request.null_order = DORA_NULL_ORDER_PROPAGATE;
        binding.upper.result_type = (struct dora_logical_type){.id = DORA_TYPE_BOOLEAN};
        binding.upper.input_physical = binding.lower.input_physical;
        status = type_scalar_physical(upper.right_type, &binding.upper.right_physical);
        if (status == DORA_OK) {
            status = compare_bind_tag(&binding.upper, &upper_tag);
        }
    }
    if (status == DORA_OK) {
        status = range_tag(&binding, lower_tag, upper_tag, &tag);
    }
    if (status == DORA_OK) {
        binding.fill = range_leaves[tag][request->range_bounds];
        *out_binding = binding;
    }
    return status;
}
