// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/packet.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Operands
//
//------------------------------------------------------------------------------

enum boolean_mix : uint8_t {
    BOOLEAN_MIX_ALL_NULL,
    BOOLEAN_MIX_ALL_FALSE,
    BOOLEAN_MIX_ALL_TRUE,
    BOOLEAN_MIX_MIXED
};

struct boolean_operand {
    const uint8_t* values;
    const uint64_t* validity;
    const uint64_t* words;
    uint64_t value_offset;
    uint64_t constant_word;
    uint32_t count;
    enum kernel_shape shape;
    enum boolean_mix mix;
};

static inline enum boolean_mix boolean_packet_mix(const struct kernel_packet* packet) {
    if (packet->shape != KERNEL_SHAPE_CONSTANT) {
        return BOOLEAN_MIX_MIXED;
    }
    if (packet->validity != nullptr && bitmap_word_bit(packet->validity[0], 0) == 0) {
        return BOOLEAN_MIX_ALL_NULL;
    }
    return dora_bitmap_test(packet->values, packet->value_offset) ? BOOLEAN_MIX_ALL_TRUE : BOOLEAN_MIX_ALL_FALSE;
}

static inline struct boolean_operand boolean_operand_init(const struct kernel_packet* packet,
                                                          uint64_t (*out_words)[KERNEL_CHUNK_WORD_COUNT]) {
    struct boolean_operand operand = {
        .values = packet->values,
        .validity = packet->validity,
        .value_offset = packet->value_offset,
        .count = packet->count,
        .shape = packet->shape,
        .mix = boolean_packet_mix(packet),
    };
    if (operand.mix == BOOLEAN_MIX_ALL_TRUE) {
        operand.constant_word = UINT64_MAX;
    }
    if (packet->shape == KERNEL_SHAPE_GATHERED) {
        const uint64_t* active = packet->validity;
        if (active == nullptr) {
            bitmap_fill_bytes((uint8_t*)*out_words, sizeof *out_words, true);
            active = *out_words;
        }
        dora_bitmap_gather_words_u32(*out_words,
                                     packet->values,
                                     packet->value_offset,
                                     packet->rows,
                                     active,
                                     packet->count);
        operand.words = *out_words;
    }
    return operand;
}

static inline uint64_t boolean_value_word(const struct boolean_operand* operand, uint32_t word) {
    switch (operand->shape) {
    case KERNEL_SHAPE_FLAT: {
        const uint32_t lanes = operand->count - word * 64 < 64 ? operand->count - word * 64 : 64;
        return dora_bitmap_load_window(operand->values, operand->value_offset + (uint64_t)word * 64, lanes);
    }
    case KERNEL_SHAPE_CONSTANT:
        return operand->constant_word;
    case KERNEL_SHAPE_GATHERED:
        return operand->words[word];
    }
    return 0;
}

static inline uint64_t boolean_present_word(const struct boolean_operand* operand, uint32_t word) {
    return operand->validity != nullptr ? operand->validity[word] : UINT64_MAX;
}

//------------------------------------------------------------------------------
//
// Words
//
//------------------------------------------------------------------------------

struct boolean_word {
    uint64_t value;
    uint64_t present;
    uint64_t active;
};

struct boolean_test {
    uint64_t value;
    uint64_t present;
};

static inline struct boolean_test boolean_test_word(const struct boolean_operand* operand, uint32_t word) {
    return (struct boolean_test){
        .value = boolean_value_word(operand, word),
        .present = boolean_present_word(operand, word),
    };
}

static inline void boolean_update_and(struct boolean_word* in_out_word, struct boolean_test test) {
    const uint64_t test_false = ~test.value & test.present;
    in_out_word->value &= ~in_out_word->active | ~test_false;
    in_out_word->present |= in_out_word->active & test_false;
    const uint64_t result_true = in_out_word->value & in_out_word->present;
    in_out_word->present &= ~in_out_word->active | ~(in_out_word->present & result_true & ~test.present);
    in_out_word->active &= ~test_false;
}

static inline void boolean_update_or(struct boolean_word* in_out_word, struct boolean_test test) {
    const uint64_t test_true = test.value & test.present;
    in_out_word->value |= in_out_word->active & test_true;
    in_out_word->present |= in_out_word->active & test_true;
    const uint64_t result_false = ~in_out_word->value & in_out_word->present;
    in_out_word->present &= ~in_out_word->active | ~(in_out_word->present & result_false & ~test.present);
    in_out_word->active &= ~test_true;
}

//------------------------------------------------------------------------------
//
// Leaves
//
//------------------------------------------------------------------------------

static inline int boolean_conjunct(const struct kernel_packet* left,
                                   const struct kernel_packet* right,
                                   struct kernel_result* out_result,
                                   bool is_and) {
    const uint32_t count = left->count;
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    uint64_t* const values = out_result->values;
    uint64_t* const validity = out_result->validity;
    const enum boolean_mix decides = is_and ? BOOLEAN_MIX_ALL_FALSE : BOOLEAN_MIX_ALL_TRUE;
    uint32_t null_count = 0;
    if (boolean_packet_mix(left) == decides || boolean_packet_mix(right) == decides) {
        for (uint32_t word = 0; word < word_count; ++word) {
            const uint64_t tail = bitmap_word_tail_mask(count, word);
            values[word] = is_and ? 0 : tail;
            if (validity != nullptr) {
                validity[word] = tail;
            }
        }
    } else {
        uint64_t gathered[2][KERNEL_CHUNK_WORD_COUNT];
        const struct boolean_operand operands[2] = {
            boolean_operand_init(left, &gathered[0]),
            boolean_operand_init(right, &gathered[1]),
        };
        const struct boolean_operand* mixed[2] = {};
        uint32_t mixed_count = 0;
        uint64_t all_null = 0;
        for (uint32_t side = 0; side < 2; ++side) {
            if (operands[side].mix == BOOLEAN_MIX_MIXED) {
                mixed[mixed_count++] = &operands[side];
            } else if (operands[side].mix == BOOLEAN_MIX_ALL_NULL) {
                all_null = UINT64_MAX;
            }
        }
        for (uint32_t word = 0; word < word_count; ++word) {
            const uint64_t tail = bitmap_word_tail_mask(count, word);
            struct boolean_word result = {.value = is_and ? tail : 0, .present = tail, .active = tail};
            for (uint32_t index = 0; index < mixed_count; ++index) {
                if (is_and) {
                    boolean_update_and(&result, boolean_test_word(mixed[index], word));
                } else {
                    boolean_update_or(&result, boolean_test_word(mixed[index], word));
                }
            }
            result.present &= ~(result.active & all_null);
            values[word] = result.value;
            if (validity != nullptr) {
                validity[word] = result.present;
            }
            null_count += dora_bitmap_word_count_ones(~result.present & tail);
        }
    }
    out_result->null_count = null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

static int boolean_and(const struct kernel_packet* left,
                       const struct kernel_packet* right,
                       const struct kernel_binary_binding*,
                       struct kernel_result* out_result) {
    return boolean_conjunct(left, right, out_result, true);
}

static int boolean_or(const struct kernel_packet* left,
                      const struct kernel_packet* right,
                      const struct kernel_binary_binding*,
                      struct kernel_result* out_result) {
    return boolean_conjunct(left, right, out_result, false);
}

static int boolean_xor(const struct kernel_packet* left,
                       const struct kernel_packet* right,
                       const struct kernel_binary_binding*,
                       struct kernel_result* out_result) {
    const uint32_t count = left->count;
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    uint64_t gathered[2][KERNEL_CHUNK_WORD_COUNT];
    const struct boolean_operand operands[2] = {
        boolean_operand_init(left, &gathered[0]),
        boolean_operand_init(right, &gathered[1]),
    };
    uint64_t* const values = out_result->values;
    uint64_t* const validity = out_result->validity;
    uint32_t null_count = 0;
    for (uint32_t word = 0; word < word_count; ++word) {
        const uint64_t tail = bitmap_word_tail_mask(count, word);
        const uint64_t valid =
            boolean_present_word(&operands[0], word) & boolean_present_word(&operands[1], word) & tail;
        values[word] = (boolean_value_word(&operands[0], word) ^ boolean_value_word(&operands[1], word)) & valid;
        if (validity != nullptr) {
            validity[word] = valid;
        }
        null_count += dora_bitmap_word_count_ones(~valid & tail);
    }
    out_result->null_count = null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

static int boolean_not(const struct kernel_packet* source,
                       const struct kernel_packet*,
                       const struct kernel_binary_binding*,
                       struct kernel_result* out_result) {
    const uint32_t count = source->count;
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    uint64_t gathered[KERNEL_CHUNK_WORD_COUNT];
    const struct boolean_operand operand = boolean_operand_init(source, &gathered);
    uint64_t* const values = out_result->values;
    uint64_t* const validity = out_result->validity;
    uint32_t null_count = 0;
    for (uint32_t word = 0; word < word_count; ++word) {
        const uint64_t tail = bitmap_word_tail_mask(count, word);
        values[word] = ~boolean_value_word(&operand, word) & tail;
        const uint64_t valid = boolean_present_word(&operand, word) & tail;
        if (validity != nullptr) {
            validity[word] = valid;
        }
        null_count += dora_bitmap_word_count_ones(~valid & tail);
    }
    out_result->null_count = null_count;
    out_result->failure_bits = 0;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

static typeof(&boolean_and) const boolean_leaves[KERNEL_BOOLEAN_NOT + 1] = {
    [KERNEL_BOOLEAN_AND] = boolean_and,
    [KERNEL_BOOLEAN_OR] = boolean_or,
    [KERNEL_BOOLEAN_XOR] = boolean_xor,
    [KERNEL_BOOLEAN_NOT] = boolean_not,
};

int dora_kernel_boolean_bind(enum kernel_boolean_operation operation, struct kernel_binary_binding* out_binding) {
    if (out_binding == nullptr || operation > KERNEL_BOOLEAN_NOT) {
        return EINVAL;
    }
    *out_binding = (struct kernel_binary_binding){
        .fill = boolean_leaves[operation],
        .result_type = {.id = DORA_TYPE_BOOLEAN},
    };
    return DORA_OK;
}
