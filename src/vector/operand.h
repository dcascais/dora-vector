// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Bound vector operands.

#ifndef DORA_VECTOR_OPERAND_H
#define DORA_VECTOR_OPERAND_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/binding/read.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/vector_internal.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Operands
//
//------------------------------------------------------------------------------

static inline struct dora_range vector_operand_chunk(uint64_t first, uint64_t stored) {
    const uint64_t remaining = stored - first;
    return (struct dora_range){
        .offset = first,
        .length = remaining < KERNEL_CHUNK_ROW_COUNT ? remaining : KERNEL_CHUNK_ROW_COUNT,
    };
}

static inline bool vector_view_has_encoding_child(const struct dora_vector_view* view) {
    return view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END;
}

static inline uint32_t vector_input_encoding_child(const struct vector_bound_input* input, uint32_t node) {
    return dora_vector_input_child(node, input, input->nodes[node].view.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1);
}

static inline uint32_t vector_input_value_node(const struct vector_bound_input* input, uint32_t node) {
    while (vector_view_has_encoding_child(&input->nodes[node].view)) {
        node = vector_input_encoding_child(input, node);
    }
    return node;
}

enum vector_operand_requirement : uint8_t {
    VECTOR_OPERAND_REQUIRE_PACKET = 1,
    VECTOR_OPERAND_REQUIRE_TEXT = 2,
    VECTOR_OPERAND_REQUIRE_PACKET_TEXT = 3,
    VECTOR_OPERAND_REQUIRE_VALIDITY = 4,
    VECTOR_OPERAND_REQUIRE_ACTIVE = 8,
    VECTOR_OPERAND_REQUIRE_LENGTH = 16,
    VECTOR_OPERAND_REQUIRE_ROW = 32
};

enum vector_operand_slot_component : uint8_t {
    VECTOR_OPERAND_SLOT_OPERAND,
    VECTOR_OPERAND_SLOT_TEXT,
    VECTOR_OPERAND_SLOT_CACHE,
    VECTOR_OPERAND_SLOT_ROWS,
    VECTOR_OPERAND_SLOT_NODES,
    VECTOR_OPERAND_SLOT_COUNT
};

// Each component may be acquired once per scope.
static inline size_t vector_operand_slot(size_t operand, enum vector_operand_slot_component component) {
    return operand * VECTOR_OPERAND_SLOT_COUNT + component;
}

enum vector_operand_kind : uint8_t {
    VECTOR_OPERAND_DIRECT,
    VECTOR_OPERAND_CONSTANT,
    VECTOR_OPERAND_CONSTANT_NULL,
    VECTOR_OPERAND_ENCODED,
    VECTOR_OPERAND_BYTES,
    VECTOR_OPERAND_NESTED
};

struct vector_operand {
    const struct vector_bound_input* input;
    struct kernel_decode_binding* chain;
    struct kernel_gather_binding gather;
    struct kernel_bytes_binding bytes;
    uint64_t* rows64;
    uint32_t* map32;
    uint64_t* chunk_validity;
    void* payload;
    const void* values;
    uint64_t value_offset;
    struct dora_bytes constant_bytes;
    struct dora_bytes* byte_buffers;
    uint8_t constant_value[sizeof(struct dora_i256)];
    uint32_t chain_length;
    uint32_t terminal_node;
    enum vector_operand_kind kind;
    bool is_all_valid;
    bool has_referenced_indices;
};

struct vector_operand_rows {
    uint64_t* rows;
    uint64_t* slots;
    uint64_t* validity;
    uint32_t count;
};

int dora_vector_input_from_nodes(struct scratch_scope* scope,
                                 const struct dora_node_topology* nodes,
                                 uint32_t node_count,
                                 uint32_t root,
                                 struct vector_bound_input* out_input);
int dora_vector_operand_bind(struct scratch_scope* scope,
                             size_t position,
                             const struct vector_bound_input* input,
                             uint8_t requirements,
                             struct vector_operand* out_operand);
int dora_vector_operand_bind_constant(uint32_t value_node, struct vector_operand* in_out_operand);
int dora_vector_operand_bind_singleton(const struct dora_vector* source,
                                       const struct vector_bound_input* input,
                                       struct vector_operand* out_operand);
int dora_vector_operand_prepare(struct vector_operand* operand,
                                struct dora_range rows,
                                const uint64_t* active,
                                struct kernel_packet* out_packet);

int dora_vector_operand_prepare_validity(struct vector_operand* operand,
                                         struct dora_range rows,
                                         struct kernel_packet* out_packet);
int dora_vector_operand_measure(struct vector_operand* operand,
                                struct dora_range rows,
                                bool unicode,
                                size_t* out_lengths,
                                struct dora_utf8_mask* out_mask);
void dora_vector_operand_utf8_batch(const struct kernel_packet* packet,
                                    enum dora_utf8_proof proof,
                                    struct dora_utf8_batch* out_batch);
int dora_vector_operand_prepare_utf8(struct vector_operand* operand,
                                     struct dora_range rows,
                                     const uint64_t* active,
                                     struct dora_utf8_batch* out_batch);
int dora_vector_operand_prepare_rows(struct vector_operand* operand,
                                     uint32_t node,
                                     struct kernel_packet* out_packet,
                                     uint32_t count);
int dora_vector_operand_resolve_slots(struct vector_operand* operand,
                                      uint32_t node,
                                      const struct vector_operand_rows* rows,
                                      uint32_t* out_terminal);
bool dora_vector_operands_are_all_valid(const struct vector_operand* operands, uint32_t operand_count);
void dora_vector_operand_reset(struct vector_operand* operand);

#endif
