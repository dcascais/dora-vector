// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector comparison and hash plans.

#ifndef DORA_VECTOR_KEY_H
#define DORA_VECTOR_KEY_H

#include "dora/vector/types.h"
#include "kernel/binding/elementwise.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "vector/image.h"
#include "vector/operand.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Plans
//
//------------------------------------------------------------------------------

enum vector_key_kind : uint8_t {
    VECTOR_KEY_LEAF,
    VECTOR_KEY_STRUCT,
    VECTOR_KEY_SEQUENCE,
    VECTOR_KEY_UNION
};

struct vector_key_node {
    // Comparison plans bind order; hash plans bind hashing.
    union {
        struct kernel_binary_binding order;

        struct {
            struct kernel_unary_binding hash;
            struct kernel_binary_binding combine;
        } hashing;
    } binding;

    uint32_t input_nodes[2];
    uint32_t base_nodes[2];
    uint32_t child_begin;
    uint32_t child_count;
    enum vector_key_kind kind;
    bool is_nullable[2];
};

struct vector_comparison_plan {
    const struct vector_bound_input* left;
    const struct vector_bound_input* right;
    struct vector_key_node* nodes;
    uint32_t* children;
    uint32_t node_count;
    uint32_t max_depth;
    enum dora_compare_op comparison;
    enum dora_null_order null_order;
    bool should_skip_right_on_left_null;
    bool should_broadcast_right;
};

struct vector_hash_plan {
    const struct vector_bound_input* input;
    struct vector_key_node* nodes;
    uint32_t* children;
    uint64_t seed;
    uint32_t node_count;
    uint32_t max_depth;
};

//------------------------------------------------------------------------------
//
// Frames
//
//------------------------------------------------------------------------------

struct vector_comparison_lane {
    uint64_t left_row;
    uint64_t right_row;
    struct dora_range left_elements;
    struct dora_range right_elements;
    uint32_t result_row;
};

struct vector_comparison_frame {
    struct vector_comparison_lane* lanes;
    int8_t* orders;
    uint32_t* members;
    uint64_t element;
    uint32_t node;
    uint32_t input_count;
    uint32_t active_count;
    uint32_t child_ordinal;
    bool is_entered;
};

struct vector_hash_lane {
    uint64_t row;
    struct dora_range elements;
    uint32_t result_row;
};

struct vector_hash_frame {
    struct vector_hash_lane* lanes;
    uint64_t* running;
    uint64_t* hashes;
    uint32_t* members;
    uint64_t element;
    uint32_t node;
    uint32_t input_count;
    uint32_t active_count;
    uint32_t child_ordinal;
    bool is_entered;
};

//------------------------------------------------------------------------------
//
// Workspace
//
//------------------------------------------------------------------------------

struct vector_key_side {
    struct vector_operand prepare;
    uint64_t* slots;
    struct dora_range* ranges;
    uint32_t* members;
};

struct vector_key_workspace {
    struct vector_key_side sides[2];
    struct vector_comparison_frame* comparison_frames;
    struct vector_hash_frame* hash_frames;
    uint64_t root_validity[KERNEL_CHUNK_WORD_COUNT];
    uint32_t row_capacity;
};

//------------------------------------------------------------------------------
//
// Comparison
//
//------------------------------------------------------------------------------

int dora_vector_key_plan_init(struct scratch_scope* scope,
                              const struct vector_bound_input* left,
                              const struct vector_bound_input* right,
                              enum dora_compare_op comparison,
                              enum dora_null_order null_order,
                              struct vector_comparison_plan* out_plan);
int dora_vector_key_workspace_init(struct scratch_scope* scope,
                                   const struct vector_comparison_plan* plan,
                                   uint32_t row_capacity,
                                   struct vector_key_workspace* out_workspace);
int dora_vector_key_validity_init(struct scratch_scope* scope,
                                  const struct vector_bound_input* input,
                                  uint32_t row_capacity,
                                  bool packets,
                                  struct vector_key_workspace* out_workspace);
int dora_vector_key_compare_rows(const struct vector_comparison_plan* plan,
                                 struct dora_range rows,
                                 struct vector_key_workspace* workspace,
                                 struct kernel_result* out_result);
int dora_vector_key_order_rows(const struct vector_comparison_plan* plan,
                               struct dora_range rows,
                               struct vector_key_workspace* workspace,
                               struct kernel_result* out_result);
int dora_vector_key_order(const struct vector_comparison_plan* plan,
                          uint64_t left_row,
                          uint64_t right_row,
                          struct vector_key_workspace* workspace,
                          int* out_order);
int dora_vector_key_validity(struct vector_key_workspace* workspace, struct dora_range rows, uint64_t* out_validity);
int dora_vector_key_constant_null(struct scratch_scope* scope,
                                  const struct vector_bound_input* inputs,
                                  struct vector_operand* operands,
                                  uint32_t count,
                                  bool* out_null);

//------------------------------------------------------------------------------
//
// Hash
//
//------------------------------------------------------------------------------

int dora_vector_key_hash_plan_init(struct scratch_scope* scope,
                                   const struct vector_bound_input* input,
                                   uint64_t seed,
                                   struct vector_hash_plan* out_plan);
int dora_vector_key_hash_workspace_init(struct scratch_scope* scope,
                                        const struct vector_hash_plan* plan,
                                        struct vector_key_workspace* out_workspace);
int dora_vector_key_hash_rows(const struct vector_hash_plan* plan,
                              struct dora_range rows,
                              struct vector_key_workspace* workspace,
                              uint64_t* out_hashes);

#endif
