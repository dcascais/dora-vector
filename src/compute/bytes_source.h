// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Borrowed byte inputs and text conversion.

#ifndef DORA_COMPUTE_BYTES_SOURCE_H
#define DORA_COMPUTE_BYTES_SOURCE_H

#include "compute/output.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/binding/elementwise.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/load.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/operand.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Sources
//
//------------------------------------------------------------------------------

struct compute_bytes_source {
    struct vector_bound_node leaf;
    struct vector_bound_input input;
    struct dora_vector converted;
    struct kernel_packet packet;
    struct vector_operand operand;
    struct kernel_unary_binding text;
    void* text_storage;
    const struct dora_bytes* pieces;
    const uint64_t* validity;
    bool is_constant;
    enum dora_utf8_proof utf8_proof;
};

struct compute_bytes_slice {
    int64_t offset;
    uint64_t count;
};

extern const struct dora_logical_type dora_compute_utf8_type;

static inline bool compute_bytes_piece_is_valid(const struct compute_bytes_source* source, uint32_t lane) {
    return source->validity == nullptr ||
           dora_bitmap_test((const uint8_t*)source->validity, source->is_constant ? 0 : lane);
}

static inline struct dora_bytes compute_bytes_piece(const struct compute_bytes_source* source, uint32_t lane) {
    return source->pieces == nullptr ? (struct dora_bytes){}
                                     : type_load_bytes((const uint8_t*)source->pieces, source->is_constant ? 0 : lane);
}

static inline bool compute_bytes_operand_is_all_valid(const struct vector_operand* operand) {
    return operand->kind == VECTOR_OPERAND_CONSTANT || (operand->kind == VECTOR_OPERAND_BYTES && operand->is_all_valid);
}

static inline bool compute_bytes_mask_is_valid(struct dora_utf8_mask mask, uint64_t lane) {
    return mask.bytes.data == nullptr || dora_bitmap_test(mask.bytes.data, mask.bit_offset + lane);
}

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

void dora_compute_bytes_source_init(struct compute_bytes_source* source);

// Utf8, Binary, FixedBinary; concatenation accepts scalars castable to Utf8 except Binary and FixedBinary
int dora_compute_bytes_bind(struct scratch_scope* scope,
                            size_t position,
                            const struct dora_vector* sources,
                            uint32_t source_count,
                            bool should_concatenate,
                            uint8_t requirements,
                            bool should_bind_singleton,
                            struct compute_bytes_source* out_inputs);

int dora_compute_bytes_prepare(struct compute_bytes_source* source, struct dora_range rows);

void dora_compute_bytes_release(struct compute_bytes_source* inputs, uint32_t source_count);

enum vector_content_state dora_compute_bytes_content(const struct compute_bytes_source* sources, uint32_t count);

int dora_compute_bytes_capacity(struct compute_bytes_source* source, struct compute_bytes_result* result);

int dora_compute_bytes_binary_batch(struct compute_bytes_source* source,
                                    struct dora_range rows,
                                    struct dora_utf8_batch* out_batch);

int dora_compute_bytes_prepare_validate(struct compute_bytes_source* source,
                                        struct dora_range rows,
                                        enum dora_utf8_proof proof,
                                        bool* out_is_ascii);

const struct dora_view_buffer* dora_compute_bytes_backing(const struct compute_bytes_source* source,
                                                          uint32_t* out_count);

bool dora_compute_bytes_has_owned_backing(const struct compute_bytes_source* source, bool should_use_view_descriptors);

int dora_compute_bytes_view_windows(struct scratch_scope* scope,
                                    const struct dora_view_buffer* buffer,
                                    const struct dora_view_buffer** out_backing,
                                    uint32_t* out_count);

void dora_compute_bytes_view_entry(struct dora_bytes piece,
                                   const uint8_t* buffer,
                                   uint32_t buffer_index,
                                   struct dora_binary_view_entry* out_entry);

const uint8_t* dora_compute_bytes_view_records(const struct compute_bytes_source* source);

const uint8_t* dora_compute_bytes_view_record(const struct compute_bytes_source* source,
                                              const uint8_t* records,
                                              uint64_t first,
                                              uint32_t lane);

int dora_compute_bytes_bind_literal(struct scratch_scope* scope,
                                    const struct dora_vector* source,
                                    const struct dora_vector* scalar,
                                    uint8_t source_requirements,
                                    struct compute_bytes_source inputs[2]);

int dora_compute_bytes_bind_i32(struct scratch_scope* scope,
                                size_t position,
                                const struct dora_vector* source,
                                uint8_t requirements,
                                struct compute_bytes_source* input);

int dora_compute_bytes_prepare_singleton(struct compute_bytes_source* source);

int dora_compute_bytes_prepare_current(struct compute_bytes_source* source, struct dora_range rows);

#endif
