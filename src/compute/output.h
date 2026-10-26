// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Computation result storage and publication.

#ifndef DORA_COMPUTE_OUTPUT_H
#define DORA_COMPUTE_OUTPUT_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/operand.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Outputs
//
//------------------------------------------------------------------------------

struct compute_output {
    struct dora_allocation buffers[2];
    struct dora_logical_type type;
    struct dora_pool pool;
    const struct dora_vector* validity_source;
    uint64_t length;
    uint64_t stored;
    uint64_t null_count;
    uint32_t byte_width;
    bool is_constant;
    bool is_nullable;
    uint64_t chunk_validity[KERNEL_CHUNK_WORD_COUNT];
};

static inline enum vector_content_state compute_proof_content(enum dora_utf8_proof proof) {
    return proof == DORA_UTF8_VALIDATED_ASCII ? VECTOR_CONTENT_VALIDATED_ASCII
           : proof >= DORA_UTF8_VALIDATED     ? VECTOR_CONTENT_VALIDATED
                                              : VECTOR_CONTENT_PROMISED;
}

struct compute_bytes_result {
    struct dora_allocation buffers[3];
    const struct dora_vector* validity_source;
    const struct dora_vector* buffer_source;
    const struct dora_view_buffer* backing;
    uint32_t backing_count;
    struct dora_utf8_mask validity;
    uint64_t stored;
    uint64_t nulls;
    size_t data_bytes;
    bool is_all_ascii;
};

uint64_t dora_compute_bytes_store_validity(void* validity, uint64_t first, const struct kernel_packet* packet);

int dora_compute_bytes_check_validity(const struct compute_bytes_result* result,
                                      uint64_t first,
                                      const struct kernel_packet* packet);

int dora_compute_bytes_preserve_validity(const struct dora_vector* source,
                                         bool constant,
                                         struct compute_bytes_result* result);

int dora_compute_bytes_publish(struct scratch_scope* scope,
                               struct compute_bytes_result* result,
                               const struct dora_logical_type* type,
                               enum dora_layout layout,
                               enum vector_content_state content_state,
                               bool constant,
                               uint64_t length,
                               struct dora_vector* out_result);

int dora_compute_output_init(struct dora_pool pool,
                             const struct dora_logical_type* type,
                             uint64_t length,
                             bool constant,
                             bool all_valid,
                             struct compute_output* out_output);

int dora_compute_output_preserve_validity(struct compute_output* output, const struct dora_vector* source);

int dora_compute_output_preserve_operands_validity(struct compute_output* output,
                                                   const struct dora_vector* const* sources,
                                                   const struct vector_operand* operands,
                                                   uint32_t count);

void dora_compute_output_zero_values(struct compute_output* output);

int dora_compute_output_commit(struct compute_output* output, uint64_t first, const struct kernel_result* result);

int dora_compute_nested_publish(struct scratch_scope* scope,
                                const struct vector_input* input,
                                struct dora_allocation* buffers,
                                struct dora_vector* out_result);

int dora_compute_output_set_null(struct compute_output* output, uint64_t row);

int dora_compute_output_null(struct compute_output* output);

int dora_compute_output_finish(struct compute_output* output, struct dora_vector* out_result);

int dora_compute_output_share(struct compute_output* output,
                              const struct dora_vector* source,
                              uint32_t shared_index,
                              struct dora_vector* out_result);

void dora_compute_output_release(struct compute_output* output);

struct kernel_result dora_compute_output_result(struct compute_output* output, uint64_t first, uint32_t count);

int dora_compute_bytes_offsets(struct dora_pool pool, struct compute_bytes_result* in_out_result);

int dora_compute_bytes_result_init(struct dora_pool pool,
                                   bool should_allocate_validity,
                                   struct compute_bytes_result* result);

void dora_compute_bytes_result_release(struct compute_bytes_result* result);

int dora_compute_bytes_compact(struct dora_pool pool, struct compute_bytes_result* result);

int dora_compute_output_commit_utf8_mask(struct compute_output* output,
                                         uint64_t first,
                                         uint32_t count,
                                         struct dora_utf8_mask mask);

#endif
