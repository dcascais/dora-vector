// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/output.h"
#include "compute/peel.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "vector/image.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Route
//
//------------------------------------------------------------------------------

static int compute_boolean(struct dora_pool pool,
                           const struct dora_vector* left,
                           const struct dora_vector* right,
                           enum kernel_boolean_operation operation,
                           struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result)) {
        return EINVAL;
    }
    struct vector_bound_node leaves[2];
    struct vector_bound_input inputs[2] = {};
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, left, &leaves[0], &inputs[0]);
    }
    const uint32_t input_count = operation == KERNEL_BOOLEAN_NOT ? 1 : 2;
    if (status == DORA_OK && input_count == 2) {
        status = dora_vector_bind_input(&scope, right, &leaves[1], &inputs[1]);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        status = dora_vector_input_admit(&inputs[index], 0, true);
        if (status == DORA_OK &&
            (inputs[index].nodes[0].field.type.id != DORA_TYPE_BOOLEAN || inputs[index].length != inputs[0].length)) {
            status = EINVAL;
        }
    }
    bool peeled = false;
    if (status == DORA_OK && input_count == 1) {
        struct dora_vector values = {};
        struct dora_vector transformed = {};
        status = dora_compute_peel_values(left, &inputs[0], false, &values, &peeled);
        if (status == DORA_OK && peeled) {
            status = compute_boolean(pool, &values, nullptr, operation, &transformed);
        }
        if (status == DORA_OK && peeled) {
            status = dora_compute_peel_publish(pool, left, &inputs[0], &transformed, &published);
        }
        dora_vector_release(&transformed);
    }
    if (peeled || status != DORA_OK) {
        dora_scratch_scope_end(&scope);
        return vector_finish_publication(&published, out_result, status);
    }
    struct kernel_binary_binding binding = {};
    struct vector_operand operands[2] = {};
    if (status == DORA_OK) {
        status = dora_kernel_boolean_bind(operation, &binding);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        status =
            dora_vector_operand_bind(&scope, index, &inputs[index], VECTOR_OPERAND_REQUIRE_PACKET, &operands[index]);
    }
    const bool constant = inputs[0].is_constant && (input_count == 1 || inputs[1].is_constant);
    const bool all_valid = dora_vector_operands_are_all_valid(operands, input_count);
    const struct dora_logical_type type = {.id = DORA_TYPE_BOOLEAN};
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, inputs[0].length, constant, all_valid, &output);
    }
    if (status == DORA_OK && input_count == 1) {
        status = dora_compute_output_preserve_operands_validity(&output,
                                                                (const struct dora_vector* const[]){left},
                                                                operands,
                                                                1);
    }
    const uint64_t stored = output.stored;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_packet packets[2];
        for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
            status = dora_vector_operand_prepare(&operands[index], rows, nullptr, &packets[index]);
        }
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        if (status == DORA_OK) {
            status = binding.fill(&packets[0], &packets[input_count - 1], &binding, &result);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &result);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Element-Wise
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_compute_and(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_boolean(pool, left, right, KERNEL_BOOLEAN_AND, out_result);
}

int dora_compute_or(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_boolean(pool, left, right, KERNEL_BOOLEAN_OR, out_result);
}

int dora_compute_xor(const struct dora_vector* left, const struct dora_vector* right, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_boolean(pool, left, right, KERNEL_BOOLEAN_XOR, out_result);
}

int dora_compute_not(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_boolean(pool, source, nullptr, KERNEL_BOOLEAN_NOT, out_result);
}
