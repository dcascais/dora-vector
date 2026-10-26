// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Routes
//
//------------------------------------------------------------------------------

enum compare_mode : uint8_t {
    COMPARE_MODE_PREDICATE,
    COMPARE_MODE_DISTINCT,
    COMPARE_MODE_ORDER
};

struct compare_request {
    enum compare_mode mode;
    enum dora_compare_op op;
    enum dora_null_order null_order;
};

struct compare_route {
    struct scratch_scope scope;
    struct vector_bound_node leaves[2];
    struct vector_bound_input inputs[2];
    struct vector_operand operands[2];
    struct kernel_binary_binding binding;
    struct vector_comparison_plan plan;
    struct vector_key_workspace workspace;
    struct compute_output output;
    struct compare_request request;
};

static int compare_kernel_bind(struct compare_route* route) {
    const struct dora_logical_type* const left = &route->inputs[0].nodes[0].field.type;
    const struct dora_logical_type* const right = &route->inputs[1].nodes[0].field.type;
    struct kernel_bind_request request = {
        .input_type = left,
        .right_type = right,
        .comparison = route->request.op,
        .null_order = route->request.null_order,
    };
    switch (route->request.mode) {
    case COMPARE_MODE_PREDICATE:
        return dora_kernel_compare_bind(&request, &route->binding);
    case COMPARE_MODE_DISTINCT:
        return dora_kernel_compare_distinct_bind(&request, &route->binding);
    case COMPARE_MODE_ORDER:
        return dora_kernel_order_bind(&request, &route->binding);
    }
    return EINVAL;
}

static int compare_chunks(struct compare_route* route) {
    struct compute_output* const output = &route->output;
    const uint64_t stored = output->stored;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_packet packets[2];
        status = dora_vector_operand_prepare(&route->operands[0], rows, nullptr, &packets[0]);
        if (status == DORA_OK) {
            status = dora_vector_operand_prepare(&route->operands[1], rows, nullptr, &packets[1]);
        }
        struct kernel_result result = dora_compute_output_result(output, first, (uint32_t)rows.length);
        if (status == DORA_OK) {
            status = route->binding.fill(&packets[0], &packets[1], &route->binding, &result);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(output, first, &result);
        }
    }
    return status;
}

static int compare_nested_bind(struct compare_route* route) {
    const struct compare_request request = route->request;
    const enum dora_null_order null_order =
        request.mode == COMPARE_MODE_PREDICATE ? DORA_NULL_ORDER_PROPAGATE : request.null_order;
    int status = dora_vector_key_plan_init(&route->scope,
                                           &route->inputs[0],
                                           &route->inputs[1],
                                           request.op,
                                           null_order,
                                           &route->plan);
    if (status == DORA_OK) {
        status = dora_vector_key_workspace_init(&route->scope, &route->plan, KERNEL_CHUNK_ROW_COUNT, &route->workspace);
    }
    return status;
}

static int compare_nested(struct compare_route* route) {
    struct compute_output* const output = &route->output;
    const uint64_t stored = output->stored;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_result result = dora_compute_output_result(output, first, (uint32_t)rows.length);
        status = route->request.mode == COMPARE_MODE_ORDER
                     ? dora_vector_key_order_rows(&route->plan, rows, &route->workspace, &result)
                     : dora_vector_key_compare_rows(&route->plan, rows, &route->workspace, &result);
        if (status == DORA_OK) {
            status = dora_compute_output_commit(output, first, &result);
        }
    }
    return status;
}

static int compute_comparison(struct dora_pool pool,
                              const struct dora_vector* left,
                              const struct dora_vector* right,
                              struct compare_request request,
                              struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result) || request.op > DORA_COMPARE_OP_GE ||
        request.null_order > DORA_NULL_ORDER_PROPAGATE ||
        (request.mode == COMPARE_MODE_DISTINCT && request.null_order == DORA_NULL_ORDER_PROPAGATE)) {
        return EINVAL;
    }
    struct compare_route route = {.request = request};
    struct dora_vector published = {};
    int status = dora_scratch_scope_begin(pool, &route.scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, left, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, right, &route.leaves[1], &route.inputs[1]);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit_pair(&route.inputs[0],
                                              &route.inputs[1],
                                              request.mode == COMPARE_MODE_ORDER || (request.op != DORA_COMPARE_OP_EQ &&
                                                                                     request.op != DORA_COMPARE_OP_NE));
    }
    for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
        status = dora_vector_operand_bind(&route.scope,
                                          index,
                                          &route.inputs[index],
                                          VECTOR_OPERAND_REQUIRE_PACKET,
                                          &route.operands[index]);
    }
    bool null_constant = false;
    if (status == DORA_OK && request.mode == COMPARE_MODE_PREDICATE) {
        status = dora_vector_key_constant_null(&route.scope, route.inputs, route.operands, 2, &null_constant);
    }
    const bool constant = null_constant || (route.inputs[0].is_constant && route.inputs[1].is_constant);
    const bool nested =
        route.operands[0].kind == VECTOR_OPERAND_NESTED || route.operands[1].kind == VECTOR_OPERAND_NESTED;
    if (status == DORA_OK && !null_constant) {
        status = nested ? compare_nested_bind(&route) : compare_kernel_bind(&route);
    }
    const struct dora_logical_type type = {
        .id = request.mode == COMPARE_MODE_ORDER ? DORA_TYPE_INT8 : DORA_TYPE_BOOLEAN,
    };
    const bool all_valid =
        request.mode == COMPARE_MODE_DISTINCT ||
        (request.mode == COMPARE_MODE_ORDER && request.null_order != DORA_NULL_ORDER_PROPAGATE) ||
        (request.mode == COMPARE_MODE_PREDICATE && dora_vector_operands_are_all_valid(route.operands, 2));
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, route.inputs[0].length, constant, all_valid, &route.output);
    }
    if (status == DORA_OK && !nested && request.mode == COMPARE_MODE_PREDICATE) {
        status = dora_compute_output_preserve_operands_validity(&route.output,
                                                                (const struct dora_vector* const[]){left, right},
                                                                route.operands,
                                                                2);
    }
    if (status == DORA_OK && null_constant) {
        status = dora_compute_output_null(&route.output);
    } else if (status == DORA_OK) {
        status = nested ? compare_nested(&route) : compare_chunks(&route);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&route.output, &published);
    }
    dora_compute_output_release(&route.output);

    dora_scratch_scope_end(&route.scope);
    return vector_finish_publication(&published, out_result, status);
}

static int compare_validity_words(struct vector_operand* operand, struct dora_range rows, uint64_t* out_words) {
    struct kernel_packet packet;
    const int status = dora_vector_operand_prepare_validity(operand, rows, &packet);
    if (status == DORA_OK && packet.validity == nullptr) {
        dora_bitmap_set_range(out_words, 0, rows.length, true);
    } else if (status == DORA_OK) {
        dora_bitmap_copy_aligned_words(out_words, packet.validity, (size_t)bitmap_words_for_bits(rows.length));
    }
    return status;
}

static int compute_null_test(struct dora_pool pool,
                             const struct dora_vector* source,
                             bool is_null,
                             struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result)) {
        return EINVAL;
    }
    struct vector_bound_node leaf;
    struct vector_bound_input input = {};
    struct vector_operand operand = {};
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, source, &leaf, &input);
    }
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(&scope, 0, &input, VECTOR_OPERAND_REQUIRE_VALIDITY, &operand);
    }
    const bool shared = status == DORA_OK && !is_null && !input.is_constant &&
                        source->handle.pool_index == pool.index &&
                        (operand.kind == VECTOR_OPERAND_DIRECT ||
                         (operand.kind == VECTOR_OPERAND_BYTES && operand.chain_length == 0)) &&
                        input.nodes[0].view.validity.data != nullptr;
    const struct dora_logical_type type = {.id = DORA_TYPE_BOOLEAN};
    struct compute_output output = {.type = type, .length = input.length};
    if (status == DORA_OK && !shared) {
        status = dora_compute_output_init(pool, &type, input.length, input.is_constant, true, &output);
        if (status == DORA_OK) {
            dora_compute_output_zero_values(&output);
        }
    }
    const uint64_t stored = shared ? 0 : input.is_constant && output.length != 0 ? 1 : output.length;
    struct vector_key_workspace workspace = {};
    if (status == DORA_OK && operand.kind == VECTOR_OPERAND_NESTED) {
        status = dora_vector_key_validity_init(&scope, &input, KERNEL_CHUNK_ROW_COUNT, false, &workspace);
    }
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        const struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        status = operand.kind == VECTOR_OPERAND_NESTED ? dora_vector_key_validity(&workspace, rows, result.values)
                                                       : compare_validity_words(&operand, rows, result.values);
        if (status == DORA_OK && is_null) {
            dora_bitmap_not_words(result.values, rows.length);
        }
    }

    if (status == DORA_OK && shared) {
        status = dora_compute_output_share(&output, source, 1, &published);
    } else if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Comparison
//
//------------------------------------------------------------------------------

int dora_compute_compare(const struct dora_vector* left,
                         enum dora_compare_op op,
                         const struct dora_vector* right,
                         struct dora_vector* out_mask) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct compare_request request = {
        .mode = COMPARE_MODE_PREDICATE,
        .op = op,
        .null_order = DORA_NULL_ORDER_PROPAGATE,
    };
    return compute_comparison(pool, left, right, request, out_mask);
}

int dora_compute_compare_distinct(const struct dora_vector* left,
                                  enum dora_compare_op op,
                                  const struct dora_vector* right,
                                  enum dora_null_order null_order,
                                  struct dora_vector* out_mask) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct compare_request request = {.mode = COMPARE_MODE_DISTINCT, .op = op, .null_order = null_order};
    return compute_comparison(pool, left, right, request, out_mask);
}

int dora_compute_order(const struct dora_vector* left,
                       const struct dora_vector* right,
                       enum dora_null_order null_order,
                       struct dora_vector* out_order) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct compare_request request = {.mode = COMPARE_MODE_ORDER, .null_order = null_order};
    return compute_comparison(pool, left, right, request, out_order);
}

//------------------------------------------------------------------------------
//
// Null Predicates
//
//------------------------------------------------------------------------------

int dora_compute_is_null(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_null_test(pool, source, true, out_result);
}

int dora_compute_is_not_null(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_null_test(pool, source, false, out_result);
}
