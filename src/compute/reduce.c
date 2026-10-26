// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/binding/reduce.h"

#include "bitmap/bitmap_internal.h"
#include "compute/output.h"
#include "compute/proof.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/compute.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "statistics/bounds.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Route
//
//------------------------------------------------------------------------------

static bool reduce_is_extremum(enum kernel_reduction operation) {
    return operation == KERNEL_REDUCTION_MIN || operation == KERNEL_REDUCTION_MAX;
}

static int reduce_bind(const struct vector_bound_input* input,
                       enum kernel_reduction operation,
                       const struct dora_logical_type* target,
                       enum dora_failure_policy policy,
                       struct kernel_reduction_binding* out_binding) {
    struct kernel_bind_request request = {
        .input_type = &input->nodes[0].field.type,
        .result_type = target,
        .policy = policy,
    };
    return dora_kernel_reduction_bind(operation, &request, out_binding);
}

static int reduce_scan(const struct kernel_reduction_binding* binding,
                       struct vector_operand* operand,
                       uint64_t length,
                       struct kernel_reduction_state* in_out_state) {
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, length);
        struct kernel_packet packet;
        status = binding->operation == KERNEL_REDUCTION_COUNT_VALID
                     ? dora_vector_operand_prepare_validity(operand, rows, &packet)
                     : dora_vector_operand_prepare(operand, rows, nullptr, &packet);
        if (status == DORA_OK) {
            status = binding->update(&packet, binding, in_out_state);
        }
    }
    return status;
}

static int reduce_nested_count(struct scratch_scope* scope,
                               const struct vector_bound_input* input,
                               struct kernel_reduction_state* in_out_state) {
    struct vector_key_workspace workspace = {};
    int status = dora_vector_key_validity_init(scope, input, KERNEL_CHUNK_ROW_COUNT, false, &workspace);
    for (uint64_t first = 0; status == DORA_OK && first < input->length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, input->length);
        uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
        status = dora_vector_key_validity(&workspace, rows, validity);
        const uint64_t valid = status == DORA_OK ? dora_bitmap_count_set_words(validity, rows.length) : 0;
        in_out_state->value.count += valid;
        in_out_state->has_value = in_out_state->has_value || valid != 0;
    }

    return status;
}

static int reduce_nested_winner(struct scratch_scope* scope,
                                const struct vector_bound_input* input,
                                bool maximum,
                                bool* out_seen,
                                uint64_t* out_winner) {
    const uint64_t length = input->is_constant && input->length != 0 ? 1 : input->length;
    struct vector_comparison_plan plan = {};
    struct vector_key_workspace workspace = {};
    int status = dora_vector_key_plan_init(scope, input, input, DORA_COMPARE_OP_LT, DORA_NULL_ORDER_LAST, &plan);
    if (status == DORA_OK) {
        status = dora_vector_key_workspace_init(scope, &plan, 1, &workspace);
    }
    bool seen = false;
    uint64_t winner = 0;
    for (uint64_t first = 0; status == DORA_OK && first < length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, length);
        uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
        status = dora_vector_key_validity(&workspace, rows, validity);
        for (uint32_t word = 0; status == DORA_OK && word < bitmap_words_for_bits(rows.length); ++word) {
            uint64_t active = validity[word];
            while (status == DORA_OK && active != 0) {
                const uint64_t row = first + (uint64_t)word * 64 + dora_bitmap_word_pop_lowest_set(&active);
                int order = 0;
                if (seen) {
                    status = dora_vector_key_order(&plan, row, winner, &workspace, &order);
                }
                if (status == DORA_OK && (!seen || (maximum ? order > 0 : order < 0))) {
                    winner = row;
                }
                seen = true;
            }
        }
    }

    *out_seen = seen;
    *out_winner = winner;
    return status;
}

static int reduce_nested_extremum(struct dora_pool pool,
                                  struct scratch_scope* scope,
                                  const struct dora_vector* source,
                                  const struct vector_bound_input* input,
                                  bool maximum,
                                  struct dora_vector* out_result) {
    bool seen = false;
    uint64_t winner = 0;
    int status = reduce_nested_winner(scope, input, maximum, &seen, &winner);
    struct dora_vector row = {};
    if (status == DORA_OK && seen) {
        status = dora_vector_slice(source, (struct dora_range){.offset = winner, .length = 1}, &row);
    } else if (status == DORA_OK) {
        const int8_t zero = 0;
        const bool valid = false;
        const struct dora_logical_type index_type = {.id = DORA_TYPE_INT8};
        struct dora_vector index = {};
        status = dora_vector_init_copy_i8(&index, pool, &index_type, &zero, &valid, 1);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_dictionary_in_pool(&row, pool, &index, source);
        }
        dora_vector_release(&index);
    }
    if (status == DORA_OK) {
        status = dora_vector_decode_error(&row, pool, out_result, nullptr);
    }
    dora_vector_release(&row);
    return status;
}

static int reduce_bytes_publish(struct scratch_scope* scope,
                                const struct vector_bound_input* input,
                                uint32_t terminal_node,
                                const struct kernel_reduction_state* state,
                                struct dora_vector* out_result) {
    const struct vector_bound_node* const terminal = &input->nodes[terminal_node];
    const struct dora_logical_type* const type = &input->nodes[0].field.type;
    const enum dora_layout layout =
        terminal->view.layout >= DORA_LAYOUT_FIXED && terminal->view.layout <= DORA_LAYOUT_BINARY_VIEW
            ? terminal->view.layout
        : type->id == DORA_TYPE_FIXED_BINARY ? DORA_LAYOUT_FIXED
                                             : DORA_LAYOUT_BINARY;
    struct dora_bytes value = {};
    if (state->has_value) {
        memcpy(&value, state->value.extremum, sizeof value);
    }
    enum vector_content_state content = compute_proof_content(vector_utf8_proof(terminal->image));
    if (type->extension_name.byte_count != 0) {
        const struct vector_image* const image = input->nodes[0].image;
        const enum vector_content_state declared =
            image == nullptr ? VECTOR_CONTENT_PROMISED : image->node.content_state;
        if (declared < content) {
            content = declared;
        }
    }
    return dora_vector_init_copy_bytes_standing(out_result,
                                                scope->pool,
                                                type,
                                                layout,
                                                &value,
                                                &state->has_value,
                                                1,
                                                content);
}

static int reduce_value(struct scratch_scope* scope,
                        const struct vector_bound_input* input,
                        enum kernel_reduction operation,
                        const struct dora_logical_type* target,
                        enum dora_failure_policy policy,
                        struct dora_vector* out_result) {
    const struct dora_pool pool = scope->pool;
    struct kernel_reduction_binding binding = {};
    int status = reduce_bind(input, operation, target, policy, &binding);
    struct vector_operand operand = {};
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(scope,
                                          0,
                                          input,
                                          operation == KERNEL_REDUCTION_COUNT_VALID ? VECTOR_OPERAND_REQUIRE_VALIDITY
                                                                                    : VECTOR_OPERAND_REQUIRE_PACKET,
                                          &operand);
    }
    const struct dora_logical_type type = binding.result_type;
    const bool bytes = reduce_is_extremum(operation) && type_is_bytes(type.id);
    struct compute_output output = {};
    if (status == DORA_OK && !bytes) {
        status = dora_compute_output_init(pool, &type, 1, false, operation == KERNEL_REDUCTION_COUNT_VALID, &output);
    }
    struct kernel_reduction_state state = {};
    if (status == DORA_OK) {
        const uint64_t length =
            reduce_is_extremum(operation) && input->is_constant && input->length != 0 ? 1 : input->length;
        dora_kernel_reduction_state_init(&binding, &state);
        status = operand.kind == VECTOR_OPERAND_NESTED ? reduce_nested_count(scope, input, &state)
                                                       : reduce_scan(&binding, &operand, length, &state);
    }
    if (status == DORA_OK && bytes) {
        status = reduce_bytes_publish(scope, input, operand.terminal_node, &state, out_result);
    } else if (status == DORA_OK) {
        struct kernel_result result = dora_compute_output_result(&output, 0, 1);
        status = binding.finish(&binding, &state, &result);
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, 0, &result);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_finish(&output, out_result);
        }
    }
    dora_compute_output_release(&output);
    return status;
}

//--------------------------------------
// Statistics Proof
//--------------------------------------

bool dora_compute_sum_is_proven(const struct dora_vector* source,
                                const struct dora_logical_type* type,
                                uint64_t row_count) {
    struct statistics_bounds bounds = {};
    if ((type->id != DORA_TYPE_INT32 && type->id != DORA_TYPE_INT64) ||
        !dora_statistics_try_get_exact_bounds(source, type, &bounds)) {
        return false;
    }
    const _BitInt(128) minimum = type->id == DORA_TYPE_INT32 ? bounds.minimum.i32 : bounds.minimum.i64;
    const _BitInt(128) maximum = type->id == DORA_TYPE_INT32 ? bounds.maximum.i32 : bounds.maximum.i64;
    const _BitInt(128) count = (_BitInt(128))row_count;
    return minimum * count > INT64_MIN && maximum * count < INT64_MAX;
}

static int compute_reduce(struct dora_pool pool,
                          const struct dora_vector* source,
                          enum kernel_reduction operation,
                          const struct dora_logical_type* target,
                          struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result)) {
        return EINVAL;
    }
    struct vector_bound_node leaf;
    struct vector_bound_input input = {};
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, source, &leaf, &input);
    }
    if (status == DORA_OK && reduce_is_extremum(operation)) {
        status = dora_vector_input_admit(&input, DORA_EXTENSION_FLAG_STORAGE_ORDER, false);
    } else if (status == DORA_OK && operation != KERNEL_REDUCTION_COUNT_VALID) {
        status = dora_vector_input_admit(&input, 0, true);
    }
    if (status == DORA_OK && reduce_is_extremum(operation) && input.nodes[0].field.type.id == DORA_TYPE_NULL) {
        status = dora_vector_init_null(&published, pool, 1);
    } else if (status == DORA_OK && reduce_is_extremum(operation) && input.nodes[0].field.type.id >= DORA_TYPE_LIST) {
        status = reduce_nested_extremum(pool, &scope, source, &input, operation == KERNEL_REDUCTION_MAX, &published);
    } else if (status == DORA_OK) {
        enum dora_failure_policy policy = DORA_FAILURE_POLICY_ERROR;
        if (operation == KERNEL_REDUCTION_SUM &&
            dora_compute_sum_is_proven(source, &input.nodes[0].field.type, input.length)) {
            policy = DORA_FAILURE_POLICY_WRAP;
        }
        status = reduce_value(&scope, &input, operation, target, policy, &published);
    }
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Aggregates
//
//------------------------------------------------------------------------------

int dora_compute_any(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_ANY, nullptr, out_result);
}

int dora_compute_all(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_ALL, nullptr, out_result);
}

int dora_compute_sum(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_SUM, nullptr, out_result);
}

int dora_compute_min(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_MIN, nullptr, out_result);
}

int dora_compute_max(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_MAX, nullptr, out_result);
}

int dora_compute_count(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_COUNT_VALID, nullptr, out_result);
}

int dora_compute_product(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return compute_reduce(pool, source, KERNEL_REDUCTION_PRODUCT, nullptr, out_result);
}

int dora_compute_product_decimal(const struct dora_vector* source,
                                 const struct dora_logical_type* target,
                                 struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || target == nullptr || target->id != DORA_TYPE_DECIMAL) {
        return EINVAL;
    }
    if (target->extension_name.byte_count != 0) {
        return ENOTSUP;
    }
    int status = dora_type_check(target);
    if (status == DORA_OK) {
        status = compute_reduce(pool, source, KERNEL_REDUCTION_PRODUCT_DECIMAL, target, out_result);
    }
    return status;
}
