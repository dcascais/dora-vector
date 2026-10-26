// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/compute.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/generate.h"
#include "kernel/packet.h"
#include "kernel/selection.h"
#include "memory/scratch.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Routes
//
//------------------------------------------------------------------------------

enum selection_kind : uint8_t {
    SELECTION_PREDICATE,
    SELECTION_NULL,
    SELECTION_MASK,
    SELECTION_RANGE
};

enum : uint32_t {
    SELECTION_INPUT_CAPACITY = 3,
    SELECTION_OUTPUT_COUNT = 3,
    SELECTION_IDENTITY_BLOCK = 65536
};

struct selection_route {
    struct scratch_scope scope;
    struct vector_bound_node leaves[SELECTION_INPUT_CAPACITY];
    struct vector_bound_input inputs[SELECTION_INPUT_CAPACITY];
    struct vector_operand operands[SELECTION_INPUT_CAPACITY];
    struct kernel_binary_binding bindings[2];
    struct kernel_range_binding range;
    struct vector_comparison_plan plans[2];
    struct vector_key_workspace workspaces[2];
    struct kernel_packet bounds[2];
    enum dora_compare_op ops[2];
    uint32_t input_count;
    uint32_t binding_count;
    uint8_t range_flags;
    enum selection_kind kind;
    bool is_null;
    bool is_constant;
    bool is_nested;
    bool is_all_null;
};

struct selection_words {
    uint64_t matches[KERNEL_CHUNK_WORD_COUNT];
    uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
    uint64_t bound_matches[KERNEL_CHUNK_WORD_COUNT];
    uint64_t bound_validity[KERNEL_CHUNK_WORD_COUNT];
    const uint64_t* valid;
};

static int selection_operands(struct selection_route* route) {
    int status = DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < route->input_count; ++index) {
        status = dora_vector_operand_bind(
            &route->scope,
            index,
            &route->inputs[index],
            VECTOR_OPERAND_REQUIRE_PACKET |
                (index == 1 && route->kind == SELECTION_PREDICATE ? VECTOR_OPERAND_REQUIRE_ACTIVE : 0U),
            &route->operands[index]);
        route->is_nested = route->is_nested || route->operands[index].kind == VECTOR_OPERAND_NESTED;
    }
    route->is_constant = route->inputs[0].is_constant;
    for (uint32_t index = 1; route->kind != SELECTION_RANGE && index < route->input_count; ++index) {
        route->is_constant = route->is_constant && route->inputs[index].is_constant;
    }
    return status;
}

static int selection_bind(struct selection_route* route, uint32_t binding, const struct dora_logical_type* right) {
    const struct dora_logical_type* const left = &route->inputs[0].nodes[0].field.type;
    struct kernel_bind_request request = {.input_type = left, .right_type = right, .comparison = route->ops[binding]};
    return dora_kernel_compare_bind(&request, &route->bindings[binding]);
}

static int selection_range_bind(struct selection_route* route) {
    const struct dora_logical_type* const source = &route->inputs[0].nodes[0].field.type;
    struct kernel_bind_request request = {
        .input_type = source,
        .right_type = &route->inputs[1].nodes[0].field.type,
        .upper_type = &route->inputs[2].nodes[0].field.type,
        .range_bounds = route->range_flags,
    };
    return dora_kernel_range_bind(&request, &route->range);
}

static int selection_bound(struct vector_operand* operand, struct kernel_packet* out_packet, bool* out_null) {
    struct kernel_packet packet;
    const int status = dora_vector_operand_prepare(operand, (struct dora_range){.length = 1}, nullptr, &packet);
    if (status != DORA_OK) {
        return status;
    }
    *out_null = packet.validity != nullptr && dora_bitmap_count_set_words(packet.validity, 1) == 0;
    *out_packet = (struct kernel_packet){
        .values = packet.values,
        .value_offset = packet.value_offset + (packet.shape == KERNEL_SHAPE_GATHERED ? packet.rows[0] : 0),
        .shape = KERNEL_SHAPE_CONSTANT,
        .placement = packet.placement,
    };
    return DORA_OK;
}

static int selection_nested_bind(struct selection_route* route) {
    if (route->kind == SELECTION_NULL || route->binding_count == 0) {
        return dora_vector_key_validity_init(&route->scope,
                                             &route->inputs[0],
                                             KERNEL_CHUNK_ROW_COUNT,
                                             false,
                                             &route->workspaces[0]);
    }
    int status = DORA_OK;
    for (uint32_t binding = 0; status == DORA_OK && binding < route->binding_count; ++binding) {
        struct vector_comparison_plan* const plan = &route->plans[binding];
        status = dora_vector_key_plan_init(&route->scope,
                                           &route->inputs[0],
                                           &route->inputs[binding + 1],
                                           route->ops[binding],
                                           DORA_NULL_ORDER_PROPAGATE,
                                           plan);
        plan->should_skip_right_on_left_null = route->kind == SELECTION_PREDICATE;
        plan->should_broadcast_right = route->kind == SELECTION_RANGE;
        if (status == DORA_OK) {
            status = dora_vector_key_workspace_init(&route->scope,
                                                    plan,
                                                    KERNEL_CHUNK_ROW_COUNT,
                                                    &route->workspaces[binding]);
        }
    }
    return status;
}

//--------------------------------------
// Chunks
//--------------------------------------

static int
selection_nested_words(struct selection_route* route, struct dora_range rows, struct selection_words* out_words) {
    const size_t word_count = bitmap_words_for_bits(rows.length);
    if (route->kind == SELECTION_NULL || route->binding_count == 0) {
        const int status = dora_vector_key_validity(&route->workspaces[0], rows, out_words->validity);
        dora_bitmap_copy_aligned_words(out_words->matches, out_words->validity, word_count);
        if (route->kind == SELECTION_NULL && route->is_null) {
            dora_bitmap_not_words(out_words->matches, rows.length);
        }
        out_words->valid = route->kind == SELECTION_NULL ? nullptr : out_words->validity;
        return status;
    }
    int status = DORA_OK;
    for (uint32_t binding = 0; status == DORA_OK && binding < route->binding_count; ++binding) {
        struct kernel_result result = {
            .values = binding == 0 ? out_words->matches : out_words->bound_matches,
            .validity = binding == 0 ? out_words->validity : out_words->bound_validity,
            .row_capacity = (uint32_t)rows.length,
        };
        status = dora_vector_key_compare_rows(&route->plans[binding], rows, &route->workspaces[binding], &result);
    }
    if (route->binding_count == 2) {
        dora_bitmap_and_aligned_words(out_words->matches, out_words->bound_matches, word_count);
        dora_bitmap_and_aligned_words(out_words->validity, out_words->bound_validity, word_count);
    }
    out_words->valid = out_words->validity;
    return status;
}

static int selection_range_words(struct selection_route* route,
                                 const struct kernel_packet* source,
                                 struct selection_words* out_words) {
    if (route->binding_count == 0) {
        dora_bitmap_set_range(out_words->matches, 0, source->count, true);
        out_words->valid = source->validity;
        return DORA_OK;
    }
    struct kernel_packet lower = route->bounds[0];
    struct kernel_packet upper = route->bounds[1];
    lower.count = source->count;
    upper.count = source->count;
    struct kernel_result result = {
        .values = out_words->matches,
        .validity = out_words->validity,
        .row_capacity = source->count,
    };
    out_words->valid = out_words->validity;
    return route->binding_count == 1 ? route->bindings[0].fill(source, &lower, &route->bindings[0], &result)
                                     : route->range.fill(source, &lower, &upper, &route->range, &result);
}

static int selection_words(struct selection_route* route, struct dora_range rows, struct selection_words* out_words) {
    if (route->is_nested) {
        return selection_nested_words(route, rows, out_words);
    }
    struct kernel_packet packets[2];
    int status = dora_vector_operand_prepare(&route->operands[0], rows, nullptr, &packets[0]);
    if (status != DORA_OK) {
        return status;
    }
    switch (route->kind) {
    case SELECTION_NULL:
        if (packets[0].validity == nullptr) {
            dora_bitmap_set_range(out_words->matches, 0, rows.length, !route->is_null);
        } else {
            dora_bitmap_copy_aligned_words(out_words->matches,
                                           packets[0].validity,
                                           (size_t)bitmap_words_for_bits(rows.length));
            if (route->is_null) {
                dora_bitmap_not_words(out_words->matches, rows.length);
            }
        }
        out_words->valid = nullptr;
        return DORA_OK;
    case SELECTION_RANGE:
        return selection_range_words(route, &packets[0], out_words);
    case SELECTION_PREDICATE:
        status = dora_vector_operand_prepare(&route->operands[1], rows, packets[0].validity, &packets[1]);
        break;
    case SELECTION_MASK:
        dora_kernel_select_mask_words(&packets[0], out_words->matches);
        out_words->valid = packets[0].validity;
        return DORA_OK;
    }
    struct kernel_result result = {
        .values = out_words->matches,
        .validity = out_words->validity,
        .row_capacity = (uint32_t)rows.length,
    };
    if (status == DORA_OK) {
        status = route->bindings[0].fill(&packets[0], &packets[1], &route->bindings[0], &result);
    }
    out_words->valid = out_words->validity;
    return status;
}

static int selection_pass(struct selection_route* route, struct kernel_select_output* in_out_output) {
    const uint64_t stored = route->is_constant && route->inputs[0].length != 0 ? 1 : route->inputs[0].length;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct selection_words words = {};
        status = selection_words(route, rows, &words);
        if (status == DORA_OK) {
            dora_kernel_select_emit(words.matches, words.valid, (uint32_t)rows.length, first, in_out_output);
        }
    }
    return status;
}

//--------------------------------------
// Outputs
//--------------------------------------

static int selection_identity(void* values, uint64_t count, bool is_wide) {
    const enum dora_physical_type physical = is_wide ? DORA_PHYSICAL_TYPE_I64 : DORA_PHYSICAL_TYPE_I32;
    const size_t width = is_wide ? sizeof(uint64_t) : sizeof(uint32_t);
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < count; first += SELECTION_IDENTITY_BLOCK) {
        const uint64_t remaining = count - first;
        status = dora_kernel_sequence_fill(
            physical,
            0,
            1,
            first,
            (uint32_t)(remaining < SELECTION_IDENTITY_BLOCK ? remaining : SELECTION_IDENTITY_BLOCK),
            (uint8_t*)values + first * width);
    }
    return status;
}

static int selection_publish(struct dora_pool pool,
                             struct dora_vector* const* outputs,
                             struct dora_allocation (*buffers)[2],
                             const struct kernel_select_output* selected) {
    const struct dora_schema_field field = {
        .type = {.id = selected->is_wide ? DORA_TYPE_UINT64 : DORA_TYPE_UINT32},
        .layout = DORA_LAYOUT_FIXED,
    };
    const uint64_t counts[SELECTION_OUTPUT_COUNT] = {
        selected->true_count,
        selected->false_count,
        selected->null_count,
    };
    int status = DORA_OK;
    uint32_t published = 0;
    while (status == DORA_OK && published < SELECTION_OUTPUT_COUNT) {
        if (outputs[published] != nullptr) {
            const struct vector_input composition =
                {.field = &field, .row_count = counts[published], .allocations = buffers[published], .buffer_count = 2};
            status = dora_vector_init_move_input(outputs[published], pool, &composition, buffers[published], nullptr);
        }
        published += status == DORA_OK ? 1U : 0U;
    }
    for (uint32_t output = 0; status != DORA_OK && output < published; ++output) {
        if (outputs[output] != nullptr) {
            dora_vector_release(outputs[output]);
        }
    }
    return status;
}

static int selection_run(struct dora_pool pool, struct selection_route* route, struct dora_vector* const* outputs) {
    const bool is_wide = route->inputs[0].length > (UINT64_C(1) << 32U);
    const bool identity = route->is_constant || route->is_all_null;
    struct kernel_select_output selected = {.is_wide = is_wide};
    int status = DORA_OK;
    if (route->is_all_null) {
        selected.null_count = route->inputs[0].length;
    } else {
        status = selection_pass(route, &selected);
    }
    if (route->is_constant && !route->is_all_null) {
        selected.true_count = selected.true_count != 0 ? route->inputs[0].length : 0;
        selected.false_count = selected.false_count != 0 ? route->inputs[0].length : 0;
        selected.null_count = selected.null_count != 0 ? route->inputs[0].length : 0;
    }
    const uint64_t counts[SELECTION_OUTPUT_COUNT] = {selected.true_count, selected.false_count, selected.null_count};
    struct dora_allocation buffers[SELECTION_OUTPUT_COUNT][2] = {};
    void* positions[SELECTION_OUTPUT_COUNT] = {};
    bool fill = false;
    for (uint32_t output = 0; status == DORA_OK && output < SELECTION_OUTPUT_COUNT; ++output) {
        size_t bytes = 0;
        if (outputs[output] == nullptr) {
            continue;
        }
        uint64_t entries = 0;
        status = ckd_add(&entries, counts[output], dora_kernel_select_padding(is_wide)) ||
                         ckd_mul(&bytes, entries, is_wide ? sizeof(uint64_t) : sizeof(uint32_t))
                     ? EOVERFLOW
                     : dora_pool_allocate(pool, bytes, &buffers[output][1]);
        positions[output] = buffers[output][1].address;
        if (status == DORA_OK && identity) {
            status = selection_identity(positions[output], counts[output], is_wide);
        }
        fill = fill || counts[output] != 0;
    }
    if (status == DORA_OK && fill && !identity) {
        for (uint32_t index = 0; index < route->input_count; ++index) {
            dora_vector_operand_reset(&route->operands[index]);
        }
        struct kernel_select_output emitted = {
            .out_true = positions[0],
            .out_false = positions[1],
            .out_null = positions[2],
            .is_wide = is_wide,
        };
        status = selection_pass(route, &emitted);
        if (status == DORA_OK &&
            (emitted.true_count != counts[0] || emitted.false_count != counts[1] || emitted.null_count != counts[2])) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK) {
        status = selection_publish(pool, outputs, buffers, &selected);
    }
    for (uint32_t output = 0; output < SELECTION_OUTPUT_COUNT; ++output) {
        dora_allocation_release(&buffers[output][1]);
    }
    return status;
}

static int selection_outputs_admit(struct dora_vector* const* outputs) {
    for (uint32_t output = 0; output < SELECTION_OUTPUT_COUNT; ++output) {
        if (outputs[output] != nullptr && !dora_vector_is_empty(outputs[output])) {
            return EINVAL;
        }
        for (uint32_t other = 0; other < output; ++other) {
            if (outputs[output] != nullptr && outputs[output] == outputs[other]) {
                return EINVAL;
            }
        }
    }
    return DORA_OK;
}

static int
selection_finish(struct dora_pool pool, struct selection_route* route, struct dora_vector* const* outputs, int status) {
    struct dora_vector staged[SELECTION_OUTPUT_COUNT] = {};
    struct dora_vector* destinations[SELECTION_OUTPUT_COUNT] = {};
    for (uint32_t index = 0; index < SELECTION_OUTPUT_COUNT; ++index) {
        destinations[index] = outputs[index] == nullptr ? nullptr : &staged[index];
    }
    if (status == DORA_OK) {
        status = selection_run(pool, route, destinations);
    }
    dora_scratch_scope_end(&route->scope);
    if (status == DORA_OK) {
        status = selection_outputs_admit(outputs);
    }
    if (status == DORA_OK) {
        for (uint32_t index = 0; index < SELECTION_OUTPUT_COUNT; ++index) {
            if (outputs[index] != nullptr) {
                *outputs[index] = staged[index];
            }
        }
    } else {
        for (uint32_t index = 0; index < SELECTION_OUTPUT_COUNT; ++index) {
            dora_vector_release(&staged[index]);
        }
    }
    return status;
}

static int selection_predicate(struct dora_pool pool,
                               const struct dora_vector* left,
                               enum dora_compare_op op,
                               const struct dora_vector* right,
                               struct dora_vector* const* outputs) {
    if (right == nullptr || op > DORA_COMPARE_OP_GE) {
        return EINVAL;
    }
    int status = selection_outputs_admit(outputs);
    struct selection_route route = {.kind = SELECTION_PREDICATE, .ops = {op}, .input_count = 2, .binding_count = 1};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &route.scope);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, left, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, right, &route.leaves[1], &route.inputs[1]);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit_pair(&route.inputs[0],
                                              &route.inputs[1],
                                              op != DORA_COMPARE_OP_EQ && op != DORA_COMPARE_OP_NE);
    }
    if (status == DORA_OK) {
        status = selection_operands(&route);
    }
    if (status == DORA_OK) {
        status = dora_vector_key_constant_null(&route.scope,
                                               route.inputs,
                                               route.operands,
                                               route.input_count,
                                               &route.is_all_null);
    }
    if (status == DORA_OK && !route.is_all_null) {
        status = route.is_nested ? selection_nested_bind(&route)
                                 : selection_bind(&route, 0, &route.inputs[1].nodes[0].field.type);
    }
    return selection_finish(pool, &route, outputs, status);
}

//------------------------------------------------------------------------------
//
// Selection
//
//------------------------------------------------------------------------------

int dora_compute_select(const struct dora_vector* left,
                        enum dora_compare_op op,
                        const struct dora_vector* right,
                        struct dora_vector* out_indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    struct dora_vector* const outputs[SELECTION_OUTPUT_COUNT] = {out_indices};
    return out_indices == nullptr ? EINVAL : selection_predicate(pool, left, op, right, outputs);
}

int dora_compute_partition(const struct dora_vector* left,
                           enum dora_compare_op op,
                           const struct dora_vector* right,
                           struct dora_vector* out_true,
                           struct dora_vector* out_false,
                           struct dora_vector* out_null) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){left, right}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    struct dora_vector* const outputs[SELECTION_OUTPUT_COUNT] = {out_true, out_false, out_null};
    return selection_predicate(pool, left, op, right, outputs);
}

static int
select_nulls(struct dora_pool pool, const struct dora_vector* source, bool is_null, struct dora_vector* out_indices) {
    struct dora_vector* const outputs[SELECTION_OUTPUT_COUNT] = {out_indices};
    int status = out_indices == nullptr ? EINVAL : selection_outputs_admit(outputs);
    struct selection_route route = {.kind = SELECTION_NULL, .input_count = 1, .is_null = is_null};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &route.scope);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, source, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = selection_operands(&route);
    }
    if (status == DORA_OK && route.is_nested) {
        status = selection_nested_bind(&route);
    }
    return selection_finish(pool, &route, outputs, status);
}

int dora_compute_select_null(const struct dora_vector* source, struct dora_vector* out_indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return select_nulls(pool, source, true, out_indices);
}

int dora_compute_select_not_null(const struct dora_vector* source, struct dora_vector* out_indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return select_nulls(pool, source, false, out_indices);
}

int dora_compute_select_mask(const struct dora_vector* mask, struct dora_vector* out_indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(mask, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    struct dora_vector* const outputs[SELECTION_OUTPUT_COUNT] = {out_indices};
    int status = out_indices == nullptr ? EINVAL : selection_outputs_admit(outputs);
    struct selection_route route = {.kind = SELECTION_MASK, .input_count = 1};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &route.scope);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, mask, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&route.inputs[0], 0, true);
    }
    if (status == DORA_OK && route.inputs[0].nodes[0].field.type.id != DORA_TYPE_BOOLEAN) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = selection_operands(&route);
    }
    return selection_finish(pool, &route, outputs, status);
}

int dora_compute_select_range(const struct dora_vector* source,
                              const struct dora_vector* lower,
                              const struct dora_vector* upper,
                              enum dora_range_bound bounds,
                              struct dora_vector* out_indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    struct dora_vector* const outputs[SELECTION_OUTPUT_COUNT] = {out_indices};
    int status =
        out_indices == nullptr || bounds > DORA_RANGE_BOUND_EXCLUSIVE ? EINVAL : selection_outputs_admit(outputs);
    struct selection_route route = {.kind = SELECTION_RANGE, .input_count = 1, .range_flags = bounds};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &route.scope);
    }
    const struct dora_vector* const sides[3] = {lower, upper, source};

    const enum dora_compare_op inclusive[2] = {DORA_COMPARE_OP_GE, DORA_COMPARE_OP_LE};

    const enum dora_compare_op exclusive[2] = {DORA_COMPARE_OP_GT, DORA_COMPARE_OP_LT};

    const uint8_t flags[2] = {DORA_RANGE_BOUND_LOWER_EXCLUSIVE, DORA_RANGE_BOUND_UPPER_EXCLUSIVE};
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, sides[2], &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&route.inputs[0], DORA_EXTENSION_FLAG_STORAGE_ORDER, false);
    }
    for (uint32_t side = 0; status == DORA_OK && side < 2; ++side) {
        if (sides[side] == nullptr) {
            continue;
        }
        struct vector_bound_input* const bound = &route.inputs[route.input_count++];
        route.ops[route.binding_count++] = (bounds & flags[side]) != 0 ? exclusive[side] : inclusive[side];
        status = dora_vector_bind_input(&route.scope, sides[side], &route.leaves[route.input_count - 1], bound);
        if (status == DORA_OK && (bound->length == 0 || (bound->length != 1 && !bound->is_constant))) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            struct vector_bound_input broadcast = *bound;
            broadcast.length = route.inputs[0].length;
            status = dora_vector_input_admit_pair(&route.inputs[0], &broadcast, true);
        }
    }
    if (status == DORA_OK) {
        status = selection_operands(&route);
    }
    if (status == DORA_OK && route.is_nested) {
        status = selection_nested_bind(&route);
    }
    for (uint32_t binding = 0; status == DORA_OK && !route.is_nested && binding < route.binding_count; ++binding) {
        bool null_bound = false;
        status = selection_bound(&route.operands[binding + 1], &route.bounds[binding], &null_bound);
        route.is_all_null = route.is_all_null || null_bound;
    }
    if (status == DORA_OK && !route.is_nested && route.binding_count == 1) {
        status = selection_bind(&route, 0, &route.inputs[1].nodes[0].field.type);
    } else if (status == DORA_OK && !route.is_nested && route.binding_count == 2) {
        status = selection_range_bind(&route);
    }
    return selection_finish(pool, &route, outputs, status);
}
