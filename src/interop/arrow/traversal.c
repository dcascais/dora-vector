// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "interop/arrow/arrow_internal.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t ARROW_IMPORT_DEFAULT_DEPTH_LIMIT = 64;
static constexpr uint64_t ARROW_IMPORT_DEFAULT_NODE_LIMIT = 1'000'000;
static constexpr uint64_t ARROW_IMPORT_DEFAULT_EDGE_LIMIT = 1'000'000;
static constexpr uint64_t ARROW_IMPORT_DEFAULT_COORDINATE_WORK_LIMIT = 100'000'000;

struct dora_arrow_limits dora_arrow_resolve_limits(const struct dora_arrow_limits* limits) {
    struct dora_arrow_limits value = limits == nullptr ? (struct dora_arrow_limits){} : *limits;
    if (value.max_depth == 0) {
        value.max_depth = ARROW_IMPORT_DEFAULT_DEPTH_LIMIT;
    }
    if (value.max_node_count == 0) {
        value.max_node_count = ARROW_IMPORT_DEFAULT_NODE_LIMIT;
    }
    if (value.max_edge_count == 0) {
        value.max_edge_count = ARROW_IMPORT_DEFAULT_EDGE_LIMIT;
    }
    if (value.max_coordinate_work_count == 0) {
        value.max_coordinate_work_count = ARROW_IMPORT_DEFAULT_COORDINATE_WORK_LIMIT;
    }
    if (value.max_output_bytes == 0) {
        value.max_output_bytes = SIZE_MAX;
    }
    return value;
}

void dora_arrow_traversal_init(struct dora_pool pool,
                               const struct dora_arrow_limits* limits,
                               struct arrow_traversal_state* state) {
    *state = (struct arrow_traversal_state){
        .pool = pool,
        .depth_capacity = limits->max_depth < limits->max_node_count ? limits->max_depth : limits->max_node_count
    };
}

int dora_arrow_traversal_push(struct arrow_traversal_state* state, const void* source, uint32_t depth) {
    const void** stack =
        state->stack_storage.address != nullptr ? (const void**)state->stack_storage.address : state->inline_stack;
    for (uint32_t index = 0; index < depth; ++index) {
        if (stack[index] == source) {
            return EINVAL;
        }
    }
    if (depth >= state->depth_capacity) {
        return EINVAL;
    }
    if (depth >= 16) {
        size_t bytes = 0;
        if (ckd_mul(&bytes, (size_t)depth + 1, sizeof(void*))) {
            return EOVERFLOW;
        }
        const bool inline_source = state->stack_storage.address == nullptr;
        const int status = dora_memory_payload_reserve(state->pool,
                                                       bytes,
                                                       inline_source ? 0 : (size_t)depth * sizeof(void*),
                                                       &state->stack_storage);
        if (status != DORA_OK) {
            return status;
        }
        if (inline_source) {
            memcpy(state->stack_storage.address, (const void*)state->inline_stack, (size_t)depth * sizeof(void*));
        }
        stack = (const void**)state->stack_storage.address;
    }
    stack[depth] = source;
    return DORA_OK;
}

void dora_arrow_traversal_rewind(struct arrow_traversal_state* state) {
    if (state == nullptr) {
        return;
    }
    state->node_count = 0;
    state->edge_count = 0;
}
