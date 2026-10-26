// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "hash/hash.h"

#include "bitmap/bitmap_internal.h"
#include "compute/compute_internal.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/read.h"
#include "kernel/binding/request.h"
#include "kernel/decode.h"
#include "kernel/hash_cache.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Routes
//
//------------------------------------------------------------------------------

struct hash_route {
    struct scratch_scope scope;
    struct vector_bound_node leaves[2];
    struct vector_bound_input inputs[2];
    struct vector_operand operands[2];
    struct compute_output output;
    uint64_t* cache;
    uint32_t input_count;
};

static void hash_route_release(struct hash_route* route) {
    dora_compute_output_release(&route->output);
    dora_scratch_scope_end(&route->scope);
}

static bool hash_route_has_repeated_values(const struct hash_route* route, uint32_t count) {
    const struct vector_operand* const operand = &route->operands[route->input_count - 1];
    const struct dora_vector_view* const terminal = operand->bytes.view;
    return operand->kind == VECTOR_OPERAND_BYTES && operand->chain_length != 0 &&
           terminal->selection.indices == nullptr && terminal->length < count;
}

static int hash_route_operands(struct dora_pool pool, struct hash_route* route) {
    int status = DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < route->input_count; ++index) {
        status = dora_vector_operand_bind(&route->scope,
                                          index,
                                          &route->inputs[index],
                                          VECTOR_OPERAND_REQUIRE_PACKET,
                                          &route->operands[index]);
    }
    const bool constant = route->inputs[0].is_constant && (route->input_count == 1 || route->inputs[1].is_constant);
    const struct dora_logical_type type = {.id = DORA_TYPE_UINT64};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, route->inputs[0].length, constant, true, &route->output);
    }
    const struct vector_operand* const operand = &route->operands[route->input_count - 1];
    const uint64_t count =
        route->output.stored < KERNEL_CHUNK_ROW_COUNT ? route->output.stored : KERNEL_CHUNK_ROW_COUNT;
    if (status == DORA_OK && hash_route_has_repeated_values(route, (uint32_t)count)) {
        status = dora_scratch_slot_require(&route->scope,
                                           vector_operand_slot(route->input_count - 1, VECTOR_OPERAND_SLOT_CACHE),
                                           (void**)&route->cache,
                                           (size_t)operand->bytes.view->length * sizeof(*route->cache),
                                           64);
        if (status == DORA_OK) {
            for (uint64_t i = 0; i < operand->bytes.view->length; ++i) {
                route->cache[i] = HASH_NULL_VALUE;
            }
        }
    }
    return status;
}

static int hash_route_chunks(struct hash_route* route,
                             uint64_t seed,
                             const struct kernel_unary_binding* unary,
                             const struct kernel_binary_binding* binary) {
    const uint64_t stored = route->output.stored;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct kernel_packet packets[2];
        for (uint32_t index = 0; status == DORA_OK && index < route->input_count; ++index) {
            status = dora_vector_operand_prepare(&route->operands[index], rows, nullptr, &packets[index]);
        }
        struct kernel_result result = dora_compute_output_result(&route->output, first, (uint32_t)rows.length);
        if (status == DORA_OK && hash_route_has_repeated_values(route, (uint32_t)rows.length)) {
            const struct vector_operand* const operand = &route->operands[route->input_count - 1];
            const struct kernel_hash_cache entries = {
                .base_rows = operand->rows64,
                .entries = route->cache,
                .base_row_count = (uint32_t)operand->bytes.view->length,
            };
            dora_kernel_hash_bytes_cached(&packets[route->input_count - 1],
                                          &entries,
                                          seed,
                                          route->input_count == 1 ? nullptr : &packets[0],
                                          result.values);
        } else if (status == DORA_OK && unary != nullptr) {
            status = unary->fill(&packets[0], unary, &result);
        } else if (status == DORA_OK && binary != nullptr) {
            status = binary->fill(&packets[0], &packets[1], binary, &result);
        }
    }
    return status;
}

static int hash_route_prior(struct vector_operand* hashes, struct dora_range rows, uint64_t* in_out_hashes) {
    struct kernel_packet packet;
    const int status = dora_vector_operand_prepare(hashes, rows, nullptr, &packet);
    if (status != DORA_OK) {
        return status;
    }
    const uint8_t* const values = packet.values;
    if (packet.shape == KERNEL_SHAPE_FLAT) {
        for (uint32_t lane = 0; lane < packet.count; ++lane) {
            const uint64_t prior = type_load_u64(values, packet.value_offset + lane);
            in_out_hashes[lane] = hash_value_combine(prior, in_out_hashes[lane]);
        }
    } else if (packet.shape == KERNEL_SHAPE_CONSTANT) {
        const uint64_t prior = type_load_u64(values, packet.value_offset);
        for (uint32_t lane = 0; lane < packet.count; ++lane) {
            in_out_hashes[lane] = hash_value_combine(prior, in_out_hashes[lane]);
        }
    } else {
        for (uint32_t lane = 0; lane < packet.count; ++lane) {
            const uint64_t prior = type_load_u64(values, packet.value_offset + packet.rows[lane]);
            in_out_hashes[lane] = hash_value_combine(prior, in_out_hashes[lane]);
        }
    }
    return DORA_OK;
}

static int hash_route_nested(struct hash_route* route, uint64_t seed) {
    struct vector_hash_plan plan = {};
    struct vector_key_workspace workspace = {};
    int status = dora_vector_key_hash_plan_init(&route->scope, &route->inputs[route->input_count - 1], seed, &plan);
    if (status == DORA_OK) {
        status = dora_vector_key_hash_workspace_init(&route->scope, &plan, &workspace);
    }
    const uint64_t stored = route->output.stored;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        const struct kernel_result result = dora_compute_output_result(&route->output, first, (uint32_t)rows.length);
        status = dora_vector_key_hash_rows(&plan, rows, &workspace, result.values);
        if (status == DORA_OK && route->input_count == 2) {
            status = hash_route_prior(&route->operands[0], rows, result.values);
        }
    }

    return status;
}

//------------------------------------------------------------------------------
//
// Cached Dictionary Hashes
//
//------------------------------------------------------------------------------

static bool hash_is_cacheable(const struct vector_bound_input* input) {
    const struct dora_vector_view* const view = &input->nodes[0].view;
    if (view->layout != DORA_LAYOUT_DICTIONARY || view->selection.indices != nullptr || input->length == 0) {
        return false;
    }
    return type_is_bytes(input->nodes[dora_vector_input_child(0, input, 0)].view.type);
}

static int hash_dictionary_fill(const struct dora_vector_view* view,
                                struct dora_pool pool,
                                const struct dora_vector_view* hashes,
                                uint32_t* map,
                                uint64_t* validity,
                                struct dora_vector* out_hashes) {
    struct kernel_decode_binding indices = {};
    struct compute_output output = {};
    const struct dora_logical_type type = {.id = DORA_TYPE_UINT64};
    int status = dora_kernel_decode_bind(view, &indices);
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, view->length, false, true, &output);
    }
    for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
        const uint32_t count = (uint32_t)vector_operand_chunk(first, view->length).length;
        bitmap_prefix_words(count, validity, (size_t)bitmap_words_for_bits(count));
        indices.narrow(&indices, first, count, validity, map);
        const struct kernel_result result = dora_compute_output_result(&output, first, count);
        dora_kernel_gather_u64((const uint64_t*)hashes->values.data + hashes->offset,
                               map,
                               validity,
                               count,
                               result.values,
                               HASH_NULL_VALUE);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, out_hashes);
    }
    dora_compute_output_release(&output);
    return status;
}

//------------------------------------------------------------------------------
//
// Hashing
//
//------------------------------------------------------------------------------

int dora_compute_hash_in_pool(struct dora_pool pool,
                              const struct dora_vector* source,
                              uint64_t seed,
                              struct dora_vector* out_hashes) {
    if (!dora_vector_is_empty(out_hashes)) {
        return EINVAL;
    }
    struct hash_route route = {.input_count = 1};
    struct dora_vector published = {};
    int status = dora_scratch_scope_begin(pool, &route.scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, source, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&route.inputs[0], DORA_EXTENSION_FLAG_STORAGE_EQUALITY, false);
    }
    bool peeled = false;
    if (status == DORA_OK && hash_is_cacheable(&route.inputs[0])) {
        const struct vector_bound_input* const input = &route.inputs[0];
        struct dora_vector values = {};
        status = dora_vector_get_child(source, 0, &values);
        if (status == DORA_OK && values.rows.length <= input->length / 2 &&
            values.rows.length <= (uint64_t)UINT32_MAX + 1) {
            struct dora_vector cached = {};
            struct dora_vector_view hashes = {};
            status = dora_compute_hash_in_pool(pool, &values, seed, &cached);
            if (status == DORA_OK) {
                status = dora_vector_get_view(&cached, &hashes);
            }
            if (status == DORA_OK && hashes.layout == DORA_LAYOUT_FIXED && hashes.selection.indices == nullptr) {
                peeled = true;
                uint32_t* map = nullptr;
                uint64_t* validity = nullptr;
                const size_t capacity =
                    input->length < KERNEL_CHUNK_ROW_COUNT ? (size_t)input->length : KERNEL_CHUNK_ROW_COUNT;
                status = dora_scratch_slot_require(&route.scope,
                                                   vector_operand_slot(0, VECTOR_OPERAND_SLOT_ROWS),
                                                   (void**)&map,
                                                   capacity * sizeof(*map),
                                                   64);
                if (status == DORA_OK) {
                    status = dora_scratch_slot_require(&route.scope,
                                                       vector_operand_slot(0, VECTOR_OPERAND_SLOT_CACHE),
                                                       (void**)&validity,
                                                       (size_t)bitmap_words_for_bits(capacity) * sizeof(*validity),
                                                       64);
                }
                if (status == DORA_OK) {
                    status = hash_dictionary_fill(&input->nodes[0].view, pool, &hashes, map, validity, &published);
                }
            }
            dora_vector_release(&cached);
        }
    }
    if (status == DORA_OK && !peeled) {
        status = hash_route_operands(pool, &route);
    }
    struct kernel_bind_request request = {.seed = seed};
    struct kernel_unary_binding binding = {};
    const bool nested = route.operands[0].kind == VECTOR_OPERAND_NESTED;
    if (status == DORA_OK && !peeled && !nested) {
        request.input_type = &route.inputs[0].nodes[0].field.type;
        status = dora_kernel_hash_bind(&request, &binding);
    }
    if (status == DORA_OK && !peeled) {
        status = nested ? hash_route_nested(&route, seed) : hash_route_chunks(&route, seed, &binding, nullptr);
    }
    if (status == DORA_OK && !peeled) {
        status = dora_compute_output_finish(&route.output, &published);
    }
    hash_route_release(&route);
    return vector_finish_publication(&published, out_hashes, status);
}

int dora_compute_hash(const struct dora_vector* source, uint64_t seed, struct dora_vector* out_hashes) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_compute_hash_in_pool(pool, source, seed, out_hashes);
}

int dora_compute_hash_combine(const struct dora_vector* hashes,
                              const struct dora_vector* source,
                              struct dora_vector* out_hashes) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(hashes, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_hashes)) {
        return EINVAL;
    }
    uint64_t null_count = 0;
    struct hash_route route = {.input_count = 2};
    struct dora_vector published = {};
    int status = dora_scratch_scope_begin(pool, &route.scope);
    if (status == DORA_OK) {
        status = dora_vector_get_null_count(hashes, &null_count);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, hashes, &route.leaves[0], &route.inputs[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&route.scope, source, &route.leaves[1], &route.inputs[1]);
    }
    if (status == DORA_OK) {
        const struct dora_logical_type* const type = &route.inputs[0].nodes[0].field.type;
        status = type->id != DORA_TYPE_UINT64 || type->extension_name.byte_count != 0 || null_count != 0 ||
                         route.inputs[0].length != route.inputs[1].length
                     ? EINVAL
                     : dora_vector_input_admit(&route.inputs[1], DORA_EXTENSION_FLAG_STORAGE_EQUALITY, false);
    }
    if (status == DORA_OK) {
        status = hash_route_operands(pool, &route);
    }
    struct kernel_bind_request request = {};
    struct kernel_binary_binding binding = {};
    const bool nested = route.operands[1].kind == VECTOR_OPERAND_NESTED;
    if (status == DORA_OK && !nested) {
        request.input_type = &route.inputs[0].nodes[0].field.type;
        request.right_type = &route.inputs[1].nodes[0].field.type;
        status = dora_kernel_hash_combine_bind(&request, &binding);
    }
    if (status == DORA_OK) {
        status = nested ? hash_route_nested(&route, 0) : hash_route_chunks(&route, request.seed, nullptr, &binding);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&route.output, &published);
    }
    hash_route_release(&route);
    return vector_finish_publication(&published, out_hashes, status);
}
