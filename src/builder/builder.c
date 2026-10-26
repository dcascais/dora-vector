// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/builder.h"

#include "bitmap/bitmap_internal.h"
#include "builder/builder_internal.h"
#include "builder/state.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "extension/extension_internal.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "types/array_count.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Builder State
//
//------------------------------------------------------------------------------

static constexpr size_t BUILDER_NODE_RESERVED_BUFFER_CAPACITY = 5;

static bool builder_is_empty(const struct dora_builder* builder) {
    return builder != nullptr && handle_is_empty(builder->handle) && builder->node_index == 0;
}

// Preorder releases parent claims first; partial reclaim resumes after EBUSY.
static int builder_reclaim(struct builder_state* state) {
    struct dora_vector* const shared = state->shared_vectors.address;
    void* const* const addresses = (void* const*)state->shared_addresses.address;
    struct builder_node_state* const nodes = state->nodes;
    struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t index = 0; index < state->node_count; ++index) {
        if (shared[index].handle.slot_index == 0) {
            continue;
        }
        struct dora_allocation local[BUILDER_NODE_RESERVED_BUFFER_CAPACITY] = {};
        const uint32_t capacity = builder_node_buffer_count(&nodes[index]);
        struct dora_allocation storage = {};
        size_t bytes;
        if (ckd_mul(&bytes, capacity, sizeof(struct dora_allocation))) {
            return EOVERFLOW;
        }
        int status = DORA_OK;
        if (capacity > BUILDER_NODE_RESERVED_BUFFER_CAPACITY) {
            struct vector_image* image = nullptr;
            status = dora_vector_resolve(&shared[index], &image);
            if (status != DORA_OK || image->claims.claim_count != 1) {
                return status != DORA_OK ? status : EBUSY;
            }
            status = dora_pool_allocate(state->storage.pool, bytes, &storage);
        }
        struct dora_allocation* const rows = storage.address != nullptr ? storage.address : local;
        uint32_t count = 0;
        if (status == DORA_OK) {
            status = dora_vector_reclaim_buffers(&shared[index], rows, capacity, &count);
        }
        if (status != DORA_OK) {
            dora_allocation_release(&storage);
            return status;
        }
        for (uint32_t part = 0, number = nodes[index].first_buffer; part < count;
             ++part, number = builder_next_buffer(state, &nodes[index], number)) {
            if (part < capacity && rows[part].address != nullptr && rows[part].address == addresses[number]) {
                buffers[number].allocation = rows[part];
            } else {
                dora_allocation_release(&rows[part]);
            }
        }
        dora_allocation_release(&storage);
    }
    state->capacity_bytes = 0;
    for (uint32_t index = 0; index < state->buffer_count; ++index) {
        state->capacity_bytes += buffers[index].allocation.capacity_bytes;
    }
    dora_allocation_release(&state->shared_vectors);
    dora_allocation_release(&state->shared_addresses);
    state->phase = BUILDER_PHASE_OPEN;
    return DORA_OK;
}

static void builder_discard(struct builder_state* state) {
    if (state == nullptr) {
        return;
    }
    dora_builder_run_discard(state);
    dora_builder_discard_window(state);
    struct dora_vector* const shared = state->shared_vectors.address;
    for (uint32_t index = 0; shared != nullptr && index < state->node_count; ++index) {
        dora_vector_release(&shared[index]);
    }
    dora_allocation_release(&state->shared_vectors);
    dora_allocation_release(&state->shared_addresses);
    struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t index = 0; index < state->buffer_count; ++index) {
        dora_allocation_release(&buffers[index].allocation);
    }
    struct builder_node_state* const nodes = state->nodes;
    for (uint32_t index = 0; index < state->node_count; ++index) {
        dora_allocation_release(&nodes[index].extension_parameters);
        dora_vector_release(&nodes[index].dictionary);
    }
    dora_statistics_accumulator_release(&state->statistics);
    dora_allocation_release(&state->buffer_storage);
    dora_allocation_release(&state->bookkeeping);
    dora_schema_release(&state->schema);
    struct dora_allocation storage = state->storage;
    dora_allocation_release(&storage);
}

static int allocate_zero(struct dora_pool pool, size_t count, size_t width, struct dora_allocation* destination) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, width)) {
        return EOVERFLOW;
    }
    if (bytes == 0) {
        return DORA_OK;
    }
    const int status = dora_memory_payload_allocate(pool, bytes, destination);
    if (status == DORA_OK) {
        memset(destination->address, 0, bytes);
    }
    return status;
}

static int node_buffers(struct builder_node_state* node,
                        const struct dora_schema_field* field,
                        struct builder_buffer_state* buffers) {
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    const struct dora_logical_type type =
        field->layout == DORA_LAYOUT_DICTIONARY ? (struct dora_logical_type){.id = field->index_type} : field->type;
    int status = type_physical(&type, &physical);
    if (status != DORA_OK) {
        return status;
    }
    uint32_t width = physical_width(physical) * 8U;
    if (field->type.id == DORA_TYPE_FIXED_BINARY && field->layout != DORA_LAYOUT_DICTIONARY &&
        ckd_mul(&width, field->type.parameters.byte_width, 8U)) {
        return EOVERFLOW;
    }
    buffers[0] = (struct builder_buffer_state){.role = BUILDER_BUFFER_VALIDITY_BITS, .unit_bit_width = 1};
    buffers[1] = (struct builder_buffer_state){.role = BUILDER_BUFFER_FIXED_VALUES, .unit_bit_width = width};
    node->buffer_count = 2;
    switch (field->layout) {
    case DORA_LAYOUT_NULL:
    case DORA_LAYOUT_RUN_END:
        node->buffer_count = 0;
        break;
    case DORA_LAYOUT_STRUCT:
    case DORA_LAYOUT_FIXED_LIST:
        node->buffer_count = 1;
        break;
    case DORA_LAYOUT_BITS:
        buffers[1].role = BUILDER_BUFFER_VALUE_BITS;
        buffers[1].unit_bit_width = 1;
        break;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_LIST:
    case DORA_LAYOUT_LARGE_LIST:
        buffers[1].unit_bit_width =
            field->layout == DORA_LAYOUT_LARGE_BINARY || field->layout == DORA_LAYOUT_LARGE_LIST ? 64 : 32;
        buffers[1].role = buffers[1].unit_bit_width == 64 ? BUILDER_BUFFER_OFFSETS64 : BUILDER_BUFFER_OFFSETS32;
        if (field->layout == DORA_LAYOUT_BINARY || field->layout == DORA_LAYOUT_LARGE_BINARY) {
            buffers[2] = (struct builder_buffer_state){.role = BUILDER_BUFFER_BYTES, .unit_bit_width = 8};
            node->buffer_count = 3;
        }
        break;
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        buffers[1].unit_bit_width = field->layout == DORA_LAYOUT_LIST_VIEW ? 32 : 64;
        buffers[1].role = field->layout == DORA_LAYOUT_LIST_VIEW ? BUILDER_BUFFER_OFFSETS32 : BUILDER_BUFFER_OFFSETS64;
        buffers[2] = (struct builder_buffer_state){
            .unit_bit_width = buffers[1].unit_bit_width,
            .role = field->layout == DORA_LAYOUT_LIST_VIEW ? BUILDER_BUFFER_SIZES32 : BUILDER_BUFFER_SIZES64
        };
        node->buffer_count = 3;
        break;
    case DORA_LAYOUT_BINARY_VIEW:
        buffers[1] = (struct builder_buffer_state){.role = BUILDER_BUFFER_BINARY_VIEW, .unit_bit_width = 128};
        buffers[2] = (struct builder_buffer_state){.role = BUILDER_BUFFER_BYTES, .unit_bit_width = 8};
        buffers[3] = (struct builder_buffer_state){.role = BUILDER_BUFFER_VARIADIC_SIZES, .unit_bit_width = 64};
        node->buffer_count = 4;
        break;
    case DORA_LAYOUT_DENSE_UNION:
    case DORA_LAYOUT_SPARSE_UNION:
        buffers[0] = (struct builder_buffer_state){.role = BUILDER_BUFFER_TYPE_IDS, .unit_bit_width = 8};
        buffers[1] = (struct builder_buffer_state){.role = BUILDER_BUFFER_MEMBER_ROWS, .unit_bit_width = 32};
        node->buffer_count = field->layout == DORA_LAYOUT_DENSE_UNION ? 2 : 1;
        break;
    default:
        break;
    }
    return DORA_OK;
}

static int
logical_schema(const struct dora_schema* schema, struct dora_schema* out_values, struct dora_schema_field* out_field) {
    struct dora_schema values = dora_schema_borrow(schema);
    int status = dora_schema_get_field(&values, out_field);
    while (status == DORA_OK &&
           (out_field->layout == DORA_LAYOUT_DICTIONARY || out_field->layout == DORA_LAYOUT_RUN_END)) {
        struct dora_schema child = {};
        status = dora_schema_get_child(&values, out_field->layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
        if (status == DORA_OK) {
            values = child;
            status = dora_schema_get_field(&values, out_field);
        }
    }
    if (status == DORA_OK) {
        *out_values = values;
    }
    return status;
}

static int dictionary_schema_check(const struct dora_schema* declared,
                                   const struct dora_schema* supplied,
                                   bool* out_physical_match) {
    struct dora_schema values[2] = {dora_schema_borrow(declared), dora_schema_borrow(supplied)};
    struct dora_schema_field fields[2] = {};
    int status = dora_schema_get_field(&values[0], &fields[0]);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&values[1], &fields[1]);
    }
    if (status != DORA_OK) {
        return status;
    }
    bool paired_declarations = true;
    bool target_sorted = false;
    for (;;) {
        target_sorted = target_sorted || (fields[0].flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) != 0;
        const uint32_t semantic = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE | DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED;
        if ((paired_declarations && (fields[0].flags & semantic) != (fields[1].flags & semantic)) ||
            (fields[0].layout == DORA_LAYOUT_DICTIONARY &&
             ((fields[0].flags & DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED) !=
              (fields[1].layout == DORA_LAYOUT_DICTIONARY ? fields[1].flags & DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED
                                                          : 0)))) {
            return EINVAL;
        }
        if (fields[0].layout == DORA_LAYOUT_RUN_END && fields[1].layout == DORA_LAYOUT_RUN_END) {
            struct dora_schema ends[2] = {};
            struct dora_schema_field end_fields[2] = {};
            for (uint32_t side = 0; status == DORA_OK && side < 2; ++side) {
                status = dora_schema_get_physical_child(&values[side], 0, &ends[side]);
                if (status == DORA_OK) {
                    status = dora_schema_get_field(&ends[side], &end_fields[side]);
                }
            }
            if (status != DORA_OK) {
                return status;
            }
            *out_physical_match = *out_physical_match && end_fields[0].layout == end_fields[1].layout &&
                                  end_fields[0].flags == end_fields[1].flags &&
                                  dora_type_is_equal(&end_fields[0].type, &end_fields[1].type, 0);
        }
        *out_physical_match =
            *out_physical_match && fields[0].layout == fields[1].layout && fields[0].index_type == fields[1].index_type;
        const bool encoded[2] = {
            fields[0].layout == DORA_LAYOUT_DICTIONARY || fields[0].layout == DORA_LAYOUT_RUN_END,
            fields[1].layout == DORA_LAYOUT_DICTIONARY || fields[1].layout == DORA_LAYOUT_RUN_END
        };
        if (!encoded[0] && !encoded[1]) {
            break;
        }
        paired_declarations = paired_declarations && encoded[0] && encoded[1] && fields[0].layout == fields[1].layout;
        for (uint32_t side = 0; status == DORA_OK && side < 2; ++side) {
            if (!encoded[side]) {
                continue;
            }
            struct dora_schema child = {};
            status = dora_schema_get_physical_child(&values[side],
                                                    fields[side].layout == DORA_LAYOUT_DICTIONARY ? 0 : 1,
                                                    &child);
            if (status == DORA_OK) {
                values[side] = child;
                status = dora_schema_get_field(&values[side], &fields[side]);
            }
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    if ((!paired_declarations && target_sorted && (fields[1].flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) == 0) ||
        (fields[0].flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) !=
            (fields[1].flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED)) {
        return EINVAL;
    }
    if (fields[0].child_count != fields[1].child_count ||
        !dora_type_is_equal(&fields[0].type, &fields[1].type, fields[0].child_count)) {
        return EINVAL;
    }
    for (uint32_t child = 0; status == DORA_OK && child < fields[0].child_count; ++child) {
        struct dora_schema first = {};
        struct dora_schema second = {};
        status = dora_schema_get_physical_child(&values[0], child, &first);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(&values[1], child, &second);
        }
        if (status == DORA_OK) {
            status = dictionary_schema_check(&first, &second, out_physical_match);
        }
    }
    return status;
}

static int builder_bind(struct builder_state* state,
                        struct builder_node_state* node,
                        const struct dora_schema_field* node_field,
                        struct dora_error* out_error) {
    if (node_field->type.extension_name.byte_count == 0) {
        return DORA_OK;
    }
    int status = dora_extension_find(state->storage.pool, node_field->type.extension_name, &node->extension);
    if (status != DORA_OK || node->extension == nullptr || node->extension->bind == nullptr) {
        return status;
    }
    struct dora_schema values = {};
    struct dora_schema_field field = {};
    status = logical_schema(&node->schema, &values, &field);
    field.type.extension_name = (struct dora_bytes){};
    field.type.extension_metadata = (struct dora_bytes){};
    struct dora_schema storage = {};
    if (status == DORA_OK) {
        status = dora_schema_init_retain_field(&storage, state->storage.pool, &values, &field);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_parameters(node->extension,
                                             state->storage.pool,
                                             &storage,
                                             node_field,
                                             &node->extension_parameters,
                                             out_error);
    }
    dora_schema_release(&storage);
    return status;
}

static int builder_bookkeeping(struct builder_state* state, uint32_t count) {
    const size_t counts[] = {
        count,
        count,
        count,
        count,
        count,
        count,
        state->options.node_layout_count,
        state->options.statistics_request_count
    };
    const size_t widths[] = {
        sizeof(struct builder_node_state),
        BUILDER_NODE_RESERVED_BUFFER_CAPACITY * sizeof(struct builder_buffer_state),
        sizeof(struct builder_row_checkpoint) + sizeof(uint32_t),
        BUILDER_NODE_RESERVED_BUFFER_CAPACITY * (sizeof(struct builder_buffer_checkpoint) + 2 * sizeof(uint32_t)),
        sizeof(struct builder_column_run),
        sizeof(struct dora_node_topology) + sizeof(struct dora_vector) + sizeof(struct dora_view_buffer) +
            UNION_MEMBER_CAPACITY,
        sizeof(enum dora_layout),
        sizeof(struct dora_statistics_request)
    };
    const size_t alignments[] = {
        alignof(struct builder_node_state),
        alignof(struct builder_buffer_state),
        alignof(struct builder_row_checkpoint),
        alignof(struct builder_buffer_checkpoint),
        alignof(struct builder_column_run),
        alignof(struct dora_node_topology),
        alignof(enum dora_layout),
        alignof(struct dora_statistics_request)
    };
    static_assert(ARRAY_COUNT(widths) == ARRAY_COUNT(counts) && ARRAY_COUNT(alignments) == ARRAY_COUNT(counts));
    size_t offsets[ARRAY_COUNT(counts)];
    size_t bytes = 0;
    for (size_t segment = 0; segment < ARRAY_COUNT(counts); ++segment) {
        size_t size = 0;
        if (ckd_add(&bytes, bytes, alignments[segment] - 1) || ckd_mul(&size, counts[segment], widths[segment])) {
            return EOVERFLOW;
        }
        bytes &= ~(alignments[segment] - 1);
        offsets[segment] = bytes;
        if (ckd_add(&bytes, bytes, size)) {
            return EOVERFLOW;
        }
    }
    const int status = allocate_zero(state->storage.pool, bytes, 1, &state->bookkeeping);
    if (status != DORA_OK) {
        return status;
    }
    unsigned char* const data = state->bookkeeping.address;
    state->buffer_capacity = count * BUILDER_NODE_RESERVED_BUFFER_CAPACITY;
    state->nodes = (struct builder_node_state*)(data + offsets[0]);
    state->buffers = (struct builder_buffer_state*)(data + offsets[1]);
    state->mutation_nodes = (struct builder_row_checkpoint*)(data + offsets[2]);
    state->mutation_buffers = (struct builder_buffer_checkpoint*)(data + offsets[3]);
    state->runs.columns = (struct builder_column_run*)(data + offsets[4]);
    state->topology = (struct dora_node_topology*)(data + offsets[5]);
    if (state->options.node_layout_count != 0) {
        memcpy(data + offsets[6],
               state->options.node_layouts,
               state->options.node_layout_count * sizeof(enum dora_layout));
        state->options.node_layouts = (const enum dora_layout*)(data + offsets[6]);
    }
    if (state->options.statistics_request_count != 0) {
        memcpy(data + offsets[7],
               state->options.statistics_requests,
               state->options.statistics_request_count * sizeof(struct dora_statistics_request));
        state->options.statistics_requests = (const struct dora_statistics_request*)(data + offsets[7]);
    }
    return DORA_OK;
}

static int builder_schema(struct builder_state* state, const struct dora_schema* source, struct dora_error* out_error) {
    struct dora_allocation walk_storage = {};
    uint32_t count = 0;
    int status =
        dora_identity_claim_acquire(state->storage.pool, &source->handle, IDENTITY_KIND_SCHEMA, &state->schema.handle);
    if (status != DORA_OK) {
        return status;
    }
    state->schema.node_index = source->node_index;
    status = dora_schema_walk(state->storage.pool, &state->schema, &walk_storage, &count);
    if (status != DORA_OK) {
        return status;
    }
    if (count > UINT32_MAX / BUILDER_NODE_RESERVED_BUFFER_CAPACITY) {
        status = EOVERFLOW;
    } else if (count == 0 || (state->options.node_layout_count != 0 && state->options.node_layout_count != count)) {
        status = EINVAL;
    }
    bool changed_layouts = false;
    const struct schema_walk_node* const walk = walk_storage.address;
    for (uint32_t index = 0; status == DORA_OK && index < state->options.node_layout_count; ++index) {
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&walk[index].schema, &field);
        if (status == DORA_OK && field.layout != state->options.node_layouts[index]) {
            changed_layouts = true;
            break;
        }
    }
    if (status == DORA_OK) {
        status = builder_bookkeeping(state, count);
    }
    if (status == DORA_OK && changed_layouts) {
        struct dora_allocation schema_storage = {};
        struct dora_allocation child_storage = {};
        status = allocate_zero(state->storage.pool, count, sizeof(struct dora_schema), &schema_storage);
        if (status == DORA_OK) {
            status = allocate_zero(state->storage.pool, count, sizeof(struct dora_schema), &child_storage);
        }
        struct dora_schema* const schemas = schema_storage.address;
        struct dora_schema* const children = child_storage.address;
        if (status == DORA_OK && (schemas == nullptr || children == nullptr)) {
            status = EINVAL;
        }
        for (uint32_t cursor = count; status == DORA_OK && cursor != 0;) {
            const uint32_t node = --cursor;
            struct dora_schema_field field = {};
            status = dora_schema_get_field(&walk[node].schema, &field);
            if (status != DORA_OK) {
                break;
            }
            field.layout = state->options.node_layouts[node];
            for (uint32_t child = node + 1; child < count; ++child) {
                if (walk[child].parent_node_index == node) {
                    children[walk[child].child_index] = schemas[child];
                }
            }
            status = dora_schema_init_retain_node(&schemas[node], state->storage.pool, &field, children);
        }
        if (status == DORA_OK) {
            dora_schema_release(&state->schema);
            status = dora_identity_claim_acquire(state->storage.pool,
                                                 &schemas[0].handle,
                                                 IDENTITY_KIND_SCHEMA,
                                                 &state->schema.handle);
            state->schema.node_index = schemas[0].node_index;
        }
        for (uint32_t index = 0; schemas != nullptr && index < count; ++index) {
            dora_schema_release(&schemas[index]);
        }
        dora_allocation_release(&child_storage);
        dora_allocation_release(&schema_storage);
        dora_allocation_release(&walk_storage);
        if (status == DORA_OK) {
            status = dora_schema_walk(state->storage.pool, &state->schema, &walk_storage, &count);
        }
    }
    if (status == DORA_OK) {
        state->node_count = count;
    }
    const struct schema_walk_node* const nodes = walk_storage.address;
    struct builder_node_state* const states = state->nodes;
    struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t index = 0; status == DORA_OK && index < state->node_count; ++index) {
        struct builder_node_state* const node = &states[index];
        node->schema = nodes[index].schema;
        node->first_buffer = state->buffer_count;
        struct dora_schema_field field = {};
        const struct schema_image* image = nullptr;
        status = dora_schema_resolve_image(&node->schema, &image);
        if (status == DORA_OK) {
            node->field = &image->nodes[node->schema.node_index].field;
            field = *node->field;
            node->rows.content_state =
                field.type.id == DORA_TYPE_UTF8 ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_VALIDATED;
            status = node_buffers(node, &field, &buffers[state->buffer_count]);
        }
        state->buffer_count += node->buffer_count;
        node->frontier_buffer = UINT32_MAX;
        if (status == DORA_OK &&
            (field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW)) {
            node->frontier_buffer = state->buffer_count++;
            buffers[node->frontier_buffer] =
                (struct builder_buffer_state){.unit_bit_width = 64, .role = BUILDER_BUFFER_CHILD_FRONTIERS};
        }
        if (status == DORA_OK) {
            status = builder_bind(state, node, &field, out_error);
        }
        state->has_validation_nodes = state->has_validation_nodes ||
                                      (field.type.id == DORA_TYPE_MAP &&
                                       (field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LARGE_LIST)) ||
                                      (node->extension != nullptr && node->extension->validate != nullptr);
        node->is_run_eligible = ((field.layout == DORA_LAYOUT_FIXED || field.layout == DORA_LAYOUT_BITS) &&
                                 (node->extension == nullptr || node->extension->validate == nullptr)) ||
                                ((field.layout == DORA_LAYOUT_BINARY || field.layout == DORA_LAYOUT_LARGE_BINARY ||
                                  field.layout == DORA_LAYOUT_BINARY_VIEW) &&
                                 (field.type.id == DORA_TYPE_UTF8 || field.type.id == DORA_TYPE_BINARY) &&
                                 (node->extension == nullptr || node->extension->validate == nullptr));
    }
    if (status == DORA_OK) {
        struct dora_node_topology* const inputs = state->topology;
        struct dora_vector* const domains = (struct dora_vector*)(inputs + state->node_count);
        struct dora_view_buffer* const data = (struct dora_view_buffer*)(domains + state->node_count);
        uint8_t* const members = (uint8_t*)(data + state->node_count);
        for (uint32_t index = 0; status == DORA_OK && index < state->node_count; ++index) {
            struct dora_schema_field field = {};
            status = dora_schema_get_field(&states[index].schema, &field);
            if (status != DORA_OK) {
                break;
            }
            inputs[index] = (struct dora_node_topology){
                .parent_node_index = nodes[index].parent_node_index,
                .child_index = nodes[index].child_index
            };
            struct dora_vector_view* const view = &inputs[index].view;
            *view = (struct dora_vector_view){
                .type = field.type.id,
                .layout = field.layout,
                .list_element_count =
                    field.type.id == DORA_TYPE_FIXED_LIST ? field.type.parameters.list_element_count : 0
            };
            const struct dora_logical_type physical =
                field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END
                    ? (struct dora_logical_type){.id = field.index_type}
                    : field.type;
            status = type_physical(&physical, &view->physical_type);
            if (field.layout == DORA_LAYOUT_RUN_END) {
                view->byte_width = view->physical_type == DORA_PHYSICAL_TYPE_I16   ? 2
                                   : view->physical_type == DORA_PHYSICAL_TYPE_I32 ? 4
                                                                                   : 8;
            }
            states[index].comparison_flags =
                field.type.extension_name.byte_count == 0
                    ? DORA_EXTENSION_FLAG_STORAGE_EQUALITY | DORA_EXTENSION_FLAG_STORAGE_ORDER
                : states[index].extension == nullptr ? 0
                                                     : states[index].extension->flags;
            for (uint32_t part = 0; status == DORA_OK && part < states[index].buffer_count; ++part) {
                const struct builder_buffer_state* const buffer = &buffers[states[index].first_buffer + part];
                if (buffer->role == BUILDER_BUFFER_FIXED_VALUES) {
                    view->byte_width = buffer->unit_bit_width / 8;
                } else if (buffer->role == BUILDER_BUFFER_BYTES) {
                    view->data_buffers = &data[index];
                    view->data_buffer_count = 1;
                }
            }
            if (field.type.id == DORA_TYPE_UNION) {
                uint8_t* const map = members + (size_t)index * UNION_MEMBER_CAPACITY;
                memset(map, UINT8_MAX, UNION_MEMBER_CAPACITY);
                for (uint32_t child = 0; child < field.child_count; ++child) {
                    map[(uint8_t)field.type.parameters.type_codes[child]] = (uint8_t)child;
                }
                view->union_members = map;
            }
        }
        if (status == DORA_OK) {
            status = dora_node_topology_link(inputs, state->node_count);
            for (uint32_t cursor = state->node_count; status == DORA_OK && cursor != 0;) {
                const uint32_t index = --cursor;
                if (inputs[index].parent_node_index != UINT32_MAX) {
                    states[inputs[index].parent_node_index].comparison_flags &= states[index].comparison_flags;
                }
            }
        }
    }
    dora_allocation_release(&walk_storage);
    return status;
}

static void builder_reset_rows(struct builder_state* state, struct statistics_accumulator* statistics) {
    struct builder_node_state* const nodes = state->nodes;
    struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t index = 0; index < state->node_count; ++index) {
        nodes[index].rows = (struct builder_row_state){
            .content_state = nodes[index].field->type.id == DORA_TYPE_UTF8 ? VECTOR_CONTENT_VALIDATED_ASCII
                                                                           : VECTOR_CONTENT_VALIDATED
        };
    }
    // Bit writes merge into shared words and require zeroed storage.
    state->capacity_bytes = 0;
    for (uint32_t index = 0; index < state->buffer_count; ++index) {
        state->capacity_bytes += buffers[index].allocation.capacity_bytes;
        buffers[index].initialized_unit_count = 0;
        if (buffers[index].allocation.address != nullptr && buffers[index].unit_bit_width == 1) {
            bitmap_fill_bytes(buffers[index].allocation.address, buffers[index].allocation.capacity_bytes, false);
        }
    }
    state->written_byte_count = 0;
    for (uint32_t index = 0; index < state->node_count; ++index) {
        const struct builder_node_state* const node = &nodes[index];
        for (uint32_t buffer = 0; buffer < node->buffer_count; ++buffer) {
            struct builder_buffer_state* const offsets = &buffers[node->first_buffer + buffer];
            if (offsets->role != BUILDER_BUFFER_OFFSETS32 && offsets->role != BUILDER_BUFFER_OFFSETS64) {
                continue;
            }
            const bool has_sizes = buffer + 1 < node->buffer_count &&
                                   (buffers[node->first_buffer + buffer + 1].role == BUILDER_BUFFER_SIZES32 ||
                                    buffers[node->first_buffer + buffer + 1].role == BUILDER_BUFFER_SIZES64);
            offsets->initialized_unit_count = !has_sizes && offsets->allocation.address != nullptr ? 1 : 0;
            state->written_byte_count += offsets->initialized_unit_count * (offsets->unit_bit_width / 8);
        }
    }
    state->completed_root_row_count = 0;
    dora_statistics_accumulator_release(&state->statistics);
    state->statistics = statistics;
}

//--------------------------------------
// State Resolution
//--------------------------------------

int dora_builder_resolve_state(const struct dora_builder* builder, struct builder_state** out_state) {
    const struct identity_slot* slot = nullptr;
    if (builder == nullptr || out_state == nullptr ||
        dora_identity_resolve(&builder->handle, IDENTITY_KIND_BUILDER, &slot) != DORA_OK) {
        return EINVAL;
    }
    struct builder_state* const state = slot->claims->payload;
    if (builder->node_index >= state->node_count) {
        return EINVAL;
    }
    if (slot->is_running) {
        return EBUSY;
    }
    *out_state = state;
    return DORA_OK;
}

int dora_builder_operation_capture(const struct dora_builder* receiver, struct builder_operation* out_operation) {
    if (receiver == nullptr || out_operation == nullptr) {
        return EINVAL;
    }
    *out_operation = (struct builder_operation){.data.target = *receiver};
    return DORA_OK;
}

void dora_builder_operation_data_move(struct builder_operation_data* source, struct builder_operation_data* out_data) {
    *out_data = *source;
    *source = (struct builder_operation_data){};
}

void dora_builder_operation_reset_checkpoints(struct builder_operation* operation) {
    operation->data.written_byte_count = operation->state->written_byte_count;
    operation->data.statistics = (struct statistics_checkpoint){};
    operation->data.touched_node_count = 0;
    operation->data.touched_buffer_count = 0;
    operation->data.bound_buffer_count = 0;
}

int dora_builder_operation_resume(struct builder_operation_data* parked,
                                  enum builder_phase expected_phase,
                                  struct builder_operation* out_active) {
    struct builder_state* state = nullptr;
    int status = dora_builder_resolve_state(&parked->target, &state);
    if (status == DORA_OK && state->phase != expected_phase) {
        status = EBUSY;
    }
    struct identity_slot* slot = nullptr;
    if (status == DORA_OK) {
        status = dora_identity_slot_enter(&parked->target.handle, IDENTITY_KIND_BUILDER, &slot);
    }
    if (status == DORA_OK) {
        dora_builder_operation_data_move(parked, &out_active->data);
        out_active->state = state;
        out_active->slot = slot;
    }
    return status;
}

bool dora_builder_operation_addresses(const struct builder_operation* operation, const struct dora_builder* receiver) {
    return receiver != nullptr && dora_handle_is_same(receiver->handle, operation->data.target.handle) &&
           receiver->node_index == operation->data.target.node_index;
}

//------------------------------------------------------------------------------
//
// Completion and Lifecycle
//
//------------------------------------------------------------------------------

//--------------------------------------
// Lifecycle
//--------------------------------------

int dora_builder_move(struct dora_builder* in_out_source, struct dora_builder* out_destination) {
    struct builder_state* state = nullptr;
    const int status = dora_builder_resolve_state(in_out_source, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase == BUILDER_PHASE_PREPARED) {
        return EBUSY;
    }
    if (!in_out_source->handle.is_owner || in_out_source->node_index != 0 || !builder_is_empty(out_destination)) {
        return EINVAL;
    }
    *out_destination = *in_out_source;
    *in_out_source = (struct dora_builder){};
    return DORA_OK;
}

void dora_builder_release(struct dora_builder* builder) {
    if (builder == nullptr) {
        return;
    }
    struct dora_handle owner = builder->handle;
    *builder = (struct dora_builder){};
    const struct identity_slot* slot = nullptr;
    if (owner.is_owner && dora_identity_resolve(&owner, IDENTITY_KIND_BUILDER, &slot) == DORA_OK && slot->is_running) {
        struct builder_state* const state = slot->claims->payload;
        state->release_owner = owner;
        return;
    }
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_BUILDER, &retired);
    builder_discard(retired);
}

bool dora_builder_is_valid(const struct dora_builder* builder) {
    const struct identity_slot* slot = nullptr;
    if (builder == nullptr || dora_identity_resolve(&builder->handle, IDENTITY_KIND_BUILDER, &slot) != DORA_OK) {
        return false;
    }
    const struct builder_state* const state = slot->claims->payload;
    return builder->node_index < state->node_count;
}

//--------------------------------------
// Completion
//--------------------------------------

static int operation_leave_captured(struct identity_slot* slot,
                                    struct dora_builder target,
                                    int status,
                                    struct builder_state** out_live_state) {
    struct builder_state* const state = slot->claims->payload;
    const struct dora_handle owner = state->release_owner;
    dora_identity_slot_leave(&target.handle, slot);
    if (owner.slot_index != 0) {
        struct dora_builder builder = {.handle = owner};
        dora_builder_release(&builder);
    }
    return status == DORA_OK && out_live_state != nullptr ? dora_builder_resolve_state(&target, out_live_state)
                                                          : status;
}

int dora_builder_operation_leave(struct builder_operation* active, int status, struct builder_state** out_live_state) {
    const struct dora_builder target = active->data.target;
    struct identity_slot* const slot = active->slot;
    *active = (struct builder_operation){};
    return operation_leave_captured(slot, target, status, out_live_state);
}

int dora_builder_operation_park(struct builder_operation* active,
                                struct builder_operation_data* out_parked,
                                int status,
                                struct builder_state** out_live_state) {
    const struct dora_builder target = active->data.target;
    struct identity_slot* const slot = active->slot;
    dora_builder_operation_data_move(&active->data, out_parked);
    *active = (struct builder_operation){};
    return operation_leave_captured(slot, target, status, out_live_state);
}

int dora_builder_operation_enter(struct builder_operation* operation, enum builder_entry entry) {
    int status = dora_builder_resolve_state(&operation->data.target, &operation->state);
    if (status == DORA_OK && entry != BUILDER_ENTRY_PREPARED_ACCESS && entry != BUILDER_ENTRY_INSPECT &&
        operation->state->runs.operation.target.handle.slot_index != 0) {
        status = dora_builder_runs_end(operation->state, entry != BUILDER_ENTRY_CLEAR);
        if (status == DORA_OK) {
            status = dora_builder_resolve_state(&operation->data.target, &operation->state);
        }
    }
    const bool reclaims = entry == BUILDER_ENTRY_RECLAIM || entry == BUILDER_ENTRY_CLEAR;
    if (status == DORA_OK && entry != BUILDER_ENTRY_ANY_PHASE && entry != BUILDER_ENTRY_INSPECT &&
        entry != BUILDER_ENTRY_PREPARED_ACCESS && operation->state->phase != BUILDER_PHASE_OPEN &&
        (!reclaims || operation->state->phase != BUILDER_PHASE_SHARED)) {
        status = EBUSY;
    }
    struct identity_slot* entered_slot = nullptr;
    if (status == DORA_OK) {
        status = dora_identity_slot_enter(&operation->data.target.handle, IDENTITY_KIND_BUILDER, &entered_slot);
    }
    if (status != DORA_OK) {
        operation->state = nullptr;
        return status;
    }
    operation->slot = entered_slot;
    struct builder_state* const state = operation->state;
    if (entry == BUILDER_ENTRY_PREPARED_ACCESS) {
        const struct builder_operation_data* const window = &state->window.operation;
        if (state->phase != BUILDER_PHASE_PREPARED || window->target.node_index != operation->data.target.node_index) {
            return dora_builder_operation_leave(operation, EINVAL, nullptr);
        }
        struct identity_slot* const slot = operation->slot;
        *operation = (struct builder_operation){};
        dora_builder_operation_data_move(&state->window.operation, &operation->data);
        operation->state = state;
        operation->slot = slot;
        return DORA_OK;
    }
    if (reclaims && state->phase == BUILDER_PHASE_SHARED) {
        status = builder_reclaim(state);
        if (status != DORA_OK) {
            return dora_builder_operation_leave(operation, status, nullptr);
        }
    }
    dora_builder_operation_reset_checkpoints(operation);
    return DORA_OK;
}

int dora_builder_operation_begin(const struct dora_builder* receiver,
                                 enum builder_entry entry,
                                 struct builder_operation* out_operation) {
    const int status = dora_builder_operation_capture(receiver, out_operation);
    return status == DORA_OK ? dora_builder_operation_enter(out_operation, entry) : status;
}

static uint32_t finish_previous(const struct builder_state* state, uint32_t number) {
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_node_topology* const inputs = state->topology;
    if (number == 0) {
        return UINT32_MAX;
    }
    if (number == UINT32_MAX) {
        number = 0;
    } else {
        const uint32_t parent = inputs[number].parent_node_index;
        if (inputs[parent].first_child_index == number) {
            return parent;
        }
        --number;
        while (inputs[number].parent_node_index != parent) {
            number = inputs[number].parent_node_index;
        }
    }
    while (nodes[number].dictionary.handle.slot_index == 0 && inputs[number].first_child_index != UINT32_MAX) {
        number = inputs[number].first_child_index;
        while (inputs[number].next_sibling_index != UINT32_MAX) {
            number = inputs[number].next_sibling_index;
        }
    }
    return number;
}

static int builder_check_complete(const struct builder_state* state) {
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_node_topology* const inputs = state->topology;
    const struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t index = 0; index < state->node_count; ++index) {
        const struct builder_node_state* const node = &nodes[index];
        struct dora_schema_field field = {};
        int status = dora_schema_get_field(&node->schema, &field);
        if (status != DORA_OK) {
            return status;
        }
        uint64_t rows = node->rows.row_count;
        if (index == 0 && field.layout == DORA_LAYOUT_STRUCT && inputs[index].first_child_index != UINT32_MAX) {
            rows = nodes[inputs[index].first_child_index].rows.row_count;
        }
        if (field.layout == DORA_LAYOUT_STRUCT || field.layout == DORA_LAYOUT_SPARSE_UNION) {
            for (uint32_t child = inputs[index].first_child_index; child != UINT32_MAX;
                 child = inputs[child].next_sibling_index) {
                if (nodes[child].rows.row_count != rows) {
                    return EINVAL;
                }
            }
        } else if (field.layout == DORA_LAYOUT_DENSE_UNION || field.layout == DORA_LAYOUT_LIST_VIEW ||
                   field.layout == DORA_LAYOUT_LARGE_LIST_VIEW) {
            for (uint32_t child = inputs[index].first_child_index; child != UINT32_MAX;
                 child = inputs[child].next_sibling_index) {
                if (nodes[child].rows.row_count != nodes[child].rows.closed_row_count) {
                    return EINVAL;
                }
            }
        } else if (field.layout == DORA_LAYOUT_FIXED_LIST) {
            uint64_t length = 0;
            if (ckd_mul(&length, rows, field.type.parameters.list_element_count) ||
                length != nodes[inputs[index].first_child_index].rows.row_count) {
                return EINVAL;
            }
        } else if (field.layout == DORA_LAYOUT_RUN_END) {
            const uint32_t ends = inputs[index].first_child_index;
            if (nodes[ends].rows.row_count != nodes[inputs[ends].next_sibling_index].rows.row_count) {
                return EINVAL;
            }
        } else if (field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LARGE_LIST) {
            const struct dora_allocation offsets = buffers[node->first_buffer + 1].allocation;
            const size_t width = field.layout == DORA_LAYOUT_LIST ? sizeof(int32_t) : sizeof(int64_t);
            size_t endpoint = 0;
            int64_t end = 0;
            if (ckd_mul(&endpoint, rows, width) || ckd_add(&endpoint, endpoint, width)) {
                return EOVERFLOW;
            }
            if (offsets.address != nullptr && endpoint <= offsets.capacity_bytes) {
                if (width == sizeof(int32_t)) {
                    int32_t narrow = 0;
                    memcpy(&narrow, (const uint8_t*)offsets.address + endpoint - width, width);
                    end = narrow;
                } else {
                    memcpy(&end, (const uint8_t*)offsets.address + endpoint - width, width);
                }
            } else if (rows != 0) {
                return EINVAL;
            }
            for (uint32_t child = inputs[index].first_child_index; child != UINT32_MAX;
                 child = inputs[child].next_sibling_index) {
                if (end < 0 || (uint64_t)end != nodes[child].rows.row_count) {
                    return EINVAL;
                }
            }
        }
    }
    return DORA_OK;
}

static int builder_publish(struct builder_operation* operation,
                           struct dora_vector* out_nodes,
                           bool is_shared,
                           struct dora_error* out_error) {
    struct builder_state* const state = operation->state;
    int status = builder_check_complete(state);
    struct dora_vector statistics = {};
    struct statistics_checkpoint statistics_checkpoint = {};
    bool statistics_active = false;
    if (status == DORA_OK && state->statistics != nullptr) {
        status = dora_builder_statistics_checkpoint(operation, &statistics_checkpoint);
        statistics_active = status == DORA_OK;
        if (status == DORA_OK) {
            status = dora_statistics_accumulator_finish(state->statistics, &statistics, out_error);
        }
    }
    struct dora_allocation scratch = {};
    size_t image_bytes = 0;
    size_t child_bytes = 0;
    size_t offset_bytes = 0;
    size_t bytes = 0;
    if (status == DORA_OK && (ckd_mul(&image_bytes, state->node_count, sizeof(struct vector_image*)) ||
                              ckd_mul(&child_bytes, state->node_count, sizeof(struct dora_vector)) ||
                              ckd_mul(&offset_bytes, state->buffer_count, sizeof(struct dora_allocation)) ||
                              ckd_add(&bytes, image_bytes, child_bytes) || ckd_add(&bytes, bytes, offset_bytes) ||
                              ckd_add(&bytes, bytes, offset_bytes))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = allocate_zero(state->storage.pool, bytes, 1, &scratch);
    }
    struct dora_vector* const vectors = out_nodes;
    struct dora_vector* const children = (struct dora_vector*)scratch.address;
    struct dora_allocation* const empty_offsets =
        scratch.address == nullptr ? nullptr : (struct dora_allocation*)((unsigned char*)scratch.address + child_bytes);
    struct vector_image** const images =
        scratch.address == nullptr
            ? nullptr
            : (struct vector_image**)((unsigned char*)scratch.address + child_bytes + offset_bytes + offset_bytes);
    void** const addresses = (void**)state->shared_addresses.address;
    struct builder_node_state* const nodes = state->nodes;
    const struct dora_node_topology* const inputs = state->topology;
    struct builder_buffer_state* const buffers = state->buffers;
    if (status == DORA_OK && (vectors == nullptr || images == nullptr)) {
        status = EINVAL;
    }
    struct dora_allocation* const allocations =
        empty_offsets == nullptr ? nullptr : empty_offsets + state->buffer_count;
    for (uint32_t index = status == DORA_OK ? finish_previous(state, UINT32_MAX) : UINT32_MAX;
         status == DORA_OK && index != UINT32_MAX;
         index = finish_previous(state, index)) {
        const struct builder_node_state* const node = &nodes[index];
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&node->schema, &field);
        if (status != DORA_OK) {
            break;
        }
        uint64_t rows = node->rows.row_count;
        for (uint32_t child = inputs[index].first_child_index; child != UINT32_MAX;
             child = inputs[child].next_sibling_index) {
            children[inputs[child].child_index] = vectors[child];
        }
        if (field.layout == DORA_LAYOUT_DICTIONARY && node->dictionary.handle.slot_index != 0) {
            children[0] = node->dictionary;
        }
        if (index == 0 && field.layout == DORA_LAYOUT_STRUCT && field.child_count != 0) {
            rows = children[0].rows.length;
        }
        const uint32_t count = builder_node_buffer_count(node);
        for (uint32_t buffer = 0, number = node->first_buffer; status == DORA_OK && buffer < count;
             ++buffer, number = builder_next_buffer(state, node, number)) {
            allocations[buffer] = buffers[number].allocation;
            if (field.layout == DORA_LAYOUT_BINARY_VIEW && buffers[number].role == BUILDER_BUFFER_BYTES &&
                buffers[node->first_buffer + 3].allocation.address != nullptr) {
                ((int64_t*)buffers[node->first_buffer + 3].allocation.address)[buffer - 2] =
                    (int64_t)buffers[number].initialized_unit_count;
            }
            if (buffers[number].role == BUILDER_BUFFER_VALIDITY_BITS && node->rows.null_count == 0) {
                allocations[buffer] = (struct dora_allocation){};
            }
            const bool offsets =
                buffers[number].role == BUILDER_BUFFER_OFFSETS32 || buffers[number].role == BUILDER_BUFFER_OFFSETS64;
            if (rows == 0 && offsets && allocations[buffer].address == nullptr &&
                field.layout != DORA_LAYOUT_LIST_VIEW && field.layout != DORA_LAYOUT_LARGE_LIST_VIEW) {
                status =
                    allocate_zero(state->storage.pool, 1, buffers[number].unit_bit_width / 8, &empty_offsets[number]);
                if (status == DORA_OK) {
                    allocations[buffer] = empty_offsets[number];
                }
            }
        }
        if (status != DORA_OK) {
            break;
        }
        struct dora_allocation activity_storage = {};
        status = dora_builder_activity(operation, index, &activity_storage);
        const bool masked_required =
            node->rows.has_masked_nulls && (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0;
        const void* activity_data = masked_required ? activity_storage.address : nullptr;
        if (masked_required && node->buffer_count != 0 &&
            buffers[node->first_buffer].role == BUILDER_BUFFER_VALIDITY_BITS) {
            activity_data = buffers[node->first_buffer].allocation.address;
        }
        const struct dora_bytes activity = {.data = activity_data, .byte_count = bitmap_bytes_for_bits(rows)};
        const struct vector_input input = {
            .field = &field,
            .admitted_schema = &node->schema,
            .content_state = (node->extension != nullptr && node->extension->validate != nullptr) ||
                                     field.type.extension_name.byte_count == 0
                                 ? node->rows.content_state
                                 : VECTOR_CONTENT_PROMISED,
            .row_count = rows,
            .null_count = node->rows.null_count,
            .allocations = allocations,
            .buffer_count = count,
            .children = children,
            .child_count = field.child_count,
            .ancestor_activity = activity.data == nullptr ? nullptr : &activity,
            .are_children_folded = field.layout == DORA_LAYOUT_STRUCT
        };
        if (status == DORA_OK) {
            status = dora_vector_prepare(state->storage.pool, &input, &images[index], out_error);
        }
        dora_allocation_release(&activity_storage);
        if (status == DORA_OK && index == 0 && statistics.handle.slot_index != 0) {
            status = dora_vector_set_statistics_claim(images[index], &statistics, VECTOR_STATISTICS_COMPUTED);
        }
        if (status == DORA_OK) {
            status = dora_vector_publish(images[index], &vectors[index]);
        }
        if (status != DORA_OK) {
            dora_vector_discard(images[index]);
            images[index] = nullptr;
        }
    }
    if (status == DORA_OK && empty_offsets != nullptr) {
        for (uint32_t index = 0; status == DORA_OK && index < state->node_count; ++index) {
            if (images[index] == nullptr) {
                continue;
            }
            struct vector_buffer_row* const adopted = dora_vector_image_buffers(images[index]->buffer_image);
            for (uint32_t buffer = 0, number = nodes[index].first_buffer;
                 buffer < builder_node_buffer_count(&nodes[index]);
                 ++buffer, number = builder_next_buffer(state, &nodes[index], number)) {
                if (buffers[number].role == BUILDER_BUFFER_VALIDITY_BITS && nodes[index].rows.null_count == 0) {
                    buffers[number].next_capacity_bytes = 0;
                    dora_allocation_release(&buffers[number].allocation);
                }
                if (empty_offsets[number].address != nullptr) {
                    dora_allocation_release(&buffers[number].allocation);
                }
                struct dora_allocation* const source =
                    empty_offsets[number].address != nullptr ? &empty_offsets[number] : &buffers[number].allocation;
                if (is_shared && addresses != nullptr) {
                    addresses[number] = source == &buffers[number].allocation ? source->address : nullptr;
                }
                const size_t capacity = source->capacity_bytes;
                if (dora_allocation_adopt(source, &adopted[buffer].owner.allocation) != DORA_OK) {
                    status = EINVAL;
                    break;
                }
                if (source == &buffers[number].allocation && buffers[number].initialized_unit_count != 0 &&
                    buffers[number].role != BUILDER_BUFFER_BYTES) {
                    buffers[number].next_capacity_bytes = capacity;
                }
            }
        }
    }
    for (uint32_t index = 0; status != DORA_OK && vectors != nullptr && index < state->node_count; ++index) {
        dora_vector_release(&vectors[index]);
    }
    for (uint32_t index = 0; empty_offsets != nullptr && index < state->buffer_count; ++index) {
        dora_allocation_release(&empty_offsets[index]);
    }
    dora_vector_release(&statistics);
    if (statistics_active) {
        if (status == DORA_OK && !is_shared) {
            dora_statistics_checkpoint_commit(state->statistics, &statistics_checkpoint);
        } else {
            dora_statistics_checkpoint_rollback(state->statistics, &statistics_checkpoint);
        }
    }
    dora_allocation_release(&scratch);
    return status;
}

// Releases staged on failure or callback-filled destination; preserves the original error.
static int builder_publish_vector(struct dora_vector* destination, struct dora_vector* staged, int status) {
    if (status == DORA_OK) {
        status = dora_vector_move(staged, destination);
    }
    dora_vector_release(staged);
    return status;
}

static int builder_finish(struct dora_builder* in_out_builder,
                          struct dora_vector* out_vector,
                          struct dora_error* out_error,
                          bool keep_metadata) {
    struct builder_operation operation = {};
    int status = dora_builder_operation_capture(in_out_builder, &operation);
    if (status != DORA_OK) {
        return status;
    }
    // Argument failure must preserve shared storage.
    if (!operation.data.target.handle.is_owner || operation.data.target.node_index != 0 ||
        !dora_vector_is_empty(out_vector)) {
        return EINVAL;
    }
    status = dora_builder_operation_enter(&operation, BUILDER_ENTRY_RECLAIM);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = operation.state;
    // Accumulator failure must leave rows unpublished.
    struct statistics_accumulator* statistics = nullptr;
    if (keep_metadata && state->statistics != nullptr) {
        status = dora_statistics_accumulator_open(state->storage.pool,
                                                  &state->schema,
                                                  state->options.statistics_requests,
                                                  state->options.statistics_request_count,
                                                  &statistics);
    }
    struct dora_allocation vector_storage = {};
    if (status == DORA_OK) {
        status = allocate_zero(state->storage.pool, state->node_count, sizeof(struct dora_vector), &vector_storage);
    }
    struct dora_vector* const vectors = vector_storage.address;
    if (status == DORA_OK) {
        status = vectors == nullptr ? EINVAL : builder_publish(&operation, vectors, false, out_error);
    }
    struct dora_vector staged = {};
    if (status == DORA_OK) {
        staged = vectors[0];
        vectors[0] = (struct dora_vector){};
        if (keep_metadata) {
            builder_reset_rows(state, statistics);
            statistics = nullptr;
        } else {
            state->release_owner = operation.data.target.handle;
            if (dora_builder_operation_addresses(&operation, in_out_builder)) {
                *in_out_builder = (struct dora_builder){};
            }
        }
    }
    for (uint32_t index = 0; vectors != nullptr && index < state->node_count; ++index) {
        dora_vector_release(&vectors[index]);
    }
    dora_allocation_release(&vector_storage);
    dora_statistics_accumulator_release(&statistics);
    status = dora_builder_operation_leave(&operation, status, nullptr);
    return builder_publish_vector(out_vector, &staged, status);
}

int dora_builder_finish_error(struct dora_builder* in_out_builder,
                              struct dora_vector* out_vector,
                              struct dora_error* out_error) {
    return builder_finish(in_out_builder, out_vector, out_error, false);
}

int dora_builder_finish_batch(struct dora_builder* builder, struct dora_vector* out_vector) {
    return builder_finish(builder, out_vector, nullptr, true);
}

int dora_builder_share(struct dora_builder* builder, struct dora_vector* out_vector) {
    struct builder_operation operation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &operation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = operation.state;
    if (!operation.data.target.handle.is_owner || operation.data.target.node_index != 0 ||
        !dora_vector_is_empty(out_vector)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status =
            allocate_zero(state->storage.pool, state->node_count, sizeof(struct dora_vector), &state->shared_vectors);
    }
    if (status == DORA_OK) {
        status = allocate_zero(state->storage.pool, state->buffer_count, sizeof(void*), &state->shared_addresses);
    }
    if (status == DORA_OK) {
        status = builder_publish(&operation, state->shared_vectors.address, true, nullptr);
    }
    struct dora_vector staged = {};
    if (status == DORA_OK) {
        status = dora_vector_retain(&((struct dora_vector*)state->shared_vectors.address)[0], &staged);
    }
    // Callback-filled destinations must be rejected before storage becomes unreclaimable.
    if (status == DORA_OK && !dora_vector_is_empty(out_vector)) {
        dora_vector_release(&staged);
        status = EINVAL;
    }
    if (status == DORA_OK) {
        state->phase = BUILDER_PHASE_SHARED;
    } else if (state->shared_vectors.address == nullptr || builder_reclaim(state) != DORA_OK) {
        struct dora_vector* const shared = state->shared_vectors.address;
        for (uint32_t index = 0; shared != nullptr && index < state->node_count; ++index) {
            dora_vector_release(&shared[index]);
        }
        dora_allocation_release(&state->shared_vectors);
        dora_allocation_release(&state->shared_addresses);
    }
    status = dora_builder_operation_leave(&operation, status, nullptr);
    return builder_publish_vector(out_vector, &staged, status);
}

int dora_builder_finish(struct dora_builder* in_out_builder, struct dora_vector* out_vector) {
    return dora_builder_finish_error(in_out_builder, out_vector, nullptr);
}

int dora_builder_clear(struct dora_builder* builder) {
    struct builder_operation operation = {};
    int status = dora_builder_operation_capture(builder, &operation);
    if (status != DORA_OK) {
        return status;
    }
    // Rejected child clears must preserve shared storage.
    if (operation.data.target.node_index != 0) {
        return EINVAL;
    }
    status = dora_builder_operation_enter(&operation, BUILDER_ENTRY_CLEAR);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = operation.state;
    struct statistics_accumulator* statistics = nullptr;
    if (state->statistics != nullptr) {
        status = dora_statistics_accumulator_open(state->storage.pool,
                                                  &state->schema,
                                                  state->options.statistics_requests,
                                                  state->options.statistics_request_count,
                                                  &statistics);
    }
    if (status == DORA_OK) {
        builder_reset_rows(state, statistics);
    }
    return dora_builder_operation_leave(&operation, status, nullptr);
}

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

static int builder_construct_schema(struct dora_pool pool,
                                    const struct dora_schema* schema,
                                    const struct dora_builder_options* options,
                                    struct dora_builder* out_staged,
                                    struct dora_error* out_error) {
    if (!dora_schema_is_valid(schema) ||
        (options != nullptr && ((options->content_policy == DORA_FAILURE_POLICY_WRAP ||
                                 options->content_policy > DORA_FAILURE_POLICY_TRUST) ||
                                (options->node_layout_count != 0 && options->node_layouts == nullptr) ||
                                (options->statistics_request_count != 0 && options->statistics_requests == nullptr)))) {
        return EINVAL;
    }
    struct dora_allocation storage = {};
    int status = dora_memory_payload_allocate(pool, sizeof(struct builder_state), &storage);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = storage.address;
    *state = (struct builder_state){.storage = storage};
    state->claims.payload = state;
    if (options != nullptr) {
        state->options = *options;
    }
    status = builder_schema(state, schema, out_error);
    if (status == DORA_OK && state->options.statistics_request_count != 0) {
        status = dora_statistics_accumulator_open(pool,
                                                  &state->schema,
                                                  state->options.statistics_requests,
                                                  state->options.statistics_request_count,
                                                  &state->statistics);
    }
    if (status == DORA_OK) {
        struct builder_node_state* const nodes = state->nodes;
        for (uint32_t index = 0; status == DORA_OK && index < state->node_count; ++index) {
            struct dora_allocation adopted = {};
            status = dora_allocation_adopt(&nodes[index].extension_parameters, &adopted);
            nodes[index].extension_parameters = adopted;
        }
    }
    struct dora_handle owner = {};
    if (status == DORA_OK) {
        status = dora_identity_slot_publish(pool, IDENTITY_KIND_BUILDER, &state->claims, &owner);
    }
    if (status == DORA_OK) {
        *out_staged = (struct dora_builder){.handle = owner};
    }
    if (status != DORA_OK) {
        builder_discard(state);
    }
    return status;
}

// Rechecks destination and staged after callbacks; releases rejected staged results.
static int builder_publish_staged(struct dora_builder* destination, struct dora_builder* staged, int status) {
    struct builder_state* state = nullptr;
    if (status == DORA_OK &&
        (!builder_is_empty(destination) || dora_builder_resolve_state(staged, &state) != DORA_OK)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        *destination = *staged;
    } else {
        dora_builder_release(staged);
    }
    return status;
}

int dora_builder_init_type(struct dora_builder* builder,
                           struct dora_pool pool,
                           const struct dora_logical_type* type,
                           const struct dora_builder_options* options) {
    if (!builder_is_empty(builder) || type == nullptr) {
        return EINVAL;
    }
    struct dora_schema schema = {};
    struct dora_builder staged = {};
    const struct dora_schema_field field = {.type = *type};
    int status = dora_schema_init_retain_node(&schema, pool, &field, nullptr);
    if (status == DORA_OK) {
        status = builder_construct_schema(pool, &schema, options, &staged, nullptr);
    }
    dora_schema_release(&schema);
    return builder_publish_staged(builder, &staged, status);
}

int dora_builder_init_fields_error(struct dora_builder* builder,
                                   struct dora_pool pool,
                                   const struct dora_schema_input* input,
                                   const struct dora_builder_options* options,
                                   struct dora_error* out_error) {
    if (!builder_is_empty(builder)) {
        return EINVAL;
    }
    struct dora_schema schema = {};
    struct dora_builder staged = {};
    int status = dora_schema_init_copy(&schema, pool, input);
    if (status == DORA_OK) {
        status = builder_construct_schema(pool, &schema, options, &staged, out_error);
    }
    dora_schema_release(&schema);
    return builder_publish_staged(builder, &staged, status);
}

int dora_builder_init_fields(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema_input* input,
                             const struct dora_builder_options* options) {
    return dora_builder_init_fields_error(builder, pool, input, options, nullptr);
}

int dora_builder_init_schema_error(struct dora_builder* builder,
                                   struct dora_pool pool,
                                   const struct dora_schema* schema,
                                   const struct dora_builder_options* options,
                                   struct dora_error* out_error) {
    if (!builder_is_empty(builder)) {
        return EINVAL;
    }
    struct dora_builder staged = {};
    const int status = builder_construct_schema(pool, schema, options, &staged, out_error);
    return builder_publish_staged(builder, &staged, status);
}

int dora_builder_init_schema(struct dora_builder* builder,
                             struct dora_pool pool,
                             const struct dora_schema* schema,
                             const struct dora_builder_options* options) {
    return dora_builder_init_schema_error(builder, pool, schema, options, nullptr);
}

//------------------------------------------------------------------------------
//
// Navigation and Setup
//
//------------------------------------------------------------------------------

//--------------------------------------
// Navigation
//--------------------------------------

static int builder_child_parent(const struct builder_state* state, uint32_t node, uint32_t* out_parent) {
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_node_topology* const inputs = state->topology;
    struct dora_schema_field field = {};
    const int status = dora_schema_get_field(&nodes[node].schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    *out_parent = node;
    if (field.type.id == DORA_TYPE_MAP &&
        (field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LARGE_LIST)) {
        for (uint32_t index = node + 1; index < state->node_count; ++index) {
            if (inputs[index].parent_node_index == node && inputs[index].child_index == 0) {
                *out_parent = index;
                return DORA_OK;
            }
        }
        return EINVAL;
    }
    return DORA_OK;
}

int dora_builder_get_child(const struct dora_builder* builder, uint32_t child_index, struct dora_builder* out_child) {
    struct builder_state* state = nullptr;
    int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase == BUILDER_PHASE_PREPARED) {
        return EBUSY;
    }
    if (!builder_is_empty(out_child)) {
        return EINVAL;
    }
    uint32_t parent = 0;
    status = builder_child_parent(state, builder->node_index, &parent);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_node_topology* const inputs = state->topology;
    for (uint32_t index = parent + 1; index < state->node_count; ++index) {
        if (inputs[index].parent_node_index == parent && inputs[index].child_index == child_index) {
            *out_child = (struct dora_builder){.handle = builder->handle, .node_index = index};
            out_child->handle.is_owner = false;
            return DORA_OK;
        }
    }
    return EINVAL;
}

int dora_builder_find_child(const struct dora_builder* builder, const char* name, struct dora_builder* out_child) {
    struct builder_state* state = nullptr;
    int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase == BUILDER_PHASE_PREPARED) {
        return EBUSY;
    }
    if (name == nullptr || !builder_is_empty(out_child)) {
        return EINVAL;
    }
    uint32_t parent = 0;
    status = builder_child_parent(state, builder->node_index, &parent);
    if (status != DORA_OK) {
        return status;
    }
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_node_topology* const inputs = state->topology;
    for (uint32_t index = parent + 1; index < state->node_count; ++index) {
        if (inputs[index].parent_node_index != parent) {
            continue;
        }
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&nodes[index].schema, &field);
        if (status != DORA_OK) {
            return status;
        }
        if (field.name != nullptr && strcmp(field.name, name) == 0) {
            *out_child = (struct dora_builder){.handle = builder->handle, .node_index = index};
            out_child->handle.is_owner = false;
            return DORA_OK;
        }
    }
    return ENOENT;
}

static int builder_publish_schema(struct dora_schema* destination, struct dora_schema* staged, int status) {
    if (status == DORA_OK) {
        status = dora_schema_move(staged, destination);
    }
    dora_schema_release(staged);
    return status;
}

int dora_builder_retain_schema(const struct dora_builder* builder, struct dora_schema* out_schema) {
    struct builder_operation operation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_INSPECT, &operation);
    if (status != DORA_OK) {
        return status;
    }
    const struct builder_node_state* const nodes = operation.state->nodes;
    struct dora_schema staged = {};
    status = dora_schema_retain(&nodes[operation.data.target.node_index].schema, &staged);
    status = dora_builder_operation_leave(&operation, status, nullptr);
    return builder_publish_schema(out_schema, &staged, status);
}

//--------------------------------------
// Inspection
//--------------------------------------

int dora_builder_get_metrics(const struct dora_builder* builder, struct dora_builder_metrics* out_metrics) {
    struct builder_operation operation = {};
    const int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_ANY_PHASE, &operation);
    if (status != DORA_OK) {
        return status;
    }
    if (out_metrics == nullptr) {
        return dora_builder_operation_leave(&operation, EINVAL, nullptr);
    }
    const struct builder_state* const state = operation.state;
    const struct builder_node_state* const nodes = state->nodes;
    *out_metrics = (struct dora_builder_metrics){
        .row_count = nodes[operation.data.target.node_index].rows.row_count,
        .capacity_bytes = state->capacity_bytes,
        .written_byte_count = state->written_byte_count,
        .null_count = nodes[operation.data.target.node_index].rows.null_count
    };
    return dora_builder_operation_leave(&operation, DORA_OK, nullptr);
}

//--------------------------------------
// Dictionary Values
//--------------------------------------

int dora_builder_set_dictionary_error(struct dora_builder* builder,
                                      const struct dora_vector* values,
                                      struct dora_error* out_error) {
    struct builder_operation operation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &operation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = operation.state;
    struct builder_node_state* const node = &state->nodes[operation.data.target.node_index];
    struct dora_schema declared = {};
    struct dora_schema supplied = {};
    struct dora_handle claim = {};
    struct dora_vector normalized = {};
    struct dora_schema_field field = {};
    status = dora_schema_get_field(&node->schema, &field);
    if (status == DORA_OK) {
        if (field.layout != DORA_LAYOUT_DICTIONARY || node->rows.row_count != 0) {
            status = EINVAL;
        } else {
            status = dora_schema_get_child(&node->schema, 0, &declared);
        }
    }
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(values, &supplied);
    }
    bool matches = true;
    if (status == DORA_OK) {
        status = dictionary_schema_check(&declared, &supplied, &matches);
    }
    bool exact = false;
    if (status == DORA_OK && matches) {
        status = dora_schema_declarations_equal(&declared, &supplied, &exact);
    }
    bool transport = false;
    if (status == DORA_OK && matches && state->statistics != nullptr) {
        status = dora_vector_copy_requires_transport(values, &transport);
    }
    if (status == DORA_OK && (!matches || transport)) {
        status = dora_vector_copy_schema_error(values, state->storage.pool, &declared, &normalized, out_error);
    } else if (status == DORA_OK && !exact) {
        struct vector_image* image = nullptr;
        status = dora_vector_resolve(values, &image);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_schema_error(&normalized,
                                                          state->storage.pool,
                                                          values,
                                                          &declared,
                                                          image->node.content_state,
                                                          out_error);
        }
    }
    if (status == DORA_OK && normalized.handle.slot_index != 0) {
        values = &normalized;
    }
    if (status == DORA_OK) {
        status = dora_identity_claim_acquire(state->storage.pool, &values->handle, IDENTITY_KIND_VECTOR, &claim);
    }
    if (status == DORA_OK) {
        dora_vector_release(&node->dictionary);
        node->dictionary = *values;
        node->dictionary.handle = claim;
        struct vector_image* dictionary = nullptr;
        status = dora_vector_resolve(values, &dictionary);
        if (status == DORA_OK && dictionary->node.content_state < node->rows.content_state) {
            node->rows.content_state = dictionary->node.content_state;
        }
    }
    dora_vector_release(&normalized);
    dora_schema_release(&supplied);
    return dora_builder_operation_leave(&operation, status, nullptr);
}

int dora_builder_set_dictionary(struct dora_builder* builder, const struct dora_vector* values) {
    return dora_builder_set_dictionary_error(builder, values, nullptr);
}

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(alignof(struct dora_node_topology) >= alignof(struct dora_vector));
static_assert(alignof(struct dora_vector) >= alignof(struct dora_view_buffer));
static_assert(alignof(struct dora_vector) >= alignof(struct dora_allocation));
static_assert(alignof(struct dora_allocation) >= alignof(struct vector_image*));
