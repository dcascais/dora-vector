// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/device.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "extension/extension_internal.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

int dora_vector_bind_coordinates(const struct dora_vector* vector, struct dora_vector_view* out_view) {
    struct vector_image* image = nullptr;
    if (out_view == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    struct dora_vector_view view = image->ownership.view;
    if (view.selection.indices != nullptr) {
        const size_t width =
            view.selection.physical_type == DORA_PHYSICAL_TYPE_U32 ? sizeof(uint32_t) : sizeof(uint64_t);
        view.selection.indices = (const uint8_t*)view.selection.indices + vector->rows.offset * width;
    } else {
        view.offset += vector->rows.offset;
        view.validity_bit_offset += vector->rows.offset;
    }
    if (view.null_count == (int64_t)view.length) {
        view.null_count = vector->rows.length;
    } else if (view.null_count != 0 && (vector->rows.offset != 0 || vector->rows.length != view.length)) {
        view.null_count = -1;
    }
    view.length = vector->rows.length;
    *out_view = view;
    return DORA_OK;
}

static int bind_view(const struct dora_vector* vector, struct dora_vector_view* out_view, bool device) {
    struct vector_image* image = nullptr;
    if (out_view == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    struct dora_vector_view view = image->ownership.view;
    if (image->ownership.requires_effective_validity) {
        return ENOTSUP;
    }
    if (device) {
        if (view.placement.access != DORA_ACCESS_DEVICE && view.placement.access != DORA_ACCESS_HOST_AND_DEVICE) {
            return ENOTSUP;
        }
        struct vector_image* provider = image;
        while (view.validity.data != nullptr) {
            if (dora_vector_owns_host_validity(provider, view.validity)) {
                return ENOTSUP;
            }
            if (provider->ownership.backing.handle.slot_index == 0) {
                break;
            }
            const int status = dora_vector_resolve(&provider->ownership.backing, &provider);
            if (status != DORA_OK) {
                return status;
            }
        }
    } else {
        if (view.placement.access != DORA_ACCESS_HOST && view.placement.access != DORA_ACCESS_HOST_AND_DEVICE) {
            return ENOTSUP;
        }
        const int status = dora_vector_settle(vector);
        if (status != DORA_OK) {
            return status;
        }
    }
    const int status = dora_vector_bind_coordinates(vector, &view);
    if (status != DORA_OK) {
        return status;
    }
    if (!device) {
        view.readiness = (struct dora_readiness){};
    }
    *out_view = view;
    return DORA_OK;
}

int dora_vector_get_view(const struct dora_vector* vector, struct dora_vector_view* out_view) {
    return bind_view(vector, out_view, false);
}

int dora_vector_get_device_view(const struct dora_vector* vector, struct dora_vector_view* out_view) {
    return bind_view(vector, out_view, true);
}

//------------------------------------------------------------------------------
//
// Extension Binding
//
//------------------------------------------------------------------------------

static void write_bind_error(struct dora_error* out_error, const char* name, const struct dora_error* hook_error) {
    if (out_error == nullptr || out_error->message[0] != 0) {
        return;
    }
    const char* const parts[] = {
        "Field '",
        name != nullptr && name[0] != 0 ? name : "<unnamed>",
        "': ",
        hook_error->message[0] != 0 ? hook_error->message : "extension binding failed"
    };
    size_t written = 0;
    for (size_t part = 0; part < ARRAY_COUNT(parts); ++part) {
        for (const char* next = parts[part]; *next != 0 && written < sizeof(out_error->message) - 1; ++next) {
            out_error->message[written++] = *next;
        }
    }
    out_error->message[written] = 0;
}

enum vector_binding : uint8_t {
    VECTOR_BIND_SETTLED,
    VECTOR_BIND_ADMISSION
};

int dora_vector_bind_parameters(const struct dora_extension_definition* definition,
                                struct dora_pool pool,
                                const struct dora_schema* storage,
                                const struct dora_schema_field* field,
                                struct dora_allocation* out_parameters,
                                struct dora_error* out_error) {
    if (definition == nullptr || field == nullptr || out_parameters == nullptr || out_parameters->address != nullptr ||
        out_parameters->capacity_bytes != 0 || out_parameters->pool.index != 0 ||
        out_parameters->pool.generation != 0) {
        return EINVAL;
    }
    struct dora_allocation parameters = {};
    struct dora_error hook_error = {};
    if (definition->bind != nullptr) {
        dora_pool_callback_enter(pool);
        const int status = definition->bind(definition,
                                            pool,
                                            storage,
                                            field->type.extension_metadata,
                                            &parameters,
                                            out_error != nullptr ? &hook_error : nullptr);
        dora_pool_callback_leave(pool);
        hook_error.message[sizeof(hook_error.message) - 1] = 0;
        if (status != DORA_OK) {
            dora_allocation_release(&parameters);
            write_bind_error(out_error, field->name, &hook_error);
            return status;
        }
    }
    if ((parameters.address != nullptr || parameters.capacity_bytes != 0) && !pool_is_same(parameters.pool, pool)) {
        dora_allocation_release(&parameters);
        write_bind_error(out_error, field->name, &hook_error);
        return EINVAL;
    }
    const int status = dora_allocation_move(&parameters, out_parameters);
    if (status != DORA_OK) {
        dora_allocation_release(&parameters);
        write_bind_error(out_error, field->name, &hook_error);
        return EINVAL;
    }
    return DORA_OK;
}

int dora_vector_bind_extension(struct vector_image* image, struct dora_error* out_error) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(&image->schema, &field);
    if (status != DORA_OK || field.type.extension_name.byte_count == 0) {
        return status;
    }
    const struct dora_extension_definition* definition = nullptr;
    status = dora_extension_find(image->allocation.pool, field.type.extension_name, &definition);
    if (status != DORA_OK || definition == nullptr) {
        return status;
    }
    struct dora_schema storage = {};
    struct dora_schema values = {};
    if (definition->bind != nullptr) {
        status = dora_vector_retain_schema(&image->ownership.storage, &storage);
        values = dora_schema_borrow(&storage);
        while (status == DORA_OK) {
            struct dora_schema_field storage_field = {};
            status = dora_schema_get_field(&values, &storage_field);
            if (status != DORA_OK ||
                (storage_field.layout != DORA_LAYOUT_DICTIONARY && storage_field.layout != DORA_LAYOUT_RUN_END)) {
                break;
            }
            struct dora_schema child = {};
            status = dora_schema_get_child(&values, storage_field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
            if (status == DORA_OK) {
                values = child;
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_parameters(definition,
                                             image->allocation.pool,
                                             &values,
                                             &field,
                                             &image->ownership.extension_parameters,
                                             out_error);
    }
    dora_schema_release(&storage);
    if (status == DORA_OK) {
        image->ownership.extension = definition;
    }
    return status;
}

int dora_vector_get_extension(const struct dora_vector* vector,
                              const struct dora_extension_definition** out_definition,
                              const struct dora_allocation** out_parameters) {
    struct vector_image* image = nullptr;
    if (out_definition == nullptr || out_parameters == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.extension == nullptr) {
        return ENOTSUP;
    }
    *out_definition = image->ownership.extension;
    *out_parameters = &image->ownership.extension_parameters;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Bound Inputs
//
//------------------------------------------------------------------------------

int dora_vector_input_admit(const struct vector_bound_input* input, uint8_t required_flags, bool reject_extensions) {
    for (uint32_t index = 0; index < input->node_count; ++index) {
        const struct vector_bound_node* const node = &input->nodes[index];
        if (node->field.type.extension_name.byte_count != 0 &&
            (reject_extensions || node->extension == nullptr ||
             (node->extension->flags & required_flags) != required_flags)) {
            return ENOTSUP;
        }
        if ((required_flags & DORA_EXTENSION_FLAG_STORAGE_ORDER) != 0 && node->field.type.id == DORA_TYPE_MAP) {
            return ENOTSUP;
        }
    }
    return DORA_OK;
}

static int input_tree_bind(const struct dora_vector* source,
                           struct vector_bound_input* input,
                           uint32_t* next,
                           enum vector_binding binding) {
    const uint32_t index = (*next)++;
    struct vector_bound_node* const node = &input->nodes[index];
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    if (status == DORA_OK) {
        node->image = image;
        node->source = binding == VECTOR_BIND_ADMISSION ? source : nullptr;
        status = dora_schema_get_field(&image->schema, &node->field);
    }
    if (status == DORA_OK) {
        status = binding == VECTOR_BIND_ADMISSION ? dora_vector_bind_coordinates(source, &node->view)
                                                  : dora_vector_get_view(source, &node->view);
    }
    if (status == DORA_OK && node->field.type.extension_name.byte_count != 0) {
        status = dora_vector_get_extension(source, &node->extension, &node->parameters);
        if (status == ENOTSUP) {
            status = DORA_OK;
        }
    }
    for (uint32_t ordinal = 0; status == DORA_OK && ordinal < node->field.child_count; ++ordinal) {
        struct dora_vector child = {};
        status = node->field.layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, ordinal, &child)
                                                          : dora_vector_get_physical_child(source, ordinal, &child);
        if (status == DORA_OK) {
            const uint32_t child_node = *next;
            status = input_tree_bind(&child, input, next, binding);
            if (binding == VECTOR_BIND_ADMISSION) {
                input->nodes[child_node].source = &dora_vector_image_children(image)[ordinal];
            }
        }
    }
    node->subtree_end_node_index = *next;
    return status;
}

static int
vector_bind_leaf(const struct dora_vector* source, struct vector_bound_node* out_leaf, enum vector_binding binding) {
    struct vector_image* image = nullptr;
    if (out_leaf == nullptr || dora_vector_resolve(source, &image) != DORA_OK) {
        return EINVAL;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(&image->schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    if (field.child_count != 0 || field.type.extension_name.byte_count != 0) {
        return ENOTSUP;
    }
    struct dora_vector_view view;
    status = binding == VECTOR_BIND_ADMISSION ? dora_vector_bind_coordinates(source, &view)
                                              : dora_vector_get_view(source, &view);
    if (status == DORA_OK) {
        *out_leaf = (struct vector_bound_node){
            .image = image,
            .field = field,
            .view = view,
            .subtree_end_node_index = 1,
            .source = source
        };
    }
    return status;
}

static int input_tree_storage(struct scratch_scope* scope, uint32_t count, struct vector_bound_input* out_input) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, sizeof(struct vector_bound_node))) {
        return EOVERFLOW;
    }
    void* nodes = nullptr;
    const int status = dora_scratch_allocate(scope, bytes, &nodes);
    if (status == DORA_OK) {
        memset(nodes, 0, bytes);
        *out_input = (struct vector_bound_input){.nodes = nodes, .node_count = count};
    }
    return status;
}

static int input_tree_count(const struct dora_vector* source, uint32_t* count) {
    if (*count == UINT32_MAX) {
        return EOVERFLOW;
    }
    ++*count;
    struct vector_image* image = nullptr;
    struct dora_schema_field field = {};
    int status = dora_vector_resolve(source, &image);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&image->schema, &field);
    }
    for (uint32_t ordinal = 0; status == DORA_OK && ordinal < field.child_count; ++ordinal) {
        struct dora_vector child = {};
        status = dora_vector_get_physical_child(source, ordinal, &child);
        if (status == DORA_OK) {
            status = input_tree_count(&child, count);
        }
    }
    return status;
}

int dora_vector_bind_input(struct scratch_scope* scope,
                           const struct dora_vector* source,
                           struct vector_bound_node* leaf,
                           struct vector_bound_input* out_input) {
    if (out_input == nullptr || !dora_vector_is_valid(source)) {
        return EINVAL;
    }
    if (leaf != nullptr) {
        const int status = vector_bind_leaf(source, leaf, VECTOR_BIND_SETTLED);
        if (status != ENOTSUP) {
            if (status == DORA_OK) {
                *out_input = (struct vector_bound_input){.nodes = leaf, .node_count = 1, .length = source->rows.length};
            }
            return status;
        }
    }
    uint32_t count = 0;
    int status = input_tree_count(source, &count);
    struct vector_bound_input input = {};
    if (status == DORA_OK) {
        status = input_tree_storage(scope, count, &input);
    }
    if (status != DORA_OK) {
        return status;
    }
    input.length = source->rows.length;
    uint32_t next = 0;
    status = input_tree_bind(source, &input, &next, VECTOR_BIND_SETTLED);
    if (status == DORA_OK && input.nodes[0].view.layout == DORA_LAYOUT_RUN_END) {
        const struct dora_vector_view* const view = &input.nodes[0].view;
        const size_t width = view->byte_width;
        input.is_constant = width != 0 && view->values.byte_count / width - view->run_end_element_offset == 1 &&
                            view->validity.data == nullptr;
    }
    if (status == DORA_OK) {
        input.nodes[0].source = source;
        *out_input = input;
    }
    return status;
}

int dora_vector_bind_children(struct scratch_scope* scope,
                              const struct dora_vector* children,
                              uint32_t child_count,
                              struct vector_bound_node* leaves,
                              struct vector_bound_input* out_input) {
    if (child_count == 1 && leaves != nullptr) {
        const int status = vector_bind_leaf(children, &leaves[1], VECTOR_BIND_ADMISSION);
        if (status != ENOTSUP) {
            if (status == DORA_OK) {
                leaves[0].subtree_end_node_index = 2;
                leaves[1].subtree_end_node_index = 2;
                *out_input = (struct vector_bound_input){.nodes = leaves, .node_count = 2};
            }
            return status;
        }
    }
    uint32_t count = 1;
    int status = DORA_OK;
    for (uint32_t ordinal = 0; status == DORA_OK && ordinal < child_count; ++ordinal) {
        status = input_tree_count(&children[ordinal], &count);
    }
    struct vector_bound_input input = {};
    if (status == DORA_OK) {
        status = input_tree_storage(scope, count, &input);
    }
    if (status != DORA_OK) {
        return status;
    }
    uint32_t next = 1;
    for (uint32_t ordinal = 0; status == DORA_OK && ordinal < child_count; ++ordinal) {
        status = input_tree_bind(&children[ordinal], &input, &next, VECTOR_BIND_ADMISSION);
    }
    input.nodes[0].subtree_end_node_index = next;
    if (status == DORA_OK) {
        *out_input = input;
    }
    return status;
}

uint32_t dora_vector_input_child(uint32_t node, const struct vector_bound_input* input, uint32_t ordinal) {
    uint32_t child = node + 1;
    for (uint32_t index = 0; index < ordinal; ++index) {
        child = input->nodes[child].subtree_end_node_index;
    }
    return child;
}

static bool vector_inputs_have_matching_trees(const struct vector_bound_input* left,
                                              uint32_t left_node,
                                              const struct vector_bound_input* right,
                                              uint32_t right_node) {
    const struct dora_logical_type* const left_type = &left->nodes[left_node].field.type;
    const struct dora_logical_type* const right_type = &right->nodes[right_node].field.type;
    left_node = vector_input_value_node(left, left_node);
    right_node = vector_input_value_node(right, right_node);
    const uint32_t count = left->nodes[left_node].field.child_count;
    struct dora_logical_type right_match = *right_type;
    if (left_type->id == DORA_TYPE_DECIMAL && right_type->id == DORA_TYPE_DECIMAL) {
        // Decimal coercion must preserve extension identity.
        right_match.parameters = left_type->parameters;
    }
    if (count != right->nodes[right_node].field.child_count || !dora_type_is_equal(left_type, &right_match, count)) {
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (!vector_inputs_have_matching_trees(left,
                                               dora_vector_input_child(left_node, left, index),
                                               right,
                                               dora_vector_input_child(right_node, right, index))) {
            return false;
        }
    }
    return true;
}

int dora_vector_input_admit_pair(const struct vector_bound_input* left,
                                 const struct vector_bound_input* right,
                                 bool order) {
    const uint8_t required = order ? DORA_EXTENSION_FLAG_STORAGE_ORDER : DORA_EXTENSION_FLAG_STORAGE_EQUALITY;
    int status = dora_vector_input_admit(left, required, false);
    if (status == DORA_OK) {
        status = dora_vector_input_admit(right, required, false);
    }
    if (status == DORA_OK && (left->length != right->length || !vector_inputs_have_matching_trees(left, 0, right, 0))) {
        status = EINVAL;
    }
    return status;
}
