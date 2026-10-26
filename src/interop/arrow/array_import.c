// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "interop/arrow/arrow_internal.h"
#include "memory/memory_internal.h"
#include "schema/schema_internal.h"
#include "status/status_internal.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct array_import_context {
    struct dora_pool pool;
    struct dora_arrow_limits limits;
    struct arrow_traversal_state* traversal;
    struct dora_error* error;
    struct arrow_adopted_array* adopted;
    bool should_copy;
    struct vector_bound_input foreign;
    struct dora_allocation nodes_storage;
    struct dora_allocation views_storage;
    struct dora_view_buffer* views;
    uint32_t node_capacity;
    uint32_t view_count;
    uint32_t view_capacity;
};

static void array_owner_release(struct dora_buffer_owner* owner) {
    struct arrow_adopted_array* const adopted = (struct arrow_adopted_array*)owner;
    if (--adopted->claim_count != 0) {
        return;
    }
    if (adopted->array.release != nullptr) {
        adopted->array.release(&adopted->array);
    }
    struct dora_allocation storage = adopted->self_storage;
    dora_allocation_release(&storage);
}

static int array_owner_read(struct dora_buffer_owner* owner, struct dora_device_region source, uint8_t* out_bytes) {
    struct arrow_adopted_array* const adopted = (struct arrow_adopted_array*)owner;
    if (source.byte_count > SIZE_MAX || source.byte_offset > SIZE_MAX ||
        source.byte_count > SIZE_MAX - source.byte_offset) {
        return EOVERFLOW;
    }
    if (source.byte_count == 0) {
        return DORA_OK;
    }
    if (source.data == nullptr || out_bytes == nullptr) {
        return EINVAL;
    }
    if (owner->readiness.settle != nullptr) {
        const int status = owner->readiness.settle(&owner->readiness);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (owner->placement.access == DORA_ACCESS_HOST || owner->placement.access == DORA_ACCESS_HOST_AND_DEVICE) {
        memcpy(out_bytes, (const uint8_t*)source.data + (size_t)source.byte_offset, (size_t)source.byte_count);
        return DORA_OK;
    }
    return adopted->access == nullptr || adopted->access->read_host == nullptr
               ? ENOTSUP
               : adopted->access->read_host(adopted->access, owner->placement, source, out_bytes);
}

static int buffer_extent(const void* pointer, uint64_t count, uint32_t width, struct dora_bytes* out_buffer) {
    size_t bytes = 0;
    if (count > SIZE_MAX || ckd_mul(&bytes, (size_t)count, width)) {
        return EOVERFLOW;
    }
    if (bytes != 0 && pointer == nullptr) {
        return EINVAL;
    }
    *out_buffer = (struct dora_bytes){.data = pointer, .byte_count = bytes};
    return DORA_OK;
}

static int array_integer(struct array_import_context* context,
                         const void* data,
                         uint64_t index,
                         uint32_t width,
                         int64_t* out_value) {
    uint64_t offset = 0;
    if (ckd_mul(&offset, index, width)) {
        return EOVERFLOW;
    }

    union arrow_integer {
        int16_t small;
        int32_t narrow;
        int64_t wide;
    } value = {};

    const int status =
        array_owner_read(&context->adopted->owner,
                         (struct dora_device_region){.data = data, .byte_offset = offset, .byte_count = width},
                         (uint8_t*)&value);
    if (status == DORA_OK) {
        *out_value = width == 2 ? value.small : width == 4 ? value.narrow : value.wide;
    }
    return status;
}

static int array_header(const struct ArrowArray* source) {
    if (source == nullptr || source->release == nullptr || source->length < 0 || source->offset < 0 ||
        source->null_count < -1 || source->null_count > source->length || source->n_buffers < 0 ||
        source->n_buffers > UINT32_MAX || source->n_children < 0 || source->n_children > UINT32_MAX ||
        (source->n_buffers != 0 && source->buffers == nullptr) ||
        (source->n_children != 0 && source->children == nullptr)) {
        return EINVAL;
    }
    uint64_t endpoint = 0;
    return ckd_add(&endpoint, (uint64_t)source->offset, (uint64_t)source->length) || endpoint > INT64_MAX ? EINVAL
                                                                                                          : DORA_OK;
}

static int array_enter(struct array_import_context* context, const struct ArrowArray* source, uint32_t depth) {
    const int status = array_header(source);
    if (status != DORA_OK) {
        return status;
    }
    if (depth >= context->limits.max_depth || context->traversal->node_count >= context->limits.max_node_count) {
        return ERANGE;
    }
    const int stack_status = dora_arrow_traversal_push(context->traversal, source, depth);
    if (stack_status != DORA_OK) {
        return stack_status;
    }
    ++context->traversal->node_count;
    const uint64_t edges = (uint64_t)source->n_children + (source->dictionary == nullptr ? 0U : 1U);
    if (edges > context->limits.max_edge_count - context->traversal->edge_count) {
        return ERANGE;
    }
    context->traversal->edge_count += (uint32_t)edges;
    uint64_t work = 0;
    if (ckd_add(&work, (uint64_t)source->offset, (uint64_t)source->length) || ckd_add(&work, work, 1)) {
        return EOVERFLOW;
    }
    return arrow_traversal_charge(&context->limits, context->traversal, work, 0);
}

static int activity_required(struct array_import_context* context,
                             const struct dora_schema* schema,
                             const struct ArrowArray* source,
                             uint32_t depth,
                             bool* out_required) {
    if (depth >= context->limits.max_depth) {
        return ERANGE;
    }
    int status = array_header(source);
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(schema, &field);
    }
    if (status != DORA_OK) {
        return status;
    }
    const bool dictionary = field.layout == DORA_LAYOUT_DICTIONARY;
    const bool run_end = field.layout == DORA_LAYOUT_RUN_END;
    const bool map = field.type.id == DORA_TYPE_MAP && field.layout == DORA_LAYOUT_LIST;
    const bool encoded =
        dictionary || run_end || field.layout == DORA_LAYOUT_DENSE_UNION || field.layout == DORA_LAYOUT_SPARSE_UNION;
    const bool null_free = source->null_count == 0 && field.layout != DORA_LAYOUT_NULL && !encoded;
    *out_required =
        source->length != 0 && (map || ((field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0 && !null_free));
    if (*out_required) {
        return DORA_OK;
    }
    const uint32_t expected_children = dictionary ? 0 : field.child_count;
    if ((uint64_t)source->n_children != expected_children ||
        (dictionary ? source->dictionary == nullptr : source->dictionary != nullptr)) {
        return EINVAL;
    }
    for (uint32_t child = run_end ? 1U : 0U; status == DORA_OK && !*out_required && child < field.child_count;
         ++child) {
        struct dora_schema child_schema = {};
        status = dora_schema_get_physical_child(schema, child, &child_schema);
        const struct ArrowArray* const child_source = dictionary ? source->dictionary : source->children[child];
        if (status == DORA_OK) {
            status = activity_required(context, &child_schema, child_source, depth + 1, out_required);
        }
    }
    return status;
}

static int activity_row(struct array_import_context* context,
                        const struct ArrowArray* source,
                        enum dora_layout layout,
                        const struct dora_bytes* ancestor_activity,
                        uint64_t row,
                        bool* out_active) {
    *out_active = ancestor_activity == nullptr || dora_bitmap_test(ancestor_activity->data, row);
    if (!*out_active) {
        return DORA_OK;
    }
    if (layout == DORA_LAYOUT_NULL) {
        *out_active = false;
        return DORA_OK;
    }
    if (layout == DORA_LAYOUT_RUN_END || layout == DORA_LAYOUT_DENSE_UNION || layout == DORA_LAYOUT_SPARSE_UNION ||
        source->null_count == 0 || source->buffers[0] == nullptr) {
        return DORA_OK;
    }
    const uint64_t bit = (uint64_t)source->offset + row;
    uint8_t byte = 0;
    const int status = array_owner_read(
        &context->adopted->owner,
        (struct dora_device_region){.data = source->buffers[0], .byte_offset = bitmap_byte_of(bit), .byte_count = 1},
        &byte);
    if (status == DORA_OK) {
        *out_active = dora_bitmap_test(&byte, bitmap_bit_in_byte(bit));
    }
    return status;
}

static int activity_index(struct array_import_context* context,
                          const struct ArrowArray* source,
                          uint32_t width,
                          const struct dora_schema_field* field,
                          uint64_t physical,
                          uint64_t* out_index) {
    union arrow_index {
        int8_t i8;
        uint8_t u8;
        int16_t i16;
        uint16_t u16;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        uint64_t u64;
    } value = {};

    uint64_t offset = 0;
    if (width == 0 || width > sizeof(value)) {
        return EINVAL;
    }
    if (ckd_mul(&offset, physical, width)) {
        return EOVERFLOW;
    }
    const int status = array_owner_read(
        &context->adopted->owner,
        (struct dora_device_region){.data = source->buffers[1], .byte_offset = offset, .byte_count = width},
        (uint8_t*)&value);
    if (status != DORA_OK) {
        return status;
    }
    switch (field->index_type) {
    case DORA_TYPE_INT8:
        if (value.i8 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i8;
        break;
    case DORA_TYPE_UINT8:
        *out_index = value.u8;
        break;
    case DORA_TYPE_INT16:
        if (value.i16 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i16;
        break;
    case DORA_TYPE_UINT16:
        *out_index = value.u16;
        break;
    case DORA_TYPE_INT32:
        if (value.i32 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i32;
        break;
    case DORA_TYPE_UINT32:
        *out_index = value.u32;
        break;
    case DORA_TYPE_INT64:
        if (value.i64 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i64;
        break;
    case DORA_TYPE_UINT64:
        *out_index = value.u64;
        break;
    default:
        return EINVAL;
    }
    return DORA_OK;
}

static int activity_runs(struct array_import_context* context,
                         const struct ArrowArray* source,
                         const struct dora_schema_field* field,
                         const struct ArrowArray* values,
                         const struct dora_bytes* ancestor_activity,
                         uint64_t* mask) {
    const struct ArrowArray* const runs = source->children[0];
    int status = array_header(runs);
    if (status != DORA_OK) {
        return status;
    }
    if (runs->n_buffers != 2 || runs->n_children != 0 || runs->dictionary != nullptr || runs->length > values->length) {
        return EINVAL;
    }
    const uint32_t width = field->index_type == DORA_TYPE_INT16   ? 2
                           : field->index_type == DORA_TYPE_INT32 ? 4
                           : field->index_type == DORA_TYPE_INT64 ? 8
                                                                  : 0;
    if (width == 0) {
        return EINVAL;
    }
    if (source->length == 0) {
        return DORA_OK;
    }
    struct dora_bytes extent = {};
    status = buffer_extent(runs->buffers[1], (uint64_t)runs->offset + (uint64_t)runs->length, width, &extent);
    const uint64_t first = (uint64_t)source->offset;
    const uint64_t last = first + (uint64_t)source->length;
    uint64_t previous = 0;
    for (uint64_t run = 0; status == DORA_OK && run < (uint64_t)runs->length; ++run) {
        int64_t endpoint = 0;
        status = array_integer(context, extent.data, (uint64_t)runs->offset + run, width, &endpoint);
        if (status != DORA_OK) {
            break;
        }
        if (endpoint <= 0 || (uint64_t)endpoint <= previous) {
            return EINVAL;
        }
        const uint64_t begin = previous > first ? previous : first;
        const uint64_t end = (uint64_t)endpoint < last ? (uint64_t)endpoint : last;
        bool active = begin < end && ancestor_activity == nullptr;
        for (uint64_t row = begin; status == DORA_OK && !active && row < end; ++row) {
            active = dora_bitmap_test(ancestor_activity->data, row - first);
        }
        if (status == DORA_OK && active) {
            bitmap_set(mask, run);
        }
        previous = (uint64_t)endpoint;
    }
    return status != DORA_OK ? status : previous < last ? EINVAL : DORA_OK;
}

static int child_activity(struct array_import_context* context,
                          const struct ArrowArray* source,
                          const struct dora_schema_field* field,
                          const struct ArrowArray* child,
                          uint32_t child_index,
                          const struct dora_bytes* ancestor_activity,
                          uint32_t index_width,
                          struct dora_allocation* out_storage,
                          struct dora_bytes* out_activity) {
    int status = array_header(child);
    if (status != DORA_OK) {
        return status;
    }
    const uint64_t child_count = (uint64_t)child->length;
    size_t bytes = 0;
    if (ckd_mul(&bytes, bitmap_words_for_bits(child_count), sizeof(uint64_t))) {
        return EOVERFLOW;
    }
    status = arrow_traversal_charge(&context->limits, context->traversal, 0, bytes);
    if (status == DORA_OK) {
        status = dora_pool_allocate(context->pool, bytes, out_storage);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (bytes != 0) {
        bitmap_fill_bytes(out_storage->address, bytes, false);
    }
    *out_activity = (struct dora_bytes){.data = out_storage->address, .byte_count = bytes};
    uint64_t* const mask = out_storage->address;
    if (field->layout == DORA_LAYOUT_RUN_END) {
        return activity_runs(context, source, field, child, ancestor_activity, mask);
    }
    const bool list_view = field->layout == DORA_LAYOUT_LIST_VIEW || field->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    const bool list = list_view || field->layout == DORA_LAYOUT_LIST || field->layout == DORA_LAYOUT_LARGE_LIST;
    for (uint64_t row = 0; status == DORA_OK && row < (uint64_t)source->length; ++row) {
        const uint64_t physical = (uint64_t)source->offset + row;
        uint64_t begin = 0;
        uint64_t count = 1;
        if (list) {
            const uint32_t width = field->layout == DORA_LAYOUT_LIST || field->layout == DORA_LAYOUT_LIST_VIEW ? 4 : 8;
            int64_t first = 0;
            int64_t last = 0;
            status = array_integer(context, source->buffers[1], physical, width, &first);
            if (status == DORA_OK) {
                status = array_integer(context,
                                       source->buffers[list_view ? 2 : 1],
                                       physical + (list_view ? 0U : 1U),
                                       width,
                                       &last);
            }
            if (status == DORA_OK && (first < 0 || last < 0 || (!list_view && last < first))) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                begin = (uint64_t)first;
                count = list_view ? (uint64_t)last : (uint64_t)last - (uint64_t)first;
                if (begin > child_count || count > child_count - begin) {
                    status = EINVAL;
                }
            }
            if (status == DORA_OK) {
                status = arrow_traversal_charge(&context->limits, context->traversal, count, 0);
            }
            if (status != DORA_OK) {
                break;
            }
        }
        bool active = false;
        status = activity_row(context, source, field->layout, ancestor_activity, row, &active);
        if (status != DORA_OK || !active) {
            continue;
        }
        switch (field->layout) {
        case DORA_LAYOUT_STRUCT:
            begin = physical;
            break;
        case DORA_LAYOUT_FIXED_LIST:
            count = field->type.parameters.list_element_count;
            if (ckd_mul(&begin, physical, count)) {
                status = EOVERFLOW;
            }
            break;
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW:
            break;
        case DORA_LAYOUT_DICTIONARY:
            status = activity_index(context, source, index_width, field, physical, &begin);
            break;
        case DORA_LAYOUT_DENSE_UNION:
        case DORA_LAYOUT_SPARSE_UNION: {
            int8_t code = 0;
            status = array_owner_read(
                &context->adopted->owner,
                (struct dora_device_region){.data = source->buffers[0], .byte_offset = physical, .byte_count = 1},
                (uint8_t*)&code);
            uint32_t selected = 0;
            for (; status == DORA_OK && selected < field->child_count; ++selected) {
                const int8_t declared = field->type.parameters.type_codes == nullptr
                                            ? (int8_t)selected
                                            : field->type.parameters.type_codes[selected];
                if (code == declared) {
                    break;
                }
            }
            if (status == DORA_OK && (code < 0 || selected == field->child_count)) {
                status = EINVAL;
            }
            if (status != DORA_OK || selected != child_index) {
                continue;
            }
            begin = physical;
            if (field->layout == DORA_LAYOUT_DENSE_UNION) {
                int64_t member = 0;
                status = array_integer(context, source->buffers[1], physical, 4, &member);
                if (status == DORA_OK && member < 0) {
                    status = EINVAL;
                }
                if (status == DORA_OK) {
                    begin = (uint64_t)member;
                }
            }
            break;
        }
        default:
            return EINVAL;
        }
        if (status != DORA_OK) {
            break;
        }
        if (begin > child_count || count > child_count - begin) {
            return EINVAL;
        }
        if (status == DORA_OK) {
            dora_bitmap_set_range(mask, begin, begin + count, true);
        }
    }
    return status;
}

static int array_window_work(struct array_import_context* context,
                             const struct ArrowArray* source,
                             const struct dora_schema_field* field) {
    const bool binary = field->layout == DORA_LAYOUT_BINARY || field->layout == DORA_LAYOUT_LARGE_BINARY;
    const bool view = field->layout == DORA_LAYOUT_LIST_VIEW || field->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    const bool list = field->layout == DORA_LAYOUT_LIST || field->layout == DORA_LAYOUT_LARGE_LIST || view;
    const bool byte_view = field->layout == DORA_LAYOUT_BINARY_VIEW;
    if ((!binary && !list && !byte_view) || source->length == 0) {
        return DORA_OK;
    }
    const uint32_t width = field->layout == DORA_LAYOUT_BINARY || field->layout == DORA_LAYOUT_LIST ||
                                   field->layout == DORA_LAYOUT_LIST_VIEW
                               ? 4
                               : 8;
    if (binary) {
        int64_t first = 0;
        int64_t last = 0;
        int status = array_integer(context, source->buffers[1], (uint64_t)source->offset, width, &first);
        if (status == DORA_OK) {
            status = array_integer(context,
                                   source->buffers[1],
                                   (uint64_t)source->offset + (uint64_t)source->length,
                                   width,
                                   &last);
        }
        if (status == DORA_OK && (first < 0 || last < first)) {
            status = EINVAL;
        }
        return status == DORA_OK
                   ? arrow_traversal_charge(&context->limits, context->traversal, (uint64_t)last - (uint64_t)first, 0)
                   : status;
    }
    for (uint64_t row = 0; row < (uint64_t)source->length; ++row) {
        int status;
        const uint64_t physical = (uint64_t)source->offset + row;
        uint64_t count = 0;
        if (byte_view) {
            bool active = false;
            status = activity_row(context, source, field->layout, nullptr, row, &active);
            if (status != DORA_OK) {
                return status;
            }
            if (!active) {
                continue;
            }
            struct dora_binary_view_entry entry = {};
            uint64_t offset = 0;
            if (ckd_mul(&offset, physical, sizeof(entry))) {
                return EOVERFLOW;
            }
            status = array_owner_read(&context->adopted->owner,
                                      (struct dora_device_region){
                                          .data = source->buffers[1],
                                          .byte_offset = offset,
                                          .byte_count = sizeof(entry)
                                      },
                                      (uint8_t*)&entry);
            if (status == DORA_OK && entry.byte_count < 0) {
                status = EINVAL;
            }
            count = (uint32_t)entry.byte_count;
        } else {
            int64_t first = 0;
            int64_t last = 0;
            status = array_integer(context, source->buffers[1], physical, width, &first);
            if (status == DORA_OK) {
                status =
                    array_integer(context, source->buffers[view ? 2 : 1], physical + (view ? 0U : 1U), width, &last);
            }
            if (status == DORA_OK && (first < 0 || last < 0 || (!view && last < first))) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                count = view ? (uint64_t)last : (uint64_t)last - (uint64_t)first;
            }
        }
        if (status == DORA_OK) {
            status = arrow_traversal_charge(&context->limits, context->traversal, count, 0);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int array_field_charge(struct array_import_context* context, const struct dora_schema_field* field) {
    if (field->type.extension_name.byte_count != 0) {
        struct dora_schema_field storage = *field;
        storage.type.extension_name = (struct dora_bytes){};
        storage.type.extension_metadata = (struct dora_bytes){};
        const int status = dora_arrow_field_charge(&context->limits, context->traversal, &storage);
        if (status != DORA_OK) {
            return status;
        }
    }
    return dora_arrow_field_charge(&context->limits, context->traversal, field);
}

static int array_import(struct array_import_context* context,
                        const struct dora_schema* schema,
                        const struct ArrowArray* source,
                        uint32_t depth,
                        const struct dora_bytes* ancestor_activity,
                        struct dora_vector* out_vector) {
    int status = array_enter(context, source, depth);
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(schema, &field);
    }
    if (status != DORA_OK) {
        return status;
    }
    const bool dictionary = field.layout == DORA_LAYOUT_DICTIONARY;
    const uint32_t expected_children = dictionary ? 0 : field.child_count;
    if ((uint64_t)source->n_children != expected_children || (dictionary && source->dictionary == nullptr) ||
        (!dictionary && source->dictionary != nullptr)) {
        return EINVAL;
    }
    const uint32_t position = context->foreign.node_count;
    if (context->should_copy) {
        if (position == context->node_capacity) {
            uint32_t required = 0;
            size_t node_bytes = 0;
            size_t current = 0;
            struct dora_allocation storage = {};
            if (ckd_add(&required, position, 1U) || ckd_mul(&node_bytes, required, sizeof(struct vector_bound_node)) ||
                ckd_mul(&current, context->node_capacity, sizeof(struct vector_bound_node))) {
                return EOVERFLOW;
            }
            if (context->nodes_storage.address != nullptr) {
                current = context->nodes_storage.capacity_bytes;
            }
            status = dora_memory_payload_allocate_preferred(context->pool, current, node_bytes, &storage);
            if (status != DORA_OK) {
                return status;
            }
            const size_t available = storage.capacity_bytes / sizeof(struct vector_bound_node);
            const uint32_t capacity = available > UINT32_MAX ? UINT32_MAX : (uint32_t)available;
            memcpy(storage.address, context->foreign.nodes, position * sizeof(struct vector_bound_node));
            dora_allocation_release(&context->nodes_storage);
            context->nodes_storage = storage;
            context->foreign.nodes = storage.address;
            context->node_capacity = capacity;
        }
        context->foreign.nodes[context->foreign.node_count++] = (struct vector_bound_node){};
    }
    uint32_t expected_buffers = 0;
    switch (field.layout) {
    case DORA_LAYOUT_NULL:
        expected_buffers = (uint32_t)source->n_buffers;
        break;
    case DORA_LAYOUT_RUN_END:
        expected_buffers = 0;
        break;
    case DORA_LAYOUT_STRUCT:
    case DORA_LAYOUT_FIXED_LIST:
    case DORA_LAYOUT_SPARSE_UNION:
        expected_buffers = 1;
        break;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        expected_buffers = 3;
        break;
    case DORA_LAYOUT_BINARY_VIEW:
        expected_buffers = (uint32_t)source->n_buffers;
        break;
    default:
        expected_buffers = 2;
        break;
    }
    if ((uint64_t)source->n_buffers != expected_buffers || (field.layout == DORA_LAYOUT_NULL && expected_buffers > 1) ||
        (field.layout == DORA_LAYOUT_BINARY_VIEW && expected_buffers < 3)) {
        status = EINVAL;
    }
    struct dora_bytes inline_buffers[8];
    struct dora_vector inline_children[8];
    struct dora_bytes* buffers = nullptr;
    struct dora_vector* children = nullptr;
    struct dora_allocation buffers_storage = {};
    struct dora_allocation children_storage = {};
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, expected_buffers, sizeof(struct dora_bytes))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context->limits, context->traversal, 0, bytes);
    }
    if (status == DORA_OK && expected_buffers > 8) {
        status = dora_pool_allocate(context->pool, bytes, &buffers_storage);
    }
    if (status == DORA_OK) {
        buffers = expected_buffers > 8 ? buffers_storage.address : inline_buffers;
        if (bytes != 0) {
            memset(buffers, 0, bytes);
        }
    }
    if (status == DORA_OK && ckd_mul(&bytes, field.child_count, sizeof(struct dora_vector))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context->limits, context->traversal, 0, bytes);
    }
    if (status == DORA_OK && !context->should_copy && field.child_count > 8) {
        status = dora_pool_allocate(context->pool, bytes, &children_storage);
    }
    if (status == DORA_OK) {
        children = !context->should_copy && field.child_count > 8 ? children_storage.address : inline_children;
        if (!context->should_copy && bytes != 0) {
            memset(children, 0, bytes);
        }
    }
    uint64_t endpoint = 0;
    if (status == DORA_OK && ckd_add(&endpoint, (uint64_t)source->offset, (uint64_t)source->length)) {
        status = EOVERFLOW;
    }
    const bool validity = field.layout != DORA_LAYOUT_NULL && field.layout != DORA_LAYOUT_RUN_END &&
                          field.layout != DORA_LAYOUT_DENSE_UNION && field.layout != DORA_LAYOUT_SPARSE_UNION;
    if (status == DORA_OK && validity) {
        if (source->buffers[0] != nullptr) {
            status = buffer_extent(source->buffers[0], bitmap_bytes_for_bits(endpoint), 1, &buffers[0]);
        } else if (source->null_count != 0 && endpoint != 0) {
            status = EINVAL;
        }
    }
    uint32_t byte_width = 0;
    if (status == DORA_OK) {
        switch (field.layout) {
        case DORA_LAYOUT_BITS:
            status = buffer_extent(source->buffers[1], bitmap_bytes_for_bits(endpoint), 1, &buffers[1]);
            break;
        case DORA_LAYOUT_FIXED:
        case DORA_LAYOUT_DICTIONARY: {
            const struct dora_logical_type type =
                dictionary ? (struct dora_logical_type){.id = field.index_type} : field.type;
            enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
            status = type_physical(&type, &physical);
            if (status == DORA_OK) {
                byte_width = type.id == DORA_TYPE_FIXED_BINARY ? type.parameters.byte_width : physical_width(physical);
                status = buffer_extent(source->buffers[1], endpoint, byte_width, &buffers[1]);
            }
            break;
        }
        case DORA_LAYOUT_DENSE_UNION:
        case DORA_LAYOUT_SPARSE_UNION:
            status = buffer_extent(source->buffers[0], endpoint, 1, &buffers[0]);
            if (status == DORA_OK && field.layout == DORA_LAYOUT_DENSE_UNION) {
                status = buffer_extent(source->buffers[1], endpoint, 4, &buffers[1]);
            }
            break;
        case DORA_LAYOUT_BINARY:
        case DORA_LAYOUT_LARGE_BINARY:
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW: {
            const bool wide = field.layout == DORA_LAYOUT_LARGE_BINARY || field.layout == DORA_LAYOUT_LARGE_LIST ||
                              field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
            const bool view = field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
            const bool binary = field.layout == DORA_LAYOUT_BINARY || field.layout == DORA_LAYOUT_LARGE_BINARY;
            uint64_t count = endpoint;
            if (!view && ckd_add(&count, count, 1)) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK) {
                status = buffer_extent(source->buffers[1], count, wide ? 8 : 4, &buffers[1]);
            }
            if (status == DORA_OK && view) {
                status = buffer_extent(source->buffers[2], endpoint, wide ? 8 : 4, &buffers[2]);
            }
            int64_t data_bytes = 0;
            if (status == DORA_OK && binary) {
                status = array_integer(context, source->buffers[1], endpoint, wide ? 8 : 4, &data_bytes);
                if (status == DORA_OK && data_bytes < 0) {
                    status = EINVAL;
                }
                if (status == DORA_OK) {
                    status = buffer_extent(source->buffers[2], (uint64_t)data_bytes, 1, &buffers[2]);
                }
            }
            break;
        }
        case DORA_LAYOUT_BINARY_VIEW: {
            status = buffer_extent(source->buffers[1], endpoint, 16, &buffers[1]);
            const uint32_t count = expected_buffers - 3;
            if (status == DORA_OK) {
                status = buffer_extent(source->buffers[expected_buffers - 1], count, 8, &buffers[expected_buffers - 1]);
            }
            for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
                int64_t size = 0;
                status = array_integer(context, source->buffers[expected_buffers - 1], index, 8, &size);
                if (status == DORA_OK && size < 0) {
                    status = EINVAL;
                }
                if (status == DORA_OK) {
                    status = buffer_extent(source->buffers[index + 2], (uint64_t)size, 1, &buffers[index + 2]);
                }
            }
            break;
        }
        default:
            break;
        }
    }
    bool geometry_charged = false;
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema child_schema = {};
        status = dora_schema_get_physical_child(schema, index, &child_schema);
        const struct ArrowArray* const child_source = dictionary ? source->dictionary : source->children[index];
        const uint32_t child_depth = depth + 1;
        bool required = false;
        if (status == DORA_OK && !(field.layout == DORA_LAYOUT_RUN_END && index == 0)) {
            status = activity_required(context, &child_schema, child_source, child_depth, &required);
        }
        struct dora_allocation activity_storage = {};
        struct dora_bytes activity = {};
        if (status == DORA_OK && required) {
            status = child_activity(context,
                                    source,
                                    &field,
                                    child_source,
                                    index,
                                    ancestor_activity,
                                    byte_width,
                                    &activity_storage,
                                    &activity);
            geometry_charged = field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LARGE_LIST ||
                               field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        }
        if (status == DORA_OK) {
            status = array_import(context,
                                  &child_schema,
                                  child_source,
                                  child_depth,
                                  required ? &activity : nullptr,
                                  context->should_copy ? nullptr : &children[index]);
        }
        dora_allocation_release(&activity_storage);
    }
    if (status == DORA_OK && !geometry_charged) {
        status = array_window_work(context, source, &field);
    }
    if (status == DORA_OK && context->should_copy) {
        for (uint32_t index = 0; status == DORA_OK && index < expected_buffers; ++index) {
            status = arrow_traversal_charge(&context->limits,
                                            context->traversal,
                                            buffers[index].byte_count,
                                            buffers[index].byte_count);
        }
    }
    if (status == DORA_OK) {
        status = array_field_charge(context, &field);
    }
    if (status == DORA_OK) {
        if (!context->should_copy && context->adopted->claim_count == UINT32_MAX) {
            status = EOVERFLOW;
        } else {
            const struct vector_input input = {
                .field = &field,
                .admitted_schema = schema->handle.pool_index == context->pool.index ? schema : nullptr,
                .row_count = (uint64_t)source->length,
                .null_count = source->null_count,
                .offset = (uint64_t)source->offset,
                .validity_bit_offset = (uint64_t)source->offset,
                .buffers = buffers,
                .buffer_count = expected_buffers,
                .children = context->should_copy ? nullptr : children,
                .foreign = context->should_copy ? &context->foreign : nullptr,
                .foreign_node_index = position,
                .child_count = field.child_count,
                .ancestor_activity = ancestor_activity,
                .owner = &context->adopted->owner,
                .should_scan_geometry = true
            };
            if (context->should_copy) {
                const bool is_union =
                    field.layout == DORA_LAYOUT_DENSE_UNION || field.layout == DORA_LAYOUT_SPARSE_UNION;
                const uint32_t member_buffers = is_union
                                                    ? (UNION_MEMBER_CAPACITY + sizeof(struct dora_view_buffer) - 1) /
                                                          sizeof(struct dora_view_buffer)
                                                    : 0;
                uint32_t needed = 0;
                if (ckd_add(&needed, context->view_count, expected_buffers) ||
                    ckd_add(&needed, needed, member_buffers)) {
                    status = EOVERFLOW;
                }
                if (status == DORA_OK && needed > context->view_capacity) {
                    size_t view_bytes = 0;
                    size_t current = 0;
                    struct dora_allocation storage = {};
                    if (ckd_mul(&view_bytes, needed, sizeof(struct dora_view_buffer)) ||
                        ckd_mul(&current, context->view_capacity, sizeof(struct dora_view_buffer))) {
                        status = EOVERFLOW;
                    } else {
                        if (context->views_storage.address != nullptr) {
                            current = context->views_storage.capacity_bytes;
                        }
                        status = dora_memory_payload_allocate_preferred(context->pool, current, view_bytes, &storage);
                    }
                    if (status == DORA_OK) {
                        memcpy(storage.address, context->views, context->view_count * sizeof(struct dora_view_buffer));
                        for (uint32_t node = 0; node < context->foreign.node_count; ++node) {
                            struct dora_vector_view* const view = &context->foreign.nodes[node].view;
                            if (view->data_buffers != nullptr) {
                                view->data_buffers =
                                    (struct dora_view_buffer*)storage.address + (view->data_buffers - context->views);
                            }
                            if (view->union_members != nullptr) {
                                view->union_members = (uint8_t*)storage.address + ((const uint8_t*)view->union_members -
                                                                                   (const uint8_t*)context->views);
                            }
                        }
                        dora_allocation_release(&context->views_storage);
                        context->views_storage = storage;
                        context->views = storage.address;
                        const size_t available = storage.capacity_bytes / sizeof(struct dora_view_buffer);
                        context->view_capacity = available > UINT32_MAX ? UINT32_MAX : (uint32_t)available;
                    }
                }
                if (status == DORA_OK) {
                    struct vector_bound_node* const node = &context->foreign.nodes[position];
                    node->subtree_end_node_index = context->foreign.node_count;
                    status = dora_vector_admit_foreign(context->pool,
                                                       schema,
                                                       &input,
                                                       &context->views[context->view_count],
                                                       node,
                                                       context->error);
                    context->view_count = needed;
                }
            } else {
                ++context->adopted->claim_count;
                status = dora_vector_init_wrap_input(out_vector, context->pool, &input, context->error);
            }
        }
    }
    const bool device_only = context->adopted->owner.placement.access == DORA_ACCESS_DEVICE;
    const bool readable =
        !device_only || (context->adopted->access != nullptr && context->adopted->access->read_host != nullptr);
    if (status == DORA_OK && !context->should_copy && readable && field.type.id == DORA_TYPE_MAP &&
        field.layout != DORA_LAYOUT_DICTIONARY && field.layout != DORA_LAYOUT_RUN_END) {
        status = dora_vector_check_map_vector(context->pool, out_vector);
        if (status != DORA_OK) {
            dora_vector_release(out_vector);
        }
    }
    for (uint32_t child = 0; !context->should_copy && children != nullptr && child < field.child_count; ++child) {
        dora_vector_release(&children[child]);
    }
    dora_allocation_release(&children_storage);
    dora_allocation_release(&buffers_storage);
    return status;
}

int dora_arrow_import_array_impl(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 struct arrow_adopted_array** out_adopted,
                                 const struct ArrowDeviceArray* device,
                                 struct dora_arrow_device_access* access,
                                 const struct dora_arrow_limits* limits,
                                 struct arrow_traversal_state* state,
                                 bool should_copy,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    if (!dora_vector_is_empty(out_vector) || !dora_schema_is_valid(schema) || source == nullptr ||
        source->release == nullptr || (device != nullptr && (access == nullptr || access->describe == nullptr))) {
        return EINVAL;
    }
    if (out_adopted != nullptr) {
        *out_adopted = nullptr;
    }
    // Device callbacks require import-pool reset and release to stay blocked.
    dora_pool_callback_enter(pool);
    struct dora_schema retained_schema = {};
    int status = dora_schema_retain(schema, &retained_schema);
    schema = &retained_schema;
    struct array_import_context context =
        {.pool = pool, .limits = *limits, .traversal = state, .error = out_error, .should_copy = should_copy};
    dora_arrow_traversal_rewind(state);
    struct vector_bound_node inline_nodes[8];
    struct dora_view_buffer inline_views[32];
    context.foreign.nodes = inline_nodes;
    context.node_capacity = 8;
    context.views = inline_views;
    context.view_capacity = 32;
    struct arrow_adopted_array local_adopted = {};
    struct dora_allocation storage = {};
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context.limits, context.traversal, 0, sizeof(struct arrow_adopted_array));
    }
    if (status == DORA_OK && !should_copy) {
        status = dora_memory_payload_allocate(pool, sizeof(struct arrow_adopted_array), &storage);
    }
    if (status == DORA_OK) {
        context.adopted = should_copy ? &local_adopted : storage.address;
        *context.adopted = (struct arrow_adopted_array){
            .self_storage = storage,
            .access = access,
            .claim_count = 1,
            .owner = {
                .placement = {.device = {.type = DORA_DEVICE_TYPE_CPU}, .access = DORA_ACCESS_HOST},
                .release = array_owner_release,
                .read_host = array_owner_read
            }
        };
        if (device != nullptr) {
            status =
                access->describe(access, device, &context.adopted->owner.placement, &context.adopted->owner.readiness);
            if (status == DORA_OK && (context.adopted->owner.placement.device.type != device->device_type ||
                                      context.adopted->owner.placement.device.id != device->device_id ||
                                      context.adopted->owner.placement.access == DORA_ACCESS_NONE ||
                                      context.adopted->owner.placement.access > DORA_ACCESS_HOST_AND_DEVICE ||
                                      (context.adopted->owner.readiness.event != nullptr &&
                                       context.adopted->owner.readiness.settle == nullptr))) {
                status = EINVAL;
            }
        }
    }
    struct dora_vector candidate = {};
    if (status == DORA_OK) {
        status = array_import(&context, schema, source, 0, nullptr, &candidate);
    }
    if (status == DORA_OK && should_copy) {
        status =
            dora_vector_copy_foreign(pool, schema, &context.foreign, 0, &context.adopted->owner, &candidate, out_error);
    }
    dora_allocation_release(&context.views_storage);
    dora_allocation_release(&context.nodes_storage);
    if (context.adopted != nullptr) {
        array_owner_release(&context.adopted->owner);
    }
    dora_schema_release(&retained_schema);
    status = vector_finish_publication(&candidate, out_vector, status);
    if (status == DORA_OK && !should_copy && out_adopted != nullptr) {
        *out_adopted = context.adopted;
    }
    dora_pool_callback_leave(pool);
    if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
        error_message_copy_bounded(out_error, "Arrow array admission failed");
    }
    return status;
}

static int import_array(struct dora_pool pool,
                        const struct dora_schema* schema,
                        const struct ArrowArray* source,
                        struct ArrowArray* move_source,
                        const struct ArrowDeviceArray* device,
                        struct dora_arrow_device_access* access,
                        const struct dora_arrow_limits* limits,
                        bool should_copy,
                        struct dora_vector* out_vector,
                        struct dora_error* out_error) {
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    if (!dora_vector_is_empty(out_vector)) {
        return EINVAL;
    }
    struct dora_vector candidate = {};
    struct arrow_adopted_array* adopted = nullptr;
    const struct dora_arrow_limits resolved = dora_arrow_resolve_limits(limits);
    struct arrow_traversal_state state = {};
    dora_arrow_traversal_init(pool, &resolved, &state);
    int status = dora_arrow_import_array_impl(pool,
                                              schema,
                                              source,
                                              move_source != nullptr ? &adopted : nullptr,
                                              device,
                                              access,
                                              &resolved,
                                              &state,
                                              should_copy,
                                              &candidate,
                                              out_error);
    dora_allocation_release(&state.stack_storage);
    if (status == DORA_OK && !dora_vector_is_empty(out_vector)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        if (move_source != nullptr) {
            adopted->array = *move_source;
            move_source->release = nullptr;
        }
        *out_vector = candidate;
    } else {
        dora_vector_release(&candidate);
    }
    if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
        error_message_copy_bounded(out_error, "Arrow array admission failed");
    }
    return status;
}

int dora_arrow_import_copy_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    return import_array(pool, schema, source, nullptr, nullptr, nullptr, limits, true, out_vector, out_error);
}

int dora_arrow_import_move_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 struct ArrowArray* in_out_source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    return import_array(pool,
                        schema,
                        in_out_source,
                        in_out_source,
                        nullptr,
                        nullptr,
                        limits,
                        false,
                        out_vector,
                        out_error);
}

int dora_arrow_import_wrap_array(struct dora_pool pool,
                                 const struct dora_schema* schema,
                                 const struct ArrowArray* source,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    return import_array(pool, schema, source, nullptr, nullptr, nullptr, limits, false, out_vector, out_error);
}

int dora_arrow_import_copy_device_array(struct dora_pool pool,
                                        const struct dora_schema* schema,
                                        const struct ArrowDeviceArray* source,
                                        struct dora_arrow_device_access* access,
                                        const struct dora_arrow_limits* limits,
                                        struct dora_vector* out_vector,
                                        struct dora_error* out_error) {
    return source == nullptr ? EINVAL
                             : import_array(pool,
                                            schema,
                                            &source->array,
                                            nullptr,
                                            source,
                                            access,
                                            limits,
                                            true,
                                            out_vector,
                                            out_error);
}

int dora_arrow_import_move_device_array(struct dora_pool pool,
                                        const struct dora_schema* schema,
                                        struct ArrowDeviceArray* in_out_source,
                                        struct dora_arrow_device_access* access,
                                        const struct dora_arrow_limits* limits,
                                        struct dora_vector* out_vector,
                                        struct dora_error* out_error) {
    return in_out_source == nullptr ? EINVAL
                                    : import_array(pool,
                                                   schema,
                                                   &in_out_source->array,
                                                   &in_out_source->array,
                                                   in_out_source,
                                                   access,
                                                   limits,
                                                   false,
                                                   out_vector,
                                                   out_error);
}
