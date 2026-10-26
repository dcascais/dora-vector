// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
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
#include "hash/hash.h"
#include "hash/keyset.h"
#include "identity/identity.h"
#include "kernel/binding/validate.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"
#include "types/load.h"
#include "types/order.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int extent_check(struct dora_view_buffer buffer, uint64_t count, uint32_t width) {
    uint64_t bytes = 0;
    if (ckd_mul(&bytes, count, width)) {
        return EINVAL;
    }
    if (bytes > SIZE_MAX) {
        return EOVERFLOW;
    }
    return bytes <= buffer.byte_count && (bytes == 0 || buffer.data != nullptr) ? DORA_OK : EINVAL;
}

static bool owner_allows_direct_host_access(const struct dora_buffer_owner* owner) {
    return owner == nullptr || owner->read_host == nullptr || owner->placement.access == DORA_ACCESS_HOST ||
           owner->placement.access == DORA_ACCESS_HOST_AND_DEVICE;
}

static int input_read(struct dora_buffer_owner* owner,
                      struct dora_view_buffer buffer,
                      struct dora_range bytes,
                      void* destination) {
    if (bytes.offset > buffer.byte_count || bytes.length > buffer.byte_count - bytes.offset) {
        return EINVAL;
    }
    if (!owner_allows_direct_host_access(owner)) {
        const struct dora_device_region region =
            {.data = buffer.data, .byte_offset = bytes.offset, .byte_count = bytes.length};
        return owner->read_host(owner, region, destination);
    }
    if (bytes.length != 0) {
        memcpy(destination, (const uint8_t*)buffer.data + bytes.offset, (size_t)bytes.length);
    }
    return DORA_OK;
}

static int input_integer(struct dora_buffer_owner* owner,
                         struct dora_view_buffer buffer,
                         uint64_t row,
                         uint32_t width,
                         int64_t* out_value) {
    int32_t small = 0;
    int64_t large = 0;
    const int status = input_read(owner,
                                  buffer,
                                  (struct dora_range){.offset = row * width, .length = width},
                                  width == 4 ? (void*)&small : (void*)&large);
    if (status == DORA_OK) {
        *out_value = width == 4 ? small : large;
    }
    return status;
}

static int
input_null(struct dora_buffer_owner* owner, const struct dora_vector_view* view, uint64_t row, bool* out_is_null) {
    *out_is_null = view->layout == DORA_LAYOUT_NULL;
    if (view->validity.data == nullptr || view->null_count == 0) {
        return DORA_OK;
    }
    const uint64_t bit = view->validity_bit_offset + row;
    uint8_t byte = 0;
    const int status =
        input_read(owner, view->validity, (struct dora_range){.offset = bitmap_byte_of(bit), .length = 1}, &byte);
    if (status == DORA_OK) {
        *out_is_null = !dora_bitmap_test(&byte, bitmap_bit_in_byte(bit));
    }
    return status;
}

static struct dora_buffer_owner* input_owner(const struct vector_input* input, const void* data) {
    return data == input->data_buffer_sizes ? nullptr : input->owner;
}

static int input_stage(struct dora_pool pool,
                       struct dora_buffer_owner* owner,
                       uint64_t byte_count,
                       struct dora_view_buffer* in_out_buffer,
                       struct dora_allocation* out_copy) {
    if (byte_count == 0 || in_out_buffer->data == nullptr || owner_allows_direct_host_access(owner)) {
        return DORA_OK;
    }
    if (byte_count > SIZE_MAX) {
        return EOVERFLOW;
    }
    int status = dora_memory_payload_allocate(pool, (size_t)byte_count, out_copy);
    if (status == DORA_OK) {
        status = input_read(owner, *in_out_buffer, (struct dora_range){.length = byte_count}, out_copy->address);
    }
    if (status == DORA_OK) {
        in_out_buffer->data = out_copy->address;
    }
    return status;
}

static int admission_scan(struct kernel_validate_binding* binding, uint64_t row_count) {
    uint64_t first = 0;
    int status = DORA_OK;
    do {
        const uint32_t count =
            row_count - first < KERNEL_CHUNK_ROW_COUNT ? (uint32_t)(row_count - first) : KERNEL_CHUNK_ROW_COUNT;
        uint64_t bad_row = 0;
        status = binding->check(binding, first, count, &bad_row);
        first += count;
    } while (status == DORA_OK && first < row_count);
    return status;
}

static int offsets_check(struct dora_pool pool,
                         const struct vector_input* input,
                         const struct dora_vector_view* view,
                         uint64_t limit,
                         bool list_view) {
    const uint32_t width = view->layout == DORA_LAYOUT_LARGE_BINARY || view->layout == DORA_LAYOUT_LARGE_LIST ||
                                   view->layout == DORA_LAYOUT_LARGE_LIST_VIEW
                               ? 8
                               : 4;
    uint64_t endpoint = view->offset + view->length;
    if (!list_view) {
        ++endpoint;
    }
    int status = extent_check(view->offsets, endpoint, width);
    if (status == DORA_OK && list_view) {
        status = extent_check(view->sizes, endpoint, width);
    }
    if (status != DORA_OK || !input->should_scan_geometry) {
        return status;
    }
    struct dora_vector_view host = *view;
    struct dora_allocation copies[2] = {};
    status = input_stage(pool, input_owner(input, host.offsets.data), endpoint * width, &host.offsets, &copies[0]);
    if (status == DORA_OK && list_view) {
        status = input_stage(pool, input_owner(input, host.sizes.data), endpoint * width, &host.sizes, &copies[1]);
    }
    const struct dora_vector_view child = {.length = limit};
    struct kernel_validate_binding binding = {};
    if (status == DORA_OK) {
        status = dora_kernel_validate_bind(&host, nullptr, &child, &binding);
    }
    if (status == DORA_OK) {
        status = admission_scan(&binding, endpoint - view->offset);
    }
    dora_allocation_release(&copies[1]);
    dora_allocation_release(&copies[0]);
    return status;
}

static int input_scan(struct dora_pool pool,
                      const struct dora_vector_view* view,
                      const struct vector_input* input,
                      const struct dora_vector_view* child) {
    struct dora_vector_view host = *view;
    if (host.null_count == 0) {
        host.validity = (struct dora_view_buffer){};
    }
    struct dora_allocation copies[2] = {};
    int status = input_stage(pool,
                             input_owner(input, host.values.data),
                             (view->offset + view->length) * view->byte_width,
                             &host.values,
                             &copies[0]);
    if (status == DORA_OK) {
        status = input_stage(pool,
                             input_owner(input, host.validity.data),
                             bitmap_bytes_for_bits(host.validity_bit_offset + host.length),
                             &host.validity,
                             &copies[1]);
    }
    struct kernel_validate_binding binding = {};
    if (status == DORA_OK) {
        status = dora_kernel_validate_bind(&host, nullptr, child, &binding);
    }
    if (status == DORA_OK) {
        status = admission_scan(&binding, view->length);
    }
    dora_allocation_release(&copies[1]);
    dora_allocation_release(&copies[0]);
    return status;
}

static int
binary_views_check(struct dora_pool pool, const struct vector_input* input, const struct dora_vector_view* view) {
    int status = extent_check(view->values, view->offset + view->length, sizeof(struct dora_binary_view_entry));
    if (status != DORA_OK || !input->should_scan_geometry) {
        return status;
    }
    struct dora_vector_view entries = *view;
    entries.byte_width = sizeof(struct dora_binary_view_entry);
    return input_scan(pool, &entries, input, nullptr);
}

static uint64_t input_child_length(const struct vector_bound_input* source, uint32_t root, uint32_t ordinal) {
    return source->nodes[dora_vector_input_child(root, source, ordinal)].view.length;
}

static int union_check(struct dora_pool pool,
                       const struct vector_input* input,
                       const struct dora_vector_view* view,
                       const struct vector_bound_input* source,
                       uint32_t root) {
    int status = extent_check(view->type_ids, view->offset + view->length, 1);
    if (status == DORA_OK && view->layout == DORA_LAYOUT_DENSE_UNION) {
        status = extent_check(view->offsets, view->offset + view->length, 4);
    }
    if (status != DORA_OK) {
        return status;
    }
    const uint64_t endpoint = view->offset + view->length;
    struct dora_vector_view host = *view;
    struct dora_allocation copies[3] = {};
    status = input_stage(pool, input_owner(input, host.type_ids.data), endpoint, &host.type_ids, &copies[0]);
    if (status == DORA_OK && view->layout == DORA_LAYOUT_DENSE_UNION) {
        status = input_stage(pool,
                             input_owner(input, host.offsets.data),
                             endpoint * sizeof(int32_t),
                             &host.offsets,
                             &copies[1]);
    }
    if (status == DORA_OK && input->child_count != 0) {
        status = dora_memory_payload_allocate(pool,
                                              (size_t)input->child_count * sizeof(struct dora_vector_view),
                                              &copies[2]);
    }
    struct dora_vector_view none = {};
    struct dora_vector_view* const members = input->child_count == 0 ? &none : copies[2].address;
    for (uint32_t index = 0; status == DORA_OK && index < input->child_count; ++index) {
        members[index] = (struct dora_vector_view){.length = input_child_length(source, root, index)};
    }
    struct kernel_validate_binding binding = {};
    int32_t union_offsets[KERNEL_UNION_TYPE_ID_COUNT] = {};
    if (status == DORA_OK) {
        status = dora_kernel_validate_bind(&host, nullptr, members, &binding);
        binding.union_offsets = union_offsets;
    }
    if (status == DORA_OK) {
        status = admission_scan(&binding, view->length);
    }
    for (uint32_t index = 3; index != 0; --index) {
        dora_allocation_release(&copies[index - 1]);
    }
    return status;
}

static int dictionary_check(struct dora_pool pool,
                            const struct vector_input* input,
                            const struct dora_vector_view* view,
                            const struct vector_bound_input* source,
                            uint32_t root) {
    const uint32_t width = view->byte_width;
    int status = extent_check(view->values, view->offset + view->length, width);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_vector_view values = {.length = input_child_length(source, root, 0)};
    return input_scan(pool, view, input, &values);
}

struct admission_child {
    struct dora_vector_view view;
    struct dora_buffer_owner* owner;
    const struct vector_bound_input* tree;
    uint32_t node;
    bool has_host_validity;
};

static int child_provider(const struct vector_image* image, struct admission_child* out) {
    const struct vector_image* provider = image;
    for (;;) {
        out->has_host_validity = out->has_host_validity || dora_vector_owns_host_validity(provider, out->view.validity);
        if (out->owner == nullptr) {
            out->owner = provider->buffer_image->wrapped_owner;
        }
        if (provider->ownership.backing.handle.slot_index == 0) {
            return DORA_OK;
        }
        struct vector_image* backing = nullptr;
        const int status = dora_vector_resolve(&provider->ownership.backing, &backing);
        provider = backing;
        if (status != DORA_OK) {
            return status;
        }
    }
}

static void input_child(const struct admission_child* parent, uint32_t ordinal, struct admission_child* out) {
    out->owner = parent->owner;
    out->tree = parent->tree;
    out->node = dora_vector_input_child(parent->node, parent->tree, ordinal);
    out->view = parent->tree->nodes[out->node].view;
    out->has_host_validity = false;
}

static int child_bind(const struct admission_child* parent, uint32_t ordinal, struct admission_child* out) {
    input_child(parent, ordinal, out);
    const struct vector_bound_node* const node = &out->tree->nodes[out->node];
    if (node->source == nullptr) {
        return DORA_OK;
    }
    if (node->image->ownership.requires_effective_validity) {
        return ENOTSUP;
    }
    int status = dora_vector_settle(node->source);
    if (status == DORA_OK) {
        out->owner = nullptr;
        status = child_provider(node->image, out);
    }
    return status;
}

static int child_read(const struct admission_child* child,
                      struct dora_view_buffer buffer,
                      struct dora_range bytes,
                      void* destination) {
    if (child->view.placement.access == DORA_ACCESS_DEVICE &&
        (child->owner == nullptr || child->owner->read_host == nullptr)) {
        return ENOTSUP;
    }
    return input_read(child->owner, buffer, bytes, destination);
}

static int child_integer(const struct admission_child* child,
                         struct dora_view_buffer buffer,
                         uint64_t row,
                         uint32_t width,
                         int64_t* out_value) {
    int32_t small = 0;
    int64_t large = 0;
    const int status = child_read(child,
                                  buffer,
                                  (struct dora_range){.offset = row * width, .length = width},
                                  width == 4 ? (void*)&small : (void*)&large);
    if (status == DORA_OK) {
        *out_value = width == 4 ? small : large;
    }
    return status;
}

static int child_null(const struct admission_child* child, uint64_t row, bool* out_null) {
    struct dora_buffer_owner* const owner = child->has_host_validity ? nullptr : child->owner;
    if (child->view.null_count != 0 && child->view.validity.data != nullptr && !child->has_host_validity &&
        child->view.placement.access == DORA_ACCESS_DEVICE && (owner == nullptr || owner->read_host == nullptr)) {
        return ENOTSUP;
    }
    const uint64_t logical = dora_vector_view_get_physical_row(&child->view, row) - child->view.offset;
    return input_null(owner, &child->view, logical, out_null);
}

static int child_index(const struct admission_child* child, uint64_t row, uint64_t* out_index) {
    const uint32_t width = child->view.byte_width;
    uint64_t value = 0;
    const uint64_t physical = dora_vector_view_get_physical_row(&child->view, row);
    const int status =
        child_read(child, child->view.values, (struct dora_range){.offset = physical * width, .length = width}, &value);
    if (status != DORA_OK) {
        return status;
    }
    switch (child->view.physical_type) {
// Returns DORA_OK or EINVAL from the enclosing function.
// Index carriers: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_INDEX(S, T, P, NEGATIVE)                                                                                \
    case P: {                                                                                                          \
        const T index = type_load_##S((const uint8_t*)&value, 0);                                                      \
        if (NEGATIVE(index)) {                                                                                         \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        *out_index = (uint64_t)index;                                                                                  \
        return DORA_OK;                                                                                                \
    }
        SIGNED_INDEX_CARRIERS(DEFINE_INDEX, NEGATIVE_SIGNED)
        UNSIGNED_INDEX_CARRIERS(DEFINE_INDEX, NEGATIVE_UNSIGNED)
#undef DEFINE_INDEX
    default:
        return EINVAL;
    }
}

static int child_find_run(const struct admission_child* child, uint64_t row, uint64_t* out_run) {
    int status = DORA_OK;
    uint64_t first = 0;
    uint64_t last = child->view.length;
    while (status == DORA_OK && first < last) {
        const uint64_t middle = first + (last - first) / 2;
        uint64_t endpoint = 0;
        status = child_index(child, middle, &endpoint);
        if (endpoint <= row) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    if (status == DORA_OK && first == child->view.length) {
        return EINVAL;
    }
    if (status == DORA_OK) {
        *out_run = first;
    }
    return status;
}

static int child_logical_null(struct admission_child bound, uint64_t row, bool* out_null) {
    for (;;) {
        if (row >= bound.view.length) {
            return EINVAL;
        }
        int status = child_null(&bound, row, out_null);
        if (status != DORA_OK || *out_null) {
            return status;
        }
        uint32_t ordinal = 0;
        uint64_t child_row = 0;
        const uint64_t physical = dora_vector_view_get_physical_row(&bound.view, row);
        switch (bound.view.layout) {
        case DORA_LAYOUT_DICTIONARY:
            status = child_index(&bound, row, &child_row);
            break;
        case DORA_LAYOUT_RUN_END: {
            struct admission_child ends = {};
            status = child_bind(&bound, 0, &ends);
            if (status == DORA_OK) {
                status = child_find_run(&ends, physical, &child_row);
            }
            ordinal = 1;
            break;
        }
        case DORA_LAYOUT_SPARSE_UNION:
        case DORA_LAYOUT_DENSE_UNION: {
            uint8_t code = 0;
            status =
                child_read(&bound, bound.view.type_ids, (struct dora_range){.offset = physical, .length = 1}, &code);
            if (status == DORA_OK && code > INT8_MAX) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                ordinal = bound.view.union_members[code];
            }
            child_row = physical;
            if (status == DORA_OK && bound.view.layout == DORA_LAYOUT_DENSE_UNION) {
                int64_t member = 0;
                status = child_integer(&bound, bound.view.offsets, physical, 4, &member);
                child_row = (uint64_t)member;
            }
            break;
        }
        default:
            return DORA_OK;
        }
        if (status != DORA_OK) {
            return status;
        }
        struct admission_child child = {};
        status = child_bind(&bound, ordinal, &child);
        if (status != DORA_OK) {
            return status;
        }
        bound = child;
        row = child_row;
    }
}

static int run_ends_check(struct dora_pool pool, const struct admission_child* bound, struct dora_vector_view* view) {
    const struct vector_bound_input* const source = bound->tree;
    const uint32_t root = bound->node;
    const uint64_t count = input_child_length(source, root, 0);
    if ((count == 0) != (view->length == 0) || count > input_child_length(source, root, 1)) {
        return EINVAL;
    }
    struct admission_child child = {};
    int status = child_bind(bound, 0, &child);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_vector_view runs = child.view;
    if (runs.layout != DORA_LAYOUT_FIXED || runs.selection.indices != nullptr) {
        return EINVAL;
    }
    view->physical_type = runs.physical_type;
    view->placement = runs.placement;
    view->readiness = runs.readiness;
    view->byte_width = physical_width(runs.physical_type);
    uint64_t run_extent = 0;
    size_t run_bytes = 0;
    if (ckd_add(&run_extent, runs.offset, runs.length)) {
        return EINVAL;
    }
    if (run_extent > SIZE_MAX || ckd_mul(&run_bytes, (size_t)run_extent, view->byte_width)) {
        return EOVERFLOW;
    }
    if (run_bytes > runs.values.byte_count) {
        return EINVAL;
    }
    view->run_end_element_offset = runs.offset;
    view->values = runs.values;
    view->values.byte_count = run_bytes;
    const uint64_t maximum = runs.physical_type == DORA_PHYSICAL_TYPE_I16   ? INT16_MAX
                             : runs.physical_type == DORA_PHYSICAL_TYPE_I32 ? INT32_MAX
                                                                            : INT64_MAX;
    if (view->offset + view->length > maximum) {
        return EINVAL;
    }
    struct dora_vector_view host = runs;
    if (host.null_count == 0) {
        host.validity = (struct dora_view_buffer){};
    }
    struct dora_buffer_owner* const validity_owner = child.has_host_validity ? nullptr : child.owner;
    const bool device = runs.placement.access == DORA_ACCESS_DEVICE;
    if ((device && host.validity.data != nullptr && !child.has_host_validity &&
         (validity_owner == nullptr || validity_owner->read_host == nullptr)) ||
        (device && runs.length != 0 && (child.owner == nullptr || child.owner->read_host == nullptr))) {
        return ENOTSUP;
    }
    struct dora_allocation copies[2] = {};
    status = input_stage(pool, child.owner, run_bytes, &host.values, &copies[0]);
    if (status == DORA_OK) {
        status = input_stage(pool,
                             validity_owner,
                             bitmap_bytes_for_bits(host.validity_bit_offset + host.length),
                             &host.validity,
                             &copies[1]);
    }
    struct kernel_validate_binding binding = {};
    if (status == DORA_OK) {
        status = dora_kernel_validate_bind(view, nullptr, &host, &binding);
    }
    if (status == DORA_OK) {
        status = admission_scan(&binding, runs.length);
    }
    dora_allocation_release(&copies[1]);
    dora_allocation_release(&copies[0]);
    return status;
}

static int ordinal_dictionary(struct dora_pool pool,
                              const struct dora_vector* source,
                              struct dora_vector* out_dictionary,
                              struct dora_error* out_error) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, source->rows.length, sizeof(uint64_t))) {
        return EOVERFLOW;
    }
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(source, &type);
    struct vector_image* source_image = nullptr;
    if (status == DORA_OK) {
        status = dora_vector_resolve(source, &source_image);
    }
    struct dora_allocation buffers[2] = {};
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, &buffers[1]);
    }
    if (status == DORA_OK) {
        for (uint64_t row = 0; row < source->rows.length; ++row) {
            ((uint64_t*)buffers[1].address)[row] = row;
        }
        const struct dora_schema_field field =
            {.type = type, .layout = DORA_LAYOUT_DICTIONARY, .index_type = DORA_TYPE_UINT64, .child_count = 1};
        const struct vector_input input = {
            .field = &field,
            .row_count = source->rows.length,
            .allocations = buffers,
            .buffer_count = 2,
            .children = source,
            .child_count = 1,
            .content_state = source_image->node.content_state
        };
        status = dora_vector_init_move_input(out_dictionary, pool, &input, buffers, out_error);
    }
    dora_allocation_release(&buffers[1]);
    return status;
}

static int gather_selected_dictionary(struct vector_image* image, const struct dora_vector_view* source) {
    struct vector_node_ownership* const owned = &image->ownership;
    size_t value_bytes = 0;
    size_t mask_offset = 0;
    size_t mask_bytes = 0;
    size_t bytes = 0;
    if (source->length > SIZE_MAX || ckd_mul(&value_bytes, (size_t)source->length, source->byte_width)) {
        return EOVERFLOW;
    }
    const bool has_mask = owned->own_null_count != 0;
    if (has_mask) {
        const uint64_t words = bitmap_words_for_bits(source->length);
        if (words > SIZE_MAX || ckd_add(&mask_offset, value_bytes, sizeof(uint64_t) - 1) ||
            ckd_mul(&mask_bytes, (size_t)words, sizeof(uint64_t))) {
            return EOVERFLOW;
        }
        mask_offset &= ~(sizeof(uint64_t) - 1);
        if (source->length != 0 && owned->own_validity.data == nullptr) {
            return EINVAL;
        }
    } else {
        mask_offset = value_bytes;
    }
    if (ckd_add(&bytes, mask_offset, mask_bytes)) {
        return EOVERFLOW;
    }
    int status = dora_memory_payload_allocate(image->allocation.pool, bytes, &owned->selection_storage);
    if (status != DORA_OK) {
        return status;
    }
    if (bytes != 0) {
        memset(owned->selection_storage.address, 0, mask_offset);
        if (has_mask) {
            bitmap_fill_bytes((uint8_t*)owned->selection_storage.address + mask_offset, mask_bytes, false);
        }
    }
    uint8_t* const values = owned->selection_storage.address;
    uint64_t* const mask = has_mask && mask_bytes != 0 ? (uint64_t*)(values + mask_offset) : nullptr;
    int64_t null_count = 0;
    for (uint64_t row = 0; row < source->length; ++row) {
        const uint64_t physical = dora_vector_view_get_physical_row(source, row);
        uint64_t bit = 0;
        if (has_mask && (ckd_add(&bit, owned->own_validity_bit_offset, physical - source->offset) ||
                         bitmap_byte_of(bit) >= owned->own_validity.byte_count)) {
            return EINVAL;
        }
        const bool valid = !has_mask || dora_bitmap_test(owned->own_validity.data, bit);
        if (has_mask) {
            bitmap_assign(mask, row, valid);
        }
        if (valid) {
            uint64_t byte = 0;
            if (ckd_mul(&byte, physical, source->byte_width) || byte > source->values.byte_count ||
                source->byte_width > source->values.byte_count - byte) {
                return EINVAL;
            }
            memcpy(values + row * source->byte_width, (const uint8_t*)source->values.data + byte, source->byte_width);
        } else {
            ++null_count;
        }
    }
    owned->own_validity =
        (struct dora_view_buffer){.data = mask, .byte_count = mask_bytes, .readable_byte_count = mask_bytes};
    owned->own_validity_bit_offset = 0;
    owned->own_null_count = null_count;
    owned->view.values =
        (struct dora_view_buffer){.data = values, .byte_count = value_bytes, .readable_byte_count = mask_offset};
    owned->view.selection = (struct dora_vector_selection){};
    owned->view.offset = 0;
    owned->view.validity = owned->own_validity;
    owned->view.validity_bit_offset = 0;
    owned->view.null_count = null_count;
    owned->view.placement = (struct dora_placement){.device.type = DORA_DEVICE_TYPE_CPU, .access = DORA_ACCESS_HOST};
    owned->view.readiness = (struct dora_readiness){};
    return DORA_OK;
}

int dora_vector_defer_validity(struct vector_image* image,
                               const struct dora_vector* source,
                               const struct dora_vector* ancestor,
                               uint64_t row_count,
                               uint64_t** out_selection) {
    size_t bytes = sizeof(struct vector_deferred_validity);
    size_t selection_bytes = 0;
    if (out_selection != nullptr &&
        (row_count > SIZE_MAX || ckd_mul(&selection_bytes, (size_t)row_count, sizeof(uint64_t)) ||
         ckd_add(&bytes, bytes, selection_bytes))) {
        return EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    int status = dora_memory_payload_allocate(image->allocation.pool, bytes, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_deferred_validity* const deferred = allocation.address;
    *deferred = (struct vector_deferred_validity){.allocation = allocation};
    image->ownership.deferred_validity = deferred;
    if (out_selection != nullptr) {
        *out_selection = (uint64_t*)(deferred + 1);
        deferred->selection = *out_selection;
    }
    const struct dora_vector* const inputs[2] = {source, ancestor};
    for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
        if (inputs[index] == nullptr) {
            continue;
        }
        status = dora_identity_claim_acquire(image->allocation.pool,
                                             &inputs[index]->handle,
                                             IDENTITY_KIND_VECTOR,
                                             &deferred->inputs[index].handle);
        deferred->inputs[index].rows = inputs[index]->rows;
    }
    if (status == DORA_OK) {
        image->ownership.requires_effective_validity = true;
        image->ownership.view.validity_bit_offset = 0;
        image->ownership.view.null_count = -1;
    }
    return status;
}

static int freeze_parent(struct vector_image** in_out_image, struct dora_error* out_error) {
    struct vector_image* const original = *in_out_image;
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(&original->schema, &field);
    struct dora_vector source = {};
    if (status == DORA_OK) {
        status = dora_vector_publish(original, &source);
    }
    if (status == DORA_OK) {
        *in_out_image = nullptr;
        status = dora_vector_image_share(original->allocation.pool, &source, &field, in_out_image, out_error);
    }
    dora_vector_release(&source);
    return status;
}

int dora_vector_derive_children(struct vector_image** in_out_image,
                                bool are_children_folded,
                                struct dora_error* out_error) {
    struct vector_image* image = *in_out_image;
    const struct dora_vector_view* parent = &image->ownership.view;
    if (parent->layout != DORA_LAYOUT_STRUCT ||
        (parent->null_count == 0 && !image->ownership.requires_effective_validity)) {
        return DORA_OK;
    }
    const bool deferred_parent =
        image->ownership.requires_effective_validity ||
        (parent->placement.access != DORA_ACCESS_HOST && parent->placement.access != DORA_ACCESS_HOST_AND_DEVICE);
    bool needs_raw_parent = deferred_parent;
    for (uint32_t index = 0; !needs_raw_parent && index < image->child_vectors.entry_count; ++index) {
        struct vector_image* child = nullptr;
        const int status = dora_vector_resolve(&dora_vector_image_children(image)[index], &child);
        if (status != DORA_OK) {
            return status;
        }
        needs_raw_parent = child->ownership.requires_effective_validity ||
                           child->ownership.view.placement.access == DORA_ACCESS_DEVICE;
    }
    if (needs_raw_parent) {
        const int status = freeze_parent(in_out_image, out_error);
        if (status != DORA_OK) {
            return status;
        }
        image = *in_out_image;
        parent = &image->ownership.view;
    }
    struct dora_vector* const children = dora_vector_image_children(image);
    for (uint32_t index = 0; index < image->child_vectors.entry_count; ++index) {
        if (are_children_folded) {
            struct vector_image* folded = nullptr;
            const int folded_status = dora_vector_resolve(&children[index], &folded);
            if (folded_status != DORA_OK) {
                return folded_status;
            }
            const enum dora_layout layout = folded->ownership.view.layout;
            if (layout != DORA_LAYOUT_DENSE_UNION && layout != DORA_LAYOUT_SPARSE_UNION &&
                layout != DORA_LAYOUT_RUN_END) {
                continue;
            }
        }
        struct dora_vector source = {};
        int status = dora_vector_slice(&children[index],
                                       (struct dora_range){.offset = image->node.offset, .length = parent->length},
                                       &source);
        struct vector_image* source_image = nullptr;
        if (status == DORA_OK) {
            status = dora_vector_resolve(&source, &source_image);
        }
        struct dora_schema_field field = {};
        if (status == DORA_OK) {
            status = dora_schema_get_field(&source_image->schema, &field);
        }
        const struct dora_logical_type logical_type = field.type;
        const struct dora_schema annotated_schema =
            status == DORA_OK ? dora_schema_borrow(&source_image->schema) : (struct dora_schema){};
        if (status == DORA_OK && logical_type.extension_name.byte_count != 0) {
            struct dora_vector storage = {};
            status = dora_vector_get_storage(&source, &storage);
            if (status == DORA_OK) {
                source = storage;
                status = dora_vector_resolve(&source, &source_image);
            }
            if (status == DORA_OK) {
                status = dora_schema_get_field(&source_image->schema, &field);
            }
        }
        struct dora_vector_view original_view = {};
        bool deferred = deferred_parent;
        if (status == DORA_OK) {
            deferred = deferred || source_image->ownership.requires_effective_validity ||
                       source_image->ownership.view.placement.access == DORA_ACCESS_DEVICE;
            status = deferred ? dora_vector_bind_coordinates(&source, &original_view)
                              : dora_vector_get_view(&source, &original_view);
        }
        const struct dora_vector original_source = source;
        struct dora_vector indexed = {};
        if (status == DORA_OK && original_view.selection.indices != nullptr &&
            original_view.layout != DORA_LAYOUT_DICTIONARY) {
            status = ordinal_dictionary(image->allocation.pool, &source, &indexed, out_error);
            if (status == DORA_OK) {
                source = dora_vector_borrow(&indexed);
                field.layout = DORA_LAYOUT_DICTIONARY;
                field.index_type = DORA_TYPE_UINT64;
                field.child_count = 1;
            }
        }
        struct vector_image* child_image = nullptr;
        if (status == DORA_OK) {
            status = dora_vector_image_share(image->allocation.pool, &source, &field, &child_image, out_error);
        }
        dora_vector_release(&indexed);
        if (status != DORA_OK) {
            return status;
        }
        struct vector_node_ownership* const owned = &child_image->ownership;
        if (!deferred && original_view.selection.indices != nullptr && original_view.layout == DORA_LAYOUT_DICTIONARY) {
            status = gather_selected_dictionary(child_image, &original_view);
            if (status != DORA_OK) {
                dora_vector_discard(child_image);
                return status;
            }
        }
        if (deferred) {
            status = dora_vector_defer_validity(child_image, &original_source, &image->ownership.backing, 0, nullptr);
        } else {
            size_t bytes = 0;
            if (ckd_mul(&bytes, bitmap_words_for_bits(parent->length), sizeof(uint64_t))) {
                dora_vector_discard(child_image);
                return EOVERFLOW;
            }
            status = dora_memory_payload_allocate(image->allocation.pool, bytes, &owned->effective_validity);
            if (status != DORA_OK) {
                dora_vector_discard(child_image);
                return status;
            }
            if (bytes != 0) {
                bitmap_fill_bytes(owned->effective_validity.address, bytes, false);
            }
            const uint64_t words = bitmap_words_for_bits(parent->length);
            uint64_t* const effective = owned->effective_validity.address;
            for (uint64_t word = 0; word < words; ++word) {
                effective[word] = dora_vector_view_get_validity_word(parent, word) &
                                  dora_vector_view_get_validity_word(&original_view, word);
            }
            const int64_t null_count =
                (int64_t)(parent->length - dora_bitmap_count_set_range(effective, 0, parent->length));
            owned->view.validity = (struct dora_view_buffer){
                .data = owned->effective_validity.address,
                .byte_count = bytes,
                .readable_byte_count = owned->effective_validity.capacity_bytes
            };
            owned->view.validity_bit_offset = 0;
            owned->view.null_count = null_count;
            if (null_count == 0) {
                dora_allocation_release(&owned->effective_validity);
                owned->view.validity = (struct dora_view_buffer){};
            }
        }
        if (status == DORA_OK) {
            status = dora_vector_derive_children(&child_image, false, out_error);
        }
        if (status == DORA_OK) {
            status = dora_vector_bind_extension(child_image, out_error);
        }
        struct dora_vector derived = {};
        if (status == DORA_OK) {
            status = dora_vector_publish(child_image, &derived);
            if (status == DORA_OK) {
                child_image = nullptr;
            }
        }
        if (status == DORA_OK && logical_type.extension_name.byte_count != 0) {
            struct dora_vector extension = {};
            status = dora_vector_init_retain_extension_error(&extension,
                                                             image->allocation.pool,
                                                             logical_type.extension_name,
                                                             logical_type.extension_metadata,
                                                             &derived,
                                                             &annotated_schema,
                                                             out_error);
            if (status == DORA_OK) {
                dora_vector_release(&derived);
                derived = extension;
            }
        }
        if (status == DORA_OK) {
            status = dora_identity_claim_transfer(image->allocation.pool, &derived.handle, IDENTITY_KIND_VECTOR);
        }
        if (status == DORA_OK) {
            dora_vector_release(&children[index]);
            children[index] = derived;
            derived = (struct dora_vector){};
        }
        dora_vector_release(&derived);
        dora_vector_discard(child_image);
        if (status != DORA_OK) {
            return status;
        }
    }
    image->node.offset = 0;
    return dora_vector_refresh_schema(image);
}

static int activity_extent(const struct vector_input* input, uint64_t row_count) {
    if (input->ancestor_activity == nullptr) {
        return input->activity_bit_offset == 0 ? DORA_OK : EINVAL;
    }
    uint64_t end = 0;
    if (ckd_add(&end, input->activity_bit_offset, row_count)) {
        return EINVAL;
    }
    const uint64_t bytes = bitmap_bytes_for_bits(end);
    if (bytes > SIZE_MAX) {
        return EOVERFLOW;
    }
    return bytes <= input->ancestor_activity->byte_count &&
                   (input->ancestor_activity->data != nullptr || input->ancestor_activity->byte_count == 0)
               ? DORA_OK
               : EINVAL;
}

static bool input_row_is_active(const struct vector_input* input, uint64_t row) {
    return input->ancestor_activity == nullptr ||
           dora_bitmap_test(input->ancestor_activity->data, input->activity_bit_offset + row);
}

static int input_activity(const struct vector_input* input, uint64_t row_count, bool* out_active) {
    const int status = activity_extent(input, row_count);
    if (status != DORA_OK) {
        return status;
    }
    *out_active = false;
    for (uint64_t row = 0; !*out_active && row < row_count; ++row) {
        *out_active = input_row_is_active(input, row);
    }
    return DORA_OK;
}

static bool node_may_be_null(const struct vector_bound_node* node) {
    return (node->image != nullptr && node->image->ownership.requires_effective_validity) ||
           node->view.null_count != 0 || vector_layout_derives_nulls(node->view.layout);
}

static bool map_needs_activity(const struct admission_child* bound, const struct dora_schema_field* field) {
    if (field->type.id != DORA_TYPE_MAP ||
        (field->layout != DORA_LAYOUT_LIST && field->layout != DORA_LAYOUT_LARGE_LIST)) {
        return false;
    }
    const uint32_t entries = dora_vector_input_child(bound->node, bound->tree, 0);
    return node_may_be_null(&bound->tree->nodes[dora_vector_input_child(entries, bound->tree, 0)]);
}

static int active_null_check(const struct admission_child* bound,
                             const struct vector_input* input,
                             const struct dora_schema_field* field) {
    const struct dora_vector_view* const view = &bound->view;
    const bool required =
        (field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0 && !input->are_required_rows_admitted;
    const bool map =
        !(input->are_map_rows_admitted && field->type.id == DORA_TYPE_MAP) && map_needs_activity(bound, field);
    if (!required && !map) {
        return DORA_OK;
    }
    if (required && !map && view->null_count == 0 && !vector_layout_derives_nulls(view->layout)) {
        return DORA_OK;
    }
    struct admission_child keys = {};
    if (map) {
        struct admission_child entries = {};
        int status = child_bind(bound, 0, &entries);
        if (status == DORA_OK) {
            status = child_bind(&entries, 0, &keys);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    for (uint64_t row = 0; row < view->length; ++row) {
        if (!input_row_is_active(input, row)) {
            continue;
        }
        bool is_null = false;
        int status = child_null(bound, row, &is_null);
        const uint64_t physical = dora_vector_view_get_physical_row(view, row);
        if (status != DORA_OK) {
            return status;
        }
        if (is_null) {
            if (required) {
                return EINVAL;
            }
            continue;
        }
        if (map) {
            const uint32_t width = field->layout == DORA_LAYOUT_LIST ? 4 : 8;
            int64_t begin = 0;
            int64_t end = 0;
            status = child_integer(bound, view->offsets, physical, width, &begin);
            if (status == DORA_OK) {
                status = child_integer(bound, view->offsets, physical + 1, width, &end);
            }
            if (status != DORA_OK) {
                return status;
            }
            for (uint64_t key = (uint64_t)begin; key < (uint64_t)end; ++key) {
                status = child_logical_null(keys, key, &is_null);
                if (status != DORA_OK || is_null) {
                    return status != DORA_OK ? status : EINVAL;
                }
            }
        } else if (required) {
            uint32_t child = 0;
            uint64_t child_row = 0;
            switch (view->layout) {
            case DORA_LAYOUT_DICTIONARY:
                status = child_index(bound, row, &child_row);
                break;
            case DORA_LAYOUT_DENSE_UNION:
            case DORA_LAYOUT_SPARSE_UNION: {
                uint8_t code = 0;
                status = child_read(bound, view->type_ids, (struct dora_range){.offset = physical, .length = 1}, &code);
                if (status == DORA_OK && code > INT8_MAX) {
                    status = EINVAL;
                }
                if (status == DORA_OK) {
                    child = view->union_members[code];
                    child_row = physical;
                }
                if (status == DORA_OK && view->layout == DORA_LAYOUT_DENSE_UNION) {
                    int64_t member = 0;
                    status = child_integer(bound, view->offsets, physical, 4, &member);
                    child_row = (uint64_t)member;
                }
                break;
            }
            case DORA_LAYOUT_RUN_END:
                child = 1;
                struct admission_child ends = {};
                status = child_bind(bound, 0, &ends);
                if (status == DORA_OK) {
                    status = child_find_run(&ends, physical, &child_row);
                }
                break;
            default:
                continue;
            }
            if (status == DORA_OK) {
                struct admission_child selected = {};
                status = child_bind(bound, child, &selected);
                if (status == DORA_OK) {
                    status = child_logical_null(selected, child_row, &is_null);
                }
            }
            if (status != DORA_OK || is_null) {
                return status != DORA_OK ? status : EINVAL;
            }
        }
    }
    return DORA_OK;
}

static bool node_needs_activity(const struct admission_child* bound, bool descendants) {
    const struct vector_bound_node* const node = &bound->tree->nodes[bound->node];
    const struct dora_schema_field* const field = &node->field;
    if (((field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0 && node_may_be_null(node)) ||
        map_needs_activity(bound, field)) {
        return true;
    }
    for (uint32_t ordinal = field->layout == DORA_LAYOUT_RUN_END ? 1 : 0; descendants && ordinal < field->child_count;
         ++ordinal) {
        struct admission_child child = {};
        input_child(bound, ordinal, &child);
        if (node_needs_activity(&child, true)) {
            return true;
        }
    }
    return false;
}

static int activity_child_range(uint32_t child,
                                const struct admission_child* bound,
                                uint64_t window_offset,
                                uint64_t row,
                                struct dora_range* out_rows) {
    const struct dora_vector_view* const view = &bound->view;
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    *out_rows = (struct dora_range){.length = 1};
    switch (view->layout) {
    case DORA_LAYOUT_STRUCT:
        return ckd_add(&out_rows->offset, window_offset, row) ? EINVAL : DORA_OK;
    case DORA_LAYOUT_FIXED_LIST:
        out_rows->length = view->list_element_count;
        return ckd_mul(&out_rows->offset, physical, view->list_element_count) ? EINVAL : DORA_OK;
    case DORA_LAYOUT_LIST:
    case DORA_LAYOUT_LARGE_LIST:
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW: {
        const bool large = view->layout == DORA_LAYOUT_LARGE_LIST || view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        const bool list_view = view->layout == DORA_LAYOUT_LIST_VIEW || view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        const uint32_t width = large ? 8 : 4;
        int64_t begin = 0;
        int64_t end = 0;
        int status = child_integer(bound, view->offsets, physical, width, &begin);
        if (status == DORA_OK) {
            status = child_integer(bound,
                                   list_view ? view->sizes : view->offsets,
                                   list_view ? physical : physical + 1,
                                   width,
                                   &end);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (begin < 0 || end < 0 || (!list_view && end < begin)) {
            return EINVAL;
        }
        out_rows->offset = (uint64_t)begin;
        out_rows->length = (uint64_t)(list_view ? end : end - begin);
        return DORA_OK;
    }
    case DORA_LAYOUT_DICTIONARY:
        return child_index(bound, row, &out_rows->offset);
    case DORA_LAYOUT_SPARSE_UNION:
    case DORA_LAYOUT_DENSE_UNION: {
        uint8_t code = 0;
        int status = child_read(bound, view->type_ids, (struct dora_range){.offset = physical, .length = 1}, &code);
        if (status != DORA_OK) {
            return status;
        }
        const uint32_t child_count = bound->tree->nodes[bound->node].field.child_count;
        if (code > INT8_MAX || view->union_members[code] >= child_count) {
            return EINVAL;
        }
        if (view->union_members[code] != child) {
            out_rows->length = 0;
            return DORA_OK;
        }
        out_rows->offset = physical;
        if (view->layout == DORA_LAYOUT_DENSE_UNION) {
            int64_t member = 0;
            status = child_integer(bound, view->offsets, physical, 4, &member);
            if (status == DORA_OK && member < 0) {
                status = EINVAL;
            }
            out_rows->offset = (uint64_t)member;
        }
        return status;
    }
    case DORA_LAYOUT_RUN_END: {
        struct admission_child ends = {};
        const int status = child_bind(bound, 0, &ends);
        return status == DORA_OK ? child_find_run(&ends, physical, &out_rows->offset) : status;
    }
    default:
        return EINVAL;
    }
}

static int activity_tree(struct dora_pool pool,
                         const struct admission_child* bound,
                         uint64_t window_offset,
                         const struct vector_input* input) {
    const struct dora_schema_field* const field = &bound->tree->nodes[bound->node].field;
    int status = active_null_check(bound, input, field);
    if (status != DORA_OK || !input->should_check_child_activity) {
        return status;
    }
    for (uint32_t index = field->layout == DORA_LAYOUT_RUN_END ? 1 : 0; index < field->child_count; ++index) {
        struct admission_child child = {};
        input_child(bound, index, &child);
        if (!node_needs_activity(&child, true)) {
            continue;
        }
        const uint64_t words = bitmap_words_for_bits(child.view.length);
        size_t bytes = 0;
        if (words > SIZE_MAX || ckd_mul(&bytes, (size_t)words, sizeof(uint64_t))) {
            return EOVERFLOW;
        }
        struct dora_allocation activity = {};
        status = dora_pool_allocate(pool, bytes, &activity);
        if (status != DORA_OK) {
            return status;
        }
        if (bytes != 0) {
            bitmap_fill_bytes(activity.address, bytes, false);
        }
        bool any_active = false;
        for (uint64_t row = 0; status == DORA_OK && row < bound->view.length; ++row) {
            if (!input_row_is_active(input, row)) {
                continue;
            }
            bool is_null = false;
            status = child_null(bound, row, &is_null);
            if (status != DORA_OK || is_null) {
                continue;
            }
            struct dora_range reached = {};
            status = activity_child_range(index, bound, window_offset, row, &reached);
            if (status == DORA_OK &&
                (reached.offset > child.view.length || reached.length > child.view.length - reached.offset)) {
                status = EINVAL;
            }
            if (status == DORA_OK && reached.length != 0) {
                dora_bitmap_set_range(activity.address, reached.offset, reached.offset + reached.length, true);
                any_active = true;
            }
        }
        if (status == DORA_OK && any_active) {
            status = child_bind(bound, index, &child);
            const struct dora_bytes bitmap = {.data = activity.address, .byte_count = bytes};
            const struct vector_input child_input = {.ancestor_activity = &bitmap, .should_check_child_activity = true};
            if (status == DORA_OK) {
                status = activity_tree(pool, &child, 0, &child_input);
            }
        }
        dora_allocation_release(&activity);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int activity_check(struct dora_pool pool,
                          struct admission_child bound,
                          uint64_t window_offset,
                          const struct vector_input* input) {
    if (!node_needs_activity(&bound, input->should_check_child_activity)) {
        return DORA_OK;
    }
    const struct vector_image* const image = bound.tree->nodes[bound.node].image;
    if (image != nullptr) {
        if (image->ownership.requires_effective_validity) {
            return ENOTSUP;
        }
        const int status = child_provider(image, &bound);
        if (status != DORA_OK) {
            return status;
        }
    }
    return activity_tree(pool, &bound, window_offset, input);
}

int dora_vector_check_activity(struct vector_image* image, const struct vector_input* input) {
    bool active = false;
    int status = input_activity(input, image->ownership.view.length, &active);
    if (status != DORA_OK || !active) {
        return status;
    }
    struct vector_bound_node root_nodes[2] = {
        {.image = image, .view = image->ownership.view, .subtree_end_node_index = 1},
        {.view.placement.device.type = DORA_DEVICE_TYPE_CPU}
    };
    struct scratch_scope scope = {};
    status = dora_schema_get_field(&image->schema, &root_nodes[0].field);
    struct vector_bound_input source = {.nodes = root_nodes, .node_count = 1};
    if (status == DORA_OK && image->child_vectors.entry_count != 0) {
        status = dora_scratch_scope_begin(image->allocation.pool, &scope);
        if (status == DORA_OK) {
            status = dora_vector_bind_children(&scope,
                                               dora_vector_image_children(image),
                                               image->child_vectors.entry_count,
                                               root_nodes,
                                               &source);
        }
    }
    if (status == DORA_OK) {
        source.nodes[0].image = image;
        source.nodes[0].field = root_nodes[0].field;
        source.nodes[0].view = root_nodes[0].view;
        const struct admission_child bound = {.view = root_nodes[0].view, .tree = &source};
        status = activity_check(image->allocation.pool, bound, image->node.offset, input);
    }
    dora_scratch_scope_end(&scope);
    return status;
}

int dora_vector_admit(struct dora_pool pool,
                      const struct dora_schema* schema,
                      struct vector_node_ownership* owned,
                      const struct vector_input* input,
                      struct dora_view_buffer* buffers) {
    for (uint32_t index = 0; index < input->buffer_count; ++index) {
        const struct dora_bytes bytes = input->buffers != nullptr
                                            ? input->buffers[index]
                                            : (struct dora_bytes){
                                                  .data = input->allocations[index].address,
                                                  .byte_count = input->allocations[index].capacity_bytes
                                              };
        if (bytes.data == nullptr && bytes.byte_count != 0) {
            return EINVAL;
        }
        buffers[index] = (struct dora_view_buffer){
            .data = bytes.data,
            .byte_count = bytes.byte_count,
            .readable_byte_count = bytes.byte_count
        };
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    status = activity_extent(input, input->row_count);
    if (status != DORA_OK) {
        return status;
    }
    uint64_t endpoint = 0;
    if (ckd_add(&endpoint, input->offset, input->row_count) || endpoint > INT64_MAX || input->null_count < -1 ||
        (input->null_count >= 0 && (uint64_t)input->null_count > input->row_count)) {
        return EINVAL;
    }
    struct dora_vector_view* const view = &owned->view;
    *view = (struct dora_vector_view){
        .type = field.type.id,
        .layout = field.layout,
        .length = input->row_count,
        .null_count = input->null_count,
        .offset = input->offset,
        .validity_bit_offset = input->validity_bit_offset,
        .placement = {.device.type = DORA_DEVICE_TYPE_CPU, .access = DORA_ACCESS_HOST}
    };
    if (input->owner != nullptr) {
        view->placement = input->owner->placement;
        if (input->owner->read_host == nullptr) {
            view->placement.access =
                view->placement.access == DORA_ACCESS_DEVICE || view->placement.access == DORA_ACCESS_HOST_AND_DEVICE
                    ? DORA_ACCESS_HOST_AND_DEVICE
                    : DORA_ACCESS_HOST;
        }
        if (view->placement.access > DORA_ACCESS_HOST_AND_DEVICE || view->placement.access == DORA_ACCESS_NONE) {
            return EINVAL;
        }
        view->readiness = input->owner->readiness;
        if (view->readiness.event != nullptr) {
            if (view->readiness.settle == nullptr) {
                return EINVAL;
            }
            status = view->readiness.settle(&view->readiness);
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    const struct dora_logical_type physical =
        field.layout == DORA_LAYOUT_DICTIONARY ? (struct dora_logical_type){.id = field.index_type} : field.type;
    status = type_physical(&physical, &view->physical_type);
    if (status != DORA_OK) {
        return status;
    }
    view->byte_width = field.type.id == DORA_TYPE_FIXED_BINARY && field.layout != DORA_LAYOUT_DICTIONARY
                           ? field.type.parameters.byte_width
                           : physical_width(view->physical_type);
    view->list_element_count = field.type.id == DORA_TYPE_FIXED_LIST ? field.type.parameters.list_element_count : 0;
    uint32_t expected = 1;
    switch (field.layout) {
    case DORA_LAYOUT_NULL:
        expected = input->buffer_count == 1 ? 1 : 0;
        break;
    case DORA_LAYOUT_RUN_END:
        expected = 0;
        break;
    case DORA_LAYOUT_BITS:
    case DORA_LAYOUT_FIXED:
    case DORA_LAYOUT_LIST:
    case DORA_LAYOUT_LARGE_LIST:
    case DORA_LAYOUT_DICTIONARY:
    case DORA_LAYOUT_DENSE_UNION:
        expected = 2;
        break;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        expected = 3;
        break;
    case DORA_LAYOUT_BINARY_VIEW:
        if (input->buffer_count < 3) {
            return EINVAL;
        }
        expected = input->buffer_count;
        break;
    default:
        break;
    }
    if (input->buffer_count != expected) {
        return EINVAL;
    }
    const bool has_validity = field.layout != DORA_LAYOUT_NULL && field.layout != DORA_LAYOUT_RUN_END &&
                              field.layout != DORA_LAYOUT_DENSE_UNION && field.layout != DORA_LAYOUT_SPARSE_UNION;
    if (has_validity) {
        owned->own_validity = buffers[0];
        owned->own_validity_bit_offset = view->validity_bit_offset;
        if (buffers[0].data == nullptr) {
            if (view->null_count > 0) {
                return EINVAL;
            }
            view->null_count = 0;
        } else {
            uint64_t validity_end = 0;
            if (ckd_add(&validity_end, view->validity_bit_offset, view->length)) {
                return EINVAL;
            }
            status = extent_check(buffers[0], bitmap_bytes_for_bits(validity_end), 1);
            if (status != DORA_OK) {
                return status;
            }
            if (view->null_count != 0) {
                view->validity = buffers[0];
            }
        }
    } else if (field.layout == DORA_LAYOUT_NULL) {
        if (expected == 1 && buffers[0].data != nullptr) {
            return EINVAL;
        }
        view->null_count = view->length;
    } else if (view->null_count != 0) {
        return EINVAL;
    }
    owned->own_null_count = view->null_count;
    if (status == DORA_OK && input->effective_validity != nullptr) {
        if (input->offset != 0 || input->validity_bit_offset != 0) {
            return EINVAL;
        }
        const struct dora_allocation* const effective = input->effective_validity;
        view->validity = (struct dora_view_buffer){
            .data = effective->address,
            .byte_count = effective->capacity_bytes,
            .readable_byte_count = effective->capacity_bytes
        };
        status = extent_check(view->validity, bitmap_bytes_for_bits(view->length), 1);
        if (status != DORA_OK) {
            return status;
        }
        view->null_count = input->effective_null_count;
    }
    struct vector_bound_node root_nodes[2] = {
        {.field = field, .view = *view, .subtree_end_node_index = 1},
        {.view.placement.device.type = DORA_DEVICE_TYPE_CPU}
    };
    struct vector_bound_input source = {.nodes = root_nodes, .node_count = 1};
    struct scratch_scope scope = {};
    const uint32_t root = input->foreign != nullptr ? input->foreign_node_index : 0;
    if (input->foreign != nullptr) {
        source = *input->foreign;
    } else if (input->child_count != 0) {
        status = dora_scratch_scope_begin(pool, &scope);
        if (status == DORA_OK) {
            status = dora_vector_bind_children(&scope, input->children, input->child_count, root_nodes, &source);
        }
        if (status != DORA_OK) {
            dora_scratch_scope_end(&scope);
            return status;
        }
    }
    source.nodes[root].field = field;
    source.nodes[root].view = *view;
    struct admission_child bound = {
        .view = *view,
        .owner = input->owner,
        .tree = &source,
        .node = root,
        .has_host_validity = input->effective_validity != nullptr || view->validity.data == input->data_buffer_sizes
    };
    switch (field.layout) {
    case DORA_LAYOUT_BITS:
        view->values = buffers[1];
        status = extent_check(view->values, bitmap_bytes_for_bits(endpoint), 1);
        break;
    case DORA_LAYOUT_FIXED:
        view->values = buffers[1];
        status = extent_check(view->values, endpoint, view->byte_width);
        break;
    case DORA_LAYOUT_DICTIONARY:
        view->values = buffers[1];
        status = dictionary_check(pool, input, view, &source, root);
        break;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
        view->offsets = buffers[1];
        view->data_buffers = &buffers[2];
        view->data_buffer_count = 1;
        status = offsets_check(pool, input, view, buffers[2].byte_count, false);
        break;
    case DORA_LAYOUT_BINARY_VIEW:
        view->values = buffers[1];
        view->data_buffers = &buffers[2];
        view->data_buffer_count = input->buffer_count - 3;
        status = extent_check(buffers[input->buffer_count - 1], view->data_buffer_count, sizeof(int64_t));
        for (uint32_t index = 0; status == DORA_OK && index < view->data_buffer_count; ++index) {
            int64_t size = 0;
            status = input_integer(input_owner(input, buffers[input->buffer_count - 1].data),
                                   buffers[input->buffer_count - 1],
                                   index,
                                   sizeof(int64_t),
                                   &size);
            if (status == DORA_OK && (size < 0 || (uint64_t)size > buffers[index + 2].byte_count)) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                buffers[index + 2].byte_count = size;
                if (input->allocations == nullptr) {
                    buffers[index + 2].readable_byte_count = size;
                }
            }
        }
        if (status == DORA_OK) {
            status = binary_views_check(pool, input, view);
        }
        break;
    case DORA_LAYOUT_LIST:
    case DORA_LAYOUT_LARGE_LIST:
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW: {
        view->offsets = buffers[1];
        const bool is_view = field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        if (is_view) {
            view->sizes = buffers[2];
        }
        const uint64_t limit = input_child_length(&source, root, 0);
        status = offsets_check(pool, input, view, limit, is_view);
        break;
    }
    case DORA_LAYOUT_FIXED_LIST: {
        uint64_t count = 0;
        if (ckd_mul(&count, endpoint, view->list_element_count) || count > input_child_length(&source, root, 0)) {
            status = EINVAL;
        }
        break;
    }
    case DORA_LAYOUT_STRUCT:
    case DORA_LAYOUT_SPARSE_UNION:
        for (uint32_t index = 0; index < input->child_count; ++index) {
            if (input_child_length(&source, root, index) < endpoint) {
                status = EINVAL;
                break;
            }
        }
        break;
    case DORA_LAYOUT_RUN_END:
        status = run_ends_check(pool, &bound, view);
        break;
    default:
        break;
    }
    if (status != DORA_OK) {
        dora_scratch_scope_end(&scope);
        return status;
    }
    if (field.layout == DORA_LAYOUT_DENSE_UNION || field.layout == DORA_LAYOUT_SPARSE_UNION) {
        memset(owned->union_members, UINT8_MAX, sizeof(owned->union_members));
        for (uint32_t index = 0; index < field.child_count; ++index) {
            owned->union_members[(uint8_t)field.type.parameters.type_codes[index]] = (uint8_t)index;
        }
        view->union_members = owned->union_members;
        view->type_ids = buffers[0];
        if (field.layout == DORA_LAYOUT_DENSE_UNION) {
            view->offsets = buffers[1];
        }
        status = union_check(pool, input, view, &source, root);
    }
    if (status == DORA_OK) {
        source.nodes[root].view = *view;
        bound.view = *view;
        bool active = false;
        status = input_activity(input, view->length, &active);
        if (status == DORA_OK && active) {
            status = activity_check(pool, bound, input->offset, input);
        }
    }
    if (status == DORA_OK && input->allocations != nullptr) {
        if (has_validity && buffers[0].data != nullptr) {
            buffers[0].byte_count = bitmap_bytes_for_bits(view->validity_bit_offset + view->length);
            owned->own_validity = buffers[0];
            if (owned->own_null_count == 0) {
                owned->own_validity = (struct dora_view_buffer){};
                buffers[0] = (struct dora_view_buffer){};
            } else if (input->effective_validity == nullptr) {
                view->validity = buffers[0];
            }
        }
        switch (view->layout) {
        case DORA_LAYOUT_BITS:
            view->values.byte_count = bitmap_bytes_for_bits(endpoint);
            break;
        case DORA_LAYOUT_FIXED:
        case DORA_LAYOUT_DICTIONARY:
            view->values.byte_count = endpoint * view->byte_width;
            break;
        case DORA_LAYOUT_BINARY_VIEW:
            view->values.byte_count = endpoint * sizeof(struct dora_binary_view_entry);
            break;
        case DORA_LAYOUT_BINARY:
        case DORA_LAYOUT_LARGE_BINARY: {
            const uint32_t width = view->layout == DORA_LAYOUT_BINARY ? 4 : 8;
            view->offsets.byte_count = (endpoint + 1) * width;
            int64_t end = 0;
            status = input_integer(input_owner(input, view->offsets.data), view->offsets, endpoint, width, &end);
            if (status == DORA_OK && (end < 0 || (uint64_t)end > buffers[2].byte_count)) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                buffers[2].byte_count = end;
            }
            break;
        }
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
            view->offsets.byte_count = (endpoint + 1) * (view->layout == DORA_LAYOUT_LIST ? 4U : 8U);
            break;
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW:
            view->offsets.byte_count = endpoint * (view->layout == DORA_LAYOUT_LIST_VIEW ? 4U : 8U);
            view->sizes.byte_count = view->offsets.byte_count;
            break;
        case DORA_LAYOUT_DENSE_UNION:
            view->offsets.byte_count = endpoint * sizeof(int32_t);
            view->type_ids.byte_count = endpoint;
            break;
        case DORA_LAYOUT_SPARSE_UNION:
            view->type_ids.byte_count = endpoint;
            break;
        default:
            break;
        }
    }
    dora_scratch_scope_end(&scope);
    return status;
}

static int map_key_row(struct admission_child* child, uint64_t* row, bool* out_null) {
    for (;;) {
        if (*row >= child->view.length) {
            return EINVAL;
        }
        const struct vector_bound_node* const node = &child->tree->nodes[child->node];
        if (node->field.type.extension_name.byte_count != 0 &&
            (node->extension == nullptr || (node->extension->flags & DORA_EXTENSION_FLAG_STORAGE_EQUALITY) == 0)) {
            return ENOTSUP;
        }
        int status = child_null(child, *row, out_null);
        if (status != DORA_OK || *out_null) {
            return status;
        }
        const uint64_t physical = dora_vector_view_get_physical_row(&child->view, *row);
        uint32_t ordinal = 0;
        if (child->view.layout == DORA_LAYOUT_DICTIONARY) {
            status = child_index(child, *row, row);
        } else if (child->view.layout == DORA_LAYOUT_RUN_END) {
            struct admission_child ends = {};
            input_child(child, 0, &ends);
            status = child_find_run(&ends, physical, row);
            ordinal = 1;
        } else {
            return DORA_OK;
        }
        struct admission_child next = {};
        if (status == DORA_OK) {
            input_child(child, ordinal, &next);
        }
        if (status != DORA_OK) {
            return status;
        }
        *child = next;
    }
}

static int map_key_member(const struct admission_child* child,
                          uint64_t row,
                          struct admission_child* out_member,
                          uint8_t code,
                          uint64_t* out_row) {
    const struct dora_vector_view* const view = &child->view;
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    int status = DORA_OK;
    *out_row = physical;
    if (view->layout == DORA_LAYOUT_DENSE_UNION) {
        int64_t value = 0;
        status = child_integer(child, view->offsets, physical, 4, &value);
        *out_row = (uint64_t)value;
    }
    if (status == DORA_OK && out_member != nullptr) {
        input_child(child, view->union_members[code], out_member);
    }
    return status;
}

static int map_key_scalar(const struct admission_child* child, uint64_t row, union decoded_row_value* out) {
    const struct dora_vector_view* const view = &child->view;
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    if (view->layout == DORA_LAYOUT_BITS) {
        uint8_t byte = 0;
        const int status = child_read(child,
                                      view->values,
                                      (struct dora_range){.offset = bitmap_byte_of(physical), .length = 1},
                                      &byte);
        out->boolean = dora_bitmap_test(&byte, bitmap_bit_in_byte(physical));
        return status;
    }
    return child_read(child,
                      view->values,
                      (struct dora_range){.offset = physical * view->byte_width, .length = view->byte_width},
                      out);
}

static int map_key_equal(void* context, uint64_t left_row, uint64_t right_row, bool* out_equal) {
    struct admission_child left = *(struct admission_child*)context;
    struct admission_child right = left;
    bool left_null = false, right_null = false;
    int status = map_key_row(&left, &left_row, &left_null);
    if (status == DORA_OK) {
        status = map_key_row(&right, &right_row, &right_null);
    }
    if (status != DORA_OK || left_null || right_null) {
        *out_equal = left_null == right_null;
        return status;
    }
    const struct dora_schema_field* const field = &left.tree->nodes[left.node].field;
    const struct dora_vector_view* const view = &left.view;
    if (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION) {
        uint8_t codes[2] = {};
        uint64_t rows[2] = {};
        struct admission_child member = {};
        status =
            child_read(&left,
                       view->type_ids,
                       (struct dora_range){.offset = dora_vector_view_get_physical_row(view, left_row), .length = 1},
                       &codes[0]);
        if (status == DORA_OK) {
            status = child_read(
                &right,
                right.view.type_ids,
                (struct dora_range){.offset = dora_vector_view_get_physical_row(&right.view, right_row), .length = 1},
                &codes[1]);
        }
        if (status != DORA_OK || codes[0] != codes[1]) {
            *out_equal = false;
            return status;
        }
        status = map_key_member(&left, left_row, &member, codes[0], &rows[0]);
        if (status == DORA_OK) {
            status = map_key_member(&right, right_row, nullptr, codes[1], &rows[1]);
        }
        if (status != DORA_OK) {
            return status;
        }
        return map_key_equal(&member, rows[0], rows[1], out_equal);
    }

    if (view->type == DORA_TYPE_STRUCT || view->type == DORA_TYPE_LIST || view->type == DORA_TYPE_FIXED_LIST ||
        view->type == DORA_TYPE_MAP) {
        struct dora_range ranges[2] = {{.offset = left_row, .length = 1}, {.offset = right_row, .length = 1}};
        if (view->type != DORA_TYPE_STRUCT) {
            status = activity_child_range(0, &left, 0, left_row, &ranges[0]);
            if (status == DORA_OK) {
                status = activity_child_range(0, &right, 0, right_row, &ranges[1]);
            }
        }
        *out_equal = ranges[0].length == ranges[1].length;
        for (uint64_t row = 0; status == DORA_OK && *out_equal && row < ranges[0].length; ++row) {
            for (uint32_t ordinal = 0; status == DORA_OK && *out_equal && ordinal < field->child_count; ++ordinal) {
                struct admission_child child = {};
                input_child(&left, ordinal, &child);
                status = map_key_equal(&child, ranges[0].offset + row, ranges[1].offset + row, out_equal);
            }
        }
        return status;
    }
    if (view->type == DORA_TYPE_BINARY || view->type == DORA_TYPE_UTF8 || view->type == DORA_TYPE_FIXED_BINARY) {
        return type_bytes_matches(dora_vector_view_get_bytes(view, left_row),
                                  dora_vector_view_get_bytes(&right.view, right_row),
                                  DORA_COMPARE_OP_EQ,
                                  out_equal);
    }
    union decoded_row_value values[2] = {};
    status = map_key_scalar(&left, left_row, &values[0]);
    if (status == DORA_OK) {
        status = map_key_scalar(&right, right_row, &values[1]);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (view->layout == DORA_LAYOUT_BITS) {
        *out_equal = values[0].boolean == values[1].boolean;
        return DORA_OK;
    }
    switch (view->physical_type) {
// Returns DORA_OK from the enclosing function.
#define MAP_EQUAL_CASE(S, T, P, ...)                                                                                   \
    case P:                                                                                                            \
        *out_equal = type_order_##S(values[0].S, values[1].S) == 0;                                                    \
        return DORA_OK;
        FIXED_CARRIERS(MAP_EQUAL_CASE)
#undef MAP_EQUAL_CASE
    default:
        return ENOTSUP;
    }
}

static int map_key_hash(void* context, uint64_t row, uint64_t* out_hash) {
    struct admission_child child = *(struct admission_child*)context;
    bool is_null = false;
    int status = map_key_row(&child, &row, &is_null);
    if (status != DORA_OK || is_null) {
        *out_hash = HASH_NULL_VALUE;
        return status;
    }
    const struct dora_vector_view* const view = &child.view;
    const struct dora_schema_field* const field = &child.tree->nodes[child.node].field;
    if (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION) {
        uint8_t code = 0;
        uint64_t selected = 0;
        struct admission_child member = {};
        status = child_read(&child,
                            view->type_ids,
                            (struct dora_range){.offset = dora_vector_view_get_physical_row(view, row), .length = 1},
                            &code);
        if (status == DORA_OK) {
            status = map_key_member(&child, row, &member, code, &selected);
        }
        return status == DORA_OK ? map_key_hash(&member, selected, out_hash) : status;
    }

    if (view->type == DORA_TYPE_STRUCT || view->type == DORA_TYPE_LIST || view->type == DORA_TYPE_FIXED_LIST ||
        view->type == DORA_TYPE_MAP) {
        struct dora_range range = {.offset = row, .length = 1};
        if (view->type != DORA_TYPE_STRUCT) {
            status = activity_child_range(0, &child, 0, row, &range);
        }
        uint64_t hash = 0;
        for (uint64_t item = 0; status == DORA_OK && item < range.length; ++item) {
            for (uint32_t ordinal = 0; status == DORA_OK && ordinal < field->child_count; ++ordinal) {
                struct admission_child member = {};
                input_child(&child, ordinal, &member);
                uint64_t value = 0;
                status = map_key_hash(&member, range.offset + item, &value);
                hash = hash_value_combine(hash, value);
            }
        }
        *out_hash = hash;
        return status;
    }
    if (view->type == DORA_TYPE_BINARY || view->type == DORA_TYPE_UTF8 || view->type == DORA_TYPE_FIXED_BINARY) {
        *out_hash = hash_value_bytes(dora_vector_view_get_bytes(view, row), 0);
        return DORA_OK;
    }
    union decoded_row_value value = {};
    status = map_key_scalar(&child, row, &value);
    if (status != DORA_OK) {
        return status;
    }
    if (view->layout == DORA_LAYOUT_BITS) {
        *out_hash = hash_value_bool(value.boolean, 0);
        return DORA_OK;
    }
    switch (view->physical_type) {
// Returns DORA_OK from the enclosing function.
#define MAP_HASH_CASE(S, T, P, ...)                                                                                    \
    case P:                                                                                                            \
        *out_hash = hash_value_##S(value.S, 0);                                                                        \
        return DORA_OK;
        FIXED_CARRIERS(MAP_HASH_CASE)
#undef MAP_HASH_CASE
    default:
        return ENOTSUP;
    }
}

static int map_key_check(struct dora_pool pool,
                         struct admission_child keys,
                         struct dora_range range,
                         struct vector_map_scratch* scratch) {
    int status = DORA_OK;
    const bool pairwise = range.length <= KEYSET_PAIRWISE_LIMIT;
    if (!pairwise) {
        if (range.length > scratch->set.slot_capacity / 2) {
            dora_keyset_release(&scratch->set);
            status = dora_keyset_init(&scratch->set, pool, range.length);
        } else {
            keyset_clear(&scratch->set);
        }
        if (status == DORA_OK && !scratch->are_hashes_admitted) {
            const uint32_t first = keys.node;
            const struct vector_bound_input checked = {
                .nodes = &keys.tree->nodes[first],
                .node_count = keys.tree->nodes[first].subtree_end_node_index - first
            };
            status = dora_vector_input_admit(&checked, DORA_EXTENSION_FLAG_STORAGE_EQUALITY, false);
            scratch->are_hashes_admitted = status == DORA_OK;
        }
    }
    const struct dora_keyset_rows callbacks = {.context = &keys, .hash = map_key_hash, .equal = map_key_equal};
    for (uint64_t entry = 0; status == DORA_OK && entry < range.length; ++entry) {
        bool is_null = false;
        status = child_logical_null(keys, range.offset + entry, &is_null);
        if (status == DORA_OK && is_null) {
            status = EINVAL;
        }
        if (pairwise) {
            for (uint64_t previous = 0; status == DORA_OK && previous < entry; ++previous) {
                bool equal = false;
                status = map_key_equal(&keys, range.offset + previous, range.offset + entry, &equal);
                if (status == DORA_OK && equal) {
                    status = EILSEQ;
                }
            }
        } else if (status == DORA_OK) {
            uint64_t existing = 0;
            bool duplicate = false;
            status = dora_keyset_insert(&scratch->set, &callbacks, range.offset + entry, entry, &existing, &duplicate);
            if (status == DORA_OK && duplicate) {
                status = EILSEQ;
            }
        }
    }
    return status;
}

static uint32_t map_key_width(const struct vector_map_scratch* scratch) {
    const struct dora_vector_view* const view = &scratch->view;
    return scratch->can_bind_directly && view->layout == DORA_LAYOUT_FIXED &&
                   ((view->physical_type >= DORA_PHYSICAL_TYPE_I8 && view->physical_type <= DORA_PHYSICAL_TYPE_I64) ||
                    (view->physical_type >= DORA_PHYSICAL_TYPE_U8 && view->physical_type <= DORA_PHYSICAL_TYPE_U64))
               ? physical_width(view->physical_type)
               : 0;
}

static void map_key_bind_view(const struct dora_logical_type* type, struct vector_map_scratch* scratch) {
    const struct dora_vector_view* const view = &scratch->view;
    scratch->can_bind_directly = type->extension_name.byte_count == 0 && !vector_layout_derives_nulls(view->layout);
}

static inline bool map_key_is_bound(const struct dora_vector* keys, const struct vector_map_scratch* scratch) {
    return scratch->keys.handle.slot_index != 0 && dora_handle_is_same(scratch->keys.handle, keys->handle) &&
           scratch->keys.rows.offset == keys->rows.offset && scratch->keys.rows.length == keys->rows.length;
}

// Rejects scratch bound to another vector or range with EINVAL.
static int map_key_bind(const struct dora_vector* keys, struct vector_map_scratch* scratch) {
    if (scratch->keys.handle.slot_index != 0) {
        return EINVAL;
    }
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(keys, &type);
    if (status == DORA_OK) {
        status = dora_vector_get_view(keys, &scratch->view);
    }
    if (status != DORA_OK) {
        return status;
    }
    map_key_bind_view(&type, scratch);
    scratch->keys = dora_vector_borrow(keys);
    return DORA_OK;
}

int dora_vector_check_map_keys(struct dora_pool pool,
                               const struct dora_vector* keys,
                               struct dora_range entries,
                               bool sorted,
                               struct vector_map_scratch* scratch) {
    if (entries.length == 0 && scratch->keys.handle.slot_index == 0) {
        return DORA_OK;
    }
    int status = map_key_is_bound(keys, scratch) ? DORA_OK : map_key_bind(keys, scratch);
    if (status != DORA_OK || entries.length == 0) {
        return status;
    }
    if (sorted) {
        for (uint64_t entry = 0; status == DORA_OK && entry < entries.length; ++entry) {
            bool is_null = false;
            const uint64_t key = entries.offset + entry;
            status = dora_vector_is_null(keys, key, &is_null);
            if (status == DORA_OK && is_null) {
                status = EINVAL;
            }
            if (status == DORA_OK && entry != 0) {
                bool ordered = false;
                status = dora_vector_compare_rows(keys, key - 1, key, DORA_COMPARE_OP_LT, &ordered);
                if (status == DORA_OK && !ordered) {
                    status = EILSEQ;
                }
            }
        }
        return status;
    }
    struct vector_bound_node leaf =
        {.field.type.id = scratch->view.type, .view = scratch->view, .subtree_end_node_index = 1};
    const bool scalar =
        scratch->can_bind_directly &&
        (scratch->view.layout == DORA_LAYOUT_FIXED || scratch->view.layout == DORA_LAYOUT_BITS ||
         scratch->view.layout == DORA_LAYOUT_BINARY || scratch->view.layout == DORA_LAYOUT_LARGE_BINARY ||
         scratch->view.layout == DORA_LAYOUT_BINARY_VIEW || scratch->view.layout == DORA_LAYOUT_NULL);
    const struct vector_bound_input direct = {.nodes = &leaf, .node_count = 1, .length = scratch->view.length};
    if (!scalar && scratch->input.nodes == nullptr) {
        if (scratch->scope.pool.index == 0) {
            status = dora_scratch_scope_begin(pool, &scratch->scope);
        }
        if (status == DORA_OK) {
            status = dora_vector_bind_input(&scratch->scope, keys, nullptr, &scratch->input);
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    const struct admission_child bound = {.view = scratch->view, .tree = scalar ? &direct : &scratch->input};
    return map_key_check(pool, bound, entries, scratch);
}

// Key carriers: UInt8, UInt16, UInt32, UInt64
#define DEFINE_MAP_SMALL_KEYS(SUFFIX, TYPE, PHYSICAL)                                                                  \
    static int map_small_keys_##SUFFIX(struct dora_pool pool,                                                          \
                                       const struct dora_vector* keys,                                                 \
                                       const struct dora_vector_view* maps,                                            \
                                       struct vector_map_scratch* scratch,                                             \
                                       uint64_t* out_next_row) {                                                       \
        const uint8_t* const data = scratch->view.values.data;                                                         \
        for (uint64_t row = 0; row < maps->length; ++row) {                                                            \
            if (dora_vector_view_is_null(maps, row)) {                                                                 \
                continue;                                                                                              \
            }                                                                                                          \
            const struct dora_range entries = dora_vector_view_get_elements(maps, row);                                \
            if (entries.length > KEYSET_PAIRWISE_LIMIT) {                                                              \
                if (keys == nullptr) {                                                                                 \
                    *out_next_row = row;                                                                               \
                    return DORA_OK;                                                                                    \
                }                                                                                                      \
                const int status = dora_vector_check_map_keys(pool, keys, entries, false, scratch);                    \
                if (status != DORA_OK) {                                                                               \
                    return status;                                                                                     \
                }                                                                                                      \
                continue;                                                                                              \
            }                                                                                                          \
            for (uint64_t entry = 1; entry < entries.length; ++entry) {                                                \
                const TYPE value = type_load_##SUFFIX(data, scratch->view.offset + entries.offset + entry);            \
                for (uint64_t previous = 0; previous < entry; ++previous) {                                            \
                    const TYPE other = type_load_##SUFFIX(data, scratch->view.offset + entries.offset + previous);     \
                    if (value == other) {                                                                              \
                        return EILSEQ;                                                                                 \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (out_next_row != nullptr) {                                                                                 \
            *out_next_row = maps->length;                                                                              \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
UNSIGNED_INDEX_CARRIERS(DEFINE_MAP_SMALL_KEYS)
#undef DEFINE_MAP_SMALL_KEYS

static int map_small_rows_check(struct dora_pool pool,
                                const struct dora_vector* keys,
                                const struct dora_vector_view* maps,
                                struct vector_map_scratch* scratch,
                                uint32_t width,
                                uint64_t* out_next_row) {
    switch (width) {
    case 1:
        return map_small_keys_u8(pool, keys, maps, scratch, out_next_row);
    case 2:
        return map_small_keys_u16(pool, keys, maps, scratch, out_next_row);
    case 4:
        return map_small_keys_u32(pool, keys, maps, scratch, out_next_row);
    case 8:
        return map_small_keys_u64(pool, keys, maps, scratch, out_next_row);
    default:
        return EINVAL;
    }
}

void dora_vector_map_scratch_release(struct vector_map_scratch* scratch) {
    dora_keyset_release(&scratch->set);
    dora_scratch_scope_end(&scratch->scope);
    *scratch = (struct vector_map_scratch){};
}

int dora_vector_check_map_rows(struct dora_pool pool,
                               const struct dora_vector* keys,
                               const struct dora_vector_view* maps,
                               struct vector_map_scratch* scratch) {
    const bool empty = maps->length == 0 || maps->null_count == (int64_t)maps->length;
    if (empty && scratch->keys.handle.slot_index == 0) {
        return DORA_OK;
    }
    int status = map_key_is_bound(keys, scratch) ? DORA_OK : map_key_bind(keys, scratch);
    if (status != DORA_OK || empty) {
        return status;
    }
    if (scratch->view.null_count == 0 && scratch->view.validity.data == nullptr &&
        scratch->view.selection.indices == nullptr) {
        const uint32_t width = map_key_width(scratch);
        if (width != 0) {
            return map_small_rows_check(pool, keys, maps, scratch, width, nullptr);
        }
    }
    for (uint64_t row = 0; status == DORA_OK && row < maps->length; ++row) {
        if (!dora_vector_view_is_null(maps, row)) {
            status = dora_vector_check_map_keys(pool, keys, dora_vector_view_get_elements(maps, row), false, scratch);
        }
    }
    return status;
}

int dora_vector_check_map_vector(struct dora_pool pool, const struct dora_vector* map) {
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(map, &image);
    struct dora_vector host = {}, keys = {};
    struct dora_vector_view maps = {};
    struct vector_map_scratch scratch = {};
    const bool device = status == DORA_OK && image->ownership.view.placement.access == DORA_ACCESS_DEVICE;
    if (status == DORA_OK && device) {
        status = dora_vector_copy(map, pool, &host);
    }
    const struct dora_vector* const checked = device ? &host : map;
    if (status == DORA_OK) {
        status = dora_vector_get_child(checked, 0, &keys);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(checked, &maps);
    }
    if (status == DORA_OK) {
        status = dora_vector_check_map_rows(pool, &keys, &maps, &scratch);
    }
    dora_vector_map_scratch_release(&scratch);
    dora_vector_release(&host);
    return status;
}

static int foreign_map(struct dora_pool pool,
                       const struct dora_schema* schema,
                       const struct vector_input* input,
                       const struct dora_vector_view* view) {
    if (view->length == 0 || view->null_count == (int64_t)view->length) {
        return DORA_OK;
    }
    if (view->placement.access == DORA_ACCESS_DEVICE) {
        struct dora_vector host = {};
        int status = dora_vector_copy_foreign(pool,
                                              schema,
                                              input->foreign,
                                              input->foreign_node_index,
                                              input->owner,
                                              &host,
                                              nullptr);
        if (status == DORA_OK) {
            status = dora_vector_check_map_vector(pool, &host);
        }
        dora_vector_release(&host);
        return status;
    }
    const struct admission_child root =
        {.view = *view, .owner = input->owner, .tree = input->foreign, .node = input->foreign_node_index};
    struct admission_child entries = {}, keys = {};
    input_child(&root, 0, &entries);
    input_child(&entries, 0, &keys);
    int status = DORA_OK;
    struct vector_map_scratch scratch = {.view = keys.view};
    map_key_bind_view(&input->foreign->nodes[keys.node].field.type, &scratch);
    uint64_t first = 0;
    const uint32_t width = map_key_width(&scratch);
    if (scratch.view.null_count == 0 && scratch.view.validity.data == nullptr &&
        scratch.view.selection.indices == nullptr && width != 0) {
        status = map_small_rows_check(pool, nullptr, view, &scratch, width, &first);
    }
    for (uint64_t row = first; status == DORA_OK && row < view->length; ++row) {
        if (!dora_vector_view_is_null(view, row)) {
            status = map_key_check(pool, keys, dora_vector_view_get_elements(view, row), &scratch);
        }
    }
    dora_vector_map_scratch_release(&scratch);
    return status;
}

static void child_slice(const struct vector_bound_input* tree, uint32_t index, struct dora_range rows) {
    struct vector_bound_node* const node = &tree->nodes[index];
    struct dora_vector_view* const view = &node->view;
    view->offset += rows.offset;
    view->validity_bit_offset += rows.offset;
    if (view->null_count == (int64_t)view->length) {
        view->null_count = (int64_t)rows.length;
    } else if (view->null_count != 0 && (rows.offset != 0 || rows.length != view->length)) {
        view->null_count = -1;
    }
    view->length = rows.length;
    for (uint32_t child = index + 1; node->field.layout == DORA_LAYOUT_STRUCT && child < node->subtree_end_node_index;
         child = tree->nodes[child].subtree_end_node_index) {
        child_slice(tree, child, rows);
    }
}

int dora_vector_admit_foreign(struct dora_pool pool,
                              const struct dora_schema* schema,
                              const struct vector_input* input,
                              struct dora_view_buffer* buffers,
                              struct vector_bound_node* in_out_node,
                              struct dora_error* out_error) {
    struct vector_node_ownership owned = {};
    int status = dora_vector_admit(pool, schema, &owned, input, buffers);
    if (status == DORA_OK && input->field->layout == DORA_LAYOUT_STRUCT) {
        const struct vector_bound_input* const tree = input->foreign;
        for (uint32_t child = input->foreign_node_index + 1; child < in_out_node->subtree_end_node_index;
             child = tree->nodes[child].subtree_end_node_index) {
            child_slice(tree, child, (struct dora_range){.offset = input->offset, .length = input->row_count});
        }
    }
    struct dora_schema storage_schema = {};
    struct dora_allocation parameters = {};
    if (status == DORA_OK && input->field->type.extension_name.byte_count != 0) {
        status = dora_extension_find(pool, input->field->type.extension_name, &in_out_node->extension);
        if (status == DORA_OK && in_out_node->extension != nullptr && in_out_node->extension->bind != nullptr) {
            struct dora_schema values = dora_schema_borrow(schema);
            struct dora_schema_field storage = *input->field;
            while (status == DORA_OK &&
                   (storage.layout == DORA_LAYOUT_DICTIONARY || storage.layout == DORA_LAYOUT_RUN_END)) {
                struct dora_schema child = {};
                status =
                    dora_schema_get_physical_child(&values, storage.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
                if (status == DORA_OK) {
                    values = child;
                    status = dora_schema_get_field(&values, &storage);
                }
            }
            storage.type.extension_name = (struct dora_bytes){};
            storage.type.extension_metadata = (struct dora_bytes){};
            if (status == DORA_OK) {
                status = dora_schema_init_retain_field(&storage_schema, pool, &values, &storage);
            }
            if (status == DORA_OK) {
                status = dora_vector_bind_parameters(in_out_node->extension,
                                                     pool,
                                                     &storage_schema,
                                                     input->field,
                                                     &parameters,
                                                     out_error);
            }
        }
    }
    dora_allocation_release(&parameters);
    dora_schema_release(&storage_schema);
    in_out_node->field = *input->field;
    in_out_node->view = owned.view;
    if (in_out_node->view.union_members != nullptr) {
        uint8_t* const members = (uint8_t*)&buffers[input->buffer_count];
        memcpy(members, owned.union_members, sizeof(owned.union_members));
        in_out_node->view.union_members = members;
    }
    if (status == DORA_OK && input->field->type.id == DORA_TYPE_MAP && input->field->layout != DORA_LAYOUT_DICTIONARY &&
        input->field->layout != DORA_LAYOUT_RUN_END) {
        status = foreign_map(pool, schema, input, &in_out_node->view);
    }
    return status;
}
