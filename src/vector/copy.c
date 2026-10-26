// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/compute_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "hash/keyset.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "schema/schema_internal.h"
#include "types/array_count.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t COPY_VALIDITY_CACHE_BYTES = 256;

struct copy_rows {
    const uint64_t* indices;
    const uint8_t* activity;
    const uint8_t* admission_activity;
    const uint8_t* effective_validity;
    uint64_t length;
};

struct copy_facts {
    enum vector_content_state content_state;
    bool are_required_rows_admitted;
    bool are_map_rows_admitted;
};

struct copy_plan {
    struct dora_pool pool;
    struct dora_error* error;
    const enum dora_layout* layouts;
    const struct dora_schema* schema;
    const struct dora_schema* admitted_schema;
    const struct vector_bound_input* foreign;
    struct dora_buffer_owner* foreign_owner;
    bool should_decode;
};

static uint64_t copy_row(struct copy_rows rows, uint64_t index) {
    return rows.indices == nullptr ? index : rows.indices[index];
}

static uint64_t copy_active_row(struct copy_rows rows, uint64_t index) {
    return rows.activity != nullptr && !dora_bitmap_test(rows.activity, index) ? UINT64_MAX - 1 : copy_row(rows, index);
}

static bool copy_is_null(const struct dora_vector_view* view, struct copy_rows rows, uint64_t index) {
    if (rows.effective_validity != nullptr) {
        return !dora_bitmap_test(rows.effective_validity, index);
    }
    return dora_vector_view_is_null(view, copy_row(rows, index));
}

static bool copy_row_is_admitted(struct copy_rows rows, uint64_t index) {
    return copy_active_row(rows, index) != UINT64_MAX - 1 &&
           (rows.admission_activity == nullptr || dora_bitmap_test(rows.admission_activity, index));
}

static int
copy_allocate(struct dora_pool pool, uint64_t count, size_t width, bool bitmap, struct dora_allocation* out) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, width)) {
        return EOVERFLOW;
    }
    const int status = dora_pool_allocate(pool, bytes, out);
    if (status == DORA_OK && bitmap) {
        bitmap_fill_bytes(out->address, bytes, false);
    }
    return status;
}

static int
copy_activity_row(struct dora_pool pool, struct copy_rows rows, uint64_t index, struct dora_allocation* out_activity) {
    if (copy_row_is_admitted(rows, index)) {
        return DORA_OK;
    }
    if (out_activity->address == nullptr) {
        const int status =
            copy_allocate(pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, out_activity);
        if (status != DORA_OK) {
            return status;
        }
        dora_bitmap_set_range(out_activity->address, 0, rows.length, true);
    }
    bitmap_clear(out_activity->address, index);
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Copies
//
//------------------------------------------------------------------------------

struct copy_transport {
    const struct dora_schema* schema;
    struct dora_vector_view view;
    struct dora_vector_view own;
    struct dora_buffer_owner* owner;
    const uint8_t* effective_validity;
    struct dora_allocation metadata[5];
    struct dora_pool source_pool;
    enum vector_content_state content_state;
};

static int
copy_read(struct dora_buffer_owner* owner, struct dora_view_buffer buffer, struct dora_range bytes, void* destination) {
    if (bytes.offset > buffer.byte_count || bytes.length > buffer.byte_count - bytes.offset) {
        return EINVAL;
    }
    if (bytes.length == 0) {
        return DORA_OK;
    }
    if (owner != nullptr) {
        const struct dora_device_region region =
            {.data = buffer.data, .byte_offset = bytes.offset, .byte_count = bytes.length};
        return owner->read_host(owner, region, destination);
    }
    memcpy(destination, (const uint8_t*)buffer.data + bytes.offset, bytes.length);
    return DORA_OK;
}

static int copy_stage(const struct copy_plan* plan,
                      struct copy_transport* transport,
                      struct dora_view_buffer* buffer,
                      uint32_t slot,
                      struct copy_rows rows,
                      size_t width,
                      bool pairs) {
    if (buffer->data == nullptr) {
        return DORA_OK;
    }
    const struct dora_view_buffer original = *buffer;
    int status = copy_allocate(plan->pool, original.byte_count, 1, width == 0, &transport->metadata[slot]);
    uint8_t* const destination = transport->metadata[slot].address;
    if (status == DORA_OK && width == 0) {
        status = copy_read(transport->owner, original, (struct dora_range){.length = original.byte_count}, destination);
    } else {
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            if (row >= UINT64_MAX - 1 || copy_is_null(&transport->view, rows, index)) {
                continue;
            }
            const uint64_t offset = dora_vector_view_get_physical_row(&transport->view, row) * width;
            status = copy_read(transport->owner,
                               original,
                               (struct dora_range){.offset = offset, .length = width * (pairs ? 2 : 1)},
                               destination + offset);
        }
    }
    if (status == DORA_OK) {
        *buffer = (struct dora_view_buffer){
            .data = destination,
            .byte_count = original.byte_count,
            .readable_byte_count = transport->metadata[slot].capacity_bytes
        };
    }
    return status;
}

static int copy_validity_provider(struct vector_image* image,
                                  struct dora_view_buffer bitmap,
                                  struct dora_buffer_owner** out_owner,
                                  bool* out_host) {
    *out_owner = nullptr;
    *out_host = image->ownership.view.placement.access != DORA_ACCESS_DEVICE;
    for (;;) {
        if (dora_vector_owns_host_validity(image, bitmap)) {
            *out_host = true;
        }
        if (*out_owner == nullptr) {
            *out_owner = image->buffer_image->wrapped_owner;
        }
        if (image->ownership.backing.handle.slot_index == 0) {
            return DORA_OK;
        }
        const int status = dora_vector_resolve(&image->ownership.backing, &image);
        if (status != DORA_OK) {
            return status;
        }
    }
}

static int
copy_effective_word(const struct dora_vector* source, const uint64_t* rows, uint32_t count, uint64_t* out_word) {
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = dora_vector_bind_coordinates(source, &view);
    }
    if (status != DORA_OK) {
        return status;
    }
    uint64_t mapped[64] = {};
    for (uint32_t index = 0; index < count; ++index) {
        if (rows[index] >= view.length) {
            return EINVAL;
        }
    }
    if (image->ownership.requires_effective_validity) {
        struct vector_image* recipe = image;
        uint64_t offset = source->rows.offset;
        while (recipe->ownership.deferred_validity == nullptr && recipe->ownership.backing.handle.slot_index != 0) {
            if (ckd_add(&offset, offset, recipe->ownership.backing.rows.offset)) {
                return EOVERFLOW;
            }
            status = dora_vector_resolve(&recipe->ownership.backing, &recipe);
            if (status != DORA_OK) {
                return status;
            }
        }
        const struct vector_deferred_validity* const inputs = recipe->ownership.deferred_validity;
        if (inputs == nullptr) {
            return EINVAL;
        }
        for (uint32_t index = 0; index < count; ++index) {
            if (ckd_add(&mapped[index], offset, rows[index]) || mapped[index] >= recipe->ownership.view.length) {
                return EINVAL;
            }
            if (inputs->selection != nullptr) {
                mapped[index] = inputs->selection[mapped[index]];
            }
        }
        uint64_t own = 0;
        uint64_t ancestor = UINT64_MAX;
        status = copy_effective_word(&inputs->inputs[0], mapped, count, &own);
        if (status == DORA_OK && inputs->inputs[1].handle.slot_index != 0) {
            status = copy_effective_word(&inputs->inputs[1], mapped, count, &ancestor);
        }
        if (status == DORA_OK) {
            *out_word = own & ancestor;
        }
        return status;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const uint64_t row = dora_vector_view_get_physical_row(&view, rows[index]) - view.offset;
        if (ckd_add(&mapped[index], view.validity_bit_offset, row)) {
            return EOVERFLOW;
        }
    }
    if (view.layout == DORA_LAYOUT_NULL) {
        *out_word = 0;
        return DORA_OK;
    }
    if (view.validity.data == nullptr || view.null_count == 0) {
        *out_word = bitmap_mask_low(count);
        return DORA_OK;
    }
    struct dora_buffer_owner* owner = nullptr;
    bool host = false;
    status = copy_validity_provider(image, view.validity, &owner, &host);
    if (status != DORA_OK || (!host && (owner == nullptr || owner->read_host == nullptr))) {
        return status != DORA_OK ? status : ENOTSUP;
    }
    uint8_t cache[COPY_VALIDITY_CACHE_BYTES] = {};
    uint64_t cache_offset = 0;
    uint64_t cache_length = 0;
    uint64_t word = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint64_t byte = bitmap_byte_of(mapped[index]);
        if (byte >= view.validity.byte_count) {
            return EINVAL;
        }
        bool valid = false;
        if (host) {
            valid = dora_bitmap_test(view.validity.data, mapped[index]);
        } else {
            if (byte < cache_offset || byte - cache_offset >= cache_length) {
                cache_offset = byte / sizeof(cache) * sizeof(cache);
                cache_length = view.validity.byte_count - cache_offset;
                if (cache_length > sizeof(cache)) {
                    cache_length = sizeof(cache);
                }
                status = copy_read(owner,
                                   view.validity,
                                   (struct dora_range){.offset = cache_offset, .length = cache_length},
                                   cache);
                if (status != DORA_OK) {
                    return status;
                }
            }
            valid = dora_bitmap_test(cache, mapped[index] - cache_offset * 8);
        }
        bitmap_assign(&word, index, valid);
    }
    *out_word = word;
    return DORA_OK;
}

static int copy_deferred_mask(const struct copy_plan* plan,
                              const struct dora_vector* source,
                              struct copy_rows rows,
                              struct copy_transport* transport) {
    int status =
        copy_allocate(plan->pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, &transport->metadata[1]);
    uint64_t source_rows[64] = {};
    uint64_t output_bits[64] = {};
    uint64_t position = 0;
    while (status == DORA_OK && position < rows.length) {
        uint32_t count = 0;
        while (position < rows.length && count < 64) {
            const uint64_t output = position++;
            const uint64_t row = copy_active_row(rows, output);
            if (row < UINT64_MAX - 1) {
                source_rows[count] = row;
                output_bits[count] = output;
                ++count;
            }
        }
        uint64_t word = 0;
        if (count != 0) {
            status = copy_effective_word(source, source_rows, count, &word);
        }
        for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
            bitmap_assign(transport->metadata[1].address,
                          output_bits[index],
                          dora_bitmap_test((const uint8_t*)&word, index));
        }
    }
    if (status == DORA_OK) {
        transport->effective_validity = transport->metadata[1].address;
    }
    return status;
}

static int copy_transport_stage(const struct copy_plan* plan,
                                struct copy_rows rows,
                                struct copy_transport* transport,
                                bool effective_host) {
    const void* const own_validity = transport->own.validity.data;
    int status = copy_stage(plan, transport, &transport->own.validity, 0, rows, 0, false);
    if (status == DORA_OK && transport->view.validity.data != nullptr) {
        if (transport->view.validity.data == own_validity) {
            transport->view.validity = transport->own.validity;
        } else if (!effective_host) {
            status = copy_stage(plan, transport, &transport->view.validity, 1, rows, 0, false);
        }
    }
    const enum dora_layout layout = transport->view.layout;
    if (status == DORA_OK && transport->view.offsets.data != nullptr) {
        const size_t width = layout == DORA_LAYOUT_LARGE_BINARY || layout == DORA_LAYOUT_LARGE_LIST ||
                                     layout == DORA_LAYOUT_LARGE_LIST_VIEW
                                 ? 8
                                 : 4;
        const bool pairs = layout == DORA_LAYOUT_BINARY || layout == DORA_LAYOUT_LARGE_BINARY ||
                           layout == DORA_LAYOUT_LIST || layout == DORA_LAYOUT_LARGE_LIST;
        status = copy_stage(plan, transport, &transport->view.offsets, 2, rows, width, pairs);
    }
    if (status == DORA_OK && transport->view.sizes.data != nullptr) {
        status = copy_stage(plan,
                            transport,
                            &transport->view.sizes,
                            3,
                            rows,
                            layout == DORA_LAYOUT_LARGE_LIST_VIEW ? 8 : 4,
                            false);
    } else if (status == DORA_OK && transport->view.type_ids.data != nullptr) {
        status = copy_stage(plan, transport, &transport->view.type_ids, 3, rows, 1, false);
    }
    if (status == DORA_OK && layout == DORA_LAYOUT_BINARY_VIEW) {
        status =
            copy_stage(plan, transport, &transport->view.values, 4, rows, sizeof(struct dora_binary_view_entry), false);
    }
    return status;
}

static int copy_transport_bind(const struct copy_plan* plan,
                               const struct dora_vector* source,
                               struct copy_rows rows,
                               struct copy_transport* transport) {
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    if (status != DORA_OK) {
        return status;
    }
    transport->schema = &image->schema;
    transport->source_pool = image->allocation.pool;
    transport->content_state = image->node.content_state;
    const bool device = image->ownership.view.placement.access == DORA_ACCESS_DEVICE;
    const bool deferred = image->ownership.requires_effective_validity;
    if (!device && !deferred) {
        status = dora_vector_get_view(source, &transport->view);
    } else {
        status = dora_vector_settle(source);
        if (status == DORA_OK) {
            status = dora_vector_bind_coordinates(source, &transport->view);
        }
    }
    transport->own = transport->view;
    transport->own.validity =
        image->ownership.own_null_count == 0 ? (struct dora_view_buffer){} : image->ownership.own_validity;
    transport->own.validity_bit_offset = image->ownership.own_validity_bit_offset +
                                         (image->ownership.view.selection.indices == nullptr ? source->rows.offset : 0);
    transport->own.null_count = image->ownership.own_null_count;
    if (status == DORA_OK && deferred) {
        status = copy_deferred_mask(plan, source, rows, transport);
        rows.effective_validity = transport->effective_validity;
    }
    if (status != DORA_OK || !device) {
        return status;
    }
    bool effective_host = false;
    status = copy_validity_provider(image, transport->view.validity, &transport->owner, &effective_host);
    effective_host = effective_host || deferred;
    if (status != DORA_OK) {
        return status;
    }
    if ((transport->owner == nullptr || transport->owner->read_host == nullptr) &&
        transport->view.layout != DORA_LAYOUT_RUN_END) {
        return ENOTSUP;
    }
    return copy_transport_stage(plan, rows, transport, effective_host);
}

static int copy_stage_run_ends(const struct copy_plan* plan,
                               struct dora_buffer_owner* owner,
                               const struct dora_vector_view* runs,
                               struct copy_transport* transport) {
    int status = DORA_OK;
    if (owner != nullptr) {
        const size_t width = transport->view.byte_width;
        status = copy_allocate(plan->pool, runs->length, width, false, &transport->metadata[4]);
        const size_t bytes = runs->length * width;
        if (status == DORA_OK) {
            status = copy_read(owner,
                               runs->values,
                               (struct dora_range){.offset = runs->offset * width, .length = bytes},
                               transport->metadata[4].address);
        }
        if (status == DORA_OK) {
            transport->view.values = (struct dora_view_buffer){
                .data = transport->metadata[4].address,
                .byte_count = bytes,
                .readable_byte_count = transport->metadata[4].capacity_bytes
            };
            transport->view.run_end_element_offset = 0;
        }
    }
    return status;
}

static int
copy_stage_runs(const struct copy_plan* plan, const struct dora_vector* source, struct copy_transport* transport) {
    if (transport->view.layout != DORA_LAYOUT_RUN_END) {
        return DORA_OK;
    }
    struct dora_vector runs = {};
    int status = dora_vector_get_physical_child(source, 0, &runs);
    struct copy_transport endpoints = {};
    if (status == DORA_OK) {
        status = copy_transport_bind(plan, &runs, (struct copy_rows){}, &endpoints);
    }
    if (status == DORA_OK) {
        status = copy_stage_run_ends(plan, endpoints.owner, &endpoints.view, transport);
    }
    for (uint32_t index = 0; index < ARRAY_COUNT(endpoints.metadata); ++index) {
        dora_allocation_release(&endpoints.metadata[index]);
    }
    return status;
}

static struct dora_bytes copy_bytes_value(const struct copy_transport* transport, uint64_t row) {
    const struct dora_vector_view* const view = &transport->view;
    if (transport->owner == nullptr) {
        return dora_vector_view_get_bytes(view, row);
    }
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    if (view->layout == DORA_LAYOUT_BINARY_VIEW) {
        int32_t count = 0;
        memcpy(&count,
               (const uint8_t*)view->values.data + physical * sizeof(struct dora_binary_view_entry),
               sizeof(count));
        return (struct dora_bytes){.byte_count = count};
    }
    if (view->layout == DORA_LAYOUT_BINARY) {
        int32_t offsets[2] = {};
        memcpy(offsets, (const uint8_t*)view->offsets.data + physical * sizeof(int32_t), sizeof(offsets));
        return (struct dora_bytes){.byte_count = offsets[1] - offsets[0]};
    }
    int64_t offsets[2] = {};
    memcpy(offsets, (const uint8_t*)view->offsets.data + physical * sizeof(int64_t), sizeof(offsets));
    return (struct dora_bytes){.byte_count = offsets[1] - offsets[0]};
}

static int
copy_bytes_read(const struct copy_transport* transport, uint64_t row, struct dora_bytes bytes, void* destination) {
    const struct dora_vector_view* const view = &transport->view;
    if (transport->owner == nullptr || bytes.byte_count == 0) {
        if (bytes.byte_count != 0 && bytes.data == nullptr) {
            return EINVAL;
        }
        if (bytes.byte_count != 0) {
            memcpy(destination, bytes.data, bytes.byte_count);
        }
        return DORA_OK;
    }
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    uint32_t buffer = 0;
    uint64_t offset = 0;
    if (view->layout == DORA_LAYOUT_BINARY_VIEW) {
        struct dora_binary_view_entry entry = {};
        memcpy(&entry, (const uint8_t*)view->values.data + physical * sizeof(entry), sizeof(entry));
        if (entry.byte_count <= VIEW_INLINE_BYTE_COUNT) {
            memcpy(destination, entry.payload.inline_bytes, bytes.byte_count);
            return DORA_OK;
        }
        buffer = entry.payload.reference.buffer_index;
        offset = entry.payload.reference.byte_offset;
    } else if (view->layout == DORA_LAYOUT_BINARY) {
        int32_t begin = 0;
        memcpy(&begin, (const uint8_t*)view->offsets.data + physical * sizeof(begin), sizeof(begin));
        offset = begin;
    } else {
        int64_t begin = 0;
        memcpy(&begin, (const uint8_t*)view->offsets.data + physical * sizeof(begin), sizeof(begin));
        offset = begin;
    }
    return copy_read(transport->owner,
                     view->data_buffers[buffer],
                     (struct dora_range){.offset = offset, .length = bytes.byte_count},
                     destination);
}

static int copy_tree_size(const struct dora_vector* source, uint32_t* out_count) {
    uint32_t count = 1;
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    const uint32_t children = status == DORA_OK ? image->child_vectors.entry_count : 0;
    for (uint32_t index = 0; status == DORA_OK && index < children; ++index) {
        struct dora_vector child = {};
        uint32_t child_count = 0;
        status = dora_vector_get_physical_child(source, index, &child);
        if (status == DORA_OK) {
            status = copy_tree_size(&child, &child_count);
        }
        if (status == DORA_OK && ckd_add(&count, count, child_count)) {
            status = EOVERFLOW;
        }
    }
    if (status == DORA_OK) {
        *out_count = count;
    }
    return status;
}

static int
copy_child_position(const struct dora_vector* source, uint32_t position, uint32_t* out_position, uint32_t child_index) {
    if (position == UINT32_MAX) {
        *out_position = UINT32_MAX;
        return DORA_OK;
    }
    uint32_t next = position + 1;
    for (uint32_t index = 0; index < child_index; ++index) {
        struct dora_vector child = {};
        uint32_t count = 0;
        int status = dora_vector_get_physical_child(source, index, &child);
        if (status == DORA_OK) {
            status = copy_tree_size(&child, &count);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (ckd_add(&next, next, count)) {
            return EOVERFLOW;
        }
    }
    *out_position = next;
    return DORA_OK;
}

static bool
copy_is_contiguous(struct copy_rows rows, const struct dora_vector_view* view, const struct copy_transport* transport) {
    return rows.indices == nullptr && rows.activity == nullptr && rows.admission_activity == nullptr &&
           rows.effective_validity == nullptr && view->selection.indices == nullptr && transport->owner == nullptr;
}

static int copy_mask(struct dora_pool pool,
                     const struct dora_vector_view* view,
                     struct copy_rows rows,
                     const struct copy_transport* transport,
                     struct dora_allocation* out_own,
                     int64_t* out_own_count,
                     struct dora_allocation* out_activity) {
    const struct dora_vector_view* const own = &transport->own;
    if (copy_is_contiguous(rows, view, transport) && own->validity.data == view->validity.data &&
        own->validity_bit_offset == view->validity_bit_offset) {
        *out_own_count = 0;
        if (own->validity.data == nullptr || rows.length == 0) {
            return DORA_OK;
        }
        const int status = copy_allocate(pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, out_own);
        if (status != DORA_OK) {
            return status;
        }
        *out_own_count = (int64_t)(rows.length - dora_bitmap_copy_count(rows.length,
                                                                        out_own->address,
                                                                        0,
                                                                        own->validity.data,
                                                                        own->validity_bit_offset));
        return DORA_OK;
    }
    int64_t own_count = 0;
    for (uint64_t index = 0; index < rows.length; ++index) {
        const int activity_status = copy_activity_row(pool, rows, index, out_activity);
        if (activity_status != DORA_OK) {
            return activity_status;
        }
        const uint64_t row = copy_row(rows, index);
        const bool inactive =
            row == UINT64_MAX - 1 || (rows.activity != nullptr && !dora_bitmap_test(rows.activity, index));
        const bool own_null =
            row == UINT64_MAX || inactive || (row < UINT64_MAX - 1 && copy_is_null(view, rows, index));
        if (own_null && out_own->address == nullptr) {
            const int status = copy_allocate(pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, out_own);
            if (status != DORA_OK) {
                return status;
            }
            dora_bitmap_set_range(out_own->address, 0, rows.length, true);
        }
        if (own_null) {
            bitmap_clear(out_own->address, index);
            ++own_count;
        }
    }
    *out_own_count = own_count;
    return DORA_OK;
}

static int copy_inherited_mask(struct dora_pool pool,
                               const struct dora_vector_view* view,
                               const struct dora_vector_view* own,
                               struct copy_rows rows,
                               struct dora_allocation* out,
                               int64_t* out_count,
                               struct dora_allocation* out_activity) {
    int64_t count = 0;
    for (uint64_t index = 0; index < rows.length; ++index) {
        const int activity_status = copy_activity_row(pool, rows, index, out_activity);
        if (activity_status != DORA_OK) {
            return activity_status;
        }
        const uint64_t row = copy_active_row(rows, index);
        const bool inactive = row == UINT64_MAX - 1 || (row < UINT64_MAX - 1 && copy_is_null(view, rows, index) &&
                                                        !dora_vector_view_is_null(own, row));
        if (inactive && out->address == nullptr) {
            const int status = copy_allocate(pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, out);
            if (status != DORA_OK) {
                return status;
            }
            dora_bitmap_set_range(out->address, 0, rows.length, true);
        }
        if (inactive) {
            bitmap_clear(out->address, index);
            ++count;
        }
    }
    *out_count = count;
    return DORA_OK;
}

static int copy_store_index(enum dora_type type, void* data, uint64_t row, uint64_t value) {
    switch (type) {
// Returns DORA_OK or EOVERFLOW from the enclosing function.
// Index carriers: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define COPY_INDEX(K, T, LIMIT)                                                                                        \
    case K: {                                                                                                          \
        if (value > (LIMIT)) {                                                                                         \
            return EOVERFLOW;                                                                                          \
        }                                                                                                              \
        memcpy((uint8_t*)data + row * sizeof(T), &(T){(T)value}, sizeof(T));                                           \
        return DORA_OK;                                                                                                \
    }
        COPY_INDEX(DORA_TYPE_INT8, int8_t, INT8_MAX)
        COPY_INDEX(DORA_TYPE_INT16, int16_t, INT16_MAX)
        COPY_INDEX(DORA_TYPE_INT32, int32_t, INT32_MAX)
        COPY_INDEX(DORA_TYPE_INT64, int64_t, INT64_MAX)
        COPY_INDEX(DORA_TYPE_UINT8, uint8_t, UINT8_MAX)
        COPY_INDEX(DORA_TYPE_UINT16, uint16_t, UINT16_MAX)
        COPY_INDEX(DORA_TYPE_UINT32, uint32_t, UINT32_MAX)
        COPY_INDEX(DORA_TYPE_UINT64, uint64_t, UINT64_MAX)
#undef COPY_INDEX
    default:
        return EINVAL;
    }
}

static size_t copy_index_width(enum dora_type type) {
    switch (type) {
    case DORA_TYPE_INT8:
    case DORA_TYPE_UINT8:
        return 1;
    case DORA_TYPE_INT16:
    case DORA_TYPE_UINT16:
        return 2;
    case DORA_TYPE_INT32:
    case DORA_TYPE_UINT32:
        return 4;
    case DORA_TYPE_INT64:
    case DORA_TYPE_UINT64:
        return 8;
    default:
        return 0;
    }
}

static bool copy_dictionary_ordinals_fit(const struct dora_schema_field* field, uint64_t count) {
    if (count == 0) {
        return true;
    }
    switch (field->index_type) {
    case DORA_TYPE_INT8:
        return count - 1 <= INT8_MAX;
    case DORA_TYPE_INT16:
        return count - 1 <= INT16_MAX;
    case DORA_TYPE_INT32:
        return count - 1 <= INT32_MAX;
    case DORA_TYPE_INT64:
        return count - 1 <= INT64_MAX;
    case DORA_TYPE_UINT8:
        return count - 1 <= UINT8_MAX;
    case DORA_TYPE_UINT16:
        return count - 1 <= UINT16_MAX;
    case DORA_TYPE_UINT32:
        return count - 1 <= UINT32_MAX;
    case DORA_TYPE_UINT64:
        return true;
    default:
        return false;
    }
}

static int copy_publish(const struct copy_plan* plan,
                        const struct copy_facts* facts,
                        struct dora_schema_field field,
                        struct copy_rows rows,
                        struct dora_allocation* buffers,
                        uint32_t buffer_count,
                        struct dora_vector* children,
                        int64_t own_count,
                        struct dora_allocation* effective,
                        int64_t effective_count,
                        const struct dora_allocation* activity,
                        struct dora_vector* out) {
    const struct dora_bytes active_rows = {.data = activity->address, .byte_count = bitmap_bytes_for_bits(rows.length)};
    const struct vector_input input = {
        .field = &field,
        .admitted_schema = plan->admitted_schema,
        .row_count = rows.length,
        .null_count = own_count,
        .allocations = buffers,
        .buffer_count = buffer_count,
        .children = children,
        .child_count = field.child_count,
        .ancestor_activity = activity->address == nullptr ? nullptr : &active_rows,
        .are_required_rows_admitted = facts->are_required_rows_admitted,
        .are_map_rows_admitted = facts->are_map_rows_admitted,
        .are_children_folded = field.layout == DORA_LAYOUT_STRUCT,
        .effective_validity = effective != nullptr && effective->address != nullptr ? effective : nullptr,
        .effective_null_count = effective_count,
        .content_state = facts->content_state
    };
    return dora_vector_init_move_input(out, plan->pool, &input, buffers, plan->error);
}

struct copy_child_input {
    struct dora_vector source;
    struct dora_schema schema;
    struct dora_schema_field field;
    uint32_t position;
};

static int copy_child_prepare(const struct copy_plan* plan,
                              const struct dora_vector* source,
                              uint32_t child_index,
                              uint32_t position,
                              struct copy_child_input* out) {
    if (plan->foreign != nullptr) {
        out->position = dora_vector_input_child(position, plan->foreign, child_index);
        out->field = plan->foreign->nodes[out->position].field;
        return dora_schema_get_physical_child(plan->schema, child_index, &out->schema);
    }
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    if (status == DORA_OK) {
        status = image->ownership.view.layout == DORA_LAYOUT_STRUCT
                     ? dora_vector_get_child(source, child_index, &out->source)
                     : dora_vector_get_physical_child(source, child_index, &out->source);
    }
    if (status == DORA_OK) {
        status = copy_child_position(source, position, &out->position, child_index);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_physical_child(plan->schema != nullptr ? plan->schema : &image->schema,
                                                child_index,
                                                &out->schema);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(&out->schema, &out->field);
    }
    return status;
}

// Dictionary indices: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_COPY_PEEL_DICTIONARY(S, T, P, NEGATIVE)                                                                 \
    static int copy_peel_dictionary_##S(const struct dora_vector_view* view,                                           \
                                        struct copy_rows rows,                                                         \
                                        uint64_t* mapped) {                                                            \
        for (uint64_t index = 0; index < rows.length; ++index) {                                                       \
            const uint64_t row = copy_active_row(rows, index);                                                         \
            if (row >= UINT64_MAX - 1 || copy_is_null(view, rows, index)) {                                            \
                mapped[index] = row == UINT64_MAX - 1 ? UINT64_MAX - 1 : UINT64_MAX;                                   \
                continue;                                                                                              \
            }                                                                                                          \
            const T value = type_load_##S(view->values.data, dora_vector_view_get_physical_row(view, row));            \
            if (NEGATIVE(value)) {                                                                                     \
                return EINVAL;                                                                                         \
            }                                                                                                          \
            mapped[index] = (uint64_t)value;                                                                           \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }

SIGNED_INDEX_CARRIERS(DEFINE_COPY_PEEL_DICTIONARY, NEGATIVE_SIGNED)
UNSIGNED_INDEX_CARRIERS(DEFINE_COPY_PEEL_DICTIONARY, NEGATIVE_UNSIGNED)

#undef DEFINE_COPY_PEEL_DICTIONARY

static int copy_peel_prepare(const struct copy_plan* plan,
                             const struct dora_vector* source,
                             const struct dora_vector_view* view,
                             struct copy_rows rows,
                             uint32_t position,
                             struct dora_allocation* mapping,
                             struct copy_child_input* child) {
    int status = copy_allocate(plan->pool, rows.length, sizeof(uint64_t), false, mapping);
    uint64_t* const mapped = mapping->address;
    if (status == DORA_OK && view->layout == DORA_LAYOUT_DICTIONARY) {
        switch (view->physical_type) {
#define COPY_PEEL_DICTIONARY_CASE(S, T, P, ...)                                                                        \
    case P:                                                                                                            \
        status = copy_peel_dictionary_##S(__VA_ARGS__);                                                                \
        break;
            SIGNED_INDEX_CARRIERS(COPY_PEEL_DICTIONARY_CASE, view, rows, mapped)
            UNSIGNED_INDEX_CARRIERS(COPY_PEEL_DICTIONARY_CASE, view, rows, mapped)
#undef COPY_PEEL_DICTIONARY_CASE
        default:
            status = EINVAL;
            break;
        }
    } else if (status == DORA_OK && rows.indices == nullptr && rows.activity == nullptr &&
               rows.effective_validity == nullptr && view->validity.data == nullptr &&
               view->selection.indices == nullptr) {
        struct dora_run_cursor cursor = {};
        uint64_t index = 0;
        while (index < rows.length) {
            uint64_t run = UINT64_MAX;
            const struct dora_range range = dora_vector_view_next_run(view, &cursor, &run);
            const uint64_t end = range.length == 0 || range.length > rows.length - range.offset
                                     ? rows.length
                                     : range.offset + range.length;
            for (; index < end; ++index) {
                mapped[index] = run;
            }
        }
    } else {
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            mapped[index] = row >= UINT64_MAX - 1 || copy_is_null(view, rows, index)
                                ? (row == UINT64_MAX - 1 ? UINT64_MAX - 1 : UINT64_MAX)
                                : dora_vector_view_find_run(view, row);
        }
    }
    const uint32_t child_index = view->layout == DORA_LAYOUT_DICTIONARY ? 0 : 1;
    if (status == DORA_OK) {
        status = dora_vector_get_physical_child(source, child_index, &child->source);
    }
    if (status == DORA_OK) {
        status = copy_child_position(source, position, &child->position, child_index);
    }
    return status;
}

struct copy_dedupe_rows {
    const struct dora_vector* source;
    const uint64_t* hashes;
};

static int copy_dedupe_hash(void* context, uint64_t row, uint64_t* out_hash) {
    const struct copy_dedupe_rows* const rows = context;
    bool is_null = row >= UINT64_MAX - 2;
    if (!is_null) {
        const int status = dora_vector_is_null(rows->source, row, &is_null);
        if (status != DORA_OK) {
            return status;
        }
    }
    *out_hash = is_null ? 0 : rows->hashes[row];
    return DORA_OK;
}

static int copy_dedupe_equal(void* context, uint64_t left_row, uint64_t right_row, bool* out_equal) {
    const struct copy_dedupe_rows* const rows = context;
    bool left_null = left_row >= UINT64_MAX - 2;
    bool right_null = right_row >= UINT64_MAX - 2;
    int status = left_null ? DORA_OK : dora_vector_is_null(rows->source, left_row, &left_null);
    if (status == DORA_OK && !right_null) {
        status = dora_vector_is_null(rows->source, right_row, &right_null);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (left_null || right_null) {
        *out_equal = left_null && right_null;
        return DORA_OK;
    }
    return dora_vector_compare_rows(rows->source, left_row, right_row, DORA_COMPARE_OP_EQ, out_equal);
}

// Encoded hashes must be indexed by logical row.
static int copy_hash_prepare(struct dora_pool pool,
                             const struct dora_vector* source,
                             struct dora_vector* out_hashes,
                             struct dora_vector_view* out_view) {
    int status = dora_compute_hash_in_pool(pool, source, 0, out_hashes);
    if (status == DORA_OK) {
        status = dora_vector_get_view(out_hashes, out_view);
    }
    return status;
}

struct copy_encoding {
    struct dora_allocation mapping;
    struct dora_allocation buffers[2];
    struct dora_allocation effective;
    struct dora_allocation activity;
    struct dora_allocation child_activity;
    struct dora_vector children[2];
    struct dora_vector hashes;
    struct dora_keyset set;
    struct dora_schema_field field;
    struct dora_schema_field value_field;
    struct dora_schema value_schema;
    uint64_t value_count;
    int64_t null_count;
    int64_t effective_count;
    bool should_deduplicate;
};

static int copy_encoding_prepare(const struct copy_plan* plan,
                                 const struct copy_transport* transport,
                                 struct dora_schema_field field,
                                 struct copy_rows rows,
                                 struct copy_encoding* work) {
    const struct dora_vector_view* view = &transport->view;
    const struct dora_vector_view* own = &transport->own;
    work->field = field;
    int status = copy_allocate(plan->pool, rows.length, sizeof(uint64_t), false, &work->mapping);
    if (status == DORA_OK && work->field.layout == DORA_LAYOUT_DICTIONARY) {
        status = copy_mask(plan->pool, view, rows, transport, &work->buffers[0], &work->null_count, &work->activity);
    } else if (status == DORA_OK) {
        status =
            copy_inherited_mask(plan->pool, view, own, rows, &work->effective, &work->effective_count, &work->activity);
    }
    if (status == DORA_OK && work->activity.address != nullptr) {
        status = copy_allocate(plan->pool,
                               bitmap_words_for_bits(rows.length),
                               sizeof(uint64_t),
                               true,
                               &work->child_activity);
    }
    if (plan->schema == nullptr) {
        work->field.index_type = work->field.layout == DORA_LAYOUT_DICTIONARY
                                     ? (rows.length <= UINT32_MAX ? DORA_TYPE_UINT32 : DORA_TYPE_UINT64)
                                     : DORA_TYPE_INT64;
    }
    if (status == DORA_OK) {
        status =
            copy_allocate(plan->pool, rows.length, copy_index_width(work->field.index_type), false, &work->buffers[1]);
    }
    work->should_deduplicate =
        plan->schema == nullptr ||
        (work->field.layout == DORA_LAYOUT_DICTIONARY &&
         !copy_dictionary_ordinals_fit(&work->field, rows.length - (uint64_t)work->effective_count));
    if (status == DORA_OK && work->field.layout == DORA_LAYOUT_DICTIONARY && work->should_deduplicate) {
        status = dora_keyset_init(&work->set, plan->pool, rows.length);
    }
    return status;
}

static int copy_encoding_map(const struct copy_plan* plan,
                             const struct dora_vector* source,
                             const struct copy_transport* transport,
                             struct copy_rows rows,
                             const uint64_t* hashes,
                             struct copy_encoding* work) {
    const struct dora_vector_view* view = &transport->view;
    const struct dora_vector_view* own = &transport->own;
    int status = DORA_OK;
    uint64_t* const values = work->mapping.address;
    struct copy_dedupe_rows dedupe_rows = {.source = source, .hashes = hashes};
    const struct dora_keyset_rows callbacks =
        {.context = &dedupe_rows, .hash = copy_dedupe_hash, .equal = copy_dedupe_equal};
    for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
        const uint64_t row = copy_active_row(rows, index);
        bool is_null = row >= UINT64_MAX - 1;
        if (!is_null) {
            status = dora_vector_is_null(source, row, &is_null);
        }
        if (status != DORA_OK) {
            break;
        }
        const bool own_null = row == UINT64_MAX || (row < UINT64_MAX - 1 && dora_vector_view_is_null(own, row));
        const bool inactive =
            row == UINT64_MAX - 1 || (row < UINT64_MAX - 1 && copy_is_null(view, rows, index) && !own_null);
        if (work->field.layout == DORA_LAYOUT_DICTIONARY && (own_null || inactive)) {
            status = copy_store_index(work->field.index_type, work->buffers[1].address, index, 0);
            continue;
        }
        uint64_t value = work->value_count;
        if (work->field.layout == DORA_LAYOUT_DICTIONARY && work->should_deduplicate) {
            const uint64_t probe = inactive ? UINT64_MAX - 1 : row;
            uint64_t existing = 0;
            bool duplicate = false;
            status = dora_keyset_insert(&work->set,
                                        &callbacks,
                                        probe == UINT64_MAX - 1 ? UINT64_MAX - 2 : probe,
                                        work->value_count,
                                        &existing,
                                        &duplicate);
            if (status == DORA_OK && duplicate) {
                value = existing;
            }
        } else if (work->field.layout != DORA_LAYOUT_DICTIONARY && work->should_deduplicate && work->value_count != 0) {
            value = work->value_count - 1;
            bool equal = false;
            bool prior_null = values[value] >= UINT64_MAX - 1;
            if (!prior_null) {
                status = dora_vector_is_null(source, values[value], &prior_null);
            }
            if (status == DORA_OK && (is_null || prior_null)) {
                equal = is_null && prior_null;
            } else if (status == DORA_OK) {
                status = dora_vector_compare_rows(source, row, values[value], DORA_COMPARE_OP_EQ, &equal);
            }
            if (!equal) {
                value = work->value_count;
            }
        } else if (!work->should_deduplicate) {
            value = work->value_count;
        }
        if (status != DORA_OK) {
            break;
        }
        if (value == work->value_count) {
            values[work->value_count++] = inactive ? UINT64_MAX - 1 : row;
        }
        if (!inactive && values[value] == UINT64_MAX - 1) {
            values[value] = row;
        }
        if (work->child_activity.address != nullptr && copy_row_is_admitted(rows, index) && !inactive) {
            bitmap_set(work->child_activity.address, value);
        }
        status = copy_store_index(work->field.index_type,
                                  work->buffers[1].address,
                                  work->field.layout == DORA_LAYOUT_DICTIONARY ? index : value,
                                  work->field.layout == DORA_LAYOUT_DICTIONARY ? value : index + 1);
    }
    work->value_field = work->field;
    work->value_field.layout = DORA_LAYOUT_DEFAULT;
    if (status == DORA_OK) {
        status =
            dora_type_resolve_layout(&work->field.type, DORA_LAYOUT_DEFAULT, DORA_TYPE_NULL, &work->value_field.layout);
    }
    work->value_field.index_type = DORA_TYPE_NULL;
    work->value_field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED;
    if (status == DORA_OK && plan->schema != nullptr) {
        status = dora_schema_get_physical_child(plan->schema,
                                                work->field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1,
                                                &work->value_schema);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&work->value_schema, &work->value_field);
        }
    }
    return status;
}

static int copy_encoding_publish(const struct copy_plan* plan,
                                 const struct copy_facts* facts,
                                 struct copy_rows rows,
                                 struct copy_encoding* work,
                                 struct dora_vector* out) {
    int status = DORA_OK;
    if (status == DORA_OK && work->field.layout != DORA_LAYOUT_DICTIONARY) {
        struct dora_schema_field run_field = {
            .type.id = work->field.index_type,
            .layout = DORA_LAYOUT_FIXED,
            .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE
        };
        struct dora_schema run_schema = {};
        if (plan->schema != nullptr) {
            status = dora_schema_get_physical_child(plan->schema, 0, &run_schema);
            if (status == DORA_OK) {
                status = dora_schema_get_field(&run_schema, &run_field);
            }
        }
        if (status == DORA_OK) {
            const struct vector_input input = {
                .field = &run_field,
                .row_count = work->value_count,
                .allocations = work->buffers,
                .buffer_count = 2,
                .content_state = facts->content_state
            };
            status = dora_vector_init_move_input(&work->children[0], plan->pool, &input, work->buffers, plan->error);
        }
    }
    work->field.child_count = work->field.layout == DORA_LAYOUT_DICTIONARY ? 1 : 2;
    if (status == DORA_OK) {
        status = copy_publish(plan,
                              facts,
                              work->field,
                              rows,
                              work->field.layout == DORA_LAYOUT_DICTIONARY ? work->buffers : nullptr,
                              work->field.layout == DORA_LAYOUT_DICTIONARY ? 2 : 0,
                              work->children,
                              work->null_count,
                              &work->effective,
                              work->effective_count,
                              &work->activity,
                              out);
    }
    return status;
}

static void copy_encoding_release(struct copy_encoding* work) {
    dora_vector_release(&work->children[0]);
    dora_vector_release(&work->children[1]);
    dora_allocation_release(&work->buffers[0]);
    dora_allocation_release(&work->buffers[1]);
    dora_allocation_release(&work->effective);
    dora_allocation_release(&work->activity);
    dora_allocation_release(&work->child_activity);
    dora_allocation_release(&work->mapping);
    dora_keyset_release(&work->set);
    dora_vector_release(&work->hashes);
}

struct copy_run {
    struct dora_allocation mapping;
    struct dora_allocation buffers[2];
    struct dora_allocation effective;
    struct dora_allocation activity;
    struct dora_allocation child_activity;
    struct dora_vector children[2];
    uint64_t count;
    int64_t effective_count;
};

static int copy_run_prepare(const struct copy_plan* plan,
                            const struct dora_vector* source,
                            const struct dora_vector_view* view,
                            struct dora_schema_field field,
                            struct copy_rows rows,
                            uint32_t position,
                            struct copy_run* work) {
    struct dora_vector run_source = {};
    struct vector_image* run_image = nullptr;
    struct dora_schema_field run_field = {};
    struct dora_schema run_schema = {};
    int status = DORA_OK;
    if (plan->foreign == nullptr) {
        status = dora_vector_get_physical_child(source, 0, &run_source);
        if (status == DORA_OK) {
            status = dora_vector_resolve(&run_source, &run_image);
        }
    }
    if (status == DORA_OK && plan->schema != nullptr) {
        status = dora_schema_get_physical_child(plan->schema, 0, &run_schema);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(plan->schema != nullptr ? &run_schema : &run_image->schema, &run_field);
    }
    if (status == DORA_OK && plan->layouts != nullptr && position != UINT32_MAX &&
        plan->layouts[position + 1] != DORA_LAYOUT_DEFAULT && plan->layouts[position + 1] != DORA_LAYOUT_FIXED) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = copy_allocate(plan->pool, rows.length, sizeof(uint64_t), false, &work->mapping);
    }
    if (status == DORA_OK) {
        status = copy_allocate(plan->pool, rows.length, copy_index_width(field.index_type), false, &work->buffers[1]);
    }
    struct dora_vector_view own = *view;
    own.validity = (struct dora_view_buffer){};
    if (status == DORA_OK) {
        status = copy_inherited_mask(plan->pool,
                                     view,
                                     &own,
                                     rows,
                                     &work->effective,
                                     &work->effective_count,
                                     &work->activity);
    }
    if (status == DORA_OK && work->activity.address != nullptr) {
        status = copy_allocate(plan->pool,
                               bitmap_words_for_bits(rows.length),
                               sizeof(uint64_t),
                               true,
                               &work->child_activity);
    }
    uint64_t previous = 0;
    for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
        const uint64_t row = copy_active_row(rows, index);
        const uint64_t run = row >= UINT64_MAX - 1 ? row : dora_vector_view_find_run(view, row);
        const bool inactive = row == UINT64_MAX - 1 || (row < UINT64_MAX - 1 && copy_is_null(view, rows, index));
        if (work->count == 0 || run != previous) {
            ((uint64_t*)work->mapping.address)[work->count++] = inactive ? UINT64_MAX - 1 : run;
            previous = run;
        } else if (!inactive) {
            ((uint64_t*)work->mapping.address)[work->count - 1] = run;
        }
        if (work->child_activity.address != nullptr && copy_row_is_admitted(rows, index) && !inactive) {
            bitmap_set(work->child_activity.address, work->count - 1);
        }
        status = copy_store_index(field.index_type, work->buffers[1].address, work->count - 1, index + 1);
    }
    if (status == DORA_OK) {
        const struct vector_input input = {
            .field = &run_field,
            .row_count = work->count,
            .allocations = work->buffers,
            .buffer_count = 2,
            .content_state = run_image != nullptr ? run_image->node.content_state : VECTOR_CONTENT_PROMISED
        };
        status = dora_vector_init_move_input(&work->children[0], plan->pool, &input, work->buffers, plan->error);
    }
    return status;
}

static void copy_run_release(struct copy_run* work) {
    dora_vector_release(&work->children[0]);
    dora_vector_release(&work->children[1]);
    dora_allocation_release(&work->buffers[1]);
    dora_allocation_release(&work->effective);
    dora_allocation_release(&work->activity);
    dora_allocation_release(&work->child_activity);
    dora_allocation_release(&work->mapping);
}

// Utf8, Binary; offset carriers: Int32, Int64
#define DEFINE_COPY_REBASE(S, D, T)                                                                                    \
    static void copy_rebase_##S##_##D(const struct dora_vector_view* view,                                             \
                                      uint64_t length,                                                                 \
                                      T* output,                                                                       \
                                      int64_t origin) {                                                                \
        for (uint64_t index = 0; index <= length; ++index) {                                                           \
            const int64_t offset = type_load_##S(view->offsets.data, view->offset + index);                            \
            output[index] = (T)(offset - origin);                                                                      \
        }                                                                                                              \
    }

DEFINE_COPY_REBASE(i32, i32, int32_t)
DEFINE_COPY_REBASE(i32, i64, int64_t)
DEFINE_COPY_REBASE(i64, i32, int32_t)
DEFINE_COPY_REBASE(i64, i64, int64_t)

#undef DEFINE_COPY_REBASE

static int copy_bytes_node(const struct copy_plan* plan,
                           const struct copy_facts* facts,
                           const struct copy_transport* transport,
                           struct copy_rows rows,
                           struct dora_schema_field field,
                           struct dora_vector* out) {
    const struct dora_vector_view* const view = &transport->view;
    const bool views = field.layout == DORA_LAYOUT_BINARY_VIEW;
    uint64_t total = 0;
    bool bulk = !views && copy_is_contiguous(rows, view, transport) &&
                (view->layout == DORA_LAYOUT_BINARY || view->layout == DORA_LAYOUT_LARGE_BINARY);
    const bool dense = bulk && view->validity.data == nullptr;
    const uint8_t* span = nullptr;
    int64_t origin = 0;
    if (dense && rows.length != 0) {
        int64_t end = 0;
        if (view->layout == DORA_LAYOUT_BINARY) {
            origin = type_load_i32(view->offsets.data, view->offset);
            end = type_load_i32(view->offsets.data, view->offset + rows.length);
        } else {
            origin = type_load_i64(view->offsets.data, view->offset);
            end = type_load_i64(view->offsets.data, view->offset + rows.length);
        }
        total = (uint64_t)(end - origin);
        if (total > SIZE_MAX || (field.layout == DORA_LAYOUT_BINARY && total > INT32_MAX)) {
            return EOVERFLOW;
        }
        if (total != 0) {
            span = (const uint8_t*)view->data_buffers[0].data + origin;
        }
    }
    uint64_t chunk = 0;
    uint32_t data_count = 0;
    for (uint64_t index = 0; !dense && index < rows.length; ++index) {
        const uint64_t row = copy_active_row(rows, index);
        if (row >= UINT64_MAX - 1 || copy_is_null(view, rows, index)) {
            continue;
        }
        const struct dora_bytes bytes = copy_bytes_value(transport, row);
        if (views && bytes.byte_count <= VIEW_INLINE_BYTE_COUNT) {
            continue;
        }
        if (bulk && bytes.byte_count != 0) {
            if (span == nullptr) {
                span = bytes.data;
            } else if ((uintptr_t)bytes.data < (uintptr_t)span || (uintptr_t)bytes.data - (uintptr_t)span != total) {
                bulk = false;
            }
        }
        if (ckd_add(&total, total, bytes.byte_count) || total > SIZE_MAX || (views && bytes.byte_count > INT32_MAX) ||
            (field.layout == DORA_LAYOUT_BINARY && total > INT32_MAX)) {
            return EOVERFLOW;
        }
        if (views) {
            if (data_count == 0 || bytes.byte_count > (uint64_t)INT32_MAX - chunk) {
                if (data_count == INT32_MAX) {
                    return EOVERFLOW;
                }
                ++data_count;
                chunk = 0;
            }
            chunk += bytes.byte_count;
        }
    }
    const uint32_t buffer_count = views ? data_count + 3 : 3;
    struct dora_allocation inline_buffers[8];
    struct dora_allocation buffer_array = {};
    int status = DORA_OK;
    if (buffer_count > 8) {
        status = copy_allocate(plan->pool, buffer_count, sizeof(struct dora_allocation), false, &buffer_array);
    }
    struct dora_allocation* const buffers = buffer_count > 8 ? buffer_array.address : inline_buffers;
    if (status == DORA_OK) {
        memset(buffers, 0, buffer_count * sizeof(*buffers));
    }
    struct dora_allocation effective = {};
    struct dora_allocation activity = {};
    int64_t own_count = 0;
    int64_t effective_count = 0;
    if (status == DORA_OK) {
        status = copy_mask(plan->pool, view, rows, transport, &buffers[0], &own_count, &activity);
    }
    const size_t width = views ? sizeof(struct dora_binary_view_entry) : field.layout == DORA_LAYOUT_BINARY ? 4 : 8;
    if (status == DORA_OK) {
        status = copy_allocate(plan->pool, rows.length + (views ? 0 : 1), width, false, &buffers[1]);
    }
    if (status == DORA_OK && !views) {
        status = copy_store_index(width == 4 ? DORA_TYPE_INT32 : DORA_TYPE_INT64, buffers[1].address, 0, 0);
    }
    if (status == DORA_OK && !views) {
        status = copy_allocate(plan->pool, total, 1, false, &buffers[2]);
    } else if (status == DORA_OK) {
        status = copy_allocate(plan->pool, data_count, sizeof(int64_t), false, &buffers[buffer_count - 1]);
        uint32_t data_index = 0;
        chunk = 0;
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            if (row >= UINT64_MAX - 1 || copy_is_null(view, rows, index)) {
                continue;
            }
            const struct dora_bytes bytes = copy_bytes_value(transport, row);
            if (bytes.byte_count <= VIEW_INLINE_BYTE_COUNT) {
                continue;
            }
            if (bytes.byte_count > (uint64_t)INT32_MAX - chunk) {
                status = copy_allocate(plan->pool, chunk, 1, false, &buffers[2 + data_index]);
                ((int64_t*)buffers[buffer_count - 1].address)[data_index++] = (int64_t)chunk;
                chunk = 0;
            }
            chunk += bytes.byte_count;
        }
        if (status == DORA_OK && data_count != 0) {
            status = copy_allocate(plan->pool, chunk, 1, false, &buffers[2 + data_index]);
            ((int64_t*)buffers[buffer_count - 1].address)[data_index] = (int64_t)chunk;
        }
    }
    if (status == DORA_OK && dense && rows.length != 0) {
        switch ((view->layout == DORA_LAYOUT_LARGE_BINARY ? 2 : 0) +
                (field.layout == DORA_LAYOUT_LARGE_BINARY ? 1 : 0)) {
        case 0:
            copy_rebase_i32_i32(view, rows.length, buffers[1].address, origin);
            break;
        case 1:
            copy_rebase_i32_i64(view, rows.length, buffers[1].address, origin);
            break;
        case 2:
            copy_rebase_i64_i32(view, rows.length, buffers[1].address, origin);
            break;
        default:
            copy_rebase_i64_i64(view, rows.length, buffers[1].address, origin);
            break;
        }
    }

    uint32_t data_index = 0;
    chunk = 0;
    for (uint64_t index = 0; status == DORA_OK && !dense && index < rows.length; ++index) {
        const uint64_t row = copy_active_row(rows, index);
        const struct dora_bytes bytes = row >= UINT64_MAX - 1 || copy_is_null(view, rows, index)
                                            ? (struct dora_bytes){}
                                            : copy_bytes_value(transport, row);
        uint8_t* destination = nullptr;
        if (views) {
            struct dora_binary_view_entry* const entry = &((struct dora_binary_view_entry*)buffers[1].address)[index];
            *entry = (struct dora_binary_view_entry){.byte_count = (int32_t)bytes.byte_count};
            if (bytes.byte_count <= VIEW_INLINE_BYTE_COUNT) {
                destination = entry->payload.inline_bytes;
            } else {
                if (bytes.byte_count > (uint64_t)INT32_MAX - chunk) {
                    ++data_index;
                    chunk = 0;
                }
                entry->payload.reference.buffer_index = (int32_t)data_index;
                entry->payload.reference.byte_offset = (int32_t)chunk;
                destination = (uint8_t*)buffers[2 + data_index].address + chunk;
                chunk += bytes.byte_count;
            }
        } else {
            if (bytes.byte_count != 0) {
                destination = (uint8_t*)buffers[2].address + chunk;
            }
            chunk += bytes.byte_count;
            status =
                copy_store_index(width == 4 ? DORA_TYPE_INT32 : DORA_TYPE_INT64, buffers[1].address, index + 1, chunk);
        }
        if (bytes.byte_count != 0 && status == DORA_OK && !bulk) {
            status = copy_bytes_read(transport, row, bytes, destination);
            if (status == DORA_OK && views && bytes.byte_count > VIEW_INLINE_BYTE_COUNT) {
                struct dora_binary_view_entry* const entry =
                    &((struct dora_binary_view_entry*)buffers[1].address)[index];
                memcpy(entry->payload.reference.prefix, destination, 4);
            }
        }
    }
    if (status == DORA_OK && bulk && total != 0) {
        memcpy(buffers[2].address, span, total);
    }
    struct copy_facts produced = *facts;
    if (views && field.type.id == DORA_TYPE_BINARY && field.type.extension_name.byte_count == 0) {
        produced.content_state = VECTOR_CONTENT_VALIDATED;
    }
    if (status == DORA_OK) {
        status = copy_publish(plan,
                              &produced,
                              field,
                              rows,
                              buffers,
                              buffer_count,
                              nullptr,
                              own_count,
                              &effective,
                              effective_count,
                              &activity,
                              out);
    }
    if (buffers != nullptr) {
        for (uint32_t index = 0; index < buffer_count; ++index) {
            dora_allocation_release(&buffers[index]);
        }
    }
    dora_allocation_release(&effective);
    dora_allocation_release(&activity);
    dora_allocation_release(&buffer_array);
    return status;
}

static int copy_needs_transport(const struct dora_vector* source, bool* out_needed) {
    struct vector_image* image = nullptr;
    int status = dora_vector_resolve(source, &image);
    if (status != DORA_OK) {
        return status;
    }
    if (image->ownership.view.placement.access == DORA_ACCESS_DEVICE || image->ownership.requires_effective_validity) {
        *out_needed = true;
    }
    struct dora_vector* const children = dora_vector_image_children(image);
    for (uint32_t index = 0; status == DORA_OK && !*out_needed && index < image->child_vectors.entry_count; ++index) {
        status = copy_needs_transport(&children[index], out_needed);
    }
    return status;
}

static struct copy_plan copy_child_plan(const struct copy_plan* plan, const struct dora_schema* schema) {
    struct copy_plan child = *plan;
    if (plan->schema != nullptr) {
        child.schema = schema;
    }
    return child;
}

static int copy_node(const struct copy_plan* plan,
                     const struct dora_vector* source,
                     struct copy_rows rows,
                     uint32_t position,
                     const struct dora_schema_field* declaration,
                     const struct copy_facts* inherited,
                     enum dora_layout forced_layout,
                     struct dora_vector* out) {
    struct copy_transport staged = {};
    int status = DORA_OK;
    if (plan->foreign == nullptr) {
        status = copy_transport_bind(plan, source, rows, &staged);
        if (status == DORA_OK) {
            status = copy_stage_runs(plan, source, &staged);
        }
    } else {
        const struct vector_bound_node* const node = &plan->foreign->nodes[position];
        staged.schema = plan->schema;
        staged.source_pool = plan->pool;
        staged.content_state = VECTOR_CONTENT_PROMISED;
        staged.view = node->view;
        staged.own = staged.view;
        if (staged.view.placement.access == DORA_ACCESS_DEVICE) {
            staged.owner = plan->foreign_owner;
            status = staged.owner == nullptr || staged.owner->read_host == nullptr
                         ? ENOTSUP
                         : copy_transport_stage(plan, rows, &staged, false);
            if (status == DORA_OK && staged.view.layout == DORA_LAYOUT_RUN_END) {
                const uint32_t ends = dora_vector_input_child(position, plan->foreign, 0);
                const struct dora_vector_view runs = plan->foreign->nodes[ends].view;
                status = copy_stage_run_ends(plan, staged.owner, &runs, &staged);
            }
        }
    }
    if (status != DORA_OK) {
        goto release_transport;
    }
    rows.effective_validity = staged.effective_validity;
    const struct copy_transport* transport = &staged;

    struct dora_schema_field field = {};
    status = dora_schema_get_field(transport->schema, &field);
    if (status != DORA_OK) {
        goto release_transport;
    }
    const struct dora_vector_view view = transport->view;
    struct copy_plan publication = *plan;
    publication.admitted_schema = plan->schema != nullptr                            ? plan->schema
                                  : pool_is_same(plan->pool, transport->source_pool) ? transport->schema
                                                                                     : nullptr;
    if (publication.admitted_schema != nullptr && publication.admitted_schema->handle.pool_index != plan->pool.index) {
        publication.admitted_schema = nullptr;
    }
    const struct copy_facts facts =
        inherited != nullptr
            ? *inherited
            : (struct copy_facts){
                  .content_state = transport->content_state,
                  .are_required_rows_admitted = (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0,
                  .are_map_rows_admitted = field.type.id == DORA_TYPE_MAP
              };
    if (plan->schema != nullptr) {
        if (declaration != nullptr) {
            field = *declaration;
        } else {
            status = dora_schema_get_field(plan->schema, &field);
        }
    } else if (declaration != nullptr) {
        const uint32_t children = field.child_count;
        field = *declaration;
        field.child_count = children;
    }
    if (status != DORA_OK) {
        goto release_transport;
    }
    enum dora_layout layout = forced_layout;
    if (layout == DORA_LAYOUT_DEFAULT && plan->schema != nullptr) {
        layout = field.layout;
    }
    if (layout == DORA_LAYOUT_DEFAULT && plan->layouts != nullptr && position != UINT32_MAX) {
        layout = plan->layouts[position];
    }
    if (layout == DORA_LAYOUT_DEFAULT) {
        if (plan->should_decode && (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END)) {
            status = dora_type_resolve_layout(&field.type, DORA_LAYOUT_DEFAULT, DORA_TYPE_NULL, &layout);
        } else {
            layout = view.layout;
        }
    }
    const bool encoded = layout == DORA_LAYOUT_DICTIONARY || layout == DORA_LAYOUT_RUN_END;
    if (status == DORA_OK && !encoded) {
        status = dora_type_resolve_layout(&field.type, layout, DORA_TYPE_NULL, &layout);
    }
    if (status != DORA_OK) {
        goto release_transport;
    }
    const struct dora_vector_view own = transport->own;
    field.layout = layout;
    if (layout != DORA_LAYOUT_DICTIONARY) {
        field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED;
    }
    if (encoded && layout != view.layout) {
        struct copy_encoding work = {};
        status = copy_encoding_prepare(plan, transport, field, rows, &work);
        const uint64_t* hashes = nullptr;
        if (status == DORA_OK && work.field.layout == DORA_LAYOUT_DICTIONARY && work.should_deduplicate &&
            rows.length > KEYSET_PAIRWISE_LIMIT) {
            struct dora_vector_view hash_view = {};
            status = copy_hash_prepare(plan->pool, source, &work.hashes, &hash_view);
            if (status == DORA_OK &&
                (hash_view.layout == DORA_LAYOUT_DICTIONARY || hash_view.layout == DORA_LAYOUT_RUN_END)) {
                const struct copy_plan decode = {.pool = plan->pool, .should_decode = true};
                struct dora_vector host = {};
                struct dora_vector decoded = {};
                bool needs_transport = false;
                status = copy_needs_transport(&work.hashes, &needs_transport);
                if (status == DORA_OK && needs_transport) {
                    const struct copy_plan host_plan = {.pool = plan->pool};
                    status = copy_node(&host_plan,
                                       &work.hashes,
                                       (struct copy_rows){.length = work.hashes.rows.length},
                                       UINT32_MAX,
                                       nullptr,
                                       nullptr,
                                       DORA_LAYOUT_DEFAULT,
                                       &host);
                }
                if (status == DORA_OK) {
                    const struct dora_vector* input = needs_transport ? &host : &work.hashes;
                    status = copy_node(&decode,
                                       input,
                                       (struct copy_rows){.length = input->rows.length},
                                       UINT32_MAX,
                                       nullptr,
                                       nullptr,
                                       DORA_LAYOUT_DEFAULT,
                                       &decoded);
                }
                dora_vector_release(&host);
                dora_vector_release(&work.hashes);
                work.hashes = decoded;
                if (status == DORA_OK) {
                    status = dora_vector_get_view(&work.hashes, &hash_view);
                }
            }
            if (status == DORA_OK) {
                hashes = dora_vector_view_get_values_u64(&hash_view);
                if (hashes == nullptr) {
                    status = EINVAL;
                }
            }
        }
        if (status == DORA_OK) {
            status = copy_encoding_map(plan, source, transport, rows, hashes, &work);
        }
        if (status == DORA_OK) {
            const struct copy_plan value_plan = copy_child_plan(plan, &work.value_schema);
            status = copy_node(&value_plan,
                               source,
                               (struct copy_rows){
                                   .indices = work.mapping.address,
                                   .admission_activity = work.child_activity.address,
                                   .length = work.value_count
                               },
                               position,
                               &work.value_field,
                               &facts,
                               work.value_field.layout,
                               &work.children[work.field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1]);
        }
        if (status == DORA_OK) {
            status = copy_encoding_publish(&publication, &facts, rows, &work, out);
        }
        copy_encoding_release(&work);
        goto release_transport;
    }
    if (layout == DORA_LAYOUT_RUN_END) {
        struct copy_run work = {};
        status = copy_run_prepare(plan, source, &view, field, rows, position, &work);
        if (status == DORA_OK) {
            struct copy_child_input child_input = {};
            status = copy_child_prepare(plan, source, 1, position, &child_input);
            if (status == DORA_OK) {
                const struct copy_plan child_plan = copy_child_plan(plan, &child_input.schema);
                status = copy_node(&child_plan,
                                   &child_input.source,
                                   (struct copy_rows){
                                       .indices = work.mapping.address,
                                       .admission_activity = work.child_activity.address,
                                       .length = work.count
                                   },
                                   child_input.position,
                                   &child_input.field,
                                   nullptr,
                                   DORA_LAYOUT_DEFAULT,
                                   &work.children[1]);
            }
        }
        if (status == DORA_OK) {
            status = copy_publish(&publication,
                                  &facts,
                                  field,
                                  rows,
                                  nullptr,
                                  0,
                                  work.children,
                                  0,
                                  &work.effective,
                                  work.effective_count,
                                  &work.activity,
                                  out);
        }
        copy_run_release(&work);
        goto release_transport;
    }
    if (!encoded && (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END)) {
        field.index_type = DORA_TYPE_NULL;
        struct dora_allocation mapping = {};
        struct copy_child_input child = {};
        status = copy_peel_prepare(plan, source, &view, rows, position, &mapping, &child);
        if (status == DORA_OK) {
            status = copy_node(plan,
                               &child.source,
                               (struct copy_rows){
                                   .indices = mapping.address,
                                   .admission_activity = rows.admission_activity,
                                   .length = rows.length
                               },
                               child.position,
                               &field,
                               &facts,
                               layout,
                               out);
        }
        dora_allocation_release(&mapping);
        goto release_transport;
    }
    if (!encoded) {
        field.index_type = DORA_TYPE_NULL;
    }
    if (layout == DORA_LAYOUT_BINARY || layout == DORA_LAYOUT_LARGE_BINARY || layout == DORA_LAYOUT_BINARY_VIEW) {
        status = copy_bytes_node(&publication, &facts, transport, rows, field, out);
        goto release_transport;
    }
    struct dora_allocation buffers[3] = {};
    struct dora_allocation effective = {};
    struct dora_allocation activity = {};
    struct dora_allocation child_activity = {};
    struct dora_allocation children_storage = {};
    struct dora_allocation mapping = {};
    struct dora_vector inline_children[8];
    struct dora_vector* children = nullptr;
    int64_t own_count = 0;
    int64_t effective_count = 0;
    uint32_t buffer_count = 1;
    const bool union_layout = layout == DORA_LAYOUT_DENSE_UNION || layout == DORA_LAYOUT_SPARSE_UNION;
    if (layout != DORA_LAYOUT_NULL && !union_layout) {
        status = copy_mask(plan->pool, &view, rows, transport, &buffers[0], &own_count, &activity);
    } else if (union_layout) {
        status = copy_inherited_mask(plan->pool, &view, &own, rows, &effective, &effective_count, &activity);
    }
    if (status == DORA_OK && field.child_count != 0) {
        if (field.child_count > 8) {
            status = copy_allocate(plan->pool, field.child_count, sizeof(struct dora_vector), false, &children_storage);
        }
        children = field.child_count > 8 ? children_storage.address : inline_children;
        if (status == DORA_OK) {
            memset(children, 0, field.child_count * sizeof(*children));
        }
    }
    if (status == DORA_OK &&
        (layout == DORA_LAYOUT_FIXED || layout == DORA_LAYOUT_DICTIONARY || layout == DORA_LAYOUT_BITS)) {
        struct dora_vector dictionary_child = {};
        if (layout == DORA_LAYOUT_DICTIONARY) {
            if (plan->foreign != nullptr) {
                dictionary_child.rows.length =
                    plan->foreign->nodes[dora_vector_input_child(position, plan->foreign, 0)].view.length;
            } else {
                status = dora_vector_get_physical_child(source, 0, &dictionary_child);
            }
            if (status == DORA_OK) {
                status = copy_allocate(plan->pool,
                                       bitmap_words_for_bits(dictionary_child.rows.length),
                                       sizeof(uint64_t),
                                       true,
                                       &child_activity);
            }
        }
        buffer_count = 2;
        const size_t width = layout == DORA_LAYOUT_DICTIONARY ? copy_index_width(field.index_type) : view.byte_width;
        enum dora_physical_type physical_type = view.physical_type;
        if (status == DORA_OK && layout == DORA_LAYOUT_DICTIONARY) {
            const struct dora_logical_type index_type = {.id = field.index_type};
            status = type_physical(&index_type, &physical_type);
        }
        if (status == DORA_OK) {
            status = copy_allocate(plan->pool,
                                   layout == DORA_LAYOUT_BITS ? bitmap_words_for_bits(rows.length) : rows.length,
                                   layout == DORA_LAYOUT_BITS ? sizeof(uint64_t) : width,
                                   layout == DORA_LAYOUT_BITS,
                                   &buffers[1]);
        }
        const bool contiguous = status == DORA_OK && copy_is_contiguous(rows, &view, transport) &&
                                (layout != DORA_LAYOUT_DICTIONARY || physical_type == view.physical_type);
        if (contiguous && rows.length != 0) {
            if (layout == DORA_LAYOUT_BITS) {
                dora_bitmap_load_words(view.values.data, view.offset, rows.length, buffers[1].address);
            } else if (view.byte_width != 0) {
                memcpy(buffers[1].address,
                       (const uint8_t*)view.values.data + view.offset * view.byte_width,
                       (size_t)(rows.length * view.byte_width));
            }
        }
        bool nulls_initialized = false;
        for (uint64_t index = 0; status == DORA_OK && !contiguous && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            if (row >= UINT64_MAX - 1 || copy_is_null(&view, rows, index)) {
                if (layout != DORA_LAYOUT_BITS && width != 0 && !nulls_initialized) {
                    memset((uint8_t*)buffers[1].address + index * width, 0, (rows.length - index) * width);
                    nulls_initialized = true;
                }
                continue;
            }
            const uint64_t physical = dora_vector_view_get_physical_row(&view, row);
            if (layout == DORA_LAYOUT_BITS) {
                uint8_t byte = 0;
                status = copy_read(transport->owner,
                                   view.values,
                                   (struct dora_range){.offset = bitmap_byte_of(physical), .length = 1},
                                   &byte);
                if (status == DORA_OK) {
                    bitmap_assign(buffers[1].address, index, dora_bitmap_test(&byte, bitmap_bit_in_byte(physical)));
                }
            } else if (layout == DORA_LAYOUT_DICTIONARY && physical_type != view.physical_type) {
                uint64_t code = 0;
                status = dora_vector_view_read_index(&view, row, &code);
                if (status == DORA_OK) {
                    status = copy_store_index(field.index_type, buffers[1].address, index, code);
                }
            } else if (view.byte_width != 0) {
                status = copy_read(transport->owner,
                                   view.values,
                                   (struct dora_range){.offset = physical * view.byte_width, .length = view.byte_width},
                                   (uint8_t*)buffers[1].address + index * width);
            }
            if (status == DORA_OK && layout == DORA_LAYOUT_DICTIONARY && copy_row_is_admitted(rows, index)) {
                const struct dora_vector_view copied = {
                    .physical_type = physical_type,
                    .values.data = buffers[1].address
                };
                uint64_t code = 0;
                status = dora_vector_view_read_index(&copied, index, &code);
                if (status == DORA_OK && code >= dictionary_child.rows.length) {
                    status = EINVAL;
                }
                if (status == DORA_OK) {
                    bitmap_set(child_activity.address, code);
                }
            }
        }
        const bool mark_contiguous = contiguous && layout == DORA_LAYOUT_DICTIONARY;
        for (uint64_t index = 0; status == DORA_OK && mark_contiguous && index < rows.length; ++index) {
            if (copy_is_null(&view, rows, index)) {
                continue;
            }
            const struct dora_vector_view copied = {.physical_type = physical_type, .values.data = buffers[1].address};
            uint64_t code = 0;
            status = dora_vector_view_read_index(&copied, index, &code);
            if (status == DORA_OK && code >= dictionary_child.rows.length) {
                status = EINVAL;
            }
            if (status == DORA_OK) {
                bitmap_set(child_activity.address, code);
            }
        }
        if (status == DORA_OK && layout == DORA_LAYOUT_DICTIONARY) {
            struct copy_child_input child_input = {};
            status = copy_child_prepare(plan, source, 0, position, &child_input);
            if (status == DORA_OK) {
                const struct copy_plan child_plan = copy_child_plan(plan, &child_input.schema);
                status = copy_node(&child_plan,
                                   &child_input.source,
                                   (struct copy_rows){
                                       .admission_activity = child_activity.address,
                                       .length = dictionary_child.rows.length
                                   },
                                   child_input.position,
                                   &child_input.field,
                                   nullptr,
                                   DORA_LAYOUT_DEFAULT,
                                   &children[0]);
            }
        }
    } else if (status == DORA_OK && layout == DORA_LAYOUT_STRUCT) {
        const struct copy_rows child_rows = {
            .indices = rows.indices,
            .length = rows.length,
            .admission_activity = activity.address,
            .activity = effective.address != nullptr ? effective.address : buffers[0].address
        };
        for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
            struct copy_child_input child_input = {};
            status = copy_child_prepare(plan, source, child, position, &child_input);
            if (status == DORA_OK) {
                const struct copy_plan child_plan = copy_child_plan(plan, &child_input.schema);
                status = copy_node(&child_plan,
                                   &child_input.source,
                                   child_rows,
                                   child_input.position,
                                   &child_input.field,
                                   nullptr,
                                   DORA_LAYOUT_DEFAULT,
                                   &children[child]);
            }
        }
    } else if (status == DORA_OK &&
               (layout == DORA_LAYOUT_LIST || layout == DORA_LAYOUT_LARGE_LIST || layout == DORA_LAYOUT_LIST_VIEW ||
                layout == DORA_LAYOUT_LARGE_LIST_VIEW || layout == DORA_LAYOUT_FIXED_LIST)) {
        const bool fixed = layout == DORA_LAYOUT_FIXED_LIST;
        const bool list_view = layout == DORA_LAYOUT_LIST_VIEW || layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        const size_t width = layout == DORA_LAYOUT_LIST || layout == DORA_LAYOUT_LIST_VIEW ? 4 : 8;
        buffer_count = fixed ? 1 : list_view ? 3 : 2;
        uint64_t element_count = 0;
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            const struct dora_range range = row >= UINT64_MAX - 1 || copy_is_null(&view, rows, index)
                                                ? (struct dora_range){}
                                                : dora_vector_view_get_elements(&view, row);
            const uint64_t count = fixed ? field.type.parameters.list_element_count : range.length;
            if (ckd_add(&element_count, element_count, count) || element_count > INT64_MAX ||
                (!fixed && width == 4 && element_count > INT32_MAX)) {
                status = EOVERFLOW;
            }
        }
        if (status == DORA_OK) {
            status = copy_allocate(plan->pool, element_count, sizeof(uint64_t), false, &mapping);
        }
        if (status == DORA_OK && activity.address != nullptr) {
            status = copy_allocate(plan->pool,
                                   bitmap_words_for_bits(element_count),
                                   sizeof(uint64_t),
                                   true,
                                   &child_activity);
        }
        if (status == DORA_OK && !fixed) {
            status = copy_allocate(plan->pool, rows.length + (list_view ? 0 : 1), width, false, &buffers[1]);
        }
        if (status == DORA_OK && list_view) {
            status = copy_allocate(plan->pool, rows.length, width, false, &buffers[2]);
        }
        uint64_t next = 0;
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            const uint64_t row = copy_active_row(rows, index);
            const bool active = row < UINT64_MAX - 1 && !copy_is_null(&view, rows, index);
            const struct dora_range range = active ? dora_vector_view_get_elements(&view, row) : (struct dora_range){};
            const uint64_t count = fixed ? field.type.parameters.list_element_count : range.length;
            if (!fixed) {
                status =
                    copy_store_index(width == 4 ? DORA_TYPE_INT32 : DORA_TYPE_INT64, buffers[1].address, index, next);
                if (status == DORA_OK && list_view) {
                    status = copy_store_index(width == 4 ? DORA_TYPE_INT32 : DORA_TYPE_INT64,
                                              buffers[2].address,
                                              index,
                                              count);
                }
            }
            for (uint64_t item = 0; item < count; ++item) {
                if (child_activity.address != nullptr && active && copy_row_is_admitted(rows, index)) {
                    bitmap_set(child_activity.address, next);
                }
                ((uint64_t*)mapping.address)[next++] = active ? range.offset + item : UINT64_MAX - 1;
            }
        }
        if (status == DORA_OK && !fixed && !list_view) {
            status =
                copy_store_index(width == 4 ? DORA_TYPE_INT32 : DORA_TYPE_INT64, buffers[1].address, rows.length, next);
        }
        for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
            struct copy_child_input child_input = {};
            status = copy_child_prepare(plan, source, child, position, &child_input);
            if (status == DORA_OK) {
                const struct copy_plan child_plan = copy_child_plan(plan, &child_input.schema);
                status = copy_node(&child_plan,
                                   &child_input.source,
                                   (struct copy_rows){
                                       .indices = mapping.address,
                                       .admission_activity = child_activity.address,
                                       .length = element_count
                                   },
                                   child_input.position,
                                   &child_input.field,
                                   nullptr,
                                   DORA_LAYOUT_DEFAULT,
                                   &children[child]);
            }
        }
    } else if (status == DORA_OK && union_layout) {
        const bool dense = layout == DORA_LAYOUT_DENSE_UNION;
        buffer_count = dense ? 2 : 1;
        status = copy_allocate(plan->pool, rows.length, sizeof(int8_t), false, &buffers[0]);
        if (status == DORA_OK && dense) {
            status = copy_allocate(plan->pool, rows.length, sizeof(int32_t), false, &buffers[1]);
        }
        if (status == DORA_OK) {
            status = copy_allocate(plan->pool, rows.length, sizeof(uint64_t), false, &mapping);
        }
        if (status == DORA_OK && activity.address != nullptr) {
            status =
                copy_allocate(plan->pool, bitmap_words_for_bits(rows.length), sizeof(uint64_t), true, &child_activity);
        }
        for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
            if (child_activity.address != nullptr) {
                dora_bitmap_set_range(child_activity.address, 0, rows.length, false);
            }
            uint64_t count = 0;
            for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
                const uint64_t row = copy_active_row(rows, index);
                uint64_t member_row = 0;
                const bool inactive =
                    row == UINT64_MAX - 1 || (row < UINT64_MAX - 1 && copy_is_null(&view, rows, index));
                const uint32_t member =
                    row >= UINT64_MAX - 1 || inactive ? 0 : dora_vector_view_get_member(&view, row, &member_row);
                const bool selected = member == child;
                if (selected) {
                    ((int8_t*)buffers[0].address)[index] = field.type.parameters.type_codes[child];
                    if (dense) {
                        status = copy_store_index(DORA_TYPE_INT32, buffers[1].address, index, count);
                    }
                }
                if (selected || !dense) {
                    if (child_activity.address != nullptr && selected && !inactive &&
                        copy_row_is_admitted(rows, index)) {
                        bitmap_set(child_activity.address, count);
                    }
                    ((uint64_t*)mapping.address)[count++] = !selected               ? UINT64_MAX - 1
                                                            : row >= UINT64_MAX - 1 ? row
                                                            : inactive              ? UINT64_MAX - 1
                                                                                    : member_row;
                }
            }
            if (status == DORA_OK) {
                struct copy_child_input child_input = {};
                status = copy_child_prepare(plan, source, child, position, &child_input);
                if (status == DORA_OK) {
                    const struct copy_plan child_plan = copy_child_plan(plan, &child_input.schema);
                    status = copy_node(&child_plan,
                                       &child_input.source,
                                       (struct copy_rows){
                                           .indices = mapping.address,
                                           .admission_activity = child_activity.address,
                                           .length = count
                                       },
                                       child_input.position,
                                       &child_input.field,
                                       nullptr,
                                       DORA_LAYOUT_DEFAULT,
                                       &children[child]);
                }
            }
        }
    } else if (status == DORA_OK && layout == DORA_LAYOUT_NULL) {
        buffer_count = 0;
        own_count = (int64_t)rows.length;
        for (uint64_t index = 0; status == DORA_OK && index < rows.length; ++index) {
            status = copy_activity_row(plan->pool, rows, index, &activity);
        }
    } else if (status == DORA_OK) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = copy_publish(&publication,
                              &facts,
                              field,
                              rows,
                              buffers,
                              buffer_count,
                              children,
                              own_count,
                              &effective,
                              effective_count,
                              &activity,
                              out);
    }
    if (children != nullptr) {
        for (uint32_t child = 0; child < field.child_count; ++child) {
            dora_vector_release(&children[child]);
        }
    }
    for (uint32_t index = 0; index < 3; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    dora_allocation_release(&effective);
    dora_allocation_release(&activity);
    dora_allocation_release(&child_activity);
    dora_allocation_release(&mapping);
    dora_allocation_release(&children_storage);
release_transport:
    for (uint32_t index = 0; index < ARRAY_COUNT(staged.metadata); ++index) {
        dora_allocation_release(&staged.metadata[index]);
    }
    return status;
}

int dora_vector_copy_foreign(struct dora_pool pool,
                             const struct dora_schema* schema,
                             const struct vector_bound_input* input,
                             uint32_t root_node_index,
                             struct dora_buffer_owner* owner,
                             struct dora_vector* out_vector,
                             struct dora_error* out_error) {
    const struct copy_plan plan =
        {.pool = pool, .error = out_error, .schema = schema, .foreign = input, .foreign_owner = owner};
    return copy_node(&plan,
                     nullptr,
                     (struct copy_rows){.length = input->nodes[root_node_index].view.length},
                     root_node_index,
                     nullptr,
                     nullptr,
                     DORA_LAYOUT_DEFAULT,
                     out_vector);
}

static int copy_execute(const struct copy_plan* plan, const struct dora_vector* source, struct dora_vector* out) {
    // Source-pool work is needed only for device callbacks; host readers leave it untouched.
    bool is_transported = false;
    int status = copy_needs_transport(source, &is_transported);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_pool source_pool =
        is_transported
            ? (struct
               dora_pool){.index = source->handle.pool_index, .generation = dora_pool_rows[source->handle.pool_index].generation}
            : (struct dora_pool){};
    dora_pool_callback_enter(plan->pool);
    dora_pool_callback_enter(source_pool);
    struct dora_vector host = {};
    struct dora_vector published = {};
    const bool needs_transport =
        is_transported && (plan->should_decode || plan->layouts != nullptr || plan->schema != nullptr);
    if (needs_transport) {
        const struct copy_plan transport = {.pool = plan->pool, .error = plan->error};
        status = copy_node(&transport,
                           source,
                           (struct copy_rows){.length = source->rows.length},
                           UINT32_MAX,
                           nullptr,
                           nullptr,
                           DORA_LAYOUT_DEFAULT,
                           &host);
    }
    if (status == DORA_OK) {
        const struct dora_vector* const input = needs_transport ? &host : source;
        status = copy_node(plan,
                           input,
                           (struct copy_rows){.length = input->rows.length},
                           plan->layouts != nullptr ? 0 : UINT32_MAX,
                           nullptr,
                           nullptr,
                           DORA_LAYOUT_DEFAULT,
                           &published);
    }
    dora_vector_release(&host);
    dora_pool_callback_leave(source_pool);
    dora_pool_callback_leave(plan->pool);
    status = vector_finish_publication(&published, out, status);
    return status;
}

int dora_vector_copy_requires_transport(const struct dora_vector* vector, bool* out_needed) {
    if (!dora_vector_is_valid(vector) || out_needed == nullptr) {
        return EINVAL;
    }
    bool needed = false;
    const int status = copy_needs_transport(vector, &needed);
    if (status == DORA_OK) {
        *out_needed = needed;
    }
    return status;
}

int dora_vector_copy_schema_error(const struct dora_vector* vector,
                                  struct dora_pool pool,
                                  const struct dora_schema* schema,
                                  struct dora_vector* out_copy,
                                  struct dora_error* out_error) {
    if (!dora_vector_is_valid(vector) || !dora_vector_is_empty(out_copy) || !dora_schema_is_valid(schema)) {
        return EINVAL;
    }
    const struct copy_plan plan = {.pool = pool, .error = out_error, .schema = schema};
    return copy_execute(&plan, vector, out_copy);
}

int dora_vector_decode_error(const struct dora_vector* vector,
                             struct dora_pool pool,
                             struct dora_vector* out_decoded,
                             struct dora_error* out_error) {
    if (!dora_vector_is_valid(vector) || !dora_vector_is_empty(out_decoded)) {
        return EINVAL;
    }
    const struct copy_plan plan = {.pool = pool, .error = out_error, .should_decode = true};
    return copy_execute(&plan, vector, out_decoded);
}

int dora_vector_copy(const struct dora_vector* vector, struct dora_pool pool, struct dora_vector* out_copy) {
    if (!dora_vector_is_valid(vector) || !dora_vector_is_empty(out_copy)) {
        return EINVAL;
    }
    const struct copy_plan plan = {.pool = pool};
    return copy_execute(&plan, vector, out_copy);
}

int dora_vector_decode(const struct dora_vector* vector, struct dora_vector* out_decoded) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(vector, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_decode_error(vector, pool, out_decoded, nullptr);
}

int dora_vector_recode(const struct dora_vector* vector,
                       const enum dora_layout* node_layouts,
                       uint32_t node_layout_count,
                       struct dora_vector* out_recoded) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(vector, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_valid(vector) || !dora_vector_is_empty(out_recoded) ||
        (node_layout_count != 0 && node_layouts == nullptr)) {
        return EINVAL;
    }
    uint32_t count = 0;
    int status = copy_tree_size(vector, &count);
    if (status != DORA_OK || count != node_layout_count) {
        return status != DORA_OK ? status : EINVAL;
    }
    const struct copy_plan plan = {.pool = pool, .layouts = node_layouts};
    return copy_execute(&plan, vector, out_recoded);
}
