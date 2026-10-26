// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/peel.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/decode.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Binding and Publication
//
//------------------------------------------------------------------------------

int dora_compute_peel_values(const struct dora_vector* source,
                             const struct vector_bound_input* input,
                             bool error_can_fail,
                             struct dora_vector* out_values,
                             bool* out_applied) {
    const struct vector_bound_node* const root = &input->nodes[0];
    const struct dora_vector_view* const view = &root->view;
    const bool dictionary = view->layout == DORA_LAYOUT_DICTIONARY;
    *out_applied = false;
    if ((!dictionary && view->layout != DORA_LAYOUT_RUN_END) || view->selection.indices != nullptr ||
        input->length == 0 || (view->layout == DORA_LAYOUT_RUN_END && view->validity.data != nullptr)) {
        return DORA_OK;
    }
    if (error_can_fail) {
        return DORA_OK;
    }
    struct dora_vector values = {};
    int status = dora_vector_get_child(source, dictionary ? 0 : 1, &values);
    if (status != DORA_OK || (dictionary && values.rows.length > input->length / 2)) {
        return status;
    }
    if (!dictionary) {
        const uint32_t end_node = dora_vector_input_child(0, input, 0);
        const struct dora_vector_view* const ends = &input->nodes[end_node].view;
        int64_t end = 0;
        if (ends->length == 0) {
            return DORA_OK;
        }
        status = dora_kernel_run_end_last(ends, &end);
        if (status != DORA_OK) {
            return status;
        }
        if (view->offset != 0 || end < 0 || (uint64_t)end != input->length) {
            return DORA_OK;
        }
    }
    *out_values = values;
    *out_applied = true;
    return DORA_OK;
}

int dora_compute_peel_publish(struct dora_pool pool,
                              const struct dora_vector* source,
                              const struct vector_bound_input* input,
                              const struct dora_vector* transformed,
                              struct dora_vector* out_result) {
    const struct vector_bound_node* const root = &input->nodes[0];
    const struct dora_vector_view* const view = &root->view;
    const bool dictionary = view->layout == DORA_LAYOUT_DICTIONARY;
    int status = DORA_OK;
    struct dora_vector children[2] = {};
    struct dora_allocation buffers[2] = {};
    int64_t nulls = 0;
    const bool shared = dictionary && source->handle.pool_index == pool.index;
    if (status == DORA_OK && dictionary && !shared) {
        size_t bytes = 0;
        if (ckd_mul(&bytes, input->length, view->byte_width)) {
            status = EOVERFLOW;
        } else {
            status = dora_pool_allocate(pool, bytes, &buffers[1]);
        }
        if (status == DORA_OK && view->validity.data != nullptr) {
            status =
                dora_pool_allocate(pool, (size_t)bitmap_words_for_bits(input->length) * sizeof(uint64_t), &buffers[0]);
            if (status == DORA_OK) {
                dora_bitmap_load_words(view->validity.data,
                                       view->validity_bit_offset,
                                       input->length,
                                       buffers[0].address);
                nulls = (int64_t)(input->length - dora_bitmap_count_set_words(buffers[0].address, input->length));
            }
        }
        if (status == DORA_OK && bytes != 0) {
            memcpy(buffers[1].address, (const uint8_t*)view->values.data + view->offset * view->byte_width, bytes);
        }
    }
    if (status == DORA_OK && dictionary) {
        children[0] = dora_vector_borrow(transformed);
    } else if (status == DORA_OK) {
        status = dora_vector_get_child(source, 0, &children[0]);
        children[1] = dora_vector_borrow(transformed);
    }
    if (status == DORA_OK) {
        struct dora_schema_field field = root->field;
        field.flags &=
            (uint8_t)~(uint8_t)(DORA_SCHEMA_FIELD_FLAG_NONNULLABLE | DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED);
        status = dora_vector_get_type(transformed, &field.type);
        if (status == DORA_OK && shared) {
            status = dora_vector_init_share_indices(out_result, &field, source, &children[0]);
        } else if (status == DORA_OK) {
            const struct vector_input composition = {
                .field = &field,
                .row_count = input->length,
                .null_count = nulls,
                .allocations = dictionary ? buffers : nullptr,
                .buffer_count = dictionary ? 2 : 0,
                .children = children,
                .child_count = dictionary ? 1 : 2
            };
            status =
                dora_vector_init_move_input(out_result, pool, &composition, dictionary ? buffers : nullptr, nullptr);
        }
    }
    dora_allocation_release(&buffers[0]);
    dora_allocation_release(&buffers[1]);
    return status;
}
