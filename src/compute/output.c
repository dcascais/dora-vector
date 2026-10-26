// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/output.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/utf8.h"
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
// Results
//
//------------------------------------------------------------------------------

int dora_compute_bytes_offsets(struct dora_pool pool, struct compute_bytes_result* in_out_result) {
    size_t offsets_bytes = 0;
    if (in_out_result->stored == UINT64_MAX || ckd_mul(&offsets_bytes, in_out_result->stored + 1, sizeof(int64_t))) {
        return EOVERFLOW;
    }
    const int status = dora_pool_allocate(pool, offsets_bytes, &in_out_result->buffers[1]);
    if (status == DORA_OK) {
        ((int64_t*)in_out_result->buffers[1].address)[0] = 0;
    }
    return status;
}

int dora_compute_bytes_result_init(struct dora_pool pool,
                                   bool should_allocate_validity,
                                   struct compute_bytes_result* result) {
    int status = dora_compute_bytes_offsets(pool, result);
    if (status == DORA_OK && should_allocate_validity && result->validity_source == nullptr) {
        size_t bytes = 0;
        status = ckd_mul(&bytes, bitmap_words_for_bits(result->stored), sizeof(uint64_t))
                     ? EOVERFLOW
                     : dora_pool_allocate(pool, bytes, &result->buffers[0]);
        if (status == DORA_OK && result->buffers[0].address != nullptr) {
            bitmap_fill_bytes((uint8_t*)result->buffers[0].address, result->buffers[0].capacity_bytes, false);
        }
    }
    if (status == DORA_OK && result->validity_source == nullptr) {
        result->validity = (struct dora_utf8_mask){{result->buffers[0].address, result->buffers[0].capacity_bytes}, 0};
    }
    return status;
}

void dora_compute_bytes_result_release(struct compute_bytes_result* result) {
    for (uint32_t index = 0; index < 3; ++index) {
        dora_allocation_release(&result->buffers[index]);
    }
}

int dora_compute_bytes_compact(struct dora_pool pool, struct compute_bytes_result* result) {
    if (result->data_bytes > result->buffers[2].capacity_bytes ||
        (result->data_bytes != 0 && result->buffers[2].address == nullptr)) {
        return EINVAL;
    }
    size_t required = 0;
    int status = dora_memory_preferred_capacity(pool, 0, result->data_bytes, &required);
    if (status != DORA_OK || required > result->buffers[2].capacity_bytes / 2) {
        return status;
    }
    struct dora_allocation compact = {};
    status = dora_pool_allocate(pool, result->data_bytes, &compact);
    if (status == DORA_OK) {
        utf8_copy_validated_bytes((struct dora_bytes){result->buffers[2].address, result->data_bytes}, compact.address);
        dora_allocation_release(&result->buffers[2]);
        result->buffers[2] = compact;
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Outputs
//
//------------------------------------------------------------------------------

uint64_t dora_compute_bytes_store_validity(void* validity, uint64_t first, const struct kernel_packet* packet) {
    if (validity != nullptr) {
        uint64_t* const words = (uint64_t*)validity + bitmap_word_of(first);
        if (packet->validity == nullptr) {
            bitmap_prefix_words(packet->count, words, (size_t)bitmap_words_for_bits(packet->count));
            return 0;
        }
        const size_t full_words = (size_t)bitmap_word_of(packet->count);
        dora_bitmap_copy_aligned_words(words, packet->validity, full_words);
        const uint32_t tail_bits = bitmap_bit_of(packet->count);
        if (tail_bits != 0) {
            words[full_words] = packet->validity[full_words] & bitmap_mask_low(tail_bits);
        }
    }
    return packet->validity == nullptr ? 0
                                       : packet->count - dora_bitmap_count_set_words(packet->validity, packet->count);
}

int dora_compute_output_init(struct dora_pool pool,
                             const struct dora_logical_type* type,
                             uint64_t length,
                             bool constant,
                             bool all_valid,
                             struct compute_output* out_output) {
    if (length > INT64_MAX) {
        return EINVAL;
    }
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    int status = type_physical(type, &physical);
    if (status != DORA_OK) {
        return status;
    }
    const uint64_t stored = constant && length != 0 ? 1 : length;
    const size_t width = physical_width(physical);
    const bool bits = type->id == DORA_TYPE_BOOLEAN;
    if (width == 0 && !bits && type->id != DORA_TYPE_NULL) {
        return ENOTSUP;
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, bits ? bitmap_words_for_bits(stored) : stored, bits ? sizeof(uint64_t) : width)) {
        return EOVERFLOW;
    }
    out_output->buffers[0] = (struct dora_allocation){};
    out_output->buffers[1] = (struct dora_allocation){};
    out_output->type = *type;
    out_output->pool = pool;
    out_output->validity_source = nullptr;
    out_output->length = length;
    out_output->stored = stored;
    out_output->null_count = 0;
    out_output->byte_width = (uint32_t)width;
    out_output->is_constant = constant;
    out_output->is_nullable = !all_valid;
    return dora_pool_allocate(pool, bytes, &out_output->buffers[1]);
}

int dora_compute_bytes_check_validity(const struct compute_bytes_result* result,
                                      uint64_t first,
                                      const struct kernel_packet* packet) {
    if (result->validity_source != nullptr) {
        return DORA_OK;
    }
    const uint8_t* const expected = result->buffers[0].address;
    for (uint32_t lane = 0; lane < packet->count; lane += DORA_BITMAP_WORD_BIT_COUNT) {
        const uint32_t remaining = packet->count - lane;
        const uint32_t count = remaining < DORA_BITMAP_WORD_BIT_COUNT ? remaining : DORA_BITMAP_WORD_BIT_COUNT;
        const uint64_t mask = bitmap_mask_low(count);
        const uint64_t left = expected == nullptr ? mask : dora_bitmap_load_window(expected, first + lane, count);
        const uint64_t right =
            packet->validity == nullptr ? mask : bitmap_word_slice(packet->validity[bitmap_word_of(lane)], 0, count);
        if (left != right) {
            return EINVAL;
        }
    }
    return DORA_OK;
}

static int preserved_validity(const struct dora_vector* source,
                              uint64_t length,
                              bool constant,
                              const struct dora_vector** out_source,
                              struct dora_utf8_mask* out_mask) {
    if (constant) {
        return DORA_OK;
    }
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(source, &view);
    if (status != DORA_OK || view.selection.indices != nullptr || view.layout == DORA_LAYOUT_DICTIONARY ||
        view.layout == DORA_LAYOUT_RUN_END || view.length != length) {
        return status;
    }
    bool shareable = false;
    if (view.validity.data != nullptr) {
        status = dora_vector_validity_shareable(source, &shareable);
    }
    if (status == DORA_OK && (view.validity.data == nullptr || shareable)) {
        *out_source = source;
        if (out_mask != nullptr) {
            *out_mask =
                (struct dora_utf8_mask){{view.validity.data, view.validity.byte_count}, view.validity_bit_offset};
        }
    }
    return status;
}

int dora_compute_bytes_preserve_validity(const struct dora_vector* source,
                                         bool constant,
                                         struct compute_bytes_result* result) {
    return preserved_validity(source, result->stored, constant, &result->validity_source, &result->validity);
}

int dora_compute_output_preserve_validity(struct compute_output* output, const struct dora_vector* source) {
    struct dora_utf8_mask mask = {};
    const int status = preserved_validity(source, output->length, output->is_constant, &output->validity_source, &mask);
    if (status == DORA_OK && output->validity_source != nullptr) {
        output->is_nullable = false;
        if (mask.bytes.data == nullptr) {
            output->validity_source = nullptr;
        }
    }
    return status;
}

int dora_compute_output_preserve_operands_validity(struct compute_output* output,
                                                   const struct dora_vector* const* sources,
                                                   const struct vector_operand* operands,
                                                   uint32_t count) {
    const struct dora_vector* source = nullptr;
    struct dora_utf8_mask mask = {};
    for (uint32_t index = 0; index < count; ++index) {
        if (dora_vector_operands_are_all_valid(&operands[index], 1)) {
            continue;
        }
        const struct dora_vector* candidate = nullptr;
        struct dora_utf8_mask candidate_mask = {};
        const int status =
            preserved_validity(sources[index], output->length, output->is_constant, &candidate, &candidate_mask);
        if (status != DORA_OK) {
            return status;
        }
        if (candidate == nullptr) {
            return DORA_OK;
        }
        if (candidate_mask.bytes.data == nullptr) {
            continue;
        }
        if (source != nullptr &&
            (mask.bytes.data != candidate_mask.bytes.data || mask.bit_offset != candidate_mask.bit_offset)) {
            return DORA_OK;
        }
        source = candidate;
        mask = candidate_mask;
    }
    output->validity_source = source;
    return DORA_OK;
}

void dora_compute_output_zero_values(struct compute_output* output) {
    if (output->type.id == DORA_TYPE_BOOLEAN && output->buffers[1].address != nullptr) {
        bitmap_fill_bytes((uint8_t*)output->buffers[1].address, output->buffers[1].capacity_bytes, false);
    }
}

static int output_allocate_validity(struct compute_output* output) {
    return dora_pool_allocate(output->pool,
                              (size_t)bitmap_words_for_bits(output->stored) * sizeof(uint64_t),
                              &output->buffers[0]);
}

int dora_compute_output_commit(struct compute_output* output, uint64_t first, const struct kernel_result* result) {
    output->null_count += output->is_constant && result->null_count != 0 ? output->length : result->null_count;
    if (output->validity_source != nullptr || result->null_count == 0 || result->validity != output->chunk_validity) {
        return DORA_OK;
    }
    const int status = output_allocate_validity(output);
    if (status != DORA_OK) {
        return status;
    }
    uint64_t* const words = output->buffers[0].address;
    const size_t prior = (size_t)bitmap_word_of(first);
    bitmap_fill_bytes((uint8_t*)words, prior * sizeof(uint64_t), true);
    dora_bitmap_copy_aligned_words(words + prior,
                                   output->chunk_validity,
                                   (size_t)bitmap_words_for_bits(result->row_capacity));
    return DORA_OK;
}

int dora_compute_output_set_null(struct compute_output* output, uint64_t row) {
    if (output->validity_source != nullptr) {
        struct dora_vector_view view = {};
        int status = dora_vector_get_view(output->validity_source, &view);
        if (status == DORA_OK) {
            status = output_allocate_validity(output);
        }
        if (status != DORA_OK) {
            return status;
        }
        dora_bitmap_load_words(view.validity.data,
                               view.validity_bit_offset,
                               output->stored,
                               output->buffers[0].address);
        output->validity_source = nullptr;
        output->is_nullable = true;
    }
    if (output->buffers[0].address == nullptr) {
        const int status = output_allocate_validity(output);
        if (status != DORA_OK) {
            return status;
        }
        const uint64_t stored = output->stored;
        const size_t word_count = (size_t)bitmap_words_for_bits(stored);
        uint64_t* const words = output->buffers[0].address;
        bitmap_fill_bytes((uint8_t*)words, word_count * sizeof(uint64_t), true);
        words[word_count - 1] &= bitmap_tail_mask(stored);
    }
    bitmap_clear(output->buffers[0].address, output->is_constant ? 0 : row);
    output->null_count = output->is_constant ? output->length : output->null_count + 1;
    return DORA_OK;
}

int dora_compute_output_null(struct compute_output* output) {
    const int status = output_allocate_validity(output);
    if (status == DORA_OK && output->buffers[0].address != nullptr) {
        bitmap_fill_bytes((uint8_t*)output->buffers[0].address, output->buffers[0].capacity_bytes, false);
    }
    output->null_count = output->length;
    return status;
}

void dora_compute_output_release(struct compute_output* output) {
    dora_allocation_release(&output->buffers[1]);
    dora_allocation_release(&output->buffers[0]);
    *output = (struct compute_output){};
}

int dora_compute_output_share(struct compute_output* output,
                              const struct dora_vector* source,
                              uint32_t shared_index,
                              struct dora_vector* out_result) {
    const struct dora_schema_field field = {
        .type = output->type,
        .layout = output->type.id == DORA_TYPE_BOOLEAN ? DORA_LAYOUT_BITS : DORA_LAYOUT_FIXED,
    };
    const struct vector_input input = {
        .field = &field,
        .row_count = output->length,
        .null_count = (int64_t)output->null_count,
        .allocations = output->buffers,
        .buffer_count = 2,
    };
    return dora_vector_init_share_validity(out_result, &input, source, shared_index, output->buffers);
}

int dora_compute_output_finish(struct compute_output* output, struct dora_vector* out_result) {
    if (output->validity_source != nullptr && output->null_count != 0) {
        return dora_compute_output_share(output, output->validity_source, 0, out_result);
    }
    const bool constant = output->is_constant && output->length != 0;
    const uint64_t stored = output->stored;
    const int64_t nulls = (int64_t)(constant ? output->null_count != 0 : output->null_count);
    const struct dora_schema_field field = {
        .type = output->type,
        .layout = output->type.id == DORA_TYPE_BOOLEAN ? DORA_LAYOUT_BITS
                  : output->type.id == DORA_TYPE_NULL  ? DORA_LAYOUT_NULL
                                                       : DORA_LAYOUT_FIXED
    };
    struct dora_vector value = {};
    struct dora_schema schema = {};
    const struct vector_input input = {
        .field = &field,
        .row_count = stored,
        .null_count = nulls,
        .allocations = output->buffers,
        .buffer_count = field.layout == DORA_LAYOUT_NULL ? 0 : 2,
        .admitted_schema = &schema,
    };
    int status = dora_schema_node_claim(output->pool, &field, nullptr, &schema);
    if (status == DORA_OK) {
        status =
            dora_vector_init_move_input(constant ? &value : out_result, output->pool, &input, output->buffers, nullptr);
    }
    if (status == DORA_OK && constant) {
        status = dora_vector_init_retain_constant_in_pool(out_result, output->pool, &value, output->length);
    }
    dora_schema_release(&schema);
    dora_vector_release(&value);
    return status;
}

struct kernel_result dora_compute_output_result(struct compute_output* output, uint64_t first, uint32_t count) {
    const size_t width = output->byte_width;
    uint8_t* const values = output->buffers[1].address;
    uint64_t* const validity = output->buffers[0].address;
    const uint64_t value_offset =
        output->type.id == DORA_TYPE_BOOLEAN ? bitmap_word_of(first) * sizeof(uint64_t) : first * width;
    return (struct kernel_result){
        .values = values == nullptr ? nullptr : values + value_offset,
        .validity = validity != nullptr   ? validity + bitmap_word_of(first)
                    : output->is_nullable ? output->chunk_validity
                                          : nullptr,
        .row_capacity = count,
    };
}

int dora_compute_output_commit_utf8_mask(struct compute_output* output,
                                         uint64_t first,
                                         uint32_t count,
                                         struct dora_utf8_mask mask) {
    struct kernel_result result = dora_compute_output_result(output, first, count);
    if (result.validity != nullptr) {
        if (mask.bytes.data != nullptr) {
            dora_bitmap_load_words(mask.bytes.data, mask.bit_offset, count, result.validity);
        } else {
            bitmap_prefix_words(count, result.validity, (size_t)bitmap_words_for_bits(count));
        }
    }
    result.null_count =
        mask.bytes.data == nullptr ? 0 : count - dora_bitmap_count_set_bytes(mask.bytes.data, mask.bit_offset, count);
    return dora_compute_output_commit(output, first, &result);
}

int dora_compute_bytes_publish(struct scratch_scope* scope,
                               struct compute_bytes_result* result,
                               const struct dora_logical_type* type,
                               enum dora_layout layout,
                               enum vector_content_state content_state,
                               bool constant,
                               uint64_t length,
                               struct dora_vector* out_result) {
    const struct dora_pool pool = scope->pool;
    const struct dora_schema_field field = {.type = *type, .layout = layout};
    struct dora_allocation* allocations = result->buffers;
    struct dora_bytes published[3] = {
        {.data = allocations[0].address, .byte_count = allocations[0].capacity_bytes},
        {.data = allocations[1].address,
         .byte_count = layout == DORA_LAYOUT_FIXED ? result->data_bytes : allocations[1].capacity_bytes},
        {.data = allocations[2].address, .byte_count = result->data_bytes},
    };
    struct dora_bytes* buffers = published;
    uint32_t count = layout == DORA_LAYOUT_FIXED ? 2 : 3;
    struct dora_allocation sizes = {};
    struct dora_schema schema = {};
    int status = DORA_OK;
    if (layout == DORA_LAYOUT_BINARY_VIEW) {
        size_t bytes = 0;
        status =
            ckd_add(&count, result->backing_count, 3) || ckd_mul(&bytes, count, sizeof *buffers + sizeof *allocations)
                ? EOVERFLOW
                : dora_scratch_allocate(scope, bytes, (void**)&buffers);
        if (status == DORA_OK) {
            allocations = (struct dora_allocation*)(buffers + count);
        }
        if (status == DORA_OK) {
            status = ckd_mul(&bytes, result->backing_count, sizeof(int64_t)) ? EOVERFLOW
                                                                             : dora_pool_allocate(pool, bytes, &sizes);
        }
        if (status == DORA_OK) {
            memset(allocations, 0, count * sizeof *allocations);
            for (uint32_t index = 0; index < 2; ++index) {
                buffers[index] = published[index];
                allocations[index] = result->buffers[index];
            }
            for (uint32_t index = 0; index < result->backing_count; ++index) {
                buffers[index + 2] =
                    (struct dora_bytes){result->backing[index].data, result->backing[index].byte_count};
                ((int64_t*)sizes.address)[index] = (int64_t)result->backing[index].byte_count;
            }
            if (result->buffer_source == nullptr && result->backing_count != 0) {
                allocations[2] = result->buffers[2];
            }
            buffers[count - 1] = (struct dora_bytes){sizes.address, sizes.capacity_bytes};
            allocations[count - 1] = sizes;
        }
    } else if (result->buffer_source != nullptr) {
        published[0] = result->validity.bytes;
        published[2] = (struct dora_bytes){result->backing[0].data, result->backing[0].byte_count};
    }
    if (status == DORA_OK) {
        status = dora_schema_node_claim(pool, &field, nullptr, &schema);
    }
    const struct vector_input input = {
        .field = &field,
        .row_count = result->stored,
        .null_count = result->nulls,
        .validity_bit_offset =
            result->buffer_source != nullptr && layout != DORA_LAYOUT_BINARY_VIEW ? result->validity.bit_offset : 0,
        .buffers = buffers,
        .allocations = allocations,
        .buffer_count = count,
        .buffer_source = result->buffer_source,
        .content_state = content_state,
        .admitted_schema = &schema,
    };
    struct dora_vector value = {};
    if (status == DORA_OK) {
        status = result->validity_source != nullptr && result->nulls != 0 && !constant
                     ? dora_vector_init_share_validity(&value, &input, result->validity_source, 0, allocations)
                     : dora_vector_init_move_input(&value, pool, &input, allocations, nullptr);
        if (layout == DORA_LAYOUT_BINARY_VIEW) {
            for (uint32_t index = 0; index < 2; ++index) {
                result->buffers[index] = allocations[index];
            }
            if (result->buffer_source == nullptr && result->backing_count != 0) {
                result->buffers[2] = allocations[2];
            }
            sizes = allocations[count - 1];
        }
    }
    if (status == DORA_OK) {
        status = constant ? dora_vector_init_retain_constant_in_pool(out_result, pool, &value, length)
                          : dora_vector_move(&value, out_result);
    }
    dora_schema_release(&schema);
    dora_vector_release(&value);
    dora_allocation_release(&sizes);
    return status;
}

int dora_compute_nested_publish(struct scratch_scope* scope,
                                const struct vector_input* input,
                                struct dora_allocation* buffers,
                                struct dora_vector* out_result) {
    struct dora_schema* children = nullptr;
    size_t bytes = 0;
    int status = ckd_mul(&bytes, input->child_count, sizeof(struct dora_schema)) ? EOVERFLOW : DORA_OK;
    if (status == DORA_OK && bytes != 0) {
        status = dora_scratch_allocate(scope, bytes, (void**)&children);
    }
    if (children != nullptr) {
        memset(children, 0, bytes);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input->child_count; ++index) {
        status = dora_vector_retain_schema(&input->children[index], &children[index]);
    }
    struct dora_schema schema = {};
    if (status == DORA_OK) {
        status = dora_schema_node_claim(scope->pool, input->field, children, &schema);
    }
    if (status == DORA_OK) {
        struct vector_input cached = *input;
        cached.admitted_schema = &schema;
        status = dora_vector_init_move_input(out_result, scope->pool, &cached, buffers, nullptr);
    }
    dora_schema_release(&schema);
    for (uint32_t index = 0; children != nullptr && index < input->child_count; ++index) {
        dora_schema_release(&children[index]);
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(sizeof(struct dora_bytes) % alignof(struct dora_allocation) == 0);
