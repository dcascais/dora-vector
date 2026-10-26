// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/compute_internal.h"
#include "compute/output.h"
#include "compute/peel.h"
#include "compute/proof.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/compute.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/read.h"
#include "kernel/binding/request.h"
#include "kernel/decode.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "statistics/bounds.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Scalar Routes
//
//------------------------------------------------------------------------------

struct cast_call {
    const struct dora_vector* source;
    const struct dora_logical_type* target;
    enum dora_rounding_mode mode;
    enum dora_failure_policy policy;
};

//--------------------------------------
// Statistics Proof
//--------------------------------------

static _BitInt(128) cast_bound(enum dora_type type, const union decoded_row_value* value) {
    switch (type) {
    case DORA_TYPE_INT8:
        return value->i8;
    case DORA_TYPE_INT16:
        return value->i16;
    case DORA_TYPE_INT32:
        return value->i32;
    case DORA_TYPE_INT64:
        return value->i64;
    case DORA_TYPE_UINT8:
        return value->u8;
    case DORA_TYPE_UINT16:
        return value->u16;
    case DORA_TYPE_UINT32:
        return value->u32;
    default:
        return value->u64;
    }
}

struct cast_range {
    _BitInt(128) minimum;
    _BitInt(128) maximum;
};

static struct cast_range cast_limits(enum dora_type type) {
    switch (type) {
    case DORA_TYPE_INT8:
        return (struct cast_range){INT8_MIN, INT8_MAX};
    case DORA_TYPE_INT16:
        return (struct cast_range){INT16_MIN, INT16_MAX};
    case DORA_TYPE_INT32:
        return (struct cast_range){INT32_MIN, INT32_MAX};
    case DORA_TYPE_INT64:
        return (struct cast_range){INT64_MIN, INT64_MAX};
    case DORA_TYPE_UINT8:
        return (struct cast_range){0, UINT8_MAX};
    case DORA_TYPE_UINT16:
        return (struct cast_range){0, UINT16_MAX};
    case DORA_TYPE_UINT32:
        return (struct cast_range){0, UINT32_MAX};
    default:
        return (struct cast_range){0, UINT64_MAX};
    }
}

bool dora_compute_cast_is_proven(const struct dora_vector* source,
                                 const struct dora_logical_type* type,
                                 const struct dora_logical_type* target) {
    struct statistics_bounds bounds = {};
    if (!type_is_native_integer(type->id) || !type_is_native_integer(target->id) ||
        !dora_statistics_try_get_exact_bounds(source, type, &bounds)) {
        return false;
    }
    const struct cast_range range = cast_limits(target->id);
    return cast_bound(type->id, &bounds.minimum) >= range.minimum &&
           cast_bound(type->id, &bounds.maximum) <= range.maximum;
}

static int cast_bind(const struct vector_bound_input* input,
                     const struct cast_call* call,
                     bool measure,
                     struct kernel_unary_binding* out_binding) {
    struct kernel_bind_request request = {
        .input_type = &input->nodes[0].field.type,
        .result_type = call->target,
        .policy = call->policy,
        .rounding = call->mode,
    };
    return measure ? dora_kernel_cast_measure_bind(&request, out_binding)
                   : dora_kernel_cast_bind(&request, out_binding);
}

static int cast_fixed(struct dora_pool pool,
                      const struct vector_bound_input* input,
                      const struct cast_call* call,
                      struct dora_vector* out_result) {
    struct kernel_unary_binding binding = {};
    struct cast_call bound = *call;
    if (dora_compute_cast_is_proven(call->source, &input->nodes[0].field.type, call->target)) {
        bound.policy = DORA_FAILURE_POLICY_WRAP;
    }
    int status = cast_bind(input, &bound, false, &binding);
    struct scratch_scope scope = {};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &scope);
    }
    struct vector_operand operand = {};
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(&scope, 0, input, VECTOR_OPERAND_REQUIRE_PACKET, &operand);
    }
    const bool all_valid = binding.is_result_all_valid && call->target->id != DORA_TYPE_NULL &&
                           dora_vector_operands_are_all_valid(&operand, 1);
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, call->target, input->length, input->is_constant, all_valid, &output);
    }
    if (status == DORA_OK && call->policy != DORA_FAILURE_POLICY_NULL && call->target->id != DORA_TYPE_NULL) {
        status = dora_compute_output_preserve_operands_validity(&output,
                                                                (const struct dora_vector* const[]){call->source},
                                                                &operand,
                                                                1);
    }
    if (status == DORA_OK && call->target->id == DORA_TYPE_BOOLEAN) {
        dora_compute_output_zero_values(&output);
    }
    const uint64_t count = output.stored;
    for (uint64_t first = 0; status == DORA_OK && first < count; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, count);
        struct kernel_packet packet;
        status = dora_vector_operand_prepare(&operand, rows, nullptr, &packet);
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        if (status == DORA_OK) {
            status = binding.fill(&packet, &binding, &result);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &result);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, out_result);
    }
    dora_compute_output_release(&output);
    dora_scratch_scope_end(&scope);
    return status;
}

static int cast_measure(const struct kernel_unary_binding* binding,
                        struct vector_operand* operand,
                        int64_t* offsets,
                        uint64_t count,
                        size_t* out_byte_count) {
    size_t byte_count = 0;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < count; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, count);
        struct kernel_packet packet;
        status = dora_vector_operand_prepare(operand, rows, nullptr, &packet);
        struct kernel_result result = {.values = offsets + first + 1, .row_capacity = (uint32_t)rows.length};
        if (status == DORA_OK) {
            status = binding->fill(&packet, binding, &result);
        }
        for (uint64_t row = first; status == DORA_OK && row < first + rows.length; ++row) {
            if (ckd_add(&byte_count, byte_count, (size_t)offsets[row + 1])) {
                status = EOVERFLOW;
            }
        }
    }
    *out_byte_count = byte_count;
    dora_vector_operand_reset(operand);
    return status;
}

static int cast_text(struct dora_pool pool,
                     const struct vector_bound_input* input,
                     const struct cast_call* call,
                     struct dora_vector* out_result) {
    const bool fixed = call->target->id == DORA_TYPE_FIXED_BINARY;
    const bool bytes = type_is_bytes(input->nodes[0].field.type.id);
    const uint64_t count = input->is_constant && input->length != 0 ? 1 : input->length;
    struct kernel_unary_binding binding = {};
    int status = cast_bind(input, call, false, &binding);
    struct scratch_scope scope = {};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &scope);
    }
    struct vector_operand operand = {};
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(&scope, 0, input, VECTOR_OPERAND_REQUIRE_PACKET, &operand);
    }
    struct compute_bytes_result result = {.stored = count};
    struct dora_allocation* const buffers = result.buffers;
    struct dora_allocation* const data = &buffers[fixed ? 1 : 2];
    size_t offset_bytes = 0;
    if (status == DORA_OK && ckd_mul(&offset_bytes, count + 1, sizeof(int64_t))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK && !(binding.is_result_all_valid && dora_vector_operands_are_all_valid(&operand, 1))) {
        status = dora_pool_allocate(pool, (size_t)bitmap_words_for_bits(count) * sizeof(uint64_t), &buffers[0]);
    }
    if (status == DORA_OK && !fixed) {
        status = dora_pool_allocate(pool, offset_bytes, &buffers[1]);
    }
    int64_t* const offsets = fixed || status != DORA_OK ? nullptr : buffers[1].address;
    if (offsets != nullptr) {
        offsets[0] = 0;
    }
    size_t capacity = 0;
    if (status == DORA_OK && fixed) {
        status = ckd_mul(&capacity, (size_t)count, (size_t)binding.result_byte_width) ? EOVERFLOW : DORA_OK;
    } else if (status == DORA_OK && bytes && offsets != nullptr) {
        struct kernel_unary_binding measure = {};
        status = cast_bind(input, call, true, &measure);
        if (status == DORA_OK) {
            status = cast_measure(&measure, &operand, offsets, count, &capacity);
        }
    }
    if (status == DORA_OK && (uint64_t)capacity > INT64_MAX) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK && (fixed || bytes)) {
        status = dora_pool_allocate(pool, capacity, data);
    }
    // Staging avoids reserving worst-case result storage.
    const bool staged = !fixed && !bytes;
    uint8_t* staging = nullptr;
    size_t staging_bytes = 0;
    if (status == DORA_OK && staged &&
        ckd_mul(&staging_bytes,
                (size_t)(count < KERNEL_CHUNK_ROW_COUNT ? count : KERNEL_CHUNK_ROW_COUNT),
                binding.cast.maximum_text_byte_count)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK && staged) {
        status = dora_scratch_slot_require(&scope,
                                           vector_operand_slot(0, VECTOR_OPERAND_SLOT_TEXT),
                                           (void**)&staging,
                                           staging_bytes,
                                           64);
    }
    struct kernel_text_output output = {.data = data->address};
    uint64_t null_count = 0;
    for (uint64_t first = 0; status == DORA_OK && first < count; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, count);
        struct kernel_packet packet;
        status = dora_vector_operand_prepare(&operand, rows, nullptr, &packet);
        struct kernel_text_output chunk = {.end_offsets = offsets != nullptr ? offsets + first + 1 : nullptr};
        if (staged) {
            chunk.data = staging;
        } else {
            chunk.data = fixed && data->address != nullptr
                             ? (uint8_t*)data->address + (size_t)first * binding.result_byte_width
                             : data->address;
            chunk.written_byte_count = output.written_byte_count;
        }
        struct kernel_result result = {
            .values = &chunk,
            .validity = buffers[0].address == nullptr ? nullptr : (uint64_t*)buffers[0].address + bitmap_word_of(first),
            .row_capacity = (uint32_t)rows.length,
        };
        if (status == DORA_OK) {
            status = binding.fill(&packet, &binding, &result);
        }
        null_count += result.null_count;
        if (!staged) {
            output.written_byte_count = chunk.written_byte_count;
            continue;
        }
        size_t required = 0;
        if (status == DORA_OK &&
            (ckd_add(&required, (size_t)output.written_byte_count, (size_t)chunk.written_byte_count) ||
             (uint64_t)required > INT64_MAX)) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK && chunk.written_byte_count != 0) {
            status = dora_allocation_reserve(pool, required, (size_t)output.written_byte_count, data);
        }
        if (status == DORA_OK && chunk.written_byte_count != 0) {
            (memcpy)((uint8_t*)data->address + output.written_byte_count, staging, (size_t)chunk.written_byte_count);
        }
        if (status == DORA_OK) {
            for (uint32_t row = 0; row < rows.length; ++row) {
                chunk.end_offsets[row] += output.written_byte_count;
            }
            output.written_byte_count = required;
        }
    }
    if (status == DORA_OK) {
        result.nulls = null_count;
        result.data_bytes = fixed ? capacity : (size_t)output.written_byte_count;
        status = dora_compute_bytes_publish(
            &scope,
            &result,
            call->target,
            fixed ? DORA_LAYOUT_FIXED : DORA_LAYOUT_LARGE_BINARY,
            staged && call->target->id == DORA_TYPE_UTF8 ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_PROMISED,
            input->is_constant && input->length != 0,
            input->length,
            out_result);
    }
    for (uint32_t index = 0; index < 3; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    dora_scratch_scope_end(&scope);
    return status;
}

struct cast_owner {
    struct dora_buffer_owner owner;
    struct dora_allocation allocation;
    struct dora_allocation activity;
    struct dora_allocation offsets;
    struct dora_vector source;
};

static void cast_owner_release(struct dora_buffer_owner* owner) {
    struct cast_owner* const state = (struct cast_owner*)owner;
    struct dora_allocation allocation = state->allocation;
    dora_vector_release(&state->source);
    dora_allocation_release(&state->activity);
    dora_allocation_release(&state->offsets);
    dora_allocation_release(&allocation);
}

static int cast_owner_init(struct dora_pool pool, const struct dora_vector* source, struct cast_owner** out) {
    struct dora_allocation allocation = {};
    int status = dora_memory_payload_allocate(pool, sizeof(struct cast_owner), &allocation);
    if (status != DORA_OK) {
        return status;
    }
    struct cast_owner* const owner = allocation.address;
    *owner = (struct cast_owner){
        .owner =
            {.placement = {.device = {.type = DORA_DEVICE_TYPE_CPU}, .access = DORA_ACCESS_HOST},
             .release = cast_owner_release},
        .allocation = allocation
    };
    // A foreign claim keeps shared source buffers alive.
    struct dora_handle claim = {};
    status = dora_identity_claim_acquire(pool, &source->handle, IDENTITY_KIND_VECTOR, &claim);
    if (status == DORA_OK) {
        owner->source = *source;
        owner->source.handle = claim;
    }
    if (status != DORA_OK) {
        cast_owner_release(&owner->owner);
    } else {
        *out = owner;
    }
    return status;
}

struct cast_chain {
    struct kernel_decode_binding links[2];
    uint32_t length;
};

struct cast_chunk {
    uint64_t rows[KERNEL_CHUNK_ROW_COUNT];
    uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
};

static int cast_chain_bind(const struct dora_vector_view* view, struct cast_chain* out_chain) {
    struct cast_chain chain = {};
    int status = DORA_OK;
    if (view->selection.indices != nullptr) {
        status = dora_kernel_decode_bind(view, &chain.links[chain.length++]);
    }
    if (status == DORA_OK && view->layout == DORA_LAYOUT_RUN_END) {
        struct dora_vector_view plain = *view;
        plain.selection = (struct dora_vector_selection){};
        status = dora_kernel_decode_bind(&plain, &chain.links[chain.length]);
        chain.links[chain.length++].view = view;
    }
    *out_chain = chain;
    return status;
}

// Run-end links produce run indices.
static void cast_chain_resolve(struct cast_chain* chain, struct dora_range rows, struct cast_chunk* out_chunk) {
    bitmap_prefix_words(rows.length, out_chunk->validity, (size_t)bitmap_words_for_bits(rows.length));
    for (uint32_t index = 0; index < chain->length; ++index) {
        chain->links[index].resolve(&chain->links[index],
                                    index == 0 ? rows.offset : UINT64_MAX,
                                    (uint32_t)rows.length,
                                    out_chunk->rows,
                                    out_chunk->validity);
    }
}

static void
cast_chunk_validity(const struct dora_vector_view* view, struct dora_range rows, struct cast_chunk* in_out_chunk) {
    if (view->validity.data == nullptr) {
        return;
    }
    uint64_t active[KERNEL_CHUNK_WORD_COUNT];
    bitmap_copy_bytes((uint8_t*)active,
                      (const uint8_t*)in_out_chunk->validity,
                      (size_t)bitmap_words_for_bits(rows.length) * sizeof(uint64_t));
    dora_bitmap_gather_words_u64(in_out_chunk->validity,
                                 view->validity.data,
                                 view->validity_bit_offset - view->offset,
                                 in_out_chunk->rows,
                                 active,
                                 (uint32_t)rows.length);
}

static int cast_commit_validity(struct compute_output* output, struct dora_range rows, const uint64_t* words) {
    struct kernel_result result = dora_compute_output_result(output, rows.offset, (uint32_t)rows.length);
    dora_bitmap_copy_aligned_words(result.validity, words, (size_t)bitmap_words_for_bits(rows.length));
    result.null_count = (uint32_t)rows.length - (uint32_t)dora_bitmap_count_set_words(words, rows.length);
    return dora_compute_output_commit(output, rows.offset, &result);
}

static int cast_wrap_leaf(struct scratch_scope* scope,
                          const struct dora_vector* source,
                          const struct dora_vector_view* view,
                          const struct dora_logical_type* type,
                          struct dora_vector* out) {
    const struct dora_pool pool = scope->pool;
    struct cast_owner* owner = nullptr;
    int status = cast_owner_init(pool, source, &owner);
    if (status != DORA_OK) {
        return status;
    }
    uint64_t count = view->length;
    const uint8_t* validity = view->validity.data;
    uint64_t validity_offset = view->validity_bit_offset;
    const bool selected = view->selection.indices != nullptr;
    struct cast_chain chain = {};
    struct cast_chunk chunk = {};
    if (selected) {
        status = cast_chain_bind(view, &chain);
        count = 0;
        for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
            const struct dora_range rows = vector_operand_chunk(first, view->length);
            cast_chain_resolve(&chain, rows, &chunk);
            for (uint32_t lane = 0; lane < rows.length; ++lane) {
                if (chunk.rows[lane] - view->offset >= count) {
                    count = chunk.rows[lane] - view->offset + 1;
                }
            }
        }
        if (status == DORA_OK) {
            status =
                dora_pool_allocate(pool, (size_t)bitmap_words_for_bits(count) * sizeof(uint64_t), &owner->activity);
        }
        if (status == DORA_OK && owner->activity.capacity_bytes != 0) {
            bitmap_fill_bytes((uint8_t*)owner->activity.address, owner->activity.capacity_bytes, false);
        }
        for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
            const struct dora_range rows = vector_operand_chunk(first, view->length);
            cast_chain_resolve(&chain, rows, &chunk);
            cast_chunk_validity(view, rows, &chunk);
            for (uint32_t word = 0; word < (uint32_t)bitmap_words_for_bits(rows.length); ++word) {
                uint64_t valid = chunk.validity[word];
                while (valid != 0) {
                    const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&valid);
                    bitmap_set(owner->activity.address, chunk.rows[lane] - view->offset);
                }
            }
        }
        validity = owner->activity.address;
        validity_offset = 0;
    }
    struct dora_vector wrapped = {};
    bool owner_consumed = false;
    const size_t width = view->type == DORA_TYPE_FIXED_BINARY ? view->byte_width : physical_width(view->physical_type);
    const uint8_t* const values =
        view->values.data != nullptr ? (const uint8_t*)view->values.data + view->offset * width : nullptr;
    if (status == DORA_OK && view->layout == DORA_LAYOUT_BITS) {
        owner_consumed = true;
        status = dora_vector_init_wrap_packed_bool(&wrapped,
                                                   pool,
                                                   type,
                                                   view->values.data,
                                                   view->offset,
                                                   validity,
                                                   validity_offset,
                                                   count,
                                                   &owner->owner,
                                                   nullptr);
    } else if (status == DORA_OK && view->layout == DORA_LAYOUT_FIXED && view->type == DORA_TYPE_FIXED_BINARY &&
               type->id == DORA_TYPE_UTF8) {
        size_t offset_bytes = 0, data_bytes = 0;
        if (count == UINT64_MAX || ckd_mul(&offset_bytes, count + 1, sizeof(int64_t)) ||
            ckd_mul(&data_bytes, count, width) || data_bytes > INT64_MAX) {
            status = EOVERFLOW;
        } else {
            status = dora_pool_allocate(pool, offset_bytes, &owner->offsets);
        }
        if (status == DORA_OK) {
            int64_t* const offsets = owner->offsets.address;
            for (uint64_t row = 0; row <= count; ++row) {
                offsets[row] = (int64_t)(row * width);
            }
            owner_consumed = true;
            status = dora_vector_init_wrap_large_binary(&wrapped,
                                                        pool,
                                                        type,
                                                        offsets,
                                                        (struct dora_bytes){values, data_bytes},
                                                        validity,
                                                        validity_offset,
                                                        count,
                                                        &owner->owner,
                                                        nullptr);
        }
    } else if (status == DORA_OK && view->layout == DORA_LAYOUT_FIXED && type->id == DORA_TYPE_FIXED_BINARY) {
        owner_consumed = true;
        status = dora_vector_init_wrap_fixed_binary(&wrapped,
                                                    pool,
                                                    type,
                                                    values,
                                                    validity,
                                                    validity_offset,
                                                    count,
                                                    &owner->owner,
                                                    nullptr);
    } else if (status == DORA_OK && (view->layout == DORA_LAYOUT_FIXED || view->layout == DORA_LAYOUT_DICTIONARY)) {
        // Switch case; breaks after wrapping borrowed values.
        // Carriers: signed/unsigned integers, floating-point types, IntervalDayTime, IntervalMonthDayNano
#define CAST_WRAP(PHYSICAL, NAME, CTYPE)                                                                               \
    case DORA_PHYSICAL_TYPE_##PHYSICAL:                                                                                \
        owner_consumed = true;                                                                                         \
        status = dora_vector_init_wrap_##NAME(&wrapped,                                                                \
                                              pool,                                                                    \
                                              type,                                                                    \
                                              (const CTYPE*)values,                                                    \
                                              validity,                                                                \
                                              validity_offset,                                                         \
                                              count,                                                                   \
                                              &owner->owner,                                                           \
                                              nullptr);                                                                \
        break;
        switch (view->physical_type) {
            CAST_WRAP(I8, i8, int8_t)
            CAST_WRAP(I16, i16, int16_t)
            CAST_WRAP(I32, i32, int32_t)
            CAST_WRAP(I64, i64, int64_t)
            CAST_WRAP(I128, i128, struct dora_i128)
            CAST_WRAP(I256, i256, struct dora_i256)
            CAST_WRAP(U8, u8, uint8_t)
            CAST_WRAP(U16, u16, uint16_t)
            CAST_WRAP(U32, u32, uint32_t)
            CAST_WRAP(U64, u64, uint64_t)
            CAST_WRAP(U128, u128, struct dora_u128)
            CAST_WRAP(F16, f16, struct dora_f16_bits)
            CAST_WRAP(F32, f32, float)
            CAST_WRAP(F64, f64, double)
            CAST_WRAP(INTERVAL_DAY_TIME, interval_day_time, struct dora_interval_day_time)
            CAST_WRAP(INTERVAL_MONTH_DAY_NANO, interval_month_day_nano, struct dora_interval_month_day_nano)
        default:
            status = ENOTSUP;
            break;
        }
#undef CAST_WRAP
    } else if (status == DORA_OK && view->layout == DORA_LAYOUT_BINARY) {
        const struct dora_bytes data = {
            .data = view->data_buffers[0].data,
            .byte_count = view->data_buffers[0].byte_count
        };
        const int32_t* offsets = (const int32_t*)((const uint8_t*)view->offsets.data + view->offset * sizeof(int32_t));
        owner_consumed = true;
        status = dora_vector_init_wrap_binary(&wrapped,
                                              pool,
                                              type,
                                              offsets,
                                              data,
                                              validity,
                                              validity_offset,
                                              count,
                                              &owner->owner,
                                              nullptr);
    } else if (status == DORA_OK && view->layout == DORA_LAYOUT_LARGE_BINARY) {
        const struct dora_bytes data = {
            .data = view->data_buffers[0].data,
            .byte_count = view->data_buffers[0].byte_count
        };
        const int64_t* offsets = (const int64_t*)((const uint8_t*)view->offsets.data + view->offset * sizeof(int64_t));
        owner_consumed = true;
        status = dora_vector_init_wrap_large_binary(&wrapped,
                                                    pool,
                                                    type,
                                                    offsets,
                                                    data,
                                                    validity,
                                                    validity_offset,
                                                    count,
                                                    &owner->owner,
                                                    nullptr);
    } else if (status == DORA_OK && view->layout == DORA_LAYOUT_BINARY_VIEW) {
        struct dora_bytes* data;
        status = dora_scratch_allocate(scope, (size_t)view->data_buffer_count * sizeof(*data), (void**)&data);
        if (status == DORA_OK) {
            for (uint32_t index = 0; index < view->data_buffer_count; ++index) {
                data[index] = (struct dora_bytes){
                    .data = view->data_buffers[index].data,
                    .byte_count = view->data_buffers[index].byte_count
                };
            }
            const struct dora_binary_view_entry* entries =
                view->values.data != nullptr
                    ? (const struct dora_binary_view_entry*)((const uint8_t*)view->values.data +
                                                             view->offset * sizeof(struct dora_binary_view_entry))
                    : nullptr;
            owner_consumed = true;
            status = dora_vector_init_wrap_binary_view(&wrapped,
                                                       pool,
                                                       type,
                                                       entries,
                                                       data,
                                                       view->data_buffer_count,
                                                       validity,
                                                       validity_offset,
                                                       count,
                                                       &owner->owner,
                                                       nullptr);
        }
    } else if (status == DORA_OK) {
        status = ENOTSUP;
    }
    if (status != DORA_OK) {
        if (!owner_consumed) {
            cast_owner_release(&owner->owner);
        }
        return status;
    }
    if (!selected) {
        return dora_vector_move(&wrapped, out);
    }
    struct compute_output indices = {};
    const struct dora_logical_type index_type = {.id = DORA_TYPE_UINT64};
    status = dora_compute_output_init(pool, &index_type, view->length, false, false, &indices);
    for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, view->length);
        cast_chain_resolve(&chain, rows, &chunk);
        cast_chunk_validity(view, rows, &chunk);
        uint64_t* const out = (uint64_t*)indices.buffers[1].address + rows.offset;
        for (uint32_t lane = 0; lane < rows.length; ++lane) {
            out[lane] = chunk.rows[lane] - view->offset;
        }
        status = cast_commit_validity(&indices, rows, chunk.validity);
    }
    struct dora_vector index_vector = {};
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&indices, &index_vector);
    }
    if (status == DORA_OK) {
        status = dora_vector_init_retain_dictionary_in_pool(out, pool, &index_vector, &wrapped);
    }
    dora_vector_release(&index_vector);
    dora_vector_release(&wrapped);
    dora_compute_output_release(&indices);
    return status;
}

static int cast_nested(struct scratch_scope* scope,
                       const struct dora_vector* source,
                       const struct vector_bound_input* input,
                       uint32_t node,
                       struct dora_vector* out) {
    const struct vector_bound_node* const bound = &input->nodes[node];
    struct dora_vector* children;
    int status = dora_scratch_allocate(scope, (size_t)bound->field.child_count * sizeof(*children), (void**)&children);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_pool pool = scope->pool;
    const struct dora_vector_view* const view = &bound->view;
    struct dora_schema_field field = bound->field;
    field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
    field.type.extension_name = (struct dora_bytes){};
    field.type.extension_metadata = (struct dora_bytes){};
    if (field.child_count != 0) {
        memset(children, 0, (size_t)field.child_count * sizeof(*children));
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        status = field.layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, index, &children[index])
                                                    : dora_vector_get_physical_child(source, index, &children[index]);
    }
    struct dora_allocation buffers[3] = {};
    uint32_t buffer_count = 1;
    uint64_t published_rows = view->length;
    struct compute_output union_indices = {};
    struct dora_vector union_index_vector = {};
    bool masked_union = false;
    int64_t null_count = 0;
    if (status == DORA_OK && view->null_count != 0) {
        status = dora_pool_allocate(pool, (size_t)bitmap_words_for_bits(view->length) * sizeof(uint64_t), &buffers[0]);
    }
    if (status == DORA_OK && buffers[0].capacity_bytes != 0) {
        bitmap_fill_bytes((uint8_t*)buffers[0].address, buffers[0].capacity_bytes, false);
    }
    for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
        if (dora_vector_view_is_null(view, row)) {
            ++null_count;
        } else if (buffers[0].address != nullptr) {
            bitmap_set(buffers[0].address, row);
        }
    }
    if (status == DORA_OK && field.type.id == DORA_TYPE_LIST) {
        field.layout = DORA_LAYOUT_LARGE_LIST_VIEW;
        buffer_count = 3;
        size_t bytes = 0;
        if (ckd_mul(&bytes, view->length, sizeof(int64_t))) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK) {
            status = dora_pool_allocate(pool, bytes, &buffers[1]);
        }
        if (status == DORA_OK) {
            status = dora_pool_allocate(pool, bytes, &buffers[2]);
        }
        for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
            const struct dora_range range = dora_vector_view_get_elements(view, row);
            ((int64_t*)buffers[1].address)[row] = range.offset;
            ((int64_t*)buffers[2].address)[row] = range.length;
        }
    } else if (status == DORA_OK && (field.type.id == DORA_TYPE_FIXED_LIST || field.type.id == DORA_TYPE_MAP)) {
        uint64_t entries = 0;
        const bool fixed = field.type.id == DORA_TYPE_FIXED_LIST;
        if (fixed && ckd_mul(&entries, view->length, view->list_element_count)) {
            status = EOVERFLOW;
        }
        if (!fixed) {
            field.layout = DORA_LAYOUT_LARGE_LIST;
            buffer_count = 2;
            size_t bytes = 0;
            if (ckd_mul(&bytes, view->length + 1, sizeof(int64_t))) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK) {
                status = dora_pool_allocate(pool, bytes, &buffers[1]);
            }
            if (status == DORA_OK) {
                ((int64_t*)buffers[1].address)[0] = 0;
            }
            for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
                const struct dora_range range = dora_vector_view_is_null(view, row)
                                                    ? (struct dora_range){}
                                                    : dora_vector_view_get_elements(view, row);
                if (ckd_add(&entries, entries, range.length) || entries > INT64_MAX) {
                    status = EOVERFLOW;
                } else {
                    ((int64_t*)buffers[1].address)[row + 1] = entries;
                }
            }
        }
        struct compute_output indices = {};
        const struct dora_logical_type index_type = {.id = DORA_TYPE_UINT64};
        if (status == DORA_OK) {
            status = dora_compute_output_init(pool, &index_type, entries, false, true, &indices);
        }
        uint64_t destination = 0;
        for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
            if (!fixed && dora_vector_view_is_null(view, row)) {
                continue;
            }
            const struct dora_range range =
                fixed
                    ? (struct
                       dora_range){.offset = dora_vector_view_get_physical_row(view, row) * view->list_element_count, .length = view->list_element_count}
                    : dora_vector_view_get_elements(view, row);
            for (uint64_t element = 0; element < range.length; ++element) {
                ((uint64_t*)indices.buffers[1].address)[destination++] = range.offset + element;
            }
        }
        struct dora_vector index_vector = {};
        if (status == DORA_OK) {
            status = dora_compute_output_finish(&indices, &index_vector);
        }
        for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
            struct dora_vector selected = {};
            status = dora_vector_init_retain_selection_in_pool(&selected, pool, &children[index], &index_vector);
            if (status == DORA_OK) {
                dora_vector_release(&children[index]);
                status = dora_vector_move(&selected, &children[index]);
            }
            dora_vector_release(&selected);
        }
        dora_vector_release(&index_vector);
        dora_compute_output_release(&indices);
    } else if (status == DORA_OK && field.type.id == DORA_TYPE_UNION) {
        field.layout = DORA_LAYOUT_DENSE_UNION;
        buffer_count = 2;
        masked_union = null_count != 0;
        published_rows = 0;
        if (masked_union) {
            const struct dora_logical_type index_type = {.id = DORA_TYPE_UINT64};
            status = dora_compute_output_init(pool, &index_type, view->length, false, false, &union_indices);
        }
        dora_allocation_release(&buffers[0]);
        if (status == DORA_OK) {
            status = dora_pool_allocate(pool, (size_t)view->length, &buffers[0]);
        }
        size_t bytes = 0;
        if (status == DORA_OK && ckd_mul(&bytes, view->length, sizeof(int32_t))) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK) {
            status = dora_pool_allocate(pool, bytes, &buffers[1]);
        }
        for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
            if (dora_vector_view_is_null(view, row)) {
                status = dora_compute_output_set_null(&union_indices, row);
                continue;
            }
            if (masked_union) {
                ((uint64_t*)union_indices.buffers[1].address)[row] = published_rows;
            }
            const uint64_t slot = dora_vector_view_get_physical_row(view, row);
            int8_t type_id = 0;
            (memcpy)(&type_id, (const uint8_t*)view->type_ids.data + slot, sizeof(type_id));
            uint64_t member_row = slot;
            if (view->layout == DORA_LAYOUT_DENSE_UNION) {
                int32_t stored_row = 0;
                (memcpy)(&stored_row,
                         (const uint8_t*)view->offsets.data + slot * sizeof(stored_row),
                         sizeof(stored_row));
                member_row = stored_row;
            }
            if (member_row > INT32_MAX) {
                status = EOVERFLOW;
                break;
            }
            ((int8_t*)buffers[0].address)[published_rows] = type_id;
            ((int32_t*)buffers[1].address)[published_rows++] = member_row;
        }
        null_count = 0;
    } else if (status == DORA_OK && field.type.id != DORA_TYPE_STRUCT) {
        status = ENOTSUP;
    }
    struct dora_vector dense = {};
    if (status == DORA_OK) {
        const struct vector_input result = {
            .field = &field,
            .row_count = published_rows,
            .null_count = null_count,
            .allocations = buffers,
            .buffer_count = buffer_count,
            .children = children,
            .child_count = field.child_count
        };
        status = dora_compute_nested_publish(scope, &result, buffers, masked_union ? &dense : out);
    }
    if (status == DORA_OK && masked_union) {
        status = dora_compute_output_finish(&union_indices, &union_index_vector);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_dictionary_in_pool(out, pool, &union_index_vector, &dense);
        }
    }
    dora_vector_release(&dense);
    dora_vector_release(&union_index_vector);
    dora_compute_output_release(&union_indices);
    for (uint32_t index = 0; index < 3; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    for (uint32_t index = 0; index < field.child_count; ++index) {
        dora_vector_release(&children[index]);
    }
    return status;
}

static int cast_same_storage(struct scratch_scope* scope,
                             const struct dora_vector* source,
                             const struct vector_bound_input* input,
                             uint32_t node,
                             const struct dora_logical_type* target,
                             struct dora_vector* out) {
    const struct dora_pool pool = scope->pool;
    const struct vector_bound_node* const bound = &input->nodes[node];
    const struct dora_vector_view* const view = &bound->view;
    struct dora_logical_type type = target != nullptr ? *target : bound->field.type;
    type.extension_name = (struct dora_bytes){};
    type.extension_metadata = (struct dora_bytes){};
    if (view->layout == DORA_LAYOUT_NULL) {
        return dora_vector_init_null(out, pool, view->length);
    }
    if (view->layout != DORA_LAYOUT_DICTIONARY && view->layout != DORA_LAYOUT_RUN_END) {
        return type.id >= DORA_TYPE_LIST ? cast_nested(scope, source, input, node, out)
                                         : cast_wrap_leaf(scope, source, view, &type, out);
    }
    const bool runs = view->layout == DORA_LAYOUT_RUN_END;
    const uint32_t ordinal = runs ? 1 : 0;
    const uint32_t value_node = dora_vector_input_child(node, input, ordinal);
    struct dora_vector source_values = {};
    struct dora_vector values = {};
    int status = dora_vector_get_child(source, ordinal, &source_values);
    if (status == DORA_OK) {
        status = cast_same_storage(scope, &source_values, input, value_node, target, &values);
    }
    const bool constant = runs && input->is_constant && node == 0 && view->length != 0;
    bool retained_runs = false;
    if (status == DORA_OK && runs && view->selection.indices == nullptr && view->validity.data == nullptr &&
        view->offset == 0 && view->length != 0) {
        const uint32_t end_node = dora_vector_input_child(node, input, 0);
        const struct dora_vector_view* const ends = &input->nodes[end_node].view;
        if (ends->length != 0) {
            int64_t endpoint = 0;
            status = dora_kernel_run_end_last(ends, &endpoint);
            if (status == DORA_OK && endpoint >= 0 && (uint64_t)endpoint == view->length) {
                struct dora_vector children[2] = {};
                children[1] = dora_vector_borrow(&values);
                status = dora_vector_get_child(source, 0, &children[0]);
                if (status == DORA_OK) {
                    struct dora_schema_field field = bound->field;
                    field.type = type;
                    field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
                    const struct vector_input result =
                        {.field = &field, .row_count = view->length, .children = children, .child_count = 2};
                    status = dora_compute_nested_publish(scope, &result, nullptr, out);
                }
                retained_runs = true;
            }
        }
    }
    struct dora_vector indices = {};
    if (status == DORA_OK && !retained_runs && !runs) {
        const struct dora_logical_type index_type = {.id = bound->field.index_type};
        status = cast_wrap_leaf(scope, source, view, &index_type, &indices);
    }
    struct cast_chain chain = {};
    struct cast_chunk chunk = {};
    if (status == DORA_OK && !retained_runs && runs) {
        status = cast_chain_bind(view, &chain);
    }
    if (status == DORA_OK && !retained_runs && runs && !constant) {
        struct compute_output index_output = {};
        const struct dora_logical_type index_type = {.id = DORA_TYPE_UINT64};
        status = dora_compute_output_init(pool, &index_type, view->length, false, false, &index_output);
        for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
            const struct dora_range rows = vector_operand_chunk(first, view->length);
            cast_chain_resolve(&chain, rows, &chunk);
            (memcpy)((uint64_t*)index_output.buffers[1].address + rows.offset,
                     chunk.rows,
                     (size_t)rows.length * sizeof(uint64_t));
            status = cast_commit_validity(&index_output, rows, chunk.validity);
        }
        if (status == DORA_OK) {
            status = dora_compute_output_finish(&index_output, &indices);
        }
        dora_compute_output_release(&index_output);
    }
    if (status == DORA_OK && !retained_runs && constant) {
        cast_chain_resolve(&chain, (struct dora_range){.offset = 0, .length = 1}, &chunk);
        struct dora_vector value = {};
        status = dora_vector_slice(&values, (struct dora_range){.offset = chunk.rows[0], .length = 1}, &value);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_constant_in_pool(out, pool, &value, view->length);
        }
        dora_vector_release(&value);
    } else if (status == DORA_OK && !retained_runs) {
        status = dora_vector_init_retain_dictionary_in_pool(out, pool, &indices, &values);
    }
    dora_vector_release(&indices);
    dora_vector_release(&values);
    dora_vector_release(&source_values);
    return status;
}

static bool cast_name_matches(struct dora_bytes name, const char* expected) {
    const size_t size = strlen(expected);
    return name.byte_count == size && memcmp(name.data, expected, size) == 0;
}

static int cast_admit(const struct dora_logical_type* source,
                      const struct dora_logical_type* target,
                      enum dora_rounding_mode mode) {
    if (source->id == DORA_TYPE_NULL || target->id == DORA_TYPE_NULL) {
        return DORA_OK;
    }
    if (target->id >= DORA_TYPE_LIST || source->id >= DORA_TYPE_LIST) {
        return ENOTSUP;
    }
    if (target->id == DORA_TYPE_UTF8 && (source->id == DORA_TYPE_BINARY || source->id == DORA_TYPE_FIXED_BINARY)) {
        return ENOTSUP;
    }
    if (type_is_bytes(target->id)) {
        return target->id != DORA_TYPE_UTF8 && source->id >= DORA_TYPE_DATE32 ? ENOTSUP : DORA_OK;
    }
    if (target->id >= DORA_TYPE_FLOAT16 && target->id <= DORA_TYPE_FLOAT64 && mode != DORA_ROUNDING_MODE_EXACT &&
        mode != DORA_ROUNDING_MODE_HALF_EVEN) {
        return EINVAL;
    }
    if (source->id >= DORA_TYPE_INTERVAL_YEAR_MONTH || target->id >= DORA_TYPE_INTERVAL_YEAR_MONTH) {
        return source->id == target->id ? DORA_OK : ENOTSUP;
    }
    if (type_is_temporal(source->id) || type_is_temporal(target->id)) {
        if (source->id == DORA_TYPE_UTF8) {
            return DORA_OK;
        }
        if (type_is_bytes(source->id)) {
            return ENOTSUP;
        }
        if (type_is_temporal(source->id) && type_is_temporal(target->id)) {
            const bool dates =
                (source->id == DORA_TYPE_DATE32 || source->id == DORA_TYPE_DATE64 ||
                 source->id == DORA_TYPE_TIMESTAMP) &&
                (target->id == DORA_TYPE_DATE32 || target->id == DORA_TYPE_DATE64 || target->id == DORA_TYPE_TIMESTAMP);
            if (source->id != target->id && !dates) {
                return EINVAL;
            }
            if (source->id == DORA_TYPE_TIMESTAMP && target->id == DORA_TYPE_TIMESTAMP) {
                const char* first = source->parameters.timestamp.timezone;
                const char* second = target->parameters.timestamp.timezone;
                if (strcmp(first != nullptr ? first : "", second != nullptr ? second : "") != 0) {
                    return EINVAL;
                }
            } else if (type_has_timezone(source) || type_has_timezone(target)) {
                return ENOTSUP;
            }
            return DORA_OK;
        }
        const enum dora_type other = type_is_temporal(source->id) ? target->id : source->id;
        return other >= DORA_TYPE_INT8 && other <= DORA_TYPE_UINT128 ? DORA_OK : ENOTSUP;
    }
    return DORA_OK;
}

static bool cast_can_fail(const struct dora_logical_type* source, const struct dora_logical_type* target) {
    if (source->extension_name.byte_count != 0 || target->extension_name.byte_count != 0) {
        return true;
    }
    const bool ordinary = source->id >= DORA_TYPE_BOOLEAN && source->id <= DORA_TYPE_FLOAT64;
    if (ordinary && (source->id == target->id || target->id == DORA_TYPE_UTF8)) {
        return false;
    }
    if (source->id == DORA_TYPE_BOOLEAN && target->id >= DORA_TYPE_INT8 && target->id <= DORA_TYPE_FLOAT64) {
        return false;
    }
    return !(source->id >= DORA_TYPE_FLOAT16 && source->id <= DORA_TYPE_FLOAT64 && target->id >= source->id &&
             target->id <= DORA_TYPE_FLOAT64);
}

static int cast_hook_result(struct dora_pool pool,
                            uint64_t length,
                            const struct dora_logical_type* target,
                            const struct dora_vector* result) {
    if (!dora_vector_is_valid(result) || !result->handle.is_owner || result->handle.pool_index != pool.index ||
        result->rows.length != length) {
        return EINVAL;
    }
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(result, &type);
    uint32_t child_count = 0;
    if (status == DORA_OK && type.id == DORA_TYPE_UNION) {
        status = dora_vector_get_child_count(result, &child_count);
    }
    if (status == DORA_OK && target->extension_name.byte_count == 0 && type.extension_name.byte_count != 0) {
        status = EINVAL;
    }
    type.extension_name = (struct dora_bytes){};
    type.extension_metadata = (struct dora_bytes){};
    struct dora_logical_type plain = *target;
    plain.extension_name = (struct dora_bytes){};
    plain.extension_metadata = (struct dora_bytes){};
    if (status == DORA_OK && !dora_type_is_equal(&type, &plain, child_count)) {
        status = EINVAL;
    }
    return status;
}

int dora_compute_cast_in_pool(struct dora_pool pool,
                              const struct dora_vector* source,
                              const struct dora_logical_type* target,
                              enum dora_rounding_mode rounding_mode,
                              enum dora_failure_policy policy,
                              struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result) || target == nullptr || rounding_mode > DORA_ROUNDING_MODE_HALF_AWAY ||
        policy > DORA_FAILURE_POLICY_NULL || target->id > DORA_TYPE_UNION ||
        (target->extension_name.byte_count != 0 && target->extension_name.data == nullptr) ||
        (target->extension_metadata.byte_count != 0 && target->extension_metadata.data == nullptr) ||
        (target->extension_name.byte_count == 0 && target->extension_metadata.byte_count != 0)) {
        return EINVAL;
    }
    if (target->id >= DORA_TYPE_FLOAT16 && target->id <= DORA_TYPE_FLOAT64 &&
        rounding_mode != DORA_ROUNDING_MODE_EXACT && rounding_mode != DORA_ROUNDING_MODE_HALF_EVEN) {
        return EINVAL;
    }
    struct vector_bound_node leaf;
    struct vector_bound_input input = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, source, &leaf, &input);
    }
    if (status != DORA_OK) {
        dora_scratch_scope_end(&scope);
        return status;
    }
    struct dora_logical_type plain = *target;
    plain.extension_name = (struct dora_bytes){};
    plain.extension_metadata = (struct dora_bytes){};
    struct dora_logical_type source_plain = input.nodes[0].field.type;
    source_plain.extension_name = (struct dora_bytes){};
    source_plain.extension_metadata = (struct dora_bytes){};
    const bool same = dora_type_is_equal(&source_plain, &plain, input.nodes[0].field.child_count);
    const struct vector_bound_node* const root = &input.nodes[0];
    const bool annotation_equal =
        root->field.type.extension_name.byte_count == target->extension_name.byte_count &&
        root->field.type.extension_metadata.byte_count == target->extension_metadata.byte_count &&
        (target->extension_name.byte_count == 0 ||
         memcmp(root->field.type.extension_name.data, target->extension_name.data, target->extension_name.byte_count) ==
             0) &&
        (target->extension_metadata.byte_count == 0 || memcmp(root->field.type.extension_metadata.data,
                                                              target->extension_metadata.data,
                                                              target->extension_metadata.byte_count) == 0);
    struct dora_vector result = {};
    bool converted = false;
    const bool preserves_utf8 = same && plain.id == DORA_TYPE_UTF8 && root->field.type.extension_name.byte_count == 0 &&
                                target->extension_name.byte_count == 0;
    const bool binary_utf8 = plain.id == DORA_TYPE_UTF8 && root->field.type.extension_name.byte_count == 0 &&
                             (source_plain.id == DORA_TYPE_BINARY || source_plain.id == DORA_TYPE_FIXED_BINARY);
    if (root->field.type.extension_name.byte_count != 0 && root->extension == nullptr) {
        status = ENOTSUP;
    }
    if (status == DORA_OK && plain.id < DORA_TYPE_LIST) {
        status = dora_type_check(&plain);
    }
    const bool json_storage = same && target->extension_name.byte_count == 0 &&
                              cast_name_matches(root->field.type.extension_name, "arrow.json") &&
                              plain.id == DORA_TYPE_UTF8;
    if (status == DORA_OK && root->field.type.extension_name.byte_count != 0 && (!same || !annotation_equal) &&
        !json_storage) {
        if (root->extension->cast == nullptr) {
            status = ENOTSUP;
        } else {
            // Extension callbacks keep source and result pools busy.
            struct vector_image* source_image = nullptr;
            const struct dora_pool source_pool = dora_vector_resolve(source, &source_image) == DORA_OK
                                                     ? source_image->allocation.pool
                                                     : (struct dora_pool){};
            dora_pool_callback_enter(pool);
            dora_pool_callback_enter(source_pool);
            status =
                root->extension
                    ->cast(root->extension, pool, root->parameters, source, target, rounding_mode, policy, &result);
            dora_pool_callback_leave(source_pool);
            dora_pool_callback_leave(pool);
        }
        if (status == DORA_OK) {
            status = cast_hook_result(pool, input.length, target, &result);
        }
    } else if (status == DORA_OK && preserves_utf8) {
        struct vector_image* image = nullptr;
        status = dora_vector_image_share(pool, source, &root->field, &image, nullptr);
        if (status == DORA_OK) {
            status = dora_vector_publish(image, &result);
        }
        if (status != DORA_OK) {
            dora_vector_discard(image);
        }
    } else if (status == DORA_OK && binary_utf8) {
        status = cast_same_storage(&scope, source, &input, 0, &plain, &result);
    } else if (status == DORA_OK && same) {
        status = cast_same_storage(&scope, source, &input, 0, nullptr, &result);
    } else if (status == DORA_OK) {
        converted = true;
        status = cast_admit(&source_plain, &plain, rounding_mode);
        bool peeled = false;
        if (status == DORA_OK) {
            struct dora_vector values = {};
            struct dora_vector transformed = {};
            status = dora_compute_peel_values(source,
                                              &input,
                                              policy == DORA_FAILURE_POLICY_ERROR &&
                                                  cast_can_fail(&root->field.type, target),
                                              &values,
                                              &peeled);
            if (status == DORA_OK && peeled) {
                status = dora_compute_cast_in_pool(pool, &values, &plain, rounding_mode, policy, &transformed);
            }
            if (status == DORA_OK && peeled) {
                status = dora_compute_peel_publish(pool, source, &input, &transformed, &result);
            }
            dora_vector_release(&transformed);
        }
        if (status == DORA_OK && !peeled) {
            const struct cast_call call = {.source = source, .target = &plain, .mode = rounding_mode, .policy = policy};
            status = type_is_bytes(plain.id) ? cast_text(pool, &input, &call, &result)
                                             : cast_fixed(pool, &input, &call, &result);
        }
    }
    if (status == DORA_OK && target->extension_name.byte_count != 0) {
        struct dora_vector extension = {};
        status = dora_vector_init_retain_extension_error(&extension,
                                                         pool,
                                                         target->extension_name,
                                                         target->extension_metadata,
                                                         &result,
                                                         nullptr,
                                                         nullptr);
        if (status == DORA_OK) {
            dora_vector_release(&result);
            status = dora_vector_move(&extension, &result);
        }
        dora_vector_release(&extension);
    }
    if (status == DORA_OK && !preserves_utf8 && !(converted && target->extension_name.byte_count == 0)) {
        struct dora_vector validated = {};
        status = dora_vector_init_retain_validated_in_pool(&validated, pool, &result, policy, nullptr);
        dora_vector_release(&result);
        result = validated;
    }
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&result, out_result, status);
}

int dora_compute_cast(const struct dora_vector* source,
                      const struct dora_logical_type* target,
                      enum dora_rounding_mode rounding_mode,
                      enum dora_failure_policy policy,
                      struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_compute_cast_in_pool(pool, source, target, rounding_mode, policy, out_result);
}
