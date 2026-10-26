// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/bytes_source.h"

#include "bitmap/bitmap_internal.h"
#include "compute/compute_internal.h"
#include "compute/output.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/input.h"
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
// Sources
//
//------------------------------------------------------------------------------

struct bytes_text {
    int64_t ends[KERNEL_CHUNK_ROW_COUNT];
    struct dora_bytes pieces[KERNEL_CHUNK_ROW_COUNT];
    uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
    uint8_t data[];
};

const struct dora_logical_type dora_compute_utf8_type = {.id = DORA_TYPE_UTF8};

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

void dora_compute_bytes_source_init(struct compute_bytes_source* source) {
    source->input = (struct vector_bound_input){};
    source->converted = (struct dora_vector){};
    source->operand.kind = VECTOR_OPERAND_DIRECT;
    source->operand.is_all_valid = false;
    source->text.fill = nullptr;
    source->text_storage = nullptr;
    source->utf8_proof = DORA_UTF8_CHECKED;
}

static int bytes_bind_text(struct scratch_scope* scope, size_t position, struct compute_bytes_source* source) {
    const struct vector_bound_node* const node = &source->input.nodes[source->operand.terminal_node];
    if (source->operand.kind == VECTOR_OPERAND_CONSTANT_NULL || type_is_bytes(node->field.type.id)) {
        return DORA_OK;
    }
    struct kernel_bind_request request = {
        .input_type = &node->field.type,
        .result_type = &dora_compute_utf8_type,
        .policy = DORA_FAILURE_POLICY_ERROR,
        .rounding = DORA_ROUNDING_MODE_EXACT,
    };
    int status = dora_kernel_cast_bind(&request, &source->text);
    size_t bytes = 0;
    if (status == DORA_OK &&
        (ckd_mul(&bytes, (size_t)KERNEL_CHUNK_ROW_COUNT, source->text.cast.maximum_text_byte_count) ||
         ckd_add(&bytes, bytes, sizeof(struct bytes_text)))) {
        status = EOVERFLOW;
    }
    return status == DORA_OK ? dora_scratch_slot_require(scope,
                                                         vector_operand_slot(position, VECTOR_OPERAND_SLOT_TEXT),
                                                         &source->text_storage,
                                                         bytes,
                                                         64)
                             : status;
}

int dora_compute_bytes_bind(struct scratch_scope* scope,
                            size_t position,
                            const struct dora_vector* sources,
                            uint32_t source_count,
                            bool should_concatenate,
                            uint8_t requirements,
                            bool should_bind_singleton,
                            struct compute_bytes_source* out_inputs) {
    const struct dora_pool pool = scope->pool;
    int status = DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < source_count; ++index) {
        struct compute_bytes_source* const source = &out_inputs[index];
        status = dora_vector_bind_input(scope, &sources[index], &source->leaf, &source->input);
        if (status == DORA_OK && should_concatenate &&
            source->input.nodes[0].field.type.extension_name.byte_count != 0) {
            status = dora_compute_cast_in_pool(pool,
                                               &sources[index],
                                               &dora_compute_utf8_type,
                                               DORA_ROUNDING_MODE_EXACT,
                                               DORA_FAILURE_POLICY_ERROR,
                                               &source->converted);
            if (status == DORA_OK) {
                status = dora_vector_bind_input(scope, &source->converted, &source->leaf, &source->input);
            }
        }
        if (status == DORA_OK) {
            status = dora_vector_input_admit(&source->input, 0, true);
        }
        const enum dora_type type = status == DORA_OK ? source->input.nodes[0].field.type.id : DORA_TYPE_NULL;
        if (status == DORA_OK && source->input.length != out_inputs[0].input.length) {
            status = EINVAL;
        }
        if (status == DORA_OK &&
            (should_concatenate ? type == DORA_TYPE_BINARY || type == DORA_TYPE_FIXED_BINARY || type >= DORA_TYPE_LIST
                                : !type_is_bytes(type))) {
            status = ENOTSUP;
        }
        if (status == DORA_OK) {
            status =
                should_bind_singleton && source->input.length == 1
                    ? dora_vector_operand_bind_singleton(&sources[index], &source->input, &source->operand)
                    : dora_vector_operand_bind(scope, position + index, &source->input, requirements, &source->operand);
        }
        if (status == DORA_OK) {
            source->utf8_proof = vector_utf8_proof(source->input.nodes[source->operand.terminal_node].image);
        }
        if (status == DORA_OK && should_concatenate) {
            status = bytes_bind_text(scope, position + index, source);
        }
    }
    return status;
}

int dora_compute_bytes_prepare(struct compute_bytes_source* source, struct dora_range rows) {
    int status = dora_vector_operand_prepare(&source->operand, rows, nullptr, &source->packet);
    if (status != DORA_OK) {
        return status;
    }
    source->is_constant = source->packet.shape == KERNEL_SHAPE_CONSTANT;
    source->validity = source->packet.validity;
    if (source->text.fill == nullptr) {
        source->pieces = source->packet.values == nullptr
                             ? nullptr
                             : (const struct dora_bytes*)source->packet.values + source->packet.value_offset;
        return DORA_OK;
    }
    struct kernel_packet packet = source->packet;
    struct bytes_text* const text = source->text_storage;
    packet.count = source->is_constant ? 1 : packet.count;
    struct kernel_text_output output = {.end_offsets = text->ends, .data = text->data};
    struct kernel_result result = {.values = &output, .validity = text->validity, .row_capacity = packet.count};
    status = source->text.fill(&packet, &source->text, &result);
    int64_t start = 0;
    for (uint32_t lane = 0; status == DORA_OK && lane < packet.count; ++lane) {
        text->pieces[lane] = (struct dora_bytes){.data = text->data + start, .byte_count = text->ends[lane] - start};
        start = text->ends[lane];
    }
    source->pieces = text->pieces;
    source->validity = text->validity;
    return status;
}

void dora_compute_bytes_release(struct compute_bytes_source* inputs, uint32_t source_count) {
    for (uint32_t index = source_count; index != 0; --index) {
        struct compute_bytes_source* const source = &inputs[index - 1];
        dora_vector_release(&source->converted);
    }
}

enum vector_content_state dora_compute_bytes_content(const struct compute_bytes_source* sources, uint32_t count) {
    enum vector_content_state result = VECTOR_CONTENT_VALIDATED_ASCII;
    for (uint32_t index = 0; index < count; ++index) {
        const enum dora_utf8_proof proof =
            sources[index].text.fill != nullptr ? DORA_UTF8_VALIDATED_ASCII : sources[index].utf8_proof;
        const enum vector_content_state state = compute_proof_content(proof);
        if (state < result) {
            result = state;
        }
    }
    return result;
}

int dora_compute_bytes_capacity(struct compute_bytes_source* source, struct compute_bytes_result* result) {
    size_t counts[KERNEL_CHUNK_ROW_COUNT];
    result->data_bytes = 0;
    result->nulls = 0;
    dora_vector_operand_reset(&source->operand);
    for (uint64_t first = 0; first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        struct dora_utf8_mask eligible = {};
        const int status = dora_vector_operand_measure(&source->operand, rows, false, counts, &eligible);
        if (status != DORA_OK) {
            return status;
        }
        for (uint32_t lane = 0; lane < rows.length; ++lane) {
            const bool valid = compute_bytes_mask_is_valid(eligible, lane);
            if (valid &&
                (ckd_add(&result->data_bytes, result->data_bytes, counts[lane]) || result->data_bytes > INT64_MAX)) {
                return EOVERFLOW;
            }
            if (result->buffers[0].address != nullptr) {
                bitmap_assign(result->buffers[0].address, first + lane, valid);
            }
            result->nulls += (uint64_t)!valid;
        }
    }
    return DORA_OK;
}

int dora_compute_bytes_binary_batch(struct compute_bytes_source* source,
                                    struct dora_range rows,
                                    struct dora_utf8_batch* out_batch) {
    if (source->operand.kind != VECTOR_OPERAND_BYTES ||
        source->input.nodes[source->operand.terminal_node].view.layout != DORA_LAYOUT_FIXED) {
        const int status = dora_vector_operand_prepare_utf8(&source->operand, rows, nullptr, out_batch);
        if (status == DORA_OK) {
            out_batch->proof = DORA_UTF8_CHECKED;
        }
        return status;
    }
    const int status = dora_compute_bytes_prepare(source, rows);
    if (status == DORA_OK) {
        dora_vector_operand_utf8_batch(&source->packet, DORA_UTF8_CHECKED, out_batch);
    }
    return status;
}

int dora_compute_bytes_prepare_validate(struct compute_bytes_source* source,
                                        struct dora_range rows,
                                        enum dora_utf8_proof proof,
                                        bool* out_is_ascii) {
    struct dora_utf8_batch batch = {};
    int status;
    if (source->operand.payload == nullptr && source->operand.kind == VECTOR_OPERAND_BYTES) {
        status = dora_compute_bytes_binary_batch(source, rows, &batch);
    } else {
        status = dora_compute_bytes_prepare(source, rows);
        if (status != DORA_OK || (proof != DORA_UTF8_CHECKED && out_is_ascii == nullptr)) {
            return status;
        }
        dora_vector_operand_utf8_batch(&source->packet, proof, &batch);
    }
    if (status == DORA_OK && (proof == DORA_UTF8_CHECKED || out_is_ascii != nullptr)) {
        batch.proof = proof;
        status = dora_utf8_validate_batch(&batch, out_is_ascii, nullptr);
    }
    return status;
}

const struct dora_view_buffer* dora_compute_bytes_backing(const struct compute_bytes_source* source,
                                                          uint32_t* out_count) {
    const struct dora_vector_view* const view = &source->input.nodes[source->operand.terminal_node].view;
    *out_count = view->layout == DORA_LAYOUT_FIXED ? 1 : view->data_buffer_count;
    return view->layout == DORA_LAYOUT_FIXED ? &view->values : view->data_buffers;
}

bool dora_compute_bytes_has_owned_backing(const struct compute_bytes_source* source, bool should_use_view_descriptors) {
    const struct vector_bound_node* const node = &source->input.nodes[source->operand.terminal_node];
    const struct dora_vector_view* const view = &node->view;
    uint32_t count = 0;
    const struct dora_view_buffer* const buffers = dora_compute_bytes_backing(source, &count);
    if (node->image == nullptr || view->placement.access == DORA_ACCESS_DEVICE || count > INT32_MAX) {
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if ((should_use_view_descriptors && buffers[index].byte_count > INT32_MAX) ||
            (buffers[index].byte_count != 0 && !dora_vector_buffer_has_owner(node->image, buffers[index]))) {
            return false;
        }
    }
    const struct dora_view_buffer geometry = view->layout == DORA_LAYOUT_BINARY_VIEW ? view->values : view->offsets;
    return should_use_view_descriptors ||
           ((geometry.byte_count == 0 || dora_vector_buffer_has_owner(node->image, geometry)) &&
            (view->validity.data == nullptr || dora_vector_buffer_has_owner(node->image, view->validity)));
}

int dora_compute_bytes_view_windows(struct scratch_scope* scope,
                                    const struct dora_view_buffer* buffer,
                                    const struct dora_view_buffer** out_backing,
                                    uint32_t* out_count) {
    const size_t stride = (size_t)INT32_MAX + 1;
    const size_t count = buffer->byte_count / stride + (buffer->byte_count % stride != 0);
    *out_backing = buffer;
    *out_count = count == 0 ? 0 : 1;
    if (count <= 1) {
        return DORA_OK;
    }
    size_t bytes = 0;
    struct dora_view_buffer* windows = nullptr;
    int status = count > INT32_MAX || ckd_mul(&bytes, count, sizeof *windows)
                     ? EOVERFLOW
                     : dora_scratch_allocate(scope, bytes, (void**)&windows);
    for (size_t index = 0; status == DORA_OK && index < count; ++index) {
        const size_t first = index * stride;
        const size_t remaining = buffer->byte_count - first;
        windows[index] = (struct dora_view_buffer){
            .data = (const uint8_t*)buffer->data + first,
            .byte_count = remaining < UINT32_MAX ? remaining : UINT32_MAX
        };
    }
    if (status == DORA_OK) {
        *out_backing = windows;
        *out_count = (uint32_t)count;
    }
    return status;
}

void dora_compute_bytes_view_entry(struct dora_bytes piece,
                                   const uint8_t* buffer,
                                   uint32_t buffer_index,
                                   struct dora_binary_view_entry* out_entry) {
    *out_entry = (struct dora_binary_view_entry){.byte_count = (int32_t)piece.byte_count};
    if (piece.byte_count <= sizeof out_entry->payload.inline_bytes) {
        if (piece.byte_count != 0) {
            memcpy(out_entry->payload.inline_bytes, piece.data, piece.byte_count);
        }
        return;
    }
    out_entry->payload.reference.buffer_index = (int32_t)buffer_index;
    out_entry->payload.reference.byte_offset = (int32_t)((uintptr_t)piece.data - (uintptr_t)buffer);
    memcpy(out_entry->payload.reference.prefix, piece.data, sizeof out_entry->payload.reference.prefix);
}

const uint8_t* dora_compute_bytes_view_records(const struct compute_bytes_source* source) {
    const struct dora_vector_view* const view = &source->input.nodes[source->operand.terminal_node].view;
    return view->layout == DORA_LAYOUT_BINARY_VIEW && view->values.data != nullptr
               ? (const uint8_t*)view->values.data +
                     (view->selection.indices == nullptr ? view->offset : 0) * sizeof(struct dora_binary_view_entry)
               : nullptr;
}

const uint8_t* dora_compute_bytes_view_record(const struct compute_bytes_source* source,
                                              const uint8_t* records,
                                              uint64_t first,
                                              uint32_t lane) {
    const uint64_t row =
        source->operand.rows64 != nullptr ? type_load_u64((const uint8_t*)source->operand.rows64, lane) : first + lane;
    return records + row * sizeof(struct dora_binary_view_entry);
}

int dora_compute_bytes_bind_literal(struct scratch_scope* scope,
                                    const struct dora_vector* source,
                                    const struct dora_vector* scalar,
                                    uint8_t source_requirements,
                                    struct compute_bytes_source inputs[2]) {
    const struct dora_vector* const vectors[] = {source, scalar};
    for (uint32_t index = 0; index < (scalar == nullptr ? 1U : 2U); ++index) {
        const int status = dora_compute_bytes_bind(scope,
                                                   index,
                                                   vectors[index],
                                                   1,
                                                   false,
                                                   index == 0 ? source_requirements : VECTOR_OPERAND_REQUIRE_PACKET,
                                                   index != 0,
                                                   &inputs[index]);
        if (status != DORA_OK) {
            return status;
        }
        if (inputs[index].input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
            return ENOTSUP;
        }
    }
    return scalar != nullptr && inputs[1].input.length != 1 ? EINVAL : DORA_OK;
}

int dora_compute_bytes_bind_i32(struct scratch_scope* scope,
                                size_t position,
                                const struct dora_vector* source,
                                uint8_t requirements,
                                struct compute_bytes_source* input) {
    int status = dora_vector_bind_input(scope, source, &input->leaf, &input->input);
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&input->input, 0, true);
    }
    if (status == DORA_OK && input->input.nodes[0].field.type.id != DORA_TYPE_INT32) {
        status = ENOTSUP;
    }
    if (status == DORA_OK) {
        status = (requirements & VECTOR_OPERAND_REQUIRE_ROW) != 0 && input->input.length == 1
                     ? dora_vector_operand_bind_singleton(source, &input->input, &input->operand)
                     : dora_vector_operand_bind(scope, position, &input->input, requirements, &input->operand);
    }
    return status;
}

int dora_compute_bytes_prepare_singleton(struct compute_bytes_source* source) {
    const int status =
        dora_vector_operand_bind_singleton(source->input.nodes[0].source, &source->input, &source->operand);
    return status == DORA_OK ? dora_compute_bytes_prepare(source, (struct dora_range){0, 1}) : status;
}

int dora_compute_bytes_prepare_current(struct compute_bytes_source* source, struct dora_range rows) {
    if (source->operand.kind == VECTOR_OPERAND_CONSTANT || source->operand.kind == VECTOR_OPERAND_CONSTANT_NULL) {
        // Allocator callbacks can invalidate borrowed constant spans.
        const int status = dora_vector_operand_bind_constant(source->operand.terminal_node, &source->operand);
        if (status != DORA_OK) {
            return status;
        }
    }
    dora_vector_operand_reset(&source->operand);
    return dora_compute_bytes_prepare(source, rows);
}
