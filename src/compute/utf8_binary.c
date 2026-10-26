// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"
#include "types/array_count.h"
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
// Binary To Utf8
//
//------------------------------------------------------------------------------

static int utf8_from_binary_retain(struct scratch_scope* scope,
                                   const struct dora_vector* source,
                                   struct compute_bytes_source* input,
                                   bool constant,
                                   bool direct,
                                   struct compute_bytes_result* result,
                                   struct dora_vector* out_result) {
    const struct dora_vector_view* const view = &input->input.nodes[0].view;
    const uint64_t stored = result->stored;
    const bool fixed = direct && view->layout == DORA_LAYOUT_FIXED;
    const bool gather = !direct;
    if (fixed && view->byte_width != 0 && stored > (uint64_t)INT64_MAX / view->byte_width) {
        return EOVERFLOW;
    }
    result->is_all_ascii = true;
    int status = DORA_OK;
    if (gather) {
        size_t bytes = 0;
        status = ckd_mul(&bytes, stored, sizeof(struct dora_binary_view_entry))
                     ? EOVERFLOW
                     : dora_pool_allocate(scope->pool, bytes, &result->buffers[1]);
        if (status == DORA_OK && !compute_bytes_operand_is_all_valid(&input->operand)) {
            status = ckd_mul(&bytes, bitmap_words_for_bits(stored), sizeof(uint64_t))
                         ? EOVERFLOW
                         : dora_pool_allocate(scope->pool, bytes, &result->buffers[0]);
        }
    } else if (fixed) {
        status = dora_compute_bytes_offsets(scope->pool, result);
    }
    const uint8_t* const records = dora_compute_bytes_view_records(input);
    uint32_t backing_count = 0;
    const struct dora_view_buffer* backing = dora_compute_bytes_backing(input, &backing_count);
    const uint8_t* const buffer = backing_count == 0 ? nullptr : backing[0].data;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        bool ascii = false;
        status = dora_compute_bytes_prepare_validate(input, rows, DORA_UTF8_CHECKED, &ascii);
        result->is_all_ascii &= ascii;
        if (status == DORA_OK && gather) {
            result->nulls += dora_compute_bytes_store_validity(result->buffers[0].address, first, &input->packet);
            for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
                const bool valid = compute_bytes_piece_is_valid(input, lane);
                const uint64_t row = first + lane;
                struct dora_binary_view_entry* const entry =
                    &((struct dora_binary_view_entry*)result->buffers[1].address)[row];
                if (valid && records != nullptr) {
                    memcpy(entry, dora_compute_bytes_view_record(input, records, first, lane), sizeof *entry);
                } else {
                    const struct dora_bytes piece = valid ? compute_bytes_piece(input, lane) : (struct dora_bytes){};
                    dora_compute_bytes_view_entry(piece, buffer, 0, entry);
                }
            }
        }
    }
    const enum vector_content_state content =
        result->is_all_ascii ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_VALIDATED;
    if (status == DORA_OK && gather) {
        result->buffer_source = source;
        result->backing = backing;
        result->backing_count = backing_count;
        status = dora_compute_bytes_publish(scope,
                                            result,
                                            &dora_compute_utf8_type,
                                            DORA_LAYOUT_BINARY_VIEW,
                                            content,
                                            constant,
                                            input->input.length,
                                            out_result);
    } else if (status == DORA_OK && fixed) {
        int64_t* const offsets = result->buffers[1].address;
        for (uint64_t row = 0; row <= stored; ++row) {
            offsets[row] = (int64_t)(row * view->byte_width);
        }
        const struct dora_view_buffer payload = {
            .data = view->values.data == nullptr ? nullptr
                                                 : (const uint8_t*)view->values.data + view->offset * view->byte_width,
            .byte_count = view->values.byte_count - view->offset * view->byte_width
        };
        result->validity =
            (struct dora_utf8_mask){{view->validity.data, view->validity.byte_count}, view->validity_bit_offset};
        result->nulls = (uint64_t)view->null_count;
        result->buffer_source = source;
        result->backing = &payload;
        result->backing_count = 1;
        status = dora_compute_bytes_publish(scope,
                                            result,
                                            &dora_compute_utf8_type,
                                            DORA_LAYOUT_LARGE_BINARY,
                                            content,
                                            constant,
                                            input->input.length,
                                            out_result);
    } else if (status == DORA_OK) {
        const struct dora_schema_field field = {.type = dora_compute_utf8_type, .layout = view->layout};
        struct dora_schema schema = {};
        status = dora_schema_node_claim(scope->pool, &field, nullptr, &schema);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_schema_error(out_result,
                                                          scope->pool,
                                                          source,
                                                          &schema,
                                                          VECTOR_CONTENT_PROMISED,
                                                          nullptr);
        }
        struct vector_image* image = nullptr;
        if (status == DORA_OK) {
            status = dora_vector_resolve(out_result, &image);
        }
        if (status == DORA_OK) {
            image->node.content_state = content;
        }
        dora_schema_release(&schema);
    }
    return status;
}

static int
utf8_from_binary_rows(struct dora_pool pool, struct compute_bytes_source* source, struct compute_bytes_result* result) {
    size_t lengths[KERNEL_CHUNK_ROW_COUNT];
    int64_t* const offsets = result->buffers[1].address;
    int status = DORA_OK;
    result->is_all_ascii = true;
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        struct dora_utf8_batch batch;
        status = dora_compute_bytes_binary_batch(source, rows, &batch);
        size_t total = 0;
        if (status == DORA_OK) {
            batch.proof = DORA_UTF8_VALIDATED;
            status = dora_utf8_measure_batch(&batch, lengths, &total, nullptr);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const bool valid = compute_bytes_mask_is_valid(batch.eligible_rows, lane);
            if (valid &&
                (ckd_add(&result->data_bytes, result->data_bytes, lengths[lane]) || result->data_bytes > INT64_MAX)) {
                status = EOVERFLOW;
            } else {
                offsets[first + lane + 1] = result->data_bytes;
                if (result->buffers[0].address != nullptr) {
                    bitmap_assign(result->buffers[0].address, first + lane, valid);
                }
                result->nulls += (uint64_t)!valid;
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, result->data_bytes, &result->buffers[2]);
    }
    dora_vector_operand_reset(&source->operand);
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        struct dora_utf8_batch batch;
        status = dora_compute_bytes_binary_batch(source, rows, &batch);
        if (status == DORA_OK) {
            const struct dora_utf8_batch_output output = {
                .layout = DORA_UTF8_PACKED_I64,
                .packed.bytes = {result->buffers[2].address, result->buffers[2].capacity_bytes},
                .packed.offset_bytes = {(uint8_t*)(offsets + first), ((size_t)rows.length + 1) * sizeof *offsets},
                .packed.byte_offset = (size_t)offsets[first]
            };
            bool ascii = false;
            batch.proof = DORA_UTF8_CHECKED;
            status = dora_utf8_copy_batch_sized(&batch,
                                                result->data_bytes - output.packed.byte_offset,
                                                &output,
                                                nullptr,
                                                &ascii);
            result->is_all_ascii &= ascii;
        }
    }
    return status;
}

int dora_compute_utf8_from_binary(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || source == out_result) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[1];
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const input = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }
    status = dora_vector_bind_input(&scope, source, &input->leaf, &input->input);
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&input->input, 0, true);
    }
    if (status == DORA_OK && input->input.nodes[0].field.type.id != DORA_TYPE_BINARY &&
        input->input.nodes[0].field.type.id != DORA_TYPE_FIXED_BINARY) {
        status = ENOTSUP;
    }
    const struct dora_vector_view* const view = status == DORA_OK ? &input->input.nodes[0].view : nullptr;
    const bool direct =
        status == DORA_OK && view->selection.indices == nullptr && !vector_view_has_encoding_child(view);
    if (status == DORA_OK) {
        const uint8_t requirements = view->type == DORA_TYPE_FIXED_BINARY ? VECTOR_OPERAND_REQUIRE_PACKET
                                     : direct                             ? VECTOR_OPERAND_REQUIRE_TEXT
                                                                          : VECTOR_OPERAND_REQUIRE_PACKET_TEXT;
        status = dora_vector_operand_bind(&scope, 0, &input->input, requirements, &input->operand);
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    if (status == DORA_OK && dora_compute_bytes_has_owned_backing(input, !direct)) {
        status = utf8_from_binary_retain(&scope, source, input, constant, direct, &result, &published);
    } else {
        if (status == DORA_OK) {
            status = dora_compute_bytes_preserve_validity(source, constant, &result);
        }
        if (status == DORA_OK) {
            status =
                dora_compute_bytes_result_init(pool, !compute_bytes_operand_is_all_valid(&input->operand), &result);
        }
        if (status == DORA_OK) {
            status = utf8_from_binary_rows(pool, input, &result);
        }
        if (status == DORA_OK) {
            status = dora_compute_bytes_publish(&scope,
                                                &result,
                                                &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                                DORA_LAYOUT_LARGE_BINARY,
                                                result.is_all_ascii ? VECTOR_CONTENT_VALIDATED_ASCII
                                                                    : VECTOR_CONTENT_VALIDATED,
                                                constant,
                                                input->input.length,
                                                &published);
        }
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}
