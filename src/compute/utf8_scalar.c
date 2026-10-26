// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "types/load.h"
#include "utf8/input.h"
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
// Scalar Codecs
//
//------------------------------------------------------------------------------

static int utf8_chr_scalar(const struct kernel_packet* packet, uint32_t lane, uint32_t* out_scalar) {
    uint64_t slot = packet->shape == KERNEL_SHAPE_CONSTANT ? 0 : lane;
    if (packet->shape == KERNEL_SHAPE_GATHERED) {
        if (packet->rows == nullptr) {
            return EINVAL;
        }
        slot = packet->rows[lane];
    }
    size_t position;
    if (ckd_add(&slot, packet->value_offset, slot) || ckd_mul(&position, slot, sizeof(int32_t))) {
        return EOVERFLOW;
    }
    if (packet->values == nullptr) {
        return EINVAL;
    }
    const int32_t scalar = type_load_i32((const uint8_t*)packet->values + position, 0);
    if (scalar < 0) {
        return EILSEQ;
    }
    *out_scalar = (uint32_t)scalar;
    return DORA_OK;
}

static int
utf8_chr_rows(struct vector_operand* source, enum dora_utf8_emit_mode mode, struct compute_bytes_result* result) {
    dora_vector_operand_reset(source);
    int64_t* const offsets = result->buffers[1].address;
    size_t bytes = 0;
    result->nulls = 0;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        struct kernel_packet packet = {};
        status = dora_vector_operand_prepare(source, rows, nullptr, &packet);
        if (status == DORA_OK) {
            result->nulls += dora_compute_bytes_store_validity(result->buffers[0].address, first, &packet);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool valid =
                packet.validity == nullptr ||
                dora_bitmap_test((const uint8_t*)packet.validity, packet.shape == KERNEL_SHAPE_CONSTANT ? 0 : lane);
            uint32_t scalar = 0;
            size_t size = 0;
            if (valid) {
                status = utf8_chr_scalar(&packet, lane, &scalar);
                if (status == DORA_OK) {
                    status = mode == DORA_UTF8_MEASURE
                                 ? utf8_scalar_width(scalar, &size)
                                 : utf8_encode_scalar(scalar,
                                                      (struct dora_mutable_bytes){
                                                          (uint8_t*)result->buffers[2].address + bytes,
                                                          result->data_bytes - bytes
                                                      },
                                                      &size);
                }
            }
            if (status == DORA_OK && (ckd_add(&bytes, bytes, size) || bytes > INT64_MAX ||
                                      (mode == DORA_UTF8_EMIT && bytes > result->data_bytes))) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK) {
                offsets[row + 1] = (int64_t)bytes;
            }
        }
    }
    if (status == DORA_OK) {
        result->data_bytes = bytes;
    }
    return status;
}

int dora_compute_utf8_from_scalar(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source) {
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
    status = dora_compute_bytes_bind_i32(&scope, 0, source, VECTOR_OPERAND_REQUIRE_PACKET, input);
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    if (status == DORA_OK) {
        status = dora_compute_bytes_preserve_validity(source, constant, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, result.stored != 0, &result);
    }
    if (status == DORA_OK) {
        if (ckd_mul(&result.data_bytes, result.stored, 4) || result.data_bytes > INT64_MAX) {
            status = EOVERFLOW;
        }
    }
    if (status == DORA_OK) {
        bool drained = false;
        status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
        if (status == ENOMEM) {
            status = utf8_chr_rows(&input->operand, DORA_UTF8_MEASURE, &result);
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
            }
        }
    }
    if (status == DORA_OK) {
        status = utf8_chr_rows(&input->operand, DORA_UTF8_EMIT, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_compact(pool, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            result.data_bytes == result.stored - result.nulls
                                                ? VECTOR_CONTENT_VALIDATED_ASCII
                                                : VECTOR_CONTENT_VALIDATED,
                                            constant,
                                            input->input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_utf8_first_scalar(const struct dora_vector* source, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source) {
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
    status = dora_compute_bytes_bind(&scope, 0, source, 1, false, VECTOR_OPERAND_REQUIRE_PACKET, false, input);
    if (status == DORA_OK && input->input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    const uint64_t stored = constant ? 1 : input->input.length;
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_INT32},
                                          input->input.length,
                                          constant,
                                          false,
                                          &output);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_preserve_validity(&output, source);
        output.is_nullable = true;
    }
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        status = dora_compute_bytes_prepare(input, rows);
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        bitmap_fill_bytes((uint8_t*)result.validity,
                          (size_t)bitmap_words_for_bits(rows.length) * sizeof(uint64_t),
                          false);
        int32_t* const values = result.values;
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const bool valid = compute_bytes_piece_is_valid(input, lane);
            values[lane] = 0;
            result.null_count += (uint64_t)!valid;
            if (valid) {
                status = dora_utf8_first_scalar(
                    (struct dora_utf8_input){compute_bytes_piece(input, lane), input->utf8_proof},
                    &values[lane]);
                bitmap_assign(result.validity, lane, true);
            }
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &result);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}
