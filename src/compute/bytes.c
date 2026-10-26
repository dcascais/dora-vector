// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/bytes_source.h"
#include "compute/compute_internal.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "utf8/batch.h"
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
// Length
//
//------------------------------------------------------------------------------

static int bytes_length(const struct dora_vector* source, bool unicode, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || source == out_result) {
        return EINVAL;
    }
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[1];
    if (status != DORA_OK) {
        return status;
    }
    struct dora_vector published = {};
    struct compute_bytes_source* const input = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }
    status = dora_compute_bytes_bind(&scope, 0, source, 1, false, VECTOR_OPERAND_REQUIRE_LENGTH, false, input);
    if (status == DORA_OK && unicode && input->input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    const uint64_t stored = constant ? 1 : input->input.length;
    struct compute_output output;
    output.buffers[0] = (struct dora_allocation){};
    output.buffers[1] = (struct dora_allocation){};
    output.type.id = DORA_TYPE_NULL;
    const struct dora_logical_type type = {.id = DORA_TYPE_UINT64};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool, &type, input->input.length, constant, false, &output);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_preserve_validity(&output, source);
    }
    size_t counts[KERNEL_CHUNK_ROW_COUNT];
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct dora_utf8_batch batch;
        status = dora_vector_operand_measure(&input->operand, rows, unicode, counts, &batch.eligible_rows);
        batch.row_count = rows.length;
        uint64_t* const values = (uint64_t*)output.buffers[1].address + first;
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            values[lane] = 0;
            if (compute_bytes_mask_is_valid(batch.eligible_rows, lane)) {
                const size_t count = counts[lane];
                if (ckd_add(&values[lane], count, 0)) {
                    status = EOVERFLOW;
                }
            }
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit_utf8_mask(&output, first, batch.row_count, batch.eligible_rows);
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

int dora_compute_bytes_count(const struct dora_vector* source, struct dora_vector* out_result) {
    return bytes_length(source, false, out_result);
}

int dora_compute_utf8_scalar_count(const struct dora_vector* source, struct dora_vector* out_result) {
    return bytes_length(source, true, out_result);
}

//------------------------------------------------------------------------------
//
// Substring
//
//------------------------------------------------------------------------------

static struct dora_bytes bytes_cut(struct dora_bytes bytes, struct compute_bytes_slice slice) {
    const uint64_t magnitude = slice.offset < 0 ? (uint64_t)(-(slice.offset + 1)) + 1 : (uint64_t)slice.offset;
    const size_t position = slice.offset < 0
                                ? (magnitude >= bytes.byte_count ? 0 : bytes.byte_count - (size_t)magnitude)
                                : (magnitude < bytes.byte_count ? (size_t)magnitude : bytes.byte_count);
    const size_t length = slice.count < bytes.byte_count - position ? (size_t)slice.count : bytes.byte_count - position;
    return (struct dora_bytes){.data = length == 0 ? nullptr : bytes.data + position, .byte_count = length};
}

static int substring_rows(struct scratch_scope* scope,
                          struct compute_bytes_source* source,
                          struct compute_bytes_slice slice,
                          struct compute_bytes_result* in_out_result) {
    const uint64_t stored = in_out_result->stored;
    struct dora_allocation* const buffers = in_out_result->buffers;
    int status = DORA_OK;
    size_t total = 0;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        status = dora_compute_bytes_prepare(source, rows);
        if (status == DORA_OK) {
            in_out_result->nulls += dora_compute_bytes_store_validity(buffers[0].address, first, &source->packet);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            if (compute_bytes_piece_is_valid(source, lane) &&
                (ckd_add(&total, total, bytes_cut(compute_bytes_piece(source, lane), slice).byte_count) ||
                 total > INT64_MAX)) {
                status = EOVERFLOW;
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(scope->pool, total, &buffers[2]);
    }
    dora_vector_operand_reset(&source->operand);
    int64_t* const offsets = buffers[1].address;
    uint8_t* const data = buffers[2].address;
    size_t offset = 0;
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        status = dora_compute_bytes_prepare(source, rows);
        if (status == DORA_OK) {
            status = dora_compute_bytes_check_validity(in_out_result, first, &source->packet);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            if (compute_bytes_piece_is_valid(source, lane)) {
                const struct dora_bytes value = bytes_cut(compute_bytes_piece(source, lane), slice);
                if (value.byte_count > total - offset) {
                    status = EINVAL;
                    break;
                }
                if (value.byte_count != 0) {
                    memcpy(data + offset, value.data, value.byte_count);
                    offset += value.byte_count;
                }
            }
            offsets[first + lane + 1] = offset;
        }
    }
    in_out_result->data_bytes = total;

    return status;
}

int dora_compute_bytes_slice_in_pool(struct dora_pool pool,
                                     const struct dora_vector* source,
                                     int64_t byte_offset,
                                     uint64_t byte_count,
                                     bool is_text,
                                     struct dora_vector* out_result) {
    if (!dora_vector_is_empty(out_result) || source == nullptr) {
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
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    const struct vector_operand* const operand = &input->operand;
    const bool all_valid = compute_bytes_operand_is_all_valid(operand);
    if (status == DORA_OK) {
        status = dora_compute_bytes_preserve_validity(source, constant, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, !all_valid, &result);
    }
    if (status == DORA_OK) {
        status = substring_rows(&scope,
                                input,
                                (struct compute_bytes_slice){.offset = byte_offset, .count = byte_count},
                                &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(
            &scope,
            &result,
            &(const struct dora_logical_type){.id = is_text ? DORA_TYPE_UTF8 : DORA_TYPE_BINARY},
            DORA_LAYOUT_LARGE_BINARY,
            is_text ? dora_compute_bytes_content(input, 1) : VECTOR_CONTENT_VALIDATED,
            constant,
            input->input.length,
            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_bytes_substring(const struct dora_vector* source,
                                 int64_t byte_offset,
                                 uint64_t byte_count,
                                 struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int status = dora_vector_get_pool(source, &pool);
    return status == DORA_OK
               ? dora_compute_bytes_slice_in_pool(pool, source, byte_offset, byte_count, false, out_result)
               : status;
}
