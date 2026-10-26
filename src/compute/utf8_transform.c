// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "compute/bytes_source.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "utf8/input.h"
#include "utf8/transform.h"
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
// Unicode Mapping
//
//------------------------------------------------------------------------------

enum utf8_transform_operation : uint8_t {
    UTF8_LOWER,
    UTF8_UPPER,
    UTF8_REVERSE
};

static int utf8_transform_capacity(struct compute_bytes_source* source,
                                   enum utf8_transform_operation operation,
                                   struct compute_bytes_result* result) {
    int status = dora_compute_bytes_capacity(source, result);
    if (status == DORA_OK && operation != UTF8_REVERSE) {
        status = utf8_case_capacity(result->data_bytes, &result->data_bytes);
    }
    return status == DORA_OK && result->data_bytes > INT64_MAX ? EOVERFLOW : status;
}

static int utf8_transform_fill(struct compute_bytes_source* restrict source,
                               enum utf8_transform_operation operation,
                               enum dora_utf8_emit_mode mode,
                               struct compute_bytes_result* restrict result) {
    dora_vector_operand_reset(&source->operand);
    int64_t* const offsets = result->buffers[1].address;
    size_t bytes = 0;
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        status = dora_compute_bytes_prepare_current(source, rows);
        if (status == DORA_OK) {
            status = dora_compute_bytes_check_validity(result, first, &source->packet);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool valid = compute_bytes_piece_is_valid(source, lane);
            size_t size = 0;
            if (valid) {
                const struct dora_mutable_bytes output = {
                    mode == DORA_UTF8_MEASURE || result->buffers[2].address == nullptr
                        ? nullptr
                        : (uint8_t*)result->buffers[2].address + bytes,
                    mode == DORA_UTF8_MEASURE ? 0 : result->data_bytes - bytes
                };
                const struct dora_utf8_input input = {compute_bytes_piece(source, lane), source->utf8_proof};
                status = operation == UTF8_REVERSE
                             ? utf8_reverse(input, mode, output, &size)
                             : dora_utf8_case_map(input,
                                                  operation == UTF8_UPPER ? DORA_UTF8_UPPER : DORA_UTF8_LOWER,
                                                  mode,
                                                  output,
                                                  &size);
            }
            if (status == ENOBUFS) {
                status = EINVAL;
            }
            if (status == DORA_OK && (ckd_add(&bytes, bytes, size) || bytes > result->data_bytes)) {
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

static int utf8_transform(const struct dora_vector* source,
                          enum utf8_transform_operation operation,
                          struct dora_vector* out_result) {
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
    status = dora_compute_bytes_bind(&scope, 0, source, 1, false, VECTOR_OPERAND_REQUIRE_PACKET_TEXT, false, input);
    if (status == DORA_OK && input->input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    if (status == DORA_OK) {
        status = dora_compute_bytes_preserve_validity(source, constant, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, !compute_bytes_operand_is_all_valid(&input->operand), &result);
    }
    if (status == DORA_OK) {
        status = utf8_transform_capacity(input, operation, &result);
    }
    if (status == DORA_OK) {
        bool drained = false;
        status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
        if (status == ENOMEM) {
            status = utf8_transform_fill(input, operation, DORA_UTF8_MEASURE, &result);
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
            }
        }
    }
    if (status == DORA_OK) {
        status = utf8_transform_fill(input, operation, DORA_UTF8_EMIT, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_compact(pool, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(
            &scope,
            &result,
            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
            DORA_LAYOUT_LARGE_BINARY,
            input->utf8_proof == DORA_UTF8_VALIDATED_ASCII ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_VALIDATED,
            constant,
            input->input.length,
            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_utf8_lower(const struct dora_vector* source, struct dora_vector* out_result) {
    return utf8_transform(source, UTF8_LOWER, out_result);
}

int dora_compute_utf8_upper(const struct dora_vector* source, struct dora_vector* out_result) {
    return utf8_transform(source, UTF8_UPPER, out_result);
}

int dora_compute_utf8_reverse(const struct dora_vector* source, struct dora_vector* out_result) {
    return utf8_transform(source, UTF8_REVERSE, out_result);
}
