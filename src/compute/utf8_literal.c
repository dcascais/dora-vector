// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "utf8/batch.h"
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
// Concat
//
//------------------------------------------------------------------------------

static int
concat_prepare(struct compute_bytes_source* source, struct dora_range rows, struct dora_utf8_batch* out_batch) {
    if (source->text.fill == nullptr) {
        int status = dora_vector_operand_prepare_utf8(&source->operand, rows, nullptr, out_batch);
        if (status == DORA_OK) {
            out_batch->proof = source->utf8_proof;
        }
        return status;
    }
    const int status = dora_compute_bytes_prepare(source, rows);
    if (status != DORA_OK) {
        return status;
    }
    *out_batch = (struct dora_utf8_batch){
        .row_count = rows.length,
        .physical_value_count = source->is_constant ? 1 : (size_t)rows.length,
        .layout = source->is_constant ? DORA_UTF8_CONSTANT : DORA_UTF8_DESCRIPTORS,
        .proof = DORA_UTF8_VALIDATED_ASCII
    };
    if (source->is_constant) {
        out_batch->constant = compute_bytes_piece(source, 0);
        if (!compute_bytes_piece_is_valid(source, 0)) {
            out_batch->eligible_rows = (struct dora_utf8_mask){
                {(const uint8_t*)source->validity, (size_t)bitmap_bytes_for_bits(rows.length)},
                0
            };
        }
    } else {
        out_batch->descriptors = source->pieces;
        if (source->validity != nullptr) {
            out_batch->eligible_rows = (struct dora_utf8_mask){
                {(const uint8_t*)source->validity, (size_t)bitmap_bytes_for_bits(rows.length)},
                0
            };
        }
    }
    return DORA_OK;
}

static int concat_offsets(struct compute_bytes_source* inputs,
                          uint32_t input_count,
                          struct dora_range rows,
                          struct dora_utf8_batch* batches,
                          struct compute_bytes_result* in_out_result) {
    for (uint32_t index = 0; index < input_count; ++index) {
        const int status = concat_prepare(&inputs[index], rows, &batches[index]);
        if (status != DORA_OK) {
            return status;
        }
    }
    size_t total;
    int status = dora_utf8_measure_concat(batches, input_count, (struct dora_utf8_mask){}, nullptr, &total, nullptr);
    if (status != DORA_OK) {
        return status;
    }
    int64_t* offsets = (int64_t*)in_out_result->buffers[1].address + rows.offset;
    size_t end;
    if (ckd_add(&end, (size_t)offsets[0], total) || end > INT64_MAX) {
        return EOVERFLOW;
    }
    offsets[rows.length] = end;
    for (uint32_t index = 0; index < input_count; ++index) {
        if (batches[index].proof == DORA_UTF8_CHECKED) {
            batches[index].proof = DORA_UTF8_VALIDATED;
        }
    }
    return DORA_OK;
}

static int concat_copy(const struct dora_utf8_batch* batches,
                       uint32_t input_count,
                       struct dora_range rows,
                       const struct compute_bytes_result* result) {
    int64_t* offsets = (int64_t*)result->buffers[1].address + rows.offset;
    const struct dora_utf8_batch_output output = {
        .layout = DORA_UTF8_PACKED_I64,
        .packed.bytes = {result->buffers[2].address, result->buffers[2].capacity_bytes},
        .packed.offset_bytes = {(uint8_t*)offsets, ((size_t)rows.length + 1) * sizeof *offsets},
        .packed.byte_offset = (size_t)offsets[0]
    };
    return dora_utf8_concat(batches, input_count, (struct dora_utf8_mask){}, DORA_UTF8_UNCOMMITTED, &output, nullptr);
}

static int concat_chunk(struct dora_pool pool,
                        struct compute_bytes_source* inputs,
                        uint32_t input_count,
                        struct dora_range rows,
                        struct dora_utf8_batch* batches,
                        struct compute_bytes_result* in_out_result) {
    int status = concat_offsets(inputs, input_count, rows, batches, in_out_result);
    const size_t position =
        status == DORA_OK ? (size_t)((const int64_t*)in_out_result->buffers[1].address)[rows.offset + rows.length] : 0;
    if (status == DORA_OK) {
        status = dora_allocation_reserve(pool, position, in_out_result->data_bytes, &in_out_result->buffers[2]);
    }
    if (status == DORA_OK) {
        status = concat_copy(batches, input_count, rows, in_out_result);
        if (status == DORA_OK) {
            in_out_result->data_bytes = position;
        }
    }
    return status;
}

static int concat_exact(struct dora_pool pool,
                        struct compute_bytes_source* inputs,
                        uint32_t input_count,
                        struct dora_utf8_batch* batches,
                        struct compute_bytes_result* in_out_result) {
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < in_out_result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        status = concat_offsets(inputs,
                                input_count,
                                vector_operand_chunk(first, in_out_result->stored),
                                batches,
                                in_out_result);
    }
    const size_t total =
        status == DORA_OK ? (size_t)((const int64_t*)in_out_result->buffers[1].address)[in_out_result->stored] : 0;
    if (status == DORA_OK && total != 0) {
        status = dora_pool_allocate(pool, total, &in_out_result->buffers[2]);
    }
    if (status == DORA_OK) {
        in_out_result->data_bytes = total;
        for (uint32_t index = 0; index < input_count; ++index) {
            if (inputs[index].utf8_proof == DORA_UTF8_CHECKED) {
                inputs[index].utf8_proof = DORA_UTF8_VALIDATED;
            }
            dora_vector_operand_reset(&inputs[index].operand);
        }
    }
    for (uint64_t first = 0; status == DORA_OK && first < in_out_result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, in_out_result->stored);
        for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
            status = concat_prepare(&inputs[index], rows, &batches[index]);
        }
        if (status == DORA_OK) {
            status = concat_copy(batches, input_count, rows, in_out_result);
        }
    }
    return status;
}

int dora_compute_utf8_concat(const struct dora_vector* inputs, uint32_t input_count, struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool_span(inputs, input_count, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || inputs == nullptr || input_count == 0) {
        return EINVAL;
    }
    size_t workspace_bytes = 0;
    if (ckd_mul(&workspace_bytes, input_count, sizeof(struct compute_bytes_source) + sizeof(struct dora_utf8_batch))) {
        return EOVERFLOW;
    }
    void* workspace = nullptr;
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_scratch_slot_require(&scope,
                                           vector_operand_slot(0, VECTOR_OPERAND_SLOT_CACHE),
                                           &workspace,
                                           workspace_bytes,
                                           64);
        if (status != DORA_OK) {
            dora_scratch_scope_end(&scope);
        }
    }
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const sources = workspace;
    struct dora_utf8_batch* const batches = (struct dora_utf8_batch*)(sources + input_count);
    for (uint32_t index = 0; index < input_count; ++index) {
        dora_compute_bytes_source_init(&sources[index]);
    }
    status = dora_compute_bytes_bind(&scope, 0, inputs, input_count, true, VECTOR_OPERAND_REQUIRE_TEXT, false, sources);
    bool constant = status == DORA_OK && sources[0].input.length != 0;
    for (uint32_t index = 0; constant && index < input_count; ++index) {
        constant = sources[index].input.is_constant;
    }
    struct compute_bytes_result result = {.stored = constant ? 1 : sources[0].input.length};
    if (status == DORA_OK) {
        status = dora_compute_bytes_offsets(pool, &result);
    }
    bool bytes_only = status == DORA_OK;
    for (uint32_t index = 0; bytes_only && index < input_count; ++index) {
        bytes_only = sources[index].text.fill == nullptr;
    }
    if (status == DORA_OK && bytes_only) {
        status = concat_exact(pool, sources, input_count, batches, &result);
    }
    for (uint64_t first = 0; status == DORA_OK && !bytes_only && first < result.stored;
         first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result.stored);
        status = concat_chunk(pool, sources, input_count, rows, batches, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            dora_compute_bytes_content(sources, input_count),
                                            constant,
                                            sources[0].input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(sources, input_count);
    memset(workspace, 0, workspace_bytes);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Unicode Substring
//
//------------------------------------------------------------------------------

static int utf8_substring_ranges(struct compute_bytes_source* source,
                                 struct dora_range rows,
                                 struct compute_bytes_slice slice,
                                 struct dora_utf8_batch* out_batch,
                                 struct dora_range* out_ranges) {
    int status = dora_vector_operand_prepare_utf8(&source->operand, rows, nullptr, out_batch);
    out_batch->proof = source->utf8_proof;
    size_t count = 0;
    if (ckd_add(&count, slice.count, 0)) {
        count = SIZE_MAX;
    }
    if (status != DORA_OK) {
        return status;
    }
    if (slice.offset >= 0) {
        size_t start = 0;
        if (ckd_add(&start, slice.offset, 0)) {
            start = SIZE_MAX;
        }
        return dora_utf8_locate_batch(out_batch, start, count, out_ranges, nullptr);
    }
    status = dora_compute_bytes_prepare_validate(source, rows, source->utf8_proof, nullptr);
    const uint64_t magnitude = (uint64_t)(-(slice.offset + 1)) + 1;
    for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
        out_ranges[lane] = (struct dora_range){};
        if (compute_bytes_piece_is_valid(source, lane)) {
            status =
                dora_utf8_locate_tail(magnitude,
                                      (struct dora_utf8_input){compute_bytes_piece(source, lane), DORA_UTF8_VALIDATED},
                                      count,
                                      &out_ranges[lane]);
        }
    }
    return status;
}

static int utf8_substring_rows(struct compute_bytes_source* source,
                               struct compute_bytes_slice slice,
                               enum dora_utf8_emit_mode mode,
                               struct compute_bytes_result* result) {
    struct dora_range ranges[KERNEL_CHUNK_ROW_COUNT];
    int64_t* const offsets = result->buffers[1].address;
    size_t bytes = 0;
    int status = DORA_OK;
    dora_vector_operand_reset(&source->operand);
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        struct dora_utf8_batch batch;
        status = utf8_substring_ranges(source, rows, slice, &batch, ranges);
        if (status == DORA_OK && mode == DORA_UTF8_EMIT) {
            status = dora_compute_bytes_prepare(source, rows);
            if (status == DORA_OK) {
                status = dora_compute_bytes_check_validity(result, first, &source->packet);
            }
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const bool valid = compute_bytes_mask_is_valid(batch.eligible_rows, lane);
            const struct dora_range range = valid ? ranges[lane] : (struct dora_range){};
            size_t next = bytes;
            if (valid && (ckd_add(&next, bytes, range.length) || next > INT64_MAX ||
                          (mode == DORA_UTF8_EMIT && next > result->data_bytes))) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK && valid && mode == DORA_UTF8_EMIT && range.length != 0) {
                const struct dora_bytes piece = compute_bytes_piece(source, lane);
                if (range.offset > piece.byte_count || range.length > piece.byte_count - range.offset) {
                    status = EINVAL;
                } else {
                    utf8_copy_validated_bytes((struct dora_bytes){piece.data + range.offset, (size_t)range.length},
                                              (uint8_t*)result->buffers[2].address + bytes);
                }
            }
            if (status == DORA_OK) {
                offsets[first + lane + 1] = (int64_t)next;
                bytes = next;
            }
        }
    }
    if (status == DORA_OK) {
        result->data_bytes = bytes;
    }
    return status;
}

int dora_compute_utf8_substring(const struct dora_vector* source,
                                int64_t scalar_offset,
                                uint64_t scalar_count,
                                struct dora_vector* out_result) {
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
    status = dora_compute_bytes_bind(&scope, 0, source, 1, false, VECTOR_OPERAND_REQUIRE_PACKET_TEXT, false, input);
    if (status == DORA_OK && input->input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    const bool all_valid = compute_bytes_operand_is_all_valid(&input->operand);
    if (status == DORA_OK) {
        status = dora_compute_bytes_preserve_validity(source, constant, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, !all_valid, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_capacity(input, &result);
    }
    if (status == DORA_OK) {
        bool drained = false;
        status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
        if (status == ENOMEM) {
            status = utf8_substring_rows(input,
                                         (struct compute_bytes_slice){scalar_offset, scalar_count},
                                         DORA_UTF8_MEASURE,
                                         &result);
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
            }
        }
    }
    if (status == DORA_OK) {
        status = utf8_substring_rows(input,
                                     (struct compute_bytes_slice){scalar_offset, scalar_count},
                                     DORA_UTF8_EMIT,
                                     &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_compact(pool, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            dora_compute_bytes_content(input, 1),
                                            constant,
                                            input->input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(input, 1);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Literal Search
//
//------------------------------------------------------------------------------

static int utf8_search(const struct dora_vector* source,
                       const struct dora_vector* needle,
                       struct dora_vector* out_offsets,
                       struct dora_vector* out_found) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_found) || out_found == source || out_found == needle ||
        (out_offsets != nullptr && (!dora_vector_is_empty(out_offsets) || out_offsets == source ||
                                    out_offsets == needle || out_offsets == out_found))) {
        return EINVAL;
    }
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[2];
    if (status != DORA_OK) {
        return status;
    }
    struct compute_bytes_source* const inputs = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }
    status = dora_compute_bytes_bind(&scope, 0, source, 1, false, VECTOR_OPERAND_REQUIRE_TEXT, false, &inputs[0]);
    if (status == DORA_OK) {
        status = dora_compute_bytes_bind(&scope, 1, needle, 1, false, VECTOR_OPERAND_REQUIRE_PACKET, true, &inputs[1]);
    }
    if (status == DORA_OK && (inputs[0].input.nodes[0].field.type.id != DORA_TYPE_UTF8 ||
                              inputs[1].input.nodes[0].field.type.id != DORA_TYPE_UTF8)) {
        status = ENOTSUP;
    }
    if (status == DORA_OK && inputs[1].input.length != 1) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_prepare(&inputs[1], (struct dora_range){0, 1});
    }
    const bool needle_valid = status == DORA_OK && compute_bytes_piece_is_valid(&inputs[1], 0);
    struct dora_utf8_input pattern = {
        .bytes = needle_valid ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){},
        .proof = inputs[1].utf8_proof
    };
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    const uint64_t stored = constant ? 1 : inputs[0].input.length;
    struct compute_output offsets;
    offsets.buffers[0] = (struct dora_allocation){};
    offsets.buffers[1] = (struct dora_allocation){};
    offsets.type.id = DORA_TYPE_NULL;
    struct compute_output found;
    found.buffers[0] = (struct dora_allocation){};
    found.buffers[1] = (struct dora_allocation){};
    found.type.id = DORA_TYPE_NULL;
    if (status == DORA_OK && out_offsets != nullptr) {
        status = dora_compute_output_init(pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_UINT64},
                                          inputs[0].input.length,
                                          constant,
                                          false,
                                          &offsets);
    }
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_BOOLEAN},
                                          inputs[0].input.length,
                                          constant,
                                          false,
                                          &found);
        dora_compute_output_zero_values(&found);
    }
    if (status == DORA_OK && needle_valid) {
        status = dora_compute_output_preserve_validity(&found, source);
        if (status == DORA_OK && out_offsets != nullptr) {
            offsets.validity_source = found.validity_source;
            offsets.is_nullable = found.is_nullable;
        }
    }
    if (out_offsets != nullptr && !constant) {
        found.is_nullable = false;
    }
    size_t positions[KERNEL_CHUNK_ROW_COUNT];
    const uint64_t no_rows[KERNEL_CHUNK_WORD_COUNT] = {};
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        struct dora_utf8_batch batch;
        batch.row_count = rows.length;
        batch.eligible_rows = (struct dora_utf8_mask){{(const uint8_t*)no_rows, sizeof no_rows}, 0};
        struct kernel_result flags = dora_compute_output_result(&found, first, batch.row_count);
        if (needle_valid) {
            status = dora_vector_operand_prepare_utf8(&inputs[0].operand, rows, nullptr, &batch);
            if (status == DORA_OK) {
                status = dora_utf8_find_batch(&batch,
                                              pattern,
                                              positions,
                                              flags.values,
                                              (size_t)bitmap_words_for_bits(batch.row_count),
                                              nullptr);
            }
            if (status == DORA_OK && (batch.eligible_rows.bytes.data == nullptr ||
                                      dora_bitmap_count_set_bytes(batch.eligible_rows.bytes.data,
                                                                  batch.eligible_rows.bit_offset,
                                                                  batch.row_count) != 0)) {
                pattern.proof = DORA_UTF8_VALIDATED;
            }
        }
        if (out_offsets != nullptr) {
            uint64_t* const values = (uint64_t*)offsets.buffers[1].address + first;
            memset(values, 0, (size_t)rows.length * sizeof *values);
            for (size_t word = 0; status == DORA_OK && word < bitmap_words_for_bits(rows.length); ++word) {
                uint64_t matches = ((const uint64_t*)flags.values)[word];
                if (word + 1 == bitmap_words_for_bits(rows.length)) {
                    matches &= bitmap_tail_mask(rows.length);
                }
                while (matches != 0) {
                    const size_t lane = word * DORA_BITMAP_WORD_BIT_COUNT + dora_bitmap_word_pop_lowest_set(&matches);
                    if (ckd_add(&values[lane], positions[lane], 0)) {
                        status = EOVERFLOW;
                        break;
                    }
                }
            }
            if (status == DORA_OK) {
                status = dora_compute_output_commit_utf8_mask(&offsets, first, batch.row_count, batch.eligible_rows);
            }
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit_utf8_mask(&found, first, batch.row_count, batch.eligible_rows);
        }
    }
    struct dora_vector offset_vector = {}, found_vector = {};
    if (status == DORA_OK && out_offsets != nullptr) {
        status = dora_compute_output_finish(&offsets, &offset_vector);
    }
    if (status == DORA_OK) {
        if (out_offsets != nullptr && !constant && found.null_count != 0 && found.validity_source == nullptr) {
            found.validity_source = &offset_vector;
        }
        status = dora_compute_output_finish(&found, &found_vector);
    }
    dora_compute_output_release(&found);
    dora_compute_output_release(&offsets);
    dora_compute_bytes_release(inputs, 2);
    dora_scratch_scope_end(&scope);
    if (status == DORA_OK &&
        (!dora_vector_is_empty(out_found) || (out_offsets != nullptr && !dora_vector_is_empty(out_offsets)))) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        if (out_offsets != nullptr) {
            *out_offsets = offset_vector;
        }
        *out_found = found_vector;
    } else {
        dora_vector_release(&found_vector);
        dora_vector_release(&offset_vector);
    }
    return status;
}

int dora_compute_utf8_find(const struct dora_vector* source,
                           const struct dora_vector* needle,
                           struct dora_vector* out_byte_offsets,
                           struct dora_vector* out_found) {
    return out_byte_offsets == nullptr ? EINVAL : utf8_search(source, needle, out_byte_offsets, out_found);
}

int dora_compute_utf8_contains(const struct dora_vector* source,
                               const struct dora_vector* needle,
                               struct dora_vector* out_result) {
    return utf8_search(source, needle, nullptr, out_result);
}

//------------------------------------------------------------------------------
//
// Literal Replacement
//
//------------------------------------------------------------------------------

static int utf8_replace_rows(struct compute_bytes_source inputs[3],
                             enum dora_utf8_emit_mode mode,
                             struct compute_bytes_result* result) {
    struct compute_bytes_source* const source = &inputs[0];
    const bool valid_pattern =
        compute_bytes_piece_is_valid(&inputs[1], 0) && compute_bytes_piece_is_valid(&inputs[2], 0);
    const struct dora_bytes needle = valid_pattern ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){};
    const struct dora_bytes replacement = valid_pattern ? compute_bytes_piece(&inputs[2], 0) : (struct dora_bytes){};
    int64_t* const offsets = result->buffers[1].address;
    bool checked = false;
    size_t bytes = 0;
    result->nulls = 0;
    int status = DORA_OK;
    dora_vector_operand_reset(&source->operand);
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        if (valid_pattern) {
            status = dora_compute_bytes_prepare_validate(source, rows, source->utf8_proof, nullptr);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool valid = valid_pattern && compute_bytes_piece_is_valid(source, lane);
            if (valid && !checked) {
                status = utf8_check_input((struct dora_utf8_input){needle, inputs[1].utf8_proof});
                if (status == DORA_OK) {
                    status = utf8_check_input((struct dora_utf8_input){replacement, inputs[2].utf8_proof});
                }
                checked = status == DORA_OK;
            }
            size_t size = 0;
            if (status == DORA_OK && valid) {
                const struct dora_mutable_bytes output = mode == DORA_UTF8_EMIT ? (struct dora_mutable_bytes){
                    result->buffers[2].address == nullptr ? nullptr : (uint8_t*)result->buffers[2].address + bytes,
                    result->data_bytes - bytes} : (struct dora_mutable_bytes){};
                status = utf8_replace((struct dora_utf8_input){compute_bytes_piece(source, lane), DORA_UTF8_VALIDATED},
                                      (struct dora_utf8_input){needle, DORA_UTF8_VALIDATED},
                                      (struct dora_utf8_input){replacement, DORA_UTF8_VALIDATED},
                                      mode,
                                      output,
                                      &size);
            }
            if (status == DORA_OK && (ckd_add(&bytes, bytes, size) || bytes > INT64_MAX ||
                                      (mode == DORA_UTF8_EMIT && bytes > result->data_bytes))) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK) {
                offsets[row + 1] = (int64_t)bytes;
                if (result->buffers[0].address != nullptr) {
                    bitmap_assign(result->buffers[0].address, row, valid);
                }
                result->nulls += (uint64_t)!valid;
            }
        }
    }
    if (status == DORA_OK) {
        result->data_bytes = bytes;
    }
    return status;
}

int dora_compute_utf8_replace(const struct dora_vector* source,
                              const struct dora_vector* needle,
                              const struct dora_vector* replacement,
                              struct dora_vector* out_result) {
    const struct dora_vector* const vectors[] = {source, needle, replacement};
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source || out_result == needle ||
        out_result == replacement) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[3];
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const inputs = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }
    for (uint32_t index = 0; status == DORA_OK && index < 3; ++index) {
        status = dora_compute_bytes_bind(&scope,
                                         index,
                                         vectors[index],
                                         1,
                                         false,
                                         VECTOR_OPERAND_REQUIRE_PACKET_TEXT,
                                         index != 0,
                                         &inputs[index]);
        if (status == DORA_OK && inputs[index].input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
            status = ENOTSUP;
        }
        if (status == DORA_OK && index != 0) {
            status = inputs[index].input.length != 1
                         ? EINVAL
                         : dora_compute_bytes_prepare(&inputs[index], (struct dora_range){0, 1});
        }
    }
    const bool valid =
        status == DORA_OK && compute_bytes_piece_is_valid(&inputs[1], 0) && compute_bytes_piece_is_valid(&inputs[2], 0);
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : inputs[0].input.length};
    const bool all_valid = valid && compute_bytes_operand_is_all_valid(&inputs[0].operand);
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, !all_valid, &result);
    }
    if (status == DORA_OK) {
        status = valid ? dora_compute_bytes_capacity(&inputs[0], &result) : DORA_OK;
    }
    if (status == DORA_OK && valid) {
        const struct dora_bytes needle_bytes = compute_bytes_piece(&inputs[1], 0);
        const struct dora_bytes replacement_bytes = compute_bytes_piece(&inputs[2], 0);
        if (needle_bytes.byte_count != 0 && replacement_bytes.byte_count > needle_bytes.byte_count) {
            size_t growth = 0;
            if (ckd_mul(&growth,
                        result.data_bytes / needle_bytes.byte_count,
                        replacement_bytes.byte_count - needle_bytes.byte_count) ||
                ckd_add(&result.data_bytes, result.data_bytes, growth) || result.data_bytes > INT64_MAX) {
                status = utf8_replace_rows(inputs, DORA_UTF8_MEASURE, &result);
            }
        }
    }
    if (status == DORA_OK) {
        bool drained = false;
        status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
        if (status == ENOMEM) {
            status = utf8_replace_rows(inputs, DORA_UTF8_MEASURE, &result);
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
            }
        }
    }
    if (status == DORA_OK) {
        status = utf8_replace_rows(inputs, DORA_UTF8_EMIT, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_compact(pool, &result);
    }
    if (status == DORA_OK) {
        enum vector_content_state content = dora_compute_bytes_content(inputs, 3);
        if (content >= VECTOR_CONTENT_VALIDATED &&
            dora_compute_bytes_content(inputs, 1) == VECTOR_CONTENT_VALIDATED_ASCII &&
            dora_compute_bytes_content(inputs + 2, 1) == VECTOR_CONTENT_VALIDATED_ASCII) {
            content = VECTOR_CONTENT_VALIDATED_ASCII;
        }
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            content,
                                            constant,
                                            inputs[0].input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(inputs, 3);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Trim And Affixes
//
//------------------------------------------------------------------------------

static int
utf8_trim_set(struct scratch_scope* scope, struct compute_bytes_source* source, struct dora_utf8_trim_set* out_set) {
    const struct dora_bytes characters = compute_bytes_piece(source, 0);
    size_t capacity = 0;
    int status = dora_utf8_count((struct dora_utf8_input){characters, source->utf8_proof}, &capacity);
    size_t bytes = 0;
    void* storage = nullptr;
    if (status == DORA_OK) {
        status =
            ckd_mul(&bytes, capacity, sizeof(uint32_t)) ? EOVERFLOW : dora_scratch_allocate(scope, bytes, &storage);
    }
    uint32_t* const scalars = storage;
    size_t consumed = 0, count = 0;
    if (status == DORA_OK) {
        status = dora_utf8_decode(characters, scalars, capacity, &consumed, &count);
    }
    if (status == ENOBUFS) {
        const int validation = dora_utf8_validate_span(characters, nullptr);
        status = validation == DORA_OK ? EINVAL : validation;
    }
    if (status == DORA_OK) {
        *out_set = (struct dora_utf8_trim_set){scalars, count};
    }
    return status;
}

static int utf8_trim_rows(struct compute_bytes_source* source,
                          const struct dora_utf8_trim_set* characters,
                          enum dora_utf8_emit_mode mode,
                          struct compute_bytes_result* result) {
    dora_vector_operand_reset(&source->operand);
    int64_t* const offsets = result->buffers[1].address;
    size_t bytes = 0;
    for (uint64_t first = 0; first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        int status = dora_compute_bytes_prepare(source, rows);
        if (status == DORA_OK) {
            status = dora_compute_bytes_check_validity(result, first, &source->packet);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            struct dora_range range = {};
            struct dora_bytes piece = {};
            if (compute_bytes_piece_is_valid(source, lane)) {
                piece = compute_bytes_piece(source, lane);
                status = dora_utf8_trim_range((struct dora_utf8_input){piece, source->utf8_proof}, characters, &range);
            }
            size_t next = 0;
            if (status == DORA_OK && (ckd_add(&next, bytes, range.length) || next > INT64_MAX ||
                                      (mode == DORA_UTF8_EMIT && next > result->data_bytes))) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK && mode == DORA_UTF8_EMIT && range.length != 0) {
                if (range.offset > piece.byte_count || range.length > piece.byte_count - range.offset) {
                    status = EINVAL;
                } else {
                    utf8_copy_validated_bytes((struct dora_bytes){piece.data + range.offset, (size_t)range.length},
                                              (uint8_t*)result->buffers[2].address + bytes);
                }
            }
            if (status == DORA_OK) {
                offsets[first + lane + 1] = (int64_t)next;
                bytes = next;
            }
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    result->data_bytes = bytes;
    return DORA_OK;
}

int dora_compute_utf8_trim(const struct dora_vector* source,
                           const struct dora_vector* characters,
                           struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source || out_result == characters) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[2];
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const inputs = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }

    status = dora_compute_bytes_bind_literal(&scope, source, characters, VECTOR_OPERAND_REQUIRE_PACKET_TEXT, inputs);
    const bool null_scalar =
        status == DORA_OK && characters != nullptr && inputs[1].operand.kind == VECTOR_OPERAND_CONSTANT_NULL;
    const bool constant =
        status == DORA_OK && (inputs[0].input.is_constant || null_scalar) && inputs[0].input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : inputs[0].input.length};
    result.nulls = null_scalar ? result.stored : 0;
    if (status == DORA_OK && (characters == nullptr || inputs[1].operand.kind == VECTOR_OPERAND_CONSTANT)) {
        status = dora_compute_bytes_preserve_validity(source, constant, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(
            pool,
            !(compute_bytes_operand_is_all_valid(&inputs[0].operand) &&
              (characters == nullptr || inputs[1].operand.kind == VECTOR_OPERAND_CONSTANT)),
            &result);
    }
    if (status == DORA_OK && null_scalar) {
        ((int64_t*)result.buffers[1].address)[result.stored] = 0;
    }
    if (status == DORA_OK && !null_scalar) {
        status = dora_compute_bytes_capacity(&inputs[0], &result);
    }
    struct dora_utf8_trim_set trim_set = {};
    if (status == DORA_OK && !null_scalar && characters != nullptr && result.data_bytes != 0) {
        status = dora_compute_bytes_prepare(&inputs[1], (struct dora_range){0, 1});
        if (status == DORA_OK) {
            status = utf8_trim_set(&scope, &inputs[1], &trim_set);
        }
    }
    if (status == DORA_OK) {
        bool drained = false;
        status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
        if (status == ENOMEM && !null_scalar) {
            status =
                utf8_trim_rows(&inputs[0], characters != nullptr ? &trim_set : nullptr, DORA_UTF8_MEASURE, &result);
            if (status == DORA_OK) {
                status = dora_memory_payload_allocate_attempt(pool, result.data_bytes, &drained, &result.buffers[2]);
            }
        }
    }
    if (status == DORA_OK && !null_scalar) {
        status = utf8_trim_rows(&inputs[0], characters != nullptr ? &trim_set : nullptr, DORA_UTF8_EMIT, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_compact(pool, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            dora_compute_bytes_content(inputs, 1),
                                            constant,
                                            inputs[0].input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(inputs, 2);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

static int utf8_affix(const struct dora_vector* source,
                      const struct dora_vector* pattern,
                      bool suffix,
                      struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (pattern == nullptr || !dora_vector_is_empty(out_result) || out_result == source || out_result == pattern) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[2];
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const inputs = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }
    status = dora_compute_bytes_bind_literal(&scope, source, pattern, VECTOR_OPERAND_REQUIRE_PACKET, inputs);
    if (status == DORA_OK) {
        status = dora_compute_bytes_prepare(&inputs[1], (struct dora_range){0, 1});
    }
    const bool pattern_valid = status == DORA_OK && compute_bytes_piece_is_valid(&inputs[1], 0);
    const struct dora_bytes needle = pattern_valid ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){};
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    const uint64_t stored = constant ? 1 : inputs[0].input.length;
    struct compute_output output;
    output.buffers[0] = (struct dora_allocation){};
    output.buffers[1] = (struct dora_allocation){};
    output.type.id = DORA_TYPE_NULL;
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_BOOLEAN},
                                          inputs[0].input.length,
                                          constant,
                                          false,
                                          &output);
        dora_compute_output_zero_values(&output);
    }
    bool checked = false;
    if (status == DORA_OK && pattern_valid) {
        status = dora_compute_output_preserve_validity(&output, source);
        output.is_nullable = true;
    }
    for (uint64_t first = 0; status == DORA_OK && first < stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        if (pattern_valid) {
            status = dora_compute_bytes_prepare(&inputs[0], rows);
        }
        struct kernel_result flags = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        bitmap_fill_bytes((uint8_t*)flags.validity,
                          (size_t)bitmap_words_for_bits(rows.length) * sizeof(uint64_t),
                          false);
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const bool valid = pattern_valid && compute_bytes_piece_is_valid(&inputs[0], lane);
            flags.null_count += (uint64_t)!valid;
            if (!valid) {
                continue;
            }
            const struct dora_bytes piece = compute_bytes_piece(&inputs[0], lane);
            status = utf8_check_input((struct dora_utf8_input){piece, inputs[0].utf8_proof});
            if (status == DORA_OK && !checked) {
                status = utf8_check_input((struct dora_utf8_input){needle, inputs[1].utf8_proof});
                checked = status == DORA_OK;
            }
            if (status == DORA_OK) {
                const bool match = utf8_has_affix_validated(piece, needle, suffix);
                bitmap_assign(flags.values, lane, match);
                bitmap_assign(flags.validity, lane, true);
            }
        }
        if (status == DORA_OK) {
            status = dora_compute_output_commit(&output, first, &flags);
        }
    }
    if (status == DORA_OK) {
        status = dora_compute_output_finish(&output, &published);
    }
    dora_compute_output_release(&output);
    dora_compute_bytes_release(inputs, 2);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_utf8_starts_with(const struct dora_vector* source,
                                  const struct dora_vector* prefix,
                                  struct dora_vector* out_result) {
    return utf8_affix(source, prefix, false, out_result);
}

int dora_compute_utf8_ends_with(const struct dora_vector* source,
                                const struct dora_vector* suffix,
                                struct dora_vector* out_result) {
    return utf8_affix(source, suffix, true, out_result);
}

//------------------------------------------------------------------------------
//
// Scalar Padding
//
//------------------------------------------------------------------------------

static int utf8_padding_rows(struct compute_bytes_source inputs[2],
                             struct dora_utf8_padding padding,
                             bool write,
                             struct compute_bytes_result* result) {
    dora_vector_operand_reset(&inputs[0].operand);
    dora_vector_operand_reset(&inputs[1].operand);
    int status = dora_compute_bytes_prepare(&inputs[1], (struct dora_range){0, 1});
    const bool padding_valid = status == 0 && compute_bytes_piece_is_valid(&inputs[1], 0);
    padding.proof = inputs[1].utf8_proof;
    padding.bytes = padding_valid ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){};
    bool checked = false;
    size_t bytes = 0;
    int64_t* const offsets = result->buffers[1].address;
    for (uint64_t first = 0; status == 0 && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        if (padding_valid) {
            status = dora_compute_bytes_prepare(&inputs[0], rows);
        }
        for (uint32_t lane = 0; status == 0 && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool valid = padding_valid && compute_bytes_piece_is_valid(&inputs[0], lane);
            if (valid && !checked) {
                status = dora_utf8_count(
                    (struct dora_utf8_input){padding.bytes, write ? DORA_UTF8_VALIDATED : inputs[1].utf8_proof},
                    &padding.pattern_scalar_count);
                checked = status == 0;
                if (checked && padding.proof == DORA_UTF8_CHECKED) {
                    padding.proof = DORA_UTF8_VALIDATED;
                }
            }
            size_t size = 0, next = 0;
            struct dora_mutable_bytes output = {};
            if (status == 0 && write) {
                if (valid != dora_bitmap_test(result->buffers[0].address, row) || offsets[row] < 0 ||
                    offsets[row + 1] < offsets[row] || (uint64_t)offsets[row] != bytes ||
                    (uint64_t)offsets[row + 1] > result->data_bytes) {
                    return EINVAL;
                }
                output.byte_count = (size_t)(offsets[row + 1] - offsets[row]);
                output.data = output.byte_count == 0 ? nullptr : (uint8_t*)result->buffers[2].address + bytes;
            }
            if (status == 0 && valid) {
                status = utf8_pad(
                    (struct dora_utf8_input){
                        compute_bytes_piece(&inputs[0], lane),
                        write ? DORA_UTF8_VALIDATED : inputs[0].utf8_proof
                    },
                    &padding,
                    write ? DORA_UTF8_EMIT : DORA_UTF8_MEASURE,
                    output,
                    &size);
            }
            if (status == ENOBUFS) {
                status = EINVAL;
            }
            if (status == 0 && (ckd_add(&next, bytes, size) || next > INT64_MAX)) {
                status = EOVERFLOW;
            }
            if (status == 0 && write) {
                if (next != (uint64_t)offsets[row + 1]) {
                    status = EINVAL;
                }
            } else if (status == 0) {
                offsets[row + 1] = next;
                bitmap_assign(result->buffers[0].address, row, valid);
                result->nulls += (uint64_t)!valid;
            }
            bytes = next;
        }
    }
    if (status == 0 && write && bytes != result->data_bytes) {
        status = EINVAL;
    } else if (status == 0 && !write) {
        result->data_bytes = bytes;
    }
    return status;
}

static int compute_utf8_pad(const struct dora_vector* source,
                            int64_t target_scalar_count,
                            const struct dora_vector* padding,
                            bool left,
                            struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (padding == nullptr || !dora_vector_is_empty(out_result) || out_result == source || out_result == padding) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    struct compute_bytes_source local_inputs[2];
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source* const inputs = local_inputs;
    for (size_t index = 0; index < ARRAY_COUNT(local_inputs); ++index) {
        dora_compute_bytes_source_init(&local_inputs[index]);
    }

    status = dora_compute_bytes_bind_literal(&scope, source, padding, VECTOR_OPERAND_REQUIRE_PACKET, inputs);
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : inputs[0].input.length};
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, result.stored != 0, &result);
    }
    const struct dora_utf8_padding specification = {
        .target_scalar_count = target_scalar_count > 0 ? (uint64_t)target_scalar_count : 0,
        .is_left_padding = left
    };
    if (status == DORA_OK) {
        status = utf8_padding_rows(inputs, specification, false, &result);
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, result.data_bytes, &result.buffers[2]);
    }
    if (status == DORA_OK) {
        status = utf8_padding_rows(inputs, specification, true, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            dora_compute_bytes_content(inputs, 2),
                                            constant,
                                            inputs[0].input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);
    dora_compute_bytes_release(inputs, 2);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_utf8_lpad(const struct dora_vector* source,
                           int64_t target_scalar_count,
                           const struct dora_vector* padding,
                           struct dora_vector* out_result) {
    return compute_utf8_pad(source, target_scalar_count, padding, true, out_result);
}

int dora_compute_utf8_rpad(const struct dora_vector* source,
                           int64_t target_scalar_count,
                           const struct dora_vector* padding,
                           struct dora_vector* out_result) {
    return compute_utf8_pad(source, target_scalar_count, padding, false, out_result);
}
