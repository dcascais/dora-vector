// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "utf8/input.h"
#include "utf8/like.h"
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
// LIKE Patterns
//
//------------------------------------------------------------------------------

static int utf8_like_singletons(struct compute_bytes_source inputs[3], bool escaped, bool* out_valid) {
    *out_valid = false;
    int status = dora_compute_bytes_prepare_singleton(&inputs[1]);
    if (status != DORA_OK || !compute_bytes_piece_is_valid(&inputs[1], 0)) {
        return status;
    }
    if (escaped) {
        status = dora_compute_bytes_prepare_singleton(&inputs[2]);
    }
    if (status == DORA_OK) {
        *out_valid = !escaped || compute_bytes_piece_is_valid(&inputs[2], 0);
    }
    return status;
}

static int utf8_like_compile(struct scratch_scope* scope,
                             struct compute_bytes_source inputs[3],
                             bool escaped,
                             bool insensitive,
                             struct dora_utf8_like_program* pattern) {
    const struct dora_bytes measured = compute_bytes_piece(&inputs[1], 0);
    const struct dora_bytes escape_bytes = escaped ? compute_bytes_piece(&inputs[2], 0) : (struct dora_bytes){};
    struct dora_utf8_like_program probe = {};
    int status = dora_utf8_like_compile(
        (struct dora_utf8_input){measured, inputs[1].utf8_proof},
        (struct dora_utf8_input){escape_bytes, escaped ? inputs[2].utf8_proof : DORA_UTF8_VALIDATED_ASCII},
        insensitive,
        (struct dora_mutable_bytes){},
        &probe);
    if (status == ENOBUFS) {
        status = 0;
    }
    size_t size = 0;
    if (status == 0) {
        status = dora_utf8_like_workspace_size((struct dora_utf8_input){measured, inputs[1].utf8_proof}, &size);
    }
    void* storage = nullptr;
    if (status == 0) {
        status = dora_scratch_allocate(scope, size, &storage);
        pattern->tokens = storage;
    }
    bool valid = false;
    if (status == 0) {
        status = utf8_like_singletons(inputs, escaped, &valid);
    }
    if (status == 0 && !valid) {
        status = EINVAL;
    }
    const struct dora_bytes current = status == 0 ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){};
    const struct dora_bytes current_escape =
        status == 0 && escaped ? compute_bytes_piece(&inputs[2], 0) : (struct dora_bytes){};
    if (status == 0 &&
        (current.byte_count != measured.byte_count || current_escape.byte_count != escape_bytes.byte_count)) {
        status = EINVAL;
    }
    size_t current_size = 0;
    if (status == 0) {
        status = dora_utf8_like_workspace_size((struct dora_utf8_input){current, inputs[1].utf8_proof}, &current_size);
    }
    if (status == 0 && current_size != size) {
        status = EINVAL;
    }
    if (status == 0) {
        status = dora_utf8_like_compile(
            (struct dora_utf8_input){current, DORA_UTF8_VALIDATED},
            (struct dora_utf8_input){current_escape, escaped ? inputs[2].utf8_proof : DORA_UTF8_VALIDATED_ASCII},
            insensitive,
            (struct dora_mutable_bytes){storage, size},
            pattern);
    }
    return status;
}

static int utf8_like(const struct dora_vector* source,
                     const struct dora_vector* pattern,
                     const struct dora_vector* escape,
                     bool insensitive,
                     struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const struct dora_vector* const vectors[] = {source, pattern, escape};
    const uint32_t count = escape == nullptr ? 2U : 3U;
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source || out_result == pattern || out_result == escape) {
        return EINVAL;
    }
    struct dora_vector published = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status != DORA_OK) {
        return vector_finish_publication(&published, out_result, status);
    }
    struct compute_bytes_source inputs[3];
    for (uint32_t index = 0; index < count; ++index) {
        dora_compute_bytes_source_init(&inputs[index]);
    }
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        status = dora_compute_bytes_bind(&scope,
                                         index,
                                         vectors[index],
                                         1,
                                         false,
                                         VECTOR_OPERAND_REQUIRE_PACKET,
                                         index != 0,
                                         &inputs[index]);
        if (status == DORA_OK && inputs[index].input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
            status = ENOTSUP;
        }
        if (status == DORA_OK && index != 0 && inputs[index].input.length != 1) {
            status = EINVAL;
        }
    }
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    const uint64_t stored = constant ? 1 : inputs[0].input.length;
    struct compute_output output = {};
    if (status == DORA_OK) {
        status = dora_compute_output_init(pool,
                                          &(struct dora_logical_type){.id = DORA_TYPE_BOOLEAN},
                                          inputs[0].input.length,
                                          constant,
                                          false,
                                          &output);
        dora_compute_output_zero_values(&output);
    }
    if (status == DORA_OK) {
        // Allocator callbacks would invalidate the prepared pattern.
        status = dora_compute_output_null(&output);
        output.null_count = 0;
    }
    bool valid = false;
    if (status == DORA_OK && (stored != 0 || escape != nullptr)) {
        status = utf8_like_singletons(inputs, escape != nullptr, &valid);
    }
    if (status == DORA_OK && escape != nullptr && valid) {
        struct dora_utf8_like_program checked_escape = {};
        status =
            dora_utf8_like_compile((struct dora_utf8_input){.proof = DORA_UTF8_VALIDATED},
                                   (struct dora_utf8_input){compute_bytes_piece(&inputs[2], 0), inputs[2].utf8_proof},
                                   false,
                                   (struct dora_mutable_bytes){},
                                   &checked_escape);
    }
    uint64_t eligible = UINT64_MAX;
    for (uint64_t first = 0; status == DORA_OK && valid && eligible == UINT64_MAX && first < stored;
         first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        status = dora_compute_bytes_prepare_current(&inputs[0], rows);
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            if (compute_bytes_piece_is_valid(&inputs[0], lane)) {
                eligible = first + lane;
                break;
            }
        }
    }
    struct dora_utf8_like_program compiled = {};
    if (status == DORA_OK && eligible != UINT64_MAX) {
        status = utf8_like_compile(&scope, inputs, escape != nullptr, insensitive, &compiled);
        if (status == DORA_OK) {
            status = dora_compute_bytes_prepare_current(&inputs[0], (struct dora_range){eligible, 1});
        }
        if (status == DORA_OK && !compute_bytes_piece_is_valid(&inputs[0], 0)) {
            status = EINVAL;
        }
    }
    if (eligible == UINT64_MAX) {
        output.null_count = output.length;
    }
    for (uint64_t first = 0; status == DORA_OK && eligible != UINT64_MAX && first < stored;
         first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, stored);
        status = dora_compute_bytes_prepare_current(&inputs[0], rows);
        struct kernel_result result = dora_compute_output_result(&output, first, (uint32_t)rows.length);
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const bool present = compute_bytes_piece_is_valid(&inputs[0], lane);
            result.null_count += (uint64_t)!present;
            if (!present) {
                continue;
            }
            bool match = false;
            status = dora_utf8_like_match(
                (struct dora_utf8_input){compute_bytes_piece(&inputs[0], lane), inputs[0].utf8_proof},
                &compiled,
                &match);
            if (status == DORA_OK) {
                bitmap_assign(result.values, lane, match);
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
    dora_compute_bytes_release(inputs, count);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

int dora_compute_utf8_like(const struct dora_vector* source,
                           const struct dora_vector* pattern,
                           const struct dora_vector* escape,
                           struct dora_vector* out_result) {
    return utf8_like(source, pattern, escape, false, out_result);
}

int dora_compute_utf8_ilike(const struct dora_vector* source,
                            const struct dora_vector* pattern,
                            const struct dora_vector* escape,
                            struct dora_vector* out_result) {
    return utf8_like(source, pattern, escape, true, out_result);
}
