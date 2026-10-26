// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
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
// Grouped Text Aggregation
//
//------------------------------------------------------------------------------

struct utf8_group {
    struct dora_bytes key;
    uint64_t hash;
    size_t head, tail, byte_count;
};

struct utf8_group_row {
    struct dora_bytes value;
    size_t next;
};

struct utf8_grouping {
    struct utf8_group* groups;
    struct utf8_group_row* rows;
    size_t* table;
    size_t count, row_count, table_count, null_group;
};

struct utf8_group_inputs {
    const struct dora_vector* keys;
    const struct dora_vector* values;
};

struct utf8_group_sizes {
    size_t group_byte_count, row_byte_count, table_byte_count, table_count;
};

static int utf8_group_sizes(uint64_t rows, struct utf8_group_sizes* out_sizes) {
    struct utf8_group_sizes sizes = {};
    size_t count = 0, minimum = 0;
    if (ckd_add(&count, rows, 0) || ckd_mul(&minimum, count, 2) ||
        ckd_mul(&sizes.group_byte_count, count, sizeof(struct utf8_group)) ||
        ckd_mul(&sizes.row_byte_count, count, sizeof(struct utf8_group_row))) {
        return EOVERFLOW;
    }
    if (count != 0) {
        sizes.table_count = 1;
        while (sizes.table_count < minimum) {
            if (ckd_mul(&sizes.table_count, sizes.table_count, 2)) {
                return EOVERFLOW;
            }
        }
    }
    if (ckd_mul(&sizes.table_byte_count, sizes.table_count, sizeof(size_t)) || sizes.group_byte_count > PTRDIFF_MAX ||
        sizes.row_byte_count > PTRDIFF_MAX || sizes.table_byte_count > PTRDIFF_MAX) {
        return EOVERFLOW;
    }
    *out_sizes = sizes;
    return DORA_OK;
}

static int utf8_group_admit(struct dora_pool pool, const struct dora_vector* source) {
    struct vector_bound_node leaf;
    struct vector_bound_input input = {};
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, source, &leaf, &input);
    }
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&input, 0, true);
    }
    if (status == DORA_OK && input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    dora_scratch_scope_end(&scope);
    return status;
}

static int utf8_group_lookup(struct utf8_grouping* grouping,
                             struct dora_bytes key,
                             uint64_t hash,
                             bool is_null,
                             size_t* out_group) {
    size_t group = grouping->null_group, bucket = 0;
    if (!is_null) {
        if (grouping->table_count == 0) {
            return EINVAL;
        }
        bucket = (size_t)hash & (grouping->table_count - 1);
        size_t visited = 0;
        for (; visited < grouping->table_count; ++visited) {
            group = grouping->table[bucket];
            if (group == SIZE_MAX) {
                break;
            }
            if (group >= grouping->count) {
                return EINVAL;
            }
            const struct utf8_group* const existing = &grouping->groups[group];
            if (existing->hash == hash && existing->key.byte_count == key.byte_count &&
                (key.byte_count == 0 || memcmp(existing->key.data, key.data, key.byte_count) == 0)) {
                break;
            }
            bucket = (bucket + 1) & (grouping->table_count - 1);
        }
        if (visited == grouping->table_count) {
            return EINVAL;
        }
    }
    if (group == SIZE_MAX) {
        if (grouping->count >= grouping->row_count) {
            return EINVAL;
        }
        group = grouping->count++;
        grouping->groups[group] = (struct utf8_group){.key = key, .hash = hash, .head = SIZE_MAX, .tail = SIZE_MAX};
        if (is_null) {
            grouping->null_group = group;
        } else {
            grouping->table[bucket] = group;
        }
    }
    *out_group = group;
    return DORA_OK;
}

static int utf8_group_build(struct scratch_scope* scope,
                            struct utf8_grouping* grouping,
                            const struct utf8_group_inputs* inputs,
                            struct dora_bytes separator) {
    struct group_binding {
        struct compute_bytes_source bytes[2];
        struct kernel_unary_binding hash;
        uint64_t hashes[KERNEL_CHUNK_ROW_COUNT];
    };

    void* workspace = nullptr;
    int status = dora_scratch_allocate(scope, sizeof(struct group_binding), &workspace);
    if (status != DORA_OK) {
        return status;
    }
    struct group_binding* const binding = workspace;
    memset(binding, 0, sizeof *binding);
    status = dora_compute_bytes_bind(scope,
                                     0,
                                     inputs->keys,
                                     1,
                                     false,
                                     VECTOR_OPERAND_REQUIRE_PACKET,
                                     false,
                                     &binding->bytes[0]);
    if (status == DORA_OK && inputs->values != nullptr) {
        status = dora_compute_bytes_bind(scope,
                                         1,
                                         inputs->values,
                                         1,
                                         false,
                                         VECTOR_OPERAND_REQUIRE_PACKET,
                                         false,
                                         &binding->bytes[1]);
    }
    if (status == DORA_OK) {
        const struct kernel_bind_request request = {.input_type = &dora_compute_utf8_type};
        status = dora_kernel_hash_bind(&request, &binding->hash);
    }
    for (size_t first = 0; status == DORA_OK && first < grouping->row_count; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, grouping->row_count);
        status = dora_compute_bytes_prepare(&binding->bytes[0], rows);
        if (status == DORA_OK && inputs->values != nullptr) {
            status = dora_compute_bytes_prepare(&binding->bytes[1], rows);
        }
        if (status == DORA_OK) {
            struct kernel_result result = {.values = binding->hashes, .row_capacity = (uint32_t)rows.length};
            status = binding->hash.fill(&binding->bytes[0].packet, &binding->hash, &result);
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const size_t row = first + lane;
            const bool key_null = !compute_bytes_piece_is_valid(&binding->bytes[0], lane);
            const bool value_null =
                inputs->values == nullptr || !compute_bytes_piece_is_valid(&binding->bytes[1], lane);
            const struct dora_bytes key =
                key_null ? (struct dora_bytes){} : compute_bytes_piece(&binding->bytes[0], lane);
            const struct dora_bytes value =
                value_null ? (struct dora_bytes){} : compute_bytes_piece(&binding->bytes[1], lane);
            const uint64_t hash = binding->hashes[lane];
            size_t group = 0;
            status = utf8_group_lookup(grouping, key, hash, key_null, &group);
            if (status != DORA_OK) {
                break;
            }
            if (!value_null) {
                struct utf8_group* const target = &grouping->groups[group];
                size_t next = target->byte_count;
                if ((target->head != SIZE_MAX && ckd_add(&next, next, separator.byte_count)) ||
                    ckd_add(&next, next, value.byte_count) || next > INT64_MAX) {
                    status = EOVERFLOW;
                    break;
                }
                grouping->rows[row] = (struct utf8_group_row){.value = value, .next = SIZE_MAX};
                if (target->head == SIZE_MAX) {
                    target->head = row;
                } else {
                    grouping->rows[target->tail].next = row;
                }
                target->tail = row;
                target->byte_count = next;
            }
        }
    }
    dora_compute_bytes_release(binding->bytes, 2);
    memset(workspace, 0, sizeof(struct group_binding));
    return status;
}

static int utf8_group_measure(struct dora_pool pool,
                              const struct utf8_grouping* grouping,
                              bool keys,
                              struct compute_bytes_result* result) {
    result->stored = grouping->count;
    int status = dora_compute_bytes_offsets(pool, result);
    if (status == DORA_OK && result->stored != 0) {
        size_t bytes = 0;
        status = ckd_mul(&bytes, bitmap_words_for_bits(result->stored), sizeof(uint64_t))
                     ? EOVERFLOW
                     : dora_pool_allocate(pool, bytes, &result->buffers[0]);
        if (status == DORA_OK) {
            bitmap_fill_bytes((uint8_t*)result->buffers[0].address, result->buffers[0].capacity_bytes, false);
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    if (result->stored != 0 && result->buffers[0].address == nullptr) {
        return EINVAL;
    }
    int64_t* const offsets = result->buffers[1].address;
    for (size_t index = 0; index < grouping->count; ++index) {
        const struct utf8_group* const group = &grouping->groups[index];
        const bool valid = (keys && index != grouping->null_group) || (!keys && group->head != SIZE_MAX);
        const size_t size = valid ? (keys ? group->key.byte_count : group->byte_count) : 0;
        if (ckd_add(&result->data_bytes, result->data_bytes, size) || result->data_bytes > INT64_MAX) {
            return EOVERFLOW;
        }
        offsets[index + 1] = result->data_bytes;
        bitmap_assign(result->buffers[0].address, index, valid);
        result->nulls += (uint64_t)!valid;
    }
    return dora_pool_allocate(pool, result->data_bytes, &result->buffers[2]);
}

static int utf8_group_fill(const struct utf8_grouping* grouping,
                           struct dora_bytes separator,
                           bool keys,
                           struct compute_bytes_result* result) {
    const int64_t* const offsets = result->buffers[1].address;
    size_t total = 0;
    for (size_t index = 0; index < grouping->count; ++index) {
        if (offsets[index] < 0 || (uint64_t)offsets[index] != total || offsets[index + 1] < offsets[index] ||
            (uint64_t)offsets[index + 1] > result->data_bytes) {
            return EINVAL;
        }
        const struct dora_mutable_bytes output = {
            result->buffers[2].address == nullptr ? nullptr : (uint8_t*)result->buffers[2].address + total,
            (size_t)(offsets[index + 1] - offsets[index])
        };
        const struct utf8_group* const group = &grouping->groups[index];
        size_t position = 0;
        int status = DORA_OK;
        if (keys && index != grouping->null_group) {
            status = utf8_copy((struct dora_utf8_input){group->key, DORA_UTF8_VALIDATED},
                               DORA_UTF8_UNCOMMITTED,
                               output,
                               &position);
        } else if (!keys) {
            size_t visited = 0;
            struct dora_utf8_join_state joined = {};
            for (size_t row = group->head; status == DORA_OK && row != SIZE_MAX; row = grouping->rows[row].next) {
                if (row >= grouping->row_count || visited++ == grouping->row_count) {
                    return EINVAL;
                }
                status = utf8_join_append((struct dora_utf8_input){grouping->rows[row].value, DORA_UTF8_VALIDATED},
                                          (struct dora_utf8_input){separator, DORA_UTF8_VALIDATED},
                                          DORA_UTF8_EMIT,
                                          output,
                                          &joined);
                position = joined.byte_count;
            }
        }
        if (status != DORA_OK || position != output.byte_count) {
            return status == DORA_OK ? EINVAL : status;
        }
        if (ckd_add(&total, total, position) || total > result->data_bytes) {
            return EINVAL;
        }
    }
    return total == result->data_bytes ? DORA_OK : EINVAL;
}

int dora_compute_utf8_join_groups(const struct dora_vector* source,
                                  const struct dora_vector* keys,
                                  const struct dora_vector* separator,
                                  struct dora_vector* out_keys,
                                  struct dora_vector* out_result) {
    const struct dora_vector* const inputs[] = {keys, source, separator};
    const uint32_t input_count = separator == nullptr ? 2 : 3;
    struct dora_pool pool = {};
    int status = dora_vector_get_shared_pool(inputs, 2, &pool);
    if (status != DORA_OK) {
        return status;
    }
    if (!dora_vector_is_empty(out_keys) || !dora_vector_is_empty(out_result) || out_keys == out_result) {
        return EINVAL;
    }
    for (uint32_t index = 0; index < input_count; ++index) {
        if (out_keys == inputs[index] || out_result == inputs[index]) {
            return EINVAL;
        }
    }
    uint64_t length = 0, value_length = 0, separator_length = 1;
    status = dora_vector_get_length(keys, &length);
    if (status == DORA_OK) {
        status = dora_vector_get_length(source, &value_length);
    }
    if (status == DORA_OK && separator != nullptr) {
        status = dora_vector_get_length(separator, &separator_length);
    }
    if (status != DORA_OK || length != value_length || separator_length != 1) {
        return status == DORA_OK ? EINVAL : status;
    }
    struct scratch_scope scope = {};
    status = dora_scratch_scope_begin(pool, &scope);
    struct utf8_group_sizes sizes = {};
    if (status == DORA_OK) {
        status = utf8_group_sizes(length, &sizes);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        status = utf8_group_admit(pool, inputs[index]);
    }
    bool separator_null = false;
    if (status == DORA_OK && separator != nullptr) {
        status = dora_vector_is_null(separator, 0, &separator_null);
    }
    struct dora_vector key_copy = {}, value_copy = {}, separator_copy = {};
    if (status == DORA_OK) {
        status = dora_vector_retain(keys, &key_copy);
    }
    if (status == DORA_OK && !separator_null) {
        status = dora_vector_retain(source, &value_copy);
    }
    struct dora_bytes delimiter = DORA_BYTES_LITERAL(",");
    uint64_t null_values = length;
    if (status == DORA_OK && !separator_null) {
        status = dora_vector_get_null_count(&value_copy, &null_values);
    }
    if (status == DORA_OK && separator != nullptr && !separator_null && null_values != length) {
        status = dora_vector_retain(separator, &separator_copy);
        if (status == DORA_OK) {
            status = dora_vector_read_bytes(&separator_copy, 0, &delimiter, &separator_null);
        }
    }
    void* group_storage = nullptr;
    void* row_storage = nullptr;
    void* table_storage = nullptr;
    if (status == DORA_OK && length != 0) {
        status = dora_scratch_allocate(&scope, sizes.group_byte_count, &group_storage);
    }
    if (status == DORA_OK && length != 0) {
        status = dora_scratch_allocate(&scope, sizes.row_byte_count, &row_storage);
    }
    if (status == DORA_OK && length != 0) {
        status = dora_scratch_allocate(&scope, sizes.table_byte_count, &table_storage);
        if (status == DORA_OK) {
            memset(table_storage, 0xFF, sizes.table_byte_count);
        }
    }
    struct utf8_grouping grouping = {
        .groups = group_storage,
        .rows = row_storage,
        .table = table_storage,
        .row_count = length,
        .table_count = sizes.table_count,
        .null_group = SIZE_MAX
    };
    if (status == DORA_OK) {
        const struct utf8_group_inputs stable = {.keys = &key_copy, .values = separator_null ? nullptr : &value_copy};
        status = utf8_group_build(&scope, &grouping, &stable, delimiter);
    }
    struct compute_bytes_result results[2] = {};
    for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
        status = utf8_group_measure(pool, &grouping, index == 0, &results[index]);
        if (status == DORA_OK) {
            status = utf8_group_fill(&grouping, delimiter, index == 0, &results[index]);
        }
    }

    enum vector_content_state states[2] = {VECTOR_CONTENT_VALIDATED_ASCII, VECTOR_CONTENT_VALIDATED_ASCII};

    for (uint32_t index = 0; status == DORA_OK && index < input_count; ++index) {
        struct vector_image* image = nullptr;
        status = dora_vector_resolve(inputs[index], &image);
        if (status == DORA_OK) {
            const enum vector_content_state state = compute_proof_content(vector_utf8_proof(image));
            const uint32_t output = index == 0 ? 0 : 1;
            if (state < states[output]) {
                states[output] = state;
            }
        }
    }
    struct dora_vector published[2] = {};
    for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
        status = dora_compute_bytes_publish(&scope,
                                            &results[index],
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            states[index],
                                            false,
                                            grouping.count,
                                            &published[index]);
    }
    for (uint32_t index = 2; index != 0; --index) {
        dora_compute_bytes_result_release(&results[index - 1]);
    }
    dora_scratch_scope_end(&scope);
    dora_vector_release(&separator_copy);
    dora_vector_release(&value_copy);
    dora_vector_release(&key_copy);
    if (status == DORA_OK && (!dora_vector_is_empty(out_keys) || !dora_vector_is_empty(out_result))) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        *out_keys = published[0];
        *out_result = published[1];
    } else {
        dora_vector_release(&published[1]);
        dora_vector_release(&published[0]);
    }
    return status;
}
