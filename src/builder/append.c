// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "builder/builder_internal.h"
#include "builder/state.h"
#include "compute/compute_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/builder.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "hash/hash.h"
#include "hash/keyset.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "numeric/decimal.h"
#include "statistics/statistics_internal.h"
#include "temporal/temporal.h"
#include "types/order.h"
#include "types/text.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "utf8/utf8.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Appends
//
//------------------------------------------------------------------------------

struct builder_input {
    const struct dora_vector_view* view;
    uint64_t first;
    const void* values;
    const bool* validity;
    const uint8_t* bits;
    const uint8_t* validity_bits;
    uint64_t bit_offset;
    uint64_t validity_bit_offset;
    uint64_t row_count;
    enum dora_physical_type physical;
    bool is_boolean;
    bool is_byte_input;
    bool is_all_null;
    bool is_masked;
    bool are_rows_admitted;
    bool is_constant;
    bool is_in_window;
    enum vector_content_state content_state;
    bool is_content_carried;
};

static uint32_t builder_child(uint32_t node, const struct builder_state* state, uint32_t ordinal) {
    for (uint32_t index = state->topology[node].first_child_index; index != UINT32_MAX;
         index = state->topology[index].next_sibling_index) {
        if (state->topology[index].child_index == ordinal) {
            return index;
        }
    }
    return UINT32_MAX;
}

static int buffer_bytes(uint64_t count, uint32_t width, size_t* out_bytes) {
    uint64_t bits = 0;
    if (ckd_mul(&bits, count, width) || ckd_add(&bits, bits, 7U) || bits / 8 > SIZE_MAX) {
        return EOVERFLOW;
    }
    *out_bytes = (size_t)(bits / 8);
    return DORA_OK;
}

static uint32_t* mutation_node_indices(struct builder_state* state) {
    return (uint32_t*)(state->mutation_nodes + state->node_count);
}

static uint32_t* mutation_buffer_indices(struct builder_state* state) {
    return (uint32_t*)(state->mutation_buffers + state->buffer_capacity);
}

static int compare_indices(const void* left, const void* right) {
    return (*(const uint32_t*)left > *(const uint32_t*)right) - (*(const uint32_t*)left < *(const uint32_t*)right);
}

static void mutation_activate_rows(struct builder_operation* mutation, uint32_t index) {
    struct builder_row_checkpoint* const candidate = mutation->state->mutation_nodes;
    if (!candidate[index].is_active) {
        const uint32_t parent = mutation->state->topology[index].parent_node_index;
        if (parent != UINT32_MAX) {
            mutation_activate_rows(mutation, parent);
        }
        candidate[index].rows = mutation->state->nodes[index].rows;
        candidate[index].first_changed_run = UINT64_MAX;
        candidate[index].is_active = true;
        mutation_node_indices(mutation->state)[mutation->data.touched_node_count++] = index;
    }
}

static struct builder_row_state* mutation_rows(struct builder_operation* mutation, uint32_t index) {
    mutation_activate_rows(mutation, index);
    struct builder_row_checkpoint* const candidate = mutation->state->mutation_nodes;
    return &candidate[index].rows;
}

static const struct builder_row_state* mutation_read_rows(const struct builder_operation* mutation, uint32_t index) {
    const struct builder_row_checkpoint* const candidate = mutation->state->mutation_nodes;
    return candidate[index].is_active ? &candidate[index].rows : &mutation->state->nodes[index].rows;
}

static struct builder_buffer_checkpoint* mutation_buffer(struct builder_operation* mutation, uint32_t index) {
    struct builder_buffer_checkpoint* const candidate = mutation->state->mutation_buffers;
    if (!candidate[index].is_active) {
        const uint64_t committed = mutation->state->buffers[index].initialized_unit_count;
        candidate[index].initialized_unit_count = committed;
        candidate[index].reached_unit_count = committed;
        candidate[index].is_active = true;
        mutation_buffer_indices(mutation->state)[mutation->data.touched_buffer_count++] = index;
    }
    return &candidate[index];
}

static struct dora_view_buffer* bound_data(struct builder_state* state) {
    return (struct dora_view_buffer*)((struct dora_vector*)(state->topology + state->node_count) + state->node_count);
}

static struct dora_vector* bound_domains(struct builder_state* state) {
    return (struct dora_vector*)(state->topology + state->node_count);
}

static void window_release(struct builder_operation* mutation) {
    struct builder_state* const state = mutation->state;
    struct builder_buffer_checkpoint* const buffers = state->mutation_buffers;
    const uint32_t* const owned = mutation_buffer_indices(state) + state->buffer_capacity;
    for (uint32_t cursor = 0; cursor < mutation->data.bound_buffer_count; ++cursor) {
        dora_allocation_release(&buffers[owned[cursor]].bound);
    }
    mutation->data.bound_buffer_count = 0;
}

static void release_domains(struct builder_operation* mutation) {
    struct builder_state* const state = mutation->state;
    struct dora_vector* const domains = bound_domains(state);
    const uint32_t* const changed = mutation_node_indices(state);
    for (uint32_t cursor = 0; cursor < mutation->data.touched_node_count; ++cursor) {
        dora_vector_release(&domains[changed[cursor]]);
    }
}

static uint64_t committed_end(const struct builder_buffer_state* buffer) {
    return buffer->unit_bit_width == 1 ? bitmap_words_for_bits(buffer->initialized_unit_count) * 64
                                       : buffer->initialized_unit_count;
}

static void mutation_finish(struct builder_operation* mutation, bool commit) {
    window_release(mutation);
    release_domains(mutation);
    struct builder_state* const state = mutation->state;
    struct builder_buffer_state* const buffers = state->buffers;
    struct builder_buffer_checkpoint* const candidate = state->mutation_buffers;
    const uint32_t* const changed_buffers = mutation_buffer_indices(state);
    for (uint32_t cursor = 0; cursor < mutation->data.touched_buffer_count; ++cursor) {
        const uint32_t index = changed_buffers[cursor];
        struct builder_buffer_state* const buffer = &buffers[index];
        const uint64_t committed = buffer->initialized_unit_count;
        uint8_t* const live = buffer->allocation.address;
        if (commit) {
            buffer->initialized_unit_count = candidate[index].initialized_unit_count;
            if (buffer->unit_bit_width == 1 && candidate[index].initialized_unit_count < committed && live != nullptr) {
                dora_bitmap_set_range(buffer->allocation.address,
                                      candidate[index].initialized_unit_count,
                                      committed,
                                      false);
            }
        } else {
            if (candidate[index].saved.address != nullptr) {
                memcpy(live + candidate[index].saved_first * buffer->unit_bit_width / 8,
                       candidate[index].saved.address,
                       (size_t)((committed_end(buffer) - candidate[index].saved_first) * buffer->unit_bit_width / 8));
            }
            if (buffer->unit_bit_width == 1 && candidate[index].reached_unit_count > committed && live != nullptr) {
                dora_bitmap_set_range(buffer->allocation.address,
                                      committed,
                                      candidate[index].reached_unit_count,
                                      false);
            }
        }
        dora_allocation_release(&candidate[index].saved);
        candidate[index] = (struct builder_buffer_checkpoint){};
    }
    struct builder_row_checkpoint* const rows = state->mutation_nodes;
    const uint32_t* const changed_nodes = mutation_node_indices(state);
    for (uint32_t cursor = 0; cursor < mutation->data.touched_node_count; ++cursor) {
        const uint32_t index = changed_nodes[cursor];
        if (commit) {
            state->nodes[index].rows = rows[index].rows;
        }
        dora_allocation_release(&rows[index].inactive);
        rows[index] = (struct builder_row_checkpoint){};
    }
    if (commit) {
        state->written_byte_count = mutation->data.written_byte_count;
        dora_statistics_checkpoint_commit(state->statistics, &mutation->data.statistics);
    } else {
        dora_statistics_checkpoint_rollback(state->statistics, &mutation->data.statistics);
    }
    dora_builder_operation_reset_checkpoints(mutation);
}

// Preserves the first committed value across repeated writes.
static int save_committed(uint32_t index, struct builder_operation* mutation, uint64_t first) {
    const struct builder_buffer_state* const buffer = &mutation->state->buffers[index];
    struct builder_buffer_checkpoint* const candidate = mutation_buffer(mutation, index);
    if (buffer->unit_bit_width == 1) {
        first = bitmap_word_of(first) * DORA_BITMAP_WORD_BIT_COUNT;
    }
    if (buffer->allocation.address == nullptr ||
        (candidate->saved.address != nullptr && candidate->saved_first <= first)) {
        return DORA_OK;
    }
    const uint64_t end = committed_end(buffer);
    const uint64_t kept = candidate->saved.address != nullptr ? candidate->saved_first : end;
    size_t bytes = 0;
    int status = buffer_bytes(end - first, buffer->unit_bit_width, &bytes);
    struct dora_allocation saved = {};
    if (status == DORA_OK) {
        status = dora_pool_allocate(mutation->state->storage.pool, bytes, &saved);
    }
    if (status != DORA_OK) {
        return status;
    }
    const size_t live_bytes = (kept - first) * buffer->unit_bit_width / 8;
    memcpy(saved.address, (const uint8_t*)buffer->allocation.address + first * buffer->unit_bit_width / 8, live_bytes);
    if (candidate->saved.address != nullptr) {
        memcpy((uint8_t*)saved.address + live_bytes, candidate->saved.address, bytes - live_bytes);
    }
    dora_allocation_release(&candidate->saved);
    candidate->saved = saved;
    candidate->saved_first = first;
    return DORA_OK;
}

static int grow_buffer(uint32_t index, struct builder_operation* mutation, size_t required) {
    struct builder_state* const state = mutation->state;
    struct builder_buffer_state* const buffer = &state->buffers[index];
    const struct builder_buffer_checkpoint* const candidate = mutation_buffer(mutation, index);
    size_t capacity = 0;
    int status =
        dora_memory_preferred_capacity(state->storage.pool, buffer->allocation.capacity_bytes, required, &capacity);
    if (status != DORA_OK) {
        return status;
    }
    if (buffer->allocation.capacity_bytes == 0 && capacity < buffer->next_capacity_bytes) {
        capacity = buffer->next_capacity_bytes;
    }
    const size_t other = state->capacity_bytes - buffer->allocation.capacity_bytes;
    const size_t limit = state->options.limit_bytes;
    if (limit != 0 && (other > limit || capacity > limit - other)) {
        capacity = required;
    }
    if (limit != 0 && (other > limit || capacity > limit - other)) {
        return ENOSPC;
    }
    struct dora_allocation allocation = {};
    status = dora_memory_payload_allocate_capacity(state->storage.pool, required, capacity, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    size_t live_capacity = 0;
    if (ckd_add(&live_capacity, other, allocation.capacity_bytes)) {
        dora_allocation_release(&allocation);
        return EOVERFLOW;
    }
    if (limit != 0 && live_capacity > limit) {
        dora_allocation_release(&allocation);
        return ENOSPC;
    }
    uint64_t written = committed_end(buffer);
    if (candidate->initialized_unit_count > written) {
        written = candidate->initialized_unit_count;
    }
    if (candidate->reached_unit_count > written) {
        written = candidate->reached_unit_count;
    }
    size_t prefix = 0;
    if (buffer_bytes(written, buffer->unit_bit_width, &prefix) != DORA_OK ||
        prefix > buffer->allocation.capacity_bytes) {
        prefix = buffer->allocation.capacity_bytes;
    }
    if (prefix != 0) {
        memcpy(allocation.address, buffer->allocation.address, prefix);
    }
    // Bit writes merge into shared words and require zeroed tails.
    if (buffer->unit_bit_width == 1) {
        bitmap_fill_bytes((uint8_t*)allocation.address + prefix, allocation.capacity_bytes - prefix, false);
    }
    dora_allocation_release(&buffer->allocation);
    buffer->allocation = allocation;
    state->capacity_bytes = live_capacity;
    return DORA_OK;
}

static int reserve_units(uint32_t index, struct builder_operation* mutation, uint64_t first, uint64_t end) {
    const struct builder_buffer_state* const buffer = &mutation->state->buffers[index];
    struct builder_buffer_checkpoint* const candidate = mutation_buffer(mutation, index);
    if (end < first) {
        return EINVAL;
    }
    if (buffer->unit_bit_width == 0 || end == first) {
        return DORA_OK;
    }
    int status = first < buffer->initialized_unit_count ? save_committed(index, mutation, first) : DORA_OK;
    if (status != DORA_OK) {
        return status;
    }
    if (buffer->unit_bit_width == 1 && ckd_mul(&end, bitmap_words_for_bits(end), UINT64_C(64))) {
        return EOVERFLOW;
    }
    size_t required = 0;
    status = buffer_bytes(end, buffer->unit_bit_width, &required);
    if (status == DORA_OK && required > buffer->allocation.capacity_bytes) {
        status = grow_buffer(index, mutation, required);
    }
    if (status == DORA_OK && end > candidate->reached_unit_count) {
        candidate->reached_unit_count = end;
    }
    return status;
}

static void* candidate_address(struct builder_operation* mutation, uint32_t index, uint64_t unit) {
    return (uint8_t*)mutation->state->buffers[index].allocation.address +
           (size_t)(unit * mutation->state->buffers[index].unit_bit_width / 8);
}

static uint64_t* candidate_bit_words(struct builder_operation* mutation, uint32_t index) {
    return mutation->state->buffers[index].allocation.address;
}

static void candidate_assign_bit(uint32_t index, struct builder_operation* mutation, uint64_t bit, bool value) {
    uint64_t* const words = candidate_bit_words(mutation, index);
    bitmap_assign(words, bit, value);
}

static int read_units(struct builder_operation* mutation,
                      uint32_t index,
                      uint64_t first,
                      uint64_t end,
                      struct dora_bytes* out_bytes) {
    const struct builder_buffer_state* const buffer = &mutation->state->buffers[index];
    const struct builder_buffer_checkpoint* const candidates = mutation->state->mutation_buffers;
    const struct builder_buffer_checkpoint* const candidate = &candidates[index];
    if (end < first || (buffer->unit_bit_width == 1 && bitmap_bit_in_byte(first) != 0)) {
        return EINVAL;
    }
    const uint64_t initialized =
        candidate->is_active ? candidate->initialized_unit_count : buffer->initialized_unit_count;
    const bool absent = buffer->allocation.address == nullptr;
    const bool implicit =
        absent && (buffer->role == BUILDER_BUFFER_VALIDITY_BITS ||
                   ((buffer->role == BUILDER_BUFFER_OFFSETS32 || buffer->role == BUILDER_BUFFER_OFFSETS64) &&
                    first == 0 && end == 1));
    if (end > initialized && !implicit) {
        return EINVAL;
    }
    size_t absolute = 0;
    size_t bytes = 0;
    int status = buffer_bytes(first, buffer->unit_bit_width, &absolute);
    if (status == DORA_OK) {
        status = buffer_bytes(end - first, buffer->unit_bit_width, &bytes);
    }
    if (status == DORA_OK) {
        *out_bytes = (struct dora_bytes){
            .data = absent ? nullptr : (const uint8_t*)buffer->allocation.address + absolute,
            .byte_count = absent ? 0 : bytes
        };
    }
    return status;
}

static int note_inactive(struct builder_operation* mutation,
                         uint32_t number,
                         uint64_t first,
                         uint64_t count,
                         const uint8_t* selected) {
    if (count == 0) {
        return DORA_OK;
    }
    const struct builder_node_state* const node = &mutation->state->nodes[number];
    const struct dora_schema_field field = *node->field;
    if ((field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0 ||
        (field.layout != DORA_LAYOUT_RUN_END && field.layout != DORA_LAYOUT_DENSE_UNION &&
         field.layout != DORA_LAYOUT_SPARSE_UNION)) {
        return DORA_OK;
    }
    mutation_activate_rows(mutation, number);
    struct builder_row_checkpoint* const checkpoints = mutation->state->mutation_nodes;
    struct builder_row_checkpoint* const checkpoint = &checkpoints[number];
    uint64_t end = 0;
    if (ckd_add(&end, first, count)) {
        return EOVERFLOW;
    }
    uint64_t begin = bitmap_word_of(first) * DORA_BITMAP_WORD_BIT_COUNT;
    if (checkpoint->inactive.address != nullptr && checkpoint->inactive_first < begin) {
        begin = checkpoint->inactive_first;
    }
    if (checkpoint->inactive_end > end) {
        end = checkpoint->inactive_end;
    }
    size_t bytes = 0;
    int status = buffer_bytes(bitmap_words_for_bits(end - begin), 64, &bytes);
    if (status != DORA_OK) {
        return status;
    }
    if (checkpoint->inactive.address == nullptr || checkpoint->inactive_first != begin ||
        checkpoint->inactive.capacity_bytes < bytes) {
        struct dora_allocation inactive = {};
        status = dora_pool_allocate(mutation->state->storage.pool, bytes, &inactive);
        if (status != DORA_OK) {
            return status;
        }
        bitmap_fill_bytes((uint8_t*)inactive.address, inactive.capacity_bytes, false);
        if (checkpoint->inactive.address != nullptr) {
            const size_t old_bytes =
                (size_t)bitmap_bytes_for_bits(checkpoint->inactive_end - checkpoint->inactive_first);
            bitmap_copy_bytes((uint8_t*)inactive.address + bitmap_byte_of(checkpoint->inactive_first - begin),
                              checkpoint->inactive.address,
                              old_bytes);
        }
        dora_allocation_release(&checkpoint->inactive);
        checkpoint->inactive = inactive;
        checkpoint->inactive_first = begin;
    }
    checkpoint->inactive_end = end;
    if (selected == nullptr) {
        dora_bitmap_set_range(checkpoint->inactive.address, first - begin, first - begin + count, true);
    } else {
        for (uint64_t row = 0; row < count; ++row) {
            if (dora_bitmap_test(selected, row)) {
                bitmap_set(checkpoint->inactive.address, first - begin + row);
            }
        }
    }
    return DORA_OK;
}

static bool is_inactive(uint32_t number, const struct builder_operation* mutation, uint64_t row) {
    const struct builder_row_checkpoint* const checkpoints = mutation->state->mutation_nodes;
    const struct builder_row_checkpoint* const checkpoint = &checkpoints[number];
    return checkpoint->inactive.address != nullptr && row >= checkpoint->inactive_first &&
           row < checkpoint->inactive_end &&
           dora_bitmap_test(checkpoint->inactive.address, row - checkpoint->inactive_first);
}

static int
append_limit(uint32_t number, const struct builder_state* state, uint64_t current, uint64_t added, uint64_t* out_end) {
    if (ckd_add(out_end, current, added) || *out_end > INT64_MAX) {
        return EOVERFLOW;
    }
    return number == 0 && state->options.row_capacity != 0 && *out_end > state->options.row_capacity ? ENOSPC : DORA_OK;
}

// Rollback preserves this all-valid prefix: it is equivalent to the formerly absent bitmap.
static int materialize_validity(struct builder_operation* mutation, uint32_t number, uint64_t rows) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_buffer_state* const validity = &state->buffers[node->first_buffer];
    const uint64_t committed = node->rows.row_count;
    size_t bytes = 0;
    const int status = buffer_bytes(committed, 1, &bytes);
    if (status != DORA_OK || ckd_add(&state->written_byte_count, state->written_byte_count, bytes)) {
        return status != DORA_OK ? status : EOVERFLOW;
    }
    dora_bitmap_set_range(state->buffers[state->nodes[number].first_buffer].allocation.address, 0, rows, true);
    validity->initialized_unit_count = committed;
    return DORA_OK;
}

static int append_validity(struct builder_operation* mutation,
                           uint32_t number,
                           const struct dora_schema_field* field,
                           uint64_t end,
                           bool valid,
                           bool masked,
                           bool are_rows_admitted) {
    const struct builder_node_state* const node = &mutation->state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    if (!valid && !masked && !are_rows_admitted && (field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
        return EINVAL;
    }
    const struct builder_buffer_state* const buffer = &mutation->state->buffers[node->first_buffer];
    if (node->buffer_count != 0 && buffer->role == BUILDER_BUFFER_VALIDITY_BITS) {
        struct builder_buffer_checkpoint* const candidate = mutation_buffer(mutation, node->first_buffer);
        const bool absent = buffer->allocation.address == nullptr;
        if (!valid || !absent) {
            if (end > candidate->reached_unit_count) {
                const int status = reserve_units(node->first_buffer, mutation, absent ? 0 : rows->row_count, end);
                if (status != DORA_OK) {
                    return status;
                }
            }
            if (absent) {
                const int status = materialize_validity(mutation, number, rows->row_count);
                if (status != DORA_OK) {
                    return status;
                }
            }
            candidate_assign_bit(node->first_buffer, mutation, rows->row_count, valid);
            candidate->initialized_unit_count = rows->row_count + 1;
        }
    }
    rows->null_count += !valid;
    rows->has_masked_nulls = rows->has_masked_nulls || (!valid && masked);
    return DORA_OK;
}

static struct builder_content_bound fixed_content_bound(const struct dora_schema_field* field,
                                                        enum dora_physical_type physical) {
    struct builder_content_bound prepared = {.status = DORA_OK};
    if (field->type.id == DORA_TYPE_DECIMAL) {
        const uint32_t width_bits = physical == DORA_PHYSICAL_TYPE_I32    ? 32
                                    : physical == DORA_PHYSICAL_TYPE_I64  ? 64
                                    : physical == DORA_PHYSICAL_TYPE_I128 ? 128
                                    : physical == DORA_PHYSICAL_TYPE_I256 ? 256
                                                                          : 0;
        prepared.status = dora_decimal_bound_prepare_width(field->type.parameters.decimal, width_bits, &prepared.bound);
    }
    return prepared;
}

static int fixed_content(const struct dora_schema_field* field,
                         enum dora_physical_type physical,
                         const struct builder_content_bound* bound,
                         const void* value) {
    if (field->type.id == DORA_TYPE_DECIMAL) {
        if (bound->status != DORA_OK) {
            return bound->status;
        }
        bool are_rows_admitted = false;
        switch (physical) {
        case DORA_PHYSICAL_TYPE_I32: {
            int32_t coefficient;
            memcpy(&coefficient, value, sizeof(coefficient));
            are_rows_admitted = decimal_bound_admits_i32(coefficient, bound->bound);
            break;
        }
        case DORA_PHYSICAL_TYPE_I64: {
            int64_t coefficient;
            memcpy(&coefficient, value, sizeof(coefficient));
            are_rows_admitted = decimal_bound_admits_i64(coefficient, bound->bound);
            break;
        }
        case DORA_PHYSICAL_TYPE_I128: {
            struct dora_i128 coefficient;
            memcpy(&coefficient, value, sizeof(coefficient));
            are_rows_admitted = decimal_bound_admits_i128(coefficient, bound->bound);
            break;
        }
        default: {
            struct dora_i256 coefficient;
            memcpy(&coefficient, value, sizeof(coefficient));
            are_rows_admitted = decimal_bound_admits_i256(coefficient, bound->bound);
            break;
        }
        }
        return are_rows_admitted ? DORA_OK : EOVERFLOW;
    }
    if (field->type.id == DORA_TYPE_DATE64 || field->type.id == DORA_TYPE_TIME) {
        int64_t ticks = 0;
        if (physical == DORA_PHYSICAL_TYPE_I32) {
            int32_t narrow;
            memcpy(&narrow, value, sizeof(narrow));
            ticks = narrow;
        } else if (physical == DORA_PHYSICAL_TYPE_I64) {
            memcpy(&ticks, value, sizeof(ticks));
        } else {
            return EINVAL;
        }
        if (field->type.id == DORA_TYPE_DATE64) {
            return ticks % 86'400'000 == 0 ? DORA_OK : ERANGE;
        }
        int64_t limit = 86'400;
        for (enum dora_time_unit unit = DORA_TIME_UNIT_SECOND; unit < field->type.parameters.unit; ++unit) {
            limit *= 1000;
        }
        return ticks >= 0 && ticks < limit ? DORA_OK : ERANGE;
    }
    return DORA_OK;
}

static int index_value(enum dora_physical_type physical, const void* value, uint64_t* out_index) {
    switch (physical) {
        SIGNED_INDEX_CARRIERS(INDEX_READ_CASE, NEGATIVE_SIGNED, value, 0, out_index)
        UNSIGNED_INDEX_CARRIERS(INDEX_READ_CASE, NEGATIVE_UNSIGNED, value, 0, out_index)
    default:
        return EINVAL;
    }
}

static int index_scalar(enum dora_physical_type physical, union decoded_row_value* out_value, uint64_t index) {
    switch (physical) {
    case DORA_PHYSICAL_TYPE_I8:
        if (index > INT8_MAX) {
            return EOVERFLOW;
        }
        out_value->i8 = index;
        break;
    case DORA_PHYSICAL_TYPE_I16:
        if (index > INT16_MAX) {
            return EOVERFLOW;
        }
        out_value->i16 = index;
        break;
    case DORA_PHYSICAL_TYPE_I32:
        if (index > INT32_MAX) {
            return EOVERFLOW;
        }
        out_value->i32 = index;
        break;
    case DORA_PHYSICAL_TYPE_I64:
        if (index > INT64_MAX) {
            return EOVERFLOW;
        }
        out_value->i64 = index;
        break;
    case DORA_PHYSICAL_TYPE_U8:
        if (index > UINT8_MAX) {
            return EOVERFLOW;
        }
        out_value->u8 = index;
        break;
    case DORA_PHYSICAL_TYPE_U16:
        if (index > UINT16_MAX) {
            return EOVERFLOW;
        }
        out_value->u16 = index;
        break;
    case DORA_PHYSICAL_TYPE_U32:
        if (index > UINT32_MAX) {
            return EOVERFLOW;
        }
        out_value->u32 = index;
        break;
    case DORA_PHYSICAL_TYPE_U64:
        out_value->u64 = index;
        break;
    default:
        return EINVAL;
    }
    return DORA_OK;
}

static int bind_domain(struct builder_operation* mutation, uint32_t number, const struct dora_vector* domain) {
    struct builder_state* const state = mutation->state;
    struct dora_vector* const domains = bound_domains(state);
    if (domains[number].handle.slot_index != 0) {
        return DORA_OK;
    }
    int status = dora_vector_retain(domain, &domains[number]);
    if (status == DORA_OK) {
        mutation_activate_rows(mutation, number);
        status = dora_vector_get_view(&domains[number], &state->topology[number].view);
        struct vector_image* image = nullptr;
        if (status == DORA_OK) {
            status = dora_vector_resolve(domain, &image);
        }
        if (status == DORA_OK) {
            state->topology[number].utf8_proof = vector_utf8_proof(image);
        }
    }
    for (uint32_t child = state->topology[number].first_child_index; status == DORA_OK && child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        struct dora_vector values = {};
        status = state->topology[number].view.layout == DORA_LAYOUT_STRUCT
                     ? dora_vector_get_child(&domains[number], state->topology[child].child_index, &values)
                     : dora_vector_get_physical_child(&domains[number], state->topology[child].child_index, &values);
        if (status == DORA_OK) {
            status = bind_domain(mutation, child, &values);
        }
    }
    return status;
}

static int bound_null(uint32_t number, struct builder_operation* mutation, uint64_t row, bool* out_null) {
    if (number >= mutation->state->node_count) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &mutation->state->topology[number].view;
    if (row >= view->length) {
        return EINVAL;
    }
    if (dora_vector_view_is_null(view, row)) {
        *out_null = true;
        return DORA_OK;
    }
    if (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END) {
        uint64_t physical = 0;
        uint32_t ordinal = 0;
        if (view->layout == DORA_LAYOUT_DICTIONARY) {
            const int status = dora_vector_view_read_index(view, row, &physical);
            if (status != DORA_OK) {
                return status;
            }
        } else {
            ordinal = 1;
            physical = dora_vector_view_find_run(view, row);
        }
        return bound_null(builder_child(number, mutation->state, ordinal), mutation, physical, out_null);
    }
    if (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION) {
        uint64_t physical = 0;
        const uint32_t ordinal = dora_vector_view_get_member(view, row, &physical);
        const uint32_t child = builder_child(number, mutation->state, ordinal);
        return child == UINT32_MAX ? EINVAL : bound_null(child, mutation, physical, out_null);
    }
    *out_null = false;
    return DORA_OK;
}

static bool input_row_is_valid(const struct builder_input* input, uint64_t row) {
    return !input->is_all_null &&
           (input->view == nullptr ||
            !dora_vector_view_is_null(input->view, input->first + (input->is_constant ? 0 : row))) &&
           (input->validity == nullptr || input->validity[row]) &&
           (input->validity_bits == nullptr ||
            dora_bitmap_test(input->validity_bits, input->validity_bit_offset + row));
}

// Decimal, Date64, Time
static bool checks_fixed_content(const struct dora_schema_field* field) {
    return field->type.id == DORA_TYPE_DECIMAL || field->type.id == DORA_TYPE_DATE64 ||
           field->type.id == DORA_TYPE_TIME;
}

// Requires reserved values and no per-value content checks.
static int append_fixed_range(struct builder_operation* mutation,
                              uint32_t number,
                              const struct dora_schema_field* field,
                              const struct builder_input* input,
                              size_t width) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    const uint32_t values = node->first_buffer + 1;
    const uint64_t first = rows->row_count;
    const uint64_t count = input->row_count;
    const uint64_t end = first + count;
    uint8_t* const destination = candidate_address(mutation, values, first);
    if (!input->is_in_window) {
        memcpy(destination, input->values, (size_t)(count * width));
    }
    const bool has_bitmap =
        node->buffer_count != 0 && state->buffers[node->first_buffer].role == BUILDER_BUFFER_VALIDITY_BITS;
    bool present = has_bitmap && state->buffers[node->first_buffer].allocation.address != nullptr;
    int status = present ? reserve_units(node->first_buffer, mutation, first, end) : DORA_OK;
    const bool required =
        !input->is_masked && !input->are_rows_admitted && (field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0;
    const bool has_input_validity = input->validity != nullptr || input->validity_bits != nullptr;
    uint64_t nulls = 0;
    for (uint64_t at = 0; status == DORA_OK && at < count; at += 64) {
        const uint32_t lanes = count - at < 64 ? (uint32_t)(count - at) : 64U;
        uint64_t valid = bitmap_mask_low(lanes);
        if (input->validity != nullptr) {
            dora_bitmap_copy_bool8_to_words(&valid, input->validity, at, lanes);
        } else if (input->validity_bits != nullptr) {
            valid = dora_bitmap_get_word(input->validity_bits,
                                         input->validity_bit_offset + at,
                                         input->validity_bit_offset + count);
        }
        const uint64_t missing = ~valid & bitmap_mask_low(lanes);
        if (missing != 0 && required) {
            status = EINVAL;
            break;
        }
        if (missing != 0 && has_bitmap && !present) {
            status = reserve_units(node->first_buffer, mutation, 0, end);
            if (status == DORA_OK) {
                status = materialize_validity(mutation, number, first + at);
            }
            present = status == DORA_OK;
        }
        if (status != DORA_OK) {
            break;
        }
        uint64_t active = input->is_in_window ? 0 : missing;
        while (active != 0) {
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&active);
            memset(destination + (size_t)((at + lane) * width), 0, width);
        }
        if (present && has_input_validity) {
            dora_bitmap_store_bits(candidate_bit_words(mutation, node->first_buffer), first + at, valid, lanes);
        }
        nulls += dora_bitmap_word_count_ones(missing);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (present) {
        if (!has_input_validity) {
            dora_bitmap_set_range(candidate_bit_words(mutation, node->first_buffer), first, end, true);
        }
        mutation_buffer(mutation, node->first_buffer)->initialized_unit_count = end;
    }
    rows->null_count += (int64_t)nulls;
    rows->has_masked_nulls = rows->has_masked_nulls || (input->is_masked && nulls != 0);
    rows->row_count = end;
    mutation_buffer(mutation, values)->initialized_unit_count = end;
    return DORA_OK;
}

static int append_fixed(struct builder_operation* mutation,
                        uint32_t number,
                        const struct dora_schema_field* field,
                        const struct builder_input* input) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    const bool dictionary = field->layout == DORA_LAYOUT_DICTIONARY;
    const enum dora_physical_type physical = state->topology[number].view.physical_type;
    if ((input->is_boolean ? field->layout != DORA_LAYOUT_BITS : (field->layout != DORA_LAYOUT_FIXED && !dictionary)) ||
        (!input->is_boolean && physical != input->physical)) {
        return EINVAL;
    }
    uint64_t end = 0;
    int status = append_limit(number, state, rows->row_count, input->row_count, &end);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_buffer_state* const values = &state->buffers[node->first_buffer + 1];
    status = reserve_units(node->first_buffer + 1, mutation, rows->row_count, end);
    if (status != DORA_OK) {
        return status;
    }
    const size_t width = values->unit_bit_width / 8;
    size_t input_bytes = 0;
    if (ckd_mul(&input_bytes, input->row_count, input->is_boolean ? sizeof(bool) : width)) {
        return EOVERFLOW;
    }
    uint64_t packed_end = 0;
    if ((input->bits != nullptr && ckd_add(&packed_end, input->bit_offset, input->row_count)) ||
        (input->validity_bits != nullptr && ckd_add(&packed_end, input->validity_bit_offset, input->row_count))) {
        return EOVERFLOW;
    }
    if (input->row_count != 0 && checks_fixed_content(field) && !input->is_content_carried &&
        state->options.content_policy == DORA_FAILURE_POLICY_TRUST) {
        rows->content_state = VECTOR_CONTENT_PROMISED;
    }
    struct builder_input range = *input;
    const struct dora_vector_view* const flat = input->view;
    if (flat != nullptr && flat->selection.indices == nullptr && flat->layout == DORA_LAYOUT_FIXED &&
        !input->is_constant && flat->values.data != nullptr) {
        range.values = (const uint8_t*)flat->values.data + (size_t)((flat->offset + input->first) * flat->byte_width);
        range.validity_bits = flat->validity.data;
        range.validity_bit_offset = flat->validity_bit_offset + input->first;
        range.view = nullptr;
    }
    if (!dictionary && !range.is_boolean && range.view == nullptr && range.values != nullptr && !range.is_all_null &&
        width != 0 &&
        (range.is_content_carried || state->options.content_policy == DORA_FAILURE_POLICY_TRUST ||
         !checks_fixed_content(field))) {
        return append_fixed_range(mutation, number, field, &range, width);
    }
    if (dictionary && !input->are_rows_admitted && node->dictionary.handle.slot_index != 0) {
        status = bind_domain(mutation, builder_child(number, state, 0), &node->dictionary);
        if (status != DORA_OK) {
            return status;
        }
    }
    const struct builder_content_bound content_bound = fixed_content_bound(field, physical);
    for (uint64_t row = 0; row < input->row_count; ++row) {
        bool valid = input_row_is_valid(input, row);
        const void* value = nullptr;
        union decoded_row_value converted = {};
        if (valid) {
            if ((input->view == nullptr && input->values == nullptr && input->bits == nullptr &&
                 (input->is_boolean || width != 0)) ||
                (input->view != nullptr && !input->is_boolean && width != 0 && input->view->values.data == nullptr)) {
                return EINVAL;
            }
            if (!input->is_boolean) {
                value = input->view == nullptr
                            ? (input->values == nullptr ? nullptr : (const uint8_t*)input->values + row * width)
                        : input->view->values.data == nullptr
                            ? nullptr
                            : (const uint8_t*)input->view->values.data +
                                  dora_vector_view_get_physical_row(input->view,
                                                                    input->first + (input->is_constant ? 0 : row)) *
                                      input->view->byte_width;
                if (dictionary) {
                    uint64_t index = 0;
                    status = input->view == nullptr
                                 ? index_value(physical, value, &index)
                                 : dora_vector_view_read_index(input->view,
                                                               input->first + (input->is_constant ? 0 : row),
                                                               &index);
                    if (status == DORA_OK && input->view != nullptr && input->view->physical_type != physical) {
                        status = index_scalar(physical, &converted, index);
                        value = &converted;
                    }
                    if (status == DORA_OK &&
                        (node->dictionary.handle.slot_index == 0 || index >= node->dictionary.rows.length)) {
                        status = EINVAL;
                    }
                    bool is_null = false;
                    if (status == DORA_OK && !input->are_rows_admitted) {
                        status = bound_null(builder_child(number, state, 0), mutation, index, &is_null);
                    }
                    if (status == DORA_OK && is_null && !input->is_masked &&
                        (field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
                        status = EINVAL;
                    }
                } else if (!input->is_content_carried && state->options.content_policy != DORA_FAILURE_POLICY_TRUST &&
                           value != nullptr) {
                    status = fixed_content(field, physical, &content_bound, value);
                    if (status == EOVERFLOW) {
                        status = ERANGE;
                    }
                    if ((status == EILSEQ || status == ERANGE) &&
                        state->options.content_policy == DORA_FAILURE_POLICY_NULL) {
                        status = DORA_OK;
                        valid = false;
                    }
                }
                if (status != DORA_OK) {
                    return status;
                }
            }
        }
        status = append_validity(mutation, number, field, end, valid, input->is_masked, input->are_rows_admitted);
        if (status != DORA_OK) {
            return status;
        }
        if (input->is_boolean) {
            bool bit = false;
            if (valid && input->view != nullptr) {
                bit = dora_vector_view_get_bool(input->view, input->first + (input->is_constant ? 0 : row));
            } else if (valid && input->bits != nullptr) {
                bit = dora_bitmap_test(input->bits, input->bit_offset + row);
            } else if (valid) {
                bit = ((const bool*)input->values)[row];
            }
            candidate_assign_bit(node->first_buffer + 1, mutation, rows->row_count, bit);
        } else if (width != 0 && !input->is_in_window) {
            void* const destination = candidate_address(mutation, node->first_buffer + 1, rows->row_count);
            if (valid) {
                memcpy(destination, value, width);
            } else {
                memset(destination, 0, width);
            }
        }
        ++rows->row_count;
        mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = rows->row_count;
    }
    return DORA_OK;
}

static void store_offset(uint32_t width, void* destination, uint64_t value) {
    if (width == 16) {
        const int16_t narrow = (int16_t)value;
        memcpy(destination, &narrow, sizeof(narrow));
    } else if (width == 32) {
        const int32_t narrow = (int32_t)value;
        memcpy(destination, &narrow, sizeof(narrow));
    } else {
        const int64_t wide = value;
        memcpy(destination, &wide, sizeof(wide));
    }
}

static uint64_t load_offset(const struct builder_buffer_state* buffer, uint64_t row) {
    if (buffer->unit_bit_width == 16) {
        int16_t value = 0;
        memcpy(&value, (const uint8_t*)buffer->allocation.address + row * sizeof(value), sizeof(value));
        return value;
    }
    if (buffer->unit_bit_width == 32) {
        int32_t value;
        memcpy(&value, (const uint8_t*)buffer->allocation.address + row * sizeof(value), sizeof(value));
        return value;
    }
    int64_t value;
    memcpy(&value, (const uint8_t*)buffer->allocation.address + row * sizeof(value), sizeof(value));
    return value;
}

static int add_data_buffer(struct builder_operation* operation, uint32_t previous, uint32_t* out_index) {
    struct builder_state* const state = operation->state;
    if (state->buffers[previous].next_buffer != 0) {
        *out_index = state->buffers[previous].next_buffer;
        return DORA_OK;
    }
    if (state->buffer_count == state->buffer_capacity) {
        const size_t stride =
            sizeof(struct builder_buffer_state) + sizeof(struct builder_buffer_checkpoint) + 2 * sizeof(uint32_t);
        uint32_t required;
        size_t bytes;
        size_t current;
        if (ckd_add(&required, state->buffer_capacity, 1U) || ckd_mul(&bytes, required, stride) ||
            ckd_mul(&current, state->buffer_capacity, stride)) {
            return EOVERFLOW;
        }
        if (state->buffer_storage.address != nullptr) {
            current = state->buffer_storage.capacity_bytes;
        }
        struct dora_allocation storage = {};
        const int status = dora_memory_payload_allocate_preferred(state->storage.pool, current, bytes, &storage);
        if (status != DORA_OK) {
            return status;
        }
        const size_t available = storage.capacity_bytes / stride;
        const uint32_t capacity = available > UINT32_MAX ? UINT32_MAX : (uint32_t)available;
        memset(storage.address, 0, storage.capacity_bytes);
        struct builder_buffer_state* const buffers = storage.address;
        struct builder_buffer_checkpoint* const checkpoints = (struct builder_buffer_checkpoint*)(buffers + capacity);
        uint32_t* const indices = (uint32_t*)(checkpoints + capacity);
        memcpy(buffers, state->buffers, state->buffer_count * sizeof(*buffers));
        memcpy(checkpoints, state->mutation_buffers, state->buffer_count * sizeof(*checkpoints));
        memcpy(indices, mutation_buffer_indices(state), operation->data.touched_buffer_count * sizeof(*indices));
        memcpy(indices + capacity,
               mutation_buffer_indices(state) + state->buffer_capacity,
               operation->data.bound_buffer_count * sizeof(*indices));
        struct dora_allocation old = state->buffer_storage;
        state->buffer_storage = storage;
        state->buffers = buffers;
        state->mutation_buffers = checkpoints;
        state->buffer_capacity = capacity;
        for (uint32_t number = 0; number < state->node_count; ++number) {
            struct builder_column_run* const cursor = &state->runs.columns[number];
            if (!cursor->is_open) {
                continue;
            }
            const uint32_t first = state->nodes[number].first_buffer;
            cursor->values = &checkpoints[first + 1];
            cursor->validity = cursor->validity == nullptr ? nullptr : &checkpoints[first];
            cursor->bytes =
                cursor->bytes == nullptr
                    ? nullptr
                    : &checkpoints[cursor->kind == BUILDER_RUN_VIEW ? cursor->rows->last_data_buffer : first + 2];
        }
        dora_allocation_release(&old);
    }
    const uint32_t index = state->buffer_count++;
    state->buffers[index] = (struct builder_buffer_state){.unit_bit_width = 8, .role = BUILDER_BUFFER_BYTES};
    state->buffers[previous].next_buffer = index;
    *out_index = index;
    return DORA_OK;
}

static constexpr uint64_t VIEW_BUFFER_BYTE_LIMIT = 32768;

static int reserve_view(uint32_t number, struct builder_operation* operation, size_t size, uint32_t* out_index) {
    struct builder_state* const state = operation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(operation, number);
    uint32_t index = rows->data_buffer_count == 0 ? node->first_buffer + 2 : rows->last_data_buffer;
    uint64_t first = rows->data_buffer_count == 0 ? 0 : mutation_buffer(operation, index)->initialized_unit_count;
    const size_t capacity = state->buffers[index].allocation.capacity_bytes;
    const bool rollover = first != 0 && ((size > capacity - first && first + size > VIEW_BUFFER_BYTE_LIMIT) ||
                                         first > (uint64_t)INT32_MAX - size);
    const uint32_t count = rows->data_buffer_count + (rows->data_buffer_count == 0 || rollover ? 1U : 0U);
    if (count > INT32_MAX) {
        return EOVERFLOW;
    }
    int status = DORA_OK;
    if (rollover) {
        status = add_data_buffer(operation, index, &index);
        first = 0;
    }
    if (status == DORA_OK) {
        status = reserve_units(index, operation, first, first + size);
    }
    if (status == DORA_OK && count != rows->data_buffer_count) {
        const uint32_t sizes = node->first_buffer + 3;
        const uint64_t extent = state->buffers[sizes].initialized_unit_count;
        status = reserve_units(sizes, operation, extent, count > extent ? count : extent);
    }
    if (status == DORA_OK) {
        rows->data_buffer_count = count;
        rows->last_data_buffer = index;
        mutation_buffer(operation, node->first_buffer + 3)->initialized_unit_count = count;
        *out_index = index;
    }
    return status;
}

static void store_view(uint32_t number, struct builder_state* state, uint64_t row, struct dora_bytes value) {
    const uint32_t first = state->nodes[number].first_buffer;
    struct builder_row_state* const rows = &state->mutation_nodes[number].rows;
    struct dora_binary_view_entry entry = {.byte_count = (int32_t)value.byte_count};
    if (value.byte_count <= VIEW_INLINE_BYTE_COUNT) {
        if (value.byte_count != 0) {
            memcpy(entry.payload.inline_bytes, value.data, value.byte_count);
        }
    } else {
        const uint32_t data_buffer = rows->last_data_buffer;
        const uint64_t offset = state->mutation_buffers[data_buffer].initialized_unit_count;
        memcpy(entry.payload.reference.prefix, value.data, sizeof(entry.payload.reference.prefix));
        entry.payload.reference.buffer_index = (int32_t)(rows->data_buffer_count - 1);
        entry.payload.reference.byte_offset = (int32_t)offset;
        uint8_t* const destination = (uint8_t*)state->buffers[data_buffer].allocation.address + offset;
        if (destination != value.data) {
            memcpy(destination, value.data, value.byte_count);
        }
        rows->byte_count += value.byte_count;
        state->mutation_buffers[data_buffer].initialized_unit_count = offset + value.byte_count;
    }
    memcpy((struct dora_binary_view_entry*)state->buffers[first + 1].allocation.address + row, &entry, sizeof(entry));
    state->mutation_buffers[first + 1].initialized_unit_count = row + 1;
}

static int append_bytes(struct builder_operation* mutation,
                        uint32_t number,
                        const struct dora_schema_field* field,
                        const struct builder_input* input) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    const bool fixed = field->type.id == DORA_TYPE_FIXED_BINARY && field->layout == DORA_LAYOUT_FIXED;
    if (!fixed && (field->type.id != DORA_TYPE_BINARY && field->type.id != DORA_TYPE_UTF8)) {
        return EINVAL;
    }
    if (!fixed && field->layout != DORA_LAYOUT_BINARY && field->layout != DORA_LAYOUT_LARGE_BINARY &&
        field->layout != DORA_LAYOUT_BINARY_VIEW) {
        return EINVAL;
    }
    uint64_t end = 0;
    int status = append_limit(number, state, rows->row_count, input->row_count, &end);
    if (status != DORA_OK) {
        return status;
    }
    const bool view = field->layout == DORA_LAYOUT_BINARY_VIEW;
    const uint64_t first = rows->row_count;
    uint64_t byte_end = rows->byte_count;
    for (uint64_t row = 0; row < input->row_count; ++row) {
        bool valid = input_row_is_valid(input, row);
        struct dora_bytes value = {};
        if (valid) {
            if (input->view == nullptr && input->values == nullptr) {
                return EINVAL;
            }
            value = input->view == nullptr
                        ? ((const struct dora_bytes*)input->values)[row]
                        : dora_vector_view_get_bytes(input->view, input->first + (input->is_constant ? 0 : row));
            if ((value.byte_count != 0 && value.data == nullptr) ||
                (fixed && value.byte_count != field->type.parameters.byte_width)) {
                return EINVAL;
            }
            if (field->type.id == DORA_TYPE_UTF8) {
                enum vector_content_state content = input->content_state;
                if (!input->is_content_carried && state->options.content_policy != DORA_FAILURE_POLICY_TRUST) {
                    bool ascii = false;
                    status = dora_utf8_validate_span(value, &ascii);
                    content = ascii ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_VALIDATED;
                    if (status == EILSEQ && state->options.content_policy == DORA_FAILURE_POLICY_NULL) {
                        status = DORA_OK;
                        valid = false;
                    }
                    if (status != DORA_OK) {
                        return status;
                    }
                }
                if (valid && content < rows->content_state) {
                    rows->content_state = content;
                }
            }
        }
        if (valid && !fixed && (!view || value.byte_count > VIEW_INLINE_BYTE_COUNT) &&
            ckd_add(&byte_end, byte_end, value.byte_count)) {
            return EOVERFLOW;
        }
        if ((!fixed && field->layout == DORA_LAYOUT_BINARY && byte_end > INT32_MAX) ||
            (view && value.byte_count > INT32_MAX) || byte_end > INT64_MAX) {
            return EOVERFLOW;
        }
        status = append_validity(mutation, number, field, end, valid, input->is_masked, input->are_rows_admitted);
        if (status != DORA_OK) {
            return status;
        }
        ++rows->row_count;
    }
    const bool offsets = !fixed && !view;
    const uint64_t committed_offsets = mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count;
    status = reserve_units(node->first_buffer + 1,
                           mutation,
                           offsets && committed_offsets != 0 ? first + 1 : first,
                           offsets ? end + 1 : end);
    if (status == DORA_OK && !fixed && !view) {
        status = reserve_units(node->first_buffer + 2, mutation, rows->byte_count, byte_end);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (offsets && committed_offsets == 0) {
        store_offset(state->buffers[node->first_buffer + 1].unit_bit_width,
                     candidate_address(mutation, node->first_buffer + 1, 0),
                     0);
    }

    const uint64_t* const validity = candidate_bit_words(mutation, node->first_buffer);
    const uint64_t long_bytes = byte_end - rows->byte_count;
    if (view && long_bytes != 0 && long_bytes <= INT32_MAX) {
        uint32_t index;
        status = reserve_view(number, mutation, (size_t)long_bytes, &index);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (offsets && field->type.id == DORA_TYPE_UTF8 && !input->is_in_window && input->view == nullptr &&
        input->values != nullptr) {
        for (uint64_t at = 0; at < input->row_count; at += 2048) {
            const uint32_t count = (uint32_t)(input->row_count - at > 2048 ? 2048 : input->row_count - at);
            const struct dora_utf8_batch batch = {
                .row_count = count,
                .physical_value_count = count,
                .layout = DORA_UTF8_DESCRIPTORS,
                .proof = DORA_UTF8_VALIDATED,
                .descriptors = (const struct dora_bytes*)input->values + at,
                .eligible_rows = validity == nullptr
                                     ? (struct dora_utf8_mask){}
                                     : (struct dora_utf8_mask){
                                           {(const uint8_t*)validity, (size_t)bitmap_bytes_for_bits(end)},
                                           first + at
                                       }
            };
            const uint32_t width = state->buffers[node->first_buffer + 1].unit_bit_width / 8;
            const struct dora_utf8_batch_output output = {
                .layout = width == 4 ? DORA_UTF8_PACKED_I32 : DORA_UTF8_PACKED_I64,
                .packed.bytes =
                    {state->buffers[node->first_buffer + 2].allocation.address,
                     state->buffers[node->first_buffer + 2].allocation.capacity_bytes},
                .packed.offset_bytes =
                    {candidate_address(mutation, node->first_buffer + 1, first + at), ((size_t)count + 1) * width},
                .packed.byte_offset = (size_t)rows->byte_count
            };
            status =
                dora_utf8_copy_batch_sized(&batch, (size_t)(byte_end - rows->byte_count), &output, nullptr, nullptr);
            if (status != DORA_OK) {
                return status;
            }
            if (width == 4) {
                int32_t finish;
                memcpy(&finish, output.packed.offset_bytes.data + (size_t)count * width, sizeof finish);
                rows->byte_count = (uint64_t)finish;
            } else {
                int64_t finish;
                memcpy(&finish, output.packed.offset_bytes.data + (size_t)count * width, sizeof finish);
                rows->byte_count = (uint64_t)finish;
            }
        }
        mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = end + 1;
        mutation_buffer(mutation, node->first_buffer + 2)->initialized_unit_count = byte_end;
        return DORA_OK;
    }
    for (uint64_t row = 0; row < input->row_count; ++row) {
        const bool valid = validity == nullptr || dora_bitmap_test((const uint8_t*)validity, first + row);
        const struct dora_bytes value =
            !valid ? (struct dora_bytes){}
            : input->view != nullptr
                ? dora_vector_view_get_bytes(input->view, input->first + (input->is_constant ? 0 : row))
            : input->values != nullptr ? ((const struct dora_bytes*)input->values)[row]
                                       : (struct dora_bytes){};
        if (fixed) {
            const size_t width = field->type.parameters.byte_width;
            void* const destination =
                width == 0 ? nullptr : candidate_address(mutation, node->first_buffer + 1, first + row);
            if (value.data != nullptr && width != 0) {
                memcpy(destination, value.data, width);
            } else if (width != 0) {
                memset(destination, 0, width);
            }
        } else if (view) {
            if (long_bytes > INT32_MAX && value.byte_count > VIEW_INLINE_BYTE_COUNT) {
                uint32_t index;
                status = reserve_view(number, mutation, value.byte_count, &index);
                if (status != DORA_OK) {
                    return status;
                }
            }
            store_view(number, state, first + row, value);
        } else {
            if (value.byte_count != 0 && value.data != nullptr && !input->is_in_window) {
                if (field->type.id == DORA_TYPE_UTF8) {
                    size_t written;
                    status = utf8_copy((struct dora_utf8_input){value, DORA_UTF8_VALIDATED},
                                       DORA_UTF8_UNCOMMITTED,
                                       (struct dora_mutable_bytes){
                                           candidate_address(mutation, node->first_buffer + 2, rows->byte_count),
                                           value.byte_count
                                       },
                                       &written);
                    if (status != DORA_OK) {
                        return status;
                    }
                } else {
                    memcpy(candidate_address(mutation, node->first_buffer + 2, rows->byte_count),
                           value.data,
                           value.byte_count);
                }
            }
            rows->byte_count += value.byte_count;
            store_offset(state->buffers[node->first_buffer + 1].unit_bit_width,
                         candidate_address(mutation, node->first_buffer + 1, first + row + 1),
                         rows->byte_count);
        }
    }
    mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = offsets ? end + 1 : end;
    if (!fixed && !view) {
        mutation_buffer(mutation, node->first_buffer + 2)->initialized_unit_count = byte_end;
    }
    return DORA_OK;
}

static int bind_part(struct builder_operation* mutation,
                     uint32_t index,
                     uint64_t first,
                     uint64_t end,
                     struct dora_view_buffer* out_buffer) {
    struct dora_bytes span = {};
    const int status = read_units(mutation, index, first, end, &span);
    if (status == DORA_OK) {
        *out_buffer = (struct dora_view_buffer){
            .data = span.data,
            .byte_count = span.byte_count,
            .readable_byte_count = span.byte_count
        };
    }
    return status;
}

static int materialize_part(struct builder_operation* mutation, uint32_t index, struct dora_view_buffer* span) {
    struct builder_buffer_checkpoint* const candidates = mutation->state->mutation_buffers;
    struct dora_allocation* const owned = &candidates[index].bound;
    if (owned->address != nullptr || span->byte_count == 0) {
        return DORA_OK;
    }
    const int status = dora_pool_allocate(mutation->state->storage.pool, span->byte_count, owned);
    if (status == DORA_OK) {
        uint32_t* const indices = mutation_buffer_indices(mutation->state) + mutation->state->buffer_capacity;
        indices[mutation->data.bound_buffer_count++] = index;
        memcpy(owned->address, span->data, span->byte_count);
        span->data = owned->address;
        span->readable_byte_count = owned->capacity_bytes;
    }
    return status;
}

static uint64_t span_offset(uint32_t width, const struct dora_view_buffer* span, uint64_t row) {
    if (width == 16) {
        int16_t value = 0;
        memcpy(&value, (const uint8_t*)span->data + row * sizeof(value), sizeof(value));
        return value;
    }
    if (width == 32) {
        int32_t value = 0;
        memcpy(&value, (const uint8_t*)span->data + row * sizeof(value), sizeof(value));
        return value;
    }
    int64_t value = 0;
    memcpy(&value, (const uint8_t*)span->data + row * sizeof(value), sizeof(value));
    return value;
}

static int bind_offset(struct builder_operation* mutation, uint32_t index, uint64_t row, uint64_t* out_value) {
    struct dora_bytes bytes = {};
    int status = read_units(mutation, index, row, row + 1, &bytes);
    if (status == DORA_OK && bytes.data == nullptr) {
        status = row == 0 ? DORA_OK : EINVAL;
        *out_value = 0;
    } else if (status == DORA_OK) {
        *out_value = span_offset(mutation->state->buffers[index].unit_bit_width,
                                 &(struct dora_view_buffer){.data = bytes.data},
                                 0);
    }
    return status;
}

static int find_run(uint32_t number, struct builder_operation* mutation, uint64_t row, uint64_t* out_run) {
    const struct builder_node_state* const nodes = mutation->state->nodes;
    const uint32_t ends = builder_child(number, mutation->state, 0);
    const uint32_t index = nodes[ends].first_buffer + 1;
    const struct builder_buffer_state* const live = &mutation->state->buffers[index];
    const uint64_t count = mutation_read_rows(mutation, ends)->row_count;
    uint64_t first = 0;
    uint64_t end = count;
    const struct dora_view_buffer span = {.data = live->allocation.address};
    if (end == 0 || span_offset(live->unit_bit_width, &span, end - 1) <= row) {
        return EINVAL;
    }
    while (first < end) {
        const uint64_t middle = first + (end - first) / 2;
        if (span_offset(live->unit_bit_width, &span, middle) <= row) {
            first = middle + 1;
        } else {
            end = middle;
        }
    }
    if (first >= count) {
        return EINVAL;
    }
    *out_run = first;
    return DORA_OK;
}

static struct dora_view_buffer*
role_span(struct dora_vector_view* view, struct dora_view_buffer* bytes, enum builder_buffer_role role) {
    switch (role) {
    case BUILDER_BUFFER_VALIDITY_BITS:
        return &view->validity;
    case BUILDER_BUFFER_VALUE_BITS:
    case BUILDER_BUFFER_FIXED_VALUES:
    case BUILDER_BUFFER_BINARY_VIEW:
        return &view->values;
    case BUILDER_BUFFER_OFFSETS32:
    case BUILDER_BUFFER_OFFSETS64:
    case BUILDER_BUFFER_MEMBER_ROWS:
        return &view->offsets;
    case BUILDER_BUFFER_SIZES32:
    case BUILDER_BUFFER_SIZES64:
        return &view->sizes;
    case BUILDER_BUFFER_TYPE_IDS:
        return &view->type_ids;
    case BUILDER_BUFFER_BYTES:
        return view->layout == DORA_LAYOUT_BINARY_VIEW ? nullptr : bytes;
    case BUILDER_BUFFER_VARIADIC_SIZES:
    case BUILDER_BUFFER_CHILD_FRONTIERS:
        return nullptr;
    }
    return nullptr;
}

static int bind_window(struct builder_operation* mutation, uint32_t number, uint64_t first, uint64_t end) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct builder_row_state* const rows = mutation_read_rows(mutation, number);
    if (first > end || end > rows->row_count) {
        return EINVAL;
    }
    state->topology[number].utf8_proof = rows->content_state == VECTOR_CONTENT_VALIDATED_ASCII
                                             ? DORA_UTF8_VALIDATED_ASCII
                                         : rows->content_state >= VECTOR_CONTENT_VALIDATED ? DORA_UTF8_VALIDATED
                                                                                           : DORA_UTF8_PROMISED;
    struct dora_vector_view* const view = &state->topology[number].view;
    view->length = end - first;
    view->offset = 0;
    view->validity_bit_offset = 0;
    view->run_end_element_offset = 0;
    view->null_count = 0;
    view->validity = (struct dora_view_buffer){};
    view->values = (struct dora_view_buffer){};
    view->offsets = (struct dora_view_buffer){};
    view->sizes = (struct dora_view_buffer){};
    view->type_ids = (struct dora_view_buffer){};
    int status;
    const struct builder_buffer_state* const buffers = state->buffers;
    for (uint32_t part = 0; part < node->buffer_count; ++part) {
        const uint32_t index = node->first_buffer + part;
        struct dora_view_buffer* const span = role_span(view, &bound_data(state)[number], buffers[index].role);
        if (span == nullptr) {
            continue;
        }
        switch (buffers[index].role) {
        case BUILDER_BUFFER_VALIDITY_BITS:
            status = bind_part(mutation, index, bitmap_byte_of(first) * DORA_BITMAP_BYTE_BIT_COUNT, end, span);
            view->validity_bit_offset = bitmap_bit_in_byte(first);
            if (status == DORA_OK && view->validity.data != nullptr) {
                view->null_count = (int64_t)(view->length - dora_bitmap_count_set_bytes(view->validity.data,
                                                                                        view->validity_bit_offset,
                                                                                        view->length));
            }
            break;
        case BUILDER_BUFFER_VALUE_BITS:
            status = bind_part(mutation, index, bitmap_byte_of(first) * DORA_BITMAP_BYTE_BIT_COUNT, end, span);
            view->offset = bitmap_bit_in_byte(first);
            break;
        case BUILDER_BUFFER_OFFSETS32:
        case BUILDER_BUFFER_OFFSETS64:
            status =
                bind_part(mutation,
                          index,
                          first,
                          end + (view->layout != DORA_LAYOUT_LIST_VIEW && view->layout != DORA_LAYOUT_LARGE_LIST_VIEW),
                          span);
            break;
        case BUILDER_BUFFER_FIXED_VALUES:
        case BUILDER_BUFFER_BINARY_VIEW:
        case BUILDER_BUFFER_MEMBER_ROWS:
        case BUILDER_BUFFER_SIZES32:
        case BUILDER_BUFFER_SIZES64:
        case BUILDER_BUFFER_TYPE_IDS:
            status = bind_part(mutation, index, first, end, span);
            break;
        case BUILDER_BUFFER_BYTES:
        case BUILDER_BUFFER_VARIADIC_SIZES:
        case BUILDER_BUFFER_CHILD_FRONTIERS:
            continue;
        default:
            return EINVAL;
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    if (view->layout == DORA_LAYOUT_NULL) {
        view->null_count = (int64_t)view->length;
        return DORA_OK;
    }
    if (view->layout == DORA_LAYOUT_DICTIONARY) {
        if (node->dictionary.handle.slot_index == 0) {
            return view->length == 0 ? DORA_OK : EINVAL;
        }
        return bind_domain(mutation, state->topology[number].first_child_index, &node->dictionary);
    }
    if (view->layout == DORA_LAYOUT_RUN_END) {
        uint64_t run_first = 0;
        uint64_t run_end = 0;
        if (first != end) {
            status = find_run(number, mutation, first, &run_first);
            if (status == DORA_OK) {
                status = find_run(number, mutation, end - 1, &run_end);
                ++run_end;
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            status = bind_window(mutation, child, run_first, run_end);
            if (status != DORA_OK) {
                return status;
            }
        }
        const struct dora_vector_view* const ends = &state->topology[state->topology[number].first_child_index].view;
        view->offset = first;
        view->values = ends->values;
        view->physical_type = ends->physical_type;
        view->byte_width = ends->byte_width;
        return DORA_OK;
    }
    if (view->layout == DORA_LAYOUT_STRUCT || view->layout == DORA_LAYOUT_SPARSE_UNION ||
        view->layout == DORA_LAYOUT_FIXED_LIST) {
        uint64_t child_first = first;
        uint64_t child_end = end;
        if (view->layout == DORA_LAYOUT_FIXED_LIST && (ckd_mul(&child_first, first, view->list_element_count) ||
                                                       ckd_mul(&child_end, end, view->list_element_count))) {
            return EOVERFLOW;
        }
        for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            status = bind_window(mutation, child, child_first, child_end);
            if (status != DORA_OK) {
                return status;
            }
        }
        return DORA_OK;
    }
    if (view->layout == DORA_LAYOUT_DENSE_UNION) {
        uint64_t starts[UNION_MEMBER_CAPACITY];
        uint64_t ends[UNION_MEMBER_CAPACITY] = {};
        for (uint32_t member = 0; member < UNION_MEMBER_CAPACITY; ++member) {
            starts[member] = UINT64_MAX;
        }
        for (uint64_t row = 0; row < view->length; ++row) {
            uint64_t member_row = 0;
            const uint32_t member = dora_vector_view_get_member(view, row, &member_row);
            if (member >= UNION_MEMBER_CAPACITY || member_row == UINT64_MAX) {
                return EINVAL;
            }
            if (member_row < starts[member]) {
                starts[member] = member_row;
            }
            if (member_row + 1 > ends[member]) {
                ends[member] = member_row + 1;
            }
        }
        for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            const uint32_t member = state->topology[child].child_index;
            if (starts[member] == UINT64_MAX) {
                starts[member] = 0;
            }
            status = bind_window(mutation, child, starts[member], ends[member]);
            if (status != DORA_OK) {
                return status;
            }
        }
        status = materialize_part(mutation, node->first_buffer + 1, &view->offsets);
        for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
            uint64_t member_row = 0;
            const uint32_t member = dora_vector_view_get_member(view, row, &member_row);
            store_offset(32, (uint8_t*)view->offsets.data + row * 4, member_row - starts[member]);
        }
        return status;
    }
    const bool list_view = view->layout == DORA_LAYOUT_LIST_VIEW || view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    const bool list = view->layout == DORA_LAYOUT_LIST || view->layout == DORA_LAYOUT_LARGE_LIST;
    const bool binary = view->layout == DORA_LAYOUT_BINARY || view->layout == DORA_LAYOUT_LARGE_BINARY;
    if (list || list_view || binary) {
        if (view->length != 0 && (view->offsets.data == nullptr || (list_view && view->sizes.data == nullptr))) {
            return EINVAL;
        }
        const uint32_t width = buffers[node->first_buffer + 1].unit_bit_width;
        uint64_t child_first = 0;
        uint64_t child_end = 0;
        if (list_view) {
            child_first = UINT64_MAX;
            for (uint64_t row = 0; row < view->length; ++row) {
                if (dora_vector_view_is_null(view, row)) {
                    continue;
                }
                const struct dora_range elements = dora_vector_view_get_elements(view, row);
                uint64_t finish = 0;
                if (ckd_add(&finish, elements.offset, elements.length)) {
                    return EOVERFLOW;
                }
                if (elements.length != 0 && elements.offset < child_first) {
                    child_first = elements.offset;
                }
                if (elements.length != 0 && finish > child_end) {
                    child_end = finish;
                }
            }
            if (child_first == UINT64_MAX) {
                child_first = 0;
            }
        } else if (view->offsets.data != nullptr) {
            child_first = span_offset(width, &view->offsets, 0);
            child_end = span_offset(width, &view->offsets, view->length);
        } else if (view->length != 0) {
            return EINVAL;
        }
        if (binary) {
            status = bind_part(mutation, node->first_buffer + 2, child_first, child_end, &bound_data(state)[number]);
        } else {
            status = bind_window(mutation, state->topology[number].first_child_index, child_first, child_end);
        }
        if (status == DORA_OK && view->offsets.data != nullptr) {
            status = materialize_part(mutation, node->first_buffer + 1, &view->offsets);
        }
        for (uint64_t row = 0; status == DORA_OK && row < view->length + !list_view; ++row) {
            const uint64_t offset = view->offsets.data == nullptr ? 0 : span_offset(width, &view->offsets, row);
            const bool inactive =
                list_view && (dora_vector_view_is_null(view, row) || span_offset(width, &view->sizes, row) == 0);
            if (view->offsets.data != nullptr) {
                store_offset(width,
                             (uint8_t*)view->offsets.data + row * (width / 8),
                             inactive ? 0 : offset - child_first);
            }
        }
        return status;
    }
    if (view->layout == DORA_LAYOUT_BINARY_VIEW) {
        const uint32_t sizes = node->first_buffer + 3;
        struct dora_allocation* const storage = &state->mutation_buffers[sizes].bound;
        size_t bytes;
        if (ckd_mul(&bytes, rows->data_buffer_count, sizeof(struct dora_view_buffer))) {
            return EOVERFLOW;
        }
        if (bytes > storage->capacity_bytes) {
            struct dora_allocation grown = {};
            status = dora_pool_allocate(state->storage.pool, bytes, &grown);
            if (status != DORA_OK) {
                return status;
            }
            if (storage->address == nullptr) {
                uint32_t* const owned = mutation_buffer_indices(state) + state->buffer_capacity;
                owned[mutation->data.bound_buffer_count++] = sizes;
            }
            struct dora_allocation old = *storage;
            *storage = grown;
            dora_allocation_release(&old);
        }
        struct dora_view_buffer* const data = storage->address;
        uint32_t index = node->first_buffer + 2;
        for (uint32_t part = 0; part < rows->data_buffer_count; ++part) {
            const struct builder_buffer_state* const buffer = &state->buffers[index];
            const struct builder_buffer_checkpoint* const checkpoint = &state->mutation_buffers[index];
            const uint64_t length =
                checkpoint->is_active ? checkpoint->initialized_unit_count : buffer->initialized_unit_count;
            data[part] = (struct dora_view_buffer){
                .data = buffer->allocation.address,
                .byte_count = length,
                .readable_byte_count = buffer->allocation.capacity_bytes
            };
            index = buffer->next_buffer;
        }
        view->data_buffers = data;
        view->data_buffer_count = rows->data_buffer_count;
        return DORA_OK;
    }
    return DORA_OK;
}

// Requires no active checkpoints.
static int scratch_begin(struct builder_operation* operation) {
    if (operation->data.touched_node_count != 0 || operation->data.touched_buffer_count != 0 ||
        operation->data.bound_buffer_count != 0) {
        return EINVAL;
    }
    dora_builder_operation_reset_checkpoints(operation);
    return DORA_OK;
}

int dora_builder_activity(struct builder_operation* operation, uint32_t number, struct dora_allocation* out_activity) {
    struct builder_state* const state = operation->state;
    int status = scratch_begin(operation);
    if (status != DORA_OK) {
        return status;
    }
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field field = *node->field;
    if (!node->rows.has_masked_nulls || node->rows.row_count == 0 ||
        (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
        return DORA_OK;
    }
    if (node->buffer_count != 0 && state->buffers[node->first_buffer].role == BUILDER_BUFFER_VALIDITY_BITS) {
        return DORA_OK;
    }
    size_t bytes = 0;
    status = buffer_bytes(bitmap_words_for_bits(node->rows.row_count), 64, &bytes);
    struct dora_allocation activity = {};
    if (status == DORA_OK) {
        status = dora_pool_allocate(state->storage.pool, bytes, &activity);
    }
    if (status == DORA_OK) {
        bitmap_fill_bytes((uint8_t*)activity.address, activity.capacity_bytes, false);
        status = bind_window(operation, number, 0, node->rows.row_count);
    }
    for (uint64_t row = 0; status == DORA_OK && row < node->rows.row_count; ++row) {
        bool is_null = false;
        status = bound_null(number, operation, row, &is_null);
        if (status == DORA_OK && !is_null) {
            bitmap_set(activity.address, row);
        }
    }
    mutation_finish(operation, false);
    if (status == DORA_OK) {
        *out_activity = activity;
    } else {
        dora_allocation_release(&activity);
    }
    return status;
}

static int window_comparable(uint32_t number, struct builder_operation* mutation, enum dora_compare_op comparison) {
    const uint32_t required =
        comparison == DORA_COMPARE_OP_EQ ? DORA_EXTENSION_FLAG_STORAGE_EQUALITY : DORA_EXTENSION_FLAG_STORAGE_ORDER;
    return (mutation->state->nodes[number].comparison_flags & required) != 0 ? DORA_OK : ENOTSUP;
}

static int bound_compare(struct builder_operation* mutation,
                         uint32_t number,
                         uint64_t left_row,
                         uint64_t right_row,
                         enum dora_compare_op comparison,
                         bool* out_matches) {
    struct builder_state* const state = mutation->state;
    bool left_null = false;
    bool right_null = false;
    int status = bound_null(number, mutation, left_row, &left_null);
    if (status == DORA_OK) {
        status = bound_null(number, mutation, right_row, &right_null);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (left_null || right_null) {
        *out_matches = (comparison == DORA_COMPARE_OP_EQ && left_null == right_null) ||
                       (comparison == DORA_COMPARE_OP_LT && !left_null && right_null);
        return DORA_OK;
    }
    const struct dora_vector_view view = mutation->state->topology[number].view;
    if (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END) {
        uint64_t left = 0;
        uint64_t right = 0;
        uint32_t child_index = 0;
        if (view.layout == DORA_LAYOUT_DICTIONARY) {
            status = dora_vector_view_read_index(&view, left_row, &left);
            if (status == DORA_OK) {
                status = dora_vector_view_read_index(&view, right_row, &right);
            }
        } else {
            child_index = 1;
            left = dora_vector_view_find_run(&view, left_row);
            right = dora_vector_view_find_run(&view, right_row);
        }
        return status == DORA_OK ? bound_compare(mutation,
                                                 builder_child(number, mutation->state, child_index),
                                                 left,
                                                 right,
                                                 comparison,
                                                 out_matches)
                                 : status;
    }
    if (view.layout == DORA_LAYOUT_DENSE_UNION || view.layout == DORA_LAYOUT_SPARSE_UNION) {
        uint64_t left = 0;
        uint64_t right = 0;
        const uint32_t left_member = dora_vector_view_get_member(&view, left_row, &left);
        const uint32_t right_member = dora_vector_view_get_member(&view, right_row, &right);
        if (left_member != right_member) {
            *out_matches = comparison == DORA_COMPARE_OP_LT && left_member < right_member;
            return DORA_OK;
        }
        return bound_compare(mutation,
                             builder_child(number, mutation->state, left_member),
                             left,
                             right,
                             comparison,
                             out_matches);
    }
    if (view.type == DORA_TYPE_STRUCT || view.type == DORA_TYPE_LIST || view.type == DORA_TYPE_FIXED_LIST ||
        view.type == DORA_TYPE_MAP) {
        const struct dora_range left = view.type == DORA_TYPE_STRUCT ? (struct dora_range){left_row, 1}
                                                                     : dora_vector_view_get_elements(&view, left_row);
        const struct dora_range right = view.type == DORA_TYPE_STRUCT ? (struct dora_range){right_row, 1}
                                                                      : dora_vector_view_get_elements(&view, right_row);
        for (uint64_t row = 0; status == DORA_OK && row < left.length && row < right.length; ++row) {
            for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
                 child = state->topology[child].next_sibling_index) {
                bool equal = false;
                if (status == DORA_OK) {
                    status = bound_compare(mutation,
                                           child,
                                           left.offset + row,
                                           right.offset + row,
                                           DORA_COMPARE_OP_EQ,
                                           &equal);
                }
                if (status != DORA_OK) {
                    return status;
                }
                if (!equal) {
                    if (comparison == DORA_COMPARE_OP_EQ) {
                        *out_matches = false;
                        return DORA_OK;
                    }
                    return bound_compare(mutation,
                                         child,
                                         left.offset + row,
                                         right.offset + row,
                                         DORA_COMPARE_OP_LT,
                                         out_matches);
                }
            }
        }
        if (status == DORA_OK) {
            *out_matches = (comparison == DORA_COMPARE_OP_EQ && left.length == right.length) ||
                           (comparison == DORA_COMPARE_OP_LT && left.length < right.length);
        }
        return status;
    }
    if (view.type == DORA_TYPE_BINARY || view.type == DORA_TYPE_UTF8 || view.type == DORA_TYPE_FIXED_BINARY) {
        return type_bytes_matches(dora_vector_view_get_bytes(&view, left_row),
                                  dora_vector_view_get_bytes(&view, right_row),
                                  comparison,
                                  out_matches);
    }
    if (view.type == DORA_TYPE_BOOLEAN) {
        const bool left = dora_vector_view_get_bool(&view, left_row);
        const bool right = dora_vector_view_get_bool(&view, right_row);
        *out_matches =
            (comparison == DORA_COMPARE_OP_EQ && left == right) || (comparison == DORA_COMPARE_OP_LT && !left && right);
        return DORA_OK;
    }
    const uint64_t left_physical = dora_vector_view_get_physical_row(&view, left_row);
    const uint64_t right_physical = dora_vector_view_get_physical_row(&view, right_row);
    switch (view.physical_type) {
        FIXED_CARRIERS(TYPE_ORDER_ROWS_CASE, view.values.data, left_physical, right_physical, comparison, out_matches)
    default:
        return ENOTSUP;
    }
}

static int bound_hash(struct builder_operation* mutation, uint32_t number, uint64_t row, uint64_t* inout_hash) {
    bool is_null = false;
    int status = bound_null(number, mutation, row, &is_null);
    if (status != DORA_OK) {
        return status;
    }
    if (is_null) {
        *inout_hash = hash_value_u8(0, *inout_hash);
        return DORA_OK;
    }
    struct builder_state* const state = mutation->state;
    const struct dora_vector_view* const view = &state->topology[number].view;
    if (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END) {
        uint64_t physical = 0;
        if (view->layout == DORA_LAYOUT_DICTIONARY) {
            status = dora_vector_view_read_index(view, row, &physical);
        } else {
            physical = dora_vector_view_find_run(view, row);
        }
        return status != DORA_OK ? status
                                 : bound_hash(mutation,
                                              builder_child(number, state, view->layout == DORA_LAYOUT_RUN_END ? 1 : 0),
                                              physical,
                                              inout_hash);
    }
    *inout_hash = hash_value_u8(1, *inout_hash);
    if (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION) {
        uint64_t physical = 0;
        const uint32_t member = dora_vector_view_get_member(view, row, &physical);
        *inout_hash = hash_value_u32(member, *inout_hash);
        return bound_hash(mutation, builder_child(number, state, member), physical, inout_hash);
    }
    if (view->type == DORA_TYPE_STRUCT || view->type == DORA_TYPE_LIST || view->type == DORA_TYPE_FIXED_LIST ||
        view->type == DORA_TYPE_MAP) {
        const struct dora_range elements =
            view->type == DORA_TYPE_STRUCT ? (struct dora_range){row, 1} : dora_vector_view_get_elements(view, row);
        *inout_hash = hash_value_u64(elements.length, *inout_hash);
        for (uint64_t element = 0; status == DORA_OK && element < elements.length; ++element) {
            for (uint32_t child = state->topology[number].first_child_index; status == DORA_OK && child != UINT32_MAX;
                 child = state->topology[child].next_sibling_index) {
                status = bound_hash(mutation, child, elements.offset + element, inout_hash);
            }
        }
        return status;
    }
    if (view->type == DORA_TYPE_BINARY || view->type == DORA_TYPE_UTF8 || view->type == DORA_TYPE_FIXED_BINARY) {
        *inout_hash = hash_value_bytes(dora_vector_view_get_bytes(view, row), *inout_hash);
        return DORA_OK;
    }
    if (view->type == DORA_TYPE_BOOLEAN) {
        *inout_hash = hash_value_u8(dora_vector_view_get_bool(view, row) ? 1 : 0, *inout_hash);
        return DORA_OK;
    }
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    switch (view->physical_type) {
// Returns DORA_OK from the enclosing function.
#define KEY_HASH_CASE(TAG, TYPE, SUFFIX)                                                                               \
    case TAG: {                                                                                                        \
        TYPE value = {};                                                                                               \
        memcpy(&value, (const uint8_t*)view->values.data + physical * sizeof(value), sizeof(value));                   \
        *inout_hash = hash_value_##SUFFIX(value, *inout_hash);                                                         \
        return DORA_OK;                                                                                                \
    }
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I8, int8_t, i8)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I16, int16_t, i16)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I32, int32_t, i32)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I64, int64_t, i64)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I128, struct dora_i128, i128)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_I256, struct dora_i256, i256)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_U8, uint8_t, u8)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_U16, uint16_t, u16)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_U32, uint32_t, u32)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_U64, uint64_t, u64)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_U128, struct dora_u128, u128)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_F16, struct dora_f16_bits, f16)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_F32, float, f32)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_F64, double, f64)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME, struct dora_interval_day_time, interval_day_time)
        KEY_HASH_CASE(DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO,
                      struct dora_interval_month_day_nano,
                      interval_month_day_nano)
#undef KEY_HASH_CASE
    default:
        return ENOTSUP;
    }
}

struct builder_key_rows {
    struct builder_operation* mutation;
    uint32_t keys;
};

static int builder_key_hash(void* context, uint64_t row, uint64_t* out_hash) {
    struct builder_key_rows* const rows = context;
    *out_hash = 0;
    return bound_hash(rows->mutation, rows->keys, row, out_hash);
}

static int builder_key_equal(void* context, uint64_t left_row, uint64_t right_row, bool* out_equal) {
    struct builder_key_rows* const rows = context;
    return bound_compare(rows->mutation, rows->keys, left_row, right_row, DORA_COMPARE_OP_EQ, out_equal);
}

static int check_map_keys(uint32_t keys, struct builder_operation* mutation, struct dora_range rows, bool sorted) {
    int status = window_comparable(keys, mutation, sorted ? DORA_COMPARE_OP_LT : DORA_COMPARE_OP_EQ);
    struct dora_keyset set = {};
    if (status == DORA_OK && !sorted) {
        status = dora_keyset_init(&set, mutation->state->storage.pool, rows.length);
    }
    struct builder_key_rows context = {.mutation = mutation, .keys = keys};
    const struct dora_keyset_rows callbacks =
        {.context = &context, .hash = builder_key_hash, .equal = builder_key_equal};
    for (uint64_t row = rows.offset; status == DORA_OK && row < rows.offset + rows.length; ++row) {
        bool is_null = false;
        status = bound_null(keys, mutation, row, &is_null);
        if (status == DORA_OK && is_null) {
            status = EINVAL;
        }
        if (status == DORA_OK && sorted && row != rows.offset) {
            bool ordered = false;
            status = bound_compare(mutation, keys, row - 1, row, DORA_COMPARE_OP_LT, &ordered);
            if (status == DORA_OK && !ordered) {
                status = EILSEQ;
            }
        } else if (status == DORA_OK && !sorted) {
            uint64_t existing = 0;
            bool duplicate = false;
            status = dora_keyset_insert(&set, &callbacks, row, row, &existing, &duplicate);
            if (status == DORA_OK && duplicate) {
                status = EILSEQ;
            }
        }
    }
    dora_keyset_release(&set);
    return status;
}

static void view_frontier(struct builder_state* state,
                          struct builder_operation* mutation,
                          uint32_t number,
                          struct builder_row_state* rows,
                          const struct dora_binary_view_entry* entries,
                          uint64_t count) {
    rows->data_buffer_count = 0;
    rows->last_data_buffer = 0;
    uint64_t fill = 0;
    while (count != 0) {
        const struct dora_binary_view_entry* const entry = &entries[--count];
        if (entry->byte_count <= VIEW_INLINE_BYTE_COUNT) {
            continue;
        }
        rows->data_buffer_count = (uint32_t)entry->payload.reference.buffer_index + 1;
        uint32_t index = state->nodes[number].first_buffer + 2;
        for (uint32_t part = 1; part < rows->data_buffer_count; ++part) {
            index = state->buffers[index].next_buffer;
        }
        rows->last_data_buffer = index;
        fill = (uint64_t)entry->payload.reference.byte_offset + (uint64_t)entry->byte_count;
        break;
    }
    uint32_t index = state->nodes[number].first_buffer + 2;
    for (uint32_t part = 0; index != 0; ++part) {
        struct builder_buffer_state* const buffer = &state->buffers[index];
        uint64_t* const extent = mutation == nullptr ? &buffer->initialized_unit_count
                                                     : &mutation_buffer(mutation, index)->initialized_unit_count;
        const uint64_t units = part + 1 < rows->data_buffer_count    ? *extent
                               : part + 1 == rows->data_buffer_count ? fill
                                                                     : 0;
        if (mutation == nullptr) {
            state->written_byte_count -= (size_t)(*extent - units);
        }
        *extent = units;
        index = buffer->next_buffer;
    }
}

static int truncate(uint32_t number, struct builder_state* state, struct builder_operation* mutation, uint64_t count) {
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows =
        mutation == nullptr ? &state->nodes[number].rows : mutation_rows(mutation, number);
    if (count > rows->row_count) {
        return EINVAL;
    }
    const struct dora_vector_view* const shape = &state->topology[number].view;
    const struct builder_buffer_state* const buffers = state->buffers;
    uint64_t child_rows = 0;
    if (shape->layout == DORA_LAYOUT_RUN_END) {
        const uint32_t ends = builder_child(number, state, 0);
        const uint32_t index = state->nodes[ends].first_buffer + 1;
        const struct builder_buffer_state* const endpoints = &buffers[index];
        if (count != 0) {
            if (mutation != nullptr) {
                const int status = find_run(number, mutation, count - 1, &child_rows);
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                uint64_t limit = state->nodes[ends].rows.closed_row_count;
                while (child_rows < limit) {
                    const uint64_t middle = child_rows + (limit - child_rows) / 2;
                    if (load_offset(endpoints, middle) < count) {
                        child_rows = middle + 1;
                    } else {
                        limit = middle;
                    }
                }
            }
            ++child_rows;
        }
        if (mutation != nullptr) {
            struct builder_row_checkpoint* const checkpoints = state->mutation_nodes;
            const uint64_t changed = child_rows == 0 ? 0 : child_rows - 1;
            if (changed < checkpoints[number].first_changed_run) {
                checkpoints[number].first_changed_run = changed;
            }
        }
        if (child_rows != 0) {
            uint8_t* destination;
            if (mutation != nullptr) {
                const int status = reserve_units(index, mutation, child_rows - 1, child_rows);
                if (status != DORA_OK) {
                    return status;
                }
                destination = candidate_address(mutation, index, child_rows - 1);
            } else {
                destination =
                    (uint8_t*)endpoints->allocation.address + (child_rows - 1) * (endpoints->unit_bit_width / 8);
            }
            store_offset(endpoints->unit_bit_width, destination, count);
        }
    } else if (shape->layout == DORA_LAYOUT_STRUCT || shape->layout == DORA_LAYOUT_SPARSE_UNION) {
        child_rows = count;
    } else if (shape->layout == DORA_LAYOUT_FIXED_LIST) {
        if (mutation != nullptr) {
            if (ckd_mul(&child_rows, count, shape->list_element_count)) {
                return EOVERFLOW;
            }
        } else {
            child_rows = count * shape->list_element_count;
        }
    } else if (shape->layout == DORA_LAYOUT_LIST || shape->layout == DORA_LAYOUT_LARGE_LIST) {
        if (count != 0) {
            if (mutation != nullptr) {
                const int status = bind_offset(mutation, node->first_buffer + 1, count, &child_rows);
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                child_rows = load_offset(&buffers[node->first_buffer + 1], count);
            }
        }
    } else if (shape->layout == DORA_LAYOUT_LIST_VIEW || shape->layout == DORA_LAYOUT_LARGE_LIST_VIEW) {
        if (count != 0) {
            if (mutation != nullptr) {
                const int status = bind_offset(mutation, node->frontier_buffer, count - 1, &child_rows);
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                child_rows = load_offset(&buffers[node->frontier_buffer], count - 1);
            }
        }
        const uint32_t index = node->frontier_buffer;
        uint64_t* const extent = mutation == nullptr ? &state->buffers[index].initialized_unit_count
                                                     : &mutation_buffer(mutation, index)->initialized_unit_count;
        if (mutation == nullptr) {
            state->written_byte_count -= (size_t)(*extent - count) * sizeof(int64_t);
        }
        *extent = count;
    }
    uint64_t dense_ends[UNION_MEMBER_CAPACITY] = {};
    if (shape->layout == DORA_LAYOUT_DENSE_UNION) {
        for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            dense_ends[state->topology[child].child_index] =
                mutation == nullptr ? state->nodes[child].rows.closed_row_count
                                    : mutation_read_rows(mutation, child)->closed_row_count;
        }
        const uint8_t* const address = buffers[node->first_buffer].allocation.address;
        struct dora_bytes ids = {.data = address};
        int status = mutation == nullptr ? DORA_OK : read_units(mutation, node->first_buffer, 0, rows->row_count, &ids);
        for (uint64_t row = count; status == DORA_OK && row < rows->row_count; ++row) {
            const uint32_t member = shape->union_members[(uint8_t)((const int8_t*)ids.data)[row]];
            if (mutation != nullptr && (member >= UNION_MEMBER_CAPACITY || dense_ends[member] == 0)) {
                status = EINVAL;
            } else {
                --dense_ends[member];
            }
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        if (shape->layout == DORA_LAYOUT_DICTIONARY) {
            continue;
        }
        const uint64_t end =
            shape->layout == DORA_LAYOUT_DENSE_UNION ? dense_ends[state->topology[child].child_index] : child_rows;
        const int status = truncate(child, state, mutation, end);
        struct builder_row_state* const child_state =
            mutation == nullptr ? &state->nodes[child].rows : mutation_rows(mutation, child);
        child_state->closed_row_count = end;
        if (status != DORA_OK) {
            return status;
        }
    }
    if (shape->layout == DORA_LAYOUT_BINARY || shape->layout == DORA_LAYOUT_LARGE_BINARY) {
        rows->byte_count = 0;
        if (count != 0) {
            if (mutation != nullptr) {
                const int status = bind_offset(mutation, node->first_buffer + 1, count, &rows->byte_count);
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                rows->byte_count = load_offset(&buffers[node->first_buffer + 1], count);
            }
        }
    } else if (shape->layout == DORA_LAYOUT_BINARY_VIEW) {
        const uint8_t* const address = buffers[node->first_buffer + 1].allocation.address;
        struct dora_bytes entries = {.data = address};
        int status =
            mutation == nullptr ? DORA_OK : read_units(mutation, node->first_buffer + 1, 0, rows->row_count, &entries);
        for (uint64_t row = count; status == DORA_OK && row < rows->row_count; ++row) {
            struct dora_binary_view_entry entry = {};
            memcpy(&entry, entries.data + row * sizeof(entry), sizeof(entry));
            if (entry.byte_count > VIEW_INLINE_BYTE_COUNT) {
                rows->byte_count -= (uint64_t)entry.byte_count;
            }
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    if (node->buffer_count != 0 && buffers[node->first_buffer].role == BUILDER_BUFFER_VALIDITY_BITS) {
        if (buffers[node->first_buffer].allocation.address != nullptr) {
            rows->null_count -= (int64_t)(rows->row_count - count -
                                          dora_bitmap_count_set_range(buffers[node->first_buffer].allocation.address,
                                                                      count,
                                                                      rows->row_count));
        }
    } else if (shape->layout == DORA_LAYOUT_NULL) {
        rows->null_count = (int64_t)count;
    }
    if (shape->layout == DORA_LAYOUT_BINARY_VIEW) {
        const struct dora_binary_view_entry* const entries = buffers[node->first_buffer + 1].allocation.address;
        view_frontier(state, mutation, number, rows, entries, count);
        const uint32_t index = node->first_buffer + 3;
        uint64_t* const extent = mutation == nullptr ? &state->buffers[index].initialized_unit_count
                                                     : &mutation_buffer(mutation, index)->initialized_unit_count;
        if (mutation == nullptr) {
            state->written_byte_count -= (size_t)(*extent - rows->data_buffer_count) * sizeof(int64_t);
        }
        *extent = rows->data_buffer_count;
    }
    for (uint32_t part = 0; part < (shape->layout == DORA_LAYOUT_BINARY_VIEW ? 2U : node->buffer_count); ++part) {
        const uint32_t index = node->first_buffer + part;
        uint64_t* const extent = mutation == nullptr ? &state->buffers[index].initialized_unit_count
                                                     : &mutation_buffer(mutation, index)->initialized_unit_count;
        uint64_t units = buffers[index].role == BUILDER_BUFFER_BYTES ? rows->byte_count : count;
        if ((buffers[index].role == BUILDER_BUFFER_OFFSETS32 || buffers[index].role == BUILDER_BUFFER_OFFSETS64) &&
            shape->layout != DORA_LAYOUT_LIST_VIEW && shape->layout != DORA_LAYOUT_LARGE_LIST_VIEW) {
            ++units;
        }
        if (buffers[index].allocation.address == nullptr) {
            units = 0;
        }
        if (mutation == nullptr) {
            state->written_byte_count -= (size_t)((*extent * buffers[index].unit_bit_width + 7) / 8) -
                                         (size_t)((units * buffers[index].unit_bit_width + 7) / 8);
        }
        *extent = units;
    }
    rows->row_count = count;
    rows->child_row_count = child_rows;
    if (rows->closed_row_count > count) {
        rows->closed_row_count = count;
    }
    return DORA_OK;
}

static int rollback_children(struct builder_state* state, uint32_t number) {
    struct builder_node_state* const nodes = state->nodes;
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const int status = truncate(child, state, nullptr, nodes[child].rows.closed_row_count);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int check_source(uint32_t number,
                        const struct vector_bound_input* input,
                        struct builder_state* state,
                        const struct dora_vector* source,
                        uint32_t source_node) {
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_schema_field* const field = nodes[number].field;
    const struct vector_bound_node* const bound = &input->nodes[source_node];
    const struct dora_vector_view* const view = &bound->view;
    int status;
    if (field->layout == DORA_LAYOUT_RUN_END) {
        struct dora_vector values = dora_vector_borrow(source);
        uint32_t value_node = source_node;
        if (view->layout == DORA_LAYOUT_RUN_END) {
            values = (struct dora_vector){};
            status = dora_vector_get_physical_child(source, 1, &values);
            if (status != DORA_OK) {
                return status;
            }
            value_node = dora_vector_input_child(source_node, input, 1);
        }
        return check_source(builder_child(number, state, 1), input, state, &values, value_node);
    }
    if (field->layout == DORA_LAYOUT_DICTIONARY) {
        if (view->layout != DORA_LAYOUT_DICTIONARY) {
            return EINVAL;
        }
        struct dora_vector domain = {};
        bool equal = false;
        status = dora_vector_get_physical_child(source, 0, &domain);
        if (status == DORA_OK) {
            status = dora_vector_is_equal(&nodes[number].dictionary, &domain, &equal);
        }
        return status != DORA_OK ? status : equal ? DORA_OK : EINVAL;
    }
    if (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END) {
        const uint32_t ordinal = view->layout == DORA_LAYOUT_RUN_END ? 1 : 0;
        struct dora_vector values = {};
        status = dora_vector_get_physical_child(source, ordinal, &values);
        return status != DORA_OK
                   ? status
                   : check_source(number, input, state, &values, dora_vector_input_child(source_node, input, ordinal));
    }
    if (bound->field.child_count != field->child_count ||
        !dora_type_is_equal(&field->type, &bound->field.type, field->child_count)) {
        return EINVAL;
    }
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const uint32_t ordinal = state->topology[child].child_index;
        struct dora_vector child_source = {};
        status = view->layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, ordinal, &child_source)
                                                    : dora_vector_get_physical_child(source, ordinal, &child_source);
        if (status == DORA_OK) {
            status =
                check_source(child, input, state, &child_source, dora_vector_input_child(source_node, input, ordinal));
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int append_map_entries(uint32_t number, struct builder_operation* mutation, uint64_t count) {
    struct builder_state* const state = mutation->state;
    struct builder_node_state* const nodes = state->nodes;
    const uint32_t entries = builder_child(number, state, 0);
    if (entries == UINT32_MAX) {
        return EINVAL;
    }
    const uint32_t keys = builder_child(entries, state, 0);
    const uint32_t values = builder_child(entries, state, 1);
    if (keys == UINT32_MAX || values == UINT32_MAX) {
        return EINVAL;
    }
    const struct dora_schema_field bound_field = *nodes[entries].field;
    uint64_t end = 0;
    int status = append_limit(entries, state, mutation_rows(mutation, entries)->closed_row_count, count, &end);
    if (status != DORA_OK) {
        return status;
    }
    if (mutation_rows(mutation, keys)->row_count != end || mutation_rows(mutation, values)->row_count != end ||
        (mutation_rows(mutation, entries)->row_count != mutation_rows(mutation, entries)->closed_row_count &&
         mutation_rows(mutation, entries)->row_count != end)) {
        return EINVAL;
    }
    while (status == DORA_OK && mutation_rows(mutation, entries)->row_count < end) {
        status = append_validity(mutation, entries, &bound_field, end, true, false, false);
        if (status == DORA_OK) {
            ++mutation_rows(mutation, entries)->row_count;
        }
    }
    if (status == DORA_OK) {
        mutation_rows(mutation, keys)->closed_row_count = end;
        mutation_rows(mutation, values)->closed_row_count = end;
    }
    return status;
}

struct builder_container {
    struct dora_schema_field field;
    uint64_t end;
    uint64_t member_row;
    uint32_t selected;
};

static int container_prepare(uint32_t number,
                             struct builder_operation* mutation,
                             enum dora_layout expected,
                             struct dora_range elements,
                             int8_t type_id,
                             struct builder_container* out_container) {
    struct builder_state* const state = mutation->state;
    struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field field = *node->field;
    if (field.layout != expected) {
        return EINVAL;
    }
    uint64_t end = 0;
    int status = append_limit(number, state, mutation_rows(mutation, number)->row_count, 1, &end);
    if (status != DORA_OK) {
        return status;
    }
    const bool list = expected == DORA_LAYOUT_LIST || expected == DORA_LAYOUT_LARGE_LIST;
    const bool dense = expected == DORA_LAYOUT_DENSE_UNION;
    const bool sparse = expected == DORA_LAYOUT_SPARSE_UNION;
    if (list && field.type.id == DORA_TYPE_MAP) {
        status = append_map_entries(number, mutation, elements.length);
        if (status != DORA_OK) {
            return status;
        }
    }
    uint32_t selected = 0;
    if (dense || sparse) {
        while (selected < field.child_count && field.type.parameters.type_codes[selected] != type_id) {
            ++selected;
        }
        if (selected == field.child_count) {
            return EINVAL;
        }
    }
    *out_container = (struct builder_container){.field = field, .end = end, .selected = selected};
    return DORA_OK;
}

static int container_child(uint32_t number,
                           struct builder_operation* mutation,
                           struct builder_container* container,
                           struct dora_range elements,
                           uint32_t ordinal,
                           uint32_t* out_child,
                           bool* out_fill) {
    struct builder_state* const state = mutation->state;
    const struct dora_schema_field* const field = &container->field;
    const enum dora_layout expected = field->layout;
    const bool list = expected == DORA_LAYOUT_LIST || expected == DORA_LAYOUT_LARGE_LIST;
    const bool list_view = expected == DORA_LAYOUT_LIST_VIEW || expected == DORA_LAYOUT_LARGE_LIST_VIEW;
    const bool dense = expected == DORA_LAYOUT_DENSE_UNION;
    const bool sparse = expected == DORA_LAYOUT_SPARSE_UNION;
    const uint32_t child = builder_child(number, state, ordinal);
    const uint64_t added = mutation_rows(mutation, child)->row_count - mutation_rows(mutation, child)->closed_row_count;
    const bool fill = sparse && ordinal != container->selected && added == 0;
    if (!fill) {
        if (dense || sparse) {
            if (added != (ordinal == container->selected || sparse ? 1U : 0U)) {
                return EINVAL;
            }
        } else if ((expected == DORA_LAYOUT_STRUCT && added != 1) ||
                   (expected == DORA_LAYOUT_FIXED_LIST && added != field->type.parameters.list_element_count) ||
                   (list && added != elements.length) ||
                   (list_view && (elements.offset > mutation_rows(mutation, child)->row_count ||
                                  elements.length > mutation_rows(mutation, child)->row_count - elements.offset))) {
            return EINVAL;
        }
    }
    if ((dense || sparse) && ordinal == container->selected) {
        container->member_row = mutation_rows(mutation, child)->row_count - 1;
    }
    *out_child = child;
    *out_fill = fill;
    return DORA_OK;
}

static int container_publish(uint32_t number,
                             struct builder_operation* mutation,
                             const struct builder_container* container,
                             struct dora_range elements,
                             int8_t type_id,
                             bool valid,
                             bool masked,
                             bool are_rows_admitted) {
    struct builder_state* const state = mutation->state;
    struct builder_node_state* const node = &state->nodes[number];
    struct builder_buffer_state* const buffers = state->buffers;
    const struct dora_schema_field field = container->field;
    const uint64_t end = container->end;
    const uint64_t member_row = container->member_row;
    const enum dora_layout expected = field.layout;
    const bool list = expected == DORA_LAYOUT_LIST || expected == DORA_LAYOUT_LARGE_LIST;
    const bool list_view = expected == DORA_LAYOUT_LIST_VIEW || expected == DORA_LAYOUT_LARGE_LIST_VIEW;
    const bool dense = expected == DORA_LAYOUT_DENSE_UNION;
    const bool sparse = expected == DORA_LAYOUT_SPARSE_UNION;
    int status;
    if (list || list_view) {
        const uint64_t start = list ? mutation_rows(mutation, number)->child_row_count : elements.offset;
        uint64_t finish = 0;
        if (ckd_add(&finish, start, elements.length) || finish > INT64_MAX ||
            (buffers[node->first_buffer + 1].unit_bit_width == 32 && finish > INT32_MAX)) {
            return EOVERFLOW;
        }
        const uint64_t committed_offsets = mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count;
        status = reserve_units(node->first_buffer + 1,
                               mutation,
                               mutation_rows(mutation, number)->row_count + (!list_view && committed_offsets != 0),
                               end + !list_view);
        if (status == DORA_OK && list_view) {
            status = reserve_units(node->first_buffer + 2, mutation, mutation_rows(mutation, number)->row_count, end);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (list_view) {
            status = reserve_units(node->frontier_buffer, mutation, mutation_rows(mutation, number)->row_count, end);
            if (status != DORA_OK) {
                return status;
            }
            store_offset(state->buffers[node->frontier_buffer].unit_bit_width,
                         candidate_address(mutation, node->frontier_buffer, mutation_rows(mutation, number)->row_count),
                         mutation_rows(mutation, builder_child(number, state, 0))->row_count);
            mutation_buffer(mutation, node->frontier_buffer)->initialized_unit_count = end;
            store_offset(
                state->buffers[node->first_buffer + 1].unit_bit_width,
                candidate_address(mutation, node->first_buffer + 1, mutation_rows(mutation, number)->row_count),
                start);
            store_offset(
                state->buffers[node->first_buffer + 2].unit_bit_width,
                candidate_address(mutation, node->first_buffer + 2, mutation_rows(mutation, number)->row_count),
                elements.length);
            mutation_buffer(mutation, node->first_buffer + 2)->initialized_unit_count = end;
        } else {
            if (committed_offsets == 0) {
                store_offset(state->buffers[node->first_buffer + 1].unit_bit_width,
                             candidate_address(mutation, node->first_buffer + 1, 0),
                             0);
            }
            store_offset(state->buffers[node->first_buffer + 1].unit_bit_width,
                         candidate_address(mutation, node->first_buffer + 1, end),
                         finish);
        }
        mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = end + !list_view;
        mutation_rows(mutation, number)->child_row_count = finish;
    } else if (dense || sparse) {
        if (dense && member_row > INT32_MAX) {
            return EOVERFLOW;
        }
        status = reserve_units(node->first_buffer, mutation, mutation_rows(mutation, number)->row_count, end);
        if (status == DORA_OK && dense) {
            status = reserve_units(node->first_buffer + 1, mutation, mutation_rows(mutation, number)->row_count, end);
        }
        if (status != DORA_OK) {
            return status;
        }
        *(int8_t*)candidate_address(mutation, node->first_buffer, mutation_rows(mutation, number)->row_count) = type_id;
        mutation_buffer(mutation, node->first_buffer)->initialized_unit_count = end;
        if (dense) {
            store_offset(
                32,
                candidate_address(mutation, node->first_buffer + 1, mutation_rows(mutation, number)->row_count),
                member_row);
            mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = end;
        }
    }
    if (!dense && !sparse) {
        status = append_validity(mutation, number, &field, end, valid, masked, are_rows_admitted);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (masked && (dense || sparse)) {
        status = note_inactive(mutation, number, mutation_rows(mutation, number)->row_count, 1, nullptr);
        if (status != DORA_OK) {
            return status;
        }
    }
    mutation_rows(mutation, number)->row_count = end;
    mutation_rows(mutation, number)->has_masked_nulls =
        mutation_rows(mutation, number)->has_masked_nulls || (!valid && masked);
    for (uint32_t ordinal = 0; ordinal < field.child_count; ++ordinal) {
        const uint32_t child = builder_child(number, state, ordinal);
        mutation_rows(mutation, child)->closed_row_count = mutation_rows(mutation, child)->row_count;
    }
    return DORA_OK;
}

struct run_end_rows {
    uint64_t length;
    bool is_masked;
};

static int append_run_end(uint32_t number, struct builder_operation* mutation, struct run_end_rows run) {
    if (run.length == 0) {
        return DORA_OK;
    }
    struct builder_state* const state = mutation->state;
    const uint32_t ends = builder_child(number, state, 0);
    const uint32_t values = builder_child(number, state, 1);
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    struct builder_row_checkpoint* const checkpoints = state->mutation_nodes;
    const uint64_t runs = mutation_read_rows(mutation, ends)->closed_row_count;
    const uint64_t changed = runs == 0 ? 0 : runs - 1;
    if (changed < checkpoints[number].first_changed_run) {
        checkpoints[number].first_changed_run = changed;
    }
    uint64_t end = 0;
    int status = append_limit(number, state, rows->row_count, run.length, &end);
    const struct dora_vector_view* const end_type = &state->topology[ends].view;
    const uint64_t maximum = end_type->physical_type == DORA_PHYSICAL_TYPE_I16   ? INT16_MAX
                             : end_type->physical_type == DORA_PHYSICAL_TYPE_I32 ? INT32_MAX
                                                                                 : INT64_MAX;
    if (status != DORA_OK || end > maximum) {
        return status != DORA_OK ? status : EOVERFLOW;
    }
    if (status == DORA_OK && run.is_masked) {
        status = note_inactive(mutation, number, rows->row_count, run.length, nullptr);
    }
    if (status != DORA_OK) {
        return status;
    }
    union decoded_row_value value = {};
    if (end_type->physical_type == DORA_PHYSICAL_TYPE_I16) {
        value.i16 = (int16_t)end;
    } else if (end_type->physical_type == DORA_PHYSICAL_TYPE_I32) {
        value.i32 = (int32_t)end;
    } else {
        value.i64 = (int64_t)end;
    }
    const struct dora_schema_field field = *state->nodes[ends].field;
    if (status == DORA_OK) {
        status = append_fixed(
            mutation,
            ends,
            &field,
            &(struct builder_input){.values = &value, .row_count = 1, .physical = end_type->physical_type});
    }
    if (status == DORA_OK) {
        rows->row_count = end;
        rows->has_masked_nulls = rows->has_masked_nulls || run.is_masked;
        mutation_rows(mutation, ends)->closed_row_count = mutation_rows(mutation, ends)->row_count;
        mutation_rows(mutation, values)->closed_row_count = mutation_rows(mutation, values)->row_count;
    }
    return status;
}

// Null-row entry admits the parent; input entry admits the values child.
enum builder_append_phase : uint8_t {
    BUILDER_APPEND_INPUT,
    BUILDER_APPEND_NULL_ROWS
};

static int append_input(struct builder_operation* mutation,
                        uint32_t number,
                        const struct builder_input* input,
                        enum builder_append_phase phase) {
    struct builder_state* const state = mutation->state;
    const struct dora_schema_field field = *state->nodes[number].field;
    if (phase == BUILDER_APPEND_NULL_ROWS || (input->is_all_null && field.layout != DORA_LAYOUT_RUN_END)) {
        mutation_rows(mutation, number)->has_masked_nulls =
            mutation_rows(mutation, number)->has_masked_nulls || (input->is_masked && input->row_count != 0);
        if (!input->is_masked && !input->are_rows_admitted && input->row_count != 0 &&
            (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
            return EINVAL;
        }
    }
    if (field.layout == DORA_LAYOUT_RUN_END) {
        const uint32_t child = builder_child(number, state, 1);
        if (input->is_all_null && input->row_count != 0) {
            const int status = append_input(mutation,
                                            child,
                                            &(struct builder_input){
                                                .row_count = 1,
                                                .is_all_null = true,
                                                .is_masked = input->is_masked,
                                                .are_rows_admitted = input->are_rows_admitted
                                            },
                                            BUILDER_APPEND_NULL_ROWS);
            return status != DORA_OK
                       ? status
                       : append_run_end(
                             number,
                             mutation,
                             (struct run_end_rows){.length = input->row_count, .is_masked = input->is_masked});
        }
        uint32_t terminal = child;
        while (state->topology[terminal].view.layout == DORA_LAYOUT_RUN_END) {
            terminal = builder_child(terminal, state, 1);
        }
        const size_t width = input->is_byte_input ? sizeof(struct dora_bytes)
                             : input->is_boolean  ? sizeof(bool)
                                                  : state->topology[terminal].view.byte_width;
        size_t input_bytes = 0;
        uint64_t bit_end = 0;
        if (ckd_mul(&input_bytes, input->row_count, width) ||
            (input->bits != nullptr && ckd_add(&bit_end, input->bit_offset, input->row_count)) ||
            (input->validity_bits != nullptr && ckd_add(&bit_end, input->validity_bit_offset, input->row_count))) {
            return EOVERFLOW;
        }
        for (uint64_t row = 0; row < input->row_count; ++row) {
            struct builder_input value = *input;
            value.row_count = 1;
            value.values = input->values == nullptr ? nullptr : (const uint8_t*)input->values + row * width;
            value.validity = input->validity == nullptr ? nullptr : input->validity + row;
            value.bit_offset += row;
            value.validity_bit_offset += row;
            int status = append_input(mutation, child, &value, BUILDER_APPEND_INPUT);
            if (status == DORA_OK) {
                status =
                    append_run_end(number, mutation, (struct run_end_rows){.length = 1, .is_masked = input->is_masked});
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        return DORA_OK;
    }
    if (phase != BUILDER_APPEND_NULL_ROWS && !input->is_all_null) {
        return input->is_byte_input ? append_bytes(mutation, number, &field, input)
                                    : append_fixed(mutation, number, &field, input);
    }
    const uint64_t count = input->row_count;
    const bool masked = input->is_masked;
    const bool are_rows_admitted = input->are_rows_admitted;
    int status;
    if (field.layout == DORA_LAYOUT_NULL) {
        uint64_t end = 0;
        status = append_limit(number, state, mutation_rows(mutation, number)->row_count, count, &end);
        if (status == DORA_OK) {
            mutation_rows(mutation, number)->row_count = end;
            mutation_rows(mutation, number)->null_count = (int64_t)end;
        }
        return status;
    }
    if (field.layout == DORA_LAYOUT_FIXED || field.layout == DORA_LAYOUT_BITS ||
        field.layout == DORA_LAYOUT_DICTIONARY) {
        if (field.type.id == DORA_TYPE_FIXED_BINARY && field.layout != DORA_LAYOUT_DICTIONARY) {
            return append_bytes(mutation,
                                number,
                                &field,
                                &(struct builder_input){
                                    .row_count = count,
                                    .is_all_null = true,
                                    .is_masked = masked,
                                    .are_rows_admitted = are_rows_admitted,
                                    .is_byte_input = true
                                });
        }
        enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
        const struct dora_logical_type type =
            field.layout == DORA_LAYOUT_DICTIONARY ? (struct dora_logical_type){.id = field.index_type} : field.type;
        status = type_physical(&type, &physical);
        return status != DORA_OK ? status
                                 : append_fixed(mutation,
                                                number,
                                                &field,
                                                &(struct builder_input){
                                                    .row_count = count,
                                                    .is_all_null = true,
                                                    .is_masked = masked,
                                                    .are_rows_admitted = are_rows_admitted,
                                                    .physical = physical,
                                                    .is_boolean = field.layout == DORA_LAYOUT_BITS
                                                });
    }
    if (field.type.id == DORA_TYPE_BINARY || field.type.id == DORA_TYPE_UTF8) {
        return append_bytes(mutation,
                            number,
                            &field,
                            &(struct builder_input){
                                .row_count = count,
                                .is_all_null = true,
                                .is_masked = masked,
                                .are_rows_admitted = are_rows_admitted,
                                .is_byte_input = true
                            });
    }
    for (uint64_t row = 0; row < count; ++row) {
        if (field.layout == DORA_LAYOUT_STRUCT || field.layout == DORA_LAYOUT_FIXED_LIST ||
            field.layout == DORA_LAYOUT_SPARSE_UNION || field.layout == DORA_LAYOUT_DENSE_UNION) {
            const uint32_t children = field.layout == DORA_LAYOUT_DENSE_UNION ? 1 : field.child_count;
            for (uint32_t ordinal = 0; ordinal < children; ++ordinal) {
                status = append_input(mutation,
                                      builder_child(number, state, ordinal),
                                      &(struct builder_input){
                                          .row_count = field.layout == DORA_LAYOUT_FIXED_LIST
                                                           ? field.type.parameters.list_element_count
                                                           : 1,
                                          .is_all_null = true,
                                          .is_masked = true,
                                          .are_rows_admitted = are_rows_admitted
                                      },
                                      BUILDER_APPEND_NULL_ROWS);
                if (status != DORA_OK) {
                    return status;
                }
            }
        }
        const int8_t type_id = field.type.id == DORA_TYPE_UNION ? field.type.parameters.type_codes[0] : 0;
        struct builder_container container = {};
        status = container_prepare(number, mutation, field.layout, (struct dora_range){}, type_id, &container);
        if (status != DORA_OK) {
            return status;
        }
        for (uint32_t ordinal = 0; ordinal < container.field.child_count; ++ordinal) {
            uint32_t child = 0;
            bool fill = false;
            status = container_child(number, mutation, &container, (struct dora_range){}, ordinal, &child, &fill);
            if (status == DORA_OK && fill) {
                status = append_input(mutation,
                                      child,
                                      &(struct builder_input){
                                          .row_count = 1,
                                          .is_all_null = true,
                                          .is_masked = true,
                                          .are_rows_admitted = are_rows_admitted
                                      },
                                      BUILDER_APPEND_NULL_ROWS);
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        status = container_publish(number,
                                   mutation,
                                   &container,
                                   (struct dora_range){},
                                   type_id,
                                   false,
                                   masked,
                                   are_rows_admitted);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int append_container(uint32_t number,
                            struct builder_operation* mutation,
                            enum dora_layout expected,
                            struct dora_range elements,
                            int8_t type_id,
                            bool valid,
                            bool masked,
                            bool are_rows_admitted) {
    int status;
    struct builder_container container = {};
    status = container_prepare(number, mutation, expected, elements, type_id, &container);
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t ordinal = 0; ordinal < container.field.child_count; ++ordinal) {
        uint32_t child = 0;
        bool fill = false;
        status = container_child(number, mutation, &container, elements, ordinal, &child, &fill);
        if (status == DORA_OK && fill) {
            status = append_input(mutation,
                                  child,
                                  &(struct builder_input){
                                      .row_count = 1,
                                      .is_all_null = true,
                                      .is_masked = true,
                                      .are_rows_admitted = are_rows_admitted
                                  },
                                  BUILDER_APPEND_NULL_ROWS);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    status = container_publish(number, mutation, &container, elements, type_id, valid, masked, are_rows_admitted);
    return status;
}

static int own_window(struct builder_operation* mutation, uint32_t number) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct dora_vector_view* const view = &state->topology[number].view;
    if (bound_domains(state)[number].handle.slot_index != 0) {
        return DORA_OK;
    }
    for (uint32_t part = 0; part < node->buffer_count; ++part) {
        const uint32_t index = node->first_buffer + part;
        struct dora_view_buffer* const span = role_span(view, &bound_data(state)[number], state->buffers[index].role);
        if (span == nullptr) {
            continue;
        }
        if (span->data != nullptr) {
            const int status = materialize_part(mutation, index, span);
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    if (view->layout == DORA_LAYOUT_BINARY_VIEW) {
        uint32_t index = node->first_buffer + 2;
        struct dora_view_buffer* const data = state->mutation_buffers[node->first_buffer + 3].bound.address;
        for (uint32_t part = 0; part < view->data_buffer_count; ++part) {
            const int status = materialize_part(mutation, index, &data[part]);
            if (status != DORA_OK) {
                return status;
            }
            index = state->buffers[index].next_buffer;
        }
    }
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const int status = own_window(mutation, child);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (view->layout == DORA_LAYOUT_RUN_END) {
        view->values = state->topology[state->topology[number].first_child_index].view.values;
    }
    return DORA_OK;
}

static int append_bound_rows(struct builder_operation* mutation, uint32_t number, uint64_t first, uint64_t count) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_vector_view view = state->topology[number].view;
    if (first > view.length || count > view.length - first) {
        return EINVAL;
    }
    const struct dora_schema_field field = *node->field;
    if (view.layout == DORA_LAYOUT_NULL) {
        return append_input(mutation,
                            number,
                            &(struct builder_input){
                                .row_count = count,
                                .is_all_null = true,
                                .is_masked = false,
                                .are_rows_admitted = true
                            },
                            BUILDER_APPEND_NULL_ROWS);
    }
    if (view.layout == DORA_LAYOUT_FIXED || view.layout == DORA_LAYOUT_BITS || view.layout == DORA_LAYOUT_DICTIONARY) {
        if (view.type != DORA_TYPE_FIXED_BINARY || view.layout == DORA_LAYOUT_DICTIONARY) {
            return append_fixed(
                mutation,
                number,
                &field,
                &(struct builder_input){
                    .values = view.layout == DORA_LAYOUT_BITS
                                  ? nullptr
                                  : (const uint8_t*)view.values.data + (view.offset + first) * view.byte_width,
                    .bits = view.layout == DORA_LAYOUT_BITS ? view.values.data : nullptr,
                    .bit_offset = view.offset + first,
                    .validity_bits = view.validity.data,
                    .validity_bit_offset = view.validity_bit_offset + first,
                    .row_count = count,
                    .physical = view.physical_type,
                    .is_boolean = view.layout == DORA_LAYOUT_BITS,
                    .are_rows_admitted = true
                });
        }
    }
    if (view.layout != DORA_LAYOUT_RUN_END &&
        (view.type == DORA_TYPE_UTF8 || view.type == DORA_TYPE_BINARY || view.type == DORA_TYPE_FIXED_BINARY)) {
        return append_bytes(mutation,
                            number,
                            &field,
                            &(struct builder_input){
                                .view = &view,
                                .first = first,
                                .row_count = count,
                                .is_byte_input = true,
                                .are_rows_admitted = true
                            });
    }
    if (view.layout == DORA_LAYOUT_RUN_END) {
        struct dora_run_cursor cursor = {.row = first};
        const uint32_t values = builder_child(number, state, 1);
        while (cursor.row < first + count) {
            uint64_t value_row = 0;
            struct dora_range run = dora_vector_view_next_run(&view, &cursor, &value_row);
            if (run.length == 0) {
                return EINVAL;
            }
            if (run.length > first + count - run.offset) {
                run.length = first + count - run.offset;
            }
            int status = append_bound_rows(mutation, values, value_row, 1);
            if (status == DORA_OK) {
                status = append_run_end(number, mutation, (struct run_end_rows){.length = run.length});
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        return DORA_OK;
    }
    for (uint64_t row = first; row < first + count; ++row) {
        int status;
        const bool valid = !dora_vector_view_is_null(&view, row);
        if (!valid && view.layout == DORA_LAYOUT_FIXED_LIST) {
            status = append_input(mutation,
                                  number,
                                  &(struct builder_input){
                                      .row_count = 1,
                                      .is_all_null = true,
                                      .is_masked = false,
                                      .are_rows_admitted = true
                                  },
                                  BUILDER_APPEND_NULL_ROWS);
            if (status != DORA_OK) {
                return status;
            }
            continue;
        }
        if (view.layout == DORA_LAYOUT_STRUCT || view.layout == DORA_LAYOUT_SPARSE_UNION) {
            for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
                 child = state->topology[child].next_sibling_index) {
                status = append_bound_rows(mutation, child, row, 1);
                if (status != DORA_OK) {
                    return status;
                }
            }
        } else if (view.layout == DORA_LAYOUT_DENSE_UNION) {
            uint64_t member_row = 0;
            const uint32_t member = dora_vector_view_get_member(&view, row, &member_row);
            status = append_bound_rows(mutation, builder_child(number, state, member), member_row, 1);
            if (status != DORA_OK) {
                return status;
            }
        } else {
            const struct dora_range elements = dora_vector_view_get_elements(&view, row);
            status = append_bound_rows(mutation,
                                       state->topology[number].first_child_index,
                                       elements.offset,
                                       elements.length);
            if (status != DORA_OK) {
                return status;
            }
        }
        struct dora_range elements = {};
        if (view.type == DORA_TYPE_LIST || view.type == DORA_TYPE_MAP) {
            const struct dora_range source = dora_vector_view_get_elements(&view, row);
            elements.length = source.length;
            if (view.layout == DORA_LAYOUT_LIST_VIEW || view.layout == DORA_LAYOUT_LARGE_LIST_VIEW) {
                elements.offset =
                    mutation_rows(mutation, state->topology[number].first_child_index)->row_count - source.length;
            }
        }
        const int8_t type_id = view.type == DORA_TYPE_UNION ? dora_vector_view_get_type_id(&view, row) : 0;
        status = append_container(number, mutation, view.layout, elements, type_id, valid, false, true);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int
mark_nulls(uint32_t number, struct builder_operation* mutation, struct dora_range range, const uint8_t* selected) {
    uint64_t first = 0;
    while (first < range.length && !dora_bitmap_test(selected, first)) {
        ++first;
    }
    if (first == range.length) {
        return DORA_OK;
    }
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    if (range.offset > rows->row_count || range.length > rows->row_count - range.offset) {
        return EINVAL;
    }
    int status = note_inactive(mutation, number, range.offset, range.length, selected);
    if (status != DORA_OK) {
        return status;
    }
    rows->has_masked_nulls = true;
    const enum dora_layout layout = state->topology[number].view.layout;
    if (layout == DORA_LAYOUT_NULL) {
        return DORA_OK;
    }
    if (layout == DORA_LAYOUT_RUN_END) {
        const uint64_t begin = range.offset + first;
        const uint64_t length = rows->row_count;
        const uint64_t closed = rows->closed_row_count;
        status = bind_window(mutation, number, begin, length);
        if (status != DORA_OK) {
            goto run_end_done;
        }
        status = own_window(mutation, number);
        if (status != DORA_OK) {
            goto run_end_done;
        }
        status = truncate(number, state, mutation, begin);
        if (status != DORA_OK) {
            goto run_end_done;
        }
        for (uint64_t row = first; status == DORA_OK && row < range.length;) {
            const bool clear = dora_bitmap_test(selected, row);
            uint64_t end = row + 1;
            while (end < range.length && dora_bitmap_test(selected, end) == clear) {
                ++end;
            }
            status = clear ? append_input(mutation,
                                          number,
                                          &(struct builder_input){
                                              .row_count = end - row,
                                              .is_all_null = true,
                                              .is_masked = true,
                                              .are_rows_admitted = false
                                          },
                                          BUILDER_APPEND_NULL_ROWS)
                           : append_bound_rows(mutation, number, row - first, end - row);
            row = end;
        }
        if (status == DORA_OK) {
            status = append_bound_rows(mutation, number, range.length - first, length - range.offset - range.length);
        }
        if (status == DORA_OK) {
            rows->closed_row_count = closed;
        }
run_end_done:
        window_release(mutation);
        return status;
    }
    if (layout == DORA_LAYOUT_DENSE_UNION || layout == DORA_LAYOUT_SPARSE_UNION) {
        const uint8_t* const union_members = state->topology[number].view.union_members;
        struct dora_bytes ids = {};
        struct dora_bytes offsets = {};
        status = read_units(mutation, node->first_buffer, range.offset, range.offset + range.length, &ids);
        if (status == DORA_OK && layout == DORA_LAYOUT_DENSE_UNION) {
            status = read_units(mutation, node->first_buffer + 1, range.offset, range.offset + range.length, &offsets);
        }
        uint64_t starts[UNION_MEMBER_CAPACITY];
        uint64_t ends[UNION_MEMBER_CAPACITY] = {};
        struct dora_allocation masks[UNION_MEMBER_CAPACITY] = {};
        for (uint32_t member = 0; member < UNION_MEMBER_CAPACITY; ++member) {
            starts[member] = UINT64_MAX;
        }
        for (uint64_t row = 0; status == DORA_OK && row < range.length; ++row) {
            if (!dora_bitmap_test(selected, row)) {
                continue;
            }
            const uint32_t ordinal = union_members[(uint8_t)((const int8_t*)ids.data)[row]];
            const uint64_t member = layout == DORA_LAYOUT_DENSE_UNION
                                        ? span_offset(32, &(struct dora_view_buffer){.data = offsets.data}, row)
                                        : range.offset + row;
            if (ordinal >= UNION_MEMBER_CAPACITY || builder_child(number, state, ordinal) == UINT32_MAX ||
                member == UINT64_MAX) {
                status = EINVAL;
            } else {
                if (member < starts[ordinal]) {
                    starts[ordinal] = member;
                }
                if (member + 1 > ends[ordinal]) {
                    ends[ordinal] = member + 1;
                }
            }
        }
        for (uint32_t member = 0; status == DORA_OK && member < UNION_MEMBER_CAPACITY; ++member) {
            if (starts[member] == UINT64_MAX) {
                continue;
            }
            size_t bytes = 0;
            status = buffer_bytes(bitmap_words_for_bits(ends[member] - starts[member]), 64, &bytes);
            if (status == DORA_OK) {
                status = dora_pool_allocate(state->storage.pool, bytes, &masks[member]);
            }
            if (status == DORA_OK) {
                bitmap_fill_bytes((uint8_t*)masks[member].address, bytes, false);
            }
        }
        for (uint64_t row = 0; status == DORA_OK && row < range.length; ++row) {
            if (dora_bitmap_test(selected, row)) {
                const uint32_t ordinal = union_members[(uint8_t)((const int8_t*)ids.data)[row]];
                const uint64_t member = layout == DORA_LAYOUT_DENSE_UNION
                                            ? span_offset(32, &(struct dora_view_buffer){.data = offsets.data}, row)
                                            : range.offset + row;
                bitmap_set(masks[ordinal].address, member - starts[ordinal]);
            }
        }
        for (uint32_t child = state->topology[number].first_child_index; status == DORA_OK && child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            const uint32_t member = state->topology[child].child_index;
            if (starts[member] != UINT64_MAX) {
                status =
                    mark_nulls(child,
                               mutation,
                               (struct dora_range){.offset = starts[member], .length = ends[member] - starts[member]},
                               masks[member].address);
            }
        }
        for (uint32_t member = 0; member < UNION_MEMBER_CAPACITY; ++member) {
            dora_allocation_release(&masks[member]);
        }
        return status;
    }
    if (node->buffer_count == 0 || state->buffers[node->first_buffer].role != BUILDER_BUFFER_VALIDITY_BITS) {
        return EINVAL;
    }
    const bool absent = state->buffers[node->first_buffer].allocation.address == nullptr;
    status = reserve_units(node->first_buffer,
                           mutation,
                           absent ? 0 : range.offset,
                           absent ? rows->row_count : range.offset + range.length);
    if (status == DORA_OK && absent) {
        status = materialize_validity(mutation, number, rows->row_count);
        // Clearing published rows requires materialized validity.
        if (status == DORA_OK) {
            status = reserve_units(node->first_buffer, mutation, range.offset, range.offset + range.length);
        }
    }

    uint64_t* const words = candidate_bit_words(mutation, node->first_buffer);
    if (status != DORA_OK || words == nullptr) {
        return status != DORA_OK ? status : EINVAL;
    }
    for (uint64_t row = 0; row < range.length; ++row) {
        if (dora_bitmap_test(selected, row) && dora_bitmap_test((const uint8_t*)words, range.offset + row)) {
            ++rows->null_count;
            bitmap_clear(words, range.offset + row);
        }
    }
    mutation_buffer(mutation, node->first_buffer)->initialized_unit_count = rows->row_count;
    if (layout == DORA_LAYOUT_STRUCT) {
        for (uint32_t child = state->topology[number].first_child_index; status == DORA_OK && child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            status = mark_nulls(child, mutation, range, selected);
        }
    } else if (layout == DORA_LAYOUT_FIXED_LIST) {
        const uint32_t list_element_count = state->topology[number].view.list_element_count;
        struct dora_range children = {};
        size_t bytes = 0;
        if (ckd_mul(&children.offset, range.offset, list_element_count) ||
            ckd_mul(&children.length, range.length, list_element_count)) {
            return EOVERFLOW;
        }
        status = buffer_bytes(bitmap_words_for_bits(children.length), 64, &bytes);
        struct dora_allocation expanded = {};
        if (status == DORA_OK) {
            status = dora_pool_allocate(state->storage.pool, bytes, &expanded);
        }
        if (status == DORA_OK && bytes != 0) {
            bitmap_fill_bytes((uint8_t*)expanded.address, bytes, false);
            for (uint64_t row = 0; row < range.length; ++row) {
                if (dora_bitmap_test(selected, row)) {
                    dora_bitmap_set_range(expanded.address,
                                          row * list_element_count,
                                          (row + 1) * list_element_count,
                                          true);
                }
            }
            status = mark_nulls(state->topology[number].first_child_index, mutation, children, expanded.address);
        }
        dora_allocation_release(&expanded);
    }
    return status;
}

static int normalize_runs(struct builder_operation* mutation, uint32_t number) {
    struct builder_state* const state = mutation->state;
    struct builder_row_checkpoint* const checkpoints = state->mutation_nodes;
    const bool encoded = state->topology[number].view.layout == DORA_LAYOUT_RUN_END;
    const uint32_t ends = encoded ? builder_child(number, state, 0) : UINT32_MAX;
    const uint64_t runs = encoded ? mutation_read_rows(mutation, ends)->closed_row_count : 0;
    if (encoded && checkpoints[number].is_active && checkpoints[number].first_changed_run != UINT64_MAX &&
        runs - checkpoints[number].first_changed_run > 1) {
        struct builder_row_state* const rows = &checkpoints[number].rows;
        const uint64_t first = checkpoints[number].first_changed_run;
        const uint64_t count = runs - first;
        const uint32_t values = builder_child(number, state, 1);
        const uint32_t end_buffer = state->nodes[ends].first_buffer + 1;
        const uint64_t closed = rows->closed_row_count;
        uint64_t prior = 0;
        if (first != 0) {
            const int status = bind_offset(mutation, end_buffer, first - 1, &prior);
            if (status != DORA_OK) {
                window_release(mutation);
                return status;
            }
        }
        int status = bind_window(mutation, values, first, first + count);
        if (status == DORA_OK) {
            status = window_comparable(values, mutation, DORA_COMPARE_OP_EQ);
        }
        bool compact = false;
        for (uint64_t run = 1; status == DORA_OK && run < count && !compact; ++run) {
            status = bound_compare(mutation, values, run - 1, run, DORA_COMPARE_OP_EQ, &compact);
        }
        if (status == DORA_OK && compact && count == 2) {
            uint64_t endpoint = 0;
            status = bind_offset(mutation, end_buffer, first + 1, &endpoint);
            if (status == DORA_OK) {
                status = truncate(values, state, mutation, first + 1);
            }
            if (status == DORA_OK) {
                status = truncate(ends, state, mutation, first + 1);
            }
            if (status == DORA_OK) {
                status = reserve_units(end_buffer, mutation, first, first + 1);
            }
            if (status == DORA_OK) {
                store_offset(state->buffers[end_buffer].unit_bit_width,
                             candidate_address(mutation, end_buffer, first),
                             endpoint);
            }
        } else if (status == DORA_OK && compact) {
            if (status == DORA_OK) {
                status = own_window(mutation, values);
            }
            struct dora_allocation owned_ends = {};
            struct dora_bytes end_bytes = {};
            if (status == DORA_OK) {
                status = read_units(mutation, end_buffer, first, first + count, &end_bytes);
            }
            if (status == DORA_OK && end_bytes.byte_count != 0) {
                status = dora_pool_allocate(state->storage.pool, end_bytes.byte_count, &owned_ends);
                if (status == DORA_OK) {
                    memcpy(owned_ends.address, end_bytes.data, end_bytes.byte_count);
                    end_bytes.data = owned_ends.address;
                }
            }
            if (status == DORA_OK) {
                status = truncate(values, state, mutation, first);
            }
            if (status == DORA_OK) {
                status = truncate(ends, state, mutation, first);
            }
            if (status == DORA_OK) {
                rows->row_count = prior;
            }
            for (uint64_t run = 0; status == DORA_OK && run < count;) {
                uint64_t end = run + 1;
                bool equal = true;
                while (status == DORA_OK && end < count && equal) {
                    status = bound_compare(mutation, values, run, end, DORA_COMPARE_OP_EQ, &equal);
                    if (equal && status == DORA_OK) {
                        ++end;
                    }
                }
                const uint64_t endpoint = span_offset(state->buffers[end_buffer].unit_bit_width,
                                                      &(struct dora_view_buffer){.data = end_bytes.data},
                                                      end - 1);
                if (status == DORA_OK) {
                    status = append_bound_rows(mutation, values, run, 1);
                }
                if (status == DORA_OK) {
                    status =
                        append_run_end(number, mutation, (struct run_end_rows){.length = endpoint - rows->row_count});
                }
                run = end;
            }
            if (status == DORA_OK) {
                rows->closed_row_count = closed;
                checkpoints[number].first_changed_run = UINT64_MAX;
            }
            dora_allocation_release(&owned_ends);
        }
        window_release(mutation);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (checkpoints[number].is_active) {
        checkpoints[number].first_changed_run = UINT64_MAX;
    }
    return DORA_OK;
}

static int publish_window(struct builder_operation* mutation,
                          uint32_t number,
                          bool storage_only,
                          const struct dora_bytes* activity,
                          const struct dora_vector_view* run_parent,
                          struct dora_vector* out_vector,
                          struct dora_allocation* out_logical_validity,
                          struct dora_error* out_error) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    struct dora_vector_view* const view = &state->topology[number].view;
    if (bound_domains(state)[number].handle.slot_index != 0) {
        const struct dora_vector* source = &bound_domains(state)[number];
        struct dora_vector storage = {};
        struct dora_logical_type type = {};
        if (storage_only) {
            int status = dora_vector_get_type(source, &type);
            if (status != DORA_OK) {
                return status;
            }
            if (type.extension_name.byte_count != 0) {
                status = dora_vector_get_storage(source, &storage);
                if (status != DORA_OK) {
                    return status;
                }
                source = &storage;
            }
        }
        return dora_vector_retain(source, out_vector);
    }
    struct dora_schema_field field = *node->field;
    if (storage_only) {
        field.type.extension_name = (struct dora_bytes){};
        field.type.extension_metadata = (struct dora_bytes){};
    }
    size_t child_bytes = 0;
    if (ckd_mul(&child_bytes, field.child_count, sizeof(struct dora_vector))) {
        return EOVERFLOW;
    }
    struct dora_allocation child_storage = {};
    int status = dora_pool_allocate(state->storage.pool, child_bytes, &child_storage);
    if (status != DORA_OK) {
        return status;
    }
    if (child_bytes != 0) {
        memset(child_storage.address, 0, child_bytes);
    }
    struct dora_vector* const children = child_storage.address;
    struct dora_allocation logical = {};
    size_t mask_bytes = 0;
    status = buffer_bytes(bitmap_words_for_bits(view->length), 64, &mask_bytes);
    if (status == DORA_OK) {
        status = dora_pool_allocate(state->storage.pool, mask_bytes, &logical);
    }
    if (status == DORA_OK && mask_bytes != 0) {
        bitmap_fill_bytes((uint8_t*)logical.address, mask_bytes, false);
    }
    for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
        bool is_null = false;
        status = bound_null(number, mutation, row, &is_null);
        const bool active = activity == nullptr || dora_bitmap_test(activity->data, row);
        if (!is_null && active) {
            bitmap_set(logical.address, row);
        }
    }
    for (uint32_t child = state->topology[number].first_child_index; status == DORA_OK && child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const uint32_t ordinal = state->topology[child].child_index;
        if (view->layout == DORA_LAYOUT_RUN_END && ordinal == 0) {
            status = publish_window(mutation, child, false, nullptr, view, &children[ordinal], nullptr, out_error);
            continue;
        }
        const uint64_t length = state->topology[child].view.length;
        size_t bytes = 0;
        status = buffer_bytes(bitmap_words_for_bits(length), 64, &bytes);
        struct dora_allocation active = {};
        if (status == DORA_OK) {
            status = dora_pool_allocate(state->storage.pool, bytes, &active);
        }
        if (status == DORA_OK && bytes != 0) {
            bitmap_fill_bytes((uint8_t*)active.address, bytes, false);
        }
        for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
            if (!dora_bitmap_test(logical.address, row)) {
                continue;
            }
            struct dora_range reached = {};
            if (view->layout == DORA_LAYOUT_STRUCT) {
                reached = (struct dora_range){.offset = row, .length = 1};
            } else if (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION) {
                uint64_t member = 0;
                if (dora_vector_view_get_member(view, row, &member) == ordinal) {
                    reached = (struct dora_range){.offset = member, .length = 1};
                }
            } else if (view->layout == DORA_LAYOUT_RUN_END) {
                reached = (struct dora_range){.offset = dora_vector_view_find_run(view, row), .length = 1};
            } else if (view->layout == DORA_LAYOUT_DICTIONARY) {
                uint64_t value = 0;
                status = dora_vector_view_read_index(view, row, &value);
                reached = (struct dora_range){.offset = value, .length = 1};
            } else {
                reached = dora_vector_view_get_elements(view, row);
            }
            if (status == DORA_OK && (reached.offset > length || reached.length > length - reached.offset)) {
                status = EINVAL;
            }
            if (status == DORA_OK && reached.length != 0) {
                dora_bitmap_set_range(active.address, reached.offset, reached.offset + reached.length, true);
            }
        }
        const bool child_storage_only = storage_only && ((view->layout == DORA_LAYOUT_DICTIONARY && ordinal == 0) ||
                                                         (view->layout == DORA_LAYOUT_RUN_END && ordinal == 1));
        const struct dora_bytes mask = {.data = active.address, .byte_count = bytes};
        if (status == DORA_OK) {
            status = publish_window(mutation,
                                    child,
                                    child_storage_only,
                                    &mask,
                                    nullptr,
                                    &children[ordinal],
                                    nullptr,
                                    out_error);
        }
        dora_allocation_release(&active);
    }
    const bool is_view = view->layout == DORA_LAYOUT_BINARY_VIEW;
    const uint32_t count = is_view ? view->data_buffer_count + 3 : node->buffer_count;
    struct dora_allocation local_allocations[4] = {};
    struct dora_bytes local_buffers[4] = {};
    struct dora_allocation buffer_storage = {};
    size_t storage_bytes;
    if (ckd_mul(&storage_bytes, count, sizeof(struct dora_allocation) + sizeof(struct dora_bytes))) {
        status = EOVERFLOW;
    } else if (count > 4 && status == DORA_OK) {
        status = dora_pool_allocate(state->storage.pool, storage_bytes, &buffer_storage);
        if (status == DORA_OK) {
            memset(buffer_storage.address, 0, storage_bytes);
        }
    }
    struct dora_allocation* const allocations = count > 4 ? buffer_storage.address : local_allocations;
    struct dora_bytes* const buffers =
        buffer_storage.address != nullptr ? (struct dora_bytes*)(allocations + count) : local_buffers;
    const struct builder_buffer_state* const formats = state->buffers;
    for (uint32_t part = 0; status == DORA_OK && part < count; ++part) {
        const uint32_t index = is_view ? (part < 2            ? node->first_buffer + part
                                          : part == count - 1 ? node->first_buffer + 3
                                                              : node->first_buffer + 2)
                                       : node->first_buffer + part;
        const struct builder_buffer_state* const format = &formats[index];
        struct dora_view_buffer source = {};
        const uint64_t zero = 0;
        struct dora_view_buffer* const span = role_span(view, &bound_data(state)[number], format->role);
        if (span != nullptr) {
            source = *span;
        }
        switch (format->role) {
        case BUILDER_BUFFER_BYTES:
            source = is_view ? view->data_buffers[part - 2] : bound_data(state)[number];
            if (is_view && state->buffers[node->first_buffer + 3].allocation.address != nullptr) {
                ((int64_t*)state->buffers[node->first_buffer + 3].allocation.address)[part - 2] =
                    (int64_t)source.byte_count;
            }
            break;
        case BUILDER_BUFFER_VARIADIC_SIZES:
            source = (struct dora_view_buffer){
                .data = state->buffers[node->first_buffer + 3].allocation.address,
                .byte_count = view->data_buffer_count * sizeof(int64_t)
            };
            break;
        default:
            break;
        }
        if (source.data == nullptr &&
            (format->role == BUILDER_BUFFER_OFFSETS32 || format->role == BUILDER_BUFFER_OFFSETS64) &&
            view->layout != DORA_LAYOUT_LIST_VIEW && view->layout != DORA_LAYOUT_LARGE_LIST_VIEW) {
            source = (struct dora_view_buffer){.data = &zero, .byte_count = format->unit_bit_width / 8};
        }
        if (source.byte_count != 0 && source.data == nullptr) {
            status = EINVAL;
            break;
        }
        if (source.byte_count != 0) {
            size_t bytes = source.byte_count;
            if (format->unit_bit_width == 1) {
                status = buffer_bytes(bitmap_words_for_bits(view->length), 64, &bytes);
            }
            if (status == DORA_OK) {
                status = dora_pool_allocate(state->storage.pool, bytes, &allocations[part]);
            }
            if (status == DORA_OK) {
                if (format->unit_bit_width == 1) {
                    bitmap_fill_bytes((uint8_t*)allocations[part].address, bytes, false);
                    dora_bitmap_load_words(source.data,
                                           format->role == BUILDER_BUFFER_VALIDITY_BITS ? view->validity_bit_offset
                                                                                        : view->offset,
                                           view->length,
                                           allocations[part].address);
                    source.byte_count = bitmap_bytes_for_bits(view->length);
                } else {
                    memcpy(allocations[part].address, source.data, source.byte_count);
                }
                if (run_parent != nullptr && format->role == BUILDER_BUFFER_FIXED_VALUES) {
                    struct dora_view_buffer ends = {.data = allocations[part].address};
                    for (uint64_t row = 0; row < view->length; ++row) {
                        uint64_t end = span_offset(format->unit_bit_width, &ends, row) - run_parent->offset;
                        if (end > run_parent->length) {
                            end = run_parent->length;
                        }
                        store_offset(format->unit_bit_width,
                                     (uint8_t*)allocations[part].address + row * (format->unit_bit_width / 8),
                                     end);
                    }
                }
                buffers[part] = (struct dora_bytes){.data = allocations[part].address, .byte_count = source.byte_count};
            }
        }
    }
    if (status == DORA_OK) {
        const struct vector_input input = {
            .field = &field,
            .admitted_schema = &node->schema,
            .row_count = view->length,
            .null_count = view->null_count,
            .buffers = buffers,
            .buffer_count = count,
            .children = children,
            .child_count = field.child_count,
            .ancestor_activity = activity,
            .content_state = storage_only || (node->extension != nullptr && node->extension->validate != nullptr) ||
                                     field.type.extension_name.byte_count == 0
                                 ? mutation_read_rows(mutation, number)->content_state
                                 : VECTOR_CONTENT_PROMISED
        };
        status = dora_vector_init_move_input(out_vector, state->storage.pool, &input, allocations, out_error);
    }
    for (uint32_t part = 0; allocations != nullptr && part < count; ++part) {
        dora_allocation_release(&allocations[part]);
    }
    dora_allocation_release(&buffer_storage);
    for (uint32_t child = 0; child < field.child_count; ++child) {
        dora_vector_release(&children[child]);
    }
    if (status == DORA_OK && out_logical_validity != nullptr) {
        *out_logical_validity = logical;
        logical = (struct dora_allocation){};
    }
    dora_allocation_release(&logical);
    dora_allocation_release(&child_storage);
    return status;
}

static int check_required_window(struct builder_operation* mutation,
                                 uint32_t number,
                                 uint64_t first,
                                 const struct dora_schema_field* field) {
    if ((field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
        return DORA_OK;
    }
    const struct dora_vector_view* const view = &mutation->state->topology[number].view;
    if (view->layout != DORA_LAYOUT_RUN_END && view->layout != DORA_LAYOUT_DENSE_UNION &&
        view->layout != DORA_LAYOUT_SPARSE_UNION) {
        return DORA_OK;
    }
    for (uint64_t row = 0; row < view->length; ++row) {
        bool is_null = false;
        const int status = bound_null(number, mutation, row, &is_null);
        if (status != DORA_OK || (is_null && !is_inactive(number, mutation, first + row))) {
            return status != DORA_OK ? status : EINVAL;
        }
    }
    return DORA_OK;
}

static int check_required(struct builder_operation* mutation) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const nodes = state->nodes;
    const uint32_t* const changed = mutation_node_indices(state);
    for (uint32_t cursor = 0; cursor < mutation->data.touched_node_count; ++cursor) {
        const uint32_t number = changed[cursor];
        const struct dora_schema_field* const field = nodes[number].field;
        const uint64_t first = nodes[number].rows.row_count;
        const uint64_t end = mutation_read_rows(mutation, number)->row_count;
        if (end <= first || (field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0 ||
            (field->layout != DORA_LAYOUT_RUN_END && field->layout != DORA_LAYOUT_DENSE_UNION &&
             field->layout != DORA_LAYOUT_SPARSE_UNION)) {
            continue;
        }
        int status = bind_window(mutation, number, first, end);
        if (status == DORA_OK) {
            status = check_required_window(mutation, number, first, field);
        }
        window_release(mutation);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int check_required_ancestors(struct builder_operation* mutation,
                                    uint32_t number,
                                    const struct dora_error* hook_error,
                                    struct dora_error* out_error) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const nodes = mutation->state->nodes;
    for (uint32_t parent = state->topology[number].parent_node_index; parent != UINT32_MAX;
         parent = state->topology[parent].parent_node_index) {
        const enum dora_layout layout = mutation->state->topology[parent].view.layout;
        if (layout != DORA_LAYOUT_DICTIONARY && layout != DORA_LAYOUT_RUN_END && layout != DORA_LAYOUT_DENSE_UNION &&
            layout != DORA_LAYOUT_SPARSE_UNION) {
            break;
        }
        struct dora_schema_field field = {};
        int status = dora_schema_get_field(&nodes[parent].schema, &field);
        const uint64_t first = nodes[parent].rows.row_count;
        const uint64_t end = mutation_read_rows(mutation, parent)->row_count;
        if (status == DORA_OK && (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0 && first < end) {
            status = bind_window(mutation, parent, first, end);
            if (status == DORA_OK) {
                status = check_required_window(mutation, parent, first, &field);
                if (status == EINVAL && out_error != nullptr && out_error->message[0] == 0) {
                    *out_error = *hook_error;
                }
            }
            window_release(mutation);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int validate_extensions(struct builder_operation* mutation, struct dora_error* out_error) {
    struct builder_state* const state = mutation->state;
    if (!state->has_validation_nodes) {
        return DORA_OK;
    }
    const struct builder_node_state* const nodes = state->nodes;
    uint32_t* const changed = mutation_node_indices(state);
    if (mutation->data.touched_node_count > 1) {
        qsort(changed, mutation->data.touched_node_count, sizeof(*changed), compare_indices);
    }
    for (uint32_t cursor = mutation->data.touched_node_count; cursor != 0;) {
        const uint32_t number = changed[--cursor];
        const struct builder_node_state* const node = &nodes[number];
        const uint64_t first = node->rows.row_count;
        const uint64_t end = mutation_read_rows(mutation, number)->row_count;
        if (end <= first) {
            continue;
        }
        const struct dora_schema_field field = *node->field;
        const bool map = field.type.id == DORA_TYPE_MAP &&
                         (field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LARGE_LIST);
        const bool hook = field.layout != DORA_LAYOUT_DICTIONARY && field.layout != DORA_LAYOUT_RUN_END &&
                          node->extension != nullptr && node->extension->validate != nullptr;
        const bool encoded = field.layout == DORA_LAYOUT_RUN_END || field.layout == DORA_LAYOUT_DENSE_UNION ||
                             field.layout == DORA_LAYOUT_SPARSE_UNION;
        if (!map && !hook) {
            continue;
        }
        if (state->options.content_policy == DORA_FAILURE_POLICY_TRUST) {
            mutation_rows(mutation, number)->content_state = VECTOR_CONTENT_PROMISED;
            continue;
        }
        int status = bind_window(mutation, number, first, end);
        if (status == DORA_OK && map) {
            const uint32_t keys = builder_child(builder_child(number, state, 0), state, 0);
            const struct dora_vector_view* const view = &state->topology[number].view;
            for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
                if (!dora_vector_view_is_null(view, row)) {
                    status = check_map_keys(keys,
                                            mutation,
                                            dora_vector_view_get_elements(view, row),
                                            (field.flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) != 0);
                }
            }
        }
        struct dora_vector independent = {};
        struct dora_allocation active = {};
        struct dora_allocation baseline = {};
        size_t bytes = 0;
        if (status == DORA_OK && hook) {
            status = buffer_bytes(bitmap_words_for_bits(end - first), 64, &bytes);
        }
        if (status == DORA_OK && hook && (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
            status = dora_pool_allocate(state->storage.pool, bytes, &active);
            if (status == DORA_OK && bytes != 0) {
                bitmap_fill_bytes((uint8_t*)active.address, bytes, false);
            }
            for (uint64_t row = 0; status == DORA_OK && row < end - first; ++row) {
                bool is_null = false;
                status = bound_null(number, mutation, row, &is_null);
                if (status == DORA_OK && (encoded ? !is_inactive(number, mutation, first + row) : !is_null)) {
                    bitmap_set(active.address, row);
                }
            }
        }
        const struct dora_bytes activity = {.data = active.address, .byte_count = bytes};
        if (status == DORA_OK && hook) {
            status = publish_window(mutation,
                                    number,
                                    true,
                                    active.address == nullptr ? nullptr : &activity,
                                    nullptr,
                                    &independent,
                                    &baseline,
                                    out_error);
        }
        window_release(mutation);
        dora_allocation_release(&active);
        struct dora_allocation mask = {};
        if (status == DORA_OK && hook) {
            status = dora_pool_allocate(state->storage.pool, bytes, &mask);
            if (status == DORA_OK && bytes != 0) {
                bitmap_copy_bytes(mask.address, baseline.address, bytes);
            }
        }
        struct dora_error hook_error = {};
        const bool null_policy = state->options.content_policy == DORA_FAILURE_POLICY_NULL;
        if (status == DORA_OK && hook) {
            dora_pool_callback_enter(state->storage.pool);
            status = node->extension->validate(node->extension,
                                               state->storage.pool,
                                               &node->extension_parameters,
                                               &independent,
                                               mask.address,
                                               out_error != nullptr ? &hook_error : nullptr);
            dora_pool_callback_leave(state->storage.pool);
            hook_error.message[sizeof(hook_error.message) - 1] = 0;
            if (null_policy && (status == EILSEQ || status == ERANGE)) {
                status = DORA_OK;
            }
            if (status != DORA_OK && out_error != nullptr && out_error->message[0] == 0) {
                *out_error = hook_error;
            }
        }
        if (status == DORA_OK && hook && null_policy) {
            bool changed = false;
            for (uint64_t word = 0; word < bitmap_words_for_bits(end - first); ++word) {
                const uint64_t failures = dora_bitmap_get_word(baseline.address, word * 64, end - first) &
                                          ~dora_bitmap_get_word(mask.address, word * 64, end - first);
                ((uint64_t*)mask.address)[word] = failures;
                changed = changed || failures != 0;
            }
            if (changed && (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
                status = EINVAL;
                if (out_error != nullptr && out_error->message[0] == 0) {
                    *out_error = hook_error;
                }
            } else if (changed) {
                status = mark_nulls(number,
                                    mutation,
                                    (struct dora_range){.offset = first, .length = end - first},
                                    mask.address);
                if (status == DORA_OK) {
                    status = check_required_ancestors(mutation, number, &hook_error, out_error);
                }
            }
        }
        dora_allocation_release(&mask);
        dora_allocation_release(&baseline);
        dora_vector_release(&independent);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static void domain_frontiers(struct builder_operation* mutation, uint32_t number) {
    struct builder_state* const state = mutation->state;
    const struct dora_vector_view* const view = &state->topology[number].view;
    uint64_t rows = view->length;
    if (state->topology[number].parent_node_index != UINT32_MAX && state->topology[number].child_index == 1 &&
        state->topology[state->topology[number].parent_node_index].view.layout == DORA_LAYOUT_RUN_END) {
        rows =
            state->topology[state->topology[state->topology[number].parent_node_index].first_child_index].view.length;
    }
    mutation_rows(mutation, number)->stable_row_count = rows;
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        domain_frontiers(mutation, child);
    }
}

static int advance_stable(struct builder_operation* mutation, uint32_t number, uint64_t limit, bool finishing) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const nodes = state->nodes;
    const struct builder_node_state* const node = &nodes[number];
    const struct dora_vector_view shape = state->topology[number].view;
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    if (limit < node->rows.stable_row_count || limit > rows->row_count) {
        return EINVAL;
    }
    rows->stable_row_count = limit;
    if (node->dictionary.handle.slot_index != 0) {
        if (limit == 0 && !finishing) {
            return DORA_OK;
        }
        const int status = bind_domain(mutation, state->topology[number].first_child_index, &node->dictionary);
        if (status == DORA_OK) {
            domain_frontiers(mutation, state->topology[number].first_child_index);
        }
        return status;
    }
    uint64_t child_limit = limit;
    if (shape.layout == DORA_LAYOUT_FIXED_LIST) {
        if (ckd_mul(&child_limit, limit, shape.list_element_count)) {
            return EOVERFLOW;
        }
    } else if (shape.layout == DORA_LAYOUT_LIST || shape.layout == DORA_LAYOUT_LARGE_LIST) {
        child_limit = 0;
        if (limit != 0) {
            const int status = bind_offset(mutation, node->first_buffer + 1, limit, &child_limit);
            if (status != DORA_OK) {
                return status;
            }
        }
    } else if (shape.layout == DORA_LAYOUT_LIST_VIEW || shape.layout == DORA_LAYOUT_LARGE_LIST_VIEW) {
        child_limit = 0;
        if (limit != 0) {
            const int status = bind_offset(mutation, node->frontier_buffer, limit - 1, &child_limit);
            if (status != DORA_OK) {
                return status;
            }
        }
    } else if (shape.layout == DORA_LAYOUT_RUN_END) {
        child_limit = 0;
        if (limit != 0) {
            const int status = find_run(number, mutation, limit - 1, &child_limit);
            if (status != DORA_OK) {
                return status;
            }
            ++child_limit;
        }
    }
    uint64_t members[UNION_MEMBER_CAPACITY] = {};
    if (shape.layout == DORA_LAYOUT_DENSE_UNION) {
        for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            members[state->topology[child].child_index] = nodes[child].rows.stable_row_count;
        }
        struct dora_bytes ids = {};
        struct dora_bytes offsets = {};
        const uint64_t first = node->rows.stable_row_count;
        int status = read_units(mutation, node->first_buffer, first, limit, &ids);
        if (status == DORA_OK) {
            status = read_units(mutation, node->first_buffer + 1, first, limit, &offsets);
        }
        for (uint64_t row = 0; status == DORA_OK && row < limit - first; ++row) {
            const uint32_t member = shape.union_members[(uint8_t)((const int8_t*)ids.data)[row]];
            const uint64_t end = span_offset(32, &(struct dora_view_buffer){.data = offsets.data}, row) + 1;
            if (member >= UNION_MEMBER_CAPACITY) {
                status = EINVAL;
            } else if (end > members[member]) {
                members[member] = end;
            }
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    for (uint32_t child = state->topology[number].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        uint64_t end =
            shape.layout == DORA_LAYOUT_DENSE_UNION ? members[state->topology[child].child_index] : child_limit;
        if (shape.layout == DORA_LAYOUT_RUN_END && state->topology[child].child_index == 0 && end != 0 && !finishing) {
            --end;
        }
        const int status = advance_stable(mutation, child, end, finishing);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static uint32_t dictionary_ancestor(const struct builder_state* state, uint32_t number) {
    const struct builder_node_state* const nodes = state->nodes;
    for (uint32_t parent = state->topology[number].parent_node_index; parent != UINT32_MAX;
         parent = state->topology[parent].parent_node_index) {
        if (nodes[parent].dictionary.handle.slot_index != 0) {
            return parent;
        }
    }
    return UINT32_MAX;
}

static int update_statistics(struct builder_operation* mutation, bool finishing) {
    struct builder_state* const state = mutation->state;
    if (state->statistics == nullptr ||
        (!finishing && mutation_read_rows(mutation, 0)->row_count == state->nodes[0].rows.stable_row_count)) {
        return DORA_OK;
    }
    int status = advance_stable(mutation, 0, mutation_read_rows(mutation, 0)->row_count, finishing);
    if (status != DORA_OK) {
        return status;
    }
    status = dora_statistics_checkpoint_begin(state->statistics, &mutation->data.statistics);
    const struct builder_node_state* const nodes = state->nodes;
    for (uint32_t number = 0; status == DORA_OK && number < state->node_count; ++number) {
        if (number != 0) {
            const enum dora_layout layout = state->topology[state->topology[number].parent_node_index].view.layout;
            if (layout != DORA_LAYOUT_DICTIONARY && layout != DORA_LAYOUT_RUN_END) {
                continue;
            }
        }
        struct builder_row_state* const rows = mutation_rows(mutation, number);
        const uint64_t first = nodes[number].rows.stable_row_count;
        const uint64_t end = rows->stable_row_count;
        if (end < first) {
            status = EINVAL;
            break;
        }
        if (end == first) {
            continue;
        }
        const uint32_t dictionary = dictionary_ancestor(state, number);
        const bool domain = dictionary != UINT32_MAX;
        if (domain) {
            status =
                bind_domain(mutation, state->topology[dictionary].first_child_index, &nodes[dictionary].dictionary);
        } else {
            status = bind_window(mutation, number, first, end);
        }
        if (status == DORA_OK) {
            status = dora_statistics_accumulator_update_nodes(
                state->statistics,
                state->topology,
                state->node_count,
                number,
                (struct dora_range){.offset = domain ? first : 0, .length = end - first});
        }
        window_release(mutation);
    }
    return status;
}

int dora_builder_statistics_checkpoint(struct builder_operation* operation,
                                       struct statistics_checkpoint* out_checkpoint) {
    int status = scratch_begin(operation);
    if (status != DORA_OK) {
        return status;
    }
    status = update_statistics(operation, true);
    if (status == DORA_OK) {
        *out_checkpoint = operation->data.statistics;
        operation->data.statistics = (struct statistics_checkpoint){};
    }
    mutation_finish(operation, false);
    return status;
}

static int complete_record(struct builder_operation* mutation) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const nodes = state->nodes;
    const struct dora_schema_field root = *nodes[0].field;
    if (root.layout != DORA_LAYOUT_STRUCT || root.child_count == 0) {
        return DORA_OK;
    }
    uint64_t count = mutation_read_rows(mutation, state->topology[0].first_child_index)->row_count;
    uint64_t fewest = count;
    bool equal = true;
    for (uint32_t child = state->topology[0].first_child_index; child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const uint64_t rows = mutation_read_rows(mutation, child)->row_count;
        equal = equal && rows == count;
        fewest = rows < fewest ? rows : fewest;
    }
    const uint64_t root_rows = mutation_read_rows(mutation, 0)->row_count;
    if (equal && count < root_rows) {
        return EINVAL;
    }
    const uint64_t records = state->completed_root_row_count < fewest ? state->completed_root_row_count : fewest;
    count = equal && count > records ? count : records;
    if (count <= root_rows) {
        return DORA_OK;
    }
    struct builder_row_state* const rows = mutation_rows(mutation, 0);
    uint64_t end = 0;
    int status = append_limit(0, state, rows->row_count, count - rows->row_count, &end);
    while (status == DORA_OK && rows->row_count < end) {
        status = append_validity(mutation, 0, &root, mutation_rows(mutation, 0)->row_count + 1, true, false, false);
        if (status == DORA_OK) {
            ++rows->row_count;
        }
    }
    for (uint32_t child = state->topology[0].first_child_index; status == DORA_OK && child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        mutation_rows(mutation, child)->closed_row_count = count;
    }
    return status;
}

static int update_written(struct builder_operation* mutation) {
    size_t written = mutation->state->written_byte_count;
    const struct builder_buffer_state* const buffers = mutation->state->buffers;
    const struct builder_buffer_checkpoint* const candidates = mutation->state->mutation_buffers;
    const uint32_t* const changed = mutation_buffer_indices(mutation->state);
    for (uint32_t cursor = 0; cursor < mutation->data.touched_buffer_count; ++cursor) {
        const uint32_t index = changed[cursor];
        size_t old_bytes = 0;
        size_t new_bytes = 0;
        int status = buffer_bytes(buffers[index].initialized_unit_count, buffers[index].unit_bit_width, &old_bytes);
        if (status == DORA_OK) {
            status = buffer_bytes(candidates[index].initialized_unit_count, buffers[index].unit_bit_width, &new_bytes);
        }
        if (status != DORA_OK || old_bytes > written) {
            return status != DORA_OK ? status : EINVAL;
        }
        written -= old_bytes;
        if (ckd_add(&written, written, new_bytes)) {
            return EOVERFLOW;
        }
    }
    mutation->data.written_byte_count = written;
    return DORA_OK;
}

static int mutation_admit(struct builder_operation* mutation, struct dora_error* out_error) {
    int status = complete_record(mutation);
    if (status == DORA_OK) {
        status = check_required(mutation);
    }
    const uint32_t checked_nodes = mutation->data.touched_node_count;
    if (status == DORA_OK && !mutation->is_content_carried) {
        status = validate_extensions(mutation, out_error);
    }
    if (status == DORA_OK) {
        uint32_t* const changed = mutation_node_indices(mutation->state);
        if (mutation->data.touched_node_count > 1 &&
            (mutation->data.touched_node_count != checked_nodes || !mutation->state->has_validation_nodes)) {
            qsort(changed, mutation->data.touched_node_count, sizeof(*changed), compare_indices);
        }
        for (uint32_t cursor = 0; status == DORA_OK && cursor < mutation->data.touched_node_count; ++cursor) {
            status = normalize_runs(mutation, changed[cursor]);
        }
    }
    if (status == DORA_OK) {
        status = update_written(mutation);
    }
    if (status == DORA_OK) {
        status = update_statistics(mutation, false);
    }
    return status;
}

static int mutation_complete(struct builder_operation* mutation,
                             int status,
                             bool close_target_children,
                             struct dora_error* out_error) {
    if (status == DORA_OK) {
        status = mutation_admit(mutation, out_error);
    }
    mutation_finish(mutation, status == DORA_OK);
    if (status != DORA_OK && close_target_children) {
        const int rollback_status = rollback_children(mutation->state, mutation->data.target.node_index);
        if (rollback_status != DORA_OK) {
            status = rollback_status;
        }
    }
    return dora_builder_operation_leave(mutation, status, nullptr);
}

int dora_builder_reserve(struct dora_builder* builder, uint64_t row_count, size_t byte_count) {
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = mutation.state;
    const struct builder_node_state* const node = &state->nodes[mutation.data.target.node_index];
    const struct dora_schema_field field = *node->field;
    const bool list_view = field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    uint64_t end = 0;
    status = append_limit(mutation.data.target.node_index, state, 0, row_count, &end);
    for (uint32_t part = 0; status == DORA_OK && part < node->buffer_count; ++part) {
        const uint32_t index = node->first_buffer + part;
        const struct builder_buffer_state* const buffer = &state->buffers[index];
        const bool offsets = buffer->role == BUILDER_BUFFER_OFFSETS32 || buffer->role == BUILDER_BUFFER_OFFSETS64;
        if (buffer->role == BUILDER_BUFFER_VARIADIC_SIZES ||
            (buffer->role == BUILDER_BUFFER_VALIDITY_BITS && buffer->allocation.address == nullptr)) {
            continue;
        }
        uint64_t units = buffer->role == BUILDER_BUFFER_BYTES ? byte_count : row_count;
        if (offsets && !list_view && ckd_add(&units, units, UINT64_C(1))) {
            status = EOVERFLOW;
            break;
        }
        size_t required = 0;
        status = buffer->unit_bit_width == 1 ? buffer_bytes(bitmap_words_for_bits(units), 64, &required)
                                             : buffer_bytes(units, buffer->unit_bit_width, &required);
        if (status != DORA_OK || required <= buffer->allocation.capacity_bytes) {
            continue;
        }
        status = reserve_units(index, &mutation, buffer->initialized_unit_count, units);
        if (status == DORA_OK && offsets && !list_view && buffer->initialized_unit_count == 0) {
            store_offset(buffer->unit_bit_width, candidate_address(&mutation, index, 0), 0);
            mutation_buffer(&mutation, index)->initialized_unit_count = 1;
        }
    }
    return mutation_complete(&mutation, status, false, nullptr);
}

static void runs_close_cursors(struct builder_state* state, const struct builder_operation* operation) {
    const uint32_t* const touched = mutation_node_indices(state);
    for (uint32_t cursor = 0; cursor < operation->data.touched_node_count; ++cursor) {
        state->runs.columns[touched[cursor]] = (struct builder_column_run){};
    }
}

int dora_builder_runs_end(struct builder_state* state, bool commit) {
    if (state->runs.operation.target.handle.slot_index == 0) {
        return DORA_OK;
    }
    struct builder_operation operation = {};
    int status = dora_builder_operation_resume(&state->runs.operation, BUILDER_PHASE_OPEN, &operation);
    if (status != DORA_OK) {
        return status;
    }
    runs_close_cursors(state, &operation);
    if (commit) {
        status = mutation_admit(&operation, nullptr);
    }
    if (!commit || status != DORA_OK) {
        state->completed_root_row_count = state->runs.records;
    }
    mutation_finish(&operation, commit && status == DORA_OK);
    return dora_builder_operation_leave(&operation, status, nullptr);
}

void dora_builder_run_discard(struct builder_state* state) {
    if (state->runs.operation.target.handle.slot_index == 0) {
        return;
    }
    struct builder_operation operation = {};
    dora_builder_operation_data_move(&state->runs.operation, &operation.data);
    operation.state = state;
    runs_close_cursors(state, &operation);
    state->completed_root_row_count = state->runs.records;
    mutation_finish(&operation, false);
}

static int
append_public_error(struct dora_builder* builder, const struct builder_input* input, struct dora_error* out_error) {
    struct builder_operation mutation = {};
    const int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    return status != DORA_OK
               ? status
               : mutation_complete(
                     &mutation,
                     append_input(&mutation, mutation.data.target.node_index, input, BUILDER_APPEND_INPUT),
                     false,
                     out_error);
}

//------------------------------------------------------------------------------
//
// Direct Fill
//
//------------------------------------------------------------------------------

static bool layout_is_bytes(enum dora_layout layout) {
    return layout == DORA_LAYOUT_BINARY || layout == DORA_LAYOUT_LARGE_BINARY || layout == DORA_LAYOUT_BINARY_VIEW;
}

int dora_builder_prepare(struct dora_builder* builder,
                         uint64_t row_count,
                         size_t byte_count,
                         struct dora_mutable_bytes* out_values) {
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = mutation.state;
    const uint32_t number = mutation.data.target.node_index;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field field = *node->field;
    const bool bytes = layout_is_bytes(field.layout);
    if (!dora_builder_operation_addresses(&mutation, builder) || out_values == nullptr ||
        (row_count == 0 && byte_count != 0) || (bytes && row_count != 1)) {
        status = EINVAL;
    } else if (field.layout == DORA_LAYOUT_BITS ||
               (!bytes && field.layout != DORA_LAYOUT_FIXED && field.layout != DORA_LAYOUT_DICTIONARY)) {
        status = ENOTSUP;
    }
    uint64_t end = 0;
    if (status == DORA_OK) {
        status = append_limit(number, state, node->rows.row_count, row_count, &end);
    }
    uint32_t index = node->first_buffer + (bytes ? 2 : 1);
    const uint32_t width = status == DORA_OK ? state->buffers[index].unit_bit_width : 0;
    uint64_t first_unit = bytes ? node->rows.byte_count : node->rows.row_count;
    const bool is_view = field.layout == DORA_LAYOUT_BINARY_VIEW;
    if (status == DORA_OK && is_view) {
        if (byte_count > INT32_MAX) {
            status = EOVERFLOW;
        } else if (byte_count > VIEW_INLINE_BYTE_COUNT) {
            status = reserve_view(number, &mutation, byte_count, &index);
        } else if (node->rows.data_buffer_count != 0) {
            index = node->rows.last_data_buffer;
        }
        first_unit = node->rows.data_buffer_count == 0 && index == node->first_buffer + 2
                         ? 0
                         : state->buffers[index].initialized_unit_count;
    }
    uint64_t end_unit = end;
    size_t required = byte_count;
    if (status == DORA_OK && !bytes) {
        status = buffer_bytes(row_count, width, &required);
        if (status == DORA_OK && byte_count != 0 && byte_count != required) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK && bytes && (ckd_add(&end_unit, first_unit, byte_count) || end_unit > INT64_MAX)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = reserve_units(index, &mutation, first_unit, end_unit);
    }
    if (status == DORA_OK && !dora_builder_operation_addresses(&mutation, builder)) {
        status = EINVAL;
    }
    if (status != DORA_OK) {
        mutation_finish(&mutation, false);
        return dora_builder_operation_leave(&mutation, status, nullptr);
    }
    const struct dora_mutable_bytes values = {
        .data = required == 0 ? nullptr : candidate_address(&mutation, index, first_unit),
        .byte_count = required
    };
    state->window = (struct builder_window){
        .row_count = row_count,
        .first_unit = first_unit,
        .byte_count = required,
        .buffer_index = index
    };
    state->phase = BUILDER_PHASE_PREPARED;
    struct builder_state* live = nullptr;
    status = dora_builder_operation_park(&mutation, &state->window.operation, DORA_OK, &live);
    if (status == DORA_OK) {
        *out_values = values;
    }
    return status;
}

static void window_clear_nulls(struct builder_operation* mutation, uint32_t number) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const size_t width = state->buffers[node->first_buffer + 1].unit_bit_width / 8;
    const uint64_t* const words = candidate_bit_words(mutation, node->first_buffer);
    const uint64_t end = state->window.first_unit + state->window.row_count;
    for (uint64_t row = state->window.first_unit; words != nullptr && width != 0 && row < end; ++row) {
        if (!dora_bitmap_test((const uint8_t*)words, row)) {
            memset(candidate_address(mutation, node->first_buffer + 1, row), 0, width);
        }
    }
}

// Failed commit preserves window storage and bytes.
static void
window_reopen(uint32_t index, struct builder_operation* mutation, const struct builder_buffer_checkpoint* window) {
    mutation_finish(mutation, false);
    *mutation_buffer(mutation, index) = *window;
}

int dora_builder_commit(struct dora_builder* builder, const bool* validity) {
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_PREPARED_ACCESS, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = mutation.state;
    const uint32_t number = mutation.data.target.node_index;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field field = *node->field;
    const bool bytes = layout_is_bytes(field.layout);
    const uint32_t index = state->window.buffer_index;
    const struct builder_buffer_checkpoint window = state->mutation_buffers[index];
    if (bytes) {
        const struct dora_bytes value = {
            .data =
                state->window.byte_count == 0 ? nullptr : candidate_address(&mutation, index, state->window.first_unit),
            .byte_count = state->window.byte_count
        };
        status = append_bytes(&mutation,
                              number,
                              &field,
                              &(struct builder_input){
                                  .values = &value,
                                  .validity = validity,
                                  .row_count = 1,
                                  .is_byte_input = true,
                                  .is_in_window = true
                              });
    } else {
        enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
        const struct dora_logical_type type =
            field.layout == DORA_LAYOUT_DICTIONARY ? (struct dora_logical_type){.id = field.index_type} : field.type;
        status = type_physical(&type, &physical);
        if (status == DORA_OK) {
            status = append_fixed(
                &mutation,
                number,
                &field,
                &(struct builder_input){
                    .values = state->window.byte_count == 0
                                  ? nullptr
                                  : candidate_address(&mutation, node->first_buffer + 1, state->window.first_unit),
                    .validity = validity,
                    .row_count = state->window.row_count,
                    .physical = physical,
                    .is_in_window = true
                });
        }
    }
    if (status == DORA_OK) {
        status = mutation_admit(&mutation, nullptr);
    }
    if (status == DORA_OK) {
        if (!bytes) {
            window_clear_nulls(&mutation, number);
        }
        mutation_finish(&mutation, true);
        state->window = (struct builder_window){};
        state->phase = BUILDER_PHASE_OPEN;
    } else {
        window_reopen(index, &mutation, &window);
        return dora_builder_operation_park(&mutation, &state->window.operation, status, nullptr);
    }
    return dora_builder_operation_leave(&mutation, status, nullptr);
}

int dora_builder_abort(struct dora_builder* builder) {
    struct builder_operation operation = {};
    const int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_PREPARED_ACCESS, &operation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = operation.state;
    mutation_finish(&operation, false);
    state->window = (struct builder_window){};
    state->phase = BUILDER_PHASE_OPEN;
    return dora_builder_operation_leave(&operation, DORA_OK, nullptr);
}

void dora_builder_discard_window(struct builder_state* state) {
    const bool is_prepared = state->phase == BUILDER_PHASE_PREPARED;
    struct builder_operation operation = {};
    dora_builder_operation_data_move(&state->window.operation, &operation.data);
    operation.state = state;
    state->window = (struct builder_window){};
    state->phase = BUILDER_PHASE_OPEN;
    if (is_prepared) {
        mutation_finish(&operation, false);
    }
}

//------------------------------------------------------------------------------
//
// Completion and Lifecycle
//
//------------------------------------------------------------------------------

static uint64_t pending_rows(const struct builder_state* state, uint32_t number) {
    const struct builder_row_checkpoint* const checkpoints = state->mutation_nodes;
    return checkpoints[number].is_active ? checkpoints[number].rows.row_count : state->nodes[number].rows.row_count;
}

int dora_builder_end_row_error(struct dora_builder* builder, struct dora_error* out_error) {
    struct builder_state* state = nullptr;
    int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase != BUILDER_PHASE_OPEN) {
        return EBUSY;
    }
    const struct dora_schema_field* const root = state->nodes[0].field;
    if (builder->node_index != 0 || root->layout != DORA_LAYOUT_STRUCT) {
        return EINVAL;
    }
    uint64_t expected = 0;
    if (ckd_add(&expected, state->completed_root_row_count, 1U)) {
        return EOVERFLOW;
    }
    if (root->child_count != 0) {
        // Record rows commit with the run set.
        if (state->options.row_capacity != 0 && expected > state->options.row_capacity) {
            return ENOSPC;
        }
        for (uint32_t child = state->topology[0].first_child_index; child != UINT32_MAX;
             child = state->topology[child].next_sibling_index) {
            if (pending_rows(state, child) != expected) {
                return EINVAL;
            }
        }
        state->completed_root_row_count = expected;
        return DORA_OK;
    }
    struct builder_operation mutation = {};
    status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    state = mutation.state;
    struct builder_row_state* const rows = mutation_rows(&mutation, 0);
    uint64_t end = 0;
    status =
        rows->row_count > expected ? EINVAL : append_limit(0, state, rows->row_count, expected - rows->row_count, &end);
    while (status == DORA_OK && rows->row_count < end) {
        status = append_validity(&mutation, 0, root, mutation_rows(&mutation, 0)->row_count + 1, true, false, false);
        if (status == DORA_OK) {
            ++rows->row_count;
        }
    }
    if (status == DORA_OK) {
        status = mutation_admit(&mutation, out_error);
    }
    mutation_finish(&mutation, status == DORA_OK);
    if (status == DORA_OK) {
        state->completed_root_row_count = expected;
    }
    return dora_builder_operation_leave(&mutation, status, nullptr);
}

int dora_builder_end_row(struct dora_builder* builder) {
    return dora_builder_end_row_error(builder, nullptr);
}

//------------------------------------------------------------------------------
//
// Append Runs
//
//------------------------------------------------------------------------------

static uint64_t run_capacity_rows(const struct builder_buffer_state* buffer) {
    return buffer->unit_bit_width == 1 ? bitmap_whole_word_capacity(buffer->allocation.capacity_bytes)
                                       : buffer->allocation.capacity_bytes / (buffer->unit_bit_width / 8);
}

static int
run_reserve(uint32_t index, struct builder_operation* mutation, uint64_t first, uint64_t end, uint64_t* in_out_limit) {
    int status = reserve_units(index, mutation, first, end);
    if (status != DORA_OK) {
        return status;
    }
    const uint64_t capacity = run_capacity_rows(&mutation->state->buffers[index]);
    if (*in_out_limit > capacity) {
        *in_out_limit = capacity;
    }
    return reserve_units(index, mutation, first, *in_out_limit);
}

static enum builder_run_kind run_kind(const struct dora_schema_field* field) {
    return field->layout == DORA_LAYOUT_BINARY_VIEW                                           ? BUILDER_RUN_VIEW
           : field->type.id == DORA_TYPE_FIXED_BINARY                                         ? BUILDER_RUN_FIXED_BINARY
           : field->layout == DORA_LAYOUT_BINARY || field->layout == DORA_LAYOUT_LARGE_BINARY ? BUILDER_RUN_BYTES
           : field->layout == DORA_LAYOUT_BITS                                                ? BUILDER_RUN_BITS
                                                                                              : BUILDER_RUN_VALUES;
}

static bool run_accepts(struct builder_state* state,
                        uint32_t number,
                        enum builder_run_kind kind,
                        enum dora_physical_type physical) {
    const struct builder_node_state* const node = &state->nodes[number];
    const struct builder_buffer_state* const buffers = &state->buffers[node->first_buffer];
    const enum dora_layout layout = node->field->layout;
    if (!node->is_run_eligible || node->buffer_count < 2 || buffers[0].role != BUILDER_BUFFER_VALIDITY_BITS) {
        return false;
    }
    if (kind == BUILDER_RUN_VALUES || kind == BUILDER_RUN_BITS) {
        return buffers[1].unit_bit_width != 0 &&
               (kind == BUILDER_RUN_BITS
                    ? layout == DORA_LAYOUT_BITS
                    : layout == DORA_LAYOUT_FIXED && state->topology[number].view.physical_type == physical);
    }
    if (layout == DORA_LAYOUT_BINARY_VIEW) {
        return buffers[1].role == BUILDER_BUFFER_BINARY_VIEW;
    }
    if (layout == DORA_LAYOUT_BINARY || layout == DORA_LAYOUT_LARGE_BINARY) {
        return node->buffer_count >= 3 &&
               (buffers[1].role == BUILDER_BUFFER_OFFSETS32 || buffers[1].role == BUILDER_BUFFER_OFFSETS64) &&
               buffers[2].role == BUILDER_BUFFER_BYTES;
    }
    return buffers[1].unit_bit_width != 0 && layout == DORA_LAYOUT_FIXED &&
           state->topology[number].view.physical_type == physical;
}

static int run_row_limit(uint32_t number,
                         const struct builder_state* state,
                         uint64_t first,
                         uint64_t maximum,
                         uint64_t* out_limit) {
    const int status = append_limit(number, state, first, 1, out_limit);
    if (status != DORA_OK) {
        return status;
    }
    *out_limit = first + BUILDER_RUN_ROW_LIMIT;
    if (number == 0 && state->options.row_capacity != 0 && *out_limit > state->options.row_capacity) {
        *out_limit = state->options.row_capacity;
    }
    if (*out_limit > maximum) {
        *out_limit = maximum;
    }
    return DORA_OK;
}

static int run_bind(struct builder_operation* operation, uint32_t number, struct builder_column_run* cursor) {
    struct builder_state* const state = operation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const uint32_t values_index = node->first_buffer + 1;
    const uint64_t first = cursor->rows->row_count;
    uint64_t limit = cursor->row_limit;
    int status = DORA_OK;
    if (cursor->kind == BUILDER_RUN_BYTES) {
        const bool has_offsets = mutation_buffer(operation, values_index)->initialized_unit_count != 0;
        uint64_t offset_limit = limit + 1;
        status = run_reserve(values_index, operation, has_offsets ? first + 1 : 0, first + 2, &offset_limit);
        if (status == DORA_OK && !has_offsets) {
            store_offset(cursor->offset_bit_width, candidate_address(operation, values_index, 0), 0);
            mutation_buffer(operation, values_index)->initialized_unit_count = 1;
        }
        if (status == DORA_OK && offset_limit - 1 < limit) {
            limit = offset_limit - 1;
        }
    } else {
        status = run_reserve(values_index, operation, first, first + 1, &limit);
    }
    const bool has_validity = state->buffers[node->first_buffer].allocation.address != nullptr;
    if (status == DORA_OK && has_validity) {
        status = run_reserve(node->first_buffer, operation, first, first + 1, &limit);
    }
    if (status != DORA_OK) {
        return status;
    }
    struct builder_buffer_checkpoint* const candidates = state->mutation_buffers;
    cursor->values = &candidates[values_index];
    cursor->validity = has_validity ? &candidates[node->first_buffer] : nullptr;
    cursor->validity_words = has_validity ? candidate_bit_words(operation, node->first_buffer) : nullptr;
    cursor->value_base = candidate_address(operation, values_index, 0);
    cursor->limit = limit;
    return DORA_OK;
}

static int run_describe(struct builder_operation* operation, uint32_t number) {
    struct builder_state* const state = operation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field* const field = node->field;
    const bool is_view = field->layout == DORA_LAYOUT_BINARY_VIEW;
    const bool is_bytes = field->layout == DORA_LAYOUT_BINARY || field->layout == DORA_LAYOUT_LARGE_BINARY;
    struct builder_row_state* const rows = mutation_rows(operation, number);
    uint64_t row_limit = 0;
    int status = run_row_limit(number, state, rows->row_count, is_bytes ? INT64_MAX - 1 : INT64_MAX, &row_limit);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t unit_bit_width = state->buffers[node->first_buffer + 1].unit_bit_width;
    const bool is_bits = field->layout == DORA_LAYOUT_BITS;
    state->runs.columns[number] = (struct builder_column_run){
        .rows = rows,
        .bytes = is_bytes ? mutation_buffer(operation, node->first_buffer + 2) : nullptr,
        .row_limit = row_limit,
        .byte_limit = is_view ? 0 : rows->byte_count,
        .width_bytes = is_bits || is_bytes ? 0 : unit_bit_width / 8,
        .offset_bit_width = is_bytes ? unit_bit_width : 0,
        .physical = is_bits || is_bytes ? DORA_PHYSICAL_TYPE_NONE : state->topology[number].view.physical_type,
        .kind = run_kind(field),
        .should_check_content =
            state->options.content_policy != DORA_FAILURE_POLICY_TRUST &&
            (((is_bytes || is_view) && field->type.id == DORA_TYPE_UTF8) || (!is_bytes && checks_fixed_content(field))),
    };
    struct builder_column_run* const cursor = &state->runs.columns[number];
    if (cursor->should_check_content && !is_bytes && !is_view) {
        cursor->content_bound = fixed_content_bound(field, cursor->physical);
    }
    return DORA_OK;
}

static const uint8_t null_value[32] = {};

struct run_row {
    const void* value;
    struct dora_bytes bytes;
    uint64_t byte_end;
    size_t width;
    bool bit;
    bool is_valid;
    enum vector_content_state content_state;
    bool is_content_carried;
};

// Requires KIND to match the cursor; mutates no storage.
#define DEFINE_RUN_CHECK(NAME, KIND, ...)                                                                              \
    static inline int NAME(const struct builder_state* state,                                                          \
                           const struct builder_column_run* cursor,                                                    \
                           uint32_t number,                                                                            \
                           struct run_row* row __VA_OPT__(, ) __VA_ARGS__) {                                           \
        int status = DORA_OK;                                                                                          \
        if ((KIND) == BUILDER_RUN_FIXED_BINARY && row->is_valid && row->bytes.byte_count != cursor->width_bytes) {     \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        if ((KIND) == BUILDER_RUN_FIXED_BINARY) {                                                                      \
            row->value = row->bytes.data;                                                                              \
            row->width = cursor->width_bytes;                                                                          \
        }                                                                                                              \
        if (row->is_valid && ((KIND) == BUILDER_RUN_BYTES || (KIND) == BUILDER_RUN_VIEW) &&                            \
            cursor->should_check_content && !row->is_content_carried) {                                                \
            bool ascii = false;                                                                                        \
            status = dora_utf8_validate_span(row->bytes, &ascii);                                                      \
            row->content_state = ascii ? VECTOR_CONTENT_VALIDATED_ASCII : VECTOR_CONTENT_VALIDATED;                    \
        } else if (row->is_valid && (KIND) == BUILDER_RUN_VALUES && cursor->should_check_content &&                    \
                   !row->is_content_carried) {                                                                         \
            row->content_state = VECTOR_CONTENT_VALIDATED;                                                             \
            status = fixed_content(state->nodes[number].field, cursor->physical, &cursor->content_bound, row->value);  \
            if (status == EOVERFLOW) {                                                                                 \
                status = ERANGE;                                                                                       \
            }                                                                                                          \
        } else if (!row->is_content_carried) {                                                                         \
            const struct dora_schema_field* field = state->nodes[number].field;                                        \
            row->content_state = state->options.content_policy == DORA_FAILURE_POLICY_TRUST &&                         \
                                         (field->type.id == DORA_TYPE_UTF8 || checks_fixed_content(field))             \
                                     ? VECTOR_CONTENT_PROMISED                                                         \
                                     : VECTOR_CONTENT_VALIDATED;                                                       \
        }                                                                                                              \
        if (status != DORA_OK) {                                                                                       \
            if ((status == EILSEQ || status == ERANGE) && state->options.content_policy == DORA_FAILURE_POLICY_NULL) { \
                row->is_valid = false;                                                                                 \
            } else {                                                                                                   \
                return status;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        if (!row->is_valid && (state->nodes[number].field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {         \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        if ((KIND) == BUILDER_RUN_BYTES || (KIND) == BUILDER_RUN_VIEW) {                                               \
            if (!row->is_valid) {                                                                                      \
                row->bytes = (struct dora_bytes){};                                                                    \
            }                                                                                                          \
            if (ckd_add(&row->byte_end,                                                                                \
                        cursor->rows->byte_count,                                                                      \
                        ((KIND) == BUILDER_RUN_VIEW && row->bytes.byte_count <= VIEW_INLINE_BYTE_COUNT)                \
                            ? 0                                                                                        \
                            : row->bytes.byte_count) ||                                                                \
                row->byte_end > (cursor->offset_bit_width == 32 ? (uint64_t)INT32_MAX : (uint64_t)INT64_MAX) ||        \
                ((KIND) == BUILDER_RUN_VIEW && row->bytes.byte_count > INT32_MAX)) {                                   \
                return EOVERFLOW;                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
DEFINE_RUN_CHECK(run_check, kind, enum builder_run_kind kind)
DEFINE_RUN_CHECK(run_check_bytes, BUILDER_RUN_BYTES)
DEFINE_RUN_CHECK(run_check_view, BUILDER_RUN_VIEW)
DEFINE_RUN_CHECK(run_check_fixed_binary, BUILDER_RUN_FIXED_BINARY)
#undef DEFINE_RUN_CHECK

static inline bool
run_fits(const struct builder_column_run* cursor, const struct run_row* row, enum builder_run_kind kind) {
    return (row->is_valid || cursor->validity_words != nullptr) &&
           (kind != BUILDER_RUN_BYTES || row->byte_end <= cursor->byte_limit) &&
           (kind != BUILDER_RUN_VIEW || row->bytes.byte_count <= VIEW_INLINE_BYTE_COUNT ||
            (cursor->bytes != nullptr &&
             row->bytes.byte_count <= cursor->byte_limit - cursor->bytes->initialized_unit_count));
}

static inline void run_count(struct builder_column_run* cursor, const struct run_row* row, uint64_t index) {
    if (cursor->validity_words != nullptr) {
        bitmap_assign(cursor->validity_words, index, row->is_valid);
        cursor->validity->initialized_unit_count = index + 1;
    }
    if (row->is_valid) {
        const enum vector_content_state current = cursor->rows->content_state;
        cursor->rows->content_state = row->content_state < current ? row->content_state : current;
    }
    cursor->rows->row_count = index + 1;
    cursor->rows->null_count += row->is_valid ? 0 : 1;
}

static inline void
run_put_values(struct builder_column_run* cursor, const struct run_row* row, enum builder_run_kind kind) {
    const uint64_t index = cursor->rows->row_count;
    if (kind == BUILDER_RUN_BITS) {
        bitmap_assign(cursor->value_base, index, row->is_valid && row->bit);
    } else if (row->is_valid) {
        memcpy((uint8_t*)cursor->value_base + (size_t)index * row->width, row->value, row->width);
    } else {
        memset((uint8_t*)cursor->value_base + (size_t)index * cursor->width_bytes, 0, cursor->width_bytes);
    }
    cursor->values->initialized_unit_count = index + 1;
    run_count(cursor, row, index);
}

static inline int run_put_bytes(struct builder_column_run* cursor, const struct run_row* row) {
    const uint64_t index = cursor->rows->row_count;
    if (row->bytes.byte_count != 0) {
        memcpy((uint8_t*)cursor->byte_base + cursor->rows->byte_count, row->bytes.data, row->bytes.byte_count);
    }
    store_offset(cursor->offset_bit_width,
                 (uint8_t*)cursor->value_base + (index + 1) * (cursor->offset_bit_width / 8),
                 row->byte_end);
    cursor->rows->byte_count = row->byte_end;
    cursor->values->initialized_unit_count = index + 2;
    cursor->bytes->initialized_unit_count = row->byte_end;
    run_count(cursor, row, index);
    return DORA_OK;
}

static inline void run_put_view(struct builder_state* state,
                                uint32_t number,
                                struct builder_column_run* cursor,
                                const struct run_row* row) {
    const uint64_t index = cursor->rows->row_count;
    store_view(number, state, index, row->bytes);
    run_count(cursor, row, index);
}

static int run_place(struct builder_operation* operation, uint32_t number, const struct run_row* row) {
    struct builder_state* const state = operation->state;
    struct builder_column_run* const cursor = &state->runs.columns[number];
    const uint32_t first = state->nodes[number].first_buffer;
    int status = cursor->rows->row_count >= cursor->limit ? run_bind(operation, number, cursor) : DORA_OK;
    if (status == DORA_OK && cursor->kind == BUILDER_RUN_BYTES && row->byte_end > cursor->byte_limit) {
        uint64_t limit = UINT64_MAX;
        status = run_reserve(first + 2, operation, cursor->rows->byte_count, row->byte_end, &limit);
        if (status == DORA_OK) {
            cursor->byte_base = candidate_address(operation, first + 2, 0);
            cursor->byte_limit = limit;
        }
    }
    if (status == DORA_OK && cursor->kind == BUILDER_RUN_VIEW && row->bytes.byte_count > VIEW_INLINE_BYTE_COUNT &&
        !run_fits(cursor, row, BUILDER_RUN_VIEW)) {
        uint32_t index;
        status = reserve_view(number, operation, row->bytes.byte_count, &index);
        if (status == DORA_OK) {
            cursor->bytes = &state->mutation_buffers[index];
            const uint64_t capacity = state->buffers[index].allocation.capacity_bytes;
            cursor->byte_limit = capacity > INT32_MAX ? INT32_MAX : capacity;
        }
    }
    if (status == DORA_OK && !row->is_valid && cursor->validity_words == nullptr) {
        status = reserve_units(first, operation, 0, cursor->limit);
        if (status == DORA_OK) {
            status = materialize_validity(operation, number, cursor->rows->row_count);
        }
        if (status == DORA_OK) {
            cursor->validity = &state->mutation_buffers[first];
            cursor->validity_words = candidate_bit_words(operation, first);
        }
    }
    if (status == DORA_OK && cursor->kind == BUILDER_RUN_BYTES) {
        status = run_put_bytes(cursor, row);
    } else if (status == DORA_OK && cursor->kind == BUILDER_RUN_VIEW) {
        run_put_view(state, number, cursor, row);
    } else if (status == DORA_OK) {
        run_put_values(cursor, row, cursor->kind);
    }
    return status;
}

static int run_slow(const struct dora_builder* builder,
                    struct builder_state* state,
                    struct run_row* row,
                    enum builder_run_kind kind) {
    const uint32_t number = builder->node_index;
    const struct builder_column_run* const current = &state->runs.columns[number];
    const bool fresh = state->runs.operation.target.handle.slot_index == 0 ||
                       (current->is_open && current->rows->row_count >= current->row_limit);
    struct builder_operation operation = {};
    int status = fresh ? dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &operation)
                       : dora_builder_operation_resume(&state->runs.operation, BUILDER_PHASE_OPEN, &operation);
    if (status != DORA_OK) {
        return status;
    }
    state = operation.state;
    if (fresh) {
        state->runs.records = state->completed_root_row_count;
    }
    struct builder_column_run* const cursor = &state->runs.columns[number];
    if (!cursor->is_open) {
        status = run_describe(&operation, number);
        if (status == DORA_OK) {
            status = run_bind(&operation, number, cursor);
        }
        if (status != DORA_OK) {
            goto failed;
        }
    }
    status = run_check(state, cursor, number, row, kind);
    if (status != DORA_OK) {
        goto failed;
    }
    cursor->is_open = true;
    status = run_place(&operation, number, row);
park:
    return dora_builder_operation_park(&operation, &state->runs.operation, status, nullptr);
failed:
    if (!cursor->is_open) {
        *cursor = (struct builder_column_run){};
    }
    if (fresh) {
        mutation_finish(&operation, false);
        return dora_builder_operation_leave(&operation, status, nullptr);
    }
    goto park;
}

static int run_grow(struct builder_state* state, uint32_t number, const struct run_row* row) {
    struct builder_operation operation = {};
    const int status = dora_builder_operation_resume(&state->runs.operation, BUILDER_PHASE_OPEN, &operation);
    if (status != DORA_OK) {
        return status;
    }
    return dora_builder_operation_park(&operation, &state->runs.operation, run_place(&operation, number, row), nullptr);
}

// Requires KIND to match the cursor and CHECK to validate row.
#define DEFINE_RUN_STORE(NAME, KIND, CHECK, ...)                                                                       \
    static inline int NAME(struct builder_state* state,                                                                \
                           struct builder_column_run* cursor,                                                          \
                           uint32_t number,                                                                            \
                           struct run_row* row __VA_OPT__(, ) __VA_ARGS__) {                                           \
        const int status = CHECK;                                                                                      \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        if (!run_fits(cursor, row, KIND)) {                                                                            \
            return run_grow(state, number, row);                                                                       \
        }                                                                                                              \
        if ((KIND) == BUILDER_RUN_BYTES) {                                                                             \
            return run_put_bytes(cursor, row);                                                                         \
        } else if ((KIND) == BUILDER_RUN_VIEW) {                                                                       \
            run_put_view(state, number, cursor, row);                                                                  \
        } else {                                                                                                       \
            run_put_values(cursor, row, KIND);                                                                         \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
DEFINE_RUN_STORE(run_store, kind, run_check(state, cursor, number, row, kind), enum builder_run_kind kind)
DEFINE_RUN_STORE(run_store_bytes, BUILDER_RUN_BYTES, run_check_bytes(state, cursor, number, row))
DEFINE_RUN_STORE(run_store_view, BUILDER_RUN_VIEW, run_check_view(state, cursor, number, row))
DEFINE_RUN_STORE(run_store_fixed_binary, BUILDER_RUN_FIXED_BINARY, run_check_fixed_binary(state, cursor, number, row))
#undef DEFINE_RUN_STORE

static inline int
run_append(struct dora_builder* builder, enum dora_physical_type physical, const void* value, size_t width) {
    struct builder_state* state = nullptr;
    const int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase != BUILDER_PHASE_OPEN) {
        return EBUSY;
    }
    struct run_row row = {.value = value, .width = width, .is_valid = true};
    struct builder_column_run* const cursor = &state->runs.columns[builder->node_index];
    // NONE excludes closed, bit and bytes cursors from fixed appends.
    if (cursor->physical != physical || cursor->rows->row_count >= cursor->limit) {
        if (!run_accepts(state, builder->node_index, BUILDER_RUN_VALUES, physical)) {
            return append_public_error(builder,
                                       &(struct builder_input){.values = value, .row_count = 1, .physical = physical},
                                       nullptr);
        }
        return run_slow(builder, state, &row, BUILDER_RUN_VALUES);
    }
    return run_store(state, cursor, builder->node_index, &row, BUILDER_RUN_VALUES);
}

static int
append_text_transaction(struct dora_builder* builder, struct dora_utf8_input input, enum temporal_parse_mode grammar) {
    const char* text = (const char*)input.bytes.data;
    const size_t byte_count = input.bytes.byte_count;
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_schema_field* const field = mutation.state->nodes[mutation.data.target.node_index].field;
    if (status == DORA_OK && text == nullptr && byte_count != 0) {
        status = EINVAL;
    }
    if (status == DORA_OK && field->layout != DORA_LAYOUT_DICTIONARY &&
        (field->type.id == DORA_TYPE_UTF8 || field->type.id == DORA_TYPE_BINARY ||
         field->type.id == DORA_TYPE_FIXED_BINARY)) {
        const struct dora_bytes value = {.data = (const uint8_t*)text, .byte_count = byte_count};
        status =
            append_input(&mutation,
                         mutation.data.target.node_index,
                         &(struct builder_input){
                             .values = &value,
                             .row_count = 1,
                             .is_byte_input = true,
                             .content_state = input.proof == DORA_UTF8_VALIDATED_ASCII ? VECTOR_CONTENT_VALIDATED_ASCII
                                              : input.proof == DORA_UTF8_VALIDATED     ? VECTOR_CONTENT_VALIDATED
                                                                                       : VECTOR_CONTENT_PROMISED,
                             .is_content_carried = input.proof != DORA_UTF8_CHECKED
                         },
                         BUILDER_APPEND_INPUT);
    } else if (status == DORA_OK) {
        struct dora_logical_type type =
            field->layout == DORA_LAYOUT_DICTIONARY ? (struct dora_logical_type){.id = field->index_type} : field->type;
        type.extension_name = (struct dora_bytes){};
        type.extension_metadata = (struct dora_bytes){};
        enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
        status = type_physical(&type, &physical);
        union decoded_row_value value = {};
        const enum dora_rounding_mode rounding = type.id >= DORA_TYPE_FLOAT16 && type.id <= DORA_TYPE_FLOAT64
                                                     ? DORA_ROUNDING_MODE_HALF_EVEN
                                                     : DORA_ROUNDING_MODE_EXACT;
        if (status == DORA_OK) {
            if (grammar == TEMPORAL_PARSE_READER && type.id >= DORA_TYPE_DATE32 && type.id <= DORA_TYPE_DURATION) {
                status =
                    dora_temporal_parse_type(&type, physical, text, byte_count, rounding, grammar, nullptr, &value);
                status = status == EINVAL ? EILSEQ : status;
            } else {
                status = dora_type_parse_scalar(&type, physical, text, byte_count, rounding, &value);
            }
        }
        if ((status == EILSEQ || status == ERANGE) &&
            mutation.state->options.content_policy == DORA_FAILURE_POLICY_NULL) {
            status = append_input(&mutation,
                                  mutation.data.target.node_index,
                                  &(struct builder_input){
                                      .row_count = 1,
                                      .is_all_null = true,
                                      .is_masked = false,
                                      .are_rows_admitted = false
                                  },
                                  BUILDER_APPEND_NULL_ROWS);
        } else if (status == EILSEQ) {
            status = EINVAL;
        } else if (status == DORA_OK) {
            status = append_input(&mutation,
                                  mutation.data.target.node_index,
                                  &(struct builder_input){
                                      .values = &value,
                                      .row_count = 1,
                                      .physical = physical,
                                      .is_boolean = type.id == DORA_TYPE_BOOLEAN
                                  },
                                  BUILDER_APPEND_INPUT);
        }
    }
    return mutation_complete(&mutation, status, false, nullptr);
}

// Text parses non-byte fields; bytes preserve per-call transactions.
#define DEFINE_RUN_APPEND_BYTES(NAME, PROOF, VALUE, IS_TEXT, GRAMMAR, ...)                                             \
    int NAME(__VA_ARGS__) {                                                                                            \
        const struct dora_utf8_input input = {VALUE, PROOF};                                                           \
        const enum dora_utf8_proof proof = input.proof;                                                                \
        const struct dora_bytes value = input.bytes;                                                                   \
        const bool is_text = IS_TEXT;                                                                                  \
        struct builder_state* state = nullptr;                                                                         \
        const int status = dora_builder_resolve_state(builder, &state);                                                \
        if (status != DORA_OK) {                                                                                       \
            return status;                                                                                             \
        }                                                                                                              \
        if (state->phase != BUILDER_PHASE_OPEN) {                                                                      \
            return EBUSY;                                                                                              \
        }                                                                                                              \
        if (value.byte_count != 0 && value.data == nullptr) {                                                          \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        struct run_row row = {                                                                                         \
            .value = null_value,                                                                                       \
            .bytes = value,                                                                                            \
            .is_valid = true,                                                                                          \
            .content_state = proof == DORA_UTF8_VALIDATED_ASCII ? VECTOR_CONTENT_VALIDATED_ASCII                       \
                             : proof == DORA_UTF8_VALIDATED     ? VECTOR_CONTENT_VALIDATED                             \
                                                                : VECTOR_CONTENT_PROMISED,                             \
            .is_content_carried = proof != DORA_UTF8_CHECKED                                                           \
        };                                                                                                             \
        struct builder_column_run* const cursor = &state->runs.columns[builder->node_index];                           \
        if (cursor->kind == BUILDER_RUN_BYTES && cursor->rows->row_count < cursor->limit) {                            \
            return run_store_bytes(state, cursor, builder->node_index, &row);                                          \
        }                                                                                                              \
        if (cursor->kind < BUILDER_RUN_BYTES || cursor->rows->row_count >= cursor->limit) {                            \
            if (!run_accepts(state, builder->node_index, BUILDER_RUN_BYTES, DORA_PHYSICAL_TYPE_NONE)) {                \
                return is_text ? append_text_transaction(builder, (struct dora_utf8_input){value, proof}, GRAMMAR)     \
                               : dora_builder_append_bytes_error(builder, value, nullptr);                             \
            }                                                                                                          \
            return run_slow(builder, state, &row, run_kind(state->nodes[builder->node_index].field));                  \
        }                                                                                                              \
        return cursor->kind == BUILDER_RUN_VIEW ? run_store_view(state, cursor, builder->node_index, &row)             \
                                                : run_store_fixed_binary(state, cursor, builder->node_index, &row);    \
    }

static int append_source_publish(uint32_t number,
                                 struct builder_operation* mutation,
                                 const struct dora_vector_view* view,
                                 struct dora_range range,
                                 uint64_t child_begin) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field* const field = node->field;
    const bool record = field->layout == DORA_LAYOUT_STRUCT;
    const bool fixed_list = field->layout == DORA_LAYOUT_FIXED_LIST;
    const bool list_view = field->layout == DORA_LAYOUT_LIST_VIEW || field->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    const uint64_t begin = rows->row_count;
    const uint64_t end = begin + range.length;
    const uint64_t stop = range.offset + range.length;
    const uint32_t first_child = state->topology[number].first_child_index;
    int status = DORA_OK;
    const uint64_t child_end = first_child == UINT32_MAX ? 0 : mutation_read_rows(mutation, first_child)->row_count;
    const bool offsets = !record && !fixed_list;
    if (offsets) {
        if (child_end > INT64_MAX ||
            (state->buffers[node->first_buffer + 1].unit_bit_width == 32 && child_end > INT32_MAX)) {
            return EOVERFLOW;
        }
        const uint64_t committed_offsets = mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count;
        status = reserve_units(node->first_buffer + 1,
                               mutation,
                               begin + (!list_view && committed_offsets != 0),
                               end + !list_view);
        if (status == DORA_OK && list_view) {
            status = reserve_units(node->first_buffer + 2, mutation, begin, end);
        }
        if (status == DORA_OK && list_view) {
            status = reserve_units(node->frontier_buffer, mutation, begin, end);
        }
        if (status == DORA_OK && !list_view && committed_offsets == 0) {
            store_offset(state->buffers[node->first_buffer + 1].unit_bit_width,
                         candidate_address(mutation, node->first_buffer + 1, 0),
                         0);
        }
    }
    uint64_t child_row = child_begin;
    for (uint64_t row = range.offset; status == DORA_OK && row < stop; ++row) {
        const bool valid = !dora_vector_view_is_null(view, row);
        const struct dora_range elements =
            record || !valid ? (struct dora_range){} : dora_vector_view_get_elements(view, row);
        if (status == DORA_OK) {
            status = append_validity(mutation, number, field, end, valid, false, false);
        }
        if (status != DORA_OK) {
            break;
        }
        if (offsets) {
            const uint32_t width = state->buffers[node->first_buffer + 1].unit_bit_width;
            store_offset(width,
                         candidate_address(mutation, node->first_buffer + 1, rows->row_count + !list_view),
                         list_view ? child_row : child_row + elements.length);
            if (list_view) {
                store_offset(width,
                             candidate_address(mutation, node->first_buffer + 2, rows->row_count),
                             elements.length);
                store_offset(64,
                             candidate_address(mutation, node->frontier_buffer, rows->row_count),
                             child_row + elements.length);
            }
            child_row += elements.length;
        }
        ++rows->row_count;
    }
    if (status == DORA_OK) {
        for (uint32_t child = first_child; child != UINT32_MAX; child = state->topology[child].next_sibling_index) {
            struct builder_row_state* const child_rows = mutation_rows(mutation, child);
            child_rows->closed_row_count = child_rows->row_count;
        }
        if (!record) {
            rows->child_row_count = child_end;
        }
        if (offsets) {
            mutation_buffer(mutation, node->first_buffer + 1)->initialized_unit_count = end + !list_view;
            if (list_view) {
                mutation_buffer(mutation, node->first_buffer + 2)->initialized_unit_count = end;
                mutation_buffer(mutation, node->frontier_buffer)->initialized_unit_count = end;
            }
        }
    }
    return status;
}

static int append_source_range(uint32_t number,
                               struct builder_operation* mutation,
                               const struct vector_bound_input* input,
                               uint32_t source,
                               struct dora_range range) {
    struct builder_state* const state = mutation->state;
    const struct builder_node_state* const node = &state->nodes[number];
    const struct dora_schema_field* const field = node->field;
    const struct dora_vector_view* const view = &input->nodes[source].view;
    if (range.offset > view->length || range.length > view->length - range.offset) {
        return EINVAL;
    }
    if (range.length == 0) {
        return DORA_OK;
    }
    struct builder_row_state* const standing = mutation_rows(mutation, number);
    const enum vector_content_state content_state = input->nodes[source].image->node.content_state;
    if (content_state < standing->content_state) {
        standing->content_state = content_state;
    }
    const uint64_t stop = range.offset + range.length;
    int status;
    if (field->layout == DORA_LAYOUT_NULL) {
        return append_input(mutation,
                            number,
                            &(struct builder_input){
                                .is_content_carried = true,
                                .content_state = content_state,
                                .row_count = range.length,
                                .is_all_null = true,
                                .is_masked = false,
                                .are_rows_admitted = false
                            },
                            BUILDER_APPEND_NULL_ROWS);
    }
    if (field->layout == DORA_LAYOUT_RUN_END) {
        const uint32_t values = builder_child(number, state, 1);
        if (view->layout == DORA_LAYOUT_RUN_END) {
            const uint32_t source_values = dora_vector_input_child(source, input, 1);
            struct dora_run_cursor cursor = {.row = range.offset};
            while (cursor.row < stop) {
                uint64_t physical = 0;
                struct dora_range run = dora_vector_view_next_run(view, &cursor, &physical);
                if (run.length == 0) {
                    return EINVAL;
                }
                if (run.length > stop - run.offset) {
                    run.length = stop - run.offset;
                }
                for (uint64_t row = run.offset; row < run.offset + run.length;) {
                    const bool valid = !dora_vector_view_is_null(view, row);
                    uint64_t end = row + 1;
                    while (end < run.offset + run.length && dora_vector_view_is_null(view, end) != valid) {
                        ++end;
                    }
                    status = valid ? append_source_range(values,
                                                         mutation,
                                                         input,
                                                         source_values,
                                                         (struct dora_range){physical, 1})
                                   : append_input(mutation,
                                                  values,
                                                  &(struct builder_input){
                                                      .is_content_carried = true,
                                                      .content_state = content_state,
                                                      .row_count = 1,
                                                      .is_all_null = true,
                                                      .is_masked = false,
                                                      .are_rows_admitted = false
                                                  },
                                                  BUILDER_APPEND_NULL_ROWS);
                    if (status == DORA_OK) {
                        status = append_run_end(number, mutation, (struct run_end_rows){.length = end - row});
                    }
                    if (status != DORA_OK) {
                        return status;
                    }
                    row = end;
                }
            }
        } else {
            for (uint64_t row = range.offset; row < stop; ++row) {
                status = append_source_range(values, mutation, input, source, (struct dora_range){row, 1});
                if (status == DORA_OK) {
                    status = append_run_end(number, mutation, (struct run_end_rows){.length = 1});
                }
                if (status != DORA_OK) {
                    return status;
                }
            }
        }
        return DORA_OK;
    }
    if (field->layout == DORA_LAYOUT_DICTIONARY) {
        if (view->layout != DORA_LAYOUT_DICTIONARY) {
            return EINVAL;
        }
        return append_fixed(mutation,
                            number,
                            field,
                            &(struct builder_input){
                                .is_content_carried = true,
                                .content_state = content_state,
                                .view = view,
                                .first = range.offset,
                                .row_count = range.length,
                                .physical = state->topology[number].view.physical_type
                            });
    }
    if (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END) {
        const uint32_t values = dora_vector_input_child(source, input, view->layout == DORA_LAYOUT_RUN_END ? 1 : 0);
        const struct dora_vector_view* const value_view = &input->nodes[values].view;
        for (uint64_t row = range.offset; row < stop;) {
            const bool valid = !dora_vector_view_is_null(view, row);
            uint64_t through = row + 1;
            if (!valid) {
                while (through < stop && dora_vector_view_is_null(view, through)) {
                    ++through;
                }
                status = append_input(mutation,
                                      number,
                                      &(struct builder_input){
                                          .is_content_carried = true,
                                          .content_state = content_state,
                                          .row_count = through - row,
                                          .is_all_null = true,
                                          .is_masked = false,
                                          .are_rows_admitted = false
                                      },
                                      BUILDER_APPEND_NULL_ROWS);
                if (status != DORA_OK) {
                    return status;
                }
                row = through;
                continue;
            }
            uint64_t physical = 0;
            if (view->layout == DORA_LAYOUT_DICTIONARY) {
                status = dora_vector_view_read_index(view, row, &physical);
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                physical = dora_vector_view_find_run(view, row);
            }
            bool constant = true;
            while (through < stop && !dora_vector_view_is_null(view, through)) {
                uint64_t next = 0;
                if (view->layout == DORA_LAYOUT_DICTIONARY) {
                    status = dora_vector_view_read_index(view, through, &next);
                    if (status != DORA_OK) {
                        return status;
                    }
                } else {
                    next = dora_vector_view_find_run(view, through);
                }
                if (through == row + 1) {
                    constant = next == physical;
                }
                if (next != physical + (constant ? 0 : through - row)) {
                    break;
                }
                ++through;
            }
            const bool plain =
                value_view->layout != DORA_LAYOUT_DICTIONARY && value_view->layout != DORA_LAYOUT_RUN_END;
            const bool bytes = field->type.id == DORA_TYPE_UTF8 || field->type.id == DORA_TYPE_BINARY ||
                               field->type.id == DORA_TYPE_FIXED_BINARY;
            if (constant && plain &&
                (bytes || field->layout == DORA_LAYOUT_FIXED || field->layout == DORA_LAYOUT_BITS)) {
                const struct builder_input repeated = {
                    .is_content_carried = true,
                    .content_state = input->nodes[values].image->node.content_state,
                    .view = value_view,
                    .first = physical,
                    .row_count = through - row,
                    .physical = value_view->physical_type,
                    .is_boolean = value_view->layout == DORA_LAYOUT_BITS,
                    .is_byte_input = bytes,
                    .is_constant = true
                };
                status = bytes ? append_bytes(mutation, number, field, &repeated)
                               : append_fixed(mutation, number, field, &repeated);
                if (status != DORA_OK) {
                    return status;
                }
            } else if (!constant) {
                status =
                    append_source_range(number, mutation, input, values, (struct dora_range){physical, through - row});
                if (status != DORA_OK) {
                    return status;
                }
            } else {
                for (uint64_t occurrence = row; occurrence < through; ++occurrence) {
                    status = append_source_range(number, mutation, input, values, (struct dora_range){physical, 1});
                    if (status != DORA_OK) {
                        return status;
                    }
                }
            }
            row = through;
        }
        return DORA_OK;
    }
    if (field->type.id == DORA_TYPE_UTF8 || field->type.id == DORA_TYPE_BINARY ||
        field->type.id == DORA_TYPE_FIXED_BINARY) {
        return append_bytes(mutation,
                            number,
                            field,
                            &(struct builder_input){
                                .is_content_carried = true,
                                .content_state = content_state,
                                .view = view,
                                .first = range.offset,
                                .row_count = range.length,
                                .is_byte_input = true
                            });
    }
    if (field->layout == DORA_LAYOUT_FIXED || field->layout == DORA_LAYOUT_BITS) {
        return append_fixed(mutation,
                            number,
                            field,
                            &(struct builder_input){
                                .is_content_carried = true,
                                .content_state = content_state,
                                .view = view,
                                .first = range.offset,
                                .row_count = range.length,
                                .physical = view->physical_type,
                                .is_boolean = view->layout == DORA_LAYOUT_BITS
                            });
    }
    if (field->type.id == DORA_TYPE_UNION) {
        for (uint64_t row = range.offset; row < stop; ++row) {
            if (dora_vector_view_is_null(view, row)) {
                status = append_input(mutation,
                                      number,
                                      &(struct builder_input){
                                          .is_content_carried = true,
                                          .content_state = content_state,
                                          .row_count = 1,
                                          .is_all_null = true,
                                          .is_masked = false,
                                          .are_rows_admitted = false
                                      },
                                      BUILDER_APPEND_NULL_ROWS);
                if (status != DORA_OK) {
                    return status;
                }
                continue;
            }
            uint64_t physical = 0;
            const uint32_t member = dora_vector_view_get_member(view, row, &physical);
            status = append_source_range(builder_child(number, state, member),
                                         mutation,
                                         input,
                                         dora_vector_input_child(source, input, member),
                                         (struct dora_range){physical, 1});
            if (status == DORA_OK) {
                status = append_container(number,
                                          mutation,
                                          field->layout,
                                          (struct dora_range){},
                                          dora_vector_view_get_type_id(view, row),
                                          true,
                                          false,
                                          false);
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        return DORA_OK;
    }
    const bool record = field->layout == DORA_LAYOUT_STRUCT;
    const bool fixed_list = field->layout == DORA_LAYOUT_FIXED_LIST;
    struct builder_row_state* const rows = mutation_rows(mutation, number);
    const uint64_t begin = rows->row_count;
    uint64_t end = 0;
    status = append_limit(number, state, begin, range.length, &end);
    const uint32_t first_child = state->topology[number].first_child_index;
    const uint64_t child_begin = first_child == UINT32_MAX ? 0 : mutation_read_rows(mutation, first_child)->row_count;
    for (uint32_t child = first_child; status == DORA_OK && child != UINT32_MAX;
         child = state->topology[child].next_sibling_index) {
        const struct builder_row_state* const child_rows = mutation_read_rows(mutation, child);
        if (child_rows->row_count != child_rows->closed_row_count) {
            return EINVAL;
        }
        const uint32_t source_child = dora_vector_input_child(source, input, state->topology[child].child_index);
        for (uint64_t row = range.offset; status == DORA_OK && row < stop;) {
            const bool valid = !dora_vector_view_is_null(view, row);
            uint64_t through = row + 1;
            struct dora_range elements = record  ? (struct dora_range){row, 1}
                                         : valid ? dora_vector_view_get_elements(view, row)
                                                 : (struct dora_range){};
            while (through < stop && dora_vector_view_is_null(view, through) != valid) {
                if (valid && !record) {
                    const struct dora_range next = dora_vector_view_get_elements(view, through);
                    if (next.offset != elements.offset + elements.length) {
                        break;
                    }
                    elements.length += next.length;
                } else if (record) {
                    ++elements.length;
                }
                ++through;
            }
            if (valid) {
                status = append_source_range(child, mutation, input, source_child, elements);
            } else if (record || fixed_list) {
                uint64_t count = through - row;
                if (fixed_list && ckd_mul(&count, count, field->type.parameters.list_element_count)) {
                    return EOVERFLOW;
                }
                status = append_input(mutation,
                                      child,
                                      &(struct builder_input){
                                          .is_content_carried = true,
                                          .content_state = content_state,
                                          .row_count = count,
                                          .is_all_null = true,
                                          .is_masked = true,
                                          .are_rows_admitted = false
                                      },
                                      BUILDER_APPEND_NULL_ROWS);
            }
            row = through;
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    return append_source_publish(number, mutation, view, range, child_begin);
}

int dora_builder_append_vector_error(struct dora_builder* builder,
                                     const struct dora_vector* source,
                                     struct dora_error* out_error) {
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    struct builder_state* const state = mutation.state;
    mutation.is_content_carried = true;
    const struct dora_schema_field* const field = state->nodes[mutation.data.target.node_index].field;
    struct dora_vector input = {};
    status = dora_vector_retain(source, &input);
    struct scratch_scope scope = {};
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(state->storage.pool, &scope);
    }
    struct vector_bound_input bound = {};
    if (status == DORA_OK) {
        status = dora_vector_bind_input(&scope, &input, nullptr, &bound);
    }
    if (status == DORA_OK) {
        status = check_source(mutation.data.target.node_index, &bound, state, &input, 0);
    }
    // Cast rounding is half away from zero for decimal scale-down, nearest even for floating-point targets.
    if (status == EINVAL && bound.node_count != 0 &&
        !dora_type_is_equal(&field->type, &bound.nodes[0].field.type, field->child_count)) {
        struct dora_vector cast = {};
        const bool floating = field->type.id >= DORA_TYPE_FLOAT16 && field->type.id <= DORA_TYPE_FLOAT64;
        status = dora_compute_cast_in_pool(state->storage.pool,
                                           &input,
                                           &field->type,
                                           floating ? DORA_ROUNDING_MODE_HALF_EVEN : DORA_ROUNDING_MODE_HALF_AWAY,
                                           state->options.content_policy == DORA_FAILURE_POLICY_TRUST
                                               ? DORA_FAILURE_POLICY_ERROR
                                               : state->options.content_policy,
                                           &cast);
        dora_vector_release(&input);
        input = cast;
        if (status == DORA_OK) {
            status = dora_vector_bind_input(&scope, &input, nullptr, &bound);
        }
        if (status == DORA_OK) {
            status = check_source(mutation.data.target.node_index, &bound, state, &input, 0);
        }
    }
    if (status == DORA_OK) {
        status = append_source_range(mutation.data.target.node_index,
                                     &mutation,
                                     &bound,
                                     0,
                                     (struct dora_range){.length = input.rows.length});
    }
    dora_scratch_scope_end(&scope);
    status = mutation_complete(&mutation, status, false, out_error);
    dora_vector_release(&input);
    return status;
}

int dora_builder_append_nulls_error(struct dora_builder* builder, uint64_t row_count, struct dora_error* out_error) {
    return append_public_error(builder,
                               &(struct builder_input){.row_count = row_count, .is_all_null = true},
                               out_error);
}

int dora_builder_append_scalar_error(struct dora_builder* builder,
                                     enum dora_physical_type physical,
                                     const void* value,
                                     struct dora_error* out_error) {
    return append_public_error(builder,
                               &(struct builder_input){.values = value, .row_count = 1, .physical = physical},
                               out_error);
}

int dora_builder_append_bytes_error(struct dora_builder* builder,
                                    struct dora_bytes value,
                                    struct dora_error* out_error) {
    return append_public_error(builder,
                               &(struct builder_input){.values = &value, .row_count = 1, .is_byte_input = true},
                               out_error);
}

int dora_builder_append_vector(struct dora_builder* builder, const struct dora_vector* source) {
    return dora_builder_append_vector_error(builder, source, nullptr);
}

int dora_builder_append_nulls(struct dora_builder* builder, uint64_t row_count) {
    return dora_builder_append_nulls_error(builder, row_count, nullptr);
}

int dora_builder_append_null(struct dora_builder* builder) {
    struct builder_state* state = nullptr;
    const int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase != BUILDER_PHASE_OPEN) {
        return EBUSY;
    }
    struct run_row row = {.value = null_value, .is_valid = false};
    struct builder_column_run* const cursor = &state->runs.columns[builder->node_index];
    if (!cursor->is_open || cursor->rows->row_count >= cursor->limit) {
        const enum builder_run_kind kind = run_kind(state->nodes[builder->node_index].field);
        if (!run_accepts(state, builder->node_index, kind, state->topology[builder->node_index].view.physical_type)) {
            return dora_builder_append_nulls(builder, 1);
        }
        return run_slow(builder, state, &row, kind);
    }
    return run_store(state, cursor, builder->node_index, &row, cursor->kind);
}

int dora_builder_append_packed_bool(struct dora_builder* builder,
                                    const uint8_t* bits,
                                    uint64_t bit_offset,
                                    const uint8_t* validity,
                                    uint64_t validity_bit_offset,
                                    uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .bits = bits,
                                   .bit_offset = bit_offset,
                                   .validity_bits = validity,
                                   .validity_bit_offset = validity_bit_offset,
                                   .row_count = row_count,
                                   .is_boolean = true
                               },
                               nullptr);
}

int dora_builder_append_values_bool(struct dora_builder* builder,
                                    const bool* values,
                                    const bool* validity,
                                    uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_NONE,
                                   .is_boolean = true
                               },
                               nullptr);
}

int dora_builder_append_bool(struct dora_builder* builder, bool value) {
    struct builder_state* state = nullptr;
    const int status = dora_builder_resolve_state(builder, &state);
    if (status != DORA_OK) {
        return status;
    }
    if (state->phase != BUILDER_PHASE_OPEN) {
        return EBUSY;
    }
    struct run_row row = {.value = null_value, .bit = value, .is_valid = true};
    struct builder_column_run* const cursor = &state->runs.columns[builder->node_index];
    if (cursor->kind != BUILDER_RUN_BITS || cursor->rows->row_count >= cursor->limit) {
        if (!run_accepts(state, builder->node_index, BUILDER_RUN_BITS, DORA_PHYSICAL_TYPE_NONE)) {
            return append_public_error(builder,
                                       &(struct builder_input){.values = &value, .row_count = 1, .is_boolean = true},
                                       nullptr);
        }
        return run_slow(builder, state, &row, BUILDER_RUN_BITS);
    }
    return run_store(state, cursor, builder->node_index, &row, BUILDER_RUN_BITS);
}

int dora_builder_append_values_i8(struct dora_builder* builder,
                                  const int8_t* values,
                                  const bool* validity,
                                  uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I8
                               },
                               nullptr);
}

int dora_builder_append_i8(struct dora_builder* builder, int8_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I8, &value, sizeof(value));
}

int dora_builder_append_values_i16(struct dora_builder* builder,
                                   const int16_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I16
                               },
                               nullptr);
}

int dora_builder_append_i16(struct dora_builder* builder, int16_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I16, &value, sizeof(value));
}

int dora_builder_append_values_i32(struct dora_builder* builder,
                                   const int32_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I32
                               },
                               nullptr);
}

int dora_builder_append_i32(struct dora_builder* builder, int32_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I32, &value, sizeof(value));
}

int dora_builder_append_values_i64(struct dora_builder* builder,
                                   const int64_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I64
                               },
                               nullptr);
}

int dora_builder_append_i64(struct dora_builder* builder, int64_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I64, &value, sizeof(value));
}

int dora_builder_append_values_i128(struct dora_builder* builder,
                                    const struct dora_i128* values,
                                    const bool* validity,
                                    uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I128
                               },
                               nullptr);
}

int dora_builder_append_i128(struct dora_builder* builder, struct dora_i128 value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I128, &value, sizeof(value));
}

int dora_builder_append_values_i256(struct dora_builder* builder,
                                    const struct dora_i256* values,
                                    const bool* validity,
                                    uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_I256
                               },
                               nullptr);
}

int dora_builder_append_i256(struct dora_builder* builder, struct dora_i256 value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_I256, &value, sizeof(value));
}

int dora_builder_append_values_u8(struct dora_builder* builder,
                                  const uint8_t* values,
                                  const bool* validity,
                                  uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_U8
                               },
                               nullptr);
}

int dora_builder_append_u8(struct dora_builder* builder, uint8_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_U8, &value, sizeof(value));
}

int dora_builder_append_values_u16(struct dora_builder* builder,
                                   const uint16_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_U16
                               },
                               nullptr);
}

int dora_builder_append_u16(struct dora_builder* builder, uint16_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_U16, &value, sizeof(value));
}

int dora_builder_append_values_u32(struct dora_builder* builder,
                                   const uint32_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_U32
                               },
                               nullptr);
}

int dora_builder_append_u32(struct dora_builder* builder, uint32_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_U32, &value, sizeof(value));
}

int dora_builder_append_values_u64(struct dora_builder* builder,
                                   const uint64_t* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_U64
                               },
                               nullptr);
}

int dora_builder_append_u64(struct dora_builder* builder, uint64_t value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_U64, &value, sizeof(value));
}

int dora_builder_append_values_u128(struct dora_builder* builder,
                                    const struct dora_u128* values,
                                    const bool* validity,
                                    uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_U128
                               },
                               nullptr);
}

int dora_builder_append_u128(struct dora_builder* builder, struct dora_u128 value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_U128, &value, sizeof(value));
}

int dora_builder_append_values_f16(struct dora_builder* builder,
                                   const struct dora_f16_bits* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_F16
                               },
                               nullptr);
}

int dora_builder_append_f16(struct dora_builder* builder, struct dora_f16_bits value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_F16, &value, sizeof(value));
}

int dora_builder_append_values_f32(struct dora_builder* builder,
                                   const float* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_F32
                               },
                               nullptr);
}

int dora_builder_append_f32(struct dora_builder* builder, float value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_F32, &value, sizeof(value));
}

int dora_builder_append_values_f64(struct dora_builder* builder,
                                   const double* values,
                                   const bool* validity,
                                   uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_F64
                               },
                               nullptr);
}

int dora_builder_append_f64(struct dora_builder* builder, double value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_F64, &value, sizeof(value));
}

int dora_builder_append_values_interval_day_time(struct dora_builder* builder,
                                                 const struct dora_interval_day_time* values,
                                                 const bool* validity,
                                                 uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME
                               },
                               nullptr);
}

int dora_builder_append_interval_day_time(struct dora_builder* builder, struct dora_interval_day_time value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME, &value, sizeof(value));
}

int dora_builder_append_values_interval_month_day_nano(struct dora_builder* builder,
                                                       const struct dora_interval_month_day_nano* values,
                                                       const bool* validity,
                                                       uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO
                               },
                               nullptr);
}

int dora_builder_append_interval_month_day_nano(struct dora_builder* builder,
                                                struct dora_interval_month_day_nano value) {
    return run_append(builder, DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO, &value, sizeof(value));
}

int dora_builder_append_values_bytes(struct dora_builder* builder,
                                     const struct dora_bytes* values,
                                     const bool* validity,
                                     uint64_t row_count) {
    return append_public_error(builder,
                               &(struct builder_input){
                                   .values = values,
                                   .validity = validity,
                                   .row_count = row_count,
                                   .physical = DORA_PHYSICAL_TYPE_NONE,
                                   .is_byte_input = true
                               },
                               nullptr);
}

DEFINE_RUN_APPEND_BYTES(dora_builder_append_bytes,
                        DORA_UTF8_CHECKED,
                        bytes,
                        false,
                        TEMPORAL_PARSE_CAST,
                        struct dora_builder* builder,
                        struct dora_bytes bytes)

int dora_builder_append_container_error(struct dora_builder* builder,
                                        enum dora_type type,
                                        struct dora_range elements,
                                        int8_t type_id,
                                        bool view,
                                        struct dora_error* out_error) {
    struct builder_operation mutation = {};
    int status = dora_builder_operation_begin(builder, BUILDER_ENTRY_OPEN, &mutation);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_schema_field field = {};
    status = dora_schema_get_field(&mutation.state->nodes[mutation.data.target.node_index].schema, &field);
    if (status == DORA_OK &&
        (field.type.id != type || (mutation.data.target.node_index == 0 && type == DORA_TYPE_STRUCT))) {
        status = EINVAL;
    }
    const bool view_layout = field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    if (status == DORA_OK && type == DORA_TYPE_LIST && ((view && !view_layout) || (!view && view_layout))) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = append_container(mutation.data.target.node_index,
                                  &mutation,
                                  field.layout,
                                  elements,
                                  type_id,
                                  true,
                                  false,
                                  false);
    }
    return mutation_complete(&mutation, status, true, out_error);
}

int dora_builder_append_list(struct dora_builder* builder, uint64_t element_count) {
    return dora_builder_append_container_error(builder,
                                               DORA_TYPE_LIST,
                                               (struct dora_range){.length = element_count},
                                               0,
                                               false,
                                               nullptr);
}

int dora_builder_append_list_view(struct dora_builder* builder, struct dora_range elements) {
    return dora_builder_append_container_error(builder, DORA_TYPE_LIST, elements, 0, true, nullptr);
}

int dora_builder_append_fixed_list(struct dora_builder* builder) {
    return dora_builder_append_container_error(builder, DORA_TYPE_FIXED_LIST, (struct dora_range){}, 0, false, nullptr);
}

int dora_builder_append_struct(struct dora_builder* builder) {
    return dora_builder_append_container_error(builder, DORA_TYPE_STRUCT, (struct dora_range){}, 0, false, nullptr);
}

int dora_builder_append_map(struct dora_builder* builder, uint64_t entry_count) {
    return dora_builder_append_container_error(builder,
                                               DORA_TYPE_MAP,
                                               (struct dora_range){.length = entry_count},
                                               0,
                                               false,
                                               nullptr);
}

int dora_builder_append_union(struct dora_builder* builder, int8_t type_id) {
    return dora_builder_append_container_error(builder,
                                               DORA_TYPE_UNION,
                                               (struct dora_range){},
                                               type_id,
                                               false,
                                               nullptr);
}

DEFINE_RUN_APPEND_BYTES(dora_builder_append_text,
                        DORA_UTF8_CHECKED,
                        ((struct dora_bytes){(const uint8_t*)text, byte_count}),
                        true,
                        TEMPORAL_PARSE_CAST,
                        struct dora_builder* builder,
                        const char* text,
                        size_t byte_count)

DEFINE_RUN_APPEND_BYTES(dora_builder_append_reader_text,
                        input_proof,
                        ((struct dora_bytes){(const uint8_t*)text, byte_count}),
                        true,
                        TEMPORAL_PARSE_READER,
                        struct dora_builder* builder,
                        const char* text,
                        size_t byte_count,
                        enum dora_utf8_proof input_proof)
#undef DEFINE_RUN_APPEND_BYTES
