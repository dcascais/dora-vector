// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "compute/bytes_source.h"
#include "compute/output.h"
#include "dora/vector/bitmap.h"
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
#include "types/array_count.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "utf8/transform.h"
#include "utf8/utf8.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Literal Split
//
//------------------------------------------------------------------------------

static int utf8_split_row(struct dora_utf8_input input,
                          const struct dora_utf8_input* separator,
                          const struct dora_view_buffer* backing,
                          uint32_t buffer_index,
                          bool write,
                          uint64_t* item,
                          size_t* byte,
                          struct compute_bytes_result* values) {
    const struct dora_bytes source = input.bytes;
    size_t start = 0;
    for (;;) {
        size_t position = source.byte_count;
        bool found = false;
        int status = DORA_OK;
        if (separator != nullptr && source.byte_count != 0) {
            if (start < source.byte_count && utf8_byte_is_continuation(source.data[start])) {
                return EILSEQ;
            }
            if (separator->bytes.byte_count == 0) {
                status = utf8_advance(source, start, &position);
                found = position < source.byte_count;
            } else {
                status = dora_utf8_find(input, *separator, start, &position, &found);
            }
        }
        struct dora_bytes piece = {source.data == nullptr ? nullptr : source.data + start, position - start};
        const bool referenced =
            piece.byte_count > sizeof(((struct dora_binary_view_entry*)nullptr)->payload.inline_bytes);
        size_t next = *byte;
        if (status == DORA_OK && (piece.byte_count > INT32_MAX || *item == INT64_MAX ||
                                  (referenced && (ckd_add(&next, next, piece.byte_count) || next > INT64_MAX)))) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK && write && (*item >= values->stored || next > values->data_bytes)) {
            status = EINVAL;
        }
        if (status == DORA_OK && write) {
            uint8_t* const copied = values->buffers[2].address;
            uint32_t index = buffer_index;
            if (referenced && copied != nullptr) {
                memcpy(copied + *byte, piece.data, piece.byte_count);
                piece.data = copied + *byte;
                index = (uint32_t)(*byte / ((size_t)INT32_MAX + 1));
            }
            dora_compute_bytes_view_entry(piece,
                                          referenced ? backing[index].data : nullptr,
                                          index,
                                          &((struct dora_binary_view_entry*)values->buffers[1].address)[*item]);
        }
        if (status != DORA_OK) {
            return status;
        }
        ++*item;
        *byte = next;
        if (!found) {
            return DORA_OK;
        }
        start = position + separator->bytes.byte_count;
    }
}

static int utf8_split_rows(struct compute_bytes_source* source,
                           struct dora_utf8_input separator,
                           bool has_separator,
                           bool write,
                           const struct dora_view_buffer* backing,
                           struct compute_bytes_result* lists,
                           struct compute_bytes_result* values) {
    const uint8_t* const records =
        write && values->buffers[2].address == nullptr ? dora_compute_bytes_view_records(source) : nullptr;
    int64_t* const offsets = lists->buffers[1].address;
    uint64_t item = 0;
    size_t byte = 0;
    bool checked = write;
    int status = DORA_OK;
    dora_vector_operand_reset(&source->operand);
    for (uint64_t first = 0; status == DORA_OK && first < lists->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, lists->stored);
        if (!write && has_separator && separator.bytes.byte_count == 0) {
            struct dora_utf8_batch batch = {};
            size_t counts[KERNEL_CHUNK_ROW_COUNT];
            status = dora_vector_operand_prepare_utf8(&source->operand, rows, nullptr, &batch);
            batch.proof = source->utf8_proof;
            if (status == DORA_OK) {
                status = dora_utf8_count_batch(&batch, counts, nullptr);
            }
            for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
                const bool valid = compute_bytes_mask_is_valid(batch.eligible_rows, lane);
                const size_t count = valid ? (counts[lane] == 0 ? 1 : counts[lane]) : 0;
                if (ckd_add(&item, item, count) || item > INT64_MAX) {
                    status = EOVERFLOW;
                } else {
                    offsets[first + lane + 1] = (int64_t)item;
                    if (lists->buffers[0].address != nullptr) {
                        bitmap_assign(lists->buffers[0].address, first + lane, valid);
                    }
                    lists->nulls += (uint64_t)!valid;
                }
            }
            continue;
        }
        status = dora_compute_bytes_prepare_validate(source,
                                                     rows,
                                                     write ? DORA_UTF8_VALIDATED : source->utf8_proof,
                                                     nullptr);
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool valid = compute_bytes_piece_is_valid(source, lane);
            if (valid && has_separator && !checked) {
                status = utf8_check_input(separator);
                checked = status == DORA_OK;
            }
            if (status == DORA_OK && valid) {
                const struct dora_bytes piece = compute_bytes_piece(source, lane);
                const uint32_t index =
                    records != nullptr && piece.byte_count > VIEW_INLINE_BYTE_COUNT
                        ? (uint32_t)type_load_i32(
                              dora_compute_bytes_view_record(source, records, first, lane) +
                                  offsetof(struct dora_binary_view_entry, payload.reference.buffer_index),
                              0)
                        : 0;
                status = utf8_split_row((struct dora_utf8_input){piece, DORA_UTF8_VALIDATED},
                                        has_separator ? &(struct dora_utf8_input){separator.bytes, DORA_UTF8_VALIDATED}
                                                      : nullptr,
                                        backing,
                                        index,
                                        write,
                                        &item,
                                        &byte,
                                        values);
            }
            if (status == DORA_OK && write) {
                if (item != (uint64_t)offsets[row + 1]) {
                    status = EINVAL;
                }
            } else if (status == DORA_OK) {
                offsets[row + 1] = (int64_t)item;
                if (lists->buffers[0].address != nullptr) {
                    bitmap_assign(lists->buffers[0].address, row, valid);
                }
                lists->nulls += (uint64_t)!valid;
            }
        }
    }
    if (status == DORA_OK && write) {
        if (item != values->stored || byte != values->data_bytes) {
            status = EINVAL;
        }
    } else if (status == DORA_OK) {
        values->stored = item;
        values->data_bytes = byte;
    }
    return status;
}

static int utf8_split_publish(struct scratch_scope* scope,
                              const struct dora_vector* source,
                              const struct dora_view_buffer* backing,
                              uint32_t backing_count,
                              struct compute_bytes_result* lists,
                              struct compute_bytes_result* values,
                              uint64_t length,
                              bool constant,
                              enum vector_content_state content_state,
                              struct dora_vector* out_result) {
    const struct dora_pool pool = scope->pool;
    struct dora_vector child = {}, value = {};
    values->buffer_source = source;
    values->backing = backing;
    values->backing_count = backing_count;
    int status = dora_compute_bytes_publish(scope,
                                            values,
                                            &dora_compute_utf8_type,
                                            DORA_LAYOUT_BINARY_VIEW,
                                            content_state,
                                            false,
                                            values->stored,
                                            &child);
    const struct dora_schema_field field =
        {.type = {.id = DORA_TYPE_LIST}, .layout = DORA_LAYOUT_LARGE_LIST, .child_count = 1};
    const struct dora_bytes buffers[] = {
        {lists->buffers[0].address, lists->buffers[0].capacity_bytes},
        {lists->buffers[1].address, lists->buffers[1].capacity_bytes}
    };
    const struct vector_input input = {
        .field = &field,
        .row_count = lists->stored,
        .null_count = lists->nulls,
        .buffers = buffers,
        .allocations = lists->buffers,
        .buffer_count = 2,
        .children = &child,
        .child_count = 1,
        .content_state = VECTOR_CONTENT_VALIDATED
    };
    if (status == DORA_OK) {
        status = dora_compute_nested_publish(scope, &input, lists->buffers, &value);
    }
    if (status == DORA_OK) {
        status = constant ? dora_vector_init_retain_constant_in_pool(out_result, pool, &value, length)
                          : dora_vector_move(&value, out_result);
    }
    dora_vector_release(&value);
    dora_vector_release(&child);
    return status;
}

int dora_compute_utf8_split(const struct dora_vector* source,
                            const struct dora_vector* separator,
                            struct dora_vector* out_result) {
    const struct dora_vector* const vectors[] = {source, separator};
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source || out_result == separator) {
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

    for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
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
        if (status == DORA_OK && index == 1) {
            status = inputs[index].input.length != 1
                         ? EINVAL
                         : dora_compute_bytes_prepare(&inputs[index], (struct dora_range){0, 1});
        }
    }
    const bool has_separator = status == DORA_OK && compute_bytes_piece_is_valid(&inputs[1], 0);
    const struct dora_bytes pattern = has_separator ? compute_bytes_piece(&inputs[1], 0) : (struct dora_bytes){};
    const bool constant = status == DORA_OK && inputs[0].input.is_constant && inputs[0].input.length != 0;
    struct compute_bytes_result lists = {.stored = constant ? 1 : inputs[0].input.length}, values = {};
    const bool all_valid = compute_bytes_operand_is_all_valid(&inputs[0].operand);
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, !all_valid, &lists);
    }
    const bool retained = status == DORA_OK && dora_compute_bytes_has_owned_backing(&inputs[0], true);
    if (status == DORA_OK) {
        status = utf8_split_rows(&inputs[0],
                                 (struct dora_utf8_input){pattern, inputs[1].utf8_proof},
                                 has_separator,
                                 false,
                                 nullptr,
                                 &lists,
                                 &values);
    }
    if (status == DORA_OK) {
        size_t bytes = 0;
        status = ckd_mul(&bytes, values.stored, sizeof(struct dora_binary_view_entry))
                     ? EOVERFLOW
                     : dora_pool_allocate(pool, bytes, &values.buffers[1]);
    }
    if (status == DORA_OK && !retained) {
        status = dora_pool_allocate(pool, values.data_bytes, &values.buffers[2]);
    }
    uint32_t backing_count = 0;
    const struct dora_view_buffer copied = {.data = values.buffers[2].address, .byte_count = values.data_bytes};
    const struct dora_view_buffer* backing = nullptr;
    if (status == DORA_OK) {
        if (retained) {
            backing = dora_compute_bytes_backing(&inputs[0], &backing_count);
        } else {
            status = dora_compute_bytes_view_windows(&scope, &copied, &backing, &backing_count);
        }
    }
    if (status == DORA_OK) {
        status = utf8_split_rows(&inputs[0],
                                 (struct dora_utf8_input){pattern, inputs[1].utf8_proof},
                                 has_separator,
                                 true,
                                 backing,
                                 &lists,
                                 &values);
    }
    if (status == DORA_OK) {
        status = utf8_split_publish(&scope,
                                    retained && values.data_bytes != 0 ? source : nullptr,
                                    backing,
                                    values.data_bytes == 0 ? 0 : backing_count,
                                    &lists,
                                    &values,
                                    inputs[0].input.length,
                                    constant,
                                    dora_compute_bytes_content(inputs, 2),
                                    &published);
    }
    dora_compute_bytes_result_release(&values);
    dora_compute_bytes_result_release(&lists);
    dora_compute_bytes_release(inputs, 2);
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}

//------------------------------------------------------------------------------
//
// Literal Join
//
//------------------------------------------------------------------------------

struct utf8_join_source {
    struct vector_bound_input input;
    struct compute_bytes_source separator;
    struct vector_key_workspace nested;
    uint32_t child;
    enum vector_content_state content_state;
};

static int utf8_join_range(const struct dora_vector_view* view,
                           uint64_t slot,
                           const struct dora_vector_view* child,
                           struct dora_range* out_range) {
    struct dora_range range = {};
    if (view->layout == DORA_LAYOUT_FIXED_LIST) {
        if (ckd_mul(&range.offset, slot, view->list_element_count)) {
            return EOVERFLOW;
        }
        range.length = view->list_element_count;
    } else {
        const bool wide = view->layout == DORA_LAYOUT_LARGE_LIST || view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        const bool sized = view->layout == DORA_LAYOUT_LIST_VIEW || view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
        const size_t width = wide ? sizeof(int64_t) : sizeof(int32_t);
        size_t position, end;
        if (ckd_mul(&position, slot, width) || ckd_add(&end, position, sized ? width : 2 * width)) {
            return EOVERFLOW;
        }
        if (end > view->offsets.byte_count || view->offsets.data == nullptr ||
            (sized && (end > view->sizes.byte_count || view->sizes.data == nullptr))) {
            return EINVAL;
        }
        int64_t begin = 0, bound = 0;
        const uint8_t* const offsets = view->offsets.data;
        const uint8_t* const limits = sized ? view->sizes.data : offsets + width;
        if (wide) {
            memcpy(&begin, offsets + position, width);
            memcpy(&bound, limits + position, width);
        } else {
            int32_t narrow_begin = 0, narrow_bound = 0;
            memcpy(&narrow_begin, offsets + position, width);
            memcpy(&narrow_bound, limits + position, width);
            begin = narrow_begin;
            bound = narrow_bound;
        }
        if (begin < 0 || bound < 0 || (!sized && bound < begin)) {
            return EINVAL;
        }
        range.offset = begin;
        range.length = sized ? (uint64_t)bound : (uint64_t)bound - (uint64_t)begin;
    }
    if (range.offset > child->length || range.length > child->length - range.offset) {
        return EINVAL;
    }
    *out_range = range;
    return DORA_OK;
}

static int utf8_join_rows(struct utf8_join_source* source, bool write, struct compute_bytes_result* result) {
    struct vector_key_workspace* const nested = &source->nested;
    struct vector_key_side* const side = &nested->sides[0];
    int64_t* const offsets = result->buffers[1].address;
    size_t bytes = 0;
    bool separator_checked = false;
    dora_vector_operand_reset(&side->prepare);
    dora_vector_operand_reset(&source->separator.operand);
    const bool supplied = source->separator.input.nodes != nullptr;
    int status = supplied ? dora_compute_bytes_prepare(&source->separator, (struct dora_range){0, 1}) : DORA_OK;
    const struct dora_bytes separator = supplied && compute_bytes_piece_is_valid(&source->separator, 0)
                                            ? compute_bytes_piece(&source->separator, 0)
                                            : (struct dora_bytes){(const uint8_t*)",", 1};
    for (uint64_t first = 0; status == DORA_OK && first < result->stored; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = vector_operand_chunk(first, result->stored);
        status = dora_vector_key_validity(nested, rows, nested->root_validity);
        const struct dora_vector_view* const view = &source->input.nodes[side->prepare.terminal_node].view;
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            if (dora_bitmap_test((const uint8_t*)nested->root_validity, lane)) {
                status = utf8_join_range(view,
                                         side->slots[lane],
                                         &source->input.nodes[source->child].view,
                                         &side->ranges[lane]);
            }
        }
        for (uint32_t lane = 0; status == DORA_OK && lane < rows.length; ++lane) {
            const uint64_t row = first + lane;
            const bool parent_valid = dora_bitmap_test((const uint8_t*)nested->root_validity, lane);
            const struct dora_range range = parent_valid ? side->ranges[lane] : (struct dora_range){};
            const size_t limit = write ? (size_t)offsets[row + 1] : SIZE_MAX;
            struct dora_utf8_join_state joined = {.byte_count = bytes};
            for (uint64_t element = 0; status == DORA_OK && element < range.length; element += KERNEL_CHUNK_ROW_COUNT) {
                const struct dora_range chunk = vector_operand_chunk(element, range.length);
                for (uint32_t child_lane = 0; child_lane < chunk.length; ++child_lane) {
                    side->prepare.rows64[child_lane] = range.offset + element + child_lane;
                }
                struct kernel_packet packet;
                status =
                    dora_vector_operand_prepare_rows(&side->prepare, source->child, &packet, (uint32_t)chunk.length);
                if (status != DORA_OK) {
                    break;
                }
                const struct dora_bytes* const pieces = packet.values;
                for (uint32_t child_lane = 0; status == DORA_OK && child_lane < chunk.length; ++child_lane) {
                    if (packet.validity != nullptr && !dora_bitmap_test((const uint8_t*)packet.validity, child_lane)) {
                        continue;
                    }
                    if (pieces == nullptr) {
                        status = EINVAL;
                        break;
                    }
                    const struct dora_bytes piece = pieces[packet.value_offset + child_lane];
                    status = utf8_check_input(
                        (struct dora_utf8_input){piece, vector_utf8_proof(source->input.nodes[source->child].image)});
                    if (status == DORA_OK && joined.has_value && !separator_checked) {
                        status = utf8_check_input((struct dora_utf8_input){separator, source->separator.utf8_proof});
                        separator_checked = status == DORA_OK;
                    }
                    if (status == DORA_OK) {
                        status =
                            utf8_join_append((struct dora_utf8_input){piece, DORA_UTF8_VALIDATED},
                                             (struct dora_utf8_input){separator, DORA_UTF8_VALIDATED},
                                             write ? DORA_UTF8_EMIT : DORA_UTF8_MEASURE,
                                             (struct dora_mutable_bytes){result->buffers[2].address, write ? limit : 0},
                                             &joined);
                        bytes = joined.byte_count;
                    }
                }
            }
            const bool valid = parent_valid && (range.length == 0 || joined.has_value);
            if (status == DORA_OK && write) {
                if (valid != dora_bitmap_test(result->buffers[0].address, row) || bytes != limit) {
                    status = EINVAL;
                }
            } else if (status == DORA_OK) {
                offsets[row + 1] = bytes;
                bitmap_assign(result->buffers[0].address, row, valid);
                result->nulls += (uint64_t)!valid;
            }
        }
    }
    if (status == DORA_OK && write && bytes != result->data_bytes) {
        status = EINVAL;
    } else if (status == DORA_OK && !write) {
        result->data_bytes = bytes;
    }
    return status;
}

int dora_compute_utf8_join(const struct dora_vector* source,
                           const struct dora_vector* separator,
                           struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || out_result == source || out_result == separator) {
        return EINVAL;
    }
    struct dora_vector published = {};
    void* workspace = nullptr;
    struct scratch_scope scope = {};
    int status = dora_scratch_scope_begin(pool, &scope);
    if (status == DORA_OK) {
        status = dora_scratch_allocate(&scope, sizeof(struct utf8_join_source), &workspace);
    }
    if (status != DORA_OK) {
        dora_scratch_scope_end(&scope);
        return vector_finish_publication(&published, out_result, status);
    }
    struct utf8_join_source* const input = workspace;
    *input = (struct utf8_join_source){};
    status = dora_vector_bind_input(&scope, source, nullptr, &input->input);
    if (status == DORA_OK) {
        status = dora_vector_input_admit(&input->input, 0, true);
    }
    uint32_t node = 0;
    if (status == DORA_OK) {
        const enum dora_type type = input->input.nodes[0].field.type.id;
        if (type != DORA_TYPE_LIST && type != DORA_TYPE_FIXED_LIST) {
            status = ENOTSUP;
        }
    }
    if (status == DORA_OK) {
        node = vector_input_value_node(&input->input, node);
    }
    if (status == DORA_OK) {
        const enum dora_layout layout = input->input.nodes[node].view.layout;
        if (input->input.nodes[node].field.child_count != 1 ||
            (layout != DORA_LAYOUT_LIST && layout != DORA_LAYOUT_LARGE_LIST && layout != DORA_LAYOUT_LIST_VIEW &&
             layout != DORA_LAYOUT_LARGE_LIST_VIEW && layout != DORA_LAYOUT_FIXED_LIST)) {
            status = ENOTSUP;
        } else {
            input->child = dora_vector_input_child(node, &input->input, 0);
            if (input->input.nodes[input->child].field.type.id != DORA_TYPE_UTF8) {
                status = ENOTSUP;
            }
        }
    }
    if (status == DORA_OK && separator != nullptr) {
        status = dora_compute_bytes_bind(&scope,
                                         0,
                                         separator,
                                         1,
                                         false,
                                         VECTOR_OPERAND_REQUIRE_PACKET,
                                         true,
                                         &input->separator);
    }
    if (status == DORA_OK && separator != nullptr && input->separator.input.nodes[0].field.type.id != DORA_TYPE_UTF8) {
        status = ENOTSUP;
    }
    if (status == DORA_OK && separator != nullptr && input->separator.input.length != 1) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_key_validity_init(&scope, &input->input, KERNEL_CHUNK_ROW_COUNT, true, &input->nested);
    }
    if (status == DORA_OK) {
        input->content_state = compute_proof_content(vector_utf8_proof(input->input.nodes[input->child].image));
        if (separator != nullptr) {
            const enum vector_content_state state = dora_compute_bytes_content(&input->separator, 1);
            if (state < input->content_state) {
                input->content_state = state;
            }
        }
    }
    const bool constant = status == DORA_OK && input->input.is_constant && input->input.length != 0;
    struct compute_bytes_result result = {.stored = constant ? 1 : input->input.length};
    if (status == DORA_OK) {
        status = dora_compute_bytes_result_init(pool, result.stored != 0, &result);
    }
    if (status == DORA_OK) {
        status = utf8_join_rows(input, false, &result);
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, result.data_bytes, &result.buffers[2]);
    }
    if (status == DORA_OK) {
        status = utf8_join_rows(input, true, &result);
    }
    if (status == DORA_OK) {
        status = dora_compute_bytes_publish(&scope,
                                            &result,
                                            &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                            DORA_LAYOUT_LARGE_BINARY,
                                            input->content_state,
                                            constant,
                                            input->input.length,
                                            &published);
    }
    dora_compute_bytes_result_release(&result);

    dora_compute_bytes_release(&input->separator, 1);
    memset(workspace, 0, sizeof(struct utf8_join_source));
    dora_scratch_scope_end(&scope);
    return vector_finish_publication(&published, out_result, status);
}
