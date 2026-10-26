// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "vector/operand.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/binding/read.h"
#include "kernel/decode.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/array_count.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Classification
//
//------------------------------------------------------------------------------

static uint32_t operand_terminal(const struct vector_bound_input* input, uint32_t* out_chain_length) {
    uint32_t node = 0;
    uint32_t chain_length = 0;
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        chain_length += view->selection.indices != nullptr ? 1U : 0U;
        if (!vector_view_has_encoding_child(view)) {
            break;
        }
        ++chain_length;
        node = vector_input_encoding_child(input, node);
    }
    *out_chain_length = chain_length;
    return node;
}

static bool operand_chain_is_all_valid(const struct vector_bound_input* input) {
    uint32_t node = 0;
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        if (view->validity.data != nullptr) {
            return false;
        }
        if (!vector_view_has_encoding_child(view)) {
            return true;
        }
        node = vector_input_encoding_child(input, node);
    }
}

static int operand_bind_chain(struct vector_operand* in_out_operand, uint32_t node, bool terminal_selection) {
    const struct vector_bound_input* const input = in_out_operand->input;
    uint32_t index = 0;
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        const bool wrapper = vector_view_has_encoding_child(view);
        if (view->selection.indices != nullptr && (wrapper || terminal_selection)) {
            const int status = dora_kernel_decode_bind(view, &in_out_operand->chain[index++]);
            if (status != DORA_OK) {
                return status;
            }
        }
        if (!wrapper) {
            break;
        }
        struct dora_vector_view plain = *view;
        plain.selection = (struct dora_vector_selection){};
        const int status = dora_kernel_decode_bind(&plain, &in_out_operand->chain[index]);
        if (status != DORA_OK) {
            return status;
        }
        in_out_operand->chain[index++].view = view;
        node = vector_input_encoding_child(input, node);
    }
    in_out_operand->chain_length = index;
    in_out_operand->terminal_node = node;
    return DORA_OK;
}

static bool operand_has_referenced_indices(const struct vector_operand* operand) {
    if (operand->kind != VECTOR_OPERAND_ENCODED || operand->chain_length != 1 ||
        operand->chain[0].kind != KERNEL_DECODE_DICTIONARY) {
        return false;
    }
    const struct dora_vector_view* const view = operand->chain[0].view;
    return (view->physical_type == DORA_PHYSICAL_TYPE_U32 || view->physical_type == DORA_PHYSICAL_TYPE_I32) &&
           view->validity.data == nullptr && (uintptr_t)view->values.data % alignof(uint32_t) == 0;
}

//------------------------------------------------------------------------------
//
// Operands
//
//------------------------------------------------------------------------------

static int operand_bind_bytes(const struct dora_vector_view* terminal, struct vector_operand* in_out_operand) {
    struct dora_vector_view validity = *terminal;
    validity.layout = DORA_LAYOUT_FIXED;
    int status = dora_kernel_gather_bind(&validity, &in_out_operand->gather);
    in_out_operand->gather.view = terminal;
    if (status == DORA_OK) {
        status = dora_kernel_bytes_bind(terminal, &in_out_operand->bytes);
    }
    return status;
}

int dora_vector_operand_bind_constant(uint32_t value_node, struct vector_operand* in_out_operand) {
    const struct vector_bound_input* const input = in_out_operand->input;
    const struct dora_vector_view* const value = &input->nodes[value_node].view;
    in_out_operand->terminal_node = value_node;
    if (value->layout == DORA_LAYOUT_NULL) {
        in_out_operand->kind = VECTOR_OPERAND_CONSTANT_NULL;
        return DORA_OK;
    }
    const bool bytes = type_is_bytes(value->type);
    int status =
        bytes ? operand_bind_bytes(value, in_out_operand) : dora_kernel_gather_bind(value, &in_out_operand->gather);
    uint64_t valid = 0;
    if (status == DORA_OK) {
        dora_kernel_load_validity(in_out_operand->gather.view, 0, 1, &valid);
    }
    if (status == DORA_OK && bytes && valid != 0) {
        status =
            in_out_operand->bytes.fill(&in_out_operand->bytes, 0, nullptr, nullptr, 1, &in_out_operand->constant_bytes);
    }
    in_out_operand->values = bytes ? (const void*)&in_out_operand->constant_bytes : value->values.data;
    in_out_operand->value_offset = bytes ? 0 : value->offset;
    in_out_operand->kind = valid != 0 ? VECTOR_OPERAND_CONSTANT : VECTOR_OPERAND_CONSTANT_NULL;
    return status;
}

int dora_vector_input_from_nodes(struct scratch_scope* scope,
                                 const struct dora_node_topology* nodes,
                                 uint32_t node_count,
                                 uint32_t root,
                                 struct vector_bound_input* out_input) {
    if (nodes == nullptr || out_input == nullptr || out_input->nodes != nullptr || root >= node_count) {
        return EINVAL;
    }
    uint32_t end = root + 1;
    while (end < node_count && nodes[end].parent_node_index >= root && nodes[end].parent_node_index < end) {
        ++end;
    }
    const uint32_t count = end - root;
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, sizeof(struct vector_bound_node))) {
        return EOVERFLOW;
    }
    void* storage = nullptr;
    const int status = dora_scratch_slot_require(scope,
                                                 vector_operand_slot(0, VECTOR_OPERAND_SLOT_NODES),
                                                 &storage,
                                                 bytes,
                                                 alignof(struct vector_bound_node));
    if (status != DORA_OK) {
        return status;
    }
    struct vector_bound_node* const bound = storage;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t sibling = nodes[root + index].next_sibling_index;
        bound[index] = (struct vector_bound_node){
            .view = nodes[root + index].view,
            .subtree_end_node_index = sibling > root && sibling < end ? sibling - root : count,
        };
    }
    *out_input = (struct vector_bound_input){
        .nodes = bound,
        .node_count = count,
        .length = nodes[root].view.length,
    };
    return DORA_OK;
}

static uint8_t operand_text_index_width(const struct vector_operand* operand) {
    if (operand->chain_length != 1) {
        return 0;
    }
    const struct dora_vector_view* const view = &operand->input->nodes[0].view;
    if (view->selection.indices != nullptr && !vector_view_has_encoding_child(view)) {
        return view->selection.physical_type == DORA_PHYSICAL_TYPE_U32 ? 4 : 8;
    }
    if (operand->bytes.view->selection.indices != nullptr || view->layout != DORA_LAYOUT_DICTIONARY ||
        view->selection.indices != nullptr) {
        return 0;
    }
    switch (view->physical_type) {
    case DORA_PHYSICAL_TYPE_I32:
    case DORA_PHYSICAL_TYPE_U32:
        return 4;
    case DORA_PHYSICAL_TYPE_I64:
    case DORA_PHYSICAL_TYPE_U64:
        return 8;
    default:
        return 0;
    }
}

static int
operand_storage(struct scratch_scope* scope, size_t position, struct vector_operand* operand, uint8_t requirements) {
    const struct vector_bound_input* const input = operand->input;
    const struct dora_vector_view* const terminal = &input->nodes[operand->terminal_node].view;
    const bool descriptors = (requirements & VECTOR_OPERAND_REQUIRE_PACKET) != 0;
    const bool reference = (requirements & VECTOR_OPERAND_REQUIRE_TEXT) != 0 ||
                           (requirements == VECTOR_OPERAND_REQUIRE_LENGTH && terminal->layout != DORA_LAYOUT_FIXED);
    const bool validity = !descriptors && !reference;
    const bool resolve_validity = (requirements & VECTOR_OPERAND_REQUIRE_VALIDITY) != 0 ||
                                  (requirements == VECTOR_OPERAND_REQUIRE_LENGTH && !reference);
    const size_t capacity = (requirements & VECTOR_OPERAND_REQUIRE_ROW) != 0 ? 1
                            : input->length < KERNEL_CHUNK_ROW_COUNT         ? (size_t)input->length
                                                                             : KERNEL_CHUNK_ROW_COUNT;
    const bool text_indices = reference && operand_text_index_width(operand) != 0;
    const bool borrowed = text_indices && !descriptors && !resolve_validity;
    const bool encoded = operand->chain_length != 0 && !borrowed;
    const struct dora_vector_view* const root = &input->nodes[0].view;
    const bool narrow = operand->chain_length == 1 && root->layout != DORA_LAYOUT_RUN_END;
    const bool indices_referenced =
        operand->chain_length == 1 && root->layout == DORA_LAYOUT_DICTIONARY &&
        (root->physical_type == DORA_PHYSICAL_TYPE_U32 || root->physical_type == DORA_PHYSICAL_TYPE_I32) &&
        root->validity.data == nullptr && (uintptr_t)root->values.data % alignof(uint32_t) == 0;
    const bool resolve_rows = encoded && (resolve_validity || (descriptors && operand->kind == VECTOR_OPERAND_BYTES) ||
                                          !operand->gather.can_use_u32_map || !narrow);
    const bool mask = (requirements & VECTOR_OPERAND_REQUIRE_ACTIVE) != 0 ||
                      ((requirements & VECTOR_OPERAND_REQUIRE_ROW) != 0 && !operand->is_all_valid) ||
                      (terminal->validity.data != nullptr && (descriptors || !reference) &&
                       (bitmap_bit_of(terminal->validity_bit_offset) != 0 ||
                        (uintptr_t)terminal->validity.data % alignof(uint64_t) != 0 ||
                        terminal->validity_bit_offset / 8 > terminal->validity.readable_byte_count ||
                        bitmap_words_for_bits(terminal->length) * sizeof(uint64_t) >
                            terminal->validity.readable_byte_count - terminal->validity_bit_offset / 8));
    size_t extents[6] = {
        resolve_rows ? capacity : 0,
        (mask || encoded || (descriptors && operand->kind == VECTOR_OPERAND_BYTES && !operand->is_all_valid) ||
         (borrowed && terminal->validity.data != nullptr))
            ? (size_t)bitmap_words_for_bits(capacity)
            : 0,
        borrowed ? 0 : operand->chain_length,
        operand->kind == VECTOR_OPERAND_BYTES ? (descriptors ? capacity : 0)
        : !validity && operand->kind == VECTOR_OPERAND_ENCODED && !operand->gather.can_use_u32_map
            ? (operand->gather.width == 0 ? (size_t)bitmap_words_for_bits(capacity) : capacity)
            : 0,
        !validity && encoded && operand->gather.can_use_u32_map && !indices_referenced &&
                (operand->kind != VECTOR_OPERAND_BYTES || (reference && !text_indices))
            ? capacity
            : 0,
        reference && operand->kind == VECTOR_OPERAND_BYTES && terminal->layout == DORA_LAYOUT_BINARY_VIEW
            ? terminal->data_buffer_count
            : 0,
    };
    const size_t widths[] = {
        sizeof(uint64_t),
        sizeof(uint64_t),
        sizeof(struct kernel_decode_binding),
        operand->kind == VECTOR_OPERAND_BYTES ? sizeof(struct dora_bytes)
        : operand->gather.width == 0          ? sizeof(uint64_t)
                                              : operand->gather.width,
        sizeof(uint32_t),
        sizeof(struct dora_bytes)
    };
    static_assert(ARRAY_COUNT(widths) == ARRAY_COUNT(extents));
    size_t bytes = 0;
    for (uint32_t slot = 0; slot < 6; ++slot) {
        if (ckd_mul(&extents[slot], extents[slot], widths[slot]) || ckd_add(&extents[slot], extents[slot], 63)) {
            return EOVERFLOW;
        }
        extents[slot] = extents[slot] / 64 * 64;
        if (ckd_add(&bytes, bytes, extents[slot])) {
            return EOVERFLOW;
        }
    }
    if (bytes == 0) {
        return DORA_OK;
    }
    void* storage = nullptr;
    int status = dora_scratch_slot_require(scope,
                                           vector_operand_slot(position, VECTOR_OPERAND_SLOT_OPERAND),
                                           &storage,
                                           bytes,
                                           64);
    if (status != DORA_OK) {
        return status;
    }
    uint8_t* block = storage;
    operand->rows64 = extents[0] == 0 ? nullptr : (uint64_t*)block;
    block += extents[0];
    operand->chunk_validity = extents[1] == 0 ? nullptr : (uint64_t*)block;
    block += extents[1];
    operand->chain = extents[2] == 0 ? nullptr : (struct kernel_decode_binding*)block;
    block += extents[2];
    operand->payload = extents[3] == 0 ? nullptr : block;
    block += extents[3];
    operand->map32 = extents[4] == 0 ? nullptr : (uint32_t*)block;
    block += extents[4];
    operand->byte_buffers = extents[5] == 0 ? nullptr : (struct dora_bytes*)block;
    for (uint32_t i = 0; extents[5] != 0 && i < terminal->data_buffer_count; ++i) {
        operand->byte_buffers[i] =
            (struct dora_bytes){terminal->data_buffers[i].data, terminal->data_buffers[i].byte_count};
    }
    if (extents[2] != 0) {
        status = operand_bind_chain(operand, 0, true);
        if (status != DORA_OK) {
            return status;
        }
    }
    operand->has_referenced_indices = !borrowed && operand_has_referenced_indices(operand);
    return DORA_OK;
}

int dora_vector_operand_bind(struct scratch_scope* scope,
                             size_t position,
                             const struct vector_bound_input* input,
                             uint8_t requirements,
                             struct vector_operand* out_operand) {
    if (input == nullptr || out_operand == nullptr || input->node_count == 0) {
        return EINVAL;
    }
    *out_operand = (struct vector_operand){.input = input};
    struct vector_operand* const operand = out_operand;
    if (input->nodes[0].view.layout == DORA_LAYOUT_NULL) {
        operand->kind = VECTOR_OPERAND_CONSTANT_NULL;
        return DORA_OK;
    }
    if (input->is_constant) {
        const uint32_t value_node = dora_vector_input_child(0, input, 1);
        const struct dora_vector_view* const value = &input->nodes[value_node].view;
        if (value->selection.indices == nullptr &&
            (type_is_bytes(value->type) ? !vector_view_has_encoding_child(value)
                                        : value->layout == DORA_LAYOUT_NULL || value->layout == DORA_LAYOUT_FIXED ||
                                              value->layout == DORA_LAYOUT_BITS)) {
            return dora_vector_operand_bind_constant(value_node, operand);
        }
    }
    int status = DORA_OK;
    operand->terminal_node = operand_terminal(input, &operand->chain_length);
    const struct dora_vector_view* const terminal = &input->nodes[operand->terminal_node].view;
    if (terminal->layout == DORA_LAYOUT_NULL) {
        operand->kind = VECTOR_OPERAND_CONSTANT_NULL;
        operand->chain_length = 0;
    } else if (type_is_bytes(terminal->type)) {
        operand->kind = VECTOR_OPERAND_BYTES;
        operand->is_all_valid = operand_chain_is_all_valid(input);
        status = operand_bind_bytes(terminal, operand);
    } else if (terminal->layout != DORA_LAYOUT_FIXED && terminal->layout != DORA_LAYOUT_BITS) {
        operand->kind = VECTOR_OPERAND_NESTED;
        operand->chain_length = 0;
    } else {
        operand->kind = operand->chain_length == 0 ? VECTOR_OPERAND_DIRECT : VECTOR_OPERAND_ENCODED;
        operand->is_all_valid = operand_chain_is_all_valid(input);
        status = dora_kernel_gather_bind(terminal, &operand->gather);
    }
    if (status == DORA_OK && operand->kind == VECTOR_OPERAND_DIRECT && operand->is_all_valid &&
        (requirements & VECTOR_OPERAND_REQUIRE_ACTIVE) == 0) {
        return DORA_OK;
    }
    if (status == DORA_OK && input->length != 0 && operand->kind != VECTOR_OPERAND_NESTED &&
        operand->kind != VECTOR_OPERAND_CONSTANT_NULL) {
        status = operand_storage(scope, position, operand, requirements);
    }
    return status;
}

int dora_vector_operand_bind_singleton(const struct dora_vector* source,
                                       const struct vector_bound_input* input,
                                       struct vector_operand* out_operand) {
    if (source == nullptr || input == nullptr || out_operand == nullptr || input->length != 1) {
        return EINVAL;
    }
    struct decoded_row value;
    const int status = dora_vector_decode_row(source, 0, &value);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t terminal = vector_input_value_node(input, 0);
    *out_operand = (struct vector_operand){
        .input = input,
        .constant_bytes = type_is_bytes(value.type) ? value.value.bytes : (struct dora_bytes){},
        .terminal_node = terminal,
        .kind = value.is_null ? VECTOR_OPERAND_CONSTANT_NULL : VECTOR_OPERAND_CONSTANT,
        .is_all_valid = !value.is_null,
    };
    if (!value.is_null && !type_is_bytes(value.type)) {
        const size_t width = value.type == DORA_TYPE_BOOLEAN ? sizeof(bool) : physical_width(value.physical_type);
        if (width == 0 || width > sizeof out_operand->constant_value) {
            return ENOTSUP;
        }
        memcpy(out_operand->constant_value, &value.value, width);
    }
    out_operand->values =
        type_is_bytes(value.type) ? (const void*)&out_operand->constant_bytes : out_operand->constant_value;
    return DORA_OK;
}

bool dora_vector_operands_are_all_valid(const struct vector_operand* operands, uint32_t operand_count) {
    for (uint32_t index = 0; index < operand_count; ++index) {
        const enum vector_operand_kind kind = operands[index].kind;
        const bool valid =
            kind == VECTOR_OPERAND_CONSTANT ||
            ((kind == VECTOR_OPERAND_DIRECT || kind == VECTOR_OPERAND_ENCODED) && operands[index].is_all_valid);
        if (!valid) {
            return false;
        }
    }
    return true;
}

static void operand_lanes(struct vector_operand* operand, uint32_t count, const uint64_t* active) {
    const size_t word_count = (size_t)bitmap_word_of((uint64_t)count + 63);
    bitmap_prefix_words(count, operand->chunk_validity, word_count);
    if (active != nullptr) {
        dora_bitmap_and_aligned_words(operand->chunk_validity, active, word_count);
    }
}

static struct vector_operand_rows operand_rows(struct vector_operand* operand, uint32_t count) {
    return (struct vector_operand_rows){.rows = operand->rows64, .validity = operand->chunk_validity, .count = count};
}

static void
operand_resolve(struct vector_operand* operand, uint64_t first_row, const struct vector_operand_rows* rows) {
    for (uint32_t index = 0; index < operand->chain_length; ++index) {
        operand->chain[index].resolve(&operand->chain[index],
                                      index == 0 ? first_row : UINT64_MAX,
                                      rows->count,
                                      rows->rows,
                                      rows->validity);
    }
}

static void operand_resolve_valid(struct vector_operand* operand, struct dora_range rows) {
    const struct vector_operand_rows list = operand_rows(operand, (uint32_t)rows.length);
    operand_resolve(operand, rows.offset, &list);
    if (operand->gather.view->validity.data != nullptr) {
        dora_kernel_gather_validity(operand->gather.view, list.rows, list.validity, list.count, list.validity);
    }
}

static const uint32_t* operand_map(struct vector_operand* operand, struct dora_range rows) {
    const struct vector_operand_rows list = operand_rows(operand, (uint32_t)rows.length);
    const struct dora_vector_view* const terminal = operand->gather.view;
    const bool contiguous = rows.offset != UINT64_MAX;
    const uint32_t* map = operand->map32;
    if (contiguous && operand->has_referenced_indices) {
        const struct dora_vector_view* const indices = operand->chain[0].view;
        map = (const uint32_t*)indices->values.data + indices->offset + rows.offset;
    } else if (contiguous && operand->chain_length == 1 && operand->chain[0].narrow != nullptr) {
        operand->chain[0].narrow(&operand->chain[0], rows.offset, list.count, list.validity, operand->map32);
    } else {
        operand_resolve(operand, rows.offset, &list);
        dora_kernel_narrow_rows(list.rows,
                                terminal->selection.indices != nullptr ? terminal->offset : 0,
                                operand->map32,
                                list.validity,
                                list.count);
    }
    if (terminal->validity.data != nullptr) {
        dora_kernel_gather_map_validity(terminal, map, list.validity, list.count, list.validity);
    }
    return map;
}

static void
operand_fixed_packet(struct vector_operand* operand, struct dora_range rows, struct kernel_packet* in_out_packet) {
    const uint32_t count = (uint32_t)rows.length;
    const struct dora_vector_view* const terminal = operand->gather.view;
    if (!operand->gather.can_use_u32_map) {
        operand_resolve_valid(operand, rows);
        operand->gather.gather(&operand->gather, operand->rows64, operand->payload, operand->chunk_validity, count);
        in_out_packet->values = operand->payload;
        in_out_packet->shape = KERNEL_SHAPE_FLAT;
        return;
    }
    const uint32_t* const map = operand_map(operand, rows);
    in_out_packet->values = terminal->values.data;
    in_out_packet->rows = map;
    in_out_packet->value_offset = terminal->offset;
    in_out_packet->shape = KERNEL_SHAPE_GATHERED;
}

static int
operand_bytes_packet(struct vector_operand* operand, struct dora_range rows, struct kernel_packet* in_out_packet) {
    operand_resolve_valid(operand, rows);
    in_out_packet->values = operand->payload;
    in_out_packet->shape = KERNEL_SHAPE_FLAT;
    return operand->bytes.fill(&operand->bytes,
                               rows.offset == UINT64_MAX ? 0 : rows.offset,
                               operand->rows64,
                               operand->chunk_validity,
                               (uint32_t)rows.length,
                               operand->payload);
}

static void operand_prepare_encoded(struct vector_operand* operand,
                                    struct dora_range rows,
                                    const uint64_t* active,
                                    struct kernel_packet* out_packet) {
    operand_lanes(operand, (uint32_t)rows.length, active);
    *out_packet = (struct kernel_packet){
        .validity = operand->is_all_valid && active == nullptr ? nullptr : operand->chunk_validity,
        .count = (uint32_t)rows.length,
        .placement = operand->gather.view->placement,
    };
    operand_fixed_packet(operand, rows, out_packet);
}

static const uint64_t null_words[KERNEL_CHUNK_WORD_COUNT] = {};

static int operand_validity(struct vector_operand* operand,
                            struct dora_range rows,
                            const uint64_t* active,
                            const uint64_t** out_words) {
    const struct dora_vector_view* const view = operand->gather.view;
    if (view->validity.data == nullptr && active == nullptr) {
        *out_words = nullptr;
        return DORA_OK;
    }
    uint64_t origin;
    if (ckd_add(&origin, view->validity_bit_offset, rows.offset)) {
        return EOVERFLOW;
    }
    if ((operand->kind == VECTOR_OPERAND_DIRECT || operand->chunk_validity == nullptr) && active == nullptr &&
        bitmap_bit_of(origin) == 0 && (uintptr_t)view->validity.data % alignof(uint64_t) == 0) {
        const uint64_t first = bitmap_byte_of(origin);
        const size_t bytes = (size_t)bitmap_words_for_bits(rows.length) * sizeof(uint64_t);
        if (first <= view->validity.readable_byte_count && bytes <= view->validity.readable_byte_count - first) {
            *out_words = (const uint64_t*)((const uint8_t*)view->validity.data + first);
            return DORA_OK;
        }
    }
    dora_kernel_load_validity(view, rows.offset, (uint32_t)rows.length, operand->chunk_validity);
    if (active != nullptr) {
        dora_bitmap_and_aligned_words(operand->chunk_validity, active, (size_t)bitmap_words_for_bits(rows.length));
    }
    *out_words = operand->chunk_validity;
    return DORA_OK;
}

static int operand_prepare_bytes(struct vector_operand* operand,
                                 struct dora_range rows,
                                 const uint64_t* active,
                                 struct kernel_packet* out_packet) {
    const uint32_t count = (uint32_t)rows.length;
    const struct dora_vector_view* const terminal = operand->bytes.view;
    *out_packet = (struct kernel_packet){
        .values = operand->payload,
        .count = count,
        .shape = KERNEL_SHAPE_FLAT,
        .placement = terminal->placement,
    };
    if (operand->chain_length != 0) {
        operand_lanes(operand, count, active);
        out_packet->validity = operand->is_all_valid && active == nullptr ? nullptr : operand->chunk_validity;
        return operand_bytes_packet(operand, rows, out_packet);
    }
    const int status = operand_validity(operand, rows, active, &out_packet->validity);
    if (status != DORA_OK) {
        return status;
    }
    const bool describe_all = active == nullptr && terminal->selection.indices == nullptr &&
                              (terminal->layout == DORA_LAYOUT_FIXED || terminal->layout == DORA_LAYOUT_BINARY ||
                               terminal->layout == DORA_LAYOUT_LARGE_BINARY) &&
                              rows.offset <= terminal->length && rows.length <= terminal->length - rows.offset;
    return operand->bytes.fill(&operand->bytes,
                               rows.offset,
                               nullptr,
                               describe_all ? nullptr : out_packet->validity,
                               count,
                               operand->payload);
}

static int operand_text_mask(struct vector_operand* operand,
                             const struct dora_vector_view* view,
                             struct dora_range rows,
                             const uint64_t* active,
                             struct dora_utf8_mask* out_mask) {
    if (view->validity.data == nullptr) {
        *out_mask = (struct dora_utf8_mask){
            {(const uint8_t*)active, active == nullptr ? 0 : (size_t)bitmap_bytes_for_bits(rows.length)},
            0
        };
        return DORA_OK;
    }
    uint64_t origin;
    if (ckd_add(&origin, view->validity_bit_offset, rows.offset)) {
        return EOVERFLOW;
    }
    if (active == nullptr) {
        *out_mask = (struct dora_utf8_mask){{view->validity.data, view->validity.byte_count}, origin};
        return 0;
    }
    uint64_t* const words = operand->chunk_validity;
    const size_t word_count = (size_t)bitmap_words_for_bits(rows.length);
    dora_bitmap_load_words(view->validity.data, origin, rows.length, words);
    dora_bitmap_and_aligned_words(words, active, word_count);
    *out_mask = (struct dora_utf8_mask){{(const uint8_t*)words, word_count * sizeof *words}, 0};
    return 0;
}

static int
operand_text_storage(struct vector_operand* operand, struct dora_range physical, struct dora_utf8_batch* out_batch) {
    const struct dora_vector_view* view = operand->bytes.view;
    size_t origin;
    if (physical.length > SIZE_MAX || physical.offset > SIZE_MAX) {
        return EOVERFLOW;
    }
    out_batch->physical_value_count = (size_t)physical.length;
    switch (view->layout) {
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY: {
        const size_t width = view->layout == DORA_LAYOUT_BINARY ? 4 : 8;
        if (ckd_mul(&origin, (size_t)physical.offset, width)) {
            return EOVERFLOW;
        }
        if (origin > view->offsets.byte_count || view->data_buffer_count == 0) {
            return EINVAL;
        }
        out_batch->layout = width == 4 ? DORA_UTF8_OFFSETS_I32 : DORA_UTF8_OFFSETS_I64;
        out_batch->rows.payload = (struct dora_bytes){view->data_buffers[0].data, view->data_buffers[0].byte_count};
        out_batch->rows.offsets =
            (struct dora_bytes){(const uint8_t*)view->offsets.data + origin, view->offsets.byte_count - origin};
        return 0;
    }
    case DORA_LAYOUT_BINARY_VIEW:
        if (ckd_mul(&origin, (size_t)physical.offset, sizeof(struct dora_binary_view_entry))) {
            return EOVERFLOW;
        }
        if (origin > view->values.byte_count) {
            return EINVAL;
        }
        out_batch->layout = DORA_UTF8_BINARY_VIEW;
        out_batch->views.records = (struct dora_bytes){
            view->values.data == nullptr ? nullptr : (const uint8_t*)view->values.data + origin,
            view->values.byte_count - origin
        };
        out_batch->views.buffers = operand->byte_buffers;
        out_batch->views.buffer_count = view->data_buffer_count;
        return 0;
    default:
        return ENOTSUP;
    }
}

// Utf8; row-index carriers: UInt32, UInt64
#define DEFINE_OPERAND_TEXT_CHILD(WIDTH, TYPE, PHYSICAL)                                                               \
    static int operand_text_child_##WIDTH(struct vector_operand* operand, struct dora_utf8_batch* batch) {             \
        if (batch->row_count != 0 && batch->row_indices.data == nullptr) {                                             \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        const struct dora_vector_view* const child = operand->bytes.view;                                              \
        if (child->validity.data == nullptr) {                                                                         \
            return 0;                                                                                                  \
        }                                                                                                              \
        uint64_t* const words = operand->chunk_validity;                                                               \
        const uint64_t word_count = bitmap_words_for_bits(batch->row_count);                                           \
        const uint64_t value_count = batch->physical_value_count;                                                      \
        const uint8_t* const indices = batch->row_indices.data;                                                        \
        const uint8_t* const validity = child->validity.data;                                                          \
        const uint64_t bit_offset = child->validity_bit_offset;                                                        \
        const uint8_t* const eligible = batch->eligible_rows.bytes.data;                                               \
        const uint64_t eligible_offset = batch->eligible_rows.bit_offset;                                              \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            const uint32_t first = word * DORA_BITMAP_WORD_BIT_COUNT;                                                  \
            const uint32_t remaining = batch->row_count - first;                                                       \
            const uint32_t count = remaining < DORA_BITMAP_WORD_BIT_COUNT ? remaining : DORA_BITMAP_WORD_BIT_COUNT;    \
            uint64_t candidates = eligible == nullptr                                                                  \
                                      ? bitmap_mask_low(count)                                                         \
                                      : dora_bitmap_load_window(eligible, eligible_offset + first, count);             \
            words[word] = candidates;                                                                                  \
            while (candidates != 0) {                                                                                  \
                const uint32_t row = word * 64 + dora_bitmap_word_pop_lowest_set(&candidates);                         \
                const TYPE index = type_load_##WIDTH(indices, row);                                                    \
                if ((uint64_t)index >= value_count) {                                                                  \
                    return EINVAL;                                                                                     \
                }                                                                                                      \
                if (!dora_bitmap_test(validity, bit_offset + index)) {                                                 \
                    bitmap_clear(words, row);                                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        batch->eligible_rows =                                                                                         \
            (struct dora_utf8_mask){{(const uint8_t*)words, (size_t)bitmap_bytes_for_bits(batch->row_count)}, 0};      \
        return 0;                                                                                                      \
    }
ROW_INDEX_CARRIERS(DEFINE_OPERAND_TEXT_CHILD)
#undef DEFINE_OPERAND_TEXT_CHILD

void dora_vector_operand_utf8_batch(const struct kernel_packet* packet,
                                    enum dora_utf8_proof proof,
                                    struct dora_utf8_batch* out_batch) {
    const bool constant = packet->shape == KERNEL_SHAPE_CONSTANT;
    const struct dora_bytes* const pieces =
        packet->values == nullptr ? nullptr : (const struct dora_bytes*)packet->values + packet->value_offset;
    *out_batch = (struct dora_utf8_batch){
        .layout = constant ? DORA_UTF8_CONSTANT : DORA_UTF8_DESCRIPTORS,
        .row_count = packet->count,
        .physical_value_count = constant ? 1 : packet->count,
        .descriptors = pieces,
        .proof = proof,
        .eligible_rows = {
            {(const uint8_t*)packet->validity,
             packet->validity == nullptr ? 0 : (size_t)bitmap_bytes_for_bits(packet->count)},
            0
        }
    };
    if (constant) {
        out_batch->constant = pieces == nullptr ? (struct dora_bytes){} : type_load_bytes((const uint8_t*)pieces, 0);
    }
}

int dora_vector_operand_prepare_utf8(struct vector_operand* operand,
                                     struct dora_range rows,
                                     const uint64_t* active,
                                     struct dora_utf8_batch* out_batch) {
    if (operand == nullptr || out_batch == nullptr || rows.length > KERNEL_CHUNK_ROW_COUNT ||
        rows.offset > operand->input->length || rows.length > operand->input->length - rows.offset) {
        return EINVAL;
    }
    struct dora_utf8_batch batch = {
        .row_count = (uint32_t)rows.length,
        .proof = vector_utf8_proof(operand->input->nodes[operand->terminal_node].image)
    };
    if (operand->kind == VECTOR_OPERAND_CONSTANT || operand->kind == VECTOR_OPERAND_CONSTANT_NULL) {
        batch.layout = DORA_UTF8_CONSTANT;
        batch.physical_value_count = 1;
        batch.constant = operand->constant_bytes;
        if (operand->kind == VECTOR_OPERAND_CONSTANT_NULL) {
            batch.eligible_rows = (struct dora_utf8_mask){{(const uint8_t*)null_words, sizeof null_words}, 0};
        } else if (active != nullptr) {
            batch.eligible_rows =
                (struct dora_utf8_mask){{(const uint8_t*)active, (size_t)bitmap_bytes_for_bits(rows.length)}, 0};
        }
        *out_batch = batch;
        return 0;
    }
    if (operand->kind != VECTOR_OPERAND_BYTES) {
        return ENOTSUP;
    }
    const struct dora_vector_view* view = operand->bytes.view;
    struct dora_range physical = {.offset = view->offset, .length = view->length};
    int status;
    if (operand->chain_length == 0) {
        if (ckd_add(&physical.offset, physical.offset, rows.offset)) {
            return EOVERFLOW;
        }
        physical.length = rows.length;
        status = operand_text_mask(operand, view, rows, active, &batch.eligible_rows);
    } else {
        const uint8_t width = operand_text_index_width(operand);
        if (width != 0) {
            const struct dora_vector_view* indices = &operand->input->nodes[0].view;
            const bool selected = indices->selection.indices != nullptr;
            uint64_t first;
            size_t origin, size;
            if (ckd_add(&first, selected ? 0 : indices->offset, rows.offset) || ckd_mul(&origin, first, width) ||
                ckd_mul(&size, rows.length, width)) {
                return EOVERFLOW;
            }
            if (!selected && (origin > indices->values.byte_count || size > indices->values.byte_count - origin)) {
                return EINVAL;
            }
            const uint8_t* const data = selected ? indices->selection.indices : indices->values.data;
            batch.index_width_bytes = width;
            batch.row_indices = (struct dora_bytes){data == nullptr ? nullptr : data + origin, size};
            if (selected) {
                batch.eligible_rows = (struct dora_utf8_mask){
                    {(const uint8_t*)active, active == nullptr ? 0 : (size_t)bitmap_bytes_for_bits(rows.length)},
                    0
                };
                status = DORA_OK;
            } else {
                status = operand_text_mask(operand, indices, rows, active, &batch.eligible_rows);
            }
        } else {
            operand_lanes(operand, batch.row_count, active);
            if (operand->gather.can_use_u32_map) {
                const uint32_t* map = operand_map(operand, rows);
                batch.row_indices = (struct dora_bytes){(const uint8_t*)map, (size_t)batch.row_count * sizeof *map};
                batch.index_width_bytes = 4;
            } else {
                operand_resolve_valid(operand, rows);
                batch.row_indices = (struct dora_bytes){
                    (const uint8_t*)operand->rows64,
                    (size_t)batch.row_count * sizeof *operand->rows64
                };
                batch.index_width_bytes = 8;
            }
            batch.eligible_rows = (struct dora_utf8_mask){
                {(const uint8_t*)operand->chunk_validity, (size_t)bitmap_bytes_for_bits(rows.length)},
                0
            };
            status = 0;
        }
    }
    if (view->selection.indices != nullptr) {
        const size_t width_bytes = view->layout == DORA_LAYOUT_BINARY         ? 4
                                   : view->layout == DORA_LAYOUT_LARGE_BINARY ? 8
                                                                              : 16;
        physical.length = view->layout == DORA_LAYOUT_BINARY_VIEW ? view->values.byte_count / width_bytes
                                                                  : view->offsets.byte_count / width_bytes - 1;
        physical.offset = batch.index_width_bytes == 4 || operand_text_index_width(operand) != 0 ? view->offset : 0;
        physical.length -= physical.offset;
    }
    if (status == 0) {
        status = operand_text_storage(operand, physical, &batch);
    }
    if (status == 0 && operand_text_index_width(operand) != 0) {
        if (batch.row_count != 0 && operand->bytes.view->validity.data != nullptr &&
            operand->chunk_validity == nullptr) {
            return EINVAL;
        }
        status = batch.index_width_bytes == 4 ? operand_text_child_u32(operand, &batch)
                                              : operand_text_child_u64(operand, &batch);
    }
    if (status == 0) {
        *out_batch = batch;
    }
    return status;
}

int dora_vector_operand_measure(struct vector_operand* operand,
                                struct dora_range rows,
                                bool unicode,
                                size_t* out_lengths,
                                struct dora_utf8_mask* out_mask) {
    if (operand == nullptr || out_lengths == nullptr || out_mask == nullptr || rows.length > KERNEL_CHUNK_ROW_COUNT ||
        rows.offset > operand->input->length || rows.length > operand->input->length - rows.offset) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &operand->input->nodes[operand->terminal_node].view;
    if (unicode || view->layout != DORA_LAYOUT_FIXED || operand->kind != VECTOR_OPERAND_BYTES) {
        struct dora_utf8_batch batch;
        int status = dora_vector_operand_prepare_utf8(operand, rows, nullptr, &batch);
        if (status == DORA_OK) {
            if (unicode) {
                status = dora_utf8_count_batch(&batch, out_lengths, nullptr);
            } else {
                batch.proof = DORA_UTF8_VALIDATED;
                size_t total;
                status = dora_utf8_measure_batch(&batch, out_lengths, &total, nullptr);
            }
        }
        if (status == DORA_OK) {
            *out_mask = batch.eligible_rows;
        }
        return status;
    }
    int status = DORA_OK;
    if (operand->chain_length == 0) {
        status = operand_text_mask(operand, view, rows, nullptr, out_mask);
    } else {
        operand_lanes(operand, (uint32_t)rows.length, nullptr);
        operand_resolve_valid(operand, rows);
        *out_mask = (struct dora_utf8_mask){
            {(const uint8_t*)operand->chunk_validity, (size_t)bitmap_bytes_for_bits(rows.length)},
            0
        };
    }
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t row = 0; row < rows.length; ++row) {
        out_lengths[row] = view->byte_width;
    }
    return DORA_OK;
}

int dora_vector_operand_prepare(struct vector_operand* operand,
                                struct dora_range rows,
                                const uint64_t* active,
                                struct kernel_packet* out_packet) {
    if (operand == nullptr || out_packet == nullptr || rows.length > KERNEL_CHUNK_ROW_COUNT) {
        return EINVAL;
    }
    const uint32_t count = (uint32_t)rows.length;
    const struct dora_vector_view* const view = operand->gather.view;
    switch (operand->kind) {
    case VECTOR_OPERAND_DIRECT: {
        const uint64_t* validity;
        const int status = operand_validity(operand, rows, active, &validity);
        if (status != DORA_OK) {
            return status;
        }
        *out_packet = (struct kernel_packet){
            .values = view->values.data,
            .validity = validity,
            .value_offset = view->offset + rows.offset,
            .count = count,
            .shape = KERNEL_SHAPE_FLAT,
            .placement = view->placement,
        };
        return DORA_OK;
    }
    case VECTOR_OPERAND_CONSTANT:
        *out_packet = (struct kernel_packet){
            .values = operand->values,
            .value_offset = operand->value_offset,
            .count = count,
            .shape = KERNEL_SHAPE_CONSTANT,
            .placement = operand->input->nodes[0].view.placement,
        };
        return DORA_OK;
    case VECTOR_OPERAND_CONSTANT_NULL:
        *out_packet = (struct kernel_packet){
            .validity = null_words,
            .count = count,
            .shape = KERNEL_SHAPE_CONSTANT,
            .placement = operand->input->nodes[0].view.placement,
        };
        return DORA_OK;
    case VECTOR_OPERAND_ENCODED: {
        operand_prepare_encoded(operand, rows, active, out_packet);
        return DORA_OK;
    }
    case VECTOR_OPERAND_BYTES:
        return operand_prepare_bytes(operand, rows, active, out_packet);
    case VECTOR_OPERAND_NESTED:
        return ENOTSUP;
    }
    return EINVAL;
}

int dora_vector_operand_prepare_validity(struct vector_operand* operand,
                                         struct dora_range rows,
                                         struct kernel_packet* out_packet) {
    if (operand == nullptr || out_packet == nullptr || rows.length > KERNEL_CHUNK_ROW_COUNT) {
        return EINVAL;
    }
    if (operand->kind != VECTOR_OPERAND_BYTES && operand->kind != VECTOR_OPERAND_ENCODED) {
        return dora_vector_operand_prepare(operand, rows, nullptr, out_packet);
    }
    const uint64_t* words = nullptr;
    int status = DORA_OK;
    if (operand->chain_length == 0) {
        status = operand_validity(operand, rows, nullptr, &words);
    } else if (!operand->is_all_valid) {
        operand_lanes(operand, (uint32_t)rows.length, nullptr);
        operand_resolve_valid(operand, rows);
        words = operand->chunk_validity;
    }
    if (status == DORA_OK) {
        *out_packet = (struct kernel_packet){.validity = words, .count = (uint32_t)rows.length};
    }
    return status;
}

int dora_vector_operand_prepare_rows(struct vector_operand* operand,
                                     uint32_t node,
                                     struct kernel_packet* out_packet,
                                     uint32_t count) {
    if (operand == nullptr || out_packet == nullptr || count > KERNEL_CHUNK_ROW_COUNT) {
        return EINVAL;
    }
    int status = operand_bind_chain(operand, node, true);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_vector_view* const view = &operand->input->nodes[operand->terminal_node].view;
    *out_packet = (struct kernel_packet){
        .validity = operand->chunk_validity,
        .count = count,
        .shape = KERNEL_SHAPE_FLAT,
        .placement = view->placement,
    };
    if (view->layout == DORA_LAYOUT_NULL) {
        out_packet->validity = null_words;
        out_packet->shape = KERNEL_SHAPE_CONSTANT;
        return DORA_OK;
    }
    operand_lanes(operand, count, nullptr);
    const bool bytes = type_is_bytes(view->type);
    status = bytes ? operand_bind_bytes(view, operand) : dora_kernel_gather_bind(view, &operand->gather);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_range rows = {.offset = UINT64_MAX, .length = count};
    if (bytes) {
        return operand_bytes_packet(operand, rows, out_packet);
    }
    operand_fixed_packet(operand, rows, out_packet);
    return DORA_OK;
}

int dora_vector_operand_resolve_slots(struct vector_operand* operand,
                                      uint32_t node,
                                      const struct vector_operand_rows* rows,
                                      uint32_t* out_terminal) {
    int status = operand_bind_chain(operand, node, false);
    *out_terminal = operand->terminal_node;
    if (status != DORA_OK) {
        return status;
    }
    operand_resolve(operand, UINT64_MAX, rows);
    const struct dora_vector_view* const view = &operand->input->nodes[operand->terminal_node].view;
    if (view->layout == DORA_LAYOUT_NULL) {
        dora_bitmap_set_range(rows->validity, 0, rows->count, false);
        return DORA_OK;
    }
    const bool selected = view->selection.indices != nullptr;
    memcpy(rows->slots, rows->rows, (size_t)rows->count * sizeof(uint64_t));
    if (selected) {
        struct kernel_decode_binding selection = {};
        status = dora_kernel_decode_bind(view, &selection);
        if (status == DORA_OK) {
            selection.resolve(&selection, UINT64_MAX, rows->count, rows->slots, rows->validity);
        }
    }
    if (status == DORA_OK && view->validity.data != nullptr) {
        struct dora_vector_view fixed = *view;
        fixed.layout = DORA_LAYOUT_FIXED;
        struct kernel_gather_binding gather = {};
        status = dora_kernel_gather_bind(&fixed, &gather);
        gather.view = view;
        if (status == DORA_OK) {
            dora_kernel_gather_validity(gather.view,
                                        selected ? rows->slots : rows->rows,
                                        rows->validity,
                                        rows->count,
                                        rows->validity);
        }
    }
    for (uint32_t lane = 0; status == DORA_OK && !selected && lane < rows->count; ++lane) {
        rows->slots[lane] += view->offset;
    }
    return status;
}

void dora_vector_operand_reset(struct vector_operand* operand) {
    for (uint32_t index = 0; operand->chain != nullptr && index < operand->chain_length; ++index) {
        if (operand->chain[index].kind == KERNEL_DECODE_RUN_END) {
            operand->chain[index].run_cursor = 0;
            operand->chain[index].run_end_cache = 0;
        }
    }
}
