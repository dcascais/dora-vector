// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/order.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/key.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Reads
//
//------------------------------------------------------------------------------

//--------------------------------------
// Equality
//--------------------------------------

static int equality_base(const struct dora_vector* vector, struct dora_vector* out_base) {
    struct dora_vector located = dora_vector_borrow(vector);
    for (;;) {
        struct vector_image* image = nullptr;
        int status = dora_vector_resolve(&located, &image);
        struct dora_schema_field field = {};
        if (status == DORA_OK) {
            status = dora_schema_get_field(&image->schema, &field);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (field.type.extension_name.byte_count != 0 &&
            (image->ownership.extension == nullptr ||
             (image->ownership.extension->flags & DORA_EXTENSION_FLAG_STORAGE_EQUALITY) == 0)) {
            return ENOTSUP;
        }
        if (field.layout != DORA_LAYOUT_DICTIONARY && field.layout != DORA_LAYOUT_RUN_END) {
            *out_base = located;
            return DORA_OK;
        }
        struct dora_vector child = {};
        status = dora_vector_get_child(&located, field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
        if (status != DORA_OK) {
            return status;
        }
        located = child;
    }
}

static int equality_types(const struct dora_vector* left, const struct dora_vector* right) {
    struct dora_vector first = {};
    struct dora_vector second = {};
    int status = equality_base(left, &first);
    if (status == DORA_OK) {
        status = equality_base(right, &second);
    }
    struct dora_logical_type left_type = {};
    struct dora_logical_type right_type = {};
    uint32_t left_children = 0;
    uint32_t right_children = 0;
    if (status == DORA_OK) {
        status = dora_vector_get_type(left, &left_type);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_type(right, &right_type);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child_count(&first, &left_children);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child_count(&second, &right_children);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (left_children != right_children || !dora_type_is_equal(&left_type, &right_type, left_children)) {
        return EINVAL;
    }
    for (uint32_t index = 0; index < left_children; ++index) {
        struct dora_vector left_child = {};
        struct dora_vector right_child = {};
        status = dora_vector_get_child(&first, index, &left_child);
        if (status == DORA_OK) {
            status = dora_vector_get_child(&second, index, &right_child);
        }
        if (status == DORA_OK) {
            status = equality_types(&left_child, &right_child);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

struct equality_operands {
    struct scratch_scope scope;
    struct vector_bound_node leaves[2];
    struct vector_bound_input inputs[2];
    struct vector_operand operands[2];
    struct kernel_binary_binding binding;
};

static int equality_bind(struct scratch_scope* scope,
                         const struct dora_vector* vector,
                         struct vector_bound_node* leaf,
                         size_t position,
                         struct vector_bound_input* out_input,
                         struct vector_operand* out_operand) {
    int status = dora_vector_bind_input(scope, vector, leaf, out_input);
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(scope, position, out_input, VECTOR_OPERAND_REQUIRE_PACKET, out_operand);
    }
    return status;
}

static bool equality_operand_is_terminal(const struct vector_bound_input* input, const struct vector_operand* operand) {
    for (uint32_t node = 0; node < input->node_count; ++node) {
        if (input->nodes[node].field.type.extension_name.byte_count != 0) {
            return false;
        }
    }
    return operand->kind != VECTOR_OPERAND_NESTED;
}

static int equality_kernel_bind(struct equality_operands* in_out_pair) {
    struct kernel_bind_request request = {
        .input_type = &in_out_pair->inputs[0].nodes[0].field.type,
        .right_type = &in_out_pair->inputs[1].nodes[0].field.type,
        .comparison = DORA_COMPARE_OP_EQ,
        .null_order = DORA_NULL_ORDER_LAST,
    };
    return dora_kernel_compare_distinct_bind(&request, &in_out_pair->binding);
}

static bool equality_try_decide(const struct dora_vector* left, const struct dora_vector* right, bool* out_is_equal) {
    const bool identical =
        left == right ||
        (left->handle.slot_index == right->handle.slot_index && left->handle.generation == right->handle.generation &&
         left->handle.pool_index == right->handle.pool_index && left->rows.offset == right->rows.offset);
    if (identical) {
        *out_is_equal = true;
        return true;
    }
    struct vector_image* left_image = nullptr;
    struct vector_image* right_image = nullptr;
    uint64_t left_nulls = 0;
    uint64_t right_nulls = 0;
    if (dora_vector_resolve(left, &left_image) == DORA_OK && dora_vector_resolve(right, &right_image) == DORA_OK &&
        dora_vector_try_get_exact_null_count(left, left_image, &left_nulls) &&
        dora_vector_try_get_exact_null_count(right, right_image, &right_nulls) && left_nulls != right_nulls) {
        *out_is_equal = false;
        return true;
    }
    return false;
}

struct equality_span {
    const uint8_t* values;
    const uint8_t* validity;
    uint64_t validity_bit;
};

// Carrier widths: 1, 2, 4, 8, 16, 32 bytes
#define DEFINE_EQUALITY_FLAT(W)                                                                                        \
    static bool equality_values_are_equal_##W(struct equality_span left, struct equality_span right, uint64_t count) { \
        for (uint64_t index = 0; index < count; index++) {                                                             \
            if (memcmp(left.values + index * (W), right.values + index * (W), (W)) != 0) {                             \
                return false;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        return true;                                                                                                   \
    }                                                                                                                  \
    static bool equality_masked_values_are_equal_##W(struct equality_span left,                                        \
                                                     struct equality_span right,                                       \
                                                     uint64_t count) {                                                 \
        for (uint64_t block_index = 0; block_index < count; block_index += 64) {                                       \
            const uint32_t block_element_count = count - block_index < 64 ? (uint32_t)(count - block_index) : 64;      \
            uint64_t value_difference = 0;                                                                             \
            for (uint32_t element_index = 0; element_index < block_element_count; element_index++) {                   \
                const uint64_t element = (block_index + element_index) * (W);                                          \
                const uint64_t values_differ =                                                                         \
                    memcmp(left.values + element, right.values + element, (W)) != 0 ? 1 : 0;                           \
                value_difference |= values_differ << element_index;                                                    \
            }                                                                                                          \
            const uint64_t validity_a =                                                                                \
                dora_bitmap_load_window(left.validity, left.validity_bit + block_index, block_element_count);          \
            const uint64_t validity_b =                                                                                \
                dora_bitmap_load_window(right.validity, right.validity_bit + block_index, block_element_count);        \
            if (validity_a != validity_b || (value_difference & validity_a) != 0) {                                    \
                return false;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        return true;                                                                                                   \
    }                                                                                                                  \
    static bool equality_spans_are_equal_##W(struct equality_span left, struct equality_span right, uint64_t count) {  \
        const bool left_has_validity = left.validity != nullptr;                                                       \
        const bool right_has_validity = right.validity != nullptr;                                                     \
        if (!left_has_validity && !right_has_validity) {                                                               \
            return count == 0 || memcmp(left.values, right.values, (size_t)count * (W)) == 0;                          \
        }                                                                                                              \
        if (left_has_validity != right_has_validity) {                                                                 \
            const struct equality_span* const masked = left_has_validity ? &left : &right;                             \
            if (dora_bitmap_count_set_bytes(masked->validity, masked->validity_bit, count) != count) {                 \
                return false;                                                                                          \
            }                                                                                                          \
            return equality_values_are_equal_##W(left, right, count);                                                  \
        }                                                                                                              \
        return equality_masked_values_are_equal_##W(left, right, count);                                               \
    }
DEFINE_EQUALITY_FLAT(1)
DEFINE_EQUALITY_FLAT(2)
DEFINE_EQUALITY_FLAT(4)
DEFINE_EQUALITY_FLAT(8)
DEFINE_EQUALITY_FLAT(16)
DEFINE_EQUALITY_FLAT(32)
#undef DEFINE_EQUALITY_FLAT

static bool equality_type_supports_flat_compare(enum dora_type type) {
    return (type >= DORA_TYPE_INT8 && type <= DORA_TYPE_UINT128) || type == DORA_TYPE_DECIMAL ||
           (type >= DORA_TYPE_DATE32 && type <= DORA_TYPE_DURATION);
}

static struct equality_span equality_span_of(const struct vector_operand* operand) {
    const struct dora_vector_view* const view = operand->gather.view;
    return (struct equality_span){
        .values = (const uint8_t*)view->values.data + view->offset * operand->gather.width,
        .validity = view->validity.data,
        .validity_bit = view->validity_bit_offset,
    };
}

// Signed integers, unsigned integers, Decimal, Date32, Date64, Time, Timestamp, Duration
static bool equality_try_compare_flat(const struct equality_operands* pair, uint64_t length, bool* out_is_equal) {
    const struct vector_operand* const left = &pair->operands[0];
    const struct vector_operand* const right = &pair->operands[1];
    if (left->kind != VECTOR_OPERAND_DIRECT || right->kind != VECTOR_OPERAND_DIRECT ||
        left->gather.view->layout != DORA_LAYOUT_FIXED || right->gather.view->layout != DORA_LAYOUT_FIXED ||
        !equality_type_supports_flat_compare(left->gather.view->type) ||
        left->gather.view->type != right->gather.view->type ||
        left->gather.view->physical_type != right->gather.view->physical_type ||
        left->gather.width != right->gather.width || left->gather.view->values.data == nullptr ||
        right->gather.view->values.data == nullptr) {
        return false;
    }
    const struct equality_span left_span = equality_span_of(left);
    const struct equality_span right_span = equality_span_of(right);
    switch (left->gather.width) {
    case 1:
        *out_is_equal = equality_spans_are_equal_1(left_span, right_span, length);
        return true;
    case 2:
        *out_is_equal = equality_spans_are_equal_2(left_span, right_span, length);
        return true;
    case 4:
        *out_is_equal = equality_spans_are_equal_4(left_span, right_span, length);
        return true;
    case 8:
        *out_is_equal = equality_spans_are_equal_8(left_span, right_span, length);
        return true;
    case 16:
        *out_is_equal = equality_spans_are_equal_16(left_span, right_span, length);
        return true;
    case 32:
        *out_is_equal = equality_spans_are_equal_32(left_span, right_span, length);
        return true;
    default:
        return false;
    }
}

static int equality_chunks(struct equality_operands* pair, uint64_t length, bool* out_is_equal) {
    uint64_t matches[KERNEL_CHUNK_WORD_COUNT];
    for (uint64_t first = 0; first < length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = {
            .offset = first,
            .length = length - first < KERNEL_CHUNK_ROW_COUNT ? length - first : KERNEL_CHUNK_ROW_COUNT,
        };
        struct kernel_packet packets[2];
        int status = dora_vector_operand_prepare(&pair->operands[0], rows, nullptr, &packets[0]);
        if (status == DORA_OK) {
            status = dora_vector_operand_prepare(&pair->operands[1], rows, nullptr, &packets[1]);
        }
        struct kernel_result result = {.values = matches, .row_capacity = (uint32_t)rows.length};
        if (status == DORA_OK) {
            status = pair->binding.fill(&packets[0], &packets[1], &pair->binding, &result);
        }
        if (status != DORA_OK) {
            return status;
        }
        const uint64_t word_count = bitmap_words_for_bits(rows.length);
        for (uint64_t word = 0; word < word_count; ++word) {
            const uint64_t mask = bitmap_word_tail_mask(rows.length, word);
            if ((matches[word] & mask) != mask) {
                *out_is_equal = false;
                return DORA_OK;
            }
        }
    }
    *out_is_equal = true;
    return DORA_OK;
}

static int equality_nested(struct equality_operands* pair,
                           const struct dora_vector* const sides[2],
                           uint64_t length,
                           bool* out_is_equal) {
    struct vector_image* left_image = nullptr;
    int status = dora_vector_resolve(sides[0], &left_image);
    struct vector_comparison_plan plan = {};
    struct vector_key_workspace workspace = {};
    if (status == DORA_OK) {
        status = dora_vector_key_plan_init(&pair->scope,
                                           &pair->inputs[0],
                                           &pair->inputs[1],
                                           DORA_COMPARE_OP_EQ,
                                           DORA_NULL_ORDER_LAST,
                                           &plan);
    }
    if (status == DORA_OK) {
        status = dora_vector_key_workspace_init(&pair->scope, &plan, KERNEL_CHUNK_ROW_COUNT, &workspace);
    }
    uint64_t matches[KERNEL_CHUNK_WORD_COUNT] = {};
    bool is_equal = true;
    const bool decided = status == DORA_OK && equality_try_decide(sides[0], sides[1], &is_equal);
    for (uint64_t first = 0; status == DORA_OK && !decided && is_equal && first < length;
         first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range rows = {
            .offset = first,
            .length = length - first < KERNEL_CHUNK_ROW_COUNT ? length - first : KERNEL_CHUNK_ROW_COUNT,
        };
        struct kernel_result result = {.values = matches, .row_capacity = (uint32_t)rows.length};
        status = dora_vector_key_compare_rows(&plan, rows, &workspace, &result);
        is_equal = dora_bitmap_count_set_words(matches, rows.length) == rows.length;
    }

    if (status == DORA_OK) {
        *out_is_equal = is_equal;
    }
    return status;
}

int dora_vector_is_equal(const struct dora_vector* left, const struct dora_vector* right, bool* out_is_equal) {
    if (out_is_equal == nullptr || !dora_vector_is_valid(left) || !dora_vector_is_valid(right)) {
        return EINVAL;
    }
    int status = equality_types(left, right);
    if (status != DORA_OK) {
        return status;
    }
    if (left->rows.length != right->rows.length) {
        *out_is_equal = false;
        return DORA_OK;
    }
    struct equality_operands pair = {};
    struct scratch_scope right_scope = {};
    struct dora_pool right_pool = {};
    struct dora_pool pool = {};
    status = dora_vector_get_pool(left, &pool);
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &pair.scope);
    }
    if (status == DORA_OK) {
        status = equality_bind(&pair.scope, left, &pair.leaves[0], 0, &pair.inputs[0], &pair.operands[0]);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_pool(right, &right_pool);
    }
    const bool same_pool = pool.index == right_pool.index && pool.generation == right_pool.generation;
    if (status == DORA_OK && !same_pool) {
        status = dora_scratch_scope_begin(right_pool, &right_scope);
    }
    if (status == DORA_OK) {
        status = equality_bind(same_pool ? &pair.scope : &right_scope,
                               right,
                               &pair.leaves[1],
                               1,
                               &pair.inputs[1],
                               &pair.operands[1]);
    }
    const bool terminal = status == DORA_OK && equality_operand_is_terminal(&pair.inputs[0], &pair.operands[0]) &&
                          equality_operand_is_terminal(&pair.inputs[1], &pair.operands[1]);
    if (terminal) {
        status = equality_kernel_bind(&pair);
    }
    const struct dora_vector* const sides[2] = {left, right};
    if (status == DORA_OK && terminal && !equality_try_decide(left, right, out_is_equal) &&
        !equality_try_compare_flat(&pair, left->rows.length, out_is_equal)) {
        status = equality_chunks(&pair, left->rows.length, out_is_equal);
    } else if (status == DORA_OK && !terminal) {
        status = equality_nested(&pair, sides, left->rows.length, out_is_equal);
    }
    dora_scratch_scope_end(&right_scope);
    dora_scratch_scope_end(&pair.scope);
    return status;
}

int dora_vector_compare_rows(const struct dora_vector* keys,
                             uint64_t left_row,
                             uint64_t right_row,
                             enum dora_compare_op comparison,
                             bool* out_matches) {
    struct dora_logical_type key_type = {};
    int status = dora_vector_get_type(keys, &key_type);
    if (status != DORA_OK) {
        return status;
    }
    if (key_type.extension_name.byte_count != 0) {
        const struct dora_extension_definition* definition = nullptr;
        const struct dora_allocation* parameters = nullptr;
        status = dora_vector_get_extension(keys, &definition, &parameters);
        const uint32_t required =
            comparison == DORA_COMPARE_OP_EQ ? DORA_EXTENSION_FLAG_STORAGE_EQUALITY : DORA_EXTENSION_FLAG_STORAGE_ORDER;
        if (status != DORA_OK || (definition->flags & required) == 0) {
            return ENOTSUP;
        }
    }
    bool left_null = false;
    bool right_null = false;
    status = dora_vector_is_null(keys, left_row, &left_null);
    if (status == DORA_OK) {
        status = dora_vector_is_null(keys, right_row, &right_null);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (left_null || right_null) {
        *out_matches = (comparison == DORA_COMPARE_OP_EQ && left_null == right_null) ||
                       (comparison == DORA_COMPARE_OP_LT && !left_null && right_null);
        return DORA_OK;
    }
    struct dora_vector_view view = {};
    status = dora_vector_get_view(keys, &view);
    if (status != DORA_OK) {
        return status;
    }
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
        struct dora_vector child = {};
        if (status == DORA_OK) {
            status = dora_vector_get_child(keys, child_index, &child);
        }
        return status == DORA_OK ? dora_vector_compare_rows(&child, left, right, comparison, out_matches) : status;
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
        struct dora_vector child = {};
        status = dora_vector_get_child(keys, left_member, &child);
        return status == DORA_OK ? dora_vector_compare_rows(&child, left, right, comparison, out_matches) : status;
    }
    if (view.type == DORA_TYPE_STRUCT || view.type == DORA_TYPE_LIST || view.type == DORA_TYPE_FIXED_LIST ||
        view.type == DORA_TYPE_MAP) {
        const struct dora_range left = view.type == DORA_TYPE_STRUCT ? (struct dora_range){left_row, 1}
                                                                     : dora_vector_view_get_elements(&view, left_row);
        const struct dora_range right = view.type == DORA_TYPE_STRUCT ? (struct dora_range){right_row, 1}
                                                                      : dora_vector_view_get_elements(&view, right_row);
        uint32_t children = 0;
        status = dora_vector_get_child_count(keys, &children);
        for (uint64_t row = 0; status == DORA_OK && row < left.length && row < right.length; ++row) {
            for (uint32_t index = 0; index < children; ++index) {
                struct dora_vector child = {};
                status = dora_vector_get_child(keys, index, &child);
                bool equal = false;
                if (status == DORA_OK) {
                    status = dora_vector_compare_rows(&child,
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
                    return dora_vector_compare_rows(&child,
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
