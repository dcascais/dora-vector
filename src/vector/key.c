// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "vector/key.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "hash/hash.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/read.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/scratch.h"
#include "types/order.h"
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
// Scratch Layout
//
//------------------------------------------------------------------------------

enum : uint32_t {
    KEY_SIDE_COUNT = 2,
    KEY_UNION_MEMBER_COUNT = 128
};

struct key_carver {
    uint8_t* base;
    size_t offset;
    bool has_overflow;
};

struct key_extent {
    size_t count;
    size_t width;
};

#define KEY_EXTENT(COUNT, TYPE) ((struct key_extent){.count = (COUNT), .width = sizeof(TYPE)})

static void* key_carve(struct key_carver* carver, struct key_extent extent) {
    if (extent.count == 0) {
        return nullptr;
    }
    size_t bytes = 0;
    const size_t start = carver->offset;
    if (ckd_mul(&bytes, extent.count, extent.width) || ckd_add(&bytes, bytes, 63) ||
        ckd_add(&carver->offset, start, bytes / 64 * 64)) {
        carver->has_overflow = true;
        return nullptr;
    }
    return carver->base == nullptr ? nullptr : carver->base + start;
}

static void key_lanes(uint64_t* words, uint32_t count) {
    dora_bitmap_set_range(words, 0, KERNEL_CHUNK_ROW_COUNT, false);
    dora_bitmap_set_range(words, 0, count, true);
}

//------------------------------------------------------------------------------
//
// Trees
//
//------------------------------------------------------------------------------

struct key_inputs {
    const struct vector_bound_input* inputs[KEY_SIDE_COUNT];
    uint32_t count;
};

struct key_entry {
    uint32_t nodes[KEY_SIDE_COUNT];
    uint32_t slot;
    uint32_t depth;
};

struct key_tree {
    struct vector_key_node* nodes;
    uint32_t* children;
    uint32_t node_count;
    uint32_t edge_count;
    uint32_t max_depth;
};

static bool key_is_nullable(const struct vector_bound_input* input, uint32_t node) {
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        if (view->validity.data != nullptr || view->layout == DORA_LAYOUT_NULL ||
            input->nodes[node].field.type.id == DORA_TYPE_UNION) {
            return true;
        }
        if (!vector_view_has_encoding_child(view)) {
            return false;
        }
        node = vector_input_encoding_child(input, node);
    }
}

static enum vector_key_kind key_kind(const struct vector_bound_node* node) {
    switch (node->field.type.id) {
    case DORA_TYPE_STRUCT:
        return VECTOR_KEY_STRUCT;
    case DORA_TYPE_LIST:
    case DORA_TYPE_FIXED_LIST:
    case DORA_TYPE_MAP:
        return VECTOR_KEY_SEQUENCE;
    case DORA_TYPE_UNION:
        return VECTOR_KEY_UNION;
    default:
        return VECTOR_KEY_LEAF;
    }
}

static int
key_tree_node(const struct key_inputs* inputs, const struct key_entry* entry, struct vector_key_node* out_node) {
    const uint32_t first_base = vector_input_value_node(inputs->inputs[0], entry->nodes[0]);
    const struct vector_bound_node* const first = &inputs->inputs[0]->nodes[first_base];
    const enum vector_key_kind kind = key_kind(first);
    *out_node = (struct vector_key_node){
        .kind = kind,
        .child_count = kind == VECTOR_KEY_LEAF       ? 0
                       : kind == VECTOR_KEY_SEQUENCE ? 1
                                                     : first->field.child_count,
    };
    if (kind == VECTOR_KEY_SEQUENCE && first->field.child_count == 0) {
        return EINVAL;
    }
    for (uint32_t side = 0; side < inputs->count; ++side) {
        const uint32_t base = vector_input_value_node(inputs->inputs[side], entry->nodes[side]);
        const struct vector_bound_node* const bound = &inputs->inputs[side]->nodes[base];
        if (key_kind(bound) != kind || bound->field.child_count != first->field.child_count) {
            return EINVAL;
        }
        out_node->input_nodes[side] = entry->nodes[side];
        out_node->base_nodes[side] = base;
        out_node->is_nullable[side] = key_is_nullable(inputs->inputs[side], entry->nodes[side]);
    }
    return DORA_OK;
}

static void key_tree_carve(struct key_carver* carver,
                           uint32_t capacity,
                           struct key_tree* in_out_tree,
                           struct key_entry** out_stack) {
    in_out_tree->nodes = key_carve(carver, KEY_EXTENT(capacity, struct vector_key_node));
    in_out_tree->children = key_carve(carver, KEY_EXTENT(capacity, uint32_t));
    *out_stack = key_carve(carver, KEY_EXTENT(capacity, struct key_entry));
}

static int key_tree_walk(const struct key_inputs* inputs, struct key_entry* stack, struct key_tree* in_out_tree) {
    const uint32_t capacity = inputs->inputs[0]->node_count;
    uint32_t stack_count = 1;
    stack[0] = (struct key_entry){.slot = UINT32_MAX, .depth = 1};
    while (stack_count != 0) {
        const struct key_entry entry = stack[--stack_count];
        if (in_out_tree->node_count == capacity) {
            return EINVAL;
        }
        const uint32_t id = in_out_tree->node_count++;
        struct vector_key_node* const node = &in_out_tree->nodes[id];
        const int status = key_tree_node(inputs, &entry, node);
        if (status != DORA_OK) {
            return status;
        }
        if (entry.slot != UINT32_MAX) {
            in_out_tree->children[entry.slot] = id;
        }
        if (node->child_count > capacity - in_out_tree->edge_count || node->child_count > capacity - stack_count) {
            return EINVAL;
        }
        node->child_begin = in_out_tree->edge_count;
        in_out_tree->edge_count += node->child_count;
        in_out_tree->max_depth = entry.depth > in_out_tree->max_depth ? entry.depth : in_out_tree->max_depth;
        for (uint32_t ordinal = node->child_count; ordinal != 0; --ordinal) {
            struct key_entry child = {.slot = node->child_begin + ordinal - 1, .depth = entry.depth + 1};
            for (uint32_t side = 0; side < inputs->count; ++side) {
                child.nodes[side] = dora_vector_input_child(node->base_nodes[side], inputs->inputs[side], ordinal - 1);
            }
            stack[stack_count++] = child;
        }
    }
    return DORA_OK;
}

static int key_tree_build(struct scratch_scope* scope, const struct key_inputs* inputs, struct key_tree* out_tree) {
    const struct vector_bound_input* const owner = inputs->inputs[0];
    struct key_carver sizing = {};
    struct key_tree tree = {};
    struct key_entry* stack = nullptr;
    key_tree_carve(&sizing, owner->node_count, &tree, &stack);
    if (sizing.has_overflow) {
        return EOVERFLOW;
    }
    void* storage = nullptr;
    int status = dora_scratch_allocate(scope, sizing.offset, &storage);
    if (status != DORA_OK) {
        return status;
    }
    struct key_carver carver = {.base = storage};
    key_tree_carve(&carver, owner->node_count, &tree, &stack);
    if (carver.has_overflow || tree.nodes == nullptr || tree.children == nullptr || stack == nullptr) {
        return EOVERFLOW;
    }
    status = key_tree_walk(inputs, stack, &tree);
    if (status != DORA_OK) {
        return status;
    }
    *out_tree = tree;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Resolution
//
//------------------------------------------------------------------------------

static uint32_t key_union_ordinal(const struct dora_vector_view* view, uint64_t slot) {
    int8_t type_id = 0;
    (memcpy)(&type_id, (const uint8_t*)view->type_ids.data + slot, sizeof(type_id));
    return type_id < 0 ? UINT32_MAX : view->union_members[(uint8_t)type_id];
}

static uint64_t key_union_row(const struct dora_vector_view* view, uint64_t slot) {
    if (view->layout != DORA_LAYOUT_DENSE_UNION) {
        return slot;
    }
    int32_t offset = 0;
    (memcpy)(&offset, (const uint8_t*)view->offsets.data + slot * sizeof(offset), sizeof(offset));
    return offset < 0 ? UINT64_MAX : (uint64_t)offset;
}

static int key_union_null(uint32_t node, struct vector_key_side* side, uint64_t slot, bool* out_null) {
    const struct vector_bound_input* const input = side->prepare.input;
    for (;;) {
        const struct vector_bound_node* const bound = &input->nodes[node];
        const uint32_t ordinal = key_union_ordinal(&bound->view, slot);
        if (ordinal >= bound->field.child_count) {
            return EINVAL;
        }
        uint64_t row = key_union_row(&bound->view, slot);
        uint64_t member_slot = 0;
        uint64_t valid = 1;
        const struct vector_operand_rows batch = {.rows = &row, .slots = &member_slot, .validity = &valid, .count = 1};
        const int status = dora_vector_operand_resolve_slots(&side->prepare,
                                                             dora_vector_input_child(node, input, ordinal),
                                                             &batch,
                                                             &node);
        if (status != DORA_OK) {
            return status;
        }
        if (valid == 0 || input->nodes[node].field.type.id != DORA_TYPE_UNION) {
            *out_null = valid == 0;
            return DORA_OK;
        }
        slot = member_slot;
    }
}

struct key_member_bits {
    const uint8_t* words;
    uint64_t origin;
};

static bool key_union_try_bind_plain_members(const struct vector_bound_input* input,
                                             uint32_t node,
                                             struct key_member_bits* out_members) {
    const uint32_t count = input->nodes[node].field.child_count;
    if (count > KEY_UNION_MEMBER_COUNT) {
        return false;
    }
    for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        const struct vector_bound_node* const member = &input->nodes[dora_vector_input_child(node, input, ordinal)];
        const struct dora_vector_view* const view = &member->view;
        if (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END ||
            view->layout == DORA_LAYOUT_NULL || view->selection.indices != nullptr ||
            member->field.type.id == DORA_TYPE_UNION) {
            return false;
        }
        out_members[ordinal] =
            (struct key_member_bits){.words = view->validity.data, .origin = view->validity_bit_offset};
    }
    return true;
}

static inline uint64_t key_union_bit(const struct dora_vector_view* view,
                                     uint32_t count,
                                     const struct key_member_bits* members,
                                     uint64_t slot,
                                     bool* in_out_fault) {
    const uint32_t ordinal = key_union_ordinal(view, slot);
    if (ordinal >= count) {
        *in_out_fault = true;
        return 0;
    }
    const struct key_member_bits member = members[ordinal];
    return member.words == nullptr || dora_bitmap_test(member.words, member.origin + key_union_row(view, slot));
}

// Union; layouts: DenseUnion, SparseUnion with plain members
static int key_union_words(const struct vector_bound_node* node,
                           const struct key_member_bits* members,
                           const struct vector_operand_rows* batch) {
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(batch->count);
    const uint32_t member_count = node->field.child_count;
    bool fault = false;
    for (uint32_t word = 0; word < word_count; ++word) {
        const uint64_t active = batch->validity[word];
        const uint64_t lanes = bitmap_word_tail_mask(batch->count, word);
        const uint64_t* const slots = batch->slots + (size_t)word * 64;
        uint64_t block = 0;
        if (active == lanes) {
            const uint32_t lane_count = batch->count - word * 64 < 64 ? batch->count - word * 64 : 64;
            for (uint32_t lane = 0; lane < lane_count; ++lane) {
                block |= key_union_bit(&node->view, member_count, members, slots[lane], &fault) << lane;
            }
        }
        for (uint64_t pending = active == lanes ? 0 : active; pending != 0;) {
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&pending);
            block |= key_union_bit(&node->view, member_count, members, slots[lane], &fault) << lane;
        }
        batch->validity[word] = active & block;
    }
    return fault ? EINVAL : DORA_OK;
}

static int key_resolve_valid(struct vector_key_side* side,
                             uint32_t entry,
                             const struct vector_operand_rows* batch,
                             uint32_t* out_base) {
    int status = dora_vector_operand_resolve_slots(&side->prepare, entry, batch, out_base);
    const uint32_t base = *out_base;
    if (status != DORA_OK || side->prepare.input->nodes[base].field.type.id != DORA_TYPE_UNION) {
        return status;
    }
    struct key_member_bits members[KEY_UNION_MEMBER_COUNT];
    if (key_union_try_bind_plain_members(side->prepare.input, base, members)) {
        return key_union_words(&side->prepare.input->nodes[base], members, batch);
    }
    const uint64_t word_count = bitmap_words_for_bits(batch->count);
    for (uint32_t word = 0; status == DORA_OK && word < word_count; ++word) {
        uint64_t active = batch->validity[word];
        while (status == DORA_OK && active != 0) {
            const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);
            bool is_null = false;
            status = key_union_null(base, side, batch->slots[lane], &is_null);
            if (status == DORA_OK && is_null) {
                bitmap_clear(batch->validity, lane);
            }
        }
    }
    return status;
}

//--------------------------------------
// Containers
//--------------------------------------

static struct dora_range key_range_fixed(const struct dora_vector_view* view, uint64_t slot) {
    return (struct dora_range){.offset = slot * view->list_element_count, .length = view->list_element_count};
}

// List, Map; layouts: List, LargeList
#define KEY_OFFSET_RANGE(S, T)                                                                                         \
    static struct dora_range key_range_##S(const struct dora_vector_view* view, uint64_t slot) {                       \
        T bounds[2] = {};                                                                                              \
        (memcpy)(bounds, (const uint8_t*)view->offsets.data + slot * sizeof(T), sizeof(bounds));                       \
        return (struct dora_range){.offset = bounds[0], .length = bounds[1] - bounds[0]};                              \
    }
KEY_OFFSET_RANGE(list32, int32_t)
KEY_OFFSET_RANGE(list64, int64_t)
#undef KEY_OFFSET_RANGE

// List; layouts: ListView, LargeListView
#define KEY_VIEW_RANGE(S, T)                                                                                           \
    static struct dora_range key_range_##S(const struct dora_vector_view* view, uint64_t slot) {                       \
        T offset = 0;                                                                                                  \
        T size = 0;                                                                                                    \
        (memcpy)(&offset, (const uint8_t*)view->offsets.data + slot * sizeof(T), sizeof(T));                           \
        (memcpy)(&size, (const uint8_t*)view->sizes.data + slot * sizeof(T), sizeof(T));                               \
        return (struct dora_range){.offset = offset, .length = size};                                                  \
    }
KEY_VIEW_RANGE(view32, int32_t)
KEY_VIEW_RANGE(view64, int64_t)
#undef KEY_VIEW_RANGE

#define KEY_RANGE_LOOP(LOAD)                                                                                           \
    for (uint32_t word = 0; word < word_count; ++word) {                                                               \
        uint64_t active = batch->validity[word];                                                                       \
        while (active != 0) {                                                                                          \
            const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                                \
            out_ranges[lane] = LOAD(view, batch->slots[lane]);                                                         \
        }                                                                                                              \
    }

// FixedList, List, Map; layouts: FixedList, List, LargeList, ListView, LargeListView
static int key_ranges(const struct dora_vector_view* view,
                      const struct vector_operand_rows* batch,
                      struct dora_range* out_ranges) {
    const uint64_t word_count = bitmap_words_for_bits(batch->count);
    switch (view->layout) {
    case DORA_LAYOUT_FIXED_LIST:
        KEY_RANGE_LOOP(key_range_fixed)
        return DORA_OK;
    case DORA_LAYOUT_LIST:
        KEY_RANGE_LOOP(key_range_list32)
        return DORA_OK;
    case DORA_LAYOUT_LARGE_LIST:
        KEY_RANGE_LOOP(key_range_list64)
        return DORA_OK;
    case DORA_LAYOUT_LIST_VIEW:
        KEY_RANGE_LOOP(key_range_view32)
        return DORA_OK;
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        KEY_RANGE_LOOP(key_range_view64)
        return DORA_OK;
    default:
        return EINVAL;
    }
}

#undef KEY_RANGE_LOOP

// Union; layouts: DenseUnion, SparseUnion
static int
key_members(const struct vector_bound_node* node, const struct vector_operand_rows* batch, uint32_t* out_members) {
    const uint64_t word_count = bitmap_words_for_bits(batch->count);
    for (uint32_t word = 0; word < word_count; ++word) {
        uint64_t active = batch->validity[word];
        while (active != 0) {
            const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);
            out_members[lane] = key_union_ordinal(&node->view, batch->slots[lane]);
            if (out_members[lane] >= node->field.child_count) {
                return EINVAL;
            }
            batch->slots[lane] = key_union_row(&node->view, batch->slots[lane]);
        }
    }
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Workspace
//
//------------------------------------------------------------------------------

static void key_workspace_carve(struct key_carver* carver,
                                uint32_t frame_capacity,
                                bool packets,
                                struct vector_key_workspace* in_out_workspace) {
    const struct vector_bound_input* const inputs[KEY_SIDE_COUNT] = {
        in_out_workspace->sides[0].prepare.input,
        in_out_workspace->sides[1].prepare.input,
    };
    const uint32_t rows = packets ? KERNEL_CHUNK_ROW_COUNT : in_out_workspace->row_capacity;
    for (uint32_t index = 0; index < KEY_SIDE_COUNT; ++index) {
        struct vector_key_side* const side = &in_out_workspace->sides[index];
        if (inputs[index] == nullptr) {
            continue;
        }
        side->prepare.chain =
            key_carve(carver, KEY_EXTENT((size_t)inputs[index]->node_count * 2, struct kernel_decode_binding));
        side->prepare.rows64 = key_carve(carver, KEY_EXTENT(rows, uint64_t));
        side->prepare.map32 = key_carve(carver, KEY_EXTENT(packets ? rows : 0, uint32_t));
        side->prepare.chunk_validity = key_carve(carver, KEY_EXTENT(KERNEL_CHUNK_WORD_COUNT, uint64_t));
        side->prepare.payload = key_carve(carver,
                                          (struct key_extent){
                                              .count = packets ? rows : 0,
                                              .width = sizeof(struct dora_i256),
                                          });
        side->slots = key_carve(carver, KEY_EXTENT(rows, uint64_t));
        side->ranges = key_carve(carver, KEY_EXTENT(packets ? rows : 0, struct dora_range));
        side->members = key_carve(carver, KEY_EXTENT(packets ? rows : 0, uint32_t));
    }
    const uint32_t capacity = in_out_workspace->row_capacity;
    const bool comparison = in_out_workspace->sides[1].prepare.input != nullptr;
    const uint32_t comparison_count = comparison ? frame_capacity : 0;
    const uint32_t hash_count = comparison ? 0 : frame_capacity;
    in_out_workspace->comparison_frames =
        key_carve(carver, KEY_EXTENT(comparison_count, struct vector_comparison_frame));
    for (uint32_t depth = 0; depth < comparison_count; ++depth) {
        const struct vector_comparison_frame frame = {
            .lanes = key_carve(carver, KEY_EXTENT(capacity, struct vector_comparison_lane)),
            .orders = key_carve(carver, KEY_EXTENT(capacity, int8_t)),
            .members = key_carve(carver, KEY_EXTENT(capacity, uint32_t)),
        };
        if (in_out_workspace->comparison_frames != nullptr) {
            in_out_workspace->comparison_frames[depth] = frame;
        }
    }
    in_out_workspace->hash_frames = key_carve(carver, KEY_EXTENT(hash_count, struct vector_hash_frame));
    for (uint32_t depth = 0; depth < hash_count; ++depth) {
        const struct vector_hash_frame frame = {
            .lanes = key_carve(carver, KEY_EXTENT(capacity, struct vector_hash_lane)),
            .running = key_carve(carver, KEY_EXTENT(capacity, uint64_t)),
            .hashes = key_carve(carver, KEY_EXTENT(capacity, uint64_t)),
            .members = key_carve(carver, KEY_EXTENT(capacity, uint32_t)),
        };
        if (in_out_workspace->hash_frames != nullptr) {
            in_out_workspace->hash_frames[depth] = frame;
        }
    }
}

#undef KEY_EXTENT

static int key_workspace_init(struct scratch_scope* scope,
                              uint32_t frame_capacity,
                              bool packets,
                              struct vector_key_workspace* in_out_workspace) {
    if (in_out_workspace->sides[0].prepare.input == nullptr || in_out_workspace->row_capacity == 0 ||
        in_out_workspace->row_capacity > KERNEL_CHUNK_ROW_COUNT) {
        *in_out_workspace = (struct vector_key_workspace){};
        return EINVAL;
    }
    struct vector_key_workspace workspace = *in_out_workspace;
    *in_out_workspace = (struct vector_key_workspace){};
    struct key_carver sizing = {};
    key_workspace_carve(&sizing, frame_capacity, packets, &workspace);
    if (sizing.has_overflow) {
        return EOVERFLOW;
    }
    void* storage = nullptr;
    const int status = dora_scratch_allocate(scope, sizing.offset, &storage);
    if (status != DORA_OK) {
        return status;
    }
    struct key_carver carver = {.base = storage};
    key_workspace_carve(&carver, frame_capacity, packets, &workspace);
    workspace.sides[0].prepare.kind = VECTOR_OPERAND_NESTED;
    workspace.sides[1].prepare.kind = VECTOR_OPERAND_NESTED;
    *in_out_workspace = workspace;
    return DORA_OK;
}

int dora_vector_key_workspace_init(struct scratch_scope* scope,
                                   const struct vector_comparison_plan* plan,
                                   uint32_t row_capacity,
                                   struct vector_key_workspace* out_workspace) {
    if (plan == nullptr || out_workspace == nullptr) {
        return EINVAL;
    }
    *out_workspace = (struct vector_key_workspace){
        .sides = {{.prepare.input = plan->left}, {.prepare.input = plan->right}},
        .row_capacity = row_capacity,
    };
    return key_workspace_init(scope, plan->max_depth, true, out_workspace);
}

int dora_vector_key_hash_workspace_init(struct scratch_scope* scope,
                                        const struct vector_hash_plan* plan,
                                        struct vector_key_workspace* out_workspace) {
    if (plan == nullptr || out_workspace == nullptr) {
        return EINVAL;
    }
    *out_workspace = (struct vector_key_workspace){
        .sides = {{.prepare.input = plan->input}},
        .row_capacity = KERNEL_CHUNK_ROW_COUNT,
    };
    return key_workspace_init(scope, plan->max_depth, true, out_workspace);
}

int dora_vector_key_validity_init(struct scratch_scope* scope,
                                  const struct vector_bound_input* input,
                                  uint32_t row_capacity,
                                  bool packets,
                                  struct vector_key_workspace* out_workspace) {
    if (out_workspace == nullptr) {
        return EINVAL;
    }
    *out_workspace = (struct vector_key_workspace){.sides = {{.prepare.input = input}}, .row_capacity = row_capacity};
    return key_workspace_init(scope, 0, packets, out_workspace);
}

int dora_vector_key_validity(struct vector_key_workspace* workspace, struct dora_range rows, uint64_t* out_validity) {
    if (workspace == nullptr || out_validity == nullptr || workspace->sides[0].prepare.rows64 == nullptr ||
        rows.length > KERNEL_CHUNK_ROW_COUNT) {
        return EINVAL;
    }
    struct vector_key_side* const side = &workspace->sides[0];
    const struct vector_operand_rows batch = {
        .rows = side->prepare.rows64,
        .slots = side->slots,
        .validity = side->prepare.chunk_validity,
        .count = (uint32_t)rows.length,
    };
    for (uint32_t lane = 0; lane < batch.count; ++lane) {
        batch.rows[lane] = rows.offset + lane;
    }
    key_lanes(batch.validity, batch.count);
    uint32_t base = 0;
    const int status = key_resolve_valid(side, 0, &batch, &base);
    if (status == DORA_OK) {
        dora_bitmap_copy_aligned_words(out_validity, batch.validity, (size_t)bitmap_words_for_bits(batch.count));
    }
    return status;
}

static int nested_first_null(struct scratch_scope* scope, struct vector_operand* operand, bool* out_null) {
    if (operand == nullptr || out_null == nullptr) {
        return EINVAL;
    }
    const struct dora_range first = {.length = 1};
    if (operand->kind != VECTOR_OPERAND_NESTED) {
        struct kernel_packet packet;
        const int status = dora_vector_operand_prepare(operand, first, nullptr, &packet);
        dora_vector_operand_reset(operand);
        *out_null =
            status == DORA_OK && packet.validity != nullptr && dora_bitmap_count_set_words(packet.validity, 1) == 0;
        return status;
    }
    uint64_t validity[KERNEL_CHUNK_WORD_COUNT];
    bitmap_fill_bytes((uint8_t*)validity, sizeof validity, false);
    struct vector_key_workspace workspace = {};
    int status = dora_vector_key_validity_init(scope, operand->input, 1, false, &workspace);
    if (status == DORA_OK) {
        status = dora_vector_key_validity(&workspace, first, validity);
    }

    *out_null = dora_bitmap_count_set_words(validity, 1) == 0;
    return status;
}

int dora_vector_key_constant_null(struct scratch_scope* scope,
                                  const struct vector_bound_input* inputs,
                                  struct vector_operand* operands,
                                  uint32_t count,
                                  bool* out_null) {
    *out_null = false;
    for (uint32_t index = 0; index < count; ++index) {
        if (inputs[index].is_constant && inputs[index].length != 0) {
            bool is_null = false;
            const int status = nested_first_null(scope, &operands[index], &is_null);
            if (status != DORA_OK) {
                return status;
            }
            *out_null |= is_null;
        }
    }
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Comparison
//
//------------------------------------------------------------------------------

//--------------------------------------
// Plans
//--------------------------------------

static int
comparison_leaf_bind(const struct vector_comparison_plan* plan, uint32_t id, struct vector_key_node* in_out_node) {
    struct kernel_bind_request request = {
        .input_type = &plan->left->nodes[in_out_node->base_nodes[0]].field.type,
        .right_type = &plan->right->nodes[in_out_node->base_nodes[1]].field.type,
        .null_order = id == 0 ? plan->null_order : DORA_NULL_ORDER_LAST,
    };
    return dora_kernel_order_bind(&request, &in_out_node->binding.order);
}

int dora_vector_key_plan_init(struct scratch_scope* scope,
                              const struct vector_bound_input* left,
                              const struct vector_bound_input* right,
                              enum dora_compare_op comparison,
                              enum dora_null_order null_order,
                              struct vector_comparison_plan* out_plan) {
    if (left == nullptr || right == nullptr || out_plan == nullptr || left->node_count == 0 || right->node_count == 0 ||
        comparison > DORA_COMPARE_OP_GE || null_order > DORA_NULL_ORDER_PROPAGATE) {
        return EINVAL;
    }
    const struct key_inputs inputs = {.inputs = {left, right}, .count = KEY_SIDE_COUNT};
    struct key_tree tree = {};
    int status = key_tree_build(scope, &inputs, &tree);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_comparison_plan plan = {
        .left = left,
        .right = right,
        .nodes = tree.nodes,
        .children = tree.children,
        .node_count = tree.node_count,
        .max_depth = tree.max_depth,
        .comparison = comparison,
        .null_order = null_order,
    };
    for (uint32_t id = 0; status == DORA_OK && id < plan.node_count; ++id) {
        if (plan.nodes[id].kind == VECTOR_KEY_LEAF) {
            status = comparison_leaf_bind(&plan, id, &plan.nodes[id]);
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_plan = plan;
    return DORA_OK;
}

//--------------------------------------
// Frames
//--------------------------------------

static int comparison_frame_leaf(const struct vector_comparison_plan* plan,
                                 struct vector_key_workspace* workspace,
                                 uint32_t depth) {
    const struct vector_comparison_frame* const frame = &workspace->comparison_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    struct kernel_packet packets[KEY_SIDE_COUNT];
    int status = DORA_OK;
    for (uint32_t side = 0; status == DORA_OK && side < KEY_SIDE_COUNT; ++side) {
        status = dora_vector_operand_prepare_rows(&workspace->sides[side].prepare,
                                                  node->input_nodes[side],
                                                  &packets[side],
                                                  frame->input_count);
    }
    struct kernel_result result = {
        .values = frame->orders,
        .validity = depth == 0 ? workspace->root_validity : nullptr,
        .row_capacity = frame->input_count,
    };
    if (status == DORA_OK) {
        status = node->binding.order.fill(&packets[0], &packets[1], &node->binding.order, &result);
    }
    return status;
}

static void comparison_frame_activate(struct vector_comparison_frame* frame,
                                      const struct vector_key_node* node,
                                      const struct vector_key_side* sides,
                                      uint32_t lane) {
    struct vector_comparison_lane* const target = &frame->lanes[frame->active_count];
    switch (node->kind) {
    case VECTOR_KEY_SEQUENCE:
        *target = (struct vector_comparison_lane){
            .left_elements = sides[0].ranges[lane],
            .right_elements = sides[1].ranges[lane],
            .result_row = lane,
        };
        break;
    case VECTOR_KEY_UNION: {
        const uint32_t left_member = sides[0].members[lane];
        const uint32_t right_member = sides[1].members[lane];
        if (left_member != right_member) {
            frame->orders[lane] = (int8_t)((left_member > right_member) - (left_member < right_member));
            return;
        }
        *target = (struct vector_comparison_lane){
            .left_row = sides[0].slots[lane],
            .right_row = sides[1].slots[lane],
            .result_row = lane,
        };
        frame->members[frame->active_count] = left_member;
        break;
    }
    default:
        *target = (struct vector_comparison_lane){
            .left_row = sides[0].prepare.rows64[lane],
            .right_row = sides[1].prepare.rows64[lane],
            .result_row = lane,
        };
        break;
    }
    ++frame->active_count;
}

static int comparison_frame_facts(const struct vector_key_node* node,
                                  struct vector_key_side* sides,
                                  const struct vector_operand_rows* batches,
                                  const uint32_t* bases) {
    int status = DORA_OK;
    for (uint32_t side = 0; status == DORA_OK && side < KEY_SIDE_COUNT; ++side) {
        const struct vector_bound_node* const base = &sides[side].prepare.input->nodes[bases[side]];
        if (node->kind == VECTOR_KEY_SEQUENCE) {
            status = key_ranges(&base->view, &batches[side], sides[side].ranges);
        } else if (node->kind == VECTOR_KEY_UNION) {
            status = key_members(base, &batches[side], sides[side].members);
        }
    }
    return status;
}

static int comparison_frame_enter(const struct vector_comparison_plan* plan,
                                  struct vector_key_workspace* workspace,
                                  uint32_t depth) {
    struct vector_comparison_frame* const frame = &workspace->comparison_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    struct vector_key_side* const sides = workspace->sides;
    const uint32_t count = frame->input_count;
    frame->is_entered = true;
    frame->active_count = 0;
    frame->child_ordinal = 0;
    frame->element = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        sides[0].prepare.rows64[lane] = frame->lanes[lane].left_row;
        sides[1].prepare.rows64[lane] = frame->lanes[lane].right_row;
    }
    if (node->kind == VECTOR_KEY_LEAF) {
        return comparison_frame_leaf(plan, workspace, depth);
    }
    struct vector_operand_rows batches[KEY_SIDE_COUNT];
    uint32_t bases[KEY_SIDE_COUNT] = {};
    for (uint32_t side = 0; side < KEY_SIDE_COUNT; ++side) {
        batches[side] = (struct vector_operand_rows){
            .rows = sides[side].prepare.rows64,
            .slots = sides[side].slots,
            .validity = sides[side].prepare.chunk_validity,
            .count = count,
        };
    }
    key_lanes(batches[0].validity, count);
    int status = key_resolve_valid(&sides[0], node->input_nodes[0], &batches[0], &bases[0]);
    if (depth == 0 && plan->should_skip_right_on_left_null) {
        dora_bitmap_copy_aligned_words(batches[1].validity, batches[0].validity, KERNEL_CHUNK_WORD_COUNT);
    } else {
        key_lanes(batches[1].validity, count);
    }
    if (status == DORA_OK) {
        status = key_resolve_valid(&sides[1], node->input_nodes[1], &batches[1], &bases[1]);
    }
    if (status == DORA_OK) {
        status = comparison_frame_facts(node, sides, batches, bases);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (depth == 0) {
        dora_bitmap_copy_aligned_words(workspace->root_validity, batches[0].validity, KERNEL_CHUNK_WORD_COUNT);
        dora_bitmap_and_aligned_words(workspace->root_validity, batches[1].validity, KERNEL_CHUNK_WORD_COUNT);
    }
    if (!node->is_nullable[0] && !node->is_nullable[1]) {
        for (uint32_t lane = 0; lane < count; ++lane) {
            comparison_frame_activate(frame, node, sides, lane);
        }
        return DORA_OK;
    }
    const int8_t left_null = depth == 0 && plan->null_order == DORA_NULL_ORDER_FIRST ? (int8_t)-1 : (int8_t)1;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const bool left_valid = dora_bitmap_test((const uint8_t*)batches[0].validity, lane);
        const bool right_valid = dora_bitmap_test((const uint8_t*)batches[1].validity, lane);
        if (left_valid && right_valid) {
            comparison_frame_activate(frame, node, sides, lane);
        } else {
            frame->orders[lane] = left_valid == right_valid ? (int8_t)0 : left_valid ? (int8_t)-left_null : left_null;
        }
    }
    return DORA_OK;
}

static int comparison_frame_push(const struct vector_comparison_plan* plan,
                                 struct vector_key_workspace* workspace,
                                 uint32_t depth,
                                 bool* out_pushed) {
    struct vector_comparison_frame* const frame = &workspace->comparison_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    struct vector_comparison_frame* const child = &workspace->comparison_frames[depth + 1];
    uint32_t count = 0;
    uint32_t ordinal = frame->child_ordinal;
    if (node->kind == VECTOR_KEY_UNION) {
        child->lanes[count++] = (struct vector_comparison_lane){
            .left_row = frame->lanes[frame->child_ordinal].left_row,
            .right_row = frame->lanes[frame->child_ordinal].right_row,
        };
        ordinal = frame->members[frame->child_ordinal];
    }
    for (uint32_t lane = 0; node->kind == VECTOR_KEY_STRUCT && lane < frame->active_count; ++lane) {
        child->lanes[count++] = (struct vector_comparison_lane){
            .left_row = frame->lanes[lane].left_row,
            .right_row = frame->lanes[lane].right_row,
        };
    }
    for (uint32_t lane = 0; node->kind == VECTOR_KEY_SEQUENCE && lane < frame->active_count; ++lane) {
        const struct vector_comparison_lane* const parent = &frame->lanes[lane];
        struct vector_comparison_lane* const target = &child->lanes[count++];
        *target = (struct vector_comparison_lane){};
        if (ckd_add(&target->left_row, parent->left_elements.offset, frame->element) ||
            ckd_add(&target->right_row, parent->right_elements.offset, frame->element)) {
            return EOVERFLOW;
        }
        ordinal = 0;
    }
    child->node = plan->children[node->child_begin + ordinal];
    child->input_count = count;
    child->is_entered = false;
    *out_pushed = true;
    return DORA_OK;
}

static void comparison_frame_settle(struct vector_comparison_frame* frame, const struct vector_key_node* node) {
    for (uint32_t lane = 0; node->kind == VECTOR_KEY_STRUCT && lane < frame->active_count; ++lane) {
        frame->orders[frame->lanes[lane].result_row] = 0;
    }
    frame->active_count = 0;
}

static int comparison_frame_child(const struct vector_comparison_plan* plan,
                                  struct vector_key_workspace* workspace,
                                  uint32_t depth,
                                  bool* out_pushed) {
    struct vector_comparison_frame* const frame = &workspace->comparison_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    *out_pushed = false;
    if (node->kind == VECTOR_KEY_LEAF || frame->active_count == 0) {
        return DORA_OK;
    }
    if (node->kind == VECTOR_KEY_SEQUENCE) {
        uint32_t kept = 0;
        for (uint32_t lane = 0; lane < frame->active_count; ++lane) {
            const struct vector_comparison_lane current = frame->lanes[lane];
            const uint64_t left_length = current.left_elements.length;
            const uint64_t right_length = current.right_elements.length;
            if (frame->element < left_length && frame->element < right_length) {
                frame->lanes[kept++] = current;
            } else {
                frame->orders[current.result_row] =
                    (int8_t)((left_length > right_length) - (left_length < right_length));
            }
        }
        frame->active_count = kept;
    }
    const uint32_t end = node->kind == VECTOR_KEY_UNION ? frame->active_count : node->child_count;
    if (frame->active_count == 0 || (node->kind != VECTOR_KEY_SEQUENCE && frame->child_ordinal == end)) {
        comparison_frame_settle(frame, node);
        return DORA_OK;
    }
    if (depth + 1 >= plan->max_depth) {
        return EINVAL;
    }
    return comparison_frame_push(plan, workspace, depth, out_pushed);
}

static void comparison_frame_return(const struct vector_comparison_plan* plan,
                                    struct vector_key_workspace* workspace,
                                    uint32_t child_depth) {
    struct vector_comparison_frame* const frame = &workspace->comparison_frames[child_depth - 1];
    const struct vector_comparison_frame* const child = &workspace->comparison_frames[child_depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    if (node->kind == VECTOR_KEY_UNION) {
        frame->orders[frame->lanes[frame->child_ordinal++].result_row] = child->orders[0];
        return;
    }
    uint32_t kept = 0;
    for (uint32_t lane = 0; lane < child->input_count; ++lane) {
        if (child->orders[lane] != 0) {
            frame->orders[frame->lanes[lane].result_row] = child->orders[lane];
        } else {
            frame->lanes[kept++] = frame->lanes[lane];
        }
    }
    frame->active_count = kept;
    if (node->kind == VECTOR_KEY_STRUCT) {
        ++frame->child_ordinal;
    } else {
        ++frame->element;
    }
}

#define DEFINE_KEY_RUN(NAME, PLAN, FRAMES)                                                                             \
    static int NAME##_run(const struct PLAN* plan, struct vector_key_workspace* workspace) {                           \
        uint32_t depth = 0;                                                                                            \
        for (;;) {                                                                                                     \
            int status = DORA_OK;                                                                                      \
            if (!workspace->FRAMES[depth].is_entered) {                                                                \
                status = NAME##_frame_enter(plan, workspace, depth);                                                   \
            }                                                                                                          \
            bool pushed = false;                                                                                       \
            if (status == DORA_OK) {                                                                                   \
                status = NAME##_frame_child(plan, workspace, depth, &pushed);                                          \
            }                                                                                                          \
            if (status != DORA_OK) {                                                                                   \
                return status;                                                                                         \
            }                                                                                                          \
            if (pushed) {                                                                                              \
                ++depth;                                                                                               \
            } else if (depth == 0) {                                                                                   \
                return DORA_OK;                                                                                        \
            } else {                                                                                                   \
                NAME##_frame_return(plan, workspace, depth--);                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }
DEFINE_KEY_RUN(comparison, vector_comparison_plan, comparison_frames)

static int
comparison_start(const struct vector_comparison_plan* plan, struct vector_key_workspace* workspace, uint32_t count) {
    if (plan == nullptr || workspace == nullptr || plan->node_count == 0 || plan->max_depth == 0 ||
        count > workspace->row_capacity) {
        return EINVAL;
    }
    struct vector_comparison_frame* const root = &workspace->comparison_frames[0];
    root->node = 0;
    root->input_count = count;
    root->is_entered = false;
    return comparison_run(plan, workspace);
}

static int comparison_rows(const struct vector_comparison_plan* plan,
                           struct dora_range rows,
                           struct vector_key_workspace* workspace) {
    if (workspace == nullptr || workspace->comparison_frames == nullptr || rows.length > workspace->row_capacity) {
        return EINVAL;
    }
    struct vector_comparison_lane* const lanes = workspace->comparison_frames[0].lanes;
    for (uint32_t lane = 0; lane < rows.length; ++lane) {
        lanes[lane] = (struct vector_comparison_lane){
            .left_row = rows.offset + lane,
            .right_row = plan->should_broadcast_right ? 0 : rows.offset + lane,
        };
    }
    return comparison_start(plan, workspace, (uint32_t)rows.length);
}

static uint32_t comparison_valid(const struct vector_comparison_plan* plan,
                                 const struct vector_key_workspace* workspace,
                                 uint32_t count,
                                 uint64_t* out_valid) {
    if (plan->null_order == DORA_NULL_ORDER_PROPAGATE) {
        dora_bitmap_copy_aligned_words(out_valid, workspace->root_validity, KERNEL_CHUNK_WORD_COUNT);
    } else {
        key_lanes(out_valid, count);
    }
    return count - (uint32_t)dora_bitmap_count_set_words(out_valid, count);
}

int dora_vector_key_compare_rows(const struct vector_comparison_plan* plan,
                                 struct dora_range rows,
                                 struct vector_key_workspace* workspace,
                                 struct kernel_result* out_result) {
    if (out_result == nullptr || out_result->values == nullptr) {
        return EINVAL;
    }
    const int status = comparison_rows(plan, rows, workspace);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t count = (uint32_t)rows.length;
    const size_t word_count = (size_t)bitmap_words_for_bits(count);
    const int8_t* const orders = workspace->comparison_frames[0].orders;
    bool truth[KERNEL_CHUNK_ROW_COUNT];
    for (uint32_t lane = 0; lane < count; ++lane) {
        truth[lane] = type_order_matches(orders[lane], plan->comparison);
    }
    uint64_t valid[KERNEL_CHUNK_WORD_COUNT];
    out_result->null_count = comparison_valid(plan, workspace, count, valid);
    dora_bitmap_copy_bool8_to_words(out_result->values, truth, 0, count);
    dora_bitmap_and_aligned_words(out_result->values, valid, word_count);
    if (out_result->validity != nullptr) {
        dora_bitmap_copy_aligned_words(out_result->validity, valid, word_count);
    }
    return DORA_OK;
}

int dora_vector_key_order_rows(const struct vector_comparison_plan* plan,
                               struct dora_range rows,
                               struct vector_key_workspace* workspace,
                               struct kernel_result* out_result) {
    if (out_result == nullptr || out_result->values == nullptr) {
        return EINVAL;
    }
    const int status = comparison_rows(plan, rows, workspace);
    if (status != DORA_OK) {
        return status;
    }
    const uint32_t count = (uint32_t)rows.length;
    memcpy(out_result->values, workspace->comparison_frames[0].orders, count);
    uint64_t valid[KERNEL_CHUNK_WORD_COUNT];
    out_result->null_count = comparison_valid(plan, workspace, count, valid);
    if (out_result->validity != nullptr) {
        dora_bitmap_copy_aligned_words(out_result->validity, valid, (size_t)bitmap_words_for_bits(count));
    }
    return DORA_OK;
}

int dora_vector_key_order(const struct vector_comparison_plan* plan,
                          uint64_t left_row,
                          uint64_t right_row,
                          struct vector_key_workspace* workspace,
                          int* out_order) {
    if (workspace == nullptr || workspace->comparison_frames == nullptr || out_order == nullptr) {
        return EINVAL;
    }
    workspace->comparison_frames[0].lanes[0] = (struct vector_comparison_lane){
        .left_row = left_row,
        .right_row = right_row,
    };
    const int status = comparison_start(plan, workspace, 1);
    if (status == DORA_OK) {
        const int8_t order = workspace->comparison_frames[0].orders[0];
        *out_order = (order > 0) - (order < 0);
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Hash
//
//------------------------------------------------------------------------------

//--------------------------------------
// Plans
//--------------------------------------

static const struct dora_logical_type hash_running_type = {.id = DORA_TYPE_UINT64};

static int hash_leaf_bind(const struct vector_hash_plan* plan, bool per_row, struct vector_key_node* in_out_node) {
    const struct dora_logical_type* const type = &plan->input->nodes[in_out_node->base_nodes[0]].field.type;
    struct kernel_bind_request request = {
        .input_type = per_row ? type : &hash_running_type,
        .right_type = per_row ? nullptr : type,
        .seed = plan->seed,
    };
    return per_row ? dora_kernel_hash_bind(&request, &in_out_node->binding.hashing.hash)
                   : dora_kernel_hash_combine_bind(&request, &in_out_node->binding.hashing.combine);
}

int dora_vector_key_hash_plan_init(struct scratch_scope* scope,
                                   const struct vector_bound_input* input,
                                   uint64_t seed,
                                   struct vector_hash_plan* out_plan) {
    if (input == nullptr || out_plan == nullptr || input->node_count == 0) {
        return EINVAL;
    }
    const struct key_inputs inputs = {.inputs = {input}, .count = 1};
    struct key_tree tree = {};
    int status = key_tree_build(scope, &inputs, &tree);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_hash_plan plan = {
        .input = input,
        .nodes = tree.nodes,
        .children = tree.children,
        .seed = seed,
        .node_count = tree.node_count,
        .max_depth = tree.max_depth,
    };
    status = plan.nodes[0].kind == VECTOR_KEY_LEAF ? EINVAL : DORA_OK;
    for (uint32_t id = 0; status == DORA_OK && id < plan.node_count; ++id) {
        const struct vector_key_node* const node = &plan.nodes[id];
        for (uint32_t ordinal = 0; status == DORA_OK && ordinal < node->child_count; ++ordinal) {
            struct vector_key_node* const child = &plan.nodes[plan.children[node->child_begin + ordinal]];
            if (child->kind == VECTOR_KEY_LEAF) {
                status = hash_leaf_bind(&plan, node->kind == VECTOR_KEY_SEQUENCE, child);
            }
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_plan = plan;
    return DORA_OK;
}

//--------------------------------------
// Frames
//--------------------------------------

static uint64_t hash_union_tag(const struct vector_bound_node* node, uint32_t member, uint64_t seed) {
    const int8_t* const codes = node->field.type.parameters.type_codes;
    return hash_value_i8(codes != nullptr ? codes[member] : (int8_t)member, seed);
}

static void hash_frame_activate(struct vector_hash_frame* frame,
                                const struct vector_key_node* node,
                                uint32_t lane,
                                const struct vector_key_side* side,
                                uint64_t seed) {
    const uint32_t active = frame->active_count++;
    struct vector_hash_lane* const target = &frame->lanes[active];
    *target = (struct vector_hash_lane){.row = side->prepare.rows64[lane], .result_row = lane};
    frame->running[active] = seed;
    if (node->kind == VECTOR_KEY_SEQUENCE) {
        target->elements = side->ranges[lane];
    } else if (node->kind == VECTOR_KEY_UNION) {
        const uint32_t member = side->members[lane];
        const struct vector_bound_node* const base = &side->prepare.input->nodes[node->base_nodes[0]];
        target->row = side->slots[lane];
        frame->running[active] = hash_union_tag(base, member, seed);
        frame->members[active] = member;
    }
}

static bool hash_is_per_row(const struct vector_hash_plan* plan, const struct vector_key_node* node) {
    return node->kind == VECTOR_KEY_SEQUENCE && plan->nodes[plan->children[node->child_begin]].kind == VECTOR_KEY_LEAF;
}

static int
hash_frame_leaf(const struct vector_hash_plan* plan, struct vector_key_workspace* workspace, uint32_t depth) {
    const struct vector_hash_frame* const frame = &workspace->hash_frames[depth];
    const struct vector_hash_frame* const parent = &workspace->hash_frames[depth - 1];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    const bool per_row = plan->nodes[parent->node].kind == VECTOR_KEY_SEQUENCE;
    const struct kernel_packet prior = {
        .values = parent->running,
        .value_offset = plan->nodes[parent->node].kind == VECTOR_KEY_UNION ? parent->child_ordinal : 0,
        .count = frame->input_count,
        .shape = KERNEL_SHAPE_FLAT,
    };
    struct kernel_packet packet;
    int status = dora_vector_operand_prepare_rows(&workspace->sides[0].prepare,
                                                  node->input_nodes[0],
                                                  &packet,
                                                  frame->input_count);
    struct kernel_result result = {.values = frame->hashes, .row_capacity = frame->input_count};
    if (status == DORA_OK && per_row) {
        status = node->binding.hashing.hash.fill(&packet, &node->binding.hashing.hash, &result);
    } else if (status == DORA_OK) {
        status = node->binding.hashing.combine.fill(&prior, &packet, &node->binding.hashing.combine, &result);
    }
    return status;
}

static int
hash_frame_enter(const struct vector_hash_plan* plan, struct vector_key_workspace* workspace, uint32_t depth) {
    struct vector_hash_frame* const frame = &workspace->hash_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    struct vector_key_side* const side = &workspace->sides[0];
    const uint32_t count = frame->input_count;
    frame->is_entered = true;
    frame->active_count = 0;
    frame->child_ordinal = 0;
    frame->element = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        side->prepare.rows64[lane] = frame->lanes[lane].row;
    }
    if (node->kind == VECTOR_KEY_LEAF) {
        return hash_frame_leaf(plan, workspace, depth);
    }
    const struct vector_operand_rows batch = {
        .rows = side->prepare.rows64,
        .slots = side->slots,
        .validity = side->prepare.chunk_validity,
        .count = count,
    };
    uint32_t base = 0;
    key_lanes(batch.validity, count);
    int status = key_resolve_valid(side, node->input_nodes[0], &batch, &base);
    if (status == DORA_OK && node->kind == VECTOR_KEY_SEQUENCE) {
        status = key_ranges(&side->prepare.input->nodes[base].view, &batch, side->ranges);
    } else if (status == DORA_OK && node->kind == VECTOR_KEY_UNION) {
        status = key_members(&side->prepare.input->nodes[base], &batch, side->members);
    }
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t lane = 0; lane < count; ++lane) {
        if (dora_bitmap_test((const uint8_t*)batch.validity, lane)) {
            hash_frame_activate(frame, node, lane, side, plan->seed);
        } else {
            frame->hashes[lane] = HASH_NULL_VALUE;
        }
    }
    return DORA_OK;
}

static int
hash_frame_rows(const struct vector_hash_frame* frame, uint32_t row_capacity, struct vector_hash_frame* in_out_child) {
    uint32_t count = 0;
    uint32_t lane = frame->child_ordinal;
    uint64_t element = frame->element;
    while (lane < frame->active_count && count < row_capacity) {
        const struct dora_range elements = frame->lanes[lane].elements;
        const uint64_t room = row_capacity - count;
        const uint64_t take = elements.length - element < room ? elements.length - element : room;
        for (uint64_t index = 0; index < take; ++index) {
            struct vector_hash_lane* const target = &in_out_child->lanes[count++];
            *target = (struct vector_hash_lane){};
            if (ckd_add(&target->row, elements.offset, element + index)) {
                return EOVERFLOW;
            }
        }
        element += take;
        if (element == elements.length) {
            ++lane;
            element = 0;
        }
    }
    in_out_child->input_count = count;
    return DORA_OK;
}

static int hash_frame_positions(const struct vector_hash_frame* frame, struct vector_hash_frame* in_out_child) {
    for (uint32_t lane = 0; lane < frame->active_count; ++lane) {
        struct vector_hash_lane* const target = &in_out_child->lanes[lane];
        *target = (struct vector_hash_lane){};
        if (ckd_add(&target->row, frame->lanes[lane].elements.offset, frame->element)) {
            return EOVERFLOW;
        }
    }
    in_out_child->input_count = frame->active_count;
    return DORA_OK;
}

static int hash_frame_push(const struct vector_hash_plan* plan,
                           struct vector_key_workspace* workspace,
                           uint32_t depth,
                           bool* out_pushed) {
    struct vector_hash_frame* const frame = &workspace->hash_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    struct vector_hash_frame* const child = &workspace->hash_frames[depth + 1];
    uint32_t ordinal = 0;
    int status = DORA_OK;
    switch (node->kind) {
    case VECTOR_KEY_UNION:
        child->lanes[0] = (struct vector_hash_lane){.row = frame->lanes[frame->child_ordinal].row};
        child->input_count = 1;
        ordinal = frame->members[frame->child_ordinal];
        break;
    case VECTOR_KEY_STRUCT:
        for (uint32_t lane = 0; lane < frame->active_count; ++lane) {
            child->lanes[lane] = (struct vector_hash_lane){.row = frame->lanes[lane].row};
        }
        child->input_count = frame->active_count;
        ordinal = frame->child_ordinal;
        break;
    default:
        status = hash_is_per_row(plan, node) ? hash_frame_rows(frame, workspace->row_capacity, child)
                                             : hash_frame_positions(frame, child);
        break;
    }
    if (status != DORA_OK) {
        return status;
    }
    child->node = plan->children[node->child_begin + ordinal];
    child->is_entered = false;
    *out_pushed = true;
    return DORA_OK;
}

static void hash_frame_publish(struct vector_hash_frame* frame) {
    for (uint32_t lane = 0; lane < frame->active_count; ++lane) {
        frame->hashes[frame->lanes[lane].result_row] = frame->running[lane];
    }
}

static int hash_frame_child(const struct vector_hash_plan* plan,
                            struct vector_key_workspace* workspace,
                            uint32_t depth,
                            bool* out_pushed) {
    struct vector_hash_frame* const frame = &workspace->hash_frames[depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    *out_pushed = false;
    if (node->kind == VECTOR_KEY_LEAF || frame->active_count == 0) {
        return DORA_OK;
    }
    if (hash_is_per_row(plan, node)) {
        while (frame->child_ordinal < frame->active_count &&
               frame->element == frame->lanes[frame->child_ordinal].elements.length) {
            frame->hashes[frame->lanes[frame->child_ordinal].result_row] = frame->running[frame->child_ordinal];
            ++frame->child_ordinal;
        }
        frame->active_count = frame->child_ordinal == frame->active_count ? 0 : frame->active_count;
    } else if (node->kind == VECTOR_KEY_SEQUENCE) {
        uint32_t kept = 0;
        for (uint32_t lane = 0; lane < frame->active_count; ++lane) {
            const struct vector_hash_lane current = frame->lanes[lane];
            const uint64_t running = frame->running[lane];
            if (frame->element < current.elements.length) {
                frame->running[kept] = running;
                frame->lanes[kept++] = current;
            } else {
                frame->hashes[current.result_row] = running;
            }
        }
        frame->active_count = kept;
    }
    const uint32_t end = node->kind == VECTOR_KEY_UNION ? frame->active_count : node->child_count;
    if (frame->active_count == 0 || (node->kind != VECTOR_KEY_SEQUENCE && frame->child_ordinal == end)) {
        if (node->kind == VECTOR_KEY_STRUCT) {
            hash_frame_publish(frame);
        }
        frame->active_count = 0;
        return DORA_OK;
    }
    if (depth + 1 >= plan->max_depth) {
        return EINVAL;
    }
    return hash_frame_push(plan, workspace, depth, out_pushed);
}

static void hash_frame_fold(struct vector_hash_frame* frame, const struct vector_hash_frame* child) {
    uint32_t lane = frame->child_ordinal;
    uint64_t element = frame->element;
    uint32_t index = 0;
    while (index < child->input_count) {
        const uint64_t length = frame->lanes[lane].elements.length;
        uint64_t running = frame->running[lane];
        for (; element < length && index < child->input_count; ++element) {
            running = hash_value_combine(running, child->hashes[index++]);
        }
        frame->running[lane] = running;
        if (element == length) {
            frame->hashes[frame->lanes[lane].result_row] = running;
            ++lane;
            element = 0;
        }
    }
    frame->child_ordinal = lane;
    frame->element = element;
}

static void
hash_frame_return(const struct vector_hash_plan* plan, struct vector_key_workspace* workspace, uint32_t child_depth) {
    struct vector_hash_frame* const frame = &workspace->hash_frames[child_depth - 1];
    const struct vector_hash_frame* const child = &workspace->hash_frames[child_depth];
    const struct vector_key_node* const node = &plan->nodes[frame->node];
    const bool leaf = plan->nodes[child->node].kind == VECTOR_KEY_LEAF;
    if (node->kind == VECTOR_KEY_UNION) {
        const uint32_t lane = frame->child_ordinal++;
        frame->hashes[frame->lanes[lane].result_row] =
            leaf ? child->hashes[0] : hash_value_combine(frame->running[lane], child->hashes[0]);
        return;
    }
    if (leaf && node->kind == VECTOR_KEY_SEQUENCE) {
        hash_frame_fold(frame, child);
        return;
    }
    if (leaf) {
        memcpy(frame->running, child->hashes, (size_t)child->input_count * sizeof(uint64_t));
    }
    for (uint32_t lane = 0; !leaf && lane < child->input_count; ++lane) {
        frame->running[lane] = hash_value_combine(frame->running[lane], child->hashes[lane]);
    }
    if (node->kind == VECTOR_KEY_STRUCT) {
        ++frame->child_ordinal;
    } else {
        ++frame->element;
    }
}

DEFINE_KEY_RUN(hash, vector_hash_plan, hash_frames)
#undef DEFINE_KEY_RUN

int dora_vector_key_hash_rows(const struct vector_hash_plan* plan,
                              struct dora_range rows,
                              struct vector_key_workspace* workspace,
                              uint64_t* out_hashes) {
    if (plan == nullptr || workspace == nullptr || out_hashes == nullptr || workspace->hash_frames == nullptr ||
        plan->node_count == 0 || plan->max_depth == 0 || rows.length > workspace->row_capacity) {
        return EINVAL;
    }
    struct vector_hash_frame* const root = &workspace->hash_frames[0];
    for (uint32_t lane = 0; lane < rows.length; ++lane) {
        root->lanes[lane] = (struct vector_hash_lane){.row = rows.offset + lane};
    }
    root->node = 0;
    root->input_count = (uint32_t)rows.length;
    root->is_entered = false;
    const int status = hash_run(plan, workspace);
    if (status == DORA_OK) {
        memcpy(out_hashes, root->hashes, (size_t)rows.length * sizeof(uint64_t));
    }
    return status;
}
