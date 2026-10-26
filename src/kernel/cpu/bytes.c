// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/binding/read.h"
#include "types/load.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Lanes and Sources
//
//------------------------------------------------------------------------------

struct bytes_lanes {
    const uint64_t* rows;
    const uint64_t* validity;
    uint64_t first_row;
    uint32_t count;
};

struct bytes_source {
    const uint8_t* values;
    const uint8_t* offsets;
    const uint8_t* data;
    const struct dora_view_buffer* buffers;
    uint64_t offset;
    uint64_t length;
    uint32_t width;
    bool is_selected;
};

static constexpr int32_t BYTES_BINARY_VIEW_INLINE_CAPACITY = (int32_t)sizeof(union dora_binary_view_payload);

static inline struct bytes_source bytes_source(const struct dora_vector_view* view) {
    return (struct bytes_source){
        .values = view->values.data,
        .offsets = view->offsets.data,
        .data = view->data_buffer_count != 0 ? view->data_buffers[0].data : nullptr,
        .buffers = view->data_buffers,
        .offset = view->offset,
        .length = view->length,
        .width = view->byte_width,
        .is_selected = view->selection.indices != nullptr,
    };
}

static inline uint64_t bytes_row(const struct bytes_lanes* lanes, uint32_t lane) {
    return lanes->rows != nullptr ? lanes->rows[lane] : lanes->first_row + lane;
}

static inline bool bytes_slot(const struct bytes_source* source, uint64_t row, uint64_t* out_slot) {
    *out_slot = source->is_selected ? row : source->offset + row;
    return source->is_selected || row < source->length;
}

//------------------------------------------------------------------------------
//
// Descriptors
//
//------------------------------------------------------------------------------

static inline void
bytes_describe_fixed(const struct bytes_source* source, uint64_t slot, struct dora_bytes* out_value) {
    *out_value = (struct dora_bytes){
        .data = source->values == nullptr ? nullptr : source->values + slot * source->width,
        .byte_count = source->width,
    };
}

static inline void
bytes_describe_binary32(const struct bytes_source* source, uint64_t slot, struct dora_bytes* out_value) {
    const int32_t start = type_load_i32(source->offsets, slot);
    const int32_t end = type_load_i32(source->offsets, slot + 1);
    *out_value = (struct dora_bytes){
        .data = source->data == nullptr ? nullptr : source->data + start,
        .byte_count = end - start,
    };
}

static inline void
bytes_describe_binary64(const struct bytes_source* source, uint64_t slot, struct dora_bytes* out_value) {
    const int64_t start = type_load_i64(source->offsets, slot);
    const int64_t end = type_load_i64(source->offsets, slot + 1);
    *out_value = (struct dora_bytes){
        .data = source->data == nullptr ? nullptr : source->data + start,
        .byte_count = end - start,
    };
}

static inline void
bytes_describe_binary_view(const struct bytes_source* source, uint64_t slot, struct dora_bytes* out_value) {
    const uint8_t* const entry = source->values + slot * sizeof(struct dora_binary_view_entry);
    int32_t byte_count = 0;
    (memcpy)(&byte_count, entry, sizeof(byte_count));
    if (byte_count <= BYTES_BINARY_VIEW_INLINE_CAPACITY) {
        *out_value = (struct dora_bytes){
            .data = entry + offsetof(struct dora_binary_view_entry, payload),
            .byte_count = byte_count,
        };
        return;
    }
    struct dora_binary_view_reference reference = {};
    (memcpy)(&reference, entry + offsetof(struct dora_binary_view_entry, payload), sizeof(reference));
    *out_value = (struct dora_bytes){
        .data = (const uint8_t*)source->buffers[reference.buffer_index].data + reference.byte_offset,
        .byte_count = byte_count,
    };
}

//------------------------------------------------------------------------------
//
// Fills
//
//------------------------------------------------------------------------------

static inline int bytes_describe_rows(const struct bytes_source* source, const struct bytes_lanes* lanes) {
    return lanes->count == 0 || source->is_selected ||
                   (lanes->first_row < source->length && lanes->count <= source->length - lanes->first_row)
               ? DORA_OK
               : EINVAL;
}

#define DEFINE_BYTES_FILL(S)                                                                                           \
    static int bytes_fill_##S(const struct kernel_bytes_binding* binding,                                              \
                              uint64_t first_row,                                                                      \
                              const uint64_t* rows,                                                                    \
                              const uint64_t* validity,                                                                \
                              uint32_t count,                                                                          \
                              struct dora_bytes* restrict out_values) {                                                \
        const struct bytes_lanes lanes = {.rows = rows, .validity = validity, .first_row = first_row, .count = count}; \
        const struct bytes_source source = bytes_source(binding->view);                                                \
        if (lanes.rows == nullptr && lanes.validity == nullptr) {                                                      \
            const int status = bytes_describe_rows(&source, &lanes);                                                   \
            const uint64_t base = source.is_selected ? lanes.first_row : source.offset + lanes.first_row;              \
            for (uint32_t lane = 0; status == DORA_OK && lane < lanes.count; ++lane) {                                 \
                bytes_describe_##S(&source, base + lane, &out_values[lane]);                                           \
            }                                                                                                          \
            return status;                                                                                             \
        }                                                                                                              \
        const uint64_t word_count = bitmap_words_for_bits(lanes.count);                                                \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            const uint32_t first = word * 64;                                                                          \
            const uint32_t end = lanes.count - first < 64 ? lanes.count : first + 64;                                  \
            const uint64_t mask = bitmap_word_tail_mask(lanes.count, word);                                            \
            uint64_t active = lanes.validity == nullptr ? mask : lanes.validity[word] & mask;                          \
            uint64_t slot = 0;                                                                                         \
            if (active == mask) {                                                                                      \
                for (uint32_t lane = first; lane < end; ++lane) {                                                      \
                    if (!bytes_slot(&source, bytes_row(&lanes, lane), &slot)) {                                        \
                        return EINVAL;                                                                                 \
                    }                                                                                                  \
                    bytes_describe_##S(&source, slot, &out_values[lane]);                                              \
                }                                                                                                      \
                continue;                                                                                              \
            }                                                                                                          \
            memset(out_values + first, 0, (size_t)(end - first) * sizeof(*out_values));                                \
            while (active != 0) {                                                                                      \
                const uint32_t lane = first + dora_bitmap_word_pop_lowest_set(&active);                                \
                if (!bytes_slot(&source, bytes_row(&lanes, lane), &slot)) {                                            \
                    return EINVAL;                                                                                     \
                }                                                                                                      \
                bytes_describe_##S(&source, slot, &out_values[lane]);                                                  \
            }                                                                                                          \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
DEFINE_BYTES_FILL(fixed)
DEFINE_BYTES_FILL(binary32)
DEFINE_BYTES_FILL(binary64)
DEFINE_BYTES_FILL(binary_view)
#undef DEFINE_BYTES_FILL

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

static typeof(&bytes_fill_fixed) const bytes_fills[DORA_LAYOUT_BINARY_VIEW + 1] = {
    [DORA_LAYOUT_FIXED] = bytes_fill_fixed,
    [DORA_LAYOUT_BINARY] = bytes_fill_binary32,
    [DORA_LAYOUT_LARGE_BINARY] = bytes_fill_binary64,
    [DORA_LAYOUT_BINARY_VIEW] = bytes_fill_binary_view,
};

int dora_kernel_bytes_bind(const struct dora_vector_view* view, struct kernel_bytes_binding* out_binding) {
    if (view == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    typeof(&bytes_fill_fixed) const fill =
        view->layout <= DORA_LAYOUT_BINARY_VIEW ? bytes_fills[view->layout] : nullptr;
    if (fill == nullptr) {
        return ENOTSUP;
    }
    *out_binding = (struct kernel_bytes_binding){.fill = fill, .view = view};
    return DORA_OK;
}
