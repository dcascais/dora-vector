// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Borrowed vector buffers and row access.

#ifndef DORA_VECTOR_VIEW_H
#define DORA_VECTOR_VIEW_H

#include "dora/vector/bitmap.h"
#include "dora/vector/device.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Buffers and Cursors
//
//------------------------------------------------------------------------------

struct dora_view_buffer {
    /// Buffer start.
    const void* data;
    /// Logical extent in bytes.
    size_t byte_count;
    /// Readable extent, including available padding.
    size_t readable_byte_count;
};

struct dora_vector_selection {
    /// Borrowed row indices; nullptr for identity selection.
    const void* indices;
    /// UInt32 or UInt64 index carrier.
    enum dora_physical_type physical_type;
};

struct dora_vector_view {
    /// Buffer placement.
    struct dora_placement placement;
    /// Buffer readiness.
    struct dora_readiness readiness;
    /// Logical row count.
    uint64_t length;
    /// Null rows; -1 when unknown.
    int64_t null_count;
    /// Physical element offset.
    uint64_t offset;
    /// Start within the validity bitmap, in bits.
    uint64_t validity_bit_offset;
    /// Start within run-end values, in elements.
    uint64_t run_end_element_offset;
    /// Logical type.
    enum dora_type type;
    /// Value or index carrier.
    enum dora_physical_type physical_type;
    /// Storage layout.
    enum dora_layout layout;
    /// FixedBinary bytes per row.
    uint32_t byte_width;
    /// FixedList elements per row.
    uint32_t list_element_count;
    /// Logical-to-physical row selection.
    struct dora_vector_selection selection;
    /// Packed row validity; absent buffers mean all valid outside Null layout.
    struct dora_view_buffer validity;
    /// Fixed values, dictionary indices, view entries or run ends.
    struct dora_view_buffer values;
    /// Byte, list or dense-union offsets.
    struct dora_view_buffer offsets;
    /// List-view element counts.
    struct dora_view_buffer sizes;
    /// Union type codes.
    struct dora_view_buffer type_ids;
    /// Type-code-to-child mapping.
    const uint8_t* union_members;
    /// Borrowed byte payload buffers.
    const struct dora_view_buffer* data_buffers;
    /// Payload buffer count.
    uint32_t data_buffer_count;
};

struct dora_run_cursor {
    /// Current run index.
    uint64_t run;
    /// Next logical row.
    uint64_t row;
};

struct dora_row_cursor {
    /// Next bitmap word index.
    uint64_t word_index;
    /// Remaining rows in the word.
    uint64_t word;
};

//------------------------------------------------------------------------------
//
// Values
//
//------------------------------------------------------------------------------

//--------------------------------------
// Signed Integers
//--------------------------------------

/// Int8; dictionary indices: Int8
static inline const int8_t* dora_vector_view_get_values_i8(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I8 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(int8_t);
    if ((uintptr_t)address % alignof(int8_t) != 0) {
        return nullptr;
    }
    return (const int8_t*)address;
}

/// Int16; dictionary indices: Int16
static inline const int16_t* dora_vector_view_get_values_i16(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I16 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(int16_t);
    if ((uintptr_t)address % alignof(int16_t) != 0) {
        return nullptr;
    }
    return (const int16_t*)address;
}

/// Int32, Decimal (32-bit), Date32 (days), Time (s/ms), IntervalYearMonth; dictionary indices: Int32
static inline const int32_t* dora_vector_view_get_values_i32(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I32 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(int32_t);
    if ((uintptr_t)address % alignof(int32_t) != 0) {
        return nullptr;
    }
    return (const int32_t*)address;
}

/// Int64, Decimal (64-bit), Date64 (ms), Time (us/ns), Timestamp, Duration; dictionary indices: Int64
static inline const int64_t* dora_vector_view_get_values_i64(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I64 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(int64_t);
    if ((uintptr_t)address % alignof(int64_t) != 0) {
        return nullptr;
    }
    return (const int64_t*)address;
}

/// Int128, Decimal (128-bit)
static inline const struct dora_i128* dora_vector_view_get_values_i128(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I128 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_i128);
    if ((uintptr_t)address % alignof(struct dora_i128) != 0) {
        return nullptr;
    }
    return (const struct dora_i128*)address;
}

/// Int256, Decimal (256-bit)
static inline const struct dora_i256* dora_vector_view_get_values_i256(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_I256 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_i256);
    if ((uintptr_t)address % alignof(struct dora_i256) != 0) {
        return nullptr;
    }
    return (const struct dora_i256*)address;
}

//--------------------------------------
// Unsigned Integers
//--------------------------------------

/// UInt8; dictionary indices: UInt8
static inline const uint8_t* dora_vector_view_get_values_u8(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_U8 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(uint8_t);
    if ((uintptr_t)address % alignof(uint8_t) != 0) {
        return nullptr;
    }
    return address;
}

/// UInt16; dictionary indices: UInt16
static inline const uint16_t* dora_vector_view_get_values_u16(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_U16 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(uint16_t);
    if ((uintptr_t)address % alignof(uint16_t) != 0) {
        return nullptr;
    }
    return (const uint16_t*)address;
}

/// UInt32; dictionary indices: UInt32
static inline const uint32_t* dora_vector_view_get_values_u32(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_U32 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(uint32_t);
    if ((uintptr_t)address % alignof(uint32_t) != 0) {
        return nullptr;
    }
    return (const uint32_t*)address;
}

/// UInt64; dictionary indices: UInt64
static inline const uint64_t* dora_vector_view_get_values_u64(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_U64 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(uint64_t);
    if ((uintptr_t)address % alignof(uint64_t) != 0) {
        return nullptr;
    }
    return (const uint64_t*)address;
}

static inline const struct dora_u128* dora_vector_view_get_values_u128(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_U128 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_u128);
    if ((uintptr_t)address % alignof(struct dora_u128) != 0) {
        return nullptr;
    }
    return (const struct dora_u128*)address;
}

//--------------------------------------
// Floating Point
//--------------------------------------

static inline const struct dora_f16_bits* dora_vector_view_get_values_f16(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_F16 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_f16_bits);
    if ((uintptr_t)address % alignof(struct dora_f16_bits) != 0) {
        return nullptr;
    }
    return (const struct dora_f16_bits*)address;
}

static inline const float* dora_vector_view_get_values_f32(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_F32 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(float);
    if ((uintptr_t)address % alignof(float) != 0) {
        return nullptr;
    }
    return (const float*)address;
}

static inline const double* dora_vector_view_get_values_f64(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_F64 || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(double);
    if ((uintptr_t)address % alignof(double) != 0) {
        return nullptr;
    }
    return (const double*)address;
}

//--------------------------------------
// Intervals
//--------------------------------------

static inline const struct dora_interval_day_time*
dora_vector_view_get_values_interval_day_time(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address = (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_interval_day_time);
    if ((uintptr_t)address % alignof(struct dora_interval_day_time) != 0) {
        return nullptr;
    }
    return (const struct dora_interval_day_time*)address;
}

static inline const struct dora_interval_month_day_nano*
dora_vector_view_get_values_interval_month_day_nano(const struct dora_vector_view* view) {
    if (view->physical_type != DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO || view->selection.indices != nullptr ||
        view->layout == DORA_LAYOUT_RUN_END || view->values.data == nullptr) {
        return nullptr;
    }
    const uint8_t* address =
        (const uint8_t*)view->values.data + view->offset * sizeof(struct dora_interval_month_day_nano);
    if ((uintptr_t)address % alignof(struct dora_interval_month_day_nano) != 0) {
        return nullptr;
    }
    return (const struct dora_interval_month_day_nano*)address;
}

//------------------------------------------------------------------------------
//
// Rows
//
//------------------------------------------------------------------------------

//--------------------------------------
// Coordinates and Nullness
//--------------------------------------

static inline uint64_t dora_vector_view_get_physical_row(const struct dora_vector_view* view, uint64_t row) {
    uint64_t index = row;
    if (view->selection.indices != nullptr) {
        if (view->selection.physical_type == DORA_PHYSICAL_TYPE_U32) {
            uint32_t selected;
            memcpy(&selected, (const uint8_t*)view->selection.indices + row * sizeof(selected), sizeof(selected));
            index = selected;
        } else {
            memcpy(&index, (const uint8_t*)view->selection.indices + row * sizeof(index), sizeof(index));
        }
    }
    return view->offset + index;
}

static inline bool dora_vector_view_is_null(const struct dora_vector_view* view, uint64_t row) {
    if (view->layout == DORA_LAYOUT_NULL) {
        return true;
    }
    if (view->validity.data == nullptr) {
        return false;
    }
    const uint64_t index = dora_vector_view_get_physical_row(view, row) - view->offset;
    return !dora_bitmap_test(view->validity.data, view->validity_bit_offset + index);
}

//--------------------------------------
// Boolean
//--------------------------------------

static inline bool dora_vector_view_get_bool(const struct dora_vector_view* view, uint64_t row) {
    if (dora_vector_view_is_null(view, row)) {
        return false;
    }
    return dora_bitmap_test(view->values.data, dora_vector_view_get_physical_row(view, row));
}

//--------------------------------------
// Bytes and Children
//--------------------------------------

/// FixedBinary, Binary, Utf8; layouts: Fixed, Binary, LargeBinary, BinaryView
static inline struct dora_bytes dora_vector_view_get_bytes(const struct dora_vector_view* view, uint64_t row) {
    if (view->layout == DORA_LAYOUT_NULL) {
        return (struct dora_bytes){};
    }
    const uint64_t slot = dora_vector_view_get_physical_row(view, row);
    if (view->validity.data != nullptr &&
        !dora_bitmap_test(view->validity.data, view->validity_bit_offset + slot - view->offset)) {
        return (struct dora_bytes){};
    }
    switch (view->layout) {
    case DORA_LAYOUT_FIXED: {
        const uint8_t* data = view->values.data;
        return (struct dora_bytes){
            .data = data == nullptr ? nullptr : data + slot * view->byte_width,
            .byte_count = view->byte_width
        };
    }
    case DORA_LAYOUT_BINARY: {
        int32_t begin;
        int32_t end;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&end, (const uint8_t*)view->offsets.data + (slot + 1) * sizeof(end), sizeof(end));
        const uint8_t* data = view->data_buffers[0].data;
        return (struct dora_bytes){.data = data == nullptr ? nullptr : data + begin, .byte_count = end - begin};
    }
    case DORA_LAYOUT_LARGE_BINARY: {
        int64_t begin;
        int64_t end;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&end, (const uint8_t*)view->offsets.data + (slot + 1) * sizeof(end), sizeof(end));
        const uint8_t* data = view->data_buffers[0].data;
        return (struct dora_bytes){.data = data == nullptr ? nullptr : data + begin, .byte_count = end - begin};
    }
    case DORA_LAYOUT_BINARY_VIEW: {
        const uint8_t* entry = (const uint8_t*)view->values.data + slot * sizeof(struct dora_binary_view_entry);
        int32_t count;
        memcpy(&count, entry, sizeof(count));
        if (count <= 12) {
            return (struct dora_bytes){
                .data = entry + offsetof(struct dora_binary_view_entry, payload),
                .byte_count = count
            };
        }
        struct dora_binary_view_reference reference;
        memcpy(&reference, entry + offsetof(struct dora_binary_view_entry, payload), sizeof(reference));
        return (struct dora_bytes){
            .data = (const uint8_t*)view->data_buffers[reference.buffer_index].data + reference.byte_offset,
            .byte_count = count
        };
    }
    default:
        return (struct dora_bytes){};
    }
}

/// List, FixedList, Map; layouts: List, LargeList, ListView, LargeListView, FixedList
static inline struct dora_range dora_vector_view_get_elements(const struct dora_vector_view* view, uint64_t row) {
    if (dora_vector_view_is_null(view, row)) {
        return (struct dora_range){};
    }
    const uint64_t slot = dora_vector_view_get_physical_row(view, row);
    switch (view->layout) {
    case DORA_LAYOUT_FIXED_LIST:
        return (struct dora_range){.offset = slot * view->list_element_count, .length = view->list_element_count};
    case DORA_LAYOUT_LIST: {
        int32_t begin;
        int32_t end;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&end, (const uint8_t*)view->offsets.data + (slot + 1) * sizeof(end), sizeof(end));
        return (struct dora_range){.offset = begin, .length = end - begin};
    }
    case DORA_LAYOUT_LARGE_LIST: {
        int64_t begin;
        int64_t end;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&end, (const uint8_t*)view->offsets.data + (slot + 1) * sizeof(end), sizeof(end));
        return (struct dora_range){.offset = begin, .length = end - begin};
    }
    case DORA_LAYOUT_LIST_VIEW: {
        int32_t begin;
        int32_t count;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&count, (const uint8_t*)view->sizes.data + slot * sizeof(count), sizeof(count));
        return (struct dora_range){.offset = begin, .length = count};
    }
    case DORA_LAYOUT_LARGE_LIST_VIEW: {
        int64_t begin;
        int64_t count;
        memcpy(&begin, (const uint8_t*)view->offsets.data + slot * sizeof(begin), sizeof(begin));
        memcpy(&count, (const uint8_t*)view->sizes.data + slot * sizeof(count), sizeof(count));
        return (struct dora_range){.offset = begin, .length = count};
    }
    default:
        return (struct dora_range){};
    }
}

/// Union; layouts: DenseUnion, SparseUnion
static inline int8_t dora_vector_view_get_type_id(const struct dora_vector_view* view, uint64_t row) {
    if (dora_vector_view_is_null(view, row)) {
        return -1;
    }
    const uint64_t slot = dora_vector_view_get_physical_row(view, row);
    int8_t type_id;
    memcpy(&type_id, (const uint8_t*)view->type_ids.data + slot, sizeof(type_id));
    return type_id;
}

/// Union; layouts: DenseUnion, SparseUnion
static inline uint32_t
dora_vector_view_get_member(const struct dora_vector_view* view, uint64_t row, uint64_t* out_member_row) {
    const int8_t type_id = dora_vector_view_get_type_id(view, row);
    if (type_id < 0) {
        return UINT32_MAX;
    }
    const uint64_t slot = dora_vector_view_get_physical_row(view, row);
    if (view->layout == DORA_LAYOUT_DENSE_UNION) {
        int32_t member_row;
        memcpy(&member_row, (const uint8_t*)view->offsets.data + slot * sizeof(member_row), sizeof(member_row));
        *out_member_row = (uint64_t)member_row;
    } else {
        *out_member_row = slot;
    }
    return view->union_members[type_id];
}

//------------------------------------------------------------------------------
//
// Validity
//
//------------------------------------------------------------------------------

static inline uint64_t dora_vector_view_get_validity_word(const struct dora_vector_view* view, uint64_t word_index) {
    if (word_index > UINT64_MAX / 64) {
        return 0;
    }
    const uint64_t first = word_index * 64;
    if (first >= view->length || view->layout == DORA_LAYOUT_NULL) {
        return 0;
    }
    const uint64_t remaining = view->length - first;
    const uint32_t count = remaining < 64 ? remaining : 64;
    if (view->validity.data == nullptr) {
        return count == 64 ? UINT64_MAX : (UINT64_C(1) << count) - 1;
    }
    if (view->selection.indices == nullptr) {
        return dora_bitmap_get_word(view->validity.data,
                                    view->validity_bit_offset + first,
                                    view->validity_bit_offset + view->length);
    }
    uint64_t word = 0;
    for (uint32_t bit = 0; bit < count; ++bit) {
        if (!dora_vector_view_is_null(view, first + bit)) {
            word |= UINT64_C(1) << bit;
        }
    }
    return word;
}

static inline bool dora_vector_view_next_valid(const struct dora_vector_view* view,
                                               struct dora_row_cursor* in_out_cursor,
                                               uint64_t* out_row) {
    while (in_out_cursor->word == 0) {
        if (view->length == 0 || in_out_cursor->word_index > (view->length - 1) / 64) {
            return false;
        }
        in_out_cursor->word = dora_vector_view_get_validity_word(view, in_out_cursor->word_index);
        ++in_out_cursor->word_index;
    }
    const uint32_t bit = dora_bitmap_word_pop_lowest_set(&in_out_cursor->word);
    *out_row = (in_out_cursor->word_index - 1) * 64 + bit;
    return true;
}

uint64_t dora_vector_view_count_nulls(const struct dora_vector_view* view, struct dora_range rows);

void dora_vector_view_copy_validity(const struct dora_vector_view* view, struct dora_range rows, bool* out_validity);

//------------------------------------------------------------------------------
//
// Runs
//
//------------------------------------------------------------------------------

/// RunEnd layout; run ends: Int16, Int32, Int64
static inline uint64_t dora_vector_view_find_run(const struct dora_vector_view* view, uint64_t row) {
    if (view->layout != DORA_LAYOUT_RUN_END || row >= view->length || view->values.data == nullptr) {
        return UINT64_MAX;
    }
    size_t width;
    switch (view->physical_type) {
    case DORA_PHYSICAL_TYPE_I16:
        width = sizeof(int16_t);
        break;
    case DORA_PHYSICAL_TYPE_I32:
        width = sizeof(int32_t);
        break;
    case DORA_PHYSICAL_TYPE_I64:
        width = sizeof(int64_t);
        break;
    default:
        return UINT64_MAX;
    }
    const uint64_t physical_row = dora_vector_view_get_physical_row(view, row);
    uint64_t begin = 0;
    uint64_t end = view->values.byte_count / width - view->run_end_element_offset;
    while (begin < end) {
        const uint64_t middle = begin + (end - begin) / 2;
        const uint64_t element = view->run_end_element_offset + middle;
        const uint8_t* address = (const uint8_t*)view->values.data + element * width;
        uint64_t run_end;
        if (width == sizeof(int16_t)) {
            int16_t value;
            memcpy(&value, address, sizeof(value));
            run_end = value;
        } else if (width == sizeof(int32_t)) {
            int32_t value;
            memcpy(&value, address, sizeof(value));
            run_end = value;
        } else {
            int64_t value;
            memcpy(&value, address, sizeof(value));
            run_end = value;
        }
        if (run_end <= physical_row) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

/// RunEnd layout; run ends: Int16, Int32, Int64
static inline struct dora_range dora_vector_view_next_run(const struct dora_vector_view* view,
                                                          struct dora_run_cursor* in_out_cursor,
                                                          uint64_t* out_value_row) {
    if (view->layout != DORA_LAYOUT_RUN_END || in_out_cursor->row >= view->length) {
        return (struct dora_range){};
    }
    const uint64_t first = in_out_cursor->row;
    const uint64_t run = dora_vector_view_find_run(view, first);
    if (run == UINT64_MAX) {
        return (struct dora_range){};
    }
    uint64_t end = first + 1;
    if (view->selection.indices == nullptr) {
        const uint64_t element = view->run_end_element_offset + run;
        uint64_t physical_end;
        switch (view->physical_type) {
        case DORA_PHYSICAL_TYPE_I16: {
            int16_t value;
            memcpy(&value, (const uint8_t*)view->values.data + element * sizeof(value), sizeof(value));
            physical_end = value;
            break;
        }
        case DORA_PHYSICAL_TYPE_I32: {
            int32_t value;
            memcpy(&value, (const uint8_t*)view->values.data + element * sizeof(value), sizeof(value));
            physical_end = value;
            break;
        }
        case DORA_PHYSICAL_TYPE_I64: {
            int64_t value;
            memcpy(&value, (const uint8_t*)view->values.data + element * sizeof(value), sizeof(value));
            physical_end = value;
            break;
        }
        default:
            return (struct dora_range){};
        }
        const uint64_t remaining = physical_end - view->offset;
        end = remaining < view->length ? remaining : view->length;
    } else {
        while (end < view->length && dora_vector_view_find_run(view, end) == run) {
            ++end;
        }
    }
    *out_value_row = run;
    *in_out_cursor = (struct dora_run_cursor){.run = run, .row = end};
    return (struct dora_range){.offset = first, .length = end - first};
}

#endif
