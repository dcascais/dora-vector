// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/binding/validate.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "numeric/decimal.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Rows and Validity
//
//------------------------------------------------------------------------------

struct validate_rows {
    uint64_t first;
    uint32_t count;
};

static inline uint64_t validate_valid(const struct kernel_validate_binding* binding, uint64_t row, uint32_t bits) {
    const struct dora_vector_view* const view = binding->view;
    if (view->validity.data == nullptr) {
        return bitmap_mask_low(bits);
    }
    if (view->selection.indices == nullptr) {
        return dora_bitmap_load_window(view->validity.data, view->validity_bit_offset + row, bits);
    }
    uint64_t word = 0;
    for (uint32_t lane = 0; lane < bits; ++lane) {
        bitmap_assign(&word, lane, !dora_vector_view_is_null(view, row + lane));
    }
    return word;
}

static inline bool validate_child_is_null(const struct dora_vector_view* child, uint64_t row) {
    return child->validity.data != nullptr && !dora_bitmap_test(child->validity.data, child->validity_bit_offset + row);
}

// Returns DORA_OK or STATUS from the enclosing function.
#define VALIDATE_LOCATE(MASKED, FAULT, STATUS)                                                                         \
    for (uint32_t done = 0; done < count; done += 64) {                                                                \
        const uint64_t row = first_row + done;                                                                         \
        const uint32_t bits = count - done < 64 ? count - done : 64;                                                   \
        const uint64_t valid = (MASKED) ? validate_valid(binding, row, bits) : bitmap_mask_low(bits);                  \
        for (uint32_t lane = 0; lane < bits; ++lane) {                                                                 \
            if (bitmap_word_bit(valid, lane) != 0 && FAULT(binding, row + lane)) {                                     \
                *out_first_bad_row = row + lane;                                                                       \
                return STATUS;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    return DORA_OK

#define DEFINE_VALIDATE_CHECK(NAME, MASKED, FAULT, STATUS)                                                             \
    static int NAME##_rows(struct kernel_validate_binding* binding,                                                    \
                           struct validate_rows rows,                                                                  \
                           uint64_t* out_first_bad_row) {                                                              \
        const uint64_t first_row = rows.first;                                                                         \
        const uint32_t count = rows.count;                                                                             \
        uint64_t error = 0;                                                                                            \
        if (!(MASKED) || binding->view->validity.data == nullptr) {                                                    \
            for (uint32_t lane = 0; lane < count; ++lane) {                                                            \
                error |= (uint64_t)FAULT(binding, first_row + lane);                                                   \
            }                                                                                                          \
        }                                                                                                              \
        for (uint32_t done = 0; (MASKED) && binding->view->validity.data != nullptr && error == 0 && done < count;     \
             done += 64) {                                                                                             \
            const uint64_t row = first_row + done;                                                                     \
            const uint32_t bits = count - done < 64 ? count - done : 64;                                               \
            uint64_t valid = validate_valid(binding, row, bits);                                                       \
            if (valid == bitmap_mask_low(bits)) {                                                                      \
                for (uint32_t lane = 0; lane < bits; ++lane) {                                                         \
                    error |= (uint64_t)FAULT(binding, row + lane);                                                     \
                }                                                                                                      \
            } else {                                                                                                   \
                while (error == 0 && valid != 0) {                                                                     \
                    error = FAULT(binding, row + dora_bitmap_word_pop_lowest_set(&valid));                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        if (error == 0) {                                                                                              \
            return DORA_OK;                                                                                            \
        }                                                                                                              \
        VALIDATE_LOCATE(MASKED, FAULT, STATUS);                                                                        \
    }                                                                                                                  \
    static int NAME(struct kernel_validate_binding* binding,                                                           \
                    uint64_t first_row,                                                                                \
                    uint32_t count,                                                                                    \
                    uint64_t* out_first_bad_row) {                                                                     \
        return NAME##_rows(binding, (struct validate_rows){.first = first_row, .count = count}, out_first_bad_row);    \
    }

//------------------------------------------------------------------------------
//
// Offsets
//
//------------------------------------------------------------------------------

#define DEFINE_OFFSETS_CHECKS(S, T)                                                                                    \
    static inline bool offsets_have_fault_##S(const struct kernel_validate_binding* binding, uint64_t entry) {         \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const T value = type_load_##S(view->offsets.data, view->offset + entry);                                       \
        const T previous = entry == 0 ? 0 : type_load_##S(view->offsets.data, view->offset + entry - 1);               \
        return value < 0 || (uint64_t)value > (uint64_t)binding->limit || value < previous;                            \
    }                                                                                                                  \
    static inline bool list_view_has_fault_##S(const struct kernel_validate_binding* binding, uint64_t row) {          \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const T offset = type_load_##S(view->offsets.data, view->offset + row);                                        \
        const T size = type_load_##S(view->sizes.data, view->offset + row);                                            \
        return offset < 0 || (uint64_t)offset > (uint64_t)binding->limit || size < 0 ||                                \
               (uint64_t)size > (uint64_t)binding->limit - (uint64_t)offset;                                           \
    }                                                                                                                  \
    static inline bool offsets_are_ordered_##S(const struct kernel_validate_binding* binding,                          \
                                               struct validate_rows rows) {                                            \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const uint64_t slot = view->offset + rows.first;                                                               \
        const T first_offset = type_load_##S(view->offsets.data, slot);                                                \
        T previous_offset = rows.first == 0 ? first_offset : type_load_##S(view->offsets.data, slot - 1);              \
        for (uint32_t window_index = 0; window_index < rows.count; window_index++) {                                   \
            const T offset = type_load_##S(view->offsets.data, slot + window_index);                                   \
            if (offset < previous_offset) {                                                                            \
                return false;                                                                                          \
            }                                                                                                          \
            previous_offset = offset;                                                                                  \
        }                                                                                                              \
        return first_offset >= 0 && (uint64_t)previous_offset <= (uint64_t)binding->limit;                             \
    }                                                                                                                  \
    static int offsets_check_##S(struct kernel_validate_binding* binding,                                              \
                                 uint64_t first_row,                                                                   \
                                 uint32_t count,                                                                       \
                                 uint64_t* out_first_bad_row) {                                                        \
        if (count == 0 ||                                                                                              \
            offsets_are_ordered_##S(binding, (struct validate_rows){.first = first_row, .count = count})) {            \
            return DORA_OK;                                                                                            \
        }                                                                                                              \
        VALIDATE_LOCATE(false, offsets_have_fault_##S, EINVAL);                                                        \
    }                                                                                                                  \
    DEFINE_VALIDATE_CHECK(list_view_check_##S, false, list_view_has_fault_##S, EINVAL)
DEFINE_OFFSETS_CHECKS(i32, int32_t)
DEFINE_OFFSETS_CHECKS(i64, int64_t)
#undef DEFINE_OFFSETS_CHECKS

//------------------------------------------------------------------------------
//
// Indices and Run Ends
//
//------------------------------------------------------------------------------

// Dictionary indices: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_INDEX_CHECKS(S, T, P, NEGATIVE)                                                                         \
    static inline bool dictionary_has_fault_##S(const struct kernel_validate_binding* binding, uint64_t row) {         \
        const T index =                                                                                                \
            type_load_##S(binding->view->values.data, dora_vector_view_get_physical_row(binding->view, row));          \
        return NEGATIVE(index) || (uint64_t)index >= (uint64_t)binding->limit;                                         \
    }                                                                                                                  \
    DEFINE_VALIDATE_CHECK(dictionary_check_##S, true, dictionary_has_fault_##S, EINVAL)
SIGNED_INDEX_CARRIERS(DEFINE_INDEX_CHECKS, NEGATIVE_SIGNED)
UNSIGNED_INDEX_CARRIERS(DEFINE_INDEX_CHECKS, NEGATIVE_UNSIGNED)
#undef DEFINE_INDEX_CHECKS

// Run ends: Int16, Int32, Int64
#define DEFINE_RUN_END_CHECK(S, T)                                                                                     \
    static inline bool run_end_has_fault_##S(const struct kernel_validate_binding* binding, uint64_t row) {            \
        const struct dora_vector_view* const runs = binding->related;                                                  \
        const T end = type_load_##S(runs->values.data, runs->offset + row);                                            \
        const T previous = row == 0 ? 0 : type_load_##S(runs->values.data, runs->offset + row - 1);                    \
        return validate_child_is_null(runs, row) || end <= previous;                                                   \
    }                                                                                                                  \
    static inline bool run_end_ascends_##S(const struct kernel_validate_binding* binding,                              \
                                           uint64_t first_row,                                                         \
                                           uint32_t count) {                                                           \
        const struct dora_vector_view* const runs = binding->related;                                                  \
        if (runs->validity.data != nullptr &&                                                                          \
            dora_bitmap_count_set_bytes(runs->validity.data, runs->validity_bit_offset + first_row, count) != count) { \
            return false;                                                                                              \
        }                                                                                                              \
        uint64_t ascent_error = 0;                                                                                     \
        T previous_end = first_row == 0 ? 0 : type_load_##S(runs->values.data, runs->offset + first_row - 1);          \
        for (uint32_t run_index = 0; run_index < count; ++run_index) {                                                 \
            const T run_end = type_load_##S(runs->values.data, runs->offset + first_row + run_index);                  \
            ascent_error |= (uint64_t)(run_end <= previous_end);                                                       \
            previous_end = run_end;                                                                                    \
        }                                                                                                              \
        return ascent_error == 0;                                                                                      \
    }                                                                                                                  \
    static int run_end_scan_##S(struct kernel_validate_binding* binding,                                               \
                                uint64_t first_row,                                                                    \
                                uint32_t count,                                                                        \
                                uint64_t* out_first_bad_row) {                                                         \
        if (run_end_ascends_##S(binding, first_row, count)) {                                                          \
            return DORA_OK;                                                                                            \
        }                                                                                                              \
        VALIDATE_LOCATE(false, run_end_has_fault_##S, EINVAL);                                                         \
    }                                                                                                                  \
    static int run_end_check_##S(struct kernel_validate_binding* binding,                                              \
                                 uint64_t first_row,                                                                   \
                                 uint32_t count,                                                                       \
                                 uint64_t* out_first_bad_row) {                                                        \
        const int status = run_end_scan_##S(binding, first_row, count, out_first_bad_row);                             \
        const struct dora_vector_view* const runs = binding->related;                                                  \
        if (status != DORA_OK || first_row + count != runs->length) {                                                  \
            return status;                                                                                             \
        }                                                                                                              \
        const uint64_t last =                                                                                          \
            runs->length == 0 ? 0 : (uint64_t)type_load_##S(runs->values.data, runs->offset + runs->length - 1);       \
        if (last < binding->view->offset + binding->view->length) {                                                    \
            *out_first_bad_row = runs->length == 0 ? 0 : runs->length - 1;                                             \
            return EINVAL;                                                                                             \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
DEFINE_RUN_END_CHECK(i16, int16_t)
DEFINE_RUN_END_CHECK(i32, int32_t)
DEFINE_RUN_END_CHECK(i64, int64_t)
#undef DEFINE_RUN_END_CHECK

//------------------------------------------------------------------------------
//
// Unions
//
//------------------------------------------------------------------------------

static inline bool union_sparse_has_fault(const struct kernel_validate_binding* binding, uint64_t row) {
    const struct dora_vector_view* const view = binding->view;
    const uint64_t slot = dora_vector_view_get_physical_row(view, row);
    const int8_t code = type_load_i8(view->type_ids.data, slot);
    const uint8_t member = code >= 0 ? view->union_members[(uint8_t)code] : UINT8_MAX;
    return member == UINT8_MAX || slot >= binding->related[member].length;
}

DEFINE_VALIDATE_CHECK(union_sparse_check, false, union_sparse_has_fault, EINVAL)

static void union_dense_fold(struct kernel_validate_binding* binding, uint64_t end_row) {
    if (end_row < binding->union_row) {
        memset(binding->union_offsets, 0, KERNEL_UNION_TYPE_ID_COUNT * sizeof(*binding->union_offsets));
        binding->union_row = 0;
    }
    const struct dora_vector_view* const view = binding->view;
    for (uint64_t row = binding->union_row; row < end_row; ++row) {
        const uint64_t slot = dora_vector_view_get_physical_row(view, row);
        const int8_t code = type_load_i8(view->type_ids.data, slot);
        const int32_t offset = type_load_i32(view->offsets.data, slot);
        if (code >= 0) {
            binding->union_offsets[(uint8_t)code] = offset;
        }
    }
    binding->union_row = end_row;
}

static int union_dense_check(struct kernel_validate_binding* binding,
                             uint64_t first_row,
                             uint32_t count,
                             uint64_t* out_first_bad_row) {
    union_dense_fold(binding, first_row);
    int32_t local_offsets[KERNEL_UNION_TYPE_ID_COUNT];
    (memcpy)(local_offsets, binding->union_offsets, sizeof(local_offsets));
    const struct dora_vector_view* const view = binding->view;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const uint64_t row = first_row + lane;
        const uint64_t slot = dora_vector_view_get_physical_row(view, row);
        const int8_t code = type_load_i8(view->type_ids.data, slot);
        const int32_t offset = type_load_i32(view->offsets.data, slot);
        const uint8_t id = (uint8_t)code;
        const uint8_t member = code >= 0 ? view->union_members[id] : UINT8_MAX;
        if (member == UINT8_MAX || offset < 0 || (uint64_t)offset >= binding->related[member].length ||
            offset < local_offsets[id]) {
            *out_first_bad_row = row;
            return EINVAL;
        }
        local_offsets[id] = offset;
    }
    (memcpy)(binding->union_offsets, local_offsets, sizeof(local_offsets));
    binding->union_row = first_row + count;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Binary View Entries
//
//------------------------------------------------------------------------------

static inline bool binary_view_has_fault(const struct kernel_validate_binding* binding, uint64_t row) {
    const struct dora_vector_view* const view = binding->view;
    struct dora_binary_view_entry entry = {};
    (memcpy)(&entry,
             (const uint8_t*)view->values.data + dora_vector_view_get_physical_row(view, row) * sizeof(entry),
             sizeof(entry));
    if (entry.byte_count <= VIEW_INLINE_BYTE_COUNT) {
        return entry.byte_count < 0;
    }
    const struct dora_binary_view_reference reference = entry.payload.reference;
    if (reference.buffer_index < 0 || (uint32_t)reference.buffer_index >= view->data_buffer_count ||
        reference.byte_offset < 0) {
        return true;
    }
    const size_t size = view->data_buffers[reference.buffer_index].byte_count;
    return (uint32_t)reference.byte_offset > size ||
           (uint32_t)entry.byte_count > size - (uint32_t)reference.byte_offset;
}

DEFINE_VALIDATE_CHECK(binary_view_check, true, binary_view_has_fault, EINVAL)

//------------------------------------------------------------------------------
//
// Content
//
//------------------------------------------------------------------------------

static constexpr int32_t VALIDATE_BINARY_VIEW_PREFIX_BYTES = 4;
static constexpr int64_t VALIDATE_DATE64_DAY_MILLISECONDS = 86'400'000;

static inline bool binary_view_content_has_fault(const struct kernel_validate_binding* binding, uint64_t row) {
    const struct dora_vector_view* const view = binding->view;
    struct dora_binary_view_entry entry = {};
    (memcpy)(&entry,
             (const uint8_t*)view->values.data + dora_vector_view_get_physical_row(view, row) * sizeof(entry),
             sizeof(entry));
    if (entry.byte_count < 0) {
        return true;
    }
    if (entry.byte_count <= VIEW_INLINE_BYTE_COUNT) {
        unsigned _BitInt(128) raw = 0;
        (memcpy)(&raw, &entry, sizeof(raw));
        return ((raw >> 32U) >> ((uint32_t)entry.byte_count * 8U)) != 0;
    }
    const struct dora_binary_view_reference reference = entry.payload.reference;
    if (reference.buffer_index < 0 || (uint32_t)reference.buffer_index >= view->data_buffer_count ||
        reference.byte_offset < 0) {
        return true;
    }
    const struct dora_view_buffer buffer = view->data_buffers[reference.buffer_index];
    if ((uint64_t)reference.byte_offset > buffer.byte_count ||
        (uint64_t)entry.byte_count > buffer.byte_count - (uint64_t)reference.byte_offset || buffer.data == nullptr) {
        return true;
    }
    return memcmp(reference.prefix,
                  (const uint8_t*)buffer.data + reference.byte_offset,
                  VALIDATE_BINARY_VIEW_PREFIX_BYTES) != 0;
}

DEFINE_VALIDATE_CHECK(binary_view_content_check, true, binary_view_content_has_fault, EILSEQ)

static inline bool date64_has_fault(const struct kernel_validate_binding* binding, uint64_t row) {
    const struct dora_vector_view* const view = binding->view;
    const int64_t value = type_load_i64(view->values.data, dora_vector_view_get_physical_row(view, row));
    return value % VALIDATE_DATE64_DAY_MILLISECONDS != 0;
}

DEFINE_VALIDATE_CHECK(date64_check, true, date64_has_fault, ERANGE)

// Time (s/ms/us/ns)
#define DEFINE_TIME_CHECK(S, T, U)                                                                                     \
    static inline bool time_has_fault_##S(const struct kernel_validate_binding* binding, uint64_t row) {               \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const T value = type_load_##S(view->values.data, dora_vector_view_get_physical_row(view, row));                \
        return (U)value >= (U)binding->limit;                                                                          \
    }                                                                                                                  \
    DEFINE_VALIDATE_CHECK(time_check_##S, true, time_has_fault_##S, ERANGE)
DEFINE_TIME_CHECK(i32, int32_t, uint32_t)
DEFINE_TIME_CHECK(i64, int64_t, uint64_t)
#undef DEFINE_TIME_CHECK

//------------------------------------------------------------------------------
//
// Decimals
//
//------------------------------------------------------------------------------

// Decimal (32/64/128/256-bit)
#define DEFINE_DECIMAL_FAULT(S, T)                                                                                     \
    static inline bool decimal_has_fault_##S(const struct kernel_validate_binding* binding, uint64_t row) {            \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const T coefficient = type_load_##S(view->values.data, dora_vector_view_get_physical_row(view, row));          \
        return !decimal_bound_admits_##S(coefficient, binding->decimal.bound);                                         \
    }                                                                                                                  \
    DEFINE_VALIDATE_CHECK(decimal_scan_##S, true, decimal_has_fault_##S, EOVERFLOW)

// Decimal (32/64/128/256-bit)
#define DEFINE_DECIMAL_CHECK(S)                                                                                        \
    static int decimal_check_##S(struct kernel_validate_binding* binding,                                              \
                                 uint64_t first_row,                                                                   \
                                 uint32_t count,                                                                       \
                                 uint64_t* out_first_bad_row) {                                                        \
        return binding->decimal.type.precision != 0 ? decimal_scan_##S(binding, first_row, count, out_first_bad_row)   \
                                                    : EINVAL;                                                          \
    }

DEFINE_DECIMAL_FAULT(i32, int32_t)
DEFINE_DECIMAL_FAULT(i64, int64_t)
DEFINE_DECIMAL_FAULT(i128, struct dora_i128)
DEFINE_DECIMAL_FAULT(i256, struct dora_i256)
DEFINE_DECIMAL_CHECK(i32)
DEFINE_DECIMAL_CHECK(i64)
DEFINE_DECIMAL_CHECK(i128)
DEFINE_DECIMAL_CHECK(i256)
#undef DEFINE_DECIMAL_CHECK
#undef DEFINE_DECIMAL_FAULT
#undef DEFINE_VALIDATE_CHECK
#undef VALIDATE_LOCATE

//------------------------------------------------------------------------------
//
// Binding
//
//------------------------------------------------------------------------------

static typeof(&offsets_check_i32) const dictionary_checks[DORA_PHYSICAL_TYPE_U64 + 1] = {
    [DORA_PHYSICAL_TYPE_I8] = dictionary_check_i8,
    [DORA_PHYSICAL_TYPE_I16] = dictionary_check_i16,
    [DORA_PHYSICAL_TYPE_I32] = dictionary_check_i32,
    [DORA_PHYSICAL_TYPE_I64] = dictionary_check_i64,
    [DORA_PHYSICAL_TYPE_U8] = dictionary_check_u8,
    [DORA_PHYSICAL_TYPE_U16] = dictionary_check_u16,
    [DORA_PHYSICAL_TYPE_U32] = dictionary_check_u32,
    [DORA_PHYSICAL_TYPE_U64] = dictionary_check_u64,
};

static typeof(&offsets_check_i32) const run_end_checks[DORA_PHYSICAL_TYPE_U64 + 1] = {
    [DORA_PHYSICAL_TYPE_I16] = run_end_check_i16,
    [DORA_PHYSICAL_TYPE_I32] = run_end_check_i32,
    [DORA_PHYSICAL_TYPE_I64] = run_end_check_i64,
};

static typeof(&offsets_check_i32) const decimal_checks[DORA_PHYSICAL_TYPE_I256 + 1] = {
    [DORA_PHYSICAL_TYPE_I32] = decimal_check_i32,
    [DORA_PHYSICAL_TYPE_I64] = decimal_check_i64,
    [DORA_PHYSICAL_TYPE_I128] = decimal_check_i128,
    [DORA_PHYSICAL_TYPE_I256] = decimal_check_i256,
};

static typeof(&offsets_check_i32) validate_select(const struct dora_vector_view* view,
                                                  const struct dora_vector_view* related) {
    const bool wide = view->layout == DORA_LAYOUT_LARGE_BINARY || view->layout == DORA_LAYOUT_LARGE_LIST ||
                      view->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    switch (view->layout) {
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_LIST:
    case DORA_LAYOUT_LARGE_LIST:
        return wide ? offsets_check_i64 : offsets_check_i32;
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        return wide ? list_view_check_i64 : list_view_check_i32;
    case DORA_LAYOUT_DICTIONARY:
        return view->physical_type <= DORA_PHYSICAL_TYPE_U64 ? dictionary_checks[view->physical_type] : nullptr;
    case DORA_LAYOUT_RUN_END:
        return related != nullptr && related->physical_type <= DORA_PHYSICAL_TYPE_U64
                   ? run_end_checks[related->physical_type]
                   : nullptr;
    case DORA_LAYOUT_DENSE_UNION:
        return union_dense_check;
    case DORA_LAYOUT_SPARSE_UNION:
        return union_sparse_check;
    case DORA_LAYOUT_BINARY_VIEW:
        return binary_view_check;
    case DORA_LAYOUT_FIXED:
        return view->type == DORA_TYPE_DECIMAL && view->physical_type <= DORA_PHYSICAL_TYPE_I256
                   ? decimal_checks[view->physical_type]
                   : nullptr;
    default:
        return nullptr;
    }
}

int dora_kernel_validate_content_bind(const struct dora_vector_view* view,
                                      const struct dora_logical_type* type,
                                      struct kernel_validate_binding* out_binding) {
    if (view == nullptr || type == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    if (!kernel_validate_has_content(view, type)) {
        return ENOTSUP;
    }
    static constexpr int64_t day_ticks[] = {86'400, 86'400'000, 86'400'000'000, 86'400'000'000'000};
    typeof(&offsets_check_i32) check = nullptr;
    int64_t limit = 0;
    if (view->layout == DORA_LAYOUT_BINARY_VIEW) {
        check = binary_view_content_check;
    } else if (view->layout == DORA_LAYOUT_FIXED && type->id == DORA_TYPE_DATE64 &&
               view->physical_type == DORA_PHYSICAL_TYPE_I64) {
        check = date64_check;
    } else if (view->layout == DORA_LAYOUT_FIXED && type->id == DORA_TYPE_TIME &&
               type->parameters.unit <= DORA_TIME_UNIT_NANOSECOND) {
        check = view->physical_type == DORA_PHYSICAL_TYPE_I32   ? time_check_i32
                : view->physical_type == DORA_PHYSICAL_TYPE_I64 ? time_check_i64
                                                                : nullptr;
        limit = day_ticks[type->parameters.unit];
    }
    if (check == nullptr) {
        return ENOTSUP;
    }
    *out_binding = (struct kernel_validate_binding){.check = check, .view = view, .limit = limit};
    return DORA_OK;
}

int dora_kernel_validate_bind(const struct dora_vector_view* view,
                              const struct dora_logical_type* type,
                              const struct dora_vector_view* related,
                              struct kernel_validate_binding* out_binding) {
    if (view == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    const bool needs_related = view->layout != DORA_LAYOUT_BINARY && view->layout != DORA_LAYOUT_LARGE_BINARY &&
                               view->layout != DORA_LAYOUT_BINARY_VIEW && view->layout != DORA_LAYOUT_FIXED;
    if (needs_related && related == nullptr) {
        return EINVAL;
    }
    typeof(&offsets_check_i32) const check = validate_select(view, related);
    if (check == nullptr) {
        return view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END ? EINVAL : ENOTSUP;
    }
    struct decimal_prepared decimal = {};
    if (view->layout == DORA_LAYOUT_FIXED) {
        if (type == nullptr || type->id != DORA_TYPE_DECIMAL) {
            return EINVAL;
        }
        const int status = dora_decimal_prepare(type->parameters.decimal, view->physical_type, &decimal);
        if (status != DORA_OK) {
            return status;
        }
    }
    uint64_t limit = 0;
    if (related != nullptr) {
        limit = related->length;
    } else if (view->data_buffer_count != 0) {
        limit = view->data_buffers[0].byte_count;
    }
    *out_binding = (struct kernel_validate_binding){
        .check = check,
        .view = view,
        .related = related,
        .decimal = decimal,
        .limit = limit > INT64_MAX ? INT64_MAX : (int64_t)limit,
    };
    return DORA_OK;
}
