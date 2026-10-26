// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "utf8/batch.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/types.h"
#include "types/types_internal.h"
#include "utf8/ascii.h"
#include "utf8/encoding.h"
#include "utf8/input.h"
#include "utf8/span.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static struct dora_bytes utf8_subspan(struct dora_bytes bytes, size_t begin, size_t size) {
    return (struct dora_bytes){size == 0 ? nullptr : bytes.data + begin, size};
}

static int utf8_mask_check(struct dora_utf8_mask mask, uint32_t count) {
    if (mask.bytes.data == nullptr && mask.bytes.byte_count == 0 && mask.bit_offset == 0) {
        return 0;
    }
    if (!bytes_are_well_formed(mask.bytes) || mask.bytes.data == nullptr) {
        return EINVAL;
    }
    uint64_t end;
    if (ckd_add(&end, mask.bit_offset, count)) {
        return EOVERFLOW;
    }
    return bitmap_bytes_for_bits(end) > mask.bytes.byte_count ? EINVAL : 0;
}

static bool utf8_mask_contains_row(struct dora_utf8_mask mask, uint32_t row) {
    return mask.bytes.data == nullptr || dora_bitmap_test(mask.bytes.data, mask.bit_offset + row);
}

#define DEFINE_UTF8_OFFSET_ACCESS(NAME, TYPE)                                                                          \
    static int64_t utf8_offset_##NAME(const struct dora_utf8_batch* batch, size_t row) {                               \
        TYPE value;                                                                                                    \
        memcpy(&value, batch->rows.offsets.data + row * sizeof value, sizeof value);                                   \
        return value;                                                                                                  \
    }                                                                                                                  \
    static struct dora_bytes utf8_value_##NAME(const struct dora_utf8_batch* batch, size_t row) {                      \
        const size_t begin = (size_t)utf8_offset_##NAME(batch, row);                                                   \
        return utf8_subspan(batch->rows.payload, begin, (size_t)utf8_offset_##NAME(batch, row + 1) - begin);           \
    } static int utf8_geometry_##NAME(const struct dora_utf8_batch* batch, size_t row) {                               \
        const int64_t begin = utf8_offset_##NAME(batch, row), end = utf8_offset_##NAME(batch, row + 1);                \
        return begin < 0 || end < begin || (uint64_t)end > batch->rows.payload.byte_count ? EINVAL : 0;                \
    }
DEFINE_UTF8_OFFSET_ACCESS(i32, int32_t)
DEFINE_UTF8_OFFSET_ACCESS(i64, int64_t)
#undef DEFINE_UTF8_OFFSET_ACCESS

static struct dora_bytes utf8_value_descriptors(const struct dora_utf8_batch* batch, size_t row) {
    return batch->descriptors[row];
}

static int utf8_geometry_descriptors(const struct dora_utf8_batch* batch, size_t row) {
    return batch->descriptors == nullptr || !bytes_are_well_formed(batch->descriptors[row]) ? EINVAL : 0;
}

// ROW is not expanded or evaluated.
#define utf8_value_constant(BATCH, ROW) ((BATCH)->constant)

static int utf8_geometry_constant(const struct dora_utf8_batch* batch, size_t) {
    return bytes_are_well_formed(batch->constant) ? 0 : EINVAL;
}

static int32_t utf8_view_field(const uint8_t* record, size_t field) {
    int32_t value;
    memcpy(&value, record + field, sizeof value);
    return value;
}

static struct dora_bytes utf8_value_view(const struct dora_utf8_batch* batch, size_t row) {
    const uint8_t* record = batch->views.records.data + row * 16;
    const size_t length = (size_t)utf8_view_field(record, 0);
    if (length <= 12) {
        return (struct dora_bytes){record + 4, length};
    }
    const size_t index = (size_t)utf8_view_field(record, 8), offset = (size_t)utf8_view_field(record, 12);
    return utf8_subspan(batch->views.buffers[index], offset, length);
}

static int utf8_geometry_view(const struct dora_utf8_batch* batch, size_t row) {
    const uint8_t* record = batch->views.records.data + row * 16;
    const int32_t length = utf8_view_field(record, 0);
    if (length < 0) {
        return EINVAL;
    }
    if (length <= 12) {
        return 0;
    }
    const int32_t index = utf8_view_field(record, 8), offset = utf8_view_field(record, 12);
    if (index < 0 || offset < 0 || (size_t)index >= batch->views.buffer_count || batch->views.buffers == nullptr) {
        return EINVAL;
    }
    const struct dora_bytes buffer = batch->views.buffers[index];
    size_t end;
    if (ckd_add(&end, (size_t)offset, (size_t)length)) {
        return EOVERFLOW;
    }
    return !bytes_are_well_formed(buffer) || end > buffer.byte_count ? EINVAL : 0;
}

static uint64_t utf8_index_direct(const struct dora_utf8_batch*, uint32_t row) {
    return row;
}

#define DEFINE_UTF8_INDEX(NAME, TYPE)                                                                                  \
    static uint64_t utf8_index_##NAME(const struct dora_utf8_batch* batch, uint32_t row) {                             \
        TYPE index;                                                                                                    \
        memcpy(&index, batch->row_indices.data + (size_t)row * sizeof index, sizeof index);                            \
        return index;                                                                                                  \
    }
DEFINE_UTF8_INDEX(u32, uint32_t)
DEFINE_UTF8_INDEX(u64, uint64_t)
#undef DEFINE_UTF8_INDEX

#define UTF8_DISPATCH_WIDTH(BATCH, OP, LAYOUT, ...)                                                                    \
    switch ((BATCH)->index_width_bytes) {                                                                              \
    case 0:                                                                                                            \
        return OP##_##LAYOUT##_direct(__VA_ARGS__);                                                                    \
    case 4:                                                                                                            \
        return OP##_##LAYOUT##_u32(__VA_ARGS__);                                                                       \
    case 8:                                                                                                            \
        return OP##_##LAYOUT##_u64(__VA_ARGS__);                                                                       \
    default:                                                                                                           \
        return EINVAL;                                                                                                 \
    }
#define UTF8_DISPATCH(BATCH, OP, ...)                                                                                  \
    switch ((BATCH)->layout) {                                                                                         \
    case DORA_UTF8_OFFSETS_I32:                                                                                        \
        UTF8_DISPATCH_WIDTH(BATCH, OP, i32, __VA_ARGS__)                                                               \
    case DORA_UTF8_OFFSETS_I64:                                                                                        \
        UTF8_DISPATCH_WIDTH(BATCH, OP, i64, __VA_ARGS__)                                                               \
    case DORA_UTF8_BINARY_VIEW:                                                                                        \
        UTF8_DISPATCH_WIDTH(BATCH, OP, view, __VA_ARGS__)                                                              \
    case DORA_UTF8_DESCRIPTORS:                                                                                        \
        UTF8_DISPATCH_WIDTH(BATCH, OP, descriptors, __VA_ARGS__)                                                       \
    case DORA_UTF8_CONSTANT:                                                                                           \
        UTF8_DISPATCH_WIDTH(BATCH, OP, constant, __VA_ARGS__)                                                          \
    default:                                                                                                           \
        return EINVAL;                                                                                                 \
    }
#define UTF8_INSTANTIATE_LAYOUT(MACRO, LAYOUT) MACRO(LAYOUT, direct) MACRO(LAYOUT, u32) MACRO(LAYOUT, u64)
#define UTF8_INSTANTIATE(MACRO)                                                                                        \
    UTF8_INSTANTIATE_LAYOUT(MACRO, i32)                                                                                \
    UTF8_INSTANTIATE_LAYOUT(MACRO, i64)                                                                                \
    UTF8_INSTANTIATE_LAYOUT(MACRO, view)                                                                               \
    UTF8_INSTANTIATE_LAYOUT(MACRO, descriptors)                                                                        \
    UTF8_INSTANTIATE_LAYOUT(MACRO, constant)

#define DEFINE_UTF8_OFFSET_RANGE(NAME)                                                                                 \
                                                                                                                       \
    static int utf8_geometry_range_##NAME(const struct dora_utf8_batch* batch) {                                       \
        bool ordered = true;                                                                                           \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            ordered = utf8_offset_##NAME(batch, row) <= utf8_offset_##NAME(batch, (size_t)row + 1) && ordered;         \
        }                                                                                                              \
        const int64_t first = utf8_offset_##NAME(batch, 0);                                                            \
        const int64_t last = utf8_offset_##NAME(batch, batch->row_count);                                              \
        return !ordered || first < 0 || (uint64_t)last > batch->rows.payload.byte_count ? EINVAL : 0;                  \
    }
DEFINE_UTF8_OFFSET_RANGE(i32)
DEFINE_UTF8_OFFSET_RANGE(i64)
#undef DEFINE_UTF8_OFFSET_RANGE

static int utf8_batch_shape(const struct dora_utf8_batch* batch) {
    if (batch == nullptr || batch->proof > DORA_UTF8_VALIDATED_ASCII || batch->layout > DORA_UTF8_CONSTANT) {
        return EINVAL;
    }
    int status = utf8_mask_check(batch->eligible_rows, batch->row_count);
    if (status != 0) {
        return status;
    }
    if (batch->index_width_bytes != 0 && batch->index_width_bytes != 4 && batch->index_width_bytes != 8) {
        return EINVAL;
    }
    size_t extent;
    if (ckd_mul(&extent, (size_t)batch->row_count, batch->index_width_bytes)) {
        return EOVERFLOW;
    }
    if (!bytes_are_well_formed(batch->row_indices) || extent > batch->row_indices.byte_count ||
        (batch->index_width_bytes == 0 && batch->row_indices.byte_count != 0)) {
        return EINVAL;
    }
    if (batch->layout == DORA_UTF8_CONSTANT) {
        return batch->physical_value_count == 1 ? 0 : EINVAL;
    }
    if (batch->index_width_bytes == 0 && batch->physical_value_count != batch->row_count) {
        return EINVAL;
    }
    if (batch->layout == DORA_UTF8_DESCRIPTORS) {
        return ckd_mul(&extent, batch->physical_value_count, sizeof(struct dora_bytes)) ? EOVERFLOW : 0;
    }
    if (batch->layout == DORA_UTF8_BINARY_VIEW) {
        if (ckd_mul(&extent, batch->physical_value_count, 16)) {
            return EOVERFLOW;
        }
        if (!bytes_are_well_formed(batch->views.records) || extent > batch->views.records.byte_count) {
            return EINVAL;
        }
        return ckd_mul(&extent, batch->views.buffer_count, sizeof(struct dora_bytes)) ? EOVERFLOW : 0;
    }
    size_t entries;
    if (ckd_add(&entries, batch->physical_value_count, 1) ||
        ckd_mul(&extent, entries, batch->layout == DORA_UTF8_OFFSETS_I32 ? 4 : 8)) {
        return EOVERFLOW;
    }
    if (!bytes_are_well_formed(batch->rows.offsets) || extent > batch->rows.offsets.byte_count ||
        !bytes_are_well_formed(batch->rows.payload)) {
        return EINVAL;
    }
    if (batch->index_width_bytes == 0) {
        return batch->layout == DORA_UTF8_OFFSETS_I32 ? utf8_geometry_range_i32(batch) : utf8_geometry_range_i64(batch);
    }
    return 0;
}

#define DEFINE_UTF8_PREFLIGHT(LAYOUT, WIDTH)                                                                           \
                                                                                                                       \
    static int utf8_preflight_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                  \
                                                 struct dora_utf8_mask requested,                                      \
                                                 uint64_t* referenced,                                                 \
                                                 uint32_t* out_eligible) {                                             \
        uint32_t count = 0;                                                                                            \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (!utf8_mask_contains_row(requested, row) || !utf8_mask_contains_row(batch->eligible_rows, row)) {       \
                continue;                                                                                              \
            }                                                                                                          \
            if (referenced != nullptr && count == 0) {                                                                 \
                bitmap_fill_bytes((uint8_t*)referenced,                                                                \
                                  (size_t)bitmap_words_for_bits(batch->physical_value_count) * sizeof *referenced,     \
                                  false);                                                                              \
            }                                                                                                          \
            const uint64_t index = utf8_index_##WIDTH(batch, row);                                                     \
            if (batch->index_width_bytes != 0 && index >= batch->physical_value_count) {                               \
                return EINVAL;                                                                                         \
            }                                                                                                          \
            if (referenced == nullptr || !dora_bitmap_test((const uint8_t*)referenced, index)) {                       \
                const int status = utf8_geometry_##LAYOUT(batch, index);                                               \
                if (status != 0) {                                                                                     \
                    return status;                                                                                     \
                }                                                                                                      \
                if (referenced != nullptr) {                                                                           \
                    bitmap_set(referenced, index);                                                                     \
                }                                                                                                      \
            }                                                                                                          \
            ++count;                                                                                                   \
        }                                                                                                              \
        if (out_eligible != nullptr) {                                                                                 \
            *out_eligible = count;                                                                                     \
        }                                                                                                              \
        return 0;                                                                                                      \
    }
UTF8_INSTANTIATE(DEFINE_UTF8_PREFLIGHT)
#undef DEFINE_UTF8_PREFLIGHT

static int utf8_preflight(const struct dora_utf8_batch* batch,
                          struct dora_utf8_mask requested,
                          uint64_t* referenced,
                          uint32_t* out_eligible) {
    int status = utf8_batch_shape(batch);
    if (status == 0) {
        status = utf8_mask_check(requested, batch->row_count);
    }
    if (status != 0) {
        return status;
    }
    if (batch->index_width_bytes == 0 &&
        (batch->layout == DORA_UTF8_OFFSETS_I32 || batch->layout == DORA_UTF8_OFFSETS_I64)) {
        if (out_eligible == nullptr) {
            return 0;
        }
        uint32_t count = 0;
        for (uint32_t row = 0; row < batch->row_count; ++row) {
            count += utf8_mask_contains_row(requested, row) && utf8_mask_contains_row(batch->eligible_rows, row);
        }
        *out_eligible = count;
        return 0;
    }
    UTF8_DISPATCH(batch, utf8_preflight, batch, requested, referenced, out_eligible)
}

static void utf8_report_bad(uint32_t row, uint32_t* in_out_count, uint64_t* invalid, uint32_t* in_out_first) {
    if (invalid != nullptr) {
        bitmap_set(invalid, row);
    }
    ++*in_out_count;
    if (in_out_first != nullptr && row < *in_out_first) {
        *in_out_first = row;
    }
}

#define DEFINE_UTF8_CLASSIFY(LAYOUT, WIDTH)                                                                            \
    static void utf8_classify_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                  \
                                                 struct dora_utf8_mask requested,                                      \
                                                 uint64_t* invalid,                                                    \
                                                 uint32_t* out_count,                                                  \
                                                 bool* out_ascii,                                                      \
                                                 uint32_t* out_first) {                                                \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if ((!utf8_mask_contains_row(requested, row) || !utf8_mask_contains_row(batch->eligible_rows, row))) {     \
                continue;                                                                                              \
            }                                                                                                          \
            const struct dora_bytes value = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row));                \
            bool ascii = true;                                                                                         \
            if (dora_utf8_validate_span(value, out_ascii != nullptr ? &ascii : nullptr) != 0) {                        \
                utf8_report_bad(row, out_count, invalid, out_first);                                                   \
                if (invalid == nullptr) {                                                                              \
                    return;                                                                                            \
                }                                                                                                      \
            } else if (out_ascii != nullptr) {                                                                         \
                *out_ascii &= ascii;                                                                                   \
            }                                                                                                          \
        }                                                                                                              \
    }
UTF8_INSTANTIATE_LAYOUT(DEFINE_UTF8_CLASSIFY, i32)
UTF8_INSTANTIATE_LAYOUT(DEFINE_UTF8_CLASSIFY, i64)
UTF8_INSTANTIATE_LAYOUT(DEFINE_UTF8_CLASSIFY, view)
UTF8_INSTANTIATE_LAYOUT(DEFINE_UTF8_CLASSIFY, descriptors)
#undef DEFINE_UTF8_CLASSIFY

// Returns bytes.byte_count for valid spans.
static size_t utf8_bad_position(struct dora_bytes bytes, bool* out_ascii_prefix) {
    size_t position = utf8_ascii_prefix(bytes);
    *out_ascii_prefix = true;
    while (position < bytes.byte_count) {
        uint32_t scalar;
        size_t size;
        if (utf8_read_scalar(utf8_subspan(bytes, position, bytes.byte_count - position), &scalar, &size) != 0) {
            return position;
        }
        *out_ascii_prefix = false;
        position += size;
        position += utf8_ascii_prefix(utf8_subspan(bytes, position, bytes.byte_count - position));
    }
    return position;
}

static size_t utf8_span_first_bad(struct dora_bytes bytes, bool* out_ascii_prefix) {
    *out_ascii_prefix = true;
    if (bytes.byte_count == 0) {
        return 0;
    }
    const size_t prefix = utf8_ascii_prefix(bytes);
    if (prefix == bytes.byte_count) {
        return bytes.byte_count;
    }
    const struct dora_bytes suffix = utf8_subspan(bytes, prefix, bytes.byte_count - prefix);
    if (dora_utf8_validate_span(suffix, nullptr) == 0) {
        *out_ascii_prefix = false;
        return bytes.byte_count;
    }
    return prefix + utf8_bad_position(suffix, out_ascii_prefix);
}

// Requires a selected malformed row.
#define DEFINE_UTF8_FIRST_BAD(LAYOUT)                                                                                  \
    static uint32_t utf8_first_bad_##LAYOUT(const struct dora_utf8_batch* batch,                                       \
                                            struct dora_range run,                                                     \
                                            bool* out_ascii) {                                                         \
        const uint32_t first = (uint32_t)run.offset, last = first + (uint32_t)run.length;                              \
        const size_t begin = (size_t)utf8_offset_##LAYOUT(batch, first);                                               \
        const size_t end = (size_t)utf8_offset_##LAYOUT(batch, last);                                                  \
        const struct dora_bytes span = utf8_subspan(batch->rows.payload, begin, end - begin);                          \
        bool ascii = true;                                                                                             \
        const size_t failure = begin + utf8_span_first_bad(span, &ascii);                                              \
        uint32_t candidate = last;                                                                                     \
        if (failure < end) {                                                                                           \
            uint32_t low = first, high = first;                                                                        \
            if ((size_t)utf8_offset_##LAYOUT(batch, (size_t)first + 1) <= failure) {                                   \
                high = last;                                                                                           \
                const size_t average = (end - begin) == run.length ? 1 : (end - begin) / run.length;                   \
                size_t estimate = average <= 1 ? (average == 0 ? 0 : failure - begin) : (failure - begin) / average;   \
                if (estimate >= run.length) {                                                                          \
                    estimate = run.length - 1;                                                                         \
                }                                                                                                      \
                low = first + (uint32_t)estimate;                                                                      \
                if ((size_t)utf8_offset_##LAYOUT(batch, low) > failure) {                                              \
                    high = low;                                                                                        \
                    low = first;                                                                                       \
                } else if ((size_t)utf8_offset_##LAYOUT(batch, (size_t)low + 1) <= failure) {                          \
                    ++low;                                                                                             \
                } else {                                                                                               \
                    high = low;                                                                                        \
                }                                                                                                      \
            }                                                                                                          \
            while (low < high) {                                                                                       \
                const uint32_t middle = low + (high - low) / 2;                                                        \
                if ((size_t)utf8_offset_##LAYOUT(batch, (size_t)middle + 1) <= failure) {                              \
                    low = middle + 1;                                                                                  \
                } else {                                                                                               \
                    high = middle;                                                                                     \
                }                                                                                                      \
            }                                                                                                          \
            candidate = low;                                                                                           \
        }                                                                                                              \
        if (!ascii) {                                                                                                  \
            uint32_t previous = first;                                                                                 \
            for (uint32_t row = first; row < last && row <= candidate; ++row) {                                        \
                const size_t start = (size_t)utf8_offset_##LAYOUT(batch, row);                                         \
                const size_t limit = (size_t)utf8_offset_##LAYOUT(batch, (size_t)row + 1);                             \
                if (start == limit) {                                                                                  \
                    continue;                                                                                          \
                }                                                                                                      \
                if (row != first && start < failure && utf8_byte_is_continuation(batch->rows.payload.data[start])) {   \
                    candidate = previous;                                                                              \
                    break;                                                                                             \
                }                                                                                                      \
                previous = row;                                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        if (candidate < last) {                                                                                        \
            return candidate;                                                                                          \
        }                                                                                                              \
        *out_ascii &= ascii;                                                                                           \
        return last;                                                                                                   \
    }
DEFINE_UTF8_FIRST_BAD(i32)
DEFINE_UTF8_FIRST_BAD(i64)
#undef DEFINE_UTF8_FIRST_BAD

#define DEFINE_UTF8_RUN_CLASSIFY(LAYOUT)                                                                               \
    static void utf8_classify_runs_##LAYOUT(const struct dora_utf8_batch* batch,                                       \
                                            struct dora_utf8_mask requested,                                           \
                                            uint64_t* invalid,                                                         \
                                            uint32_t* out_count,                                                       \
                                            bool* out_ascii,                                                           \
                                            uint32_t* out_first) {                                                     \
        uint32_t row = 0;                                                                                              \
        while (row < batch->row_count) {                                                                               \
            if ((!utf8_mask_contains_row(requested, row) || !utf8_mask_contains_row(batch->eligible_rows, row))) {     \
                ++row;                                                                                                 \
                continue;                                                                                              \
            }                                                                                                          \
            const size_t begin = (size_t)utf8_offset_##LAYOUT(batch, row);                                             \
            uint32_t last = requested.bytes.data == nullptr && batch->eligible_rows.bytes.data == nullptr              \
                                ? batch->row_count                                                                     \
                                : row + 1;                                                                             \
            size_t end = (size_t)utf8_offset_##LAYOUT(batch, last);                                                    \
            while (last < batch->row_count) {                                                                          \
                const int64_t next_begin = utf8_offset_##LAYOUT(batch, last),                                          \
                              next_end = utf8_offset_##LAYOUT(batch, (size_t)last + 1);                                \
                if (next_begin < 0 || (uint64_t)next_begin != end ||                                                   \
                    ((!utf8_mask_contains_row(requested, last) ||                                                      \
                      !utf8_mask_contains_row(batch->eligible_rows, last)) &&                                          \
                     next_end != next_begin)) {                                                                        \
                    break;                                                                                             \
                }                                                                                                      \
                end = (size_t)next_end;                                                                                \
                ++last;                                                                                                \
            }                                                                                                          \
            bool ascii = true;                                                                                         \
            if (invalid == nullptr && out_first != nullptr) {                                                          \
                const uint32_t bad = utf8_first_bad_##LAYOUT(batch, (struct dora_range){row, last - row}, &ascii);     \
                if (bad < last) {                                                                                      \
                    utf8_report_bad(bad, out_count, invalid, out_first);                                               \
                    return;                                                                                            \
                }                                                                                                      \
                if (out_ascii != nullptr) {                                                                            \
                    *out_ascii &= ascii;                                                                               \
                }                                                                                                      \
                row = last;                                                                                            \
                continue;                                                                                              \
            }                                                                                                          \
            bool valid = dora_utf8_validate_span(utf8_subspan(batch->rows.payload, begin, end - begin),                \
                                                 out_ascii != nullptr ? &ascii : nullptr) == 0;                        \
            if (valid) {                                                                                               \
                for (uint32_t i = row; i < last; ++i) {                                                                \
                    if ((!utf8_mask_contains_row(requested, i) || !utf8_mask_contains_row(batch->eligible_rows, i))) { \
                        continue;                                                                                      \
                    }                                                                                                  \
                    const size_t first = (size_t)utf8_offset_##LAYOUT(batch, i),                                       \
                                 limit = (size_t)utf8_offset_##LAYOUT(batch, (size_t)i + 1);                           \
                    if (first < limit && utf8_byte_is_continuation(batch->rows.payload.data[first])) {                 \
                        valid = false;                                                                                 \
                        break;                                                                                         \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            if (valid) {                                                                                               \
                if (out_ascii != nullptr) {                                                                            \
                    *out_ascii &= ascii;                                                                               \
                }                                                                                                      \
            } else {                                                                                                   \
                if (invalid == nullptr && out_first == nullptr) {                                                      \
                    ++*out_count;                                                                                      \
                    return;                                                                                            \
                }                                                                                                      \
                for (uint32_t i = row; i < last; ++i) {                                                                \
                    if ((!utf8_mask_contains_row(requested, i) || !utf8_mask_contains_row(batch->eligible_rows, i))) { \
                        continue;                                                                                      \
                    }                                                                                                  \
                    if (dora_utf8_validate_span(utf8_value_##LAYOUT(batch, i),                                         \
                                                out_ascii != nullptr ? &ascii : nullptr) != 0) {                       \
                        utf8_report_bad(i, out_count, invalid, out_first);                                             \
                        if (invalid == nullptr) {                                                                      \
                            return;                                                                                    \
                        }                                                                                              \
                    } else if (out_ascii != nullptr) {                                                                 \
                        *out_ascii &= ascii;                                                                           \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            row = last;                                                                                                \
        }                                                                                                              \
    }
DEFINE_UTF8_RUN_CLASSIFY(i32)
DEFINE_UTF8_RUN_CLASSIFY(i64)
#undef DEFINE_UTF8_RUN_CLASSIFY

#define UTF8_CLASSIFY_WIDTH(LAYOUT)                                                                                    \
    switch (batch->index_width_bytes) {                                                                                \
    case 0:                                                                                                            \
        utf8_classify_##LAYOUT##_direct(batch, requested, invalid, out_count, out_ascii, out_first);                   \
        return;                                                                                                        \
    case 4:                                                                                                            \
        utf8_classify_##LAYOUT##_u32(batch, requested, invalid, out_count, out_ascii, out_first);                      \
        return;                                                                                                        \
    case 8:                                                                                                            \
        utf8_classify_##LAYOUT##_u64(batch, requested, invalid, out_count, out_ascii, out_first);                      \
        return;                                                                                                        \
    default:                                                                                                           \
        return;                                                                                                        \
    }

static void utf8_classify_values(const struct dora_utf8_batch* batch,
                                 struct dora_utf8_mask requested,
                                 uint64_t* invalid,
                                 uint32_t* out_count,
                                 bool* out_ascii,
                                 uint32_t* out_first) {
    if (batch->layout == DORA_UTF8_CONSTANT) {
        bool seen = false, valid = false, ascii = true;
        for (uint32_t row = 0; row < batch->row_count; ++row) {
            if ((!utf8_mask_contains_row(requested, row) || !utf8_mask_contains_row(batch->eligible_rows, row))) {
                continue;
            }
            if (!seen) {
                valid = dora_utf8_validate_span(batch->constant, out_ascii != nullptr ? &ascii : nullptr) == 0;
                seen = true;
            }
            if (!valid) {
                utf8_report_bad(row, out_count, invalid, out_first);
                if (invalid == nullptr) {
                    return;
                }
            } else if (out_ascii != nullptr) {
                *out_ascii &= ascii;
            }
        }
        return;
    }

    switch (batch->layout) {
    case DORA_UTF8_OFFSETS_I32:
        UTF8_CLASSIFY_WIDTH(i32)
        break;
    case DORA_UTF8_OFFSETS_I64:
        UTF8_CLASSIFY_WIDTH(i64)
        break;
    case DORA_UTF8_BINARY_VIEW:
        UTF8_CLASSIFY_WIDTH(view)
        break;
    case DORA_UTF8_DESCRIPTORS:
        UTF8_CLASSIFY_WIDTH(descriptors)
        break;
    case DORA_UTF8_CONSTANT:
        break;
    }
}

#define DEFINE_UTF8_RUN_VALID(LAYOUT)                                                                                  \
    static bool utf8_run_is_valid_##LAYOUT(const struct dora_utf8_batch* batch, bool* out_ascii) {                     \
        const size_t begin = (size_t)utf8_offset_##LAYOUT(batch, 0);                                                   \
        const size_t end = (size_t)utf8_offset_##LAYOUT(batch, batch->row_count);                                      \
        bool ascii;                                                                                                    \
        if (dora_utf8_validate_span(utf8_subspan(batch->rows.payload, begin, end - begin), &ascii) != 0) {             \
            return false;                                                                                              \
        }                                                                                                              \
        if (out_ascii != nullptr) {                                                                                    \
            *out_ascii = ascii;                                                                                        \
        }                                                                                                              \
        if (ascii) {                                                                                                   \
            return true;                                                                                               \
        }                                                                                                              \
        for (uint32_t row = 1; row < batch->row_count; ++row) {                                                        \
            const size_t boundary = (size_t)utf8_offset_##LAYOUT(batch, row);                                          \
            if (boundary < end && utf8_byte_is_continuation(batch->rows.payload.data[boundary])) {                     \
                return false;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        return true;                                                                                                   \
    }
DEFINE_UTF8_RUN_VALID(i32)
DEFINE_UTF8_RUN_VALID(i64)
#undef DEFINE_UTF8_RUN_VALID

static void utf8_classify_content(const struct dora_utf8_batch* batch,
                                  struct dora_utf8_mask requested,
                                  uint64_t* invalid,
                                  uint32_t* out_count,
                                  bool* out_ascii,
                                  uint32_t* out_first) {
    if (batch->index_width_bytes == 0 && invalid == nullptr && out_first == nullptr &&
        requested.bytes.data == nullptr && batch->eligible_rows.bytes.data == nullptr) {
        if (batch->layout == DORA_UTF8_OFFSETS_I32) {
            *out_count = utf8_run_is_valid_i32(batch, out_ascii) ? 0 : 1;
            return;
        }
        if (batch->layout == DORA_UTF8_OFFSETS_I64) {
            *out_count = utf8_run_is_valid_i64(batch, out_ascii) ? 0 : 1;
            return;
        }
    }
    if (batch->index_width_bytes == 0) {
        if (batch->layout == DORA_UTF8_OFFSETS_I32) {
            utf8_classify_runs_i32(batch, requested, invalid, out_count, out_ascii, out_first);
            return;
        }
        if (batch->layout == DORA_UTF8_OFFSETS_I64) {
            utf8_classify_runs_i64(batch, requested, invalid, out_count, out_ascii, out_first);
            return;
        }
    }
    utf8_classify_values(batch, requested, invalid, out_count, out_ascii, out_first);
}

#undef UTF8_CLASSIFY_WIDTH

#define DEFINE_UTF8_SCATTER(WIDTH)                                                                                     \
    static void utf8_scatter_##WIDTH(const struct dora_utf8_batch* batch,                                              \
                                     const uint64_t* physical,                                                         \
                                     uint64_t* invalid,                                                                \
                                     uint32_t* out_count,                                                              \
                                     uint32_t* out_first) {                                                            \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (utf8_mask_contains_row(batch->eligible_rows, row) &&                                                   \
                dora_bitmap_test((const uint8_t*)physical, utf8_index_##WIDTH(batch, row))) {                          \
                utf8_report_bad(row, out_count, invalid, out_first);                                                   \
                if (invalid == nullptr) {                                                                              \
                    return;                                                                                            \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }
DEFINE_UTF8_SCATTER(u32)
DEFINE_UTF8_SCATTER(u64)
#undef DEFINE_UTF8_SCATTER

static int utf8_batch_report(const struct dora_utf8_batch* batch,
                             uint64_t* invalid,
                             uint32_t* out_count,
                             bool* out_ascii,
                             uint32_t* out_first) {
    uint64_t referenced[16], physical_bad[16];
    const bool bounded = batch != nullptr && batch->index_width_bytes != 0 && batch->physical_value_count <= 1024 &&
                         batch->layout != DORA_UTF8_CONSTANT;
    uint32_t eligible = 0;
    const int status = utf8_preflight(batch, (struct dora_utf8_mask){}, bounded ? referenced : nullptr, &eligible);
    if (status != 0) {
        return status;
    }
    uint32_t distinct = 0;
    if (bounded && eligible != 0) {
        for (size_t i = 0; i < bitmap_words_for_bits(batch->physical_value_count); ++i) {
            distinct += dora_bitmap_word_count_ones(referenced[i]);
        }
    }
    if (invalid != nullptr) {
        bitmap_fill_bytes((uint8_t*)invalid, (size_t)bitmap_words_for_bits(batch->row_count) * sizeof *invalid, false);
    }
    uint32_t count = 0, first = batch->row_count;
    bool ascii = true;
    if (bounded && eligible != 0 && distinct <= eligible / 2) {
        bitmap_fill_bytes((uint8_t*)physical_bad,
                          (size_t)bitmap_words_for_bits(batch->physical_value_count) * sizeof *physical_bad,
                          false);
        struct dora_utf8_batch physical = *batch;
        physical.row_count = (uint32_t)batch->physical_value_count;
        physical.index_width_bytes = 0;
        physical.row_indices = (struct dora_bytes){};
        physical.eligible_rows = (struct dora_utf8_mask){
            {(const uint8_t*)referenced, (size_t)bitmap_bytes_for_bits(batch->physical_value_count)},
            0
        };
        uint32_t physical_count = 0, physical_first = physical.row_count;
        utf8_classify_values(&physical,
                             (struct dora_utf8_mask){},
                             physical_bad,
                             &physical_count,
                             out_ascii != nullptr ? &ascii : nullptr,
                             &physical_first);
        if (batch->index_width_bytes == 4) {
            utf8_scatter_u32(batch, physical_bad, invalid, &count, &first);
        } else {
            utf8_scatter_u64(batch, physical_bad, invalid, &count, &first);
        }
    } else {
        utf8_classify_content(batch,
                              (struct dora_utf8_mask){},
                              invalid,
                              &count,
                              out_ascii != nullptr ? &ascii : nullptr,
                              out_first != nullptr ? &first : nullptr);
    }
    *out_count = count;
    if (out_first != nullptr) {
        *out_first = first;
    }
    if (out_ascii != nullptr) {
        *out_ascii = ascii;
    }
    return count == 0 ? 0 : EILSEQ;
}

int dora_utf8_validate_batch(const struct dora_utf8_batch* batch, bool* out_all_ascii, uint32_t* out_first_bad_row) {
    if (batch != nullptr && batch->index_width_bytes == 0 && batch->eligible_rows.bytes.data == nullptr &&
        batch->rows.payload.data != nullptr &&
        (batch->layout == DORA_UTF8_OFFSETS_I32 || batch->layout == DORA_UTF8_OFFSETS_I64)) {
        const int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, nullptr);
        if (status != 0) {
            return status;
        }
        if (out_first_bad_row != nullptr) {
            bool ascii = true;
            uint32_t bad = batch->row_count;
            if (batch->row_count != 0) {
                bad = batch->layout == DORA_UTF8_OFFSETS_I32
                          ? utf8_first_bad_i32(batch, (struct dora_range){0, batch->row_count}, &ascii)
                          : utf8_first_bad_i64(batch, (struct dora_range){0, batch->row_count}, &ascii);
            }
            *out_first_bad_row = bad;
            if (out_all_ascii != nullptr) {
                *out_all_ascii = bad == batch->row_count && ascii;
            }
            return bad == batch->row_count ? 0 : EILSEQ;
        }
        bool ascii = true;
        bool valid = true;
        if (batch->row_count != 0) {
            if (batch->layout == DORA_UTF8_OFFSETS_I32) {
                valid = utf8_run_is_valid_i32(batch, &ascii);
            } else {
                valid = utf8_run_is_valid_i64(batch, &ascii);
            }
        }
        if (out_all_ascii != nullptr) {
            *out_all_ascii = valid && ascii;
        }
        return valid ? 0 : EILSEQ;
    }
    uint32_t first = 0, count = 0;
    bool ascii = true;
    const int status = utf8_batch_report(batch,
                                         nullptr,
                                         &count,
                                         out_all_ascii != nullptr ? &ascii : nullptr,
                                         out_first_bad_row != nullptr ? &first : nullptr);
    if (status == 0 || status == EILSEQ) {
        if (out_first_bad_row != nullptr) {
            *out_first_bad_row = first;
        }
        if (out_all_ascii != nullptr) {
            *out_all_ascii = status == 0 && ascii;
        }
    }
    return status;
}

int dora_utf8_classify_batch(const struct dora_utf8_batch* batch,
                             uint64_t* invalid,
                             size_t word_capacity,
                             uint32_t* out_invalid_count,
                             bool* out_all_ascii) {
    if (batch == nullptr || out_invalid_count == nullptr || word_capacity < bitmap_words_for_bits(batch->row_count) ||
        (batch->row_count != 0 && invalid == nullptr)) {
        return EINVAL;
    }
    return utf8_batch_report(batch, invalid, out_invalid_count, out_all_ascii, nullptr);
}

static void utf8_first_bad(uint32_t* out_first, uint32_t row) {
    if (out_first != nullptr) {
        *out_first = row;
    }
}

#define DEFINE_UTF8_READS(LAYOUT, WIDTH, NAME, PROOF)                                                                  \
                                                                                                                       \
    static int utf8_counts_##LAYOUT##_##WIDTH##_##NAME(const struct dora_utf8_batch* batch,                            \
                                                       size_t* counts,                                                 \
                                                       uint32_t* out_first) {                                          \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (!utf8_mask_contains_row(batch->eligible_rows, row)) {                                                  \
                continue;                                                                                              \
            }                                                                                                          \
            struct dora_utf8_input input = {utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row)), PROOF};        \
            const int status = utf8_prove_ascii(&input);                                                               \
            if (status != 0) {                                                                                         \
                utf8_first_bad(out_first, row);                                                                        \
                return status;                                                                                         \
            }                                                                                                          \
            counts[row] = utf8_count_valid(input);                                                                     \
        }                                                                                                              \
        utf8_first_bad(out_first, batch->row_count);                                                                   \
        return 0;                                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    static int utf8_locates_##LAYOUT##_##WIDTH##_##NAME(const struct dora_utf8_batch* batch,                           \
                                                        size_t start,                                                  \
                                                        size_t count,                                                  \
                                                        struct dora_range* ranges,                                     \
                                                        uint32_t* out_first) {                                         \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (!utf8_mask_contains_row(batch->eligible_rows, row)) {                                                  \
                continue;                                                                                              \
            }                                                                                                          \
            struct dora_utf8_input input = {utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row)), PROOF};        \
            const int status = utf8_prove_ascii(&input);                                                               \
            if (status != 0) {                                                                                         \
                utf8_first_bad(out_first, row);                                                                        \
                return status;                                                                                         \
            }                                                                                                          \
            ranges[row] = utf8_locate_valid(input.proof, input.bytes, start, count);                                   \
        }                                                                                                              \
        utf8_first_bad(out_first, batch->row_count);                                                                   \
        return 0;                                                                                                      \
    }                                                                                                                  \
                                                                                                                       \
    static int utf8_finds_##LAYOUT##_##WIDTH##_##NAME(const struct dora_utf8_batch* batch,                             \
                                                      struct dora_bytes needle,                                        \
                                                      size_t* offsets,                                                 \
                                                      uint64_t* found,                                                 \
                                                      uint32_t* out_first) {                                           \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (!utf8_mask_contains_row(batch->eligible_rows, row)) {                                                  \
                continue;                                                                                              \
            }                                                                                                          \
            const struct dora_bytes value = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row));                \
            const int status = utf8_check_input((struct dora_utf8_input){value, PROOF});                               \
            if (status != 0) {                                                                                         \
                utf8_first_bad(out_first, row);                                                                        \
                return status;                                                                                         \
            }                                                                                                          \
            offsets[row] = dora_utf8_find_valid(value, needle, 0);                                                     \
            const bool match = needle.byte_count == 0 || offsets[row] < value.byte_count;                              \
            bitmap_assign(found, row, match);                                                                          \
        }                                                                                                              \
        utf8_first_bad(out_first, batch->row_count);                                                                   \
        return 0;                                                                                                      \
    }

#define DEFINE_UTF8_RAW_READS(LAYOUT, WIDTH)                                                                           \
                                                                                                                       \
    static int utf8_lengths_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                    \
                                               struct dora_utf8_mask requested,                                        \
                                               size_t out_lengths[],                                                   \
                                               size_t* out_length) {                                                   \
        size_t total = 0;                                                                                              \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            size_t size = 0;                                                                                           \
            if (utf8_mask_contains_row(requested, row) && utf8_mask_contains_row(batch->eligible_rows, row)) {         \
                size = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row)).byte_count;                          \
            }                                                                                                          \
            if (ckd_add(&total, total, size)) {                                                                        \
                return EOVERFLOW;                                                                                      \
            }                                                                                                          \
            if (out_lengths != nullptr) {                                                                              \
                out_lengths[row] = size;                                                                               \
            }                                                                                                          \
        }                                                                                                              \
        *out_length = total;                                                                                           \
        return 0;                                                                                                      \
    }

#define UTF8_PROOF_RETURN(BATCH, NAME, ...)                                                                            \
    switch ((BATCH)->proof) {                                                                                          \
    case DORA_UTF8_CHECKED:                                                                                            \
        return NAME##_checked(__VA_ARGS__);                                                                            \
    case DORA_UTF8_PROMISED:                                                                                           \
    case DORA_UTF8_VALIDATED:                                                                                          \
        return NAME##_validated(__VA_ARGS__);                                                                          \
    case DORA_UTF8_VALIDATED_ASCII:                                                                                    \
        return NAME##_ascii(__VA_ARGS__);                                                                              \
    default:                                                                                                           \
        return EINVAL;                                                                                                 \
    }
#define DEFINE_UTF8_TRUST_READS(LAYOUT, WIDTH)                                                                         \
    DEFINE_UTF8_READS(LAYOUT, WIDTH, checked, DORA_UTF8_CHECKED)                                                       \
    DEFINE_UTF8_READS(LAYOUT, WIDTH, validated, DORA_UTF8_VALIDATED)                                                   \
    DEFINE_UTF8_READS(LAYOUT, WIDTH, ascii, DORA_UTF8_VALIDATED_ASCII)                                                 \
                                                                                                                       \
    static int utf8_counts_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                     \
                                              size_t* counts,                                                          \
                                              uint32_t* out_first) {                                                   \
        UTF8_PROOF_RETURN(batch, utf8_counts_##LAYOUT##_##WIDTH, batch, counts, out_first)                             \
    }                                                                                                                  \
                                                                                                                       \
    static int utf8_locates_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                    \
                                               size_t start,                                                           \
                                               size_t count,                                                           \
                                               struct dora_range* ranges,                                              \
                                               uint32_t* out_first) {                                                  \
        UTF8_PROOF_RETURN(batch, utf8_locates_##LAYOUT##_##WIDTH, batch, start, count, ranges, out_first)              \
    }                                                                                                                  \
                                                                                                                       \
    static int utf8_finds_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                                      \
                                             struct dora_bytes needle,                                                 \
                                             size_t* offsets,                                                          \
                                             uint64_t* found,                                                          \
                                             uint32_t* out_first) {                                                    \
        UTF8_PROOF_RETURN(batch, utf8_finds_##LAYOUT##_##WIDTH, batch, needle, offsets, found, out_first)              \
    }
UTF8_INSTANTIATE(DEFINE_UTF8_TRUST_READS)
UTF8_INSTANTIATE(DEFINE_UTF8_RAW_READS)
#undef DEFINE_UTF8_TRUST_READS
#undef UTF8_PROOF_RETURN
#undef DEFINE_UTF8_RAW_READS

#undef DEFINE_UTF8_READS

static uint32_t utf8_first_eligible(const struct dora_utf8_batch* batch) {
    for (uint32_t row = 0; row < batch->row_count; ++row) {
        if (utf8_mask_contains_row(batch->eligible_rows, row)) {
            return row;
        }
    }
    return batch->row_count;
}

int dora_utf8_count_batch(const struct dora_utf8_batch* batch, size_t* counts, uint32_t* out_first_bad_row) {
    if (batch == nullptr || (batch->row_count != 0 && counts == nullptr)) {
        return EINVAL;
    }
    size_t extent;
    if (ckd_mul(&extent, (size_t)batch->row_count, sizeof *counts)) {
        return EOVERFLOW;
    }
    uint32_t eligible;
    const int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, &eligible);
    if (status != 0) {
        return status;
    }
    if (batch->layout == DORA_UTF8_CONSTANT) {
        if (eligible != 0) {
            size_t value;
            const int checked = dora_utf8_count((struct dora_utf8_input){batch->constant, batch->proof}, &value);
            if (checked != 0) {
                utf8_first_bad(out_first_bad_row, utf8_first_eligible(batch));
                return checked;
            }
            for (uint32_t row = 0; row < batch->row_count; ++row) {
                if (utf8_mask_contains_row(batch->eligible_rows, row)) {
                    counts[row] = value;
                }
            }
        }
        utf8_first_bad(out_first_bad_row, batch->row_count);
        return 0;
    }
    UTF8_DISPATCH(batch, utf8_counts, batch, counts, out_first_bad_row)
}

int dora_utf8_locate_batch(const struct dora_utf8_batch* batch,
                           size_t start_scalar_index,
                           size_t scalar_count,
                           struct dora_range* out_byte_ranges,
                           uint32_t* out_first_bad_row) {
    if (batch == nullptr || (batch->row_count != 0 && out_byte_ranges == nullptr)) {
        return EINVAL;
    }
    size_t extent;
    if (ckd_mul(&extent, (size_t)batch->row_count, sizeof *out_byte_ranges)) {
        return EOVERFLOW;
    }
    uint32_t eligible;
    const int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, &eligible);
    if (status != 0) {
        return status;
    }
    if (batch->layout == DORA_UTF8_CONSTANT) {
        if (eligible != 0) {
            struct dora_range value;
            const int checked = dora_utf8_locate((struct dora_utf8_input){batch->constant, batch->proof},
                                                 start_scalar_index,
                                                 scalar_count,
                                                 &value);
            if (checked != 0) {
                utf8_first_bad(out_first_bad_row, utf8_first_eligible(batch));
                return checked;
            }
            for (uint32_t row = 0; row < batch->row_count; ++row) {
                if (utf8_mask_contains_row(batch->eligible_rows, row)) {
                    out_byte_ranges[row] = value;
                }
            }
        }
        utf8_first_bad(out_first_bad_row, batch->row_count);
        return 0;
    }
    UTF8_DISPATCH(batch, utf8_locates, batch, start_scalar_index, scalar_count, out_byte_ranges, out_first_bad_row)
}

int dora_utf8_find_batch(const struct dora_utf8_batch* batch,
                         struct dora_utf8_input needle,
                         size_t* offsets,
                         uint64_t* found,
                         size_t word_capacity,
                         uint32_t* out_first_bad_row) {
    if (batch == nullptr || !utf8_input_is_valid(needle) ||
        (batch->row_count != 0 && (offsets == nullptr || found == nullptr)) ||
        word_capacity < bitmap_words_for_bits(batch->row_count)) {
        return EINVAL;
    }
    size_t extent;
    if (ckd_mul(&extent, (size_t)batch->row_count, sizeof *offsets)) {
        return EOVERFLOW;
    }
    uint32_t eligible;
    int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, &eligible);
    if (status != 0) {
        return status;
    }
    if (eligible != 0) {
        status = utf8_check_input(needle);
        if (status != 0) {
            for (uint32_t row = 0; row < batch->row_count; ++row) {
                if (utf8_mask_contains_row(batch->eligible_rows, row)) {
                    utf8_first_bad(out_first_bad_row, row);
                    break;
                }
            }
            return status;
        }
    }
    if (batch->layout == DORA_UTF8_CONSTANT) {
        if (eligible != 0) {
            size_t offset;
            bool match;
            status = dora_utf8_find((struct dora_utf8_input){batch->constant, batch->proof},
                                    (struct dora_utf8_input){needle.bytes, DORA_UTF8_VALIDATED},
                                    0,
                                    &offset,
                                    &match);
            if (status != 0) {
                utf8_first_bad(out_first_bad_row, utf8_first_eligible(batch));
                return status;
            }
            for (uint32_t row = 0; row < batch->row_count; ++row) {
                if (utf8_mask_contains_row(batch->eligible_rows, row)) {
                    offsets[row] = offset;
                    bitmap_assign(found, row, match);
                }
            }
        }
        utf8_first_bad(out_first_bad_row, batch->row_count);
        return 0;
    }
    UTF8_DISPATCH(batch, utf8_finds, batch, needle.bytes, offsets, found, out_first_bad_row)
}

static bool utf8_mask_covers(struct dora_utf8_mask mask, uint32_t count) {
    return mask.bytes.data == nullptr || dora_bitmap_count_set_bytes(mask.bytes.data, mask.bit_offset, count) == count;
}

#define DEFINE_UTF8_OFFSET_TOTAL(NAME)                                                                                 \
    static size_t utf8_offset_total_##NAME(const struct dora_utf8_batch* batch, struct dora_utf8_mask requested) {     \
        size_t total = 0;                                                                                              \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            const size_t size = (size_t)(utf8_offset_##NAME(batch, (size_t)row + 1) - utf8_offset_##NAME(batch, row)); \
            total += utf8_mask_contains_row(requested, row) && utf8_mask_contains_row(batch->eligible_rows, row)       \
                         ? size                                                                                        \
                         : 0;                                                                                          \
        }                                                                                                              \
        return total;                                                                                                  \
    }
DEFINE_UTF8_OFFSET_TOTAL(i32)
DEFINE_UTF8_OFFSET_TOTAL(i64)
#undef DEFINE_UTF8_OFFSET_TOTAL

static int utf8_lengths(const struct dora_utf8_batch* batch,
                        struct dora_utf8_mask requested,
                        size_t out_lengths[],
                        size_t* out_length) {
    if (out_lengths == nullptr && utf8_mask_covers(batch->eligible_rows, batch->row_count) &&
        utf8_mask_covers(requested, batch->row_count)) {
        if (batch->row_count == 0) {
            *out_length = 0;
            return 0;
        }
        if (batch->layout == DORA_UTF8_CONSTANT) {
            size_t total;
            if (ckd_mul(&total, (size_t)batch->row_count, batch->constant.byte_count)) {
                return EOVERFLOW;
            }
            *out_length = total;
            return 0;
        }
        if (batch->index_width_bytes == 0 && batch->layout == DORA_UTF8_OFFSETS_I32) {
            *out_length = (size_t)(utf8_offset_i32(batch, batch->row_count) - utf8_offset_i32(batch, 0));
            return 0;
        }
        if (batch->index_width_bytes == 0 && batch->layout == DORA_UTF8_OFFSETS_I64) {
            *out_length = (size_t)(utf8_offset_i64(batch, batch->row_count) - utf8_offset_i64(batch, 0));
            return 0;
        }
    }
    if (out_lengths == nullptr && batch->index_width_bytes == 0) {
        if (batch->layout == DORA_UTF8_OFFSETS_I32) {
            *out_length = utf8_offset_total_i32(batch, requested);
            return 0;
        }
        if (batch->layout == DORA_UTF8_OFFSETS_I64) {
            *out_length = utf8_offset_total_i64(batch, requested);
            return 0;
        }
    }
    UTF8_DISPATCH(batch, utf8_lengths, batch, requested, out_lengths, out_length)
}

int dora_utf8_measure_batch(const struct dora_utf8_batch* batch,
                            size_t out_lengths[],
                            size_t* out_length,
                            uint32_t* out_first_bad_row) {
    if (out_length == nullptr) {
        return EINVAL;
    }
    uint32_t eligible;
    int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, &eligible);
    if (status != 0) {
        return status;
    }
    size_t total;
    status = utf8_lengths(batch, (struct dora_utf8_mask){}, out_lengths, &total);
    if (status != 0) {
        return status;
    }
    if (batch->proof == DORA_UTF8_CHECKED) {
        uint32_t bad_count = 0, first = batch->row_count;
        utf8_classify_content(batch,
                              (struct dora_utf8_mask){},
                              nullptr,
                              &bad_count,
                              nullptr,
                              out_first_bad_row != nullptr ? &first : nullptr);
        if (bad_count != 0) {
            utf8_first_bad(out_first_bad_row, first);
            return EILSEQ;
        }
    }
    *out_length = total;
    utf8_first_bad(out_first_bad_row, batch->row_count);
    return 0;
}

#define DEFINE_UTF8_OUTPUT_OFFSET(NAME, TYPE)                                                                          \
    static void utf8_store_##NAME(uint32_t row, struct dora_mutable_bytes offsets, size_t value) {                     \
        const TYPE offset = (TYPE)value;                                                                               \
        memcpy(offsets.data + (size_t)row * sizeof offset, &offset, sizeof offset);                                    \
    }                                                                                                                  \
    static size_t utf8_load_##NAME(struct dora_mutable_bytes offsets, uint32_t row) {                                  \
        TYPE offset;                                                                                                   \
        memcpy(&offset, offsets.data + (size_t)row * sizeof offset, sizeof offset);                                    \
        return (size_t)offset;                                                                                         \
    }
DEFINE_UTF8_OUTPUT_OFFSET(i32, int32_t)
DEFINE_UTF8_OUTPUT_OFFSET(i64, int64_t)
#undef DEFINE_UTF8_OUTPUT_OFFSET

static int
utf8_output_shape(enum dora_utf8_write_policy policy, const struct dora_utf8_batch_output* output, uint32_t count) {
    if (output == nullptr || (policy != DORA_UTF8_ATOMIC && policy != DORA_UTF8_UNCOMMITTED) ||
        output->layout > DORA_UTF8_PACKED_I64) {
        return EINVAL;
    }
    size_t extent;
    size_t entries;
    if (ckd_add(&entries, (size_t)count, 1) ||
        ckd_mul(&extent, entries, output->layout == DORA_UTF8_PACKED_I32 ? 4 : 8)) {
        return EOVERFLOW;
    }
    if (!utf8_output_is_valid(output->packed.bytes) || !utf8_output_is_valid(output->packed.offset_bytes) ||
        output->packed.offset_bytes.byte_count < extent) {
        return EINVAL;
    }
    if (output->packed.byte_offset > (output->layout == DORA_UTF8_PACKED_I32 ? (size_t)INT32_MAX : (size_t)INT64_MAX)) {
        return EOVERFLOW;
    }
    if (policy == DORA_UTF8_UNCOMMITTED) {
        const size_t begin = output->layout == DORA_UTF8_PACKED_I32 ? utf8_load_i32(output->packed.offset_bytes, 0)
                                                                    : utf8_load_i64(output->packed.offset_bytes, 0);
        if (begin != output->packed.byte_offset) {
            return EINVAL;
        }
    }
    return 0;
}

static int utf8_packed_end(const struct dora_utf8_batch_output* output, size_t total, size_t* out_end) {
    size_t end;
    if (ckd_add(&end, output->packed.byte_offset, total) ||
        end > (output->layout == DORA_UTF8_PACKED_I32 ? (size_t)INT32_MAX : (size_t)INT64_MAX)) {
        return EOVERFLOW;
    }
    *out_end = end;
    return 0;
}

static int utf8_copy_check(struct dora_bytes value, bool* all_ascii) {
    bool ascii = false;
    const int status = dora_utf8_validate_span(value, all_ascii == nullptr ? nullptr : &ascii);
    if (all_ascii != nullptr) {
        *all_ascii &= ascii;
    }
    return status;
}

#define DEFINE_UTF8_PACKED_FILL(LAYOUT, WIDTH, DEST, PROOF, CHECK)                                                     \
                                                                                                                       \
    static int utf8_packed_##DEST##_##PROOF##_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                  \
                                                                 const struct dora_utf8_batch_output* output,          \
                                                                 uint32_t* out_first,                                  \
                                                                 bool* out_ascii) {                                    \
        size_t position = output->packed.byte_offset;                                                                  \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (utf8_mask_contains_row(batch->eligible_rows, row)) {                                                   \
                const struct dora_bytes value = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row));            \
                if (CHECK) {                                                                                           \
                    const int status = utf8_copy_check(value, out_ascii);                                              \
                    if (status != 0) {                                                                                 \
                        utf8_first_bad(out_first, row);                                                                \
                        return status;                                                                                 \
                    }                                                                                                  \
                }                                                                                                      \
                if (value.byte_count != 0) {                                                                           \
                    memcpy(output->packed.bytes.data + position, value.data, value.byte_count);                        \
                }                                                                                                      \
                position += value.byte_count;                                                                          \
            }                                                                                                          \
            utf8_store_##DEST(row + 1, output->packed.offset_bytes, position);                                         \
        }                                                                                                              \
        utf8_first_bad(out_first, batch->row_count);                                                                   \
        return 0;                                                                                                      \
    }
#define DEFINE_UTF8_PACKED_FILLS(LAYOUT, WIDTH)                                                                        \
    DEFINE_UTF8_PACKED_FILL(LAYOUT, WIDTH, i32, checked, true)                                                         \
    DEFINE_UTF8_PACKED_FILL(LAYOUT, WIDTH, i32, validated, false)                                                      \
    DEFINE_UTF8_PACKED_FILL(LAYOUT, WIDTH, i64, checked, true)                                                         \
    DEFINE_UTF8_PACKED_FILL(LAYOUT, WIDTH, i64, validated, false)
UTF8_INSTANTIATE(DEFINE_UTF8_PACKED_FILLS)
#undef DEFINE_UTF8_PACKED_FILLS
#undef DEFINE_UTF8_PACKED_FILL

#define DEFINE_UTF8_RUN_FILL(SOURCE, DEST, PROOF, CHECK)                                                               \
                                                                                                                       \
    static int utf8_run_fill_##SOURCE##_##DEST##_##PROOF(const struct dora_utf8_batch* batch,                          \
                                                         const struct dora_utf8_batch_output* output,                  \
                                                         uint32_t* out_first,                                          \
                                                         bool* out_ascii) {                                            \
        size_t position = output->packed.byte_offset;                                                                  \
        uint32_t row = 0;                                                                                              \
        while (row < batch->row_count) {                                                                               \
            if (!utf8_mask_contains_row(batch->eligible_rows, row)) {                                                  \
                utf8_store_##DEST(++row, output->packed.offset_bytes, position);                                       \
                continue;                                                                                              \
            }                                                                                                          \
            const size_t begin = (size_t)utf8_offset_##SOURCE(batch, row);                                             \
            size_t end = (size_t)utf8_offset_##SOURCE(batch, row + 1);                                                 \
            uint32_t last = row + 1;                                                                                   \
            while (last < batch->row_count) {                                                                          \
                const size_t next_begin = (size_t)utf8_offset_##SOURCE(batch, last),                                   \
                             next_end = (size_t)utf8_offset_##SOURCE(batch, (size_t)last + 1);                         \
                if (!utf8_mask_contains_row(batch->eligible_rows, last) && next_begin != next_end) {                   \
                    break;                                                                                             \
                }                                                                                                      \
                end = next_end;                                                                                        \
                ++last;                                                                                                \
            }                                                                                                          \
            if (CHECK) {                                                                                               \
                bool valid = utf8_copy_check(utf8_subspan(batch->rows.payload, begin, end - begin), out_ascii) == 0;   \
                if (valid) {                                                                                           \
                    for (uint32_t i = row; i < last; ++i) {                                                            \
                        const size_t first = (size_t)utf8_offset_##SOURCE(batch, i),                                   \
                                     limit = (size_t)utf8_offset_##SOURCE(batch, (size_t)i + 1);                       \
                        if (first < limit && utf8_byte_is_continuation(batch->rows.payload.data[first])) {             \
                            valid = false;                                                                             \
                            break;                                                                                     \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
                if (!valid) {                                                                                          \
                    for (uint32_t i = row; i < last; ++i) {                                                            \
                        if (utf8_mask_contains_row(batch->eligible_rows, i) &&                                         \
                            dora_utf8_validate_span(utf8_value_##SOURCE(batch, i), nullptr) != 0) {                    \
                            utf8_first_bad(out_first, i);                                                              \
                            return EILSEQ;                                                                             \
                        }                                                                                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            if (end != begin) {                                                                                        \
                memcpy(output->packed.bytes.data + position, batch->rows.payload.data + begin, end - begin);           \
            }                                                                                                          \
            for (uint32_t i = row; i < last; ++i) {                                                                    \
                utf8_store_##DEST(i + 1,                                                                               \
                                  output->packed.offset_bytes,                                                         \
                                  (size_t)utf8_offset_##SOURCE(batch, (size_t)i + 1) - begin + position);              \
            }                                                                                                          \
            position += end - begin;                                                                                   \
            row = last;                                                                                                \
        }                                                                                                              \
        utf8_first_bad(out_first, batch->row_count);                                                                   \
        return 0;                                                                                                      \
    }
DEFINE_UTF8_RUN_FILL(i32, i32, checked, true)
DEFINE_UTF8_RUN_FILL(i32, i32, validated, false)
DEFINE_UTF8_RUN_FILL(i32, i64, checked, true)
DEFINE_UTF8_RUN_FILL(i32, i64, validated, false)
DEFINE_UTF8_RUN_FILL(i64, i32, checked, true)
DEFINE_UTF8_RUN_FILL(i64, i32, validated, false)
DEFINE_UTF8_RUN_FILL(i64, i64, checked, true)
DEFINE_UTF8_RUN_FILL(i64, i64, validated, false)
#undef DEFINE_UTF8_RUN_FILL

#define DEFINE_UTF8_PACKED_DISPATCH(DEST, PROOF)                                                                       \
                                                                                                                       \
    static int utf8_packed_##DEST##_##PROOF(const struct dora_utf8_batch* batch,                                       \
                                            const struct dora_utf8_batch_output* output,                               \
                                            uint32_t* out_first,                                                       \
                                            bool* out_ascii) {                                                         \
        if (batch->index_width_bytes == 0) {                                                                           \
            if (batch->layout == DORA_UTF8_OFFSETS_I32) {                                                              \
                return utf8_run_fill_i32_##DEST##_##PROOF(batch, output, out_first, out_ascii);                        \
            }                                                                                                          \
            if (batch->layout == DORA_UTF8_OFFSETS_I64) {                                                              \
                return utf8_run_fill_i64_##DEST##_##PROOF(batch, output, out_first, out_ascii);                        \
            }                                                                                                          \
        }                                                                                                              \
        UTF8_DISPATCH(batch, utf8_packed_##DEST##_##PROOF, batch, output, out_first, out_ascii)                        \
    }
DEFINE_UTF8_PACKED_DISPATCH(i32, checked)
DEFINE_UTF8_PACKED_DISPATCH(i32, validated)
DEFINE_UTF8_PACKED_DISPATCH(i64, checked)
DEFINE_UTF8_PACKED_DISPATCH(i64, validated)
#undef DEFINE_UTF8_PACKED_DISPATCH
#define DEFINE_UTF8_PACKED_PROOF(DEST)                                                                                 \
                                                                                                                       \
    static int utf8_packed_##DEST(const struct dora_utf8_batch* batch,                                                 \
                                  const struct dora_utf8_batch_output* output,                                         \
                                  uint32_t* out_first,                                                                 \
                                  bool* out_ascii) {                                                                   \
        return batch->proof == DORA_UTF8_CHECKED                                                                       \
                   ? utf8_packed_##DEST##_checked(batch, output, out_first, out_ascii)                                 \
                   : utf8_packed_##DEST##_validated(batch, output, out_first, out_ascii);                              \
    }
DEFINE_UTF8_PACKED_PROOF(i32)
DEFINE_UTF8_PACKED_PROOF(i64)
#undef DEFINE_UTF8_PACKED_PROOF

int dora_utf8_copy_batch_sized(const struct dora_utf8_batch* batch,
                               size_t byte_bound,
                               const struct dora_utf8_batch_output* output,
                               uint32_t* out_first_bad_row,
                               bool* out_all_ascii) {
    uint32_t eligible;
    int status = utf8_preflight(batch, (struct dora_utf8_mask){}, nullptr, &eligible);
    if (status == 0) {
        status = utf8_output_shape(DORA_UTF8_UNCOMMITTED, output, batch->row_count);
    }
    if (status != 0) {
        return status;
    }
    size_t end;
    status = utf8_packed_end(output, byte_bound, &end);
    if (status != 0) {
        return status;
    }
    if (end > output->packed.bytes.byte_count) {
        return ENOBUFS;
    }
    if (out_all_ascii != nullptr) {
        *out_all_ascii =
            eligible == 0 || batch->proof == DORA_UTF8_VALIDATED_ASCII || batch->proof == DORA_UTF8_CHECKED;
    }
    struct dora_utf8_batch source = *batch;
    if (batch->layout == DORA_UTF8_CONSTANT && batch->proof == DORA_UTF8_CHECKED) {
        uint32_t bad_count = 0, first = batch->row_count;
        utf8_classify_content(batch,
                              (struct dora_utf8_mask){},
                              nullptr,
                              &bad_count,
                              out_all_ascii,
                              out_first_bad_row != nullptr ? &first : nullptr);
        if (bad_count != 0) {
            utf8_first_bad(out_first_bad_row, first);
            return EILSEQ;
        }
        source.proof = DORA_UTF8_VALIDATED;
    }
    if (output->layout == DORA_UTF8_PACKED_I32) {
        return utf8_packed_i32(&source, output, out_first_bad_row, out_all_ascii);
    }
    return utf8_packed_i64(&source, output, out_first_bad_row, out_all_ascii);
}

#define DEFINE_UTF8_CONCAT_LENGTHS(LAYOUT, WIDTH)                                                                      \
                                                                                                                       \
    static int utf8_concat_lengths_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                             \
                                                      size_t* lengths,                                                 \
                                                      struct dora_utf8_mask requested,                                 \
                                                      size_t* out_total) {                                             \
        size_t total = 0;                                                                                              \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (utf8_mask_contains_row(requested, row) && utf8_mask_contains_row(batch->eligible_rows, row)) {         \
                const size_t size = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row)).byte_count;             \
                if (ckd_add(&lengths[row], lengths[row], size) || ckd_add(&total, total, size)) {                      \
                    return EOVERFLOW;                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        *out_total = total;                                                                                            \
        return 0;                                                                                                      \
    }
UTF8_INSTANTIATE(DEFINE_UTF8_CONCAT_LENGTHS)
#undef DEFINE_UTF8_CONCAT_LENGTHS

static int utf8_concat_column_lengths(const struct dora_utf8_batch* batch,
                                      struct dora_utf8_mask requested,
                                      size_t* lengths,
                                      size_t* out_total) {
    UTF8_DISPATCH(batch, utf8_concat_lengths, batch, lengths, requested, out_total)
}

static int
utf8_concat_preflight(const struct dora_utf8_batch* batches, size_t input_count, struct dora_utf8_mask requested) {
    if (batches == nullptr || input_count == 0) {
        return EINVAL;
    }
    size_t extent;
    if (ckd_mul(&extent, input_count, sizeof *batches)) {
        return EOVERFLOW;
    }
    for (size_t i = 0; i < input_count; ++i) {
        if (batches[i].row_count != batches[0].row_count) {
            return EINVAL;
        }
        const int status = utf8_preflight(&batches[i], requested, nullptr, nullptr);
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

static int utf8_concat_lengths(const struct dora_utf8_batch* batches,
                               size_t input_count,
                               struct dora_utf8_mask requested,
                               size_t out_lengths[],
                               size_t* out_length) {
    size_t total = 0;
    if (out_lengths != nullptr) {
        size_t extent;
        if (ckd_mul(&extent, (size_t)batches[0].row_count, sizeof *out_lengths)) {
            return EOVERFLOW;
        }
        memset(out_lengths, 0, extent);
    }
    for (size_t i = 0; i < input_count; ++i) {
        size_t subtotal;
        const int status = out_lengths == nullptr
                               ? utf8_lengths(&batches[i], requested, nullptr, &subtotal)
                               : utf8_concat_column_lengths(&batches[i], requested, out_lengths, &subtotal);
        if (status != 0) {
            return status;
        }
        if (ckd_add(&total, total, subtotal)) {
            return EOVERFLOW;
        }
    }
    *out_length = total;
    return 0;
}

static int utf8_concat_check(const struct dora_utf8_batch* batches,
                             size_t input_count,
                             struct dora_utf8_mask requested,
                             uint32_t* out_first) {
    uint32_t first = batches[0].row_count;
    bool failed = false;
    for (size_t i = 0; i < input_count; ++i) {
        if (batches[i].proof != DORA_UTF8_CHECKED) {
            continue;
        }
        struct dora_utf8_batch source = batches[i];
        source.row_count = first;
        uint32_t count = 0, bad = first;
        utf8_classify_content(&source, requested, nullptr, &count, nullptr, &bad);
        if (count != 0) {
            first = bad;
            failed = true;
        }
    }
    utf8_first_bad(out_first, first);
    return failed ? EILSEQ : 0;
}

int dora_utf8_measure_concat(const struct dora_utf8_batch* batches,
                             size_t input_count,
                             struct dora_utf8_mask requested,
                             size_t out_lengths[],
                             size_t* out_length,
                             uint32_t* out_first_bad_row) {
    if (out_length == nullptr) {
        return EINVAL;
    }
    int status = utf8_concat_preflight(batches, input_count, requested);
    if (status != 0) {
        return status;
    }
    size_t total;
    status = utf8_concat_lengths(batches, input_count, requested, out_lengths, &total);
    if (status != 0) {
        return status;
    }
    status = utf8_concat_check(batches, input_count, requested, out_first_bad_row);
    if (status == 0) {
        *out_length = total;
    }
    return status;
}

#define DEFINE_UTF8_CONCAT_FILL(LAYOUT, WIDTH, DEST)                                                                   \
                                                                                                                       \
    static int utf8_concat_##DEST##_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                            \
                                                       struct dora_utf8_mask requested,                                \
                                                       const struct dora_utf8_batch_output* output) {                  \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            if (!utf8_mask_contains_row(requested, row) || !utf8_mask_contains_row(batch->eligible_rows, row)) {       \
                continue;                                                                                              \
            }                                                                                                          \
            const struct dora_bytes value = utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row));                \
            const size_t position = utf8_load_##DEST(output->packed.offset_bytes, row + 1);                            \
            if (value.byte_count != 0) {                                                                               \
                memcpy(output->packed.bytes.data + position, value.data, value.byte_count);                            \
            }                                                                                                          \
            utf8_store_##DEST(row + 1, output->packed.offset_bytes, position + value.byte_count);                      \
        }                                                                                                              \
        return 0;                                                                                                      \
    }
#define DEFINE_UTF8_CONCAT_FILLS(LAYOUT, WIDTH)                                                                        \
    DEFINE_UTF8_CONCAT_FILL(LAYOUT, WIDTH, i32) DEFINE_UTF8_CONCAT_FILL(LAYOUT, WIDTH, i64)
UTF8_INSTANTIATE(DEFINE_UTF8_CONCAT_FILLS)
#undef DEFINE_UTF8_CONCAT_FILLS
#undef DEFINE_UTF8_CONCAT_FILL

#define DEFINE_UTF8_CONCAT_CURSOR(LAYOUT, WIDTH, DEST)                                                                 \
                                                                                                                       \
    static int utf8_cursors_##DEST##_##LAYOUT##_##WIDTH(const struct dora_utf8_batch* batch,                           \
                                                        struct dora_utf8_mask requested,                               \
                                                        const struct dora_utf8_batch_output* output,                   \
                                                        bool first) {                                                  \
        size_t position = first ? output->packed.byte_offset : 0;                                                      \
        for (uint32_t row = 0; row < batch->row_count; ++row) {                                                        \
            utf8_store_##DEST(row + 1,                                                                                 \
                              output->packed.offset_bytes,                                                             \
                              position + (first ? 0 : utf8_load_##DEST(output->packed.offset_bytes, row + 1)));        \
            if (utf8_mask_contains_row(requested, row) && utf8_mask_contains_row(batch->eligible_rows, row)) {         \
                position += utf8_value_##LAYOUT(batch, utf8_index_##WIDTH(batch, row)).byte_count;                     \
            }                                                                                                          \
        }                                                                                                              \
        return 0;                                                                                                      \
    }
#define DEFINE_UTF8_CONCAT_CURSORS(LAYOUT, WIDTH)                                                                      \
    DEFINE_UTF8_CONCAT_CURSOR(LAYOUT, WIDTH, i32)                                                                      \
    DEFINE_UTF8_CONCAT_CURSOR(LAYOUT, WIDTH, i64)
UTF8_INSTANTIATE(DEFINE_UTF8_CONCAT_CURSORS)
#undef DEFINE_UTF8_CONCAT_CURSORS
#undef DEFINE_UTF8_CONCAT_CURSOR

#define DEFINE_UTF8_CONCAT_DISPATCH(DEST)                                                                              \
                                                                                                                       \
    static int utf8_concat_fill_##DEST(const struct dora_utf8_batch* batch,                                            \
                                       struct dora_utf8_mask requested,                                                \
                                       const struct dora_utf8_batch_output* output) {                                  \
        UTF8_DISPATCH(batch, utf8_concat_##DEST, batch, requested, output)                                             \
    }                                                                                                                  \
                                                                                                                       \
    static int utf8_concat_cursors_##DEST(const struct dora_utf8_batch* batch,                                         \
                                          struct dora_utf8_mask requested,                                             \
                                          const struct dora_utf8_batch_output* output,                                 \
                                          bool first) {                                                                \
        UTF8_DISPATCH(batch, utf8_cursors_##DEST, batch, requested, output, first)                                     \
    }
DEFINE_UTF8_CONCAT_DISPATCH(i32)
DEFINE_UTF8_CONCAT_DISPATCH(i64)
#undef DEFINE_UTF8_CONCAT_DISPATCH

#define DEFINE_UTF8_PAIR_FILL(SOURCE, DEST)                                                                            \
    static void utf8_pair_fill_##SOURCE##_##DEST(const struct dora_utf8_batch* batches,                                \
                                                 struct dora_utf8_mask requested,                                      \
                                                 const struct dora_utf8_batch_output* output) {                        \
        size_t position = output->packed.byte_offset;                                                                  \
        for (uint32_t row = 0; row < batches[0].row_count; ++row) {                                                    \
            if (utf8_mask_contains_row(requested, row)) {                                                              \
                if (utf8_mask_contains_row(batches[0].eligible_rows, row)) {                                           \
                    const struct dora_bytes value = utf8_value_##SOURCE(&batches[0], row);                             \
                    if (value.byte_count != 0) {                                                                       \
                        memcpy(output->packed.bytes.data + position, value.data, value.byte_count);                    \
                        position += value.byte_count;                                                                  \
                    }                                                                                                  \
                }                                                                                                      \
                if (utf8_mask_contains_row(batches[1].eligible_rows, row)) {                                           \
                    const struct dora_bytes value = utf8_value_##SOURCE(&batches[1], row);                             \
                    if (value.byte_count != 0) {                                                                       \
                        memcpy(output->packed.bytes.data + position, value.data, value.byte_count);                    \
                        position += value.byte_count;                                                                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            utf8_store_##DEST(row + 1, output->packed.offset_bytes, position);                                         \
        }                                                                                                              \
    }
DEFINE_UTF8_PAIR_FILL(i32, i32)
DEFINE_UTF8_PAIR_FILL(i32, i64)
DEFINE_UTF8_PAIR_FILL(i64, i32)
DEFINE_UTF8_PAIR_FILL(i64, i64)
#undef DEFINE_UTF8_PAIR_FILL

// False preserves all output.
static bool utf8_try_pair_fill(const struct dora_utf8_batch* batches,
                               size_t input_count,
                               struct dora_utf8_mask requested,
                               const struct dora_utf8_batch_output* output,
                               enum dora_utf8_write_policy policy) {
    if (input_count != 2 || batches[0].index_width_bytes != 0 || batches[1].index_width_bytes != 0 ||
        batches[0].layout != batches[1].layout ||
        (batches[0].layout != DORA_UTF8_OFFSETS_I32 && batches[0].layout != DORA_UTF8_OFFSETS_I64) ||
        (policy != DORA_UTF8_ATOMIC &&
         (batches[0].proof == DORA_UTF8_CHECKED || batches[1].proof == DORA_UTF8_CHECKED))) {
        return false;
    }
    if (output->layout == DORA_UTF8_PACKED_I32) {
        if (policy == DORA_UTF8_ATOMIC) {
            utf8_store_i32(0, output->packed.offset_bytes, output->packed.byte_offset);
        }
        if (batches[0].layout == DORA_UTF8_OFFSETS_I32) {
            utf8_pair_fill_i32_i32(batches, requested, output);
        } else {
            utf8_pair_fill_i64_i32(batches, requested, output);
        }
    } else {
        if (policy == DORA_UTF8_ATOMIC) {
            utf8_store_i64(0, output->packed.offset_bytes, output->packed.byte_offset);
        }
        if (batches[0].layout == DORA_UTF8_OFFSETS_I32) {
            utf8_pair_fill_i32_i64(batches, requested, output);
        } else {
            utf8_pair_fill_i64_i64(batches, requested, output);
        }
    }
    return true;
}

int dora_utf8_concat(const struct dora_utf8_batch* batches,
                     size_t input_count,
                     struct dora_utf8_mask requested,
                     enum dora_utf8_write_policy policy,
                     const struct dora_utf8_batch_output* output,
                     uint32_t* out_first_bad_row) {
    int status = utf8_concat_preflight(batches, input_count, requested);
    if (status == 0) {
        status = utf8_output_shape(policy, output, batches[0].row_count);
    }
    if (status != 0) {
        return status;
    }
    size_t total, end;
    status = utf8_concat_lengths(batches, input_count, requested, nullptr, &total);
    if (status == 0) {
        status = utf8_packed_end(output, total, &end);
    }
    if (status != 0) {
        return status;
    }
    if (policy == DORA_UTF8_UNCOMMITTED && end > output->packed.bytes.byte_count) {
        return ENOBUFS;
    }
    if (policy == DORA_UTF8_ATOMIC) {
        uint32_t first;
        status = utf8_concat_check(batches, input_count, requested, &first);
        if (status != 0) {
            utf8_first_bad(out_first_bad_row, first);
            return status;
        }
    }
    if (end > output->packed.bytes.byte_count) {
        return ENOBUFS;
    }
    if (utf8_try_pair_fill(batches, input_count, requested, output, policy)) {
        utf8_first_bad(out_first_bad_row, batches[0].row_count);
        return 0;
    }
    if (output->layout == DORA_UTF8_PACKED_I32) {
        if (policy == DORA_UTF8_ATOMIC) {
            utf8_store_i32(0, output->packed.offset_bytes, output->packed.byte_offset);
        }
        for (size_t i = 0; i < input_count; ++i) {
            status = utf8_concat_cursors_i32(&batches[i], requested, output, i == 0);
            if (status != 0) {
                return status;
            }
        }
    } else {
        if (policy == DORA_UTF8_ATOMIC) {
            utf8_store_i64(0, output->packed.offset_bytes, output->packed.byte_offset);
        }
        for (size_t i = 0; i < input_count; ++i) {
            status = utf8_concat_cursors_i64(&batches[i], requested, output, i == 0);
            if (status != 0) {
                return status;
            }
        }
    }
    uint32_t first = batches[0].row_count;
    for (size_t i = 0; i < input_count; ++i) {
        struct dora_utf8_batch source = batches[i];
        if (policy == DORA_UTF8_ATOMIC && source.proof == DORA_UTF8_CHECKED) {
            source.proof = DORA_UTF8_VALIDATED;
        }
        if (source.proof == DORA_UTF8_CHECKED) {
            uint32_t bad_count = 0;
            first = source.row_count;
            utf8_classify_content(&source, requested, nullptr, &bad_count, nullptr, &first);
            status = bad_count == 0 ? 0 : EILSEQ;
            source.proof = DORA_UTF8_VALIDATED;
        }
        if (status == 0) {
            status = output->layout == DORA_UTF8_PACKED_I32 ? utf8_concat_fill_i32(&source, requested, output)
                                                            : utf8_concat_fill_i64(&source, requested, output);
        }
        if (status != 0) {
            if (out_first_bad_row != nullptr) {
                for (size_t j = i + 1; j < input_count; ++j) {
                    if (batches[j].proof != DORA_UTF8_CHECKED) {
                        continue;
                    }
                    source = batches[j];
                    source.row_count = first;
                    uint32_t count = 0, bad = first;
                    utf8_classify_content(&source, requested, nullptr, &count, nullptr, &bad);
                    if (count != 0) {
                        first = bad;
                    }
                }
                *out_first_bad_row = first;
            }
            return status;
        }
    }
    utf8_first_bad(out_first_bad_row, batches[0].row_count);
    return 0;
}

#undef UTF8_DISPATCH
#undef UTF8_DISPATCH_WIDTH
#undef UTF8_INSTANTIATE
#undef UTF8_INSTANTIATE_LAYOUT
#undef utf8_value_constant
