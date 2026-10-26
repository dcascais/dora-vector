// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/view.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/types.h"
#include "types/types_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stdint.h>

int dora_vector_view_read_index(const struct dora_vector_view* view, uint64_t row, uint64_t* out_index) {
    if (view == nullptr || out_index == nullptr || view->values.data == nullptr) {
        return EINVAL;
    }
    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    const uint8_t* const data = view->values.data;
    switch (view->physical_type) {
        SIGNED_INDEX_CARRIERS(INDEX_READ_CASE, NEGATIVE_SIGNED, data, physical, out_index)
        UNSIGNED_INDEX_CARRIERS(INDEX_READ_CASE, NEGATIVE_UNSIGNED, data, physical, out_index)
    default:
        return EINVAL;
    }
}

//------------------------------------------------------------------------------
//
// Validity
//
//------------------------------------------------------------------------------

uint64_t dora_vector_view_count_nulls(const struct dora_vector_view* view, struct dora_range rows) {
    uint64_t count = 0;
    uint64_t row = rows.offset;
    uint64_t remaining = rows.length;
    while (remaining != 0) {
        const uint32_t shift = bitmap_bit_of(row);
        const uint32_t available = 64 - shift;
        const uint32_t take = remaining < available ? (uint32_t)remaining : available;
        const uint64_t word =
            bitmap_word_slice(dora_vector_view_get_validity_word(view, bitmap_word_of(row)), shift, take);
        count += take - dora_bitmap_word_count_ones(word);
        row += take;
        remaining -= take;
    }
    return count;
}

void dora_vector_view_copy_validity(const struct dora_vector_view* view, struct dora_range rows, bool* out_validity) {
    for (uint64_t row = 0; row < rows.length; ++row) {
        out_validity[row] = !dora_vector_view_is_null(view, rows.offset + row);
    }
}
