// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/types.h"

#include "dora/vector/status.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Type Rules
//
//------------------------------------------------------------------------------

// Absent and empty Arrow timezones are equivalent.
static bool text_is_equal(const char* left, const char* right) {
    return strcmp(left != nullptr ? left : "", right != nullptr ? right : "") == 0;
}

static bool type_is_index(enum dora_type id) {
    return (id >= DORA_TYPE_INT8 && id <= DORA_TYPE_INT64) || (id >= DORA_TYPE_UINT8 && id <= DORA_TYPE_UINT64);
}

static bool type_is_run_end(enum dora_type id) {
    return id == DORA_TYPE_INT16 || id == DORA_TYPE_INT32 || id == DORA_TYPE_INT64;
}

static enum dora_layout default_layout(enum dora_type id) {
    switch (id) {
    case DORA_TYPE_NULL:
        return DORA_LAYOUT_NULL;
    case DORA_TYPE_BOOLEAN:
        return DORA_LAYOUT_BITS;
    case DORA_TYPE_UTF8:
    case DORA_TYPE_BINARY:
        return DORA_LAYOUT_BINARY;
    case DORA_TYPE_LIST:
    case DORA_TYPE_MAP:
        return DORA_LAYOUT_LIST;
    case DORA_TYPE_FIXED_LIST:
        return DORA_LAYOUT_FIXED_LIST;
    case DORA_TYPE_STRUCT:
        return DORA_LAYOUT_STRUCT;
    case DORA_TYPE_UNION:
        return DORA_LAYOUT_DENSE_UNION;
    default:
        return DORA_LAYOUT_FIXED;
    }
}

static bool layout_is_allowed(enum dora_type id, enum dora_layout layout) {
    if (layout == default_layout(id)) {
        return true;
    }
    switch (id) {
    case DORA_TYPE_UTF8:
    case DORA_TYPE_BINARY:
        return layout == DORA_LAYOUT_LARGE_BINARY || layout == DORA_LAYOUT_BINARY_VIEW;
    case DORA_TYPE_LIST:
        return layout == DORA_LAYOUT_LARGE_LIST || layout == DORA_LAYOUT_LIST_VIEW ||
               layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    case DORA_TYPE_MAP:
        return layout == DORA_LAYOUT_LARGE_LIST;
    case DORA_TYPE_UNION:
        return layout == DORA_LAYOUT_SPARSE_UNION;
    default:
        return false;
    }
}

int dora_type_check(const struct dora_logical_type* type) {
    if (type == nullptr || type->id > DORA_TYPE_UNION || !bytes_are_well_formed(type->extension_name) ||
        !bytes_are_well_formed(type->extension_metadata) ||
        (type->extension_name.byte_count == 0 && type->extension_metadata.byte_count != 0)) {
        return EINVAL;
    }
    switch (type->id) {
    case DORA_TYPE_DECIMAL:
        return type_decimal_byte_width(type) != 0 ? DORA_OK : EINVAL;
    case DORA_TYPE_TIME:
    case DORA_TYPE_DURATION:
        return type->parameters.unit <= DORA_TIME_UNIT_NANOSECOND ? DORA_OK : EINVAL;
    case DORA_TYPE_TIMESTAMP:
        return type->parameters.timestamp.unit <= DORA_TIME_UNIT_NANOSECOND ? DORA_OK : EINVAL;
    default:
        return DORA_OK;
    }
}

int dora_type_resolve_layout(const struct dora_logical_type* type,
                             enum dora_layout layout,
                             enum dora_type index_type,
                             enum dora_layout* out_layout) {
    if (type == nullptr || out_layout == nullptr || type->id > DORA_TYPE_UNION) {
        return EINVAL;
    }
    if (layout == DORA_LAYOUT_DICTIONARY || layout == DORA_LAYOUT_RUN_END) {
        const bool is_index_valid = (layout == DORA_LAYOUT_DICTIONARY && type_is_index(index_type)) ||
                                    (layout == DORA_LAYOUT_RUN_END && type_is_run_end(index_type));
        if (!is_index_valid) {
            return EINVAL;
        }
        *out_layout = layout;
        return DORA_OK;
    }
    if (index_type != DORA_TYPE_NULL || layout > DORA_LAYOUT_RUN_END) {
        return EINVAL;
    }
    const enum dora_layout resolved = layout == DORA_LAYOUT_DEFAULT ? default_layout(type->id) : layout;
    if (!layout_is_allowed(type->id, resolved)) {
        return EINVAL;
    }
    *out_layout = resolved;
    return DORA_OK;
}

bool dora_type_is_equal(const struct dora_logical_type* left,
                        const struct dora_logical_type* right,
                        uint32_t union_child_count) {
    if (left->id != right->id || !bytes_are_equal(left->extension_name, right->extension_name) ||
        !bytes_are_equal(left->extension_metadata, right->extension_metadata)) {
        return false;
    }
    const union dora_type_parameters* const first = &left->parameters;
    const union dora_type_parameters* const second = &right->parameters;
    switch (left->id) {
    case DORA_TYPE_DECIMAL: {
        if (first->decimal.precision != second->decimal.precision || first->decimal.scale != second->decimal.scale) {
            return false;
        }
        const uint32_t first_width = type_decimal_byte_width(left);
        return first_width != 0 && first_width == type_decimal_byte_width(right);
    }
    case DORA_TYPE_TIMESTAMP:
        return first->timestamp.unit == second->timestamp.unit &&
               text_is_equal(first->timestamp.timezone, second->timestamp.timezone);
    case DORA_TYPE_TIME:
    case DORA_TYPE_DURATION:
        return first->unit == second->unit;
    case DORA_TYPE_FIXED_BINARY:
        return first->byte_width == second->byte_width;
    case DORA_TYPE_FIXED_LIST:
        return first->list_element_count == second->list_element_count;
    case DORA_TYPE_UNION:
        for (uint32_t index = 0; index < union_child_count; ++index) {
            const int32_t first_code = first->type_codes != nullptr ? first->type_codes[index] : (int32_t)index;
            const int32_t second_code = second->type_codes != nullptr ? second->type_codes[index] : (int32_t)index;
            if (first_code != second_code) {
                return false;
            }
        }
        return true;
    default:
        return true;
    }
}
