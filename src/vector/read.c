// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "types/text.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Navigation
//
//------------------------------------------------------------------------------

int dora_vector_get_own_validity(const struct dora_vector* vector,
                                 struct dora_view_buffer* out_validity,
                                 uint64_t* out_bit_offset) {
    struct vector_image* image = nullptr;
    if (out_validity == nullptr || out_bit_offset == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.view.selection.indices != nullptr) {
        return ENOTSUP;
    }
    *out_validity = image->ownership.own_validity;
    *out_bit_offset = image->ownership.own_validity_bit_offset + vector->rows.offset;
    return DORA_OK;
}

int dora_vector_get_binary_view_sizes(const struct dora_vector* vector, struct dora_view_buffer* out_sizes) {
    struct vector_image* image = nullptr;
    if (out_sizes == nullptr || dora_vector_resolve(vector, &image) != DORA_OK ||
        image->ownership.view.layout != DORA_LAYOUT_BINARY_VIEW) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &image->ownership.view;
    *out_sizes = view->data_buffers[view->data_buffer_count];
    return DORA_OK;
}

int dora_vector_get_own_null_count(const struct dora_vector* vector, int64_t* out_null_count) {
    struct vector_image* image = nullptr;
    if (out_null_count == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.view.selection.indices != nullptr) {
        return ENOTSUP;
    }
    const int64_t count = image->ownership.own_null_count;
    *out_null_count =
        count != 0 && (vector->rows.offset != 0 || vector->rows.length != image->ownership.view.length) ? -1 : count;
    return DORA_OK;
}

int dora_vector_get_length(const struct dora_vector* vector, uint64_t* out_length) {
    if (out_length == nullptr || !dora_vector_is_valid(vector)) {
        return EINVAL;
    }
    *out_length = vector->rows.length;
    return DORA_OK;
}

int dora_vector_get_type(const struct dora_vector* vector, struct dora_logical_type* out_type) {
    struct vector_image* image = nullptr;
    if (out_type == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    struct dora_schema_field field = {};
    const int status = dora_schema_get_field(&image->schema, &field);
    if (status == DORA_OK) {
        *out_type = field.type;
    }
    return status;
}

int dora_vector_retain_schema(const struct dora_vector* vector, struct dora_schema* out_schema) {
    struct vector_image* image = nullptr;
    const int status = dora_vector_resolve(vector, &image);
    if (status != DORA_OK) {
        return status;
    }
    return dora_schema_retain(&image->schema, out_schema);
}

int dora_vector_get_child_count(const struct dora_vector* vector, uint32_t* out_child_count) {
    struct vector_image* image = nullptr;
    if (out_child_count == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &image->ownership.view;
    *out_child_count =
        view->type == DORA_TYPE_MAP && (view->layout == DORA_LAYOUT_LIST || view->layout == DORA_LAYOUT_LARGE_LIST)
            ? 2
            : image->child_vectors.entry_count;
    return DORA_OK;
}

int dora_vector_get_physical_child(const struct dora_vector* vector,
                                   uint32_t child_index,
                                   struct dora_vector* out_child) {
    struct vector_image* image = nullptr;
    if (!dora_vector_is_empty(out_child) || dora_vector_resolve(vector, &image) != DORA_OK ||
        child_index >= image->child_vectors.entry_count) {
        return EINVAL;
    }
    const struct dora_vector* const child = &dora_vector_image_children(image)[child_index];
    *out_child = dora_vector_borrow(child);
    return DORA_OK;
}

int dora_vector_slice(const struct dora_vector* vector, struct dora_range rows, struct dora_vector* out_slice) {
    if (!dora_vector_is_empty(out_slice) || !dora_vector_is_valid(vector) || rows.offset > vector->rows.length ||
        rows.length > vector->rows.length - rows.offset) {
        return EINVAL;
    }
    *out_slice = dora_vector_borrow(vector);
    out_slice->rows.offset += rows.offset;
    out_slice->rows.length = rows.length;
    return DORA_OK;
}

int dora_vector_get_child(const struct dora_vector* vector, uint32_t child_index, struct dora_vector* out_child) {
    struct vector_image* image = nullptr;
    if (!dora_vector_is_empty(out_child) || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &image->ownership.view;
    if (view->type == DORA_TYPE_MAP && (view->layout == DORA_LAYOUT_LIST || view->layout == DORA_LAYOUT_LARGE_LIST)) {
        struct dora_vector entries = {};
        const int status = dora_vector_get_physical_child(vector, 0, &entries);
        return status == DORA_OK ? dora_vector_get_child(&entries, child_index, out_child) : status;
    }
    if (view->layout == DORA_LAYOUT_STRUCT) {
        struct dora_vector child = {};
        const int status = dora_vector_get_physical_child(vector, child_index, &child);
        if (status != DORA_OK) {
            return status;
        }
        const struct dora_range rows = {
            .offset = image->node.offset + vector->rows.offset,
            .length = vector->rows.length
        };
        return dora_vector_slice(&child, rows, out_child);
    }
    return dora_vector_get_physical_child(vector, child_index, out_child);
}

int dora_vector_find_child(const struct dora_vector* vector, const char* name, struct dora_vector* out_child) {
    struct vector_image* image = nullptr;
    if (name == nullptr || !dora_vector_is_empty(out_child) || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    uint32_t child_count = 0;
    int status = dora_vector_get_child_count(vector, &child_count);
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t index = 0; index < child_count; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_child(&image->schema, index, &child);
        struct dora_schema_field field = {};
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (field.name != nullptr && strcmp(field.name, name) == 0) {
            return dora_vector_get_child(vector, index, out_child);
        }
    }
    return ENOENT;
}

int dora_vector_get_storage(const struct dora_vector* vector, struct dora_vector* out_storage) {
    struct vector_image* image = nullptr;
    if (!dora_vector_is_empty(out_storage) || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.storage.handle.slot_index == 0) {
        return ENOTSUP;
    }
    return dora_vector_slice(&image->ownership.storage, vector->rows, out_storage);
}

int dora_vector_get_statistics(const struct dora_vector* vector, struct dora_vector* out_statistics) {
    struct vector_image* image = nullptr;
    if (!dora_vector_is_empty(out_statistics) || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.statistics.handle.slot_index == 0 || vector->rows.offset != 0 ||
        vector->rows.length != image->ownership.view.length) {
        return ENOTSUP;
    }
    *out_statistics = dora_vector_borrow(&image->ownership.statistics);
    return DORA_OK;
}

int dora_vector_get_computed_statistics(const struct dora_vector* vector, struct dora_vector* out_statistics) {
    struct vector_image* image = nullptr;
    if (!dora_vector_is_empty(out_statistics) || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    if (image->ownership.statistics_provenance != VECTOR_STATISTICS_COMPUTED) {
        return ENOTSUP;
    }
    return dora_vector_get_statistics(vector, out_statistics);
}

//------------------------------------------------------------------------------
//
// Reads
//
//------------------------------------------------------------------------------

int dora_vector_is_null(const struct dora_vector* vector, uint64_t row, bool* out_is_null) {
    if (vector == nullptr || out_is_null == nullptr || row >= vector->rows.length) {
        return EINVAL;
    }
    struct dora_vector located = dora_vector_borrow(vector);
    for (;;) {
        struct vector_image* image = nullptr;
        int status = dora_vector_resolve(&located, &image);
        if (status != DORA_OK) {
            return status;
        }
        const struct dora_vector_view* const original = &image->ownership.view;
        if (original->layout == DORA_LAYOUT_NULL) {
            *out_is_null = true;
            return DORA_OK;
        }
        if (!image->ownership.requires_effective_validity && original->null_count == 0 &&
            !vector_layout_derives_nulls(original->layout)) {
            *out_is_null = false;
            return DORA_OK;
        }
        struct dora_vector_view view = {};
        status = dora_vector_get_view(&located, &view);
        if (status != DORA_OK) {
            return status;
        }
        if (dora_vector_view_is_null(&view, row)) {
            *out_is_null = true;
            return DORA_OK;
        }
        uint32_t child_index = 0;
        uint64_t child_row = row;
        switch (view.layout) {
        case DORA_LAYOUT_DICTIONARY:
            status = dora_vector_view_read_index(&view, row, &child_row);
            break;
        case DORA_LAYOUT_DENSE_UNION:
        case DORA_LAYOUT_SPARSE_UNION:
            child_index = dora_vector_view_get_member(&view, row, &child_row);
            break;
        case DORA_LAYOUT_RUN_END:
            child_index = 1;
            child_row = dora_vector_view_find_run(&view, row);
            break;
        default:
            *out_is_null = false;
            return DORA_OK;
        }
        struct dora_vector child = {};
        if (status == DORA_OK) {
            status = dora_vector_get_child(&located, child_index, &child);
        }
        if (status != DORA_OK || child_row >= child.rows.length) {
            return status != DORA_OK ? status : EINVAL;
        }
        located = child;
        row = child_row;
    }
}

int dora_vector_decode_row(const struct dora_vector* vector, uint64_t row, struct decoded_row* out_row) {
    if (vector == nullptr || out_row == nullptr || row >= vector->rows.length) {
        return EINVAL;
    }
    struct dora_vector located = dora_vector_borrow(vector);
    for (;;) {
        struct dora_logical_type type = {};
        int status = dora_vector_get_type(&located, &type);
        struct decoded_row decoded = {.type = type.id};
        if (status == DORA_OK) {
            status = type_physical(&type, &decoded.physical_type);
        }
        struct dora_vector_view view = {};
        if (status == DORA_OK) {
            status = dora_vector_get_view(&located, &view);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (dora_vector_view_is_null(&view, row)) {
            decoded.is_null = true;
            *out_row = decoded;
            return DORA_OK;
        }
        if (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END) {
            uint64_t child_row = 0;
            uint32_t child_index = 0;
            if (view.layout == DORA_LAYOUT_DICTIONARY) {
                status = dora_vector_view_read_index(&view, row, &child_row);
            } else {
                child_index = 1;
                child_row = dora_vector_view_find_run(&view, row);
            }
            struct dora_vector child = {};
            if (status == DORA_OK) {
                status = dora_vector_get_child(&located, child_index, &child);
            }
            if (status != DORA_OK || child_row >= child.rows.length) {
                return status != DORA_OK ? status : EINVAL;
            }
            located = child;
            row = child_row;
            continue;
        }
        switch (view.layout) {
        case DORA_LAYOUT_BITS:
            decoded.value.boolean = dora_vector_view_get_bool(&view, row);
            break;
        case DORA_LAYOUT_FIXED:
            if (type.id == DORA_TYPE_FIXED_BINARY) {
                decoded.value.bytes = dora_vector_view_get_bytes(&view, row);
                break;
            }
            switch (view.physical_type) {
#define DECODE_CARRIER(S, T, P)                                                                                        \
    case P:                                                                                                            \
        memcpy(&decoded.value.S,                                                                                       \
               (const uint8_t*)view.values.data + dora_vector_view_get_physical_row(&view, row) * sizeof(T),           \
               sizeof(T));                                                                                             \
        break;
                FIXED_CARRIERS(DECODE_CARRIER)
#undef DECODE_CARRIER
            default:
                return EINVAL;
            }
            break;
        case DORA_LAYOUT_BINARY:
        case DORA_LAYOUT_LARGE_BINARY:
        case DORA_LAYOUT_BINARY_VIEW:
            decoded.value.bytes = dora_vector_view_get_bytes(&view, row);
            break;
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW:
        case DORA_LAYOUT_FIXED_LIST:
            decoded.value.elements.rows = dora_vector_view_get_elements(&view, row);
            if (type.id == DORA_TYPE_MAP) {
                decoded.value.elements.vector = located;
            } else {
                status = dora_vector_get_child(&located, 0, &decoded.value.elements.vector);
            }
            break;
        case DORA_LAYOUT_STRUCT:
            decoded.value.elements = (struct decoded_elements){.vector = located, .rows = {.offset = row, .length = 1}};
            break;
        case DORA_LAYOUT_DENSE_UNION:
        case DORA_LAYOUT_SPARSE_UNION: {
            uint64_t child_row = 0;
            const uint32_t child_index = dora_vector_view_get_member(&view, row, &child_row);
            struct dora_vector child = {};
            status = dora_vector_get_child(&located, child_index, &child);
            if (status == DORA_OK) {
                status = dora_vector_is_null(&child, child_row, &decoded.is_null);
            }
            if (status == DORA_OK && !decoded.is_null) {
                decoded.value.member = (struct decoded_member){
                    .vector = child,
                    .row = child_row,
                    .type_id = dora_vector_view_get_type_id(&view, row)
                };
            }
            break;
        }
        default:
            return EINVAL;
        }
        if (status != DORA_OK) {
            return status;
        }
        *out_row = decoded;
        return DORA_OK;
    }
}

//--------------------------------------
// Boolean
//--------------------------------------

int dora_vector_read_bool(const struct dora_vector* vector, uint64_t row, bool* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.type != DORA_TYPE_BOOLEAN) {
        return EINVAL;
    }
    *out_value = decoded.value.boolean;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_vector_read_i8(const struct dora_vector* vector, uint64_t row, int8_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I8) {
        return EINVAL;
    }
    *out_value = decoded.value.i8;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_i16(const struct dora_vector* vector, uint64_t row, int16_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I16) {
        return EINVAL;
    }
    *out_value = decoded.value.i16;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_i32(const struct dora_vector* vector, uint64_t row, int32_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I32) {
        return EINVAL;
    }
    *out_value = decoded.value.i32;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_i64(const struct dora_vector* vector, uint64_t row, int64_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I64) {
        return EINVAL;
    }
    *out_value = decoded.value.i64;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_i128(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_i128* out_value,
                          bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I128) {
        return EINVAL;
    }
    *out_value = decoded.value.i128;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_i256(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_i256* out_value,
                          bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_I256) {
        return EINVAL;
    }
    *out_value = decoded.value.i256;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_vector_read_u8(const struct dora_vector* vector, uint64_t row, uint8_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_U8) {
        return EINVAL;
    }
    *out_value = decoded.value.u8;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_u16(const struct dora_vector* vector, uint64_t row, uint16_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_U16) {
        return EINVAL;
    }
    *out_value = decoded.value.u16;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_u32(const struct dora_vector* vector, uint64_t row, uint32_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_U32) {
        return EINVAL;
    }
    *out_value = decoded.value.u32;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_u64(const struct dora_vector* vector, uint64_t row, uint64_t* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_U64) {
        return EINVAL;
    }
    *out_value = decoded.value.u64;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_u128(const struct dora_vector* vector,
                          uint64_t row,
                          struct dora_u128* out_value,
                          bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_U128) {
        return EINVAL;
    }
    *out_value = decoded.value.u128;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_vector_read_f16(const struct dora_vector* vector,
                         uint64_t row,
                         struct dora_f16_bits* out_value,
                         bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_F16) {
        return EINVAL;
    }
    *out_value = decoded.value.f16;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_f32(const struct dora_vector* vector, uint64_t row, float* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_F32) {
        return EINVAL;
    }
    *out_value = decoded.value.f32;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_f64(const struct dora_vector* vector, uint64_t row, double* out_value, bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_F64) {
        return EINVAL;
    }
    *out_value = decoded.value.f64;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_vector_read_interval_day_time(const struct dora_vector* vector,
                                       uint64_t row,
                                       struct dora_interval_day_time* out_value,
                                       bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME) {
        return EINVAL;
    }
    *out_value = decoded.value.interval_day_time;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_interval_month_day_nano(const struct dora_vector* vector,
                                             uint64_t row,
                                             struct dora_interval_month_day_nano* out_value,
                                             bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.physical_type != DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO) {
        return EINVAL;
    }
    *out_value = decoded.value.interval_month_day_nano;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_read_bytes(const struct dora_vector* vector,
                           uint64_t row,
                           struct dora_bytes* out_value,
                           bool* out_is_null) {
    if (out_value == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct decoded_row decoded = {};
    const int status = dora_vector_decode_row(vector, row, &decoded);
    if (status != DORA_OK) {
        return status;
    }
    if (decoded.type != DORA_TYPE_UTF8 && decoded.type != DORA_TYPE_BINARY && decoded.type != DORA_TYPE_FIXED_BINARY) {
        return EINVAL;
    }
    *out_value = decoded.value.bytes;
    *out_is_null = decoded.is_null;
    return DORA_OK;
}

int dora_vector_format_value(const struct dora_vector* vector,
                             uint64_t row,
                             char* out_text,
                             size_t capacity_bytes,
                             size_t* out_byte_count,
                             bool* out_is_null) {
    if (out_byte_count == nullptr || out_is_null == nullptr) {
        return EINVAL;
    }
    struct dora_logical_type type = {};
    int status = dora_vector_get_type(vector, &type);
    struct decoded_row decoded = {};
    if (status == DORA_OK) {
        status = dora_vector_decode_row(vector, row, &decoded);
    }
    if (status != DORA_OK) {
        return status;
    }
    size_t byte_count = 0;
    status = dora_type_format_scalar(&type,
                                     decoded.physical_type,
                                     &decoded.value,
                                     decoded.is_null,
                                     out_text,
                                     capacity_bytes,
                                     &byte_count);
    if (status == DORA_OK || status == ENOBUFS) {
        *out_byte_count = byte_count;
        *out_is_null = decoded.is_null;
    }
    return status;
}

//--------------------------------------
// Nullness
//--------------------------------------

bool dora_vector_try_get_exact_null_count(const struct dora_vector* vector,
                                          const struct vector_image* image,
                                          uint64_t* out_null_count) {
    const struct dora_vector_view* const source = &image->ownership.view;
    if (vector_layout_derives_nulls(source->layout) || image->ownership.requires_effective_validity ||
        source->null_count < 0) {
        return false;
    }
    if (source->null_count == 0) {
        *out_null_count = 0;
        return true;
    }
    if ((uint64_t)source->null_count == source->length) {
        *out_null_count = vector->rows.length;
        return true;
    }
    if (vector->rows.offset == 0 && vector->rows.length == source->length) {
        *out_null_count = (uint64_t)source->null_count;
        return true;
    }
    return false;
}

int dora_vector_get_null_count(const struct dora_vector* vector, uint64_t* out_null_count) {
    struct vector_image* image = nullptr;
    if (out_null_count == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    const struct dora_vector_view* const source = &image->ownership.view;
    if (dora_vector_try_get_exact_null_count(vector, image, out_null_count)) {
        return DORA_OK;
    }
    if (!vector_layout_derives_nulls(source->layout)) {
        struct dora_vector_view view = {};
        const int status = dora_vector_get_view(vector, &view);
        if (status != DORA_OK) {
            return status;
        }
        *out_null_count = dora_vector_view_count_nulls(&view, (struct dora_range){.length = vector->rows.length});
        return DORA_OK;
    }
    uint64_t count = 0;
    for (uint64_t row = 0; row < vector->rows.length; ++row) {
        bool is_null = false;
        const int status = dora_vector_is_null(vector, row, &is_null);
        if (status != DORA_OK) {
            return status;
        }
        count += is_null ? 1U : 0U;
    }
    *out_null_count = count;
    return DORA_OK;
}
