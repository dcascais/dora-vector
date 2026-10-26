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
#include "extension/storage.h"
#include "json/json.h"
#include "utf8/input.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct tensor_parameters {
    uint32_t rank;
    uint32_t data_child_index;
    uint32_t shape_child_index;
    int32_t uniform_dimensions[];
};

static int
tensor_shape_check(const uint8_t* bytes, struct dora_json_span array, uint32_t cardinality, uint32_t* out_rank) {
    if (bytes[array.begin_byte_offset] != '[') {
        return EINVAL;
    }
    uint32_t rank = 0;
    bool zero = false;
    uint32_t position = array.begin_byte_offset + 1;
    struct dora_json_span item = {};
    while ((position = dora_json_array_next(bytes, array, position, &item)) != array.end_byte_offset) {
        uint64_t dimension = 0;
        const int status = dora_json_parse_unsigned(bytes, item, &dimension);
        if (status != 0) {
            return status;
        }
        zero = zero || dimension == 0;
        ++rank;
    }
    uint64_t product = zero ? 0 : 1;
    if (!zero) {
        position = array.begin_byte_offset + 1;
        while ((position = dora_json_array_next(bytes, array, position, &item)) != array.end_byte_offset) {
            uint64_t dimension = 0;
            const int status = dora_json_parse_unsigned(bytes, item, &dimension);
            if (status != 0) {
                return status;
            }
            if (ckd_mul(&product, product, dimension)) {
                return EOVERFLOW;
            }
        }
    }
    if (product > INT32_MAX) {
        return EOVERFLOW;
    }
    if (product != cardinality) {
        return EINVAL;
    }
    *out_rank = rank;
    return 0;
}

static int tensor_dimension_names_check(const uint8_t* bytes, struct dora_json_span array, uint32_t rank) {
    if (bytes[array.begin_byte_offset] != '[') {
        return EINVAL;
    }
    uint32_t count = 0;
    uint32_t position = array.begin_byte_offset + 1;
    struct dora_json_span item = {};
    while ((position = dora_json_array_next(bytes, array, position, &item)) != array.end_byte_offset) {
        if (bytes[item.begin_byte_offset] != '"' || count == rank) {
            return EINVAL;
        }
        ++count;
    }
    return count == rank ? 0 : EINVAL;
}

static int tensor_permutation_check(const uint8_t* bytes, struct dora_json_span array, uint32_t rank) {
    if (bytes[array.begin_byte_offset] != '[') {
        return EINVAL;
    }
    uint32_t count = 0;
    uint32_t position = array.begin_byte_offset + 1;
    struct dora_json_span item = {};
    while ((position = dora_json_array_next(bytes, array, position, &item)) != array.end_byte_offset) {
        uint64_t dimension = 0;
        int status = dora_json_parse_unsigned(bytes, item, &dimension);
        if (status != 0) {
            return status;
        }
        if (dimension >= rank || count == rank) {
            return EINVAL;
        }
        uint32_t earlier_position = array.begin_byte_offset + 1;
        for (uint32_t earlier = 0; earlier < count; ++earlier) {
            struct dora_json_span previous = {};
            earlier_position = dora_json_array_next(bytes, array, earlier_position, &previous);
            uint64_t previous_dimension = 0;
            status = dora_json_parse_unsigned(bytes, previous, &previous_dimension);
            if (status != 0) {
                return status;
            }
            if (dimension == previous_dimension) {
                return EINVAL;
            }
        }
        ++count;
    }
    return count == rank ? 0 : EINVAL;
}

static int tensor_uniform_shape_check(const uint8_t* bytes, struct dora_json_span array, uint32_t rank) {
    if (bytes[array.begin_byte_offset] != '[') {
        return EINVAL;
    }
    uint32_t count = 0;
    uint32_t position = array.begin_byte_offset + 1;
    struct dora_json_span item = {};
    while ((position = dora_json_array_next(bytes, array, position, &item)) != array.end_byte_offset) {
        if (count == rank) {
            return EINVAL;
        }
        if (bytes[item.begin_byte_offset] != 'n') {
            uint64_t dimension = 0;
            const int status = dora_json_parse_unsigned(bytes, item, &dimension);
            if (status != 0) {
                return status;
            }
            if (dimension > INT32_MAX) {
                return EINVAL;
            }
        }
        ++count;
    }
    return count == rank ? 0 : EINVAL;
}

static int tensor_bind(const struct dora_extension_definition* definition,
                       struct dora_pool pool,
                       const struct dora_schema* storage,
                       struct dora_bytes metadata,
                       struct dora_allocation* out_parameters,
                       struct dora_error*) {
    const bool fixed = definition == &dora_extension_fixed_shape_tensor;
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(storage, &field);
    if (status != DORA_OK || metadata.byte_count > UINT32_MAX ||
        (metadata.byte_count != 0 && metadata.data == nullptr)) {
        return status != DORA_OK ? status : EINVAL;
    }
    uint32_t rank = 0;
    uint32_t data_child_index = 0;
    uint32_t shape_child_index = 1;
    uint32_t list_element_count = 0;
    if (fixed) {
        if (field.layout != DORA_LAYOUT_FIXED_LIST || field.type.id != DORA_TYPE_FIXED_LIST) {
            return EINVAL;
        }
        list_element_count = field.type.parameters.list_element_count;
    } else {
        if (field.layout != DORA_LAYOUT_STRUCT || field.type.id != DORA_TYPE_STRUCT || field.child_count != 2) {
            return EINVAL;
        }
        struct dora_schema data = {};
        struct dora_schema shape = {};
        status = dora_schema_get_child(storage, 0, &data);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&data, &field);
        }
        if (status != DORA_OK || field.name == nullptr) {
            return status != DORA_OK ? status : EINVAL;
        }
        if (strcmp(field.name, "shape") == 0) {
            data_child_index = 1;
            shape_child_index = 0;
            data = (struct dora_schema){};
            status = dora_schema_get_child(storage, data_child_index, &data);
        }
        if (status == DORA_OK) {
            status = dora_schema_get_field(&data, &field);
        }
        if (status != DORA_OK || field.name == nullptr || strcmp(field.name, "data") != 0) {
            return status != DORA_OK ? status : EINVAL;
        }
        status = extension_resolve_storage_field(&data, &field);
        if (status != DORA_OK || field.type.id != DORA_TYPE_LIST || field.layout != DORA_LAYOUT_LIST) {
            return status != DORA_OK ? status : EINVAL;
        }
        status = dora_schema_get_child(storage, shape_child_index, &shape);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&shape, &field);
        }
        if (status != DORA_OK || field.name == nullptr || strcmp(field.name, "shape") != 0) {
            return status != DORA_OK ? status : EINVAL;
        }
        status = extension_resolve_storage_field(&shape, &field);
        if (status != DORA_OK || field.type.id != DORA_TYPE_FIXED_LIST || field.layout != DORA_LAYOUT_FIXED_LIST) {
            return status != DORA_OK ? status : EINVAL;
        }
        rank = field.type.parameters.list_element_count;
        struct dora_schema dimension = {};
        status = dora_schema_get_child(&shape, 0, &dimension);
        if (status == DORA_OK) {
            status = extension_resolve_storage_field(&dimension, &field);
        }
        if (status != DORA_OK || field.type.id != DORA_TYPE_INT32) {
            return status != DORA_OK ? status : EINVAL;
        }
    }
    struct dora_json_span values[4] = {};
    uint32_t present = 0;
    if (metadata.byte_count == 0) {
        if (fixed) {
            return EINVAL;
        }
    } else {
        status = dora_json_validate(pool, metadata, DORA_UTF8_CHECKED);
        if (status != DORA_OK) {
            return status;
        }
        const char* const names[] =
            {fixed ? "shape" : nullptr, "dim_names", "permutation", fixed ? nullptr : "uniform_shape"};
        status = dora_json_object_members(metadata.data, (uint32_t)metadata.byte_count, names, 4, values, &present);
        if (status != DORA_OK) {
            return status;
        }
    }
    if (fixed) {
        if (bitmap_word_bit(present, 0) == 0) {
            return EINVAL;
        }
        status = tensor_shape_check(metadata.data, values[0], list_element_count, &rank);
    }
    if (status == DORA_OK && bitmap_word_bit(present, 1) != 0) {
        status = tensor_dimension_names_check(metadata.data, values[1], rank);
    }
    if (status == DORA_OK && bitmap_word_bit(present, 2) != 0) {
        status = tensor_permutation_check(metadata.data, values[2], rank);
    }
    if (status == DORA_OK && !fixed && bitmap_word_bit(present, 3) != 0) {
        status = tensor_uniform_shape_check(metadata.data, values[3], rank);
    }
    if (status != DORA_OK || fixed) {
        return status;
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, rank, sizeof(int32_t)) || ckd_add(&bytes, bytes, sizeof(struct tensor_parameters))) {
        return EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    status = dora_pool_allocate(pool, bytes, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    struct tensor_parameters* const parameters = allocation.address;
    *parameters = (struct tensor_parameters){
        .rank = rank,
        .data_child_index = data_child_index,
        .shape_child_index = shape_child_index
    };
    uint32_t position = values[3].begin_byte_offset + 1;
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
        parameters->uniform_dimensions[dimension] = -1;
        if (bitmap_word_bit(present, 3) != 0) {
            struct dora_json_span item = {};
            position = dora_json_array_next(metadata.data, values[3], position, &item);
            if (metadata.data[item.begin_byte_offset] != 'n') {
                uint64_t value = 0;
                status = dora_json_parse_unsigned(metadata.data, item, &value);
                if (status != DORA_OK) {
                    dora_allocation_release(&allocation);
                    return status;
                }
                parameters->uniform_dimensions[dimension] = (int32_t)value;
            }
        }
    }
    *out_parameters = allocation;
    return DORA_OK;
}

static int tensor_validate(const struct dora_extension_definition*,
                           struct dora_pool pool,
                           const struct dora_allocation* parameters,
                           const struct dora_vector* storage,
                           uint64_t* in_out_validity,
                           struct dora_error*) {
    if (parameters == nullptr || parameters->capacity_bytes < sizeof(struct tensor_parameters)) {
        return EINVAL;
    }
    const struct tensor_parameters* const tensor = parameters->address;
    struct dora_vector plain = {};
    struct dora_vector owned = {};
    struct dora_vector data = {};
    struct dora_vector shape = {};
    struct dora_vector data_plain = {};
    struct dora_vector data_owned = {};
    struct dora_vector shape_plain = {};
    struct dora_vector shape_owned = {};
    struct dora_vector dimensions = {};
    struct dora_vector_view view = {};
    struct dora_vector_view data_view = {};
    struct dora_vector_view shape_view = {};
    int status = extension_prepare_storage(pool, storage, &plain, &owned, &view);
    if (status == DORA_OK && view.length != 0 && in_out_validity == nullptr) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child(&plain, tensor->data_child_index, &data);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child(&plain, tensor->shape_child_index, &shape);
    }
    if (status == DORA_OK) {
        status = extension_prepare_storage(pool, &data, &data_plain, &data_owned, &data_view);
    }
    if (status == DORA_OK) {
        status = extension_prepare_storage(pool, &shape, &shape_plain, &shape_owned, &shape_view);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child(&shape_plain, 0, &dimensions);
    }
    bool invalid_content = false;
    for (uint64_t row = 0; status == DORA_OK && row < view.length; ++row) {
        if (!dora_bitmap_test((const uint8_t*)in_out_validity, row)) {
            continue;
        }
        bool invalid = dora_vector_view_is_null(&data_view, row) || dora_vector_view_is_null(&shape_view, row);
        struct dora_range data_rows = {};
        struct dora_range shape_rows = {};
        if (!invalid) {
            data_rows = dora_vector_view_get_elements(&data_view, row);
            shape_rows = dora_vector_view_get_elements(&shape_view, row);
        }
        invalid = invalid || shape_rows.length != tensor->rank;
        uint64_t product = 1;
        bool zero = false;
        bool overflow = false;
        for (uint32_t dimension = 0; !invalid && dimension < tensor->rank; ++dimension) {
            int32_t size = 0;
            bool is_null = false;
            status = dora_vector_read_i32(&dimensions, shape_rows.offset + dimension, &size, &is_null);
            if (status != DORA_OK) {
                break;
            }
            const int32_t expected = tensor->uniform_dimensions[dimension];
            invalid = invalid || is_null || size < 0 || (expected >= 0 && size != expected);
            zero = zero || size == 0;
            if (!invalid && !zero && !overflow) {
                overflow = ckd_mul(&product, product, (uint64_t)size);
            }
        }
        invalid = invalid || (zero ? 0 : product) != data_rows.length || (!zero && overflow);
        if (status == DORA_OK && invalid) {
            bitmap_clear(in_out_validity, row);
            invalid_content = true;
        }
    }
    dora_vector_release(&shape_owned);
    dora_vector_release(&data_owned);
    dora_vector_release(&owned);
    return status != DORA_OK ? status : invalid_content ? ERANGE : DORA_OK;
}

const struct dora_extension_definition dora_extension_fixed_shape_tensor = {
    .name = {.data = (const uint8_t*)"arrow.fixed_shape_tensor", .byte_count = sizeof("arrow.fixed_shape_tensor") - 1},
    .bind = tensor_bind
};

const struct dora_extension_definition dora_extension_variable_shape_tensor = {
    .name =
        {.data = (const uint8_t*)"arrow.variable_shape_tensor",
         .byte_count = sizeof("arrow.variable_shape_tensor") - 1},
    .bind = tensor_bind,
    .validate = tensor_validate
};
