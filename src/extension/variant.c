// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "extension/variant.h"

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
#include "numeric/decimal.h"
#include "types/types_internal.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static uint64_t byte_unsigned(const uint8_t* bytes, uint32_t width) {
    uint64_t value = 0;
    for (uint32_t index = 0; index < width; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

static int variant_unsigned(struct dora_bytes bytes, uint64_t offset, uint32_t width, uint64_t* out_value) {
    if (width == 0 || width > 8 || offset > bytes.byte_count || width > bytes.byte_count - offset) {
        return EINVAL;
    }
    const uint8_t* const data = bytes.data;
    *out_value = byte_unsigned(data + offset, width);
    return 0;
}

static int
variant_decimal(struct dora_bytes value, uint64_t begin, uint32_t primitive, const struct dora_i256 prepared[3]) {
    const uint8_t* const bytes = value.data;
    const uint8_t scale = bytes[begin + 1];
    if (scale > DECIMAL128_MAX_PRECISION) {
        return EINVAL;
    }
    uint64_t low;
    const uint32_t width = primitive == 8 ? 4 : 8;
    int status = variant_unsigned(value, begin + 2, width, &low);
    if (status != 0) {
        return status;
    }
    const uint32_t index = primitive == 8 ? 0 : primitive == 9 ? 1 : 2;
    const struct dora_i256 bound = prepared[index];
    bool admitted = false;
    if (primitive == 8) {
        const uint32_t bits = (uint32_t)low;
        int32_t coefficient;
        memcpy(&coefficient, &bits, sizeof(coefficient));
        admitted = decimal_bound_admits_i32(coefficient, bound);
    } else if (primitive == 9) {
        int64_t coefficient;
        memcpy(&coefficient, &low, sizeof(coefficient));
        admitted = decimal_bound_admits_i64(coefficient, bound);
    } else {
        uint64_t high;
        status = variant_unsigned(value, begin + 10, 8, &high);
        if (status != 0) {
            return status;
        }
        admitted = decimal_bound_admits_i128((struct dora_i128){.limbs = {low, high}}, bound);
    }
    return admitted ? 0 : EINVAL;
}

static int variant_metadata_header(struct dora_bytes metadata, uint32_t* out_keys, uint64_t* out_data) {
    if (!bytes_are_well_formed(metadata) || metadata.byte_count == 0) {
        return EINVAL;
    }
    if ((metadata.data[0] & 15U) != 1) {
        return ENOTSUP;
    }
    const uint32_t width = ((uint32_t)metadata.data[0] >> 6U) + 1;
    uint64_t keys = 0;
    const int status = variant_unsigned(metadata, 1, width, &keys);
    if (status != 0) {
        return status;
    }
    const uint64_t offsets = 1ULL + width;
    if (keys + 1 > (metadata.byte_count - offsets) / width) {
        return EINVAL;
    }
    const uint64_t data = offsets + (keys + 1) * width;
    if (byte_unsigned(metadata.data + offsets, width) != 0 ||
        byte_unsigned(metadata.data + offsets + keys * width, width) != metadata.byte_count - data) {
        return EINVAL;
    }
    *out_keys = (uint32_t)keys;
    *out_data = data;
    return 0;
}

static int
variant_key(struct dora_bytes metadata, uint32_t keys, uint64_t index, struct dora_bytes* out_key, uint64_t data) {
    if (index >= keys) {
        return EINVAL;
    }
    const uint32_t width = ((uint32_t)metadata.data[0] >> 6U) + 1;
    const uint8_t* const offsets = metadata.data + 1 + width + index * width;
    const uint64_t begin = byte_unsigned(offsets, width);
    const uint64_t end = byte_unsigned(offsets + width, width);
    if (begin > end || end > metadata.byte_count - data) {
        return EINVAL;
    }
    *out_key = (struct dora_bytes){.data = metadata.data + data + begin, .byte_count = end - begin};
    return 0;
}

static int variant_key_order(struct dora_bytes left, struct dora_bytes right) {
    const size_t common = left.byte_count < right.byte_count ? left.byte_count : right.byte_count;
    const int order = common == 0 ? 0 : memcmp(left.data, right.data, common);
    return order != 0 ? order : (left.byte_count > right.byte_count) - (left.byte_count < right.byte_count);
}

int dora_variant_metadata_check(struct dora_bytes metadata) {
    uint32_t keys = 0;
    uint64_t data = 0;
    int status = variant_metadata_header(metadata, &keys, &data);
    struct dora_bytes previous = {};
    for (uint64_t index = 0; status == 0 && index < keys; ++index) {
        struct dora_bytes key = {};
        status = variant_key(metadata, keys, index, &key, data);
        if (status == 0) {
            status = dora_utf8_validate_span(key, nullptr);
        }
        if (status == 0 && (metadata.data[0] & 16U) != 0 && index != 0 && variant_key_order(previous, key) >= 0) {
            status = EINVAL;
        }
        previous = key;
    }
    return status;
}

static int variant_node_check(struct dora_bytes metadata,
                              uint64_t keys,
                              uint64_t key_data,
                              struct dora_bytes value,
                              uint64_t begin,
                              uint64_t limit,
                              const struct dora_i256 decimal_bounds[3],
                              uint64_t* out_end) {
    if (begin >= limit || limit > value.byte_count) {
        return EINVAL;
    }
    const uint8_t basic = value.data[begin] & 3U;
    const uint32_t primitive = value.data[begin] >> 2U;
    uint64_t end = begin + 1;
    if (basic == 2 || basic == 3) {
        const uint32_t count_width = basic == 2 ? ((primitive & 16U) ? 4 : 1) : ((primitive & 4U) ? 4 : 1);
        const uint32_t id_width = basic == 2 ? ((primitive >> 2U) & 3U) + 1 : 0;
        const uint32_t offset_width = (primitive & 3U) + 1;
        if (count_width > limit - end) {
            return EINVAL;
        }
        const uint64_t count = byte_unsigned(value.data + end, count_width);
        const uint64_t ids = end + count_width;
        if (id_width != 0 && count > (limit - ids) / id_width) {
            return EINVAL;
        }
        const uint64_t offsets = ids + count * id_width;
        if (count + 1 > (limit - offsets) / offset_width) {
            return EINVAL;
        }
        const uint64_t data = offsets + (count + 1) * offset_width;
        const uint64_t extent = byte_unsigned(value.data + offsets + count * offset_width, offset_width);
        if (extent > limit - data || (count == 0 && extent != 0)) {
            return EINVAL;
        }
        struct dora_bytes previous = {};
        for (uint64_t index = 0; index < count; ++index) {
            if (byte_unsigned(value.data + offsets + index * offset_width, offset_width) >= extent) {
                return EINVAL;
            }
            if (basic == 2) {
                struct dora_bytes key = {};
                const uint64_t id = byte_unsigned(value.data + ids + index * id_width, id_width);
                const int status = variant_key(metadata, keys, id, &key, key_data);
                if (status != 0 || (index != 0 && variant_key_order(previous, key) >= 0)) {
                    return status != 0 ? status : EINVAL;
                }
                previous = key;
            }
        }
        end = data + extent;
    } else {
        uint64_t payload = 0;
        if (basic == 1) {
            payload = primitive;
        } else {
            switch (primitive) {
            case 0:
            case 1:
            case 2:
            case 15:
            case 16:
                break;
            case 3:
                payload = 1;
                break;
            case 4:
                payload = 2;
                break;
            case 5:
            case 11:
            case 14:
                payload = 4;
                break;
            case 6:
            case 7:
            case 12:
            case 13:
            case 17:
            case 18:
            case 19:
                payload = 8;
                break;
            case 8:
                payload = 5;
                break;
            case 9:
                payload = 9;
                break;
            case 10:
                payload = 17;
                break;
            case 20:
                payload = 16;
                break;
            default:
                return ENOTSUP;
            }
            if (primitive == 15 || primitive == 16) {
                if (limit - end < 4) {
                    return EINVAL;
                }
                payload = 4 + byte_unsigned(value.data + end, 4);
            }
        }
        if (payload > limit - end) {
            return EINVAL;
        }
        if (basic == 1 || primitive == 16) {
            const uint64_t offset = basic == 1 ? 0 : 4;
            const int status = dora_utf8_validate_span(
                (struct dora_bytes){.data = value.data + end + offset, .byte_count = payload - offset},
                nullptr);
            if (status != 0) {
                return status;
            }
        }
        end += payload;
        if (basic == 0 && primitive >= 8 && primitive <= 10) {
            const int status = variant_decimal(value, begin, primitive, decimal_bounds);
            if (status != 0) {
                return status;
            }
        }
    }
    *out_end = end;
    return 0;
}

int dora_variant_value_check(struct dora_bytes metadata,
                             struct dora_bytes value,
                             void* scratch,
                             size_t scratch_capacity_bytes,
                             const struct dora_i256 decimal_bounds[3]) {
    if (!bytes_are_well_formed(metadata) || !bytes_are_well_formed(value) || value.byte_count == 0) {
        return EINVAL;
    }
    uint32_t keys = 0;
    uint64_t key_data = 0;
    const int header_status = variant_metadata_header(metadata, &keys, &key_data);
    if (header_status != 0) {
        return header_status;
    }
    size_t frame_offset = 0;
    size_t frame_bytes = 0;
    size_t required = 0;
    if (ckd_mul(&frame_offset, bitmap_words_for_bits(value.byte_count), sizeof(uint64_t)) ||
        ckd_mul(&frame_bytes, value.byte_count, sizeof(struct variant_frame)) ||
        ckd_add(&required, frame_offset, frame_bytes)) {
        return EOVERFLOW;
    }
    if (scratch_capacity_bytes < required) {
        return ERANGE;
    }
    if (scratch == nullptr || (uintptr_t)scratch % alignof(uint64_t) != 0) {
        return EINVAL;
    }
    memset(scratch, 0, frame_offset);
    struct variant_frame* const frames = (struct variant_frame*)((uint8_t*)scratch + frame_offset);
    size_t depth = 0;
    uint64_t begin = 0;
    uint64_t limit = value.byte_count;
    while (true) {
        uint64_t end = 0;
        const int status = variant_node_check(metadata, keys, key_data, value, begin, limit, decimal_bounds, &end);
        if (status != 0) {
            return status;
        }
        if (depth == 0 && begin == 0 && end != value.byte_count) {
            return EINVAL;
        }
        if (!dora_bitmap_test(scratch, begin)) {
            const uint8_t basic = value.data[begin] & 3U;
            if (basic == 2 || basic == 3) {
                if (depth >= (scratch_capacity_bytes - frame_offset) / sizeof(struct variant_frame)) {
                    return ERANGE;
                }
                frames[depth++] = (struct variant_frame){.begin_byte_offset = begin, .end_byte_offset = end};
            } else {
                bitmap_set(scratch, begin);
            }
        }
        while (depth != 0) {
            struct variant_frame* const frame = &frames[depth - 1];
            const uint32_t primitive = value.data[frame->begin_byte_offset] >> 2U;
            const bool object = (value.data[frame->begin_byte_offset] & 3U) == 2;
            const uint32_t count_width = object ? ((primitive & 16U) ? 4 : 1) : ((primitive & 4U) ? 4 : 1);
            const uint32_t id_width = object ? ((primitive >> 2U) & 3U) + 1 : 0;
            const uint32_t offset_width = (primitive & 3U) + 1;
            const uint64_t count = byte_unsigned(value.data + frame->begin_byte_offset + 1, count_width);
            const uint64_t offsets = frame->begin_byte_offset + 1 + count_width + count * id_width;
            const uint64_t data = offsets + (count + 1) * offset_width;
            if (frame->next_child_index == count) {
                bitmap_set(scratch, frame->begin_byte_offset);
                --depth;
                continue;
            }
            const uint64_t offset =
                byte_unsigned(value.data + offsets + frame->next_child_index++ * offset_width, offset_width);
            if (offset >= frame->end_byte_offset - data) {
                return EINVAL;
            }
            begin = data + offset;
            limit = frame->end_byte_offset;
            break;
        }
        if (depth == 0) {
            return 0;
        }
    }
}

int dora_variant_object_lookup(struct dora_bytes metadata,
                               const struct dora_bytes* value,
                               struct dora_bytes key,
                               bool* out_found) {
    const uint32_t primitive = value->data[0] >> 2U;
    const uint32_t count_width = (primitive & 16U) ? 4 : 1;
    const uint32_t id_width = ((primitive >> 2U) & 3U) + 1;
    const uint32_t metadata_width = ((uint32_t)metadata.data[0] >> 6U) + 1;
    const uint64_t keys = byte_unsigned(metadata.data + 1, metadata_width);
    const uint64_t key_data = 1ULL + metadata_width + (keys + 1) * metadata_width;
    const uint64_t count = byte_unsigned(value->data + 1, count_width);
    const uint8_t* const ids = value->data + 1 + count_width;
    uint64_t first = 0;
    uint64_t last = count;
    *out_found = false;
    while (first < last) {
        const uint64_t middle = first + (last - first) / 2;
        const uint64_t id = byte_unsigned(ids + middle * id_width, id_width);
        struct dora_bytes candidate = {};
        const int status = variant_key(metadata, keys, id, &candidate, key_data);
        if (status != 0) {
            return status;
        }
        const int order = variant_key_order(candidate, key);
        if (order < 0) {
            first = middle + 1;
        } else if (order > 0) {
            last = middle;
        } else {
            *out_found = true;
            break;
        }
    }
    return 0;
}

static constexpr int32_t VARIANT_DECIMAL_PRECISIONS[3] = {
    DECIMAL32_MAX_PRECISION,
    DECIMAL64_MAX_PRECISION,
    DECIMAL128_MAX_PRECISION,
};

static int variant_decimal_bounds_prepare(struct dora_i256 out_bounds[3]) {
    int status = 0;
    for (uint32_t index = 0; status == 0 && index < 3; ++index) {
        status = dora_decimal_bound_prepare((struct dora_decimal_type){.precision = VARIANT_DECIMAL_PRECISIONS[index]},
                                            VARIANT_DECIMAL_PRECISIONS[index],
                                            &out_bounds[index]);
    }
    return status;
}

struct variant_role {
    uint32_t parent_role_index;
    uint32_t child_index;
    uint32_t value_child_index;
    uint32_t typed_child_index;
    uint32_t first_child_role_index;
    uint32_t last_child_role_index;
    uint32_t previous_sibling_role_index;
    uint32_t subtree_end_role_index;
    enum dora_layout layout;
    size_t name_byte_count;
};

struct variant_parameters {
    uint32_t metadata_child_index;
    uint32_t role_count;
    struct variant_role roles[];
};

struct variant_schema_frame {
    struct dora_schema schema;
    struct dora_schema typed;
    struct variant_role role;
    uint32_t child_count;
    uint32_t child;
};

static bool variant_field_is_primitive(const struct dora_schema_field* field) {
    if (field->type.extension_name.byte_count != 0) {
        return field->type.extension_name.byte_count == sizeof("arrow.uuid") - 1 &&
               memcmp(field->type.extension_name.data, "arrow.uuid", sizeof("arrow.uuid") - 1) == 0 &&
               field->type.id == DORA_TYPE_FIXED_BINARY && field->type.parameters.byte_width == 16;
    }
    switch (field->type.id) {
    case DORA_TYPE_NULL:
    case DORA_TYPE_BOOLEAN:
    case DORA_TYPE_INT8:
    case DORA_TYPE_INT16:
    case DORA_TYPE_INT32:
    case DORA_TYPE_INT64:
    case DORA_TYPE_UINT8:
    case DORA_TYPE_UINT16:
    case DORA_TYPE_UINT32:
    case DORA_TYPE_FLOAT32:
    case DORA_TYPE_FLOAT64:
    case DORA_TYPE_BINARY:
    case DORA_TYPE_UTF8:
    case DORA_TYPE_DATE32:
        return true;
    case DORA_TYPE_DECIMAL:
        return field->type.parameters.decimal.bit_width <= 128;
    case DORA_TYPE_TIME:
        return field->type.parameters.unit >= DORA_TIME_UNIT_MICROSECOND;
    case DORA_TYPE_TIMESTAMP:
        return field->type.parameters.timestamp.unit >= DORA_TIME_UNIT_MICROSECOND &&
               (field->type.parameters.timestamp.timezone == nullptr ||
                field->type.parameters.timestamp.timezone[0] == 0 ||
                strcmp(field->type.parameters.timestamp.timezone, "UTC") == 0);
    default:
        return false;
    }
}

static int variant_schema_roles(const struct dora_schema* schema,
                                bool root,
                                uint32_t* out_metadata_child_index,
                                struct dora_schema* out_typed,
                                struct variant_role* role) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status != DORA_OK || field.type.id != DORA_TYPE_STRUCT || field.layout != DORA_LAYOUT_STRUCT) {
        return status != DORA_OK ? status : EINVAL;
    }
    const uint32_t count = field.child_count;
    bool metadata = false;
    bool value = false;
    bool typed = false;
    for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
        struct dora_schema child = {};
        status = dora_schema_get_child(schema, ordinal, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status != DORA_OK || field.name == nullptr) {
            return status != DORA_OK ? status : EINVAL;
        }
        if (root && strcmp(field.name, "metadata") == 0) {
            if (metadata || (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
                return EINVAL;
            }
            metadata = true;
            *out_metadata_child_index = ordinal;
            status = extension_resolve_storage_field(&child, &field);
            if (status != DORA_OK || field.type.id != DORA_TYPE_BINARY) {
                return status != DORA_OK ? status : EINVAL;
            }
        } else if (strcmp(field.name, "value") == 0) {
            if (value || field.type.id != DORA_TYPE_BINARY || field.layout == DORA_LAYOUT_DICTIONARY ||
                field.layout == DORA_LAYOUT_RUN_END) {
                return EINVAL;
            }
            value = true;
            role->value_child_index = ordinal;
        } else if (strcmp(field.name, "typed_value") == 0) {
            if (typed) {
                return EINVAL;
            }
            typed = true;
            role->typed_child_index = ordinal;
            *out_typed = child;
        } else {
            return EINVAL;
        }
    }
    return (root && !metadata) || (!value && !typed) ? EINVAL : DORA_OK;
}

static int variant_bind(const struct dora_extension_definition*,
                        struct dora_pool pool,
                        const struct dora_schema* storage,
                        struct dora_bytes metadata,
                        struct dora_allocation* out_parameters,
                        struct dora_error*) {
    if (metadata.byte_count != 0) {
        return EINVAL;
    }
    struct dora_allocation stack = {};
    int status = dora_pool_allocate(pool, sizeof(struct variant_schema_frame), &stack);
    if (status != DORA_OK) {
        return status;
    }
    uint32_t count = 1;
    uint32_t cursor = 0;
    uint32_t metadata_child_index = 0;
    ((struct variant_schema_frame*)stack.address)[0] = (struct variant_schema_frame){
        .schema = dora_schema_borrow(storage),
        .role = {.parent_role_index = UINT32_MAX, .value_child_index = UINT32_MAX, .typed_child_index = UINT32_MAX}
    };
    while (status == DORA_OK) {
        struct variant_schema_frame* frame = &((struct variant_schema_frame*)stack.address)[cursor];
        status = variant_schema_roles(&frame->schema, cursor == 0, &metadata_child_index, &frame->typed, &frame->role);
        struct dora_schema_field field = {};
        if (status == DORA_OK) {
            status = dora_schema_get_field(&frame->schema, &field);
            frame->role.name_byte_count = field.name == nullptr ? 0 : strlen(field.name);
        }
        if (status == DORA_OK && frame->typed.handle.slot_index != 0) {
            status = dora_schema_get_field(&frame->typed, &field);
            if (status == DORA_OK && (field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END)) {
                status = EINVAL;
            }
            if (status == DORA_OK && !variant_field_is_primitive(&field)) {
                if (field.type.extension_name.byte_count != 0 ||
                    (field.layout != DORA_LAYOUT_STRUCT && field.layout != DORA_LAYOUT_LIST &&
                     field.layout != DORA_LAYOUT_LARGE_LIST && field.layout != DORA_LAYOUT_LIST_VIEW)) {
                    status = EINVAL;
                } else {
                    frame->role.layout = field.layout;
                    frame->child_count = field.child_count;
                    frame->role.first_child_role_index = count;
                }
            }
        }
        while (status == DORA_OK && frame->child == frame->child_count) {
            frame->role.subtree_end_role_index = count;
            if (frame->role.parent_role_index == UINT32_MAX) {
                break;
            }
            cursor = frame->role.parent_role_index;
            frame = &((struct variant_schema_frame*)stack.address)[cursor];
        }
        if (status != DORA_OK || (cursor == 0 && frame->child == frame->child_count)) {
            break;
        }
        struct dora_schema child = {};
        const uint32_t ordinal = frame->child++;
        const uint32_t previous = frame->role.last_child_role_index;
        frame->role.last_child_role_index = count;
        status = dora_schema_get_child(&frame->typed, ordinal, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == DORA_OK && (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
            status = EINVAL;
        }
        size_t bytes = 0;
        if (status == DORA_OK &&
            (count == UINT32_MAX || ckd_mul(&bytes, (size_t)count + 1, sizeof(struct variant_schema_frame)))) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK) {
            status = dora_allocation_reserve(pool, bytes, (size_t)count * sizeof(struct variant_schema_frame), &stack);
        }
        if (status == DORA_OK) {
            ((struct variant_schema_frame*)stack.address)[count] = (struct variant_schema_frame){
                .schema = child,
                .role = {
                    .parent_role_index = cursor,
                    .child_index = ordinal,
                    .previous_sibling_role_index = previous,
                    .value_child_index = UINT32_MAX,
                    .typed_child_index = UINT32_MAX
                }
            };
            cursor = count++;
        }
    }
    size_t bytes = 0;
    if (status == DORA_OK && (ckd_mul(&bytes, count, sizeof(struct variant_role)) ||
                              ckd_add(&bytes, bytes, sizeof(struct variant_parameters)))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, out_parameters);
    }
    if (status == DORA_OK) {
        struct variant_parameters* const parameters = out_parameters->address;
        parameters->metadata_child_index = metadata_child_index;
        parameters->role_count = count;
        for (uint32_t index = 0; index < count; ++index) {
            parameters->roles[index] = ((const struct variant_schema_frame*)stack.address)[index].role;
        }
    }
    dora_allocation_release(&stack);
    return status;
}

struct variant_packet_role {
    struct dora_vector_view wrapper;
    struct dora_vector_view binary;
    struct dora_vector_view typed;
    struct dora_bytes name;
    struct dora_bytes value;
    uint64_t row;
    uint64_t cursor;
    uint64_t end;
    bool is_binary_null;
    bool is_typed_null;
};

static int variant_packet_bind(const struct dora_vector* storage,
                               const struct variant_parameters* parameters,
                               struct variant_packet_role* packet) {
    int status = DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < parameters->role_count; ++index) {
        struct dora_vector wrapper = dora_vector_borrow(storage);
        uint32_t cursor = 0;
        while (status == DORA_OK && cursor != index) {
            const struct variant_role* const role = &parameters->roles[cursor];
            struct dora_vector typed = {};
            status = dora_vector_get_child(&wrapper, role->typed_child_index, &typed);
            cursor = role->first_child_role_index;
            while (parameters->roles[cursor].subtree_end_role_index <= index) {
                cursor = parameters->roles[cursor].subtree_end_role_index;
            }
            if (status == DORA_OK) {
                struct dora_vector child = {};
                status = dora_vector_get_child(&typed, parameters->roles[cursor].child_index, &child);
                if (status == DORA_OK) {
                    wrapper = child;
                }
            }
        }
        const struct variant_role* const role = &parameters->roles[index];
        struct variant_packet_role* const bound = &packet[index];
        *bound = (struct variant_packet_role){};
        if (status == DORA_OK) {
            status = dora_vector_get_view(&wrapper, &bound->wrapper);
        }
        struct dora_vector child = {};
        if (status == DORA_OK && role->value_child_index != UINT32_MAX) {
            status = dora_vector_get_child(&wrapper, role->value_child_index, &child);
            if (status == DORA_OK) {
                status = dora_vector_get_view(&child, &bound->binary);
            }
        }
        if (status == DORA_OK && role->typed_child_index != UINT32_MAX) {
            child = (struct dora_vector){};
            status = dora_vector_get_child(&wrapper, role->typed_child_index, &child);
            if (status == DORA_OK) {
                status = dora_vector_get_view(&child, &bound->typed);
            }
        }
        if (status == DORA_OK && index != 0 &&
            parameters->roles[role->parent_role_index].layout == DORA_LAYOUT_STRUCT) {
            struct dora_schema schema = {};
            struct dora_schema_field field = {};
            status = dora_vector_retain_schema(&wrapper, &schema);
            if (status == DORA_OK) {
                status = dora_schema_get_field(&schema, &field);
            }
            if (status == DORA_OK) {
                bound->name = (struct dora_bytes){(const uint8_t*)field.name, role->name_byte_count};
            }
            dora_schema_release(&schema);
        }
    }
    return status;
}

static int variant_validate_binary(struct dora_pool pool,
                                   struct dora_bytes metadata,
                                   struct dora_bytes value,
                                   struct dora_allocation* scratch,
                                   const struct dora_i256 decimal_bounds[3]) {
    size_t bytes = 0;
    size_t frames = 0;
    if (ckd_mul(&bytes, bitmap_words_for_bits(value.byte_count), sizeof(uint64_t)) ||
        ckd_mul(&frames, value.byte_count, sizeof(struct variant_frame)) || ckd_add(&bytes, bytes, frames)) {
        return EOVERFLOW;
    }
    const int status = dora_allocation_reserve(pool, bytes, 0, scratch);
    if (status != DORA_OK) {
        return status;
    }
    return dora_variant_value_check(metadata, value, scratch->address, scratch->capacity_bytes, decimal_bounds) ==
                   DORA_OK
               ? DORA_OK
               : EILSEQ;
}

static int variant_role_read(const struct variant_role* role, struct variant_packet_role* bound, uint64_t row) {
    if (dora_vector_view_is_null(&bound->wrapper, row)) {
        return EILSEQ;
    }
    bound->row = row;
    bound->is_binary_null = role->value_child_index == UINT32_MAX || dora_vector_view_is_null(&bound->binary, row);
    bound->is_typed_null = role->typed_child_index == UINT32_MAX || dora_vector_view_is_null(&bound->typed, row);
    bound->value = bound->is_binary_null ? (struct dora_bytes){} : dora_vector_view_get_bytes(&bound->binary, row);
    return !bound->is_binary_null && bound->value.data == nullptr ? EILSEQ : DORA_OK;
}

static int variant_validate_row(struct dora_pool pool,
                                const struct variant_parameters* parameters,
                                struct variant_packet_role* packet,
                                uint64_t row,
                                struct dora_allocation* scratch,
                                struct dora_bytes metadata,
                                const struct dora_i256 decimal_bounds[3]) {
    uint32_t index = 0;
    packet[0].row = row;
    for (;;) {
        const struct variant_role* const role = &parameters->roles[index];
        struct variant_packet_role* const bound = &packet[index];
        int status = variant_role_read(role, bound, bound->row);
        if (status != DORA_OK) {
            return status;
        }
        if (!bound->is_binary_null) {
            status = variant_validate_binary(pool, metadata, bound->value, scratch, decimal_bounds);
            if (status != DORA_OK) {
                return status;
            }
        }
        if (bound->is_typed_null) {
            if (bound->is_binary_null &&
                (index == 0 || parameters->roles[role->parent_role_index].layout != DORA_LAYOUT_STRUCT)) {
                return EILSEQ;
            }
        } else {
            if (!bound->is_binary_null && (role->layout != DORA_LAYOUT_STRUCT || bound->value.byte_count == 0 ||
                                           (bound->value.data[0] & 3U) != 2)) {
                return EILSEQ;
            }
            if (role->first_child_role_index != 0 && role->first_child_role_index < role->subtree_end_role_index) {
                if (role->layout == DORA_LAYOUT_STRUCT) {
                    if (!bound->is_binary_null) {
                        for (uint32_t child = role->first_child_role_index; child < role->subtree_end_role_index;
                             child = parameters->roles[child].subtree_end_role_index) {
                            struct variant_packet_role* const child_bound = &packet[child];
                            status = variant_role_read(&parameters->roles[child], child_bound, bound->row);
                            if (status != DORA_OK) {
                                return status;
                            }
                            if (!child_bound->is_binary_null || !child_bound->is_typed_null) {
                                bool found = false;
                                status = dora_variant_object_lookup(metadata, &bound->value, child_bound->name, &found);
                                if (status != DORA_OK || found) {
                                    return EILSEQ;
                                }
                            }
                        }
                    }
                    packet[role->last_child_role_index].row = bound->row;
                    index = role->last_child_role_index;
                    continue;
                }
                const struct dora_range rows = dora_vector_view_get_elements(&bound->typed, bound->row);
                bound->cursor = rows.offset + rows.length;
                bound->end = rows.offset;
                if (bound->cursor > bound->end) {
                    packet[role->first_child_role_index].row = --bound->cursor;
                    index = role->first_child_role_index;
                    continue;
                }
            }
        }
        for (;;) {
            const struct variant_role* const completed = &parameters->roles[index];
            if (completed->parent_role_index == UINT32_MAX) {
                return DORA_OK;
            }
            const uint32_t parent = completed->parent_role_index;
            const struct variant_role* const parent_role = &parameters->roles[parent];
            struct variant_packet_role* const parent_bound = &packet[parent];
            if (parent_role->layout == DORA_LAYOUT_STRUCT && completed->previous_sibling_role_index != 0) {
                index = completed->previous_sibling_role_index;
                packet[index].row = parent_bound->row;
                break;
            }
            if (parent_role->layout != DORA_LAYOUT_STRUCT && parent_bound->cursor > parent_bound->end) {
                index = parent_role->first_child_role_index;
                packet[index].row = --parent_bound->cursor;
                break;
            }
            index = parent;
        }
    }
}

static int variant_validate(const struct dora_extension_definition*,
                            struct dora_pool pool,
                            const struct dora_allocation* parameters,
                            const struct dora_vector* storage,
                            uint64_t* in_out_validity,
                            struct dora_error*) {
    if (parameters == nullptr || parameters->capacity_bytes < sizeof(struct variant_parameters)) {
        return EINVAL;
    }
    const struct variant_parameters* const variant = parameters->address;
    struct dora_vector plain = {};
    struct dora_vector owned = {};
    struct dora_vector_view view = {};
    int status = extension_prepare_storage(pool, storage, &plain, &owned, &view);
    if (status == DORA_OK && view.length != 0 && in_out_validity == nullptr) {
        status = EINVAL;
    }
    struct dora_vector metadata_vector = {};
    if (status == DORA_OK) {
        status = dora_vector_get_child(&plain, variant->metadata_child_index, &metadata_vector);
    }
    struct dora_allocation packet = {};
    struct dora_allocation scratch = {};
    bool invalid_content = false;
    struct dora_i256 decimal_bounds[3] = {};
    if (status == DORA_OK) {
        status = variant_decimal_bounds_prepare(decimal_bounds);
    }
    for (uint64_t row = 0; status == DORA_OK && row < view.length; ++row) {
        if (!dora_bitmap_test((const uint8_t*)in_out_validity, row)) {
            continue;
        }
        struct dora_bytes metadata = {};
        bool is_null = false;
        status = dora_vector_read_bytes(&metadata_vector, row, &metadata, &is_null);
        if (status == DORA_OK && (is_null || dora_variant_metadata_check(metadata) != DORA_OK)) {
            status = EILSEQ;
        }
        if (status == DORA_OK && packet.address == nullptr) {
            size_t packet_bytes = 0;
            if (ckd_mul(&packet_bytes, variant->role_count, sizeof(struct variant_packet_role))) {
                status = EOVERFLOW;
            } else {
                status = dora_pool_allocate(pool, packet_bytes, &packet);
            }
            if (status == DORA_OK) {
                status = variant_packet_bind(&plain, variant, packet.address);
            }
        }
        if (status == DORA_OK) {
            status = variant_validate_row(pool, variant, packet.address, row, &scratch, metadata, decimal_bounds);
        }
        if (status == EILSEQ) {
            bitmap_clear(in_out_validity, row);
            invalid_content = true;
            status = DORA_OK;
        }
    }
    dora_allocation_release(&scratch);
    dora_allocation_release(&packet);
    dora_vector_release(&owned);
    return status != DORA_OK ? status : invalid_content ? EILSEQ : DORA_OK;
}

const struct dora_extension_definition dora_extension_parquet_variant = {
    .name = {.data = (const uint8_t*)"arrow.parquet.variant", .byte_count = sizeof("arrow.parquet.variant") - 1},
    .bind = variant_bind,
    .validate = variant_validate
};
