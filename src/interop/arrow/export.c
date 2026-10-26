// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "interop/arrow/arrow_internal.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "schema/schema_internal.h"
#include "types/types_internal.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct arrow_text {
    char* data;
    size_t capacity_bytes;
    size_t byte_count;
};

struct arrow_bump {
    unsigned char* base;
    size_t used_bytes;
    size_t capacity_bytes;
};

struct arrow_schema_export_state {
    struct dora_allocation self_storage;
    uint32_t claim_count;
    alignas(8) unsigned char arena_bytes[];
};

struct arrow_array_export_state {
    struct dora_allocation self_storage;
    uint32_t claim_count;
    struct dora_vector root;
    alignas(8) unsigned char arena_bytes[];
};

struct arrow_export_context {
    struct arrow_bump schema_bump;
    struct arrow_bump array_bump;
    struct arrow_schema_export_state* schema_arena;
    struct arrow_array_export_state* array_arena;
    uint32_t schema_descriptor_count;
    uint32_t array_descriptor_count;
    bool is_device_export;
    struct dora_placement placement;
    struct dora_readiness readiness;
};

static int text_append(struct arrow_text* text, const char* value, size_t length) {
    size_t next = 0;
    if (ckd_add(&next, text->byte_count, length)) {
        return EOVERFLOW;
    }
    if (text->data != nullptr) {
        if (next > text->capacity_bytes) {
            return ENOBUFS;
        }
        if (length != 0) {
            memcpy(text->data + text->byte_count, value, length);
        }
    }
    text->byte_count = next;
    return DORA_OK;
}

static int text_integer(struct arrow_text* text, int64_t value) {
    char buffer[32] = {};
    const int count = snprintf(buffer, sizeof(buffer), "%lld", (long long)value);
    return count < 0 || (size_t)count >= sizeof(buffer) ? EOVERFLOW : text_append(text, buffer, (size_t)count);
}

static struct dora_metadata_entry field_extension(const struct dora_schema_field* field) {
    if (field->type.extension_name.byte_count != 0) {
        return (struct dora_metadata_entry){.key = field->type.extension_name, .value = field->type.extension_metadata};
    }
    const char* metadata = nullptr;
    switch (field->type.id) {
    case DORA_TYPE_INT128:
        metadata = "{\"type_name\":\"int128\",\"vendor_name\":\"dora\"}";
        break;
    case DORA_TYPE_UINT128:
        metadata = "{\"type_name\":\"uint128\",\"vendor_name\":\"dora\"}";
        break;
    case DORA_TYPE_INT256:
        metadata = "{\"type_name\":\"int256\",\"vendor_name\":\"dora\"}";
        break;
    default:
        break;
    }
    return metadata == nullptr
               ? (struct dora_metadata_entry){}
               : (struct dora_metadata_entry){
                     .key = {.data = (const uint8_t*)"arrow.opaque", .byte_count = sizeof("arrow.opaque") - 1},
                     .value = {.data = (const uint8_t*)metadata, .byte_count = strlen(metadata)}
                 };
}

//------------------------------------------------------------------------------
//
// Export Storage
//
//------------------------------------------------------------------------------

static void schema_arena_release(struct ArrowSchema* schema) {
    if (schema == nullptr || schema->release == nullptr) {
        return;
    }
    struct arrow_schema_export_state* const state = schema->private_data;
    schema->release = nullptr;
    for (int64_t child = 0; child < schema->n_children; ++child) {
        struct ArrowSchema* const member = schema->children[child];
        if (member != nullptr && member->release != nullptr) {
            member->release(member);
        }
    }
    if (schema->dictionary != nullptr && schema->dictionary->release != nullptr) {
        schema->dictionary->release(schema->dictionary);
    }
    if (--state->claim_count == 0) {
        struct dora_allocation allocation = state->self_storage;
        dora_allocation_release(&allocation);
    }
}

static int bump_take(struct arrow_bump* bump, size_t count, size_t size, size_t alignment, void** out) {
    if (bump == nullptr || out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return EINVAL;
    }
    size_t begin = 0;
    size_t bytes = 0;
    size_t end = 0;
    if (ckd_add(&begin, bump->used_bytes, alignment - 1) || ckd_mul(&bytes, count, size)) {
        return EOVERFLOW;
    }
    begin &= ~(alignment - 1);
    if (ckd_add(&end, begin, bytes)) {
        return EOVERFLOW;
    }
    if (bump->base != nullptr && end > bump->capacity_bytes) {
        return ENOBUFS;
    }
    *out = bump->base == nullptr || bytes == 0 ? nullptr : bump->base + begin;
    bump->used_bytes = end;
    return DORA_OK;
}

static void export_rewind(struct arrow_export_context* ctx,
                          unsigned char* schema_base,
                          size_t schema_capacity,
                          unsigned char* array_base,
                          size_t array_capacity) {
    ctx->schema_bump = (struct arrow_bump){.base = schema_base, .capacity_bytes = schema_capacity};
    ctx->array_bump = (struct arrow_bump){.base = array_base, .capacity_bytes = array_capacity};
    ctx->schema_descriptor_count = 0;
    ctx->array_descriptor_count = 0;
}

//------------------------------------------------------------------------------
//
// Schema Encoding
//
//------------------------------------------------------------------------------

static int format_append(struct arrow_text* text, const struct dora_schema_field* field, enum dora_layout layout) {
    if (text == nullptr || field == nullptr) {
        return EINVAL;
    }
    const enum dora_type type = layout == DORA_LAYOUT_DICTIONARY ? field->index_type : field->type.id;
    const enum dora_layout format_layout = layout == DORA_LAYOUT_DICTIONARY ? DORA_LAYOUT_FIXED : layout;
    if (layout == DORA_LAYOUT_RUN_END) {
        return text_append(text, "+r", 2);
    }
    for (uint32_t index = 0; index < ARROW_SIMPLE_FORMAT_COUNT; ++index) {
        const struct arrow_simple_format* const entry = &dora_arrow_simple_formats[index];
        if (entry->type_kind == type && entry->layout_kind == format_layout &&
            ((type != DORA_TYPE_TIME && type != DORA_TYPE_DURATION) ||
             entry->parameter == field->type.parameters.unit)) {
            return text_append(text, (const char*)entry->format.data, entry->format.byte_count);
        }
    }
    if (layout == DORA_LAYOUT_DICTIONARY) {
        return EINVAL;
    }
    static constexpr char units[] = {'s', 'm', 'u', 'n'};
    int status;
    switch (field->type.id) {
    case DORA_TYPE_INT128:
    case DORA_TYPE_UINT128:
        return text_append(text, "w:16", 4);
    case DORA_TYPE_INT256:
        return text_append(text, "w:32", 4);
    case DORA_TYPE_FIXED_BINARY:
    case DORA_TYPE_FIXED_LIST:
        status = text_append(text,
                             field->type.id == DORA_TYPE_FIXED_LIST ? "+w:" : "w:",
                             field->type.id == DORA_TYPE_FIXED_LIST ? 3 : 2);
        return status != DORA_OK
                   ? status
                   : text_integer(text,
                                  field->type.id == DORA_TYPE_FIXED_LIST ? field->type.parameters.list_element_count
                                                                         : field->type.parameters.byte_width);
    case DORA_TYPE_DECIMAL: {
        const uint32_t byte_width = type_decimal_byte_width(&field->type);
        if (byte_width == 0) {
            return EINVAL;
        }
        const uint32_t bit_width = 8 * byte_width;
        status = text_append(text, "d:", 2);
        if (status == DORA_OK) {
            status = text_integer(text, field->type.parameters.decimal.precision);
        }
        if (status == DORA_OK) {
            status = text_append(text, ",", 1);
        }
        if (status == DORA_OK) {
            status = text_integer(text, field->type.parameters.decimal.scale);
        }
        if (status == DORA_OK && bit_width != 128) {
            status = text_append(text, ",", 1);
            if (status == DORA_OK) {
                status = text_integer(text, bit_width);
            }
        }
        return status;
    }
    case DORA_TYPE_TIME:
    case DORA_TYPE_DURATION:
        return EINVAL;
    case DORA_TYPE_TIMESTAMP: {
        const struct dora_timestamp_type timestamp = field->type.parameters.timestamp;
        if (timestamp.unit > DORA_TIME_UNIT_NANOSECOND) {
            return EINVAL;
        }
        status = text_append(text, "ts", 2);
        if (status == DORA_OK) {
            status = text_append(text, &units[timestamp.unit], 1);
        }
        if (status == DORA_OK) {
            status = text_append(text, ":", 1);
        }
        return status != DORA_OK || timestamp.timezone == nullptr
                   ? status
                   : text_append(text, timestamp.timezone, strlen(timestamp.timezone));
    }
    case DORA_TYPE_UNION:
        if (layout != DORA_LAYOUT_DENSE_UNION && layout != DORA_LAYOUT_SPARSE_UNION) {
            return ENOTSUP;
        }
        status = text_append(text, layout == DORA_LAYOUT_DENSE_UNION ? "+ud:" : "+us:", 4);
        for (uint32_t index = 0; status == DORA_OK && index < field->child_count; ++index) {
            if (index != 0) {
                status = text_append(text, ",", 1);
            }
            if (status == DORA_OK) {
                status = text_integer(text, field->type.parameters.type_codes[index]);
            }
        }
        return status;
    default:
        return ENOTSUP;
    }
}

static void metadata_write_pair(unsigned char** cursor,
                                const void* key,
                                uint32_t key_bytes,
                                const void* value,
                                uint32_t value_bytes) {
    const int32_t lengths[] = {(int32_t)key_bytes, (int32_t)value_bytes};
    memcpy(*cursor, &lengths[0], sizeof(int32_t));
    *cursor += sizeof(int32_t);
    if (key_bytes != 0) {
        memcpy(*cursor, key, key_bytes);
        *cursor += key_bytes;
    }
    memcpy(*cursor, &lengths[1], sizeof(int32_t));
    *cursor += sizeof(int32_t);
    if (value_bytes != 0) {
        memcpy(*cursor, value, value_bytes);
        *cursor += value_bytes;
    }
}

static int metadata_pair_bytes(size_t* in_out_bytes, uint32_t key_bytes, uint32_t value_bytes) {
    if (in_out_bytes == nullptr || key_bytes > INT32_MAX || value_bytes > INT32_MAX) {
        return EINVAL;
    }
    size_t bytes = 0;
    if (ckd_add(&bytes, (size_t)key_bytes, value_bytes) || ckd_add(&bytes, bytes, 2 * sizeof(int32_t)) ||
        ckd_add(in_out_bytes, *in_out_bytes, bytes)) {
        return EOVERFLOW;
    }
    return DORA_OK;
}

static int
schema_metadata(struct arrow_export_context* context, const struct dora_schema_field* field, char** out_metadata) {
    const struct dora_metadata_entry extension = field_extension(field);
    uint32_t count = extension.key.byte_count == 0 ? 0 : 2;
    size_t bytes = sizeof(int32_t);
    if (extension.key.byte_count > INT32_MAX || extension.value.byte_count > INT32_MAX) {
        return EOVERFLOW;
    }
    if (extension.key.byte_count != 0) {
        int status =
            metadata_pair_bytes(&bytes, sizeof("ARROW:extension:name") - 1, (uint32_t)extension.key.byte_count);
        if (status == DORA_OK) {
            status = metadata_pair_bytes(&bytes,
                                         sizeof("ARROW:extension:metadata") - 1,
                                         (uint32_t)extension.value.byte_count);
        }
        if (status != DORA_OK) {
            *out_metadata = nullptr;
            return status;
        }
    }
    for (uint32_t index = 0; index < field->metadata_entry_count; ++index) {
        const struct dora_metadata_entry entry = field->metadata[index];
        if (dora_arrow_is_extension_metadata_key(entry.key)) {
            continue;
        }
        if (entry.key.byte_count > INT32_MAX || entry.value.byte_count > INT32_MAX || count == INT32_MAX) {
            return EOVERFLOW;
        }
        const int status =
            metadata_pair_bytes(&bytes, (uint32_t)entry.key.byte_count, (uint32_t)entry.value.byte_count);
        if (status != DORA_OK) {
            *out_metadata = nullptr;
            return status;
        }
        ++count;
    }
    if (count == 0) {
        *out_metadata = nullptr;
        return DORA_OK;
    }
    void* address = nullptr;
    const int status = bump_take(&context->schema_bump, bytes, 1, 1, &address);
    *out_metadata = address;
    if (status != DORA_OK || address == nullptr) {
        return status;
    }
    const int32_t pairs = (int32_t)count;
    unsigned char* cursor = address;
    memcpy(cursor, &pairs, sizeof(pairs));
    cursor += sizeof(pairs);
    if (extension.key.byte_count != 0) {
        metadata_write_pair(&cursor,
                            "ARROW:extension:name",
                            sizeof("ARROW:extension:name") - 1,
                            extension.key.data,
                            (uint32_t)extension.key.byte_count);
        metadata_write_pair(&cursor,
                            "ARROW:extension:metadata",
                            sizeof("ARROW:extension:metadata") - 1,
                            extension.value.data,
                            (uint32_t)extension.value.byte_count);
    }
    for (uint32_t index = 0; index < field->metadata_entry_count; ++index) {
        const struct dora_metadata_entry entry = field->metadata[index];
        if (!dora_arrow_is_extension_metadata_key(entry.key)) {
            metadata_write_pair(&cursor,
                                entry.key.data,
                                (uint32_t)entry.key.byte_count,
                                entry.value.data,
                                (uint32_t)entry.value.byte_count);
        }
    }
    return DORA_OK;
}

static int
schema_tree(struct arrow_export_context* context, const struct dora_schema* source, struct ArrowSchema** out_schema) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(source, &field);
    if (status != DORA_OK) {
        return status;
    }
    const bool dictionary = field.layout == DORA_LAYOUT_DICTIONARY;
    const uint32_t children = dictionary ? 0 : field.child_count;
    void* address = nullptr;
    status = bump_take(&context->schema_bump, 1, sizeof(struct ArrowSchema), alignof(struct ArrowSchema), &address);
    struct ArrowSchema* const result = address;
    if (ckd_add(&context->schema_descriptor_count, context->schema_descriptor_count, 1)) {
        return EOVERFLOW;
    }
    struct arrow_text format = {};
    if (status == DORA_OK) {
        status = format_append(&format, &field, field.layout);
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, format.byte_count, (size_t)1)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = bump_take(&context->schema_bump, bytes, 1, 1, &address);
    }
    char* const format_text = address;
    if (status == DORA_OK && format_text != nullptr) {
        format = (struct arrow_text){.data = format_text, .capacity_bytes = bytes};
        status = format_append(&format, &field, field.layout);
        format_text[format.byte_count] = '\0';
    }
    const size_t name_bytes = field.name == nullptr ? 1 : strlen(field.name) + 1;
    if (status == DORA_OK) {
        status = bump_take(&context->schema_bump, name_bytes, 1, 1, &address);
    }
    char* const name = address;
    if (status == DORA_OK && name != nullptr) {
        memcpy(name, field.name == nullptr ? "" : field.name, name_bytes);
    }
    char* metadata = nullptr;
    if (status == DORA_OK) {
        status = schema_metadata(context, &field, &metadata);
    }
    if (status == DORA_OK) {
        status = bump_take(&context->schema_bump,
                           children,
                           sizeof(struct ArrowSchema*),
                           alignof(struct ArrowSchema*),
                           &address);
    }
    struct ArrowSchema** const child_schemas = (struct ArrowSchema**)address;
    if (status == DORA_OK && result != nullptr) {
        *result = (struct ArrowSchema){
            .format = format_text,
            .name = name,
            .metadata = metadata,
            .flags = (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0 ? ARROW_FLAG_NULLABLE : 0,
            .n_children = children,
            .children = child_schemas,
            .release = schema_arena_release,
            .private_data = context->schema_arena
        };
        if ((field.flags & DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED) != 0) {
            result->flags += ARROW_FLAG_DICTIONARY_ORDERED;
        }
        if ((field.flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) != 0) {
            result->flags += ARROW_FLAG_MAP_KEYS_SORTED;
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < (dictionary ? 1 : children); ++index) {
        struct dora_schema child = {};
        struct ArrowSchema* output = nullptr;
        status = dora_schema_get_physical_child(source, index, &child);
        if (status == DORA_OK) {
            status = schema_tree(context, &child, &output);
        }
        if (status == DORA_OK && result != nullptr) {
            if (dictionary) {
                result->dictionary = output;
            } else {
                child_schemas[index] = output;
            }
        }
    }
    if (status == DORA_OK) {
        *out_schema = result;
    }
    return status;
}

static int export_schema_tree(struct dora_pool pool, const struct dora_schema* source, struct ArrowSchema* out_schema) {
    struct dora_schema retained = {};
    int status = dora_schema_retain(source, &retained);
    source = &retained;
    struct arrow_export_context context = {};
    struct ArrowSchema* root = nullptr;
    if (status == DORA_OK) {
        status = schema_tree(&context, source, &root);
    }
    const size_t capacity = context.schema_bump.used_bytes;
    const uint32_t descriptors = context.schema_descriptor_count;
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, offsetof(struct arrow_schema_export_state, arena_bytes), capacity)) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, &allocation);
    }
    if (status == DORA_OK) {
        context.schema_arena = allocation.address;
        *context.schema_arena =
            (struct arrow_schema_export_state){.self_storage = allocation, .claim_count = descriptors};
        export_rewind(&context, context.schema_arena->arena_bytes, capacity, nullptr, 0);
        status = schema_tree(&context, source, &root);
        if (status == DORA_OK &&
            (context.schema_bump.used_bytes != capacity || context.schema_descriptor_count != descriptors)) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK) {
        *out_schema = *root;
    } else {
        dora_allocation_release(&allocation);
    }
    dora_schema_release(&retained);
    return status;
}

int dora_arrow_export_schema(struct dora_pool pool, const struct dora_schema* source, struct ArrowSchema* out_schema) {
    if (out_schema == nullptr || out_schema->release != nullptr || !dora_schema_is_valid(source)) {
        return EINVAL;
    }
    struct ArrowSchema candidate = {};
    const int status = export_schema_tree(pool, source, &candidate);
    if (status == DORA_OK) {
        *out_schema = candidate;
    }
    return status;
}

int dora_arrow_device_view(const struct dora_vector* source, struct dora_vector_view* out_view) {
    int status = dora_vector_get_device_view(source, out_view);
    if (status == ENOTSUP) {
        struct dora_vector_view host = {};
        status = dora_vector_get_view(source, &host);
        if (status == DORA_OK && host.placement.device.type != DORA_DEVICE_TYPE_CPU) {
            status = ENOTSUP;
        }
        if (status == DORA_OK) {
            *out_view = host;
        }
    }
    return status;
}

static void array_arena_release(struct ArrowArray* array) {
    if (array == nullptr || array->release == nullptr) {
        return;
    }
    struct arrow_array_export_state* const state = array->private_data;
    array->release = nullptr;
    for (int64_t index = 0; index < array->n_children; ++index) {
        struct ArrowArray* const child = array->children[index];
        if (child != nullptr && child->release != nullptr) {
            child->release(child);
        }
    }
    if (array->dictionary != nullptr && array->dictionary->release != nullptr) {
        array->dictionary->release(array->dictionary);
    }
    if (--state->claim_count == 0) {
        dora_vector_release(&state->root);
        struct dora_allocation storage = state->self_storage;
        dora_allocation_release(&storage);
    }
}

static int array_validity(struct arrow_export_context* context,
                          const struct dora_vector* source,
                          const struct dora_vector_view* view,
                          uint64_t offset,
                          const void** out_validity) {
    struct dora_view_buffer validity = {};
    uint64_t origin = 0;
    int status = dora_vector_get_own_validity(source, &validity, &origin);
    if (status != DORA_OK) {
        return status;
    }
    *out_validity = validity.data;
    if (validity.data == nullptr || origin == offset || view->length == 0) {
        return DORA_OK;
    }
    if (context->is_device_export && view->placement.device.type != DORA_DEVICE_TYPE_CPU) {
        return ENOTSUP;
    }
    if (view->placement.access != DORA_ACCESS_HOST && view->placement.access != DORA_ACCESS_HOST_AND_DEVICE) {
        return ENOTSUP;
    }
    uint64_t endpoint = 0;
    if (ckd_add(&endpoint, offset, view->length)) {
        return EOVERFLOW;
    }
    const uint64_t words = bitmap_words_for_bits(endpoint);
    if (words > SIZE_MAX) {
        return EOVERFLOW;
    }
    void* address = nullptr;
    status = bump_take(&context->array_bump, (size_t)words, sizeof(uint64_t), alignof(uint64_t), &address);
    if (status == DORA_OK && address != nullptr) {
        if (view->readiness.settle != nullptr) {
            status = view->readiness.settle(&view->readiness);
        }
        if (status == DORA_OK) {
            bitmap_fill_bytes(address, (size_t)words * sizeof(uint64_t), false);
            for (uint64_t row = 0; row < view->length; ++row) {
                bitmap_assign(address, offset + row, dora_bitmap_test(validity.data, origin + row));
            }
        }
    }
    *out_validity = address;
    return status;
}

static int
array_tree(struct arrow_export_context* context, const struct dora_vector* source, struct ArrowArray** out_array) {
    struct dora_vector_view view = {};
    int status =
        context->is_device_export ? dora_arrow_device_view(source, &view) : dora_vector_get_view(source, &view);
    if (status != DORA_OK) {
        return status;
    }
    if (view.selection.indices != nullptr) {
        return ENOTSUP;
    }
    if (view.length > INT64_MAX || view.offset > INT64_MAX) {
        return EOVERFLOW;
    }
    if (context->is_device_export && (view.placement.device.type != context->placement.device.type ||
                                      view.placement.device.id != context->placement.device.id ||
                                      view.readiness.event != context->readiness.event)) {
        return ENOTSUP;
    }
    uint32_t child_count = 0;
    status = dora_vector_get_child_count(source, &child_count);
    const bool dictionary = view.layout == DORA_LAYOUT_DICTIONARY;
    const bool map =
        view.type == DORA_TYPE_MAP && view.layout != DORA_LAYOUT_DICTIONARY && view.layout != DORA_LAYOUT_RUN_END;
    if (map && view.layout != DORA_LAYOUT_LIST) {
        return ENOTSUP;
    }
    const uint64_t length = view.length;
    uint64_t offset = view.offset;
    if (status == DORA_OK && view.layout == DORA_LAYOUT_STRUCT && child_count != 0) {
        struct dora_vector physical = {};
        struct dora_vector exposed = {};
        status = dora_vector_get_physical_child(source, 0, &physical);
        if (status == DORA_OK) {
            status = dora_vector_get_child(source, 0, &exposed);
        }
        if (status == DORA_OK && exposed.rows.offset < physical.rows.offset) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            offset = exposed.rows.offset - physical.rows.offset;
        }
        if (status == DORA_OK && offset > INT64_MAX) {
            status = EOVERFLOW;
        }
    }
    int64_t null_count = 0;
    if (status == DORA_OK) {
        status = dora_vector_get_own_null_count(source, &null_count);
    }
    const uint32_t arrow_children = dictionary ? 0 : map ? 1 : child_count;
    uint32_t buffer_count = 0;
    switch (view.layout) {
    case DORA_LAYOUT_NULL:
    case DORA_LAYOUT_RUN_END:
        buffer_count = 0;
        break;
    case DORA_LAYOUT_STRUCT:
    case DORA_LAYOUT_FIXED_LIST:
    case DORA_LAYOUT_SPARSE_UNION:
        buffer_count = 1;
        break;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_LIST_VIEW:
    case DORA_LAYOUT_LARGE_LIST_VIEW:
        buffer_count = 3;
        break;
    case DORA_LAYOUT_BINARY_VIEW:
        if (ckd_add(&buffer_count, view.data_buffer_count, 3U)) {
            status = EOVERFLOW;
        }
        break;
    default:
        buffer_count = 2;
        break;
    }
    void* address = nullptr;
    if (status == DORA_OK) {
        status = bump_take(&context->array_bump, 1, sizeof(struct ArrowArray), alignof(struct ArrowArray), &address);
    }
    struct ArrowArray* const result = address;
    if (context->array_descriptor_count == UINT32_MAX) {
        return EOVERFLOW;
    }
    ++context->array_descriptor_count;
    if (status == DORA_OK) {
        status = bump_take(&context->array_bump, buffer_count, sizeof(void*), alignof(void*), &address);
    }
    const void** const buffers = (const void**)address;
    if (status == DORA_OK && buffers != nullptr) {
        memset((void*)buffers, 0, buffer_count * sizeof(void*));
    }
    const bool validity = view.layout != DORA_LAYOUT_NULL && view.layout != DORA_LAYOUT_RUN_END &&
                          view.layout != DORA_LAYOUT_DENSE_UNION && view.layout != DORA_LAYOUT_SPARSE_UNION;
    const void* bitmap = nullptr;
    if (status == DORA_OK && validity) {
        status = array_validity(context, source, &view, offset, &bitmap);
    }
    struct dora_view_buffer sizes = {};
    if (status == DORA_OK && view.layout == DORA_LAYOUT_BINARY_VIEW) {
        status = dora_vector_get_binary_view_sizes(source, &sizes);
    }
    if (status == DORA_OK && buffers != nullptr) {
        if (validity) {
            buffers[0] = bitmap;
        }
        switch (view.layout) {
        case DORA_LAYOUT_FIXED:
        case DORA_LAYOUT_BITS:
        case DORA_LAYOUT_DICTIONARY:
            buffers[1] = view.values.data;
            break;
        case DORA_LAYOUT_BINARY:
        case DORA_LAYOUT_LARGE_BINARY:
            buffers[1] = view.offsets.data;
            buffers[2] = view.data_buffers[0].data;
            break;
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
            buffers[1] = view.offsets.data;
            break;
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW:
            buffers[1] = view.offsets.data;
            buffers[2] = view.sizes.data;
            break;
        case DORA_LAYOUT_DENSE_UNION:
            buffers[0] = view.type_ids.data;
            buffers[1] = view.offsets.data;
            break;
        case DORA_LAYOUT_SPARSE_UNION:
            buffers[0] = view.type_ids.data;
            break;
        case DORA_LAYOUT_BINARY_VIEW:
            buffers[1] = view.values.data;
            for (uint32_t index = 0; index < view.data_buffer_count; ++index) {
                buffers[index + 2] = view.data_buffers[index].data;
            }
            buffers[buffer_count - 1] = sizes.data;
            break;
        default:
            break;
        }
    }
    if (status == DORA_OK) {
        status = bump_take(&context->array_bump,
                           arrow_children,
                           sizeof(struct ArrowArray*),
                           alignof(struct ArrowArray*),
                           &address);
    }
    struct ArrowArray** const children = (struct ArrowArray**)address;
    if (status == DORA_OK && result != nullptr) {
        *result = (struct ArrowArray){
            .length = (int64_t)length,
            .null_count = null_count,
            .offset = (int64_t)offset,
            .n_buffers = buffer_count,
            .n_children = arrow_children,
            .buffers = buffers,
            .children = children,
            .release = array_arena_release,
            .private_data = context->array_arena
        };
    }
    for (uint32_t index = 0; status == DORA_OK && index < (dictionary ? 1 : arrow_children); ++index) {
        struct dora_vector child = {};
        struct ArrowArray* output = nullptr;
        status = dora_vector_get_physical_child(source, index, &child);
        if (status == DORA_OK) {
            status = array_tree(context, &child, &output);
        }
        if (status == DORA_OK) {
            if (dictionary) {
                if (result != nullptr) {
                    result->dictionary = output;
                }
            } else if (children != nullptr) {
                children[index] = output;
            }
        }
    }
    if (status == DORA_OK) {
        *out_array = result;
    }
    return status;
}

static int export_array(struct dora_pool pool,
                        const struct dora_vector* source,
                        bool device,
                        const struct dora_arrow_limits* limits,
                        struct arrow_traversal_state* state,
                        struct ArrowDeviceArray* out_array) {
    // The export claim pins buffers until consumer release.
    struct dora_vector retained = {};
    struct dora_handle claim = {};
    int status = dora_vector_is_valid(source)
                     ? dora_identity_claim_acquire(pool, &source->handle, IDENTITY_KIND_VECTOR, &claim)
                     : EINVAL;
    if (status == DORA_OK) {
        retained = *source;
        retained.handle = claim;
    }
    source = &retained;
    struct dora_pool vector_pool = {};
    if (status == DORA_OK) {
        vector_pool = (struct dora_pool){
            .index = retained.handle.pool_index,
            .generation = dora_pool_rows[retained.handle.pool_index].generation
        };
    }
    dora_pool_callback_enter(vector_pool);
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = device ? dora_arrow_device_view(source, &view) : dora_vector_get_view(source, &view);
    }
    struct arrow_export_context context =
        {.is_device_export = device, .placement = view.placement, .readiness = view.readiness};
    struct ArrowArray* root = nullptr;
    if (status == DORA_OK) {
        status = array_tree(&context, source, &root);
    }
    const size_t capacity = context.array_bump.used_bytes;
    const uint32_t descriptors = context.array_descriptor_count;
    size_t bytes = 0;
    if (status == DORA_OK && ckd_add(&bytes, offsetof(struct arrow_array_export_state, arena_bytes), capacity)) {
        status = EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    if (status == DORA_OK && state != nullptr) {
        status = arrow_traversal_charge(limits, state, 0, bytes);
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, &allocation);
    }
    if (status == DORA_OK) {
        context.array_arena = allocation.address;
        *context.array_arena =
            (struct arrow_array_export_state){.self_storage = allocation, .claim_count = descriptors};
        context.array_arena->root = retained;
        retained = (struct dora_vector){};
        source = &context.array_arena->root;
    }
    if (status == DORA_OK) {
        export_rewind(&context, nullptr, 0, context.array_arena->arena_bytes, capacity);
        status = array_tree(&context, source, &root);
        if (status == DORA_OK &&
            (context.array_bump.used_bytes != capacity || context.array_descriptor_count != descriptors)) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK) {
        *out_array = (struct ArrowDeviceArray){
            .array = *root,
            .device_id = view.placement.device.id,
            .device_type = view.placement.device.type,
            .sync_event = device && view.placement.device.type != DORA_DEVICE_TYPE_CPU ? view.readiness.event : nullptr
        };
    } else {
        if (context.array_arena != nullptr) {
            dora_vector_release(&context.array_arena->root);
        }
        dora_allocation_release(&allocation);
    }
    dora_vector_release(&retained);
    dora_pool_callback_leave(vector_pool);
    return status;
}

int dora_arrow_export_array_in_pool(struct dora_pool pool,
                                    const struct dora_vector* source,
                                    struct ArrowArray* out_array) {
    if (out_array == nullptr || out_array->release != nullptr) {
        return EINVAL;
    }
    struct ArrowDeviceArray candidate = {};
    const int status = export_array(pool, source, false, nullptr, nullptr, &candidate);
    if (status == DORA_OK) {
        *out_array = candidate.array;
    }
    return status;
}

int dora_arrow_export_array(const struct dora_vector* source, struct ArrowArray* out_array) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_arrow_export_array_in_pool(pool, source, out_array);
}

int dora_arrow_export_device_array_in_pool(struct dora_pool pool,
                                           const struct dora_vector* source,
                                           struct ArrowDeviceArray* out_array) {
    if (out_array == nullptr || out_array->array.release != nullptr) {
        return EINVAL;
    }
    struct ArrowDeviceArray candidate = {};
    const int status = export_array(pool, source, true, nullptr, nullptr, &candidate);
    if (status == DORA_OK) {
        *out_array = candidate;
    }
    return status;
}

int dora_arrow_export_device_array(const struct dora_vector* source, struct ArrowDeviceArray* out_array) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_arrow_export_device_array_in_pool(pool, source, out_array);
}
