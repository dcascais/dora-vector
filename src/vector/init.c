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
#include "extension/extension_internal.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "memory/scratch.h"
#include "numeric/decimal.h"
#include "schema/schema_internal.h"
#include "types/load.h"
#include "types/types_internal.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "utf8/utf8.h"
#include "vector/image.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int table_span_append(size_t* size, uint32_t* byte_offset, uint32_t count, size_t width) {
    size_t bytes = 0;
    size_t aligned = 0;
    if (ckd_add(&aligned, *size, 7)) {
        return EOVERFLOW;
    }
    aligned = aligned / 8 * 8;
    if (aligned > UINT32_MAX || ckd_mul(&bytes, count, width) || ckd_add(size, aligned, bytes)) {
        return EOVERFLOW;
    }
    *byte_offset = (uint32_t)aligned;
    return DORA_OK;
}

static int image_allocate(struct dora_pool pool, const struct vector_input* input, struct vector_image** out_image) {
    struct vector_image shape = {
        .buffers.entry_count = input->buffer_count,
        .child_vectors.entry_count = input->child_count
    };
    size_t bytes = 0;
    int status =
        table_span_append(&bytes, &shape.buffers.byte_offset, input->buffer_count, sizeof(struct vector_buffer_row));
    if (status == DORA_OK) {
        status = table_span_append(&bytes,
                                   &shape.view_buffers_byte_offset,
                                   input->buffer_count,
                                   sizeof(struct dora_view_buffer));
    }
    if (status == DORA_OK) {
        status =
            table_span_append(&bytes, &shape.child_vectors.byte_offset, input->child_count, sizeof(struct dora_vector));
    }
    if (status != DORA_OK || ckd_add(&bytes, bytes, sizeof(struct vector_image))) {
        return status != DORA_OK ? status : EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    status = dora_memory_payload_allocate(pool, bytes, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_image* const image = allocation.address;
    memset(image, 0, bytes);
    *image = shape;
    image->allocation = allocation;
    image->buffer_image = image;
    image->claims.payload = image;
    image->node.offset = input->offset;
    image->node.content_state = input->content_state;
    if (input->content_state == VECTOR_CONTENT_PROMISED && (input->null_count == -1 || input->allocations != nullptr) &&
        input->field->layout != DORA_LAYOUT_BINARY_VIEW && input->field->type.extension_name.byte_count == 0 &&
        input->field->type.id != DORA_TYPE_UTF8 && input->field->type.id != DORA_TYPE_DECIMAL &&
        input->field->type.id != DORA_TYPE_DATE64 && input->field->type.id != DORA_TYPE_TIME &&
        input->field->type.id != DORA_TYPE_MAP) {
        image->node.content_state = VECTOR_CONTENT_VALIDATED;
    }

    *out_image = image;
    return DORA_OK;
}

static bool child_declaration_is_required(const struct vector_input* input,
                                          const struct dora_schema* declarations,
                                          uint32_t index) {
    return declarations != nullptr || input->child_names != nullptr ||
           (index == 0 && (input->field->layout == DORA_LAYOUT_RUN_END ||
                           (input->field->type.id == DORA_TYPE_MAP && input->field->layout != DORA_LAYOUT_DICTIONARY)));
}

static int child_field(const struct vector_input* input,
                       const struct dora_schema* declarations,
                       uint32_t index,
                       struct dora_schema_field* field) {
    if (declarations != nullptr) {
        struct dora_schema declaration = {};
        struct dora_schema_field source = {};
        int status = dora_schema_get_physical_child(declarations, index, &declaration);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&declaration, &source);
        }
        if (status != DORA_OK) {
            return status;
        }
        field->name = source.name;
        field->metadata = source.metadata;
        field->metadata_entry_count = source.metadata_entry_count;
        field->flags = source.flags;
    }
    if (input->child_names != nullptr) {
        field->name = input->child_names[index];
    }
    if (index == 0 && (input->field->layout == DORA_LAYOUT_RUN_END ||
                       (input->field->type.id == DORA_TYPE_MAP && input->field->layout != DORA_LAYOUT_DICTIONARY))) {
        field->flags |= DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
    }
    return DORA_OK;
}

static int child_schema_matches(const struct dora_schema* declared,
                                const struct dora_schema_field* field,
                                const struct dora_schema* actual,
                                bool* out_matches) {
    if (field == nullptr) {
        return dora_schema_declarations_equal(declared, actual, out_matches);
    }
    int status = dora_schema_field_matches(declared, field, out_matches);
    for (uint32_t child = 0; status == DORA_OK && *out_matches && child < field->child_count; ++child) {
        struct dora_schema left = {};
        struct dora_schema right = {};
        status = dora_schema_get_physical_child(declared, child, &left);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(actual, child, &right);
        }
        if (status == DORA_OK) {
            status = dora_schema_declarations_equal(&left, &right, out_matches);
        }
    }
    return status;
}

static int
image_schema(struct vector_image* image, const struct vector_input* input, const struct dora_schema* declarations) {
    const struct dora_schema* const candidate = declarations != nullptr ? declarations : input->admitted_schema;
    bool unchanged = false;
    if (candidate != nullptr) {
        const int status = dora_schema_field_matches(candidate, input->field, &unchanged);
        if (status != DORA_OK) {
            return status;
        }
    }
    int status;
    struct dora_allocation workspace = {};
    struct dora_schema* schemas = nullptr;
    for (uint32_t index = 0; index < input->child_count; ++index) {
        struct vector_image* child_image = nullptr;
        struct dora_schema_field field = {};
        const bool overlay = child_declaration_is_required(input, declarations, index);
        status = dora_vector_resolve(&input->children[index], &child_image);
        if (status == DORA_OK && overlay) {
            status = dora_schema_get_field(&child_image->schema, &field);
            if (status == DORA_OK) {
                status = child_field(input, declarations, index, &field);
            }
        }
        if (status == DORA_OK && unchanged) {
            struct dora_schema declared = {};
            status = dora_schema_get_physical_child(candidate, index, &declared);
            if (status == DORA_OK) {
                status = child_schema_matches(&declared, overlay ? &field : nullptr, &child_image->schema, &unchanged);
            }
        }
        if (status != DORA_OK) {
            goto cleanup;
        }
        if (unchanged) {
            continue;
        }
        if (schemas == nullptr) {
            size_t bytes = 0;
            if (ckd_mul(&bytes, input->child_count, sizeof(struct dora_schema))) {
                status = EOVERFLOW;
                goto cleanup;
            }
            status = dora_memory_payload_allocate(image->allocation.pool, bytes, &workspace);
            if (status != DORA_OK) {
                goto cleanup;
            }
            schemas = workspace.address;
            memset(schemas, 0, bytes);
            for (uint32_t previous = 0; status == DORA_OK && previous < index; ++previous) {
                status = dora_schema_get_physical_child(candidate, previous, &schemas[previous]);
            }
        }
        if (status == DORA_OK && overlay) {
            status =
                dora_schema_init_retain_field(&schemas[index], image->allocation.pool, &child_image->schema, &field);
        } else if (status == DORA_OK) {
            schemas[index] = dora_schema_borrow(&child_image->schema);
        }
        if (status != DORA_OK) {
            goto cleanup;
        }
    }
    if (unchanged) {
        status = dora_identity_claim_acquire(image->allocation.pool,
                                             &candidate->handle,
                                             IDENTITY_KIND_SCHEMA,
                                             &image->schema.handle);
        if (status == DORA_OK) {
            image->schema.node_index = candidate->node_index;
        }
    } else {
        struct dora_schema schema = {};
        status = dora_schema_init_retain_node(&schema, image->allocation.pool, input->field, schemas);
        if (status == DORA_OK) {
            status = dora_identity_claim_transfer(image->allocation.pool, &schema.handle, IDENTITY_KIND_SCHEMA);
        }
        if (status == DORA_OK) {
            image->schema = schema;
            schema = (struct dora_schema){};
        }
        dora_schema_release(&schema);
    }
cleanup:
    for (uint32_t index = 0; schemas != nullptr && index < input->child_count; ++index) {
        dora_schema_release(&schemas[index]);
    }
    dora_allocation_release(&workspace);
    return status;
}

int dora_vector_refresh_schema(struct vector_image* image) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(&image->schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_schema previous = image->schema;
    image->schema = (struct dora_schema){};
    const struct vector_input input = {
        .field = &field,
        .children = dora_vector_image_children(image),
        .child_count = image->child_vectors.entry_count
    };
    status = image_schema(image, &input, &previous);
    if (status == DORA_OK) {
        dora_schema_release(&previous);
    } else {
        image->schema = previous;
    }
    return status;
}

static int
allocate_values(struct dora_pool pool, uint64_t count, uint32_t width, struct dora_allocation* out_allocation) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, count, width)) {
        return EOVERFLOW;
    }
    return dora_pool_allocate(pool, bytes, out_allocation);
}

static int pack_validity(struct dora_pool pool,
                         const bool* validity,
                         uint64_t row_count,
                         struct dora_allocation* out_buffer,
                         int64_t* out_null_count) {
    if (row_count > INT64_MAX) {
        return EINVAL;
    }
    *out_null_count = 0;
    if (validity == nullptr || row_count == 0) {
        return DORA_OK;
    }
    const int status = allocate_values(pool, bitmap_words_for_bits(row_count), sizeof(uint64_t), out_buffer);
    if (status != DORA_OK) {
        return status;
    }
    dora_bitmap_copy_bool8_to_words(out_buffer->address, validity, 0, row_count);
    *out_null_count = (int64_t)(row_count - dora_bitmap_count_set_range(out_buffer->address, 0, row_count));
    if (*out_null_count == 0) {
        dora_allocation_release(out_buffer);
    }
    return DORA_OK;
}

static int extension_field(const struct dora_vector* vector,
                           struct dora_bytes name,
                           struct dora_bytes metadata,
                           const struct dora_vector* storage,
                           struct dora_schema_field* out_field) {
    if (!dora_vector_is_empty(vector) || name.byte_count == 0) {
        return EINVAL;
    }
    struct vector_image* source_image = nullptr;
    int status = dora_vector_resolve(storage, &source_image);
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(&source_image->schema, &field);
    }
    if (status != DORA_OK) {
        return status;
    }
    field.type = (struct dora_logical_type){
        .id = field.type.id,
        .parameters = field.type.parameters,
        .extension_name = name,
        .extension_metadata = metadata
    };
    *out_field = field;
    return DORA_OK;
}

static int vector_share(struct dora_pool pool,
                        const struct dora_vector* source,
                        const struct dora_schema_field* field,
                        const struct dora_schema* root_schema,
                        const struct dora_vector* rewritten_children,
                        const struct dora_schema* values_schema,
                        struct vector_image** out_image,
                        struct dora_error* out_error) {
    struct vector_image* source_image = nullptr;
    if (field == nullptr || out_image == nullptr || dora_vector_resolve(source, &source_image) != DORA_OK ||
        field->child_count != source_image->child_vectors.entry_count) {
        return EINVAL;
    }
    struct dora_vector_view view = {};
    int status = dora_vector_bind_coordinates(source, &view);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_input input = {
        .field = field,
        .row_count = source->rows.length,
        .null_count = view.null_count,
        .child_count = source_image->child_vectors.entry_count
    };
    struct vector_image* image = nullptr;
    status = image_allocate(pool, &input, &image);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_node_ownership* const owned = &image->ownership;
    status = dora_identity_claim_acquire(pool, &source->handle, IDENTITY_KIND_VECTOR, &owned->backing.handle);
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return status;
    }
    owned->backing.rows = source->rows;
    source = &owned->backing;
    const struct dora_schema* declarations = &source_image->schema;
    if (rewritten_children != nullptr) {
        input.admitted_schema = declarations;
        declarations = nullptr;
    }
    bool use_root = false;
    if (root_schema != nullptr) {
        struct dora_schema_field admitted = {};
        status = dora_schema_get_field(root_schema, &admitted);
        use_root = status == DORA_OK && admitted.layout == field->layout && admitted.child_count == field->child_count;
        if (use_root) {
            declarations = root_schema;
        }
    }
    owned->view = view;
    owned->view.type = field->type.id;
    owned->own_validity = source_image->ownership.own_validity;
    owned->own_validity_bit_offset =
        source_image->ownership.own_validity_bit_offset +
        (source_image->ownership.view.selection.indices == nullptr ? source->rows.offset : 0);
    owned->own_null_count = source_image->ownership.own_null_count;
    if (owned->own_null_count != 0 &&
        (source->rows.offset != 0 || source->rows.length != source_image->ownership.view.length)) {
        owned->own_null_count = -1;
    }
    owned->requires_effective_validity = source_image->ownership.requires_effective_validity;
    image->buffer_image = source_image->buffer_image;
    image->node.content_state = source_image->node.content_state;
    struct dora_schema_field original = {};
    status = dora_schema_get_field(&source_image->schema, &original);
    if (status == DORA_OK && !dora_type_is_equal(&original.type, &field->type, field->child_count)) {
        image->node.content_state = VECTOR_CONTENT_PROMISED;
    }
    struct dora_vector* const children = dora_vector_image_children(image);
    for (uint32_t index = 0; status == DORA_OK && index < image->child_vectors.entry_count; ++index) {
        struct dora_vector child = {};
        struct dora_vector annotated = {};
        if (rewritten_children != nullptr) {
            child = dora_vector_borrow(&rewritten_children[index]);
        } else {
            status = field->layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, index, &child)
                                                         : dora_vector_get_physical_child(source, index, &child);
        }
        if (status == DORA_OK && rewritten_children == nullptr && field->type.extension_name.byte_count != 0 &&
            ((field->layout == DORA_LAYOUT_DICTIONARY && index == 0) ||
             (field->layout == DORA_LAYOUT_RUN_END && index == 1))) {
            struct dora_logical_type child_type = {};
            status = dora_vector_get_type(&child, &child_type);
            struct dora_vector plain_child = child;
            const bool same_extension =
                status == DORA_OK &&
                (bytes_are_well_formed(child_type.extension_name) &&
                 bytes_are_well_formed(field->type.extension_name) &&
                 bytes_are_equal(child_type.extension_name, field->type.extension_name)) &&
                (bytes_are_well_formed(child_type.extension_metadata) &&
                 bytes_are_well_formed(field->type.extension_metadata) &&
                 bytes_are_equal(child_type.extension_metadata, field->type.extension_metadata));
            if (status == DORA_OK && child_type.extension_name.byte_count != 0 && !same_extension) {
                plain_child = (struct dora_vector){};
                status = dora_vector_get_storage(&child, &plain_child);
            }
            struct dora_schema declared_child = {};
            if (status == DORA_OK && root_schema != nullptr) {
                if (use_root) {
                    status = dora_schema_get_physical_child(root_schema, index, &declared_child);
                }
            } else if (status == DORA_OK && values_schema != nullptr) {
                declared_child = dora_schema_borrow(values_schema);
            }
            if (status == DORA_OK) {
                struct dora_schema_field annotated_field = {};
                status = extension_field(&annotated,
                                         field->type.extension_name,
                                         field->type.extension_metadata,
                                         &plain_child,
                                         &annotated_field);
                struct vector_image* annotated_image = nullptr;
                if (status == DORA_OK) {
                    status = vector_share(pool,
                                          &plain_child,
                                          &annotated_field,
                                          declared_child.handle.slot_index == 0 ? nullptr : &declared_child,
                                          nullptr,
                                          nullptr,
                                          &annotated_image,
                                          out_error);
                }
                if (status == DORA_OK) {
                    status = dora_vector_bind_extension(annotated_image, out_error);
                }
                if (status == DORA_OK) {
                    status = dora_vector_publish(annotated_image, &annotated);
                }
                if (status != DORA_OK) {
                    dora_vector_discard(annotated_image);
                }
                child = dora_vector_borrow(&annotated);
            }
        }
        if (status == DORA_OK) {
            struct dora_handle owner = {};
            status = dora_identity_claim_acquire(pool, &child.handle, IDENTITY_KIND_VECTOR, &owner);
            if (status == DORA_OK) {
                children[index] = child;
                children[index].handle = owner;
            }
        }
        dora_vector_release(&annotated);
    }
    input.children = children;
    if (status == DORA_OK) {
        status = image_schema(image, &input, declarations);
    }
    if (status == DORA_OK && field->type.extension_name.byte_count != 0) {
        struct dora_schema_field source_field = {};
        status = dora_schema_get_field(&source_image->schema, &source_field);
        struct dora_vector plain_source = dora_vector_borrow(source);
        if (status == DORA_OK && source_field.type.extension_name.byte_count != 0) {
            plain_source = (struct dora_vector){};
            status = dora_vector_get_storage(source, &plain_source);
        }
        if (status == DORA_OK) {
            status =
                dora_identity_claim_acquire(pool, &plain_source.handle, IDENTITY_KIND_VECTOR, &owned->storage.handle);
            owned->storage.rows = plain_source.rows;
        }
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return status;
    }
    *out_image = image;
    return DORA_OK;
}

// Declared-precision overflow returns ERANGE.
static int decimal_content(uint32_t width, const void* value, const struct dora_i256* bound) {
    bool admitted = false;
    switch (width) {
    case 4: {
        int32_t coefficient = 0;
        memcpy(&coefficient, value, sizeof(coefficient));
        admitted = decimal_bound_admits_i32(coefficient, *bound);
        break;
    }
    case 8: {
        int64_t coefficient = 0;
        memcpy(&coefficient, value, sizeof(coefficient));
        admitted = decimal_bound_admits_i64(coefficient, *bound);
        break;
    }
    case 16: {
        struct dora_i128 coefficient = {};
        memcpy(&coefficient, value, sizeof(coefficient));
        admitted = decimal_bound_admits_i128(coefficient, *bound);
        break;
    }
    default: {
        struct dora_i256 coefficient = {};
        memcpy(&coefficient, value, sizeof(coefficient));
        admitted = decimal_bound_admits_i256(coefficient, *bound);
        break;
    }
    }
    return admitted ? DORA_OK : ERANGE;
}

static int copy_scalar_content(const struct dora_logical_type* type,
                               uint32_t width,
                               const struct dora_i256* bound,
                               const void* value) {
    if (type->id == DORA_TYPE_DECIMAL) {
        return bound == nullptr ? EINVAL : decimal_content(width, value, bound);
    }
    if (type->id == DORA_TYPE_DATE64 || type->id == DORA_TYPE_TIME) {
        int64_t ticks = 0;
        if (width == 4) {
            int32_t small = 0;
            memcpy(&small, value, sizeof(small));
            ticks = small;
        } else {
            memcpy(&ticks, value, sizeof(ticks));
        }
        if (type->id == DORA_TYPE_DATE64) {
            return ticks % INT64_C(86400000) == 0 ? DORA_OK : ERANGE;
        }
        constexpr int64_t units[] = {1, 1000, 1000000, 1000000000};
        return ticks >= 0 && ticks < INT64_C(86400) * units[type->parameters.unit] ? DORA_OK : ERANGE;
    }
    return DORA_OK;
}

// Never reads values behind null rows.
static void copy_valid_values(void* destination,
                              const void* values,
                              uint32_t width,
                              const uint64_t* valid_words,
                              uint64_t row_count,
                              bool bits) {
    const uint64_t word_count = bitmap_words_for_bits(row_count);
    for (uint64_t word = 0; word < word_count; ++word) {
        const uint64_t first = word * DORA_BITMAP_WORD_BIT_COUNT;
        const uint64_t count =
            row_count - first < DORA_BITMAP_WORD_BIT_COUNT ? row_count - first : DORA_BITMAP_WORD_BIT_COUNT;
        uint64_t valid = valid_words[word];
        if (valid == bitmap_mask_low((uint32_t)count)) {
            if (bits) {
                dora_bitmap_copy_bool8_to_words((uint64_t*)destination + word, values, first, count);
            } else {
                memcpy((uint8_t*)destination + (size_t)first * width,
                       (const uint8_t*)values + (size_t)first * width,
                       (size_t)count * width);
            }
            continue;
        }
        while (valid != 0) {
            const uint64_t row = first + dora_bitmap_word_pop_lowest_set(&valid);
            if (bits) {
                bitmap_assign(destination, row, ((const bool*)values)[row]);
            } else {
                memcpy((uint8_t*)destination + (size_t)row * width,
                       (const uint8_t*)values + (size_t)row * width,
                       width);
            }
        }
    }
}

static struct dora_bytes copy_bytes_at(const struct dora_bytes* values,
                                       const char* const* strings,
                                       const bool* validity,
                                       uint64_t row,
                                       bool* out_null) {
    *out_null = validity != nullptr && !validity[row];
    if (*out_null) {
        return (struct dora_bytes){};
    }
    if (strings != nullptr) {
        *out_null = strings[row] == nullptr;
        return *out_null
                   ? (struct dora_bytes){}
                   : (struct dora_bytes){.data = (const uint8_t*)strings[row], .byte_count = strlen(strings[row])};
    }
    return values != nullptr ? values[row] : (struct dora_bytes){};
}

static int copy_bytes_fill(struct dora_pool pool,
                           struct vector_input* input,
                           struct dora_allocation* buffers,
                           const struct dora_bytes* values,
                           const char* const* strings,
                           const bool* validity,
                           size_t total,
                           bool check_content,
                           bool null_invalid) {
    const struct dora_logical_type* const type = &input->field->type;
    const enum dora_layout layout = input->field->layout;
    const uint64_t row_count = input->row_count;
    const bool views = layout == DORA_LAYOUT_BINARY_VIEW;
    const bool fixed = layout == DORA_LAYOUT_FIXED;
    const uint32_t offset_width = layout == DORA_LAYOUT_BINARY ? 4 : 8;
    if (values == nullptr && strings == nullptr && input->row_count != 0) {
        input->content_state = !(check_content)             ? VECTOR_CONTENT_PROMISED
                               : type->id == DORA_TYPE_UTF8 ? VECTOR_CONTENT_VALIDATED_ASCII
                                                            : VECTOR_CONTENT_VALIDATED;
        return input->null_count == (int64_t)input->row_count ? DORA_OK : EINVAL;
    }
    int64_t null_count = input->null_count;
    int status = DORA_OK;
    size_t current = 0;
    bool all_ascii = true;
    const bool checks_utf8 = type->id == DORA_TYPE_UTF8 && check_content;
    const bool packed_utf8 =
        type->id == DORA_TYPE_UTF8 && !(null_invalid) && !views && strings == nullptr && values != nullptr;
    if (status == DORA_OK && packed_utf8) {
        if (null_count != 0) {
            for (uint64_t row = 0; row < row_count; ++row) {
                bitmap_assign(buffers[0].address, row, validity == nullptr || validity[row]);
            }
        }
        size_t begin = 0;
        for (uint64_t first = 0; status == DORA_OK && first < row_count; first += 2048) {
            const uint32_t count = (uint32_t)(row_count - first > 2048 ? 2048 : row_count - first);
            const struct dora_utf8_batch batch = {
                .row_count = count,
                .physical_value_count = count,
                .layout = DORA_UTF8_DESCRIPTORS,
                .proof = (check_content) ? DORA_UTF8_CHECKED : DORA_UTF8_PROMISED,
                .descriptors = values + first,
                .eligible_rows = null_count == 0
                                     ? (struct dora_utf8_mask){}
                                     : (struct dora_utf8_mask){{buffers[0].address, buffers[0].capacity_bytes}, first}
            };
            const struct dora_utf8_batch_output output = {
                .layout = offset_width == 4 ? DORA_UTF8_PACKED_I32 : DORA_UTF8_PACKED_I64,
                .packed.bytes = {buffers[2].address, buffers[2].capacity_bytes},
                .packed.offset_bytes =
                    {(uint8_t*)buffers[1].address + first * offset_width, ((size_t)count + 1) * offset_width},
                .packed.byte_offset = begin
            };
            bool ascii = false;
            status = dora_utf8_copy_batch_sized(&batch, total - begin, &output, nullptr, &ascii);
            all_ascii &= ascii;
            if (status == DORA_OK) {
                begin = offset_width == 4 ? (size_t)((const int32_t*)buffers[1].address)[first + count]
                                          : (size_t)((const int64_t*)buffers[1].address)[first + count];
            }
        }
    }
    uint32_t data_index = 0;
    current = 0;
    for (uint64_t row = 0; status == DORA_OK && !packed_utf8 && row < row_count; ++row) {
        const bool is_null = (validity != nullptr && !validity[row]) || (strings != nullptr && strings[row] == nullptr);
        if (is_null) {
            if (!fixed && !views) {
                if (offset_width == 4) {
                    ((int32_t*)buffers[1].address)[row + 1] = (int32_t)current;
                } else {
                    ((int64_t*)buffers[1].address)[row + 1] = (int64_t)current;
                }
            }
            continue;
        }
        size_t size = type->parameters.byte_width;
        if (views) {
            size = (size_t)((const struct dora_binary_view_entry*)buffers[1].address)[row].byte_count;
        } else if (!fixed && offset_width == 4) {
            size = (size_t)((const int32_t*)buffers[1].address)[row + 1];
        } else if (!fixed) {
            size = (size_t)((const int64_t*)buffers[1].address)[row + 1];
        }
        const struct dora_bytes value = {
            .data = strings != nullptr ? (const uint8_t*)strings[row] : values[row].data,
            .byte_count = size
        };
        if (null_count != 0) {
            bitmap_set(buffers[0].address, row);
        }
        uint8_t* destination = nullptr;
        if (fixed) {
            if (value.byte_count != 0) {
                destination = (uint8_t*)buffers[1].address + (size_t)row * type->parameters.byte_width;
            }
        } else if (views) {
            struct dora_binary_view_entry* const entry = &((struct dora_binary_view_entry*)buffers[1].address)[row];
            if (value.byte_count <= VIEW_INLINE_BYTE_COUNT) {
                destination = entry->payload.inline_bytes;
            } else {
                if (value.byte_count > (size_t)INT32_MAX - current) {
                    ++data_index;
                    current = 0;
                }
                memcpy(entry->payload.reference.prefix, value.data, 4);
                entry->payload.reference.buffer_index = (int32_t)data_index;
                entry->payload.reference.byte_offset = (int32_t)current;
                destination = (uint8_t*)buffers[2 + data_index].address + current;
                current += value.byte_count;
            }
        } else {
            if (value.byte_count != 0) {
                destination = (uint8_t*)buffers[2].address + current;
            }
            current += value.byte_count;
            if (offset_width == 4) {
                ((int32_t*)buffers[1].address)[row + 1] = (int32_t)current;
            } else {
                ((int64_t*)buffers[1].address)[row + 1] = (int64_t)current;
            }
        }
        if (checks_utf8) {
            bool ascii = false;
            status = dora_utf8_validate_span(value, &ascii);
            if (status == EILSEQ && null_invalid) {
                if (null_count == 0) {
                    status = allocate_values(pool, bitmap_words_for_bits(row_count), sizeof(uint64_t), &buffers[0]);
                    if (status != DORA_OK) {
                        break;
                    }
                    bitmap_fill_bytes(buffers[0].address, buffers[0].capacity_bytes, true);
                }
                bitmap_clear(buffers[0].address, row);
                ++null_count;
                status = DORA_OK;
                continue;
            }
            if (status != DORA_OK) {
                break;
            }
            all_ascii &= ascii;
        }
        utf8_copy_validated_bytes(value, destination);
    }
    input->null_count = null_count;
    input->content_state = !(check_content)                          ? VECTOR_CONTENT_PROMISED
                           : type->id == DORA_TYPE_UTF8 && all_ascii ? VECTOR_CONTENT_VALIDATED_ASCII
                                                                     : VECTOR_CONTENT_VALIDATED;
    return status;
}

static int retained_field_check(const struct vector_image* image,
                                const struct dora_schema_field* field,
                                enum vector_content_state content_state) {
    struct dora_schema_field source = {};
    int status = dora_schema_get_field(&image->schema, &source);
    if (status != DORA_OK) {
        return status;
    }
    if (source.layout != field->layout || source.index_type != field->index_type || source.flags != field->flags ||
        source.child_count != field->child_count) {
        return EINVAL;
    }
    if (content_state >= VECTOR_CONTENT_VALIDATED &&
        (image->node.content_state < content_state || source.type.id != field->type.id ||
         !(bytes_are_well_formed(source.type.extension_name) && bytes_are_well_formed(field->type.extension_name) &&
           bytes_are_equal(source.type.extension_name, field->type.extension_name)) ||
         !(bytes_are_well_formed(source.type.extension_metadata) &&
           bytes_are_well_formed(field->type.extension_metadata) &&
           bytes_are_equal(source.type.extension_metadata, field->type.extension_metadata)))) {
        return EINVAL;
    }
    uint32_t union_children = source.child_count;
    if (source.type.id == DORA_TYPE_UNION) {
        struct dora_schema values = dora_schema_borrow(&image->schema);
        struct dora_schema_field value = source;
        while (status == DORA_OK && (value.layout == DORA_LAYOUT_DICTIONARY || value.layout == DORA_LAYOUT_RUN_END)) {
            struct dora_schema child = {};
            status = dora_schema_get_physical_child(&values, value.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
            if (status == DORA_OK) {
                values = child;
                status = dora_schema_get_field(&values, &value);
            }
        }
        union_children = value.child_count;
    }
    struct dora_logical_type target = field->type;
    if ((source.type.id == DORA_TYPE_BINARY || source.type.id == DORA_TYPE_UTF8) &&
        (target.id == DORA_TYPE_BINARY || target.id == DORA_TYPE_UTF8)) {
        source.type.id = target.id;
    }
    source.type.extension_name = (struct dora_bytes){};
    source.type.extension_metadata = (struct dora_bytes){};
    target.extension_name = (struct dora_bytes){};
    target.extension_metadata = (struct dora_bytes){};
    if (status == DORA_OK && !dora_type_is_equal(&source.type, &target, union_children)) {
        status = EINVAL;
    }
    return status;
}

int dora_vector_image_share(struct dora_pool pool,
                            const struct dora_vector* source,
                            const struct dora_schema_field* field,
                            struct vector_image** out_image,
                            struct dora_error* out_error) {
    return vector_share(pool, source, field, nullptr, nullptr, nullptr, out_image, out_error);
}

int dora_vector_init_retain_schema_error(struct dora_vector* vector,
                                         struct dora_pool pool,
                                         const struct dora_vector* source,
                                         const struct dora_schema* schema,
                                         enum vector_content_state content_state,
                                         struct dora_error* out_error) {
    struct vector_image* source_image = nullptr;
    if (!dora_vector_is_empty(vector) || dora_vector_resolve(source, &source_image) != DORA_OK ||
        content_state > VECTOR_CONTENT_VALIDATED_ASCII) {
        return EINVAL;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status == DORA_OK) {
        status = retained_field_check(source_image, &field, content_state);
    }
    struct dora_allocation workspace = {};
    struct dora_vector* children = nullptr;
    if (status == DORA_OK && field.child_count != 0) {
        size_t bytes = 0;
        if (ckd_mul(&bytes, field.child_count, sizeof(struct dora_vector))) {
            status = EOVERFLOW;
        } else {
            status = dora_memory_payload_allocate(pool, bytes, &workspace);
        }
        if (status == DORA_OK) {
            children = workspace.address;
            memset(children, 0, bytes);
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema declaration = {};
        struct dora_vector child = {};
        status = dora_schema_get_physical_child(schema, index, &declaration);
        if (status == DORA_OK) {
            status = field.layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, index, &child)
                                                        : dora_vector_get_physical_child(source, index, &child);
        }
        if (status == DORA_OK) {
            status = dora_vector_init_retain_schema_error(&children[index],
                                                          pool,
                                                          &child,
                                                          &declaration,
                                                          content_state,
                                                          out_error);
        }
    }
    struct vector_image* image = nullptr;
    struct dora_vector storage = {};
    const bool annotated = field.type.extension_name.byte_count != 0;
    if (status == DORA_OK && annotated) {
        struct dora_vector plain_source = dora_vector_borrow(source);
        struct dora_schema_field source_field = {};
        status = dora_schema_get_field(&source_image->schema, &source_field);
        if (status == DORA_OK && source_field.type.extension_name.byte_count != 0) {
            plain_source = (struct dora_vector){};
            status = dora_vector_get_storage(source, &plain_source);
        }
        struct dora_schema_field plain_field = field;
        plain_field.type.extension_name = (struct dora_bytes){};
        plain_field.type.extension_metadata = (struct dora_bytes){};
        struct dora_vector storage_children[2] = {};
        const struct dora_vector* plain_children = children;
        if (status == DORA_OK && (field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END)) {
            const uint32_t value_child = field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1;
            for (uint32_t index = 0; index < field.child_count; ++index) {
                storage_children[index] = dora_vector_borrow(&children[index]);
            }
            storage_children[value_child] = (struct dora_vector){};
            status = dora_vector_get_storage(&children[value_child], &storage_children[value_child]);
            plain_children = storage_children;
        }
        if (status == DORA_OK) {
            status =
                vector_share(pool, &plain_source, &plain_field, nullptr, plain_children, nullptr, &image, out_error);
        }
        if (status == DORA_OK) {
            image->node.content_state = content_state;
            status = dora_vector_publish(image, &storage);
            if (status == DORA_OK) {
                image = nullptr;
            }
        }
    }
    if (status == DORA_OK) {
        status =
            vector_share(pool, annotated ? &storage : source, &field, schema, children, nullptr, &image, out_error);
    }
    if (status == DORA_OK) {
        image->node.content_state = content_state;
        status = dora_vector_bind_extension(image, out_error);
    }
    dora_vector_release(&storage);
    for (uint32_t index = 0; children != nullptr && index < field.child_count; ++index) {
        dora_vector_release(&children[index]);
    }
    dora_allocation_release(&workspace);
    if (status == DORA_OK) {
        status = dora_vector_publish(image, vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Construction by Copy
//
//------------------------------------------------------------------------------

int dora_vector_prepare(struct dora_pool pool,
                        const struct vector_input* input,
                        struct vector_image** out_image,
                        struct dora_error* out_error) {
    if (input == nullptr || input->field == nullptr || out_image == nullptr ||
        input->field->child_count != input->child_count || (input->child_count != 0 && input->children == nullptr) ||
        (input->buffer_count != 0 && input->buffers == nullptr && input->allocations == nullptr)) {
        return EINVAL;
    }
    if (input->field->type.extension_name.byte_count != 0) {
        struct dora_schema_field field = *input->field;
        field.type.extension_name = (struct dora_bytes){};
        field.type.extension_metadata = (struct dora_bytes){};
        struct vector_input storage_input = *input;
        storage_input.field = &field;
        storage_input.admitted_schema = input->storage_schema;
        storage_input.storage_schema = nullptr;
        struct dora_vector storage_children[2] = {};
        struct dora_schema value_schema = {};
        if (field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END) {
            if (field.child_count != (field.layout == DORA_LAYOUT_DICTIONARY ? 1U : 2U)) {
                return EINVAL;
            }
            for (uint32_t index = 0; index < field.child_count; ++index) {
                storage_children[index] = dora_vector_borrow(&input->children[index]);
            }
            const uint32_t value_child = field.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1;
            struct vector_image* value_image = nullptr;
            int status = dora_vector_resolve(&input->children[value_child], &value_image);
            if (status == DORA_OK) {
                value_schema = dora_schema_borrow(&value_image->schema);
                storage_children[value_child] = (struct dora_vector){};
                status = dora_vector_get_storage(&input->children[value_child], &storage_children[value_child]);
            }
            if (status != DORA_OK) {
                return status;
            }
            storage_input.children = storage_children;
        }
        struct vector_image* storage_image = nullptr;
        struct vector_image* image = nullptr;
        struct dora_vector storage = {};
        int status = dora_vector_prepare(pool, &storage_input, &storage_image, out_error);
        if (status == DORA_OK) {
            status = dora_vector_publish(storage_image, &storage);
            if (status == DORA_OK) {
                storage_image = nullptr;
            }
        }
        if (status == DORA_OK) {
            status = vector_share(pool,
                                  &storage,
                                  input->field,
                                  input->admitted_schema,
                                  nullptr,
                                  value_schema.handle.slot_index == 0 ? nullptr : &value_schema,
                                  &image,
                                  out_error);
        }
        if (status == DORA_OK) {
            status = dora_vector_bind_extension(image, out_error);
        }
        dora_vector_release(&storage);
        dora_vector_discard(storage_image);
        if (status != DORA_OK) {
            dora_vector_discard(image);
            return status;
        }
        image->node.content_state = input->content_state;
        *out_image = image;
        return DORA_OK;
    }
    struct vector_image* image = nullptr;
    int status = image_allocate(pool, input, &image);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_vector* const children = dora_vector_image_children(image);
    for (uint32_t index = 0; status == DORA_OK && index < input->child_count; ++index) {
        struct vector_image* child_image = nullptr;
        status = dora_vector_resolve(&input->children[index], &child_image);
        if (status == DORA_OK) {
            struct dora_handle owner = {};
            status = dora_identity_claim_acquire(pool, &input->children[index].handle, IDENTITY_KIND_VECTOR, &owner);
            if (status == DORA_OK) {
                children[index] = input->children[index];
                children[index].handle = owner;
            }
        }
    }
    if (status == DORA_OK) {
        status = image_schema(image, input, nullptr);
    }
    struct dora_view_buffer* const views =
        (struct dora_view_buffer*)(image->table_bytes + image->view_buffers_byte_offset);
    // Admission callbacks require reset and release to stay blocked.
    dora_pool_callback_enter(pool);
    if (status == DORA_OK) {
        struct vector_input admission = *input;
        admission.children = children;
        status = dora_vector_admit(pool, &image->schema, &image->ownership, &admission, views);
    }
    if (status == DORA_OK) {
        status = dora_vector_derive_children(&image, input->are_children_folded, out_error);
    }
    dora_pool_callback_leave(pool);
    if (status == DORA_OK) {
        status = dora_vector_bind_extension(image, out_error);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return status;
    }
    *out_image = image;
    return DORA_OK;
}

int dora_vector_init_move_input(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct vector_input* input,
                                struct dora_allocation* in_out_buffers,
                                struct dora_error* out_error) {
    const uint32_t buffer_count = input->buffer_count;
    if (!dora_vector_is_empty(vector) || (buffer_count != 0 && in_out_buffers == nullptr)) {
        return EINVAL;
    }
    for (uint32_t index = 0; index < buffer_count; ++index) {
        const struct dora_allocation* const buffer = &in_out_buffers[index];
        if (buffer->capacity_bytes != 0 && (buffer->address == nullptr || buffer->pool.index != pool.index ||
                                            buffer->pool.generation != pool.generation)) {
            return EINVAL;
        }
        if (buffer->capacity_bytes == 0 && buffer->address != nullptr) {
            return EINVAL;
        }
        for (uint32_t other = 0; buffer->address != nullptr && other < index; ++other) {
            if (buffer->address == in_out_buffers[other].address) {
                return EINVAL;
            }
        }
    }
    if (input->effective_validity != nullptr) {
        const struct dora_allocation* const effective = input->effective_validity;
        if ((effective->capacity_bytes != 0 && (effective->address == nullptr || effective->pool.index != pool.index ||
                                                effective->pool.generation != pool.generation)) ||
            (effective->capacity_bytes == 0 && effective->address != nullptr) ||
            input->effective_null_count < input->null_count || input->effective_null_count < 0 ||
            (uint64_t)input->effective_null_count > input->row_count) {
            return EINVAL;
        }
        for (uint32_t index = 0; index < buffer_count; ++index) {
            if (effective->address != nullptr && effective->address == in_out_buffers[index].address) {
                return EINVAL;
            }
        }
    }
    struct vector_image* image = nullptr;
    struct dora_vector published = {};
    int status = dora_vector_prepare(pool, input, &image, out_error);
    if (status == DORA_OK) {
        status = dora_vector_publish(image, &published);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return status;
    }
    struct vector_buffer_row* const buffers = dora_vector_image_buffers(image->buffer_image);
    for (uint32_t index = 0; status == DORA_OK && index < buffer_count; ++index) {
        const enum dora_layout layout = image->ownership.view.layout;
        if (index == 0 && input->null_count == 0 && layout != DORA_LAYOUT_NULL && layout != DORA_LAYOUT_RUN_END &&
            layout != DORA_LAYOUT_DENSE_UNION && layout != DORA_LAYOUT_SPARSE_UNION) {
            dora_allocation_release(&in_out_buffers[index]);
            continue;
        }
        if (input->buffer_source != nullptr && in_out_buffers[index].address == nullptr && input->buffers != nullptr &&
            input->buffers[index].byte_count != 0) {
            status = dora_identity_claim_acquire(pool,
                                                 &input->buffer_source->handle,
                                                 IDENTITY_KIND_VECTOR,
                                                 &buffers[index].owner.retained);
            if (status == DORA_OK) {
                buffers[index].owner_kind = VECTOR_BUFFER_OWNER_RETAINED;
            }
        } else {
            status = dora_allocation_adopt(&in_out_buffers[index], &buffers[index].owner.allocation);
        }
    }
    if (status == DORA_OK && input->effective_validity != nullptr) {
        status = dora_allocation_adopt(input->effective_validity, &image->buffer_image->ownership.effective_validity);
    }
    if (status == DORA_OK && !dora_vector_is_empty(vector)) {
        status = EINVAL;
    }
    if (status != DORA_OK) {
        dora_vector_release(&published);
        return status;
    }
    *vector = published;
    return DORA_OK;
}

static int copy_publish(struct dora_vector* vector,
                        struct dora_pool pool,
                        const struct vector_input* input,
                        struct dora_allocation* buffers,
                        enum dora_failure_policy content_policy) {
    if (input->field->type.extension_name.byte_count == 0) {
        return dora_vector_init_move_input(vector, pool, input, buffers, nullptr);
    }
    struct dora_schema_field field = *input->field;
    field.type.extension_name = (struct dora_bytes){};
    field.type.extension_metadata = (struct dora_bytes){};
    struct vector_input storage_input = *input;
    storage_input.field = &field;
    struct dora_vector storage = {};
    int status = dora_vector_init_move_input(&storage, pool, &storage_input, buffers, nullptr);
    struct dora_schema schema = {};
    struct dora_schema_field declared = {};
    const struct dora_extension_definition* definition = nullptr;
    struct dora_allocation parameters = {};
    if (status == DORA_OK) {
        status = dora_schema_init_retain_node(&schema, pool, input->field, nullptr);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(&schema, &declared);
    }
    if (status == DORA_OK) {
        status = dora_extension_find(pool, declared.type.extension_name, &definition);
    }
    struct dora_schema storage_schema = {};
    if (status == DORA_OK && definition != nullptr && definition->bind != nullptr) {
        status = dora_vector_retain_schema(&storage, &storage_schema);
    }
    if (status == DORA_OK && definition != nullptr) {
        status = dora_vector_bind_parameters(definition, pool, &storage_schema, &declared, &parameters, nullptr);
    }
    dora_schema_release(&storage_schema);
    struct dora_allocation validity = {};
    bool validated = false;
    if (status == DORA_OK && content_policy != DORA_FAILURE_POLICY_TRUST && definition != nullptr &&
        definition->validate != nullptr) {
        struct dora_vector_view view = {};
        status = dora_vector_get_view(&storage, &view);
        if (status == DORA_OK) {
            status = allocate_values(pool, bitmap_words_for_bits(input->row_count), sizeof(uint64_t), &validity);
        }
        if (status == DORA_OK) {
            if (validity.capacity_bytes != 0) {
                bitmap_fill_bytes(validity.address, validity.capacity_bytes, true);
                if (view.validity.data != nullptr) {
                    dora_bitmap_load_words(view.validity.data,
                                           view.validity_bit_offset,
                                           input->row_count,
                                           validity.address);
                }
            }
            dora_pool_callback_enter(pool);
            status = definition->validate(definition, pool, &parameters, &storage, validity.address, nullptr);
            dora_pool_callback_leave(pool);
        }
        if (validity.address != nullptr && content_policy == DORA_FAILURE_POLICY_NULL &&
            (status == DORA_OK || status == EILSEQ || status == ERANGE)) {
            if (view.validity.data != nullptr) {
                dora_bitmap_and_aligned_words(validity.address,
                                              (const uint64_t*)view.validity.data,
                                              bitmap_words_for_bits(input->row_count));
            }
            const int64_t null_count =
                (int64_t)(input->row_count - dora_bitmap_count_set_range(validity.address, 0, input->row_count));
            status = DORA_OK;
            if (null_count != input->null_count) {
                struct vector_image* masked_image = nullptr;
                status = vector_share(pool, &storage, &field, nullptr, nullptr, nullptr, &masked_image, nullptr);
                if (status == DORA_OK) {
                    struct vector_node_ownership* const owned = &masked_image->ownership;
                    status = dora_allocation_adopt(&validity, &owned->effective_validity);
                    if (status == DORA_OK) {
                        owned->own_validity = (struct dora_view_buffer){
                            .data = owned->effective_validity.address,
                            .byte_count = bitmap_bytes_for_bits(input->row_count),
                            .readable_byte_count = owned->effective_validity.capacity_bytes
                        };
                        owned->own_validity_bit_offset = 0;
                        owned->own_null_count = null_count;
                        owned->view.validity = owned->own_validity;
                        owned->view.validity_bit_offset = 0;
                        owned->view.null_count = null_count;
                    }
                }
                struct dora_vector masked = {};
                if (status == DORA_OK) {
                    status = dora_vector_publish(masked_image, &masked);
                }
                if (status == DORA_OK) {
                    dora_vector_release(&storage);
                    storage = masked;
                } else {
                    dora_vector_discard(masked_image);
                }
            }
        }
        if (status == DORA_OK) {
            struct vector_image* storage_image = nullptr;
            status = dora_vector_resolve(&storage, &storage_image);
            validated = status == DORA_OK && storage_image->node.content_state >= VECTOR_CONTENT_VALIDATED;
        }
    }
    struct vector_image* image = nullptr;
    if (status == DORA_OK) {
        status = vector_share(pool, &storage, &declared, &schema, nullptr, nullptr, &image, nullptr);
    }
    if (status == DORA_OK) {
        image->ownership.extension = definition;
        status = dora_allocation_move(&parameters, &image->ownership.extension_parameters);
        image->node.content_state = content_policy == DORA_FAILURE_POLICY_TRUST ? input->content_state
                                    : validated                                 ? VECTOR_CONTENT_VALIDATED
                                                                                : VECTOR_CONTENT_PROMISED;
    }
    dora_allocation_release(&parameters);
    dora_allocation_release(&validity);
    dora_schema_release(&schema);
    dora_vector_release(&storage);
    if (status == DORA_OK) {
        status = dora_vector_publish(image, vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    return status;
}

//--------------------------------------
// Boolean
//--------------------------------------

static int copy_fixed(struct dora_vector* vector,
                      struct dora_pool pool,
                      const struct dora_logical_type* type,
                      enum dora_physical_type expected,
                      const void* values,
                      uint32_t width,
                      const bool* validity,
                      uint64_t row_count) {
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    if (!dora_vector_is_empty(vector) || row_count > INT64_MAX || type_physical(type, &physical) != DORA_OK ||
        physical != expected || (expected == DORA_PHYSICAL_TYPE_NONE && type->id != DORA_TYPE_BOOLEAN)) {
        return EINVAL;
    }
    size_t source_bytes = 0;
    if (ckd_mul(&source_bytes, row_count, width)) {
        return EOVERFLOW;
    }
    const bool bits = type->id == DORA_TYPE_BOOLEAN;
    const bool checked = type->id == DORA_TYPE_DECIMAL || type->id == DORA_TYPE_DATE64 || type->id == DORA_TYPE_TIME;
    struct dora_allocation buffers[2] = {};
    int64_t null_count = 0;
    int status = pack_validity(pool, validity, row_count, &buffers[0], &null_count);
    if (status == DORA_OK) {
        status = allocate_values(pool,
                                 bits ? bitmap_words_for_bits(row_count) : row_count,
                                 bits ? sizeof(uint64_t) : width,
                                 &buffers[1]);
    }
    if (status == DORA_OK && row_count != 0 && values == nullptr && (uint64_t)null_count != row_count) {
        status = EINVAL;
    }
    if (status == DORA_OK && buffers[1].capacity_bytes != 0) {
        memset(buffers[1].address, 0, buffers[1].capacity_bytes);
    }
    if (status == DORA_OK && row_count != 0 && values != nullptr) {
        if (buffers[0].address != nullptr) {
            copy_valid_values(buffers[1].address, values, width, buffers[0].address, row_count, bits);
        } else if (bits) {
            dora_bitmap_copy_bool8_to_words(buffers[1].address, values, 0, row_count);
        } else {
            memcpy(buffers[1].address, values, (size_t)row_count * width);
        }
    }
    struct dora_i256 bound = {};
    const bool bounded = type->id != DORA_TYPE_DECIMAL ||
                         dora_decimal_bound_prepare_width(type->parameters.decimal, width * 8, &bound) == DORA_OK;
    for (uint64_t row = 0; status == DORA_OK && checked && row < row_count; ++row) {
        if (buffers[0].address != nullptr && !dora_bitmap_test((const uint8_t*)buffers[0].address, row)) {
            continue;
        }
        status = copy_scalar_content(type,
                                     width,
                                     bounded ? &bound : nullptr,
                                     (const uint8_t*)buffers[1].address + (size_t)row * width);
    }
    const struct dora_schema_field field = {.type = *type, .layout = bits ? DORA_LAYOUT_BITS : DORA_LAYOUT_FIXED};
    const struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = null_count,
        .allocations = buffers,
        .buffer_count = 2,
        .content_state = VECTOR_CONTENT_VALIDATED
    };
    if (status == DORA_OK) {
        status = copy_publish(vector, pool, &input, buffers, DORA_FAILURE_POLICY_ERROR);
    }
    dora_allocation_release(&buffers[0]);
    dora_allocation_release(&buffers[1]);
    return status;
}

int dora_vector_init_copy_bool(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const bool* values,
                               const bool* validity,
                               uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_NONE, values, sizeof(bool), validity, row_count);
}

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_vector_init_copy_i8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const int8_t* values,
                             const bool* validity,
                             uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_I8, values, sizeof(int8_t), validity, row_count);
}

int dora_vector_init_copy_i16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int16_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_I16, values, sizeof(int16_t), validity, row_count);
}

int dora_vector_init_copy_i32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int32_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_I32, values, sizeof(int32_t), validity, row_count);
}

int dora_vector_init_copy_i64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int64_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_I64, values, sizeof(int64_t), validity, row_count);
}

int dora_vector_init_copy_i128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i128* values,
                               const bool* validity,
                               uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_I128,
                      values,
                      sizeof(struct dora_i128),
                      validity,
                      row_count);
}

int dora_vector_init_copy_i256(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i256* values,
                               const bool* validity,
                               uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_I256,
                      values,
                      sizeof(struct dora_i256),
                      validity,
                      row_count);
}

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_vector_init_copy_u8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const uint8_t* values,
                             const bool* validity,
                             uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_U8, values, sizeof(uint8_t), validity, row_count);
}

int dora_vector_init_copy_u16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint16_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_U16, values, sizeof(uint16_t), validity, row_count);
}

int dora_vector_init_copy_u32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint32_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_U32, values, sizeof(uint32_t), validity, row_count);
}

int dora_vector_init_copy_u64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint64_t* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_U64, values, sizeof(uint64_t), validity, row_count);
}

int dora_vector_init_copy_u128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_u128* values,
                               const bool* validity,
                               uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_U128,
                      values,
                      sizeof(struct dora_u128),
                      validity,
                      row_count);
}

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_vector_init_copy_f16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const struct dora_f16_bits* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_F16,
                      values,
                      sizeof(struct dora_f16_bits),
                      validity,
                      row_count);
}

int dora_vector_init_copy_f32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const float* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_F32, values, sizeof(float), validity, row_count);
}

int dora_vector_init_copy_f64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const double* values,
                              const bool* validity,
                              uint64_t row_count) {
    return copy_fixed(vector, pool, type, DORA_PHYSICAL_TYPE_F64, values, sizeof(double), validity, row_count);
}

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_vector_init_copy_interval_day_time(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            const struct dora_logical_type* type,
                                            const struct dora_interval_day_time* values,
                                            const bool* validity,
                                            uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME,
                      values,
                      sizeof(struct dora_interval_day_time),
                      validity,
                      row_count);
}

int dora_vector_init_copy_interval_month_day_nano(struct dora_vector* vector,
                                                  struct dora_pool pool,
                                                  const struct dora_logical_type* type,
                                                  const struct dora_interval_month_day_nano* values,
                                                  const bool* validity,
                                                  uint64_t row_count) {
    return copy_fixed(vector,
                      pool,
                      type,
                      DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO,
                      values,
                      sizeof(struct dora_interval_month_day_nano),
                      validity,
                      row_count);
}

static int copy_bytes(struct dora_vector* vector,
                      struct dora_pool pool,
                      const struct dora_logical_type* type,
                      enum dora_layout layout,
                      const struct dora_bytes* values,
                      const char* const* strings,
                      const bool* validity,
                      uint64_t row_count,
                      enum dora_failure_policy content_policy,
                      enum vector_content_state standing) {
    if (content_policy == DORA_FAILURE_POLICY_WRAP || content_policy > DORA_FAILURE_POLICY_TRUST ||
        !dora_vector_is_empty(vector) || type == nullptr || row_count > INT64_MAX ||
        (type->id != DORA_TYPE_UTF8 && type->id != DORA_TYPE_BINARY && type->id != DORA_TYPE_FIXED_BINARY)) {
        return EINVAL;
    }
    struct dora_schema_field field = {.type = *type, .layout = layout};
    int status = dora_type_resolve_layout(type, layout, DORA_TYPE_NULL, &layout);
    if (status != DORA_OK) {
        return status;
    }
    field.layout = layout;
    size_t input_bytes = 0;
    if (ckd_mul(&input_bytes, row_count, strings != nullptr ? sizeof(*strings) : sizeof(*values))) {
        return EOVERFLOW;
    }
    const bool views = layout == DORA_LAYOUT_BINARY_VIEW;
    const bool fixed = layout == DORA_LAYOUT_FIXED;
    const uint32_t offset_width = layout == DORA_LAYOUT_BINARY ? 4 : 8;
    struct dora_allocation sizes = {};
    if (!fixed) {
        status = allocate_values(pool,
                                 row_count + (views ? 0 : 1),
                                 views ? sizeof(struct dora_binary_view_entry) : offset_width,
                                 &sizes);
        if (status != DORA_OK) {
            return status;
        }
        if (sizes.capacity_bytes != 0) {
            memset(sizes.address, 0, sizes.capacity_bytes);
        }
    }
    uint32_t data_count = views ? 0 : 1;
    size_t total = 0;
    size_t current = 0;
    int64_t null_count = 0;
    for (uint64_t row = 0; status == DORA_OK && row < row_count; ++row) {
        bool is_null = false;
        const struct dora_bytes value = copy_bytes_at(values, strings, validity, row, &is_null);
        if (is_null) {
            ++null_count;
            continue;
        }
        if ((values == nullptr && strings == nullptr) || (value.byte_count != 0 && value.data == nullptr) ||
            (fixed && value.byte_count != type->parameters.byte_width)) {
            status = EINVAL;
            break;
        }
        if (views && value.byte_count > INT32_MAX) {
            status = EOVERFLOW;
            break;
        }
        if (views && value.byte_count > VIEW_INLINE_BYTE_COUNT) {
            if (data_count == 0 || value.byte_count > (size_t)INT32_MAX - current) {
                if (data_count == INT32_MAX) {
                    status = EOVERFLOW;
                    break;
                }
                ++data_count;
                current = 0;
            }
            current += value.byte_count;
        }
        if (ckd_add(&total, total, value.byte_count)) {
            status = EOVERFLOW;
            break;
        }
        if (views) {
            ((struct dora_binary_view_entry*)sizes.address)[row].byte_count = (int32_t)value.byte_count;
        } else if (!fixed && offset_width == 4) {
            ((int32_t*)sizes.address)[row + 1] = (int32_t)value.byte_count;
        } else if (!fixed) {
            ((int64_t*)sizes.address)[row + 1] = (int64_t)value.byte_count;
        }
    }
    if (status == DORA_OK && ((!views && !fixed && total > (layout == DORA_LAYOUT_BINARY ? INT32_MAX : INT64_MAX)) ||
                              (fixed && ckd_mul(&total, row_count, type->parameters.byte_width)))) {
        status = EOVERFLOW;
    }
    const uint32_t buffer_count = views ? data_count + 3 : fixed ? 2 : 3;
    struct dora_allocation workspace = {};
    if (status == DORA_OK) {
        status = allocate_values(pool, buffer_count, sizeof(struct dora_allocation), &workspace);
    }
    if (status != DORA_OK) {
        dora_allocation_release(&sizes);
        return status;
    }
    struct dora_allocation* const buffers = workspace.address;
    memset(buffers, 0, workspace.capacity_bytes);
    buffers[1] = sizes;
    sizes = (struct dora_allocation){};
    if (null_count != 0) {
        status = allocate_values(pool, bitmap_words_for_bits(row_count), sizeof(uint64_t), &buffers[0]);
        if (status == DORA_OK) {
            bitmap_fill_bytes(buffers[0].address, buffers[0].capacity_bytes, false);
        }
    }
    if (status == DORA_OK && fixed) {
        status = allocate_values(pool, total, 1, &buffers[1]);
        if (status == DORA_OK && buffers[1].capacity_bytes != 0) {
            memset(buffers[1].address, 0, buffers[1].capacity_bytes);
        }
    }
    if (status == DORA_OK && views) {
        status = allocate_values(pool, data_count, sizeof(int64_t), &buffers[buffer_count - 1]);
        if (status == DORA_OK && data_count != 0) {
            memset(buffers[buffer_count - 1].address, 0, buffers[buffer_count - 1].capacity_bytes);
        }
        const struct dora_binary_view_entry* const entries = buffers[1].address;
        uint32_t index = 0;
        current = 0;
        for (uint64_t row = 0; status == DORA_OK && row < row_count; ++row) {
            const size_t size = (size_t)entries[row].byte_count;
            if (size <= VIEW_INLINE_BYTE_COUNT) {
                continue;
            }
            if (size > (size_t)INT32_MAX - current) {
                status = allocate_values(pool, current, 1, &buffers[2 + index]);
                ((int64_t*)buffers[buffer_count - 1].address)[index++] = (int64_t)current;
                current = 0;
            }
            current += size;
        }
        if (status == DORA_OK && data_count != 0) {
            status = allocate_values(pool, current, 1, &buffers[2 + index]);
            ((int64_t*)buffers[buffer_count - 1].address)[index] = (int64_t)current;
        }
    } else if (status == DORA_OK && !fixed) {
        status = allocate_values(pool, total, 1, &buffers[2]);
    }
    struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = null_count,
        .allocations = buffers,
        .buffer_count = buffer_count
    };
    if (status == DORA_OK) {
        switch (content_policy) {
        case DORA_FAILURE_POLICY_ERROR:
            status = copy_bytes_fill(pool, &input, buffers, values, strings, validity, total, true, false);
            break;
        case DORA_FAILURE_POLICY_NULL:
            status = copy_bytes_fill(pool, &input, buffers, values, strings, validity, total, true, true);
            break;
        default:
            status = copy_bytes_fill(pool, &input, buffers, values, strings, validity, total, false, false);
            break;
        }
    }
    if (content_policy == DORA_FAILURE_POLICY_TRUST) {
        input.content_state = standing;
    }
    struct dora_vector published = {};
    if (status == DORA_OK) {
        status = copy_publish(&published, pool, &input, buffers, content_policy);
    }
    for (uint32_t index = 0; index < buffer_count; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    dora_allocation_release(&workspace);
    status = vector_finish_publication(&published, vector, status);
    return status;
}

int dora_vector_init_copy_bytes_standing(struct dora_vector* vector,
                                         struct dora_pool pool,
                                         const struct dora_logical_type* type,
                                         enum dora_layout layout,
                                         const struct dora_bytes* values,
                                         const bool* validity,
                                         uint64_t row_count,
                                         enum vector_content_state standing) {
    return copy_bytes(vector,
                      pool,
                      type,
                      layout,
                      values,
                      nullptr,
                      validity,
                      row_count,
                      DORA_FAILURE_POLICY_TRUST,
                      standing);
}

int dora_vector_init_copy_bytes(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct dora_logical_type* type,
                                enum dora_layout layout,
                                const struct dora_bytes* values,
                                const bool* validity,
                                uint64_t row_count,
                                enum dora_failure_policy content_policy) {
    const enum vector_content_state standing =
        type != nullptr && type->id != DORA_TYPE_UTF8 && type->extension_name.byte_count == 0 ? VECTOR_CONTENT_VALIDATED
                                                                                              : VECTOR_CONTENT_PROMISED;
    return copy_bytes(vector, pool, type, layout, values, nullptr, validity, row_count, content_policy, standing);
}

int dora_vector_init_copy_cstrings(struct dora_vector* vector,
                                   struct dora_pool pool,
                                   const struct dora_logical_type* type,
                                   enum dora_layout layout,
                                   const char* const* values,
                                   uint64_t row_count,
                                   enum dora_failure_policy content_policy) {
    if (row_count != 0 && values == nullptr) {
        return EINVAL;
    }
    const enum vector_content_state standing =
        type != nullptr && type->id != DORA_TYPE_UTF8 && type->extension_name.byte_count == 0 ? VECTOR_CONTENT_VALIDATED
                                                                                              : VECTOR_CONTENT_PROMISED;
    return copy_bytes(vector, pool, type, layout, nullptr, values, nullptr, row_count, content_policy, standing);
}

//------------------------------------------------------------------------------
//
// Composition
//
//------------------------------------------------------------------------------

//--------------------------------------
// Lists
//--------------------------------------

static int publish_constructed(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct vector_input* input,
                               struct dora_allocation* buffers) {
    struct vector_input composition = *input;
    composition.should_check_child_activity = true;
    composition.content_state =
        input->field->type.extension_name.byte_count == 0 ? VECTOR_CONTENT_VALIDATED : VECTOR_CONTENT_PROMISED;
    const int status = dora_vector_init_move_input(vector, pool, &composition, buffers, nullptr);
    for (uint32_t index = 0; index < input->buffer_count; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    return status;
}

static int list_construct(struct dora_vector* vector,
                          struct dora_pool pool,
                          const struct dora_schema_field* field,
                          const struct dora_vector* children,
                          const uint64_t* starts,
                          const uint64_t* counts,
                          const bool* validity,
                          uint64_t row_count) {
    if (!dora_vector_is_empty(vector) || children == nullptr || row_count > INT64_MAX ||
        (row_count != 0 && counts == nullptr)) {
        return EINVAL;
    }
    uint64_t limit = 0;
    int status = dora_vector_get_length(&children[0], &limit);
    if (status != DORA_OK) {
        return status;
    }
    const bool is_view = field->layout == DORA_LAYOUT_LIST_VIEW || field->layout == DORA_LAYOUT_LARGE_LIST_VIEW;
    if (is_view && starts == nullptr && row_count != 0) {
        return EINVAL;
    }
    const uint32_t width = field->layout == DORA_LAYOUT_LIST || field->layout == DORA_LAYOUT_LIST_VIEW ? 4 : 8;
    struct dora_allocation buffers[3] = {};
    int64_t null_count = 0;
    status = pack_validity(pool, validity, row_count, &buffers[0], &null_count);
    if (status == DORA_OK) {
        status = allocate_values(pool, row_count + (is_view ? 0 : 1), width, &buffers[1]);
    }
    if (status == DORA_OK && is_view) {
        status = allocate_values(pool, row_count, width, &buffers[2]);
    }
    uint64_t next = 0;
    for (uint64_t row = 0; status == DORA_OK && row < row_count; ++row) {
        const struct dora_range range = {.offset = is_view ? starts[row] : next, .length = counts[row]};
        const uint64_t begin = range.offset;
        if (begin > limit || range.length > limit - begin ||
            (width == 4 && (begin > INT32_MAX || range.length > INT32_MAX - begin))) {
            status = EINVAL;
            break;
        }
        next = begin + range.length;
        if (width == 4) {
            ((int32_t*)buffers[1].address)[row] = begin;
            if (is_view) {
                ((int32_t*)buffers[2].address)[row] = range.length;
            }
        } else {
            ((int64_t*)buffers[1].address)[row] = begin;
            if (is_view) {
                ((int64_t*)buffers[2].address)[row] = range.length;
            }
        }
    }
    if (status == DORA_OK && !is_view) {
        if (width == 4) {
            ((int32_t*)buffers[1].address)[row_count] = next;
        } else {
            ((int64_t*)buffers[1].address)[row_count] = next;
        }
    }
    const struct vector_input input = {
        .field = field,
        .row_count = row_count,
        .null_count = null_count,
        .allocations = buffers,
        .buffer_count = is_view ? 3 : 2,
        .children = children,
        .child_count = field->child_count,
    };
    if (status == DORA_OK) {
        return publish_constructed(vector, pool, &input, buffers);
    }
    for (uint32_t index = 0; index < 3; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    return status;
}

int dora_vector_init_retain_list(struct dora_vector* vector,
                                 enum dora_layout layout,
                                 const struct dora_vector* elements,
                                 const uint64_t* element_counts,
                                 const bool* validity,
                                 uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(elements, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (layout != DORA_LAYOUT_LIST && layout != DORA_LAYOUT_LARGE_LIST) {
        return EINVAL;
    }
    const struct dora_schema_field field = {.type.id = DORA_TYPE_LIST, .layout = layout, .child_count = 1};
    return list_construct(vector, pool, &field, elements, nullptr, element_counts, validity, row_count);
}

int dora_vector_init_retain_list_view(struct dora_vector* vector,
                                      enum dora_layout layout,
                                      const struct dora_vector* elements,
                                      const uint64_t* element_rows,
                                      const uint64_t* element_counts,
                                      const bool* validity,
                                      uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(elements, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (layout != DORA_LAYOUT_LIST_VIEW && layout != DORA_LAYOUT_LARGE_LIST_VIEW) {
        return EINVAL;
    }
    const struct dora_schema_field field = {.type.id = DORA_TYPE_LIST, .layout = layout, .child_count = 1};
    return list_construct(vector, pool, &field, elements, element_rows, element_counts, validity, row_count);
}

int dora_vector_init_retain_fixed_list(struct dora_vector* vector,
                                       uint32_t list_element_count,
                                       const struct dora_vector* elements,
                                       const bool* validity,
                                       uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(elements, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    uint64_t count = 0;
    uint64_t required = 0;
    if (!dora_vector_is_empty(vector) || dora_vector_get_length(elements, &count) != DORA_OK ||
        ckd_mul(&required, row_count, list_element_count) || required != count) {
        return EINVAL;
    }
    struct dora_allocation buffer = {};
    int64_t null_count = 0;
    const int status = pack_validity(pool, validity, row_count, &buffer, &null_count);
    if (status != DORA_OK) {
        dora_allocation_release(&buffer);
        return status;
    }
    const struct dora_schema_field field = {
        .type = {.id = DORA_TYPE_FIXED_LIST, .parameters.list_element_count = list_element_count},
        .layout = DORA_LAYOUT_FIXED_LIST,
        .child_count = 1
    };
    const struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = null_count,
        .allocations = &buffer,
        .buffer_count = 1,
        .children = elements,
        .child_count = 1
    };
    return publish_constructed(vector, pool, &input, &buffer);
}

//--------------------------------------
// Structures
//--------------------------------------

static int struct_construct(struct dora_vector* vector,
                            struct dora_pool pool,
                            const struct dora_vector* children,
                            const char* const* names,
                            uint32_t child_count,
                            const bool* validity,
                            uint64_t row_count) {
    if (!dora_vector_is_empty(vector) || (child_count != 0 && children == nullptr)) {
        return EINVAL;
    }
    for (uint32_t index = 0; index < child_count; ++index) {
        uint64_t count = 0;
        if (dora_vector_get_length(&children[index], &count) != DORA_OK || count != row_count) {
            return EINVAL;
        }
    }
    struct dora_allocation buffer = {};
    int64_t null_count = 0;
    const int status = pack_validity(pool, validity, row_count, &buffer, &null_count);
    if (status != DORA_OK) {
        dora_allocation_release(&buffer);
        return status;
    }
    const struct dora_schema_field field =
        {.type.id = DORA_TYPE_STRUCT, .layout = DORA_LAYOUT_STRUCT, .child_count = child_count};
    const struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = null_count,
        .allocations = &buffer,
        .buffer_count = 1,
        .children = children,
        .child_names = names,
        .child_count = child_count
    };
    return publish_constructed(vector, pool, &input, &buffer);
}

int dora_vector_init_retain_struct(struct dora_vector* vector,
                                   const struct dora_vector* children,
                                   const char* const* names,
                                   uint32_t child_count,
                                   const bool* validity,
                                   uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool_span(children, child_count, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return struct_construct(vector, pool, children, names, child_count, validity, row_count);
}

int dora_vector_init_fieldless_struct(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const bool* validity,
                                      uint64_t row_count) {
    return struct_construct(vector, pool, nullptr, nullptr, 0, validity, row_count);
}

//--------------------------------------
// Maps
//--------------------------------------

int dora_vector_init_retain_map(struct dora_vector* vector,
                                enum dora_layout layout,
                                bool are_keys_sorted,
                                const struct dora_vector* keys,
                                const struct dora_vector* values,
                                const uint64_t* entry_counts,
                                const bool* validity,
                                uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){keys, values}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    uint64_t key_count = 0;
    uint64_t value_count = 0;
    if (!dora_vector_is_empty(vector) || (layout != DORA_LAYOUT_LIST && layout != DORA_LAYOUT_LARGE_LIST) ||
        dora_vector_get_length(keys, &key_count) != DORA_OK ||
        dora_vector_get_length(values, &value_count) != DORA_OK || key_count != value_count || row_count > INT64_MAX ||
        (row_count != 0 && entry_counts == nullptr)) {
        return EINVAL;
    }
    struct vector_map_scratch scratch = {};
    uint64_t begin = 0;
    int status = DORA_OK;
    for (uint64_t row = 0; status == DORA_OK && row < row_count; ++row) {
        if (entry_counts[row] > key_count - begin) {
            status = EINVAL;
            break;
        }
        if (validity == nullptr || validity[row]) {
            status = dora_vector_check_map_keys(pool,
                                                keys,
                                                (struct dora_range){.offset = begin, .length = entry_counts[row]},
                                                are_keys_sorted,
                                                &scratch);
        }
        begin += entry_counts[row];
    }
    dora_vector_map_scratch_release(&scratch);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_image* key_source = nullptr;
    status = dora_vector_resolve(keys, &key_source);
    struct dora_schema_field key_field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(&key_source->schema, &key_field);
    }
    struct vector_image* key_image = nullptr;
    struct dora_vector declared_keys = {};
    if (status == DORA_OK) {
        key_field.name = "key";
        key_field.flags |= DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
        status = dora_vector_image_share(pool, keys, &key_field, &key_image, nullptr);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_extension(key_image, nullptr);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(key_image, &declared_keys);
        if (status == DORA_OK) {
            key_image = nullptr;
        }
    }
    struct dora_vector entries = {};
    if (status == DORA_OK) {
        const struct dora_vector children[2] = {declared_keys, *values};
        const char* const names[2] = {"key", "value"};
        const struct dora_schema_field entries_field = {
            .name = "entries",
            .type.id = DORA_TYPE_STRUCT,
            .layout = DORA_LAYOUT_STRUCT,
            .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
            .child_count = 2
        };
        struct dora_allocation buffer = {};
        const struct vector_input input = {
            .field = &entries_field,
            .row_count = key_count,
            .allocations = &buffer,
            .buffer_count = 1,
            .children = children,
            .child_names = names,
            .child_count = 2
        };
        status = dora_vector_init_move_input(&entries, pool, &input, &buffer, nullptr);
        dora_allocation_release(&buffer);
    }
    if (status == DORA_OK) {
        const struct dora_schema_field field = {
            .type.id = DORA_TYPE_MAP,
            .layout = layout,
            .child_count = 1,
            .flags = are_keys_sorted ? DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED : 0
        };
        status = list_construct(vector, pool, &field, &entries, nullptr, entry_counts, validity, row_count);
    }
    dora_vector_release(&entries);
    dora_vector_release(&declared_keys);
    dora_vector_discard(key_image);
    return status;
}

//--------------------------------------
// Unions
//--------------------------------------

static int union_construct(struct dora_vector* vector,
                           struct dora_pool pool,
                           const struct dora_schema_field* field,
                           const struct dora_vector* children,
                           const char* const* names,
                           const int8_t* type_ids,
                           const uint32_t* member_rows,
                           uint64_t row_count) {
    if (!dora_vector_is_empty(vector) || row_count > INT64_MAX ||
        (row_count != 0 &&
         (type_ids == nullptr || (field->layout == DORA_LAYOUT_DENSE_UNION && member_rows == nullptr)))) {
        return EINVAL;
    }
    struct dora_allocation buffers[2] = {};
    int status = allocate_values(pool, row_count, 1, &buffers[0]);
    const bool dense = field->layout == DORA_LAYOUT_DENSE_UNION;
    if (status == DORA_OK && dense) {
        status = allocate_values(pool, row_count, sizeof(int32_t), &buffers[1]);
    }
    if (status == DORA_OK && row_count != 0) {
        memcpy(buffers[0].address, type_ids, (size_t)row_count);
        if (dense) {
            for (uint64_t row = 0; row < row_count; ++row) {
                if (member_rows[row] > INT32_MAX) {
                    status = EINVAL;
                    break;
                }
                ((int32_t*)buffers[1].address)[row] = (int32_t)member_rows[row];
            }
        }
    }
    const struct vector_input input = {
        .field = field,
        .row_count = row_count,
        .allocations = buffers,
        .buffer_count = dense ? 2 : 1,
        .children = children,
        .child_names = names,
        .child_count = field->child_count
    };
    if (status == DORA_OK) {
        return publish_constructed(vector, pool, &input, buffers);
    }
    dora_allocation_release(&buffers[0]);
    dora_allocation_release(&buffers[1]);
    return status;
}

int dora_vector_init_retain_dense_union(struct dora_vector* vector,
                                        const int8_t* type_codes,
                                        const struct dora_vector* children,
                                        const char* const* names,
                                        uint32_t child_count,
                                        const int8_t* type_ids,
                                        const uint32_t* member_rows,
                                        uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool_span(children, child_count, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct dora_schema_field field = {
        .type = {.id = DORA_TYPE_UNION, .parameters.type_codes = type_codes},
        .layout = DORA_LAYOUT_DENSE_UNION,
        .child_count = child_count
    };
    return union_construct(vector, pool, &field, children, names, type_ids, member_rows, row_count);
}

int dora_vector_init_retain_sparse_union(struct dora_vector* vector,
                                         const int8_t* type_codes,
                                         const struct dora_vector* children,
                                         const char* const* names,
                                         uint32_t child_count,
                                         const int8_t* type_ids,
                                         uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool_span(children, child_count, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    const struct dora_schema_field field = {
        .type = {.id = DORA_TYPE_UNION, .parameters.type_codes = type_codes},
        .layout = DORA_LAYOUT_SPARSE_UNION,
        .child_count = child_count
    };
    return union_construct(vector, pool, &field, children, names, type_ids, nullptr, row_count);
}

//--------------------------------------
// Extensions
//--------------------------------------

int dora_vector_init_retain_extension_error(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            struct dora_bytes name,
                                            struct dora_bytes metadata,
                                            const struct dora_vector* storage,
                                            const struct dora_schema* admitted_schema,
                                            struct dora_error* out_error) {
    struct dora_schema_field field = {};
    int status = extension_field(vector, name, metadata, storage, &field);
    if (status != DORA_OK) {
        return status;
    }
    struct vector_image* image = nullptr;
    status = vector_share(pool, storage, &field, admitted_schema, nullptr, nullptr, &image, out_error);
    if (status == DORA_OK) {
        status = dora_vector_bind_extension(image, out_error);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    return status;
}

int dora_vector_init_retain_extension(struct dora_vector* vector,
                                      struct dora_bytes name,
                                      struct dora_bytes metadata,
                                      const struct dora_vector* storage) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(storage, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_init_retain_extension_error(vector, pool, name, metadata, storage, nullptr, nullptr);
}

//--------------------------------------
// Encodings
//--------------------------------------

static int bound_index(const struct vector_bound_input* input, uint64_t row, uint64_t* value, bool* is_null) {
    uint32_t node = 0;
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        if (row >= view->length) {
            return EINVAL;
        }
        if (dora_vector_view_is_null(view, row)) {
            *value = 0;
            *is_null = true;
            return DORA_OK;
        }
        if (view->layout == DORA_LAYOUT_FIXED) {
            *is_null = false;
            return dora_vector_view_read_index(view, row, value);
        }
        if (view->layout == DORA_LAYOUT_DICTIONARY) {
            const int status = dora_vector_view_read_index(view, row, &row);
            if (status != DORA_OK) {
                return status;
            }
            ++node;
        } else if (view->layout == DORA_LAYOUT_RUN_END) {
            row = dora_vector_view_find_run(view, row);
            node = dora_vector_input_child(node, input, 1);
        } else {
            return EINVAL;
        }
    }
}

static int normalize_null(struct dora_pool pool,
                          const struct dora_vector_view* view,
                          uint64_t row,
                          struct dora_allocation* validity,
                          int64_t* null_count) {
    if (validity->address == nullptr) {
        const int status = allocate_values(pool, bitmap_words_for_bits(view->length), sizeof(uint64_t), validity);
        if (status != DORA_OK) {
            return status;
        }
        bitmap_fill_bytes(validity->address, validity->capacity_bytes, true);
    }
    bitmap_clear(validity->address, row);
    ++*null_count;
    return DORA_OK;
}

// Index carriers: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_NORMALIZE_FIXED_INDICES(S, T, P, NEGATIVE)                                                              \
    static int normalize_fixed_indices_##S(struct dora_pool pool,                                                      \
                                           const struct dora_vector_view* view,                                        \
                                           struct dora_allocation buffers[static 2],                                   \
                                           int64_t* null_count) {                                                      \
        uint64_t* const output = buffers[1].address;                                                                   \
        for (uint64_t row = 0; row < view->length; ++row) {                                                            \
            const uint64_t physical = dora_vector_view_get_physical_row(view, row);                                    \
            if (view->validity.data != nullptr &&                                                                      \
                !dora_bitmap_test(view->validity.data, view->validity_bit_offset + physical - view->offset)) {         \
                const int status = normalize_null(pool, view, row, &buffers[0], null_count);                           \
                if (status != DORA_OK) {                                                                               \
                    return status;                                                                                     \
                }                                                                                                      \
                output[row] = 0;                                                                                       \
                continue;                                                                                              \
            }                                                                                                          \
            const T value = type_load_##S(view->values.data, physical);                                                \
            if (NEGATIVE(value)) {                                                                                     \
                return EINVAL;                                                                                         \
            }                                                                                                          \
            output[row] = (uint64_t)value;                                                                             \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }

SIGNED_INDEX_CARRIERS(DEFINE_NORMALIZE_FIXED_INDICES, NEGATIVE_SIGNED)
UNSIGNED_INDEX_CARRIERS(DEFINE_NORMALIZE_FIXED_INDICES, NEGATIVE_UNSIGNED)

#undef DEFINE_NORMALIZE_FIXED_INDICES

int dora_vector_validity_shareable(const struct dora_vector* source, bool* out_shareable) {
    if (out_shareable == nullptr) {
        return EINVAL;
    }
    struct dora_vector_view view = {};
    struct vector_image* image = nullptr;
    int status = dora_vector_get_view(source, &view);
    if (status == DORA_OK) {
        status = dora_vector_resolve(source, &image);
    }
    if (status == DORA_OK) {
        *out_shareable = view.validity.data != nullptr && view.selection.indices == nullptr &&
                         view.layout != DORA_LAYOUT_DICTIONARY && view.layout != DORA_LAYOUT_RUN_END &&
                         dora_vector_buffer_has_owner(image, view.validity);
    }
    return status;
}

int dora_vector_init_share_validity(struct dora_vector* vector,
                                    const struct vector_input* input,
                                    const struct dora_vector* source,
                                    uint32_t shared_index,
                                    struct dora_allocation* in_out_buffers) {
    if (!dora_vector_is_empty(vector) || input == nullptr || input->field == nullptr || input->buffer_count < 2 ||
        shared_index > 1 || in_out_buffers == nullptr || source == nullptr || input->child_count != 0 ||
        (shared_index == 1 && (input->field->layout != DORA_LAYOUT_BITS || input->field->type.id != DORA_TYPE_BOOLEAN ||
                               input->null_count != 0))) {
        return EINVAL;
    }
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(source, &view);
    if (status != DORA_OK) {
        return status;
    }
    if (view.validity.data == nullptr || view.selection.indices != nullptr || view.layout == DORA_LAYOUT_DICTIONARY ||
        view.layout == DORA_LAYOUT_RUN_END || input->row_count != view.length) {
        return EINVAL;
    }
    struct dora_pool pool = {};
    status = dora_vector_get_pool(source, &pool);
    struct scratch_scope scope;
    if (status == DORA_OK) {
        status = dora_scratch_scope_begin(pool, &scope);
    }
    if (status != DORA_OK) {
        return status;
    }
    struct dora_bytes local[3] = {};
    struct dora_bytes* buffers = local;
    if (input->buffer_count > 3) {
        size_t bytes = 0;
        status = ckd_mul(&bytes, input->buffer_count, sizeof(*buffers))
                     ? EOVERFLOW
                     : dora_scratch_allocate(&scope, bytes, (void**)&buffers);
    }
    for (uint32_t index = 0; status == DORA_OK && index < input->buffer_count; ++index) {
        const struct dora_allocation* const own = &in_out_buffers[index];
        if (index == shared_index) {
            if (own->address != nullptr || own->capacity_bytes != 0 || own->pool.index != 0 ||
                own->pool.generation != 0 || own->is_raw) {
                status = EINVAL;
            }
            continue;
        }
        const struct dora_bytes bytes =
            input->buffers == nullptr ? (struct dora_bytes){own->address, own->capacity_bytes} : input->buffers[index];
        if ((own->address == nullptr && (own->capacity_bytes != 0 || bytes.byte_count != 0 || bytes.data != nullptr ||
                                         own->pool.index != 0 || own->pool.generation != 0 || own->is_raw)) ||
            (own->address != nullptr && (!pool_is_same(own->pool, pool) || bytes.data != own->address ||
                                         own->capacity_bytes == 0 || bytes.byte_count > own->capacity_bytes ||
                                         own->capacity_bytes % pool_row_of(pool)->budget.alignment_bytes != 0))) {
            status = EINVAL;
            break;
        }
        for (uint32_t other = 0; own->address != nullptr && other < index; ++other) {
            if (own->address == in_out_buffers[other].address) {
                status = EINVAL;
            }
        }
        buffers[index] = bytes;
    }
    struct dora_allocation copied = {};
    struct dora_handle retained = {};
    bool shareable = false;
    const bool needs_mask = shared_index == 1 || input->null_count != 0;
    if (status == DORA_OK && needs_mask) {
        status = dora_vector_validity_shareable(source, &shareable);
        if (status == DORA_OK && shareable) {
            status = dora_identity_claim_acquire(pool, &source->handle, IDENTITY_KIND_VECTOR, &retained);
        } else if (status == DORA_OK) {
            size_t bytes = 0;
            status = ckd_mul(&bytes, bitmap_words_for_bits(input->row_count), sizeof(uint64_t))
                         ? EOVERFLOW
                         : dora_pool_allocate(pool, bytes, &copied);
            if (status == DORA_OK && bytes != 0) {
                dora_bitmap_load_words(view.validity.data, view.validity_bit_offset, input->row_count, copied.address);
            }
        }
    }
    struct vector_input composition = *input;
    composition.buffers = buffers;
    composition.allocations = nullptr;
    composition.offset = shared_index == 1 && shareable ? view.validity_bit_offset : 0;
    composition.validity_bit_offset = shared_index == 0 && shareable ? view.validity_bit_offset : 0;
    struct dora_schema schema = {};
    if (status == DORA_OK && composition.admitted_schema == nullptr) {
        status = dora_schema_node_claim(pool, input->field, nullptr, &schema);
        composition.admitted_schema = &schema;
    }
    if (status == DORA_OK) {
        buffers[shared_index] = !needs_mask ? (struct dora_bytes){}
                                : shareable ? (struct dora_bytes){view.validity.data, view.validity.byte_count}
                                            : (struct dora_bytes){copied.address, copied.capacity_bytes};
    }
    struct vector_image* image = nullptr;
    struct dora_vector published = {};
    if (status == DORA_OK) {
        status = dora_vector_prepare(pool, &composition, &image, nullptr);
    }
    if (status == DORA_OK) {
        struct vector_buffer_row* const rows = dora_vector_image_buffers(image->buffer_image);
        if (shareable) {
            rows[shared_index].owner_kind = VECTOR_BUFFER_OWNER_RETAINED;
            rows[shared_index].owner.retained = retained;
            retained = (struct dora_handle){};
        } else {
            status = dora_allocation_adopt(&copied, &rows[shared_index].owner.allocation);
        }
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, &published);
    }
    if (status == DORA_OK && !dora_vector_is_empty(vector)) {
        status = EINVAL;
    }
    struct vector_buffer_row* const rows = image == nullptr ? nullptr : dora_vector_image_buffers(image->buffer_image);
    uint32_t adopted = 0;
    for (; status == DORA_OK && adopted < input->buffer_count; ++adopted) {
        if (adopted == shared_index) {
            continue;
        }
        const struct dora_allocation* const own = &in_out_buffers[adopted];
        if (own->address != buffers[adopted].data || own->capacity_bytes < buffers[adopted].byte_count ||
            (own->address != nullptr && !pool_is_same(own->pool, pool))) {
            status = EINVAL;
            break;
        }
        status = dora_allocation_move(&in_out_buffers[adopted], &rows[adopted].owner.allocation);
    }
    if (status != DORA_OK) {
        for (uint32_t index = 0; index < adopted; ++index) {
            if (index != shared_index && rows[index].owner.allocation.address != nullptr) {
                const int restored = dora_allocation_move(&rows[index].owner.allocation, &in_out_buffers[index]);
                if (restored != DORA_OK) {
                    status = restored;
                }
            }
        }
    } else {
        for (uint32_t index = 0; index < input->buffer_count; ++index) {
            if (index != shared_index) {
                dora_allocation_internalize(&rows[index].owner.allocation);
            }
        }
        *vector = published;
        published = (struct dora_vector){};
    }
    if (published.handle.slot_index != 0) {
        dora_vector_release(&published);
    } else if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    void* retired = nullptr;
    dora_identity_claim_release(&retained, IDENTITY_KIND_VECTOR, &retired);
    dora_vector_discard(retired);
    dora_allocation_release(&copied);
    dora_schema_release(&schema);
    dora_scratch_scope_end(&scope);
    return status;
}

int dora_vector_init_share_indices(struct dora_vector* vector,
                                   const struct dora_schema_field* field,
                                   const struct dora_vector* source,
                                   const struct dora_vector* values) {
    if (!dora_vector_is_empty(vector) || field == nullptr || field->layout != DORA_LAYOUT_DICTIONARY ||
        source == nullptr || values == nullptr) {
        return EINVAL;
    }
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(source, &view);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_pool pool = {
        .index = source->handle.pool_index,
        .generation = dora_pool_rows[source->handle.pool_index].generation
    };
    if (view.layout != DORA_LAYOUT_DICTIONARY || view.selection.indices != nullptr) {
        return EINVAL;
    }
    const struct dora_bytes buffers[2] = {
        {.data = view.validity.data, .byte_count = view.validity.byte_count},
        {.data = view.values.data, .byte_count = view.values.byte_count},
    };
    const struct vector_input input = {
        .field = field,
        .row_count = view.length,
        .null_count = view.null_count,
        .offset = view.offset,
        .validity_bit_offset = view.validity_bit_offset,
        .buffers = buffers,
        .buffer_count = 2,
        .buffer_source = source,
        .children = values,
        .child_count = 1,
    };
    struct dora_allocation none[2] = {};
    return dora_vector_init_move_input(vector, pool, &input, none, nullptr);
}

static int selection_rows_publish(struct dora_pool pool,
                                  struct dora_allocation* in_out_rows,
                                  uint64_t length,
                                  struct dora_vector* out_rows) {
    const struct dora_schema_field field =
        {.type.id = DORA_TYPE_UINT64, .layout = DORA_LAYOUT_FIXED, .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE};
    struct dora_allocation buffers[2] = {{}, *in_out_rows};
    *in_out_rows = (struct dora_allocation){};
    const struct vector_input composition =
        {.field = &field, .row_count = length, .allocations = buffers, .buffer_count = 2};
    const int status = dora_vector_init_move_input(out_rows, pool, &composition, buffers, nullptr);
    dora_allocation_release(&buffers[1]);
    return status;
}

static int selection_adopt_rows(struct vector_image* image, const struct dora_vector* rows) {
    struct vector_node_ownership* const owned = &image->ownership;
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(rows, &view);
    if (status == DORA_OK) {
        status = dora_identity_claim_acquire(image->allocation.pool,
                                             &rows->handle,
                                             IDENTITY_KIND_VECTOR,
                                             &owned->selection_source.handle);
    }
    if (status == DORA_OK) {
        owned->selection_source.rows = rows->rows;
        owned->view.selection = (struct dora_vector_selection){
            .indices = dora_vector_view_get_values_u64(&view),
            .physical_type = DORA_PHYSICAL_TYPE_U64
        };
    }
    return status;
}

// Selection carriers: UInt32, UInt64
#define DEFINE_SELECTION_FIXED(S, T)                                                                                   \
    static int selection_fixed_##S(const struct dora_vector_view* indices,                                             \
                                   uint64_t* physical,                                                                 \
                                   const struct dora_vector_view* source,                                              \
                                   uint64_t* logical,                                                                  \
                                   uint64_t source_length) {                                                           \
        for (uint64_t row = 0; row < indices->length; ++row) {                                                         \
            const uint64_t physical_row = dora_vector_view_get_physical_row(indices, row);                             \
            if (indices->validity.data != nullptr &&                                                                   \
                !dora_bitmap_test(indices->validity.data,                                                              \
                                  indices->validity_bit_offset + physical_row - indices->offset)) {                    \
                return EINVAL;                                                                                         \
            }                                                                                                          \
            const T index = type_load_##S(indices->values.data, physical_row);                                         \
            if (index >= source_length) {                                                                              \
                return EINVAL;                                                                                         \
            }                                                                                                          \
            physical[row] = dora_vector_view_get_physical_row(source, index) - source->offset;                         \
            if (logical != nullptr) {                                                                                  \
                logical[row] = index;                                                                                  \
            }                                                                                                          \
        }                                                                                                              \
        return DORA_OK;                                                                                                \
    }
DEFINE_SELECTION_FIXED(u32, uint32_t)
DEFINE_SELECTION_FIXED(u64, uint64_t)

#undef DEFINE_SELECTION_FIXED

static int run_end_construct(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_vector* values,
                             const uint64_t* run_lengths,
                             uint64_t constant_length,
                             bool constant) {
    uint64_t count = 0;
    if (!dora_vector_is_empty(vector) || dora_vector_get_length(values, &count) != DORA_OK ||
        (constant && (count != 1 || constant_length > INT64_MAX)) ||
        (!constant && count != 0 && run_lengths == nullptr)) {
        return EINVAL;
    }
    struct dora_logical_type value_type = {};
    int status = dora_vector_get_type(values, &value_type);
    if (status != DORA_OK) {
        return status;
    }
    if (value_type.extension_name.byte_count != 0) {
        struct dora_vector storage = {};
        struct dora_vector encoded = {};
        status = dora_vector_get_storage(values, &storage);
        if (status == DORA_OK) {
            status = run_end_construct(&encoded, pool, &storage, run_lengths, constant_length, constant);
        }
        if (status == DORA_OK) {
            status = dora_vector_init_retain_extension_error(vector,
                                                             pool,
                                                             value_type.extension_name,
                                                             value_type.extension_metadata,
                                                             &encoded,
                                                             nullptr,
                                                             nullptr);
        }
        dora_vector_release(&encoded);
        return status;
    }
    const uint64_t runs = constant && constant_length == 0 ? 0 : count;
    struct dora_allocation buffers[2] = {};
    status = allocate_values(pool, runs, sizeof(int64_t), &buffers[1]);
    uint64_t total = 0;
    for (uint64_t run = 0; status == DORA_OK && run < runs; ++run) {
        const uint64_t length = constant ? constant_length : run_lengths[run];
        if (length == 0 || ckd_add(&total, total, length) || total > INT64_MAX) {
            status = EINVAL;
            break;
        }
        ((int64_t*)buffers[1].address)[run] = total;
    }
    const struct dora_schema_field run_field =
        {.type.id = DORA_TYPE_INT64, .layout = DORA_LAYOUT_FIXED, .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE};
    struct dora_vector children[2] = {{}, dora_vector_borrow(values)};
    if (constant && constant_length == 0) {
        children[1].rows.length = 0;
    }
    if (status == DORA_OK) {
        const struct vector_input composition =
            {.field = &run_field, .row_count = runs, .allocations = buffers, .buffer_count = 2};
        status = dora_vector_init_move_input(&children[0], pool, &composition, buffers, nullptr);
    }
    dora_allocation_release(&buffers[1]);
    const struct dora_schema_field field =
        {.type = value_type, .layout = DORA_LAYOUT_RUN_END, .index_type = DORA_TYPE_INT64, .child_count = 2};
    if (status == DORA_OK) {
        const struct vector_input input = {.field = &field, .row_count = total, .children = children, .child_count = 2};
        status = publish_constructed(vector, pool, &input, nullptr);
    }
    dora_vector_release(&children[0]);
    return status;
}

int dora_vector_init_retain_dictionary_in_pool(struct dora_vector* vector,
                                               struct dora_pool pool,
                                               const struct dora_vector* indices,
                                               const struct dora_vector* values) {
    if (!dora_vector_is_empty(vector)) {
        return EINVAL;
    }
    struct dora_vector_view index_view = {};
    struct dora_logical_type type = {};
    struct dora_logical_type index_type = {};
    int status = dora_vector_get_view(indices, &index_view);
    if (status == DORA_OK) {
        status = dora_vector_get_type(indices, &index_type);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_type(values, &type);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (type.extension_name.byte_count != 0) {
        struct dora_vector storage = {};
        struct dora_vector encoded = {};
        status = dora_vector_get_storage(values, &storage);
        if (status == DORA_OK) {
            status = dora_vector_init_retain_dictionary_in_pool(&encoded, pool, indices, &storage);
        }
        if (status == DORA_OK) {
            status = dora_vector_init_retain_extension_error(vector,
                                                             pool,
                                                             type.extension_name,
                                                             type.extension_metadata,
                                                             &encoded,
                                                             nullptr,
                                                             nullptr);
        }
        dora_vector_release(&encoded);
        return status;
    }
    enum dora_layout admitted_layout = DORA_LAYOUT_DEFAULT;
    status = dora_type_resolve_layout(&type, DORA_LAYOUT_DICTIONARY, index_type.id, &admitted_layout);
    if (status != DORA_OK) {
        return status;
    }
    if (index_view.layout != DORA_LAYOUT_FIXED || index_view.selection.indices != nullptr) {
        struct dora_allocation normalized[2] = {};
        status = allocate_values(pool, indices->rows.length, sizeof(uint64_t), &normalized[1]);
        struct vector_bound_input bound = {};
        struct scratch_scope scope = {};
        if (status == DORA_OK && index_view.layout != DORA_LAYOUT_FIXED) {
            status = dora_scratch_scope_begin(pool, &scope);
            if (status == DORA_OK) {
                status = dora_vector_bind_input(&scope, indices, nullptr, &bound);
            }
        }
        int64_t null_count = 0;
        if (status == DORA_OK && index_view.layout == DORA_LAYOUT_FIXED) {
            switch (index_view.physical_type) {
#define NORMALIZE_FIXED_INDICES_CASE(S, T, P, ...)                                                                     \
    case P:                                                                                                            \
        status = normalize_fixed_indices_##S(__VA_ARGS__);                                                             \
        break;
                SIGNED_INDEX_CARRIERS(NORMALIZE_FIXED_INDICES_CASE, pool, &index_view, normalized, &null_count)
                UNSIGNED_INDEX_CARRIERS(NORMALIZE_FIXED_INDICES_CASE, pool, &index_view, normalized, &null_count)
#undef NORMALIZE_FIXED_INDICES_CASE
            default:
                status = EINVAL;
                break;
            }
        }
        for (uint64_t row = 0; status == DORA_OK && bound.nodes != nullptr && row < indices->rows.length; ++row) {
            uint64_t index = 0;
            bool is_null = false;
            status = bound_index(&bound, row, &index, &is_null);
            if (status != DORA_OK) {
                break;
            }
            if (is_null) {
                status = normalize_null(pool, &index_view, row, &normalized[0], &null_count);
            }
            if (status == DORA_OK) {
                ((uint64_t*)normalized[1].address)[row] = index;
            }
        }
        dora_scratch_scope_end(&scope);
        const struct dora_schema_field normalized_field =
            {.type = type, .layout = DORA_LAYOUT_DICTIONARY, .index_type = DORA_TYPE_UINT64, .child_count = 1};
        if (status == DORA_OK) {
            const struct vector_input input = {
                .field = &normalized_field,
                .row_count = indices->rows.length,
                .null_count = null_count,
                .allocations = normalized,
                .buffer_count = 2,
                .children = values,
                .child_count = 1
            };
            status = publish_constructed(vector, pool, &input, normalized);
        }
        dora_allocation_release(&normalized[0]);
        dora_allocation_release(&normalized[1]);
        return status;
    }
    const struct dora_schema_field field =
        {.type = type, .layout = DORA_LAYOUT_DICTIONARY, .index_type = index_type.id, .child_count = 1};
    const struct dora_bytes buffers[2] = {
        {.data = index_view.validity.data, .byte_count = index_view.validity.byte_count},
        {.data = index_view.values.data, .byte_count = index_view.values.byte_count}
    };
    const struct vector_input input = {
        .field = &field,
        .row_count = indices->rows.length,
        .null_count = index_view.null_count,
        .offset = index_view.offset,
        .validity_bit_offset = index_view.validity_bit_offset,
        .buffers = buffers,
        .buffer_count = 2,
        .children = values,
        .child_count = 1,
        .should_check_child_activity = true
    };
    struct vector_image* image = nullptr;
    status = dora_vector_prepare(pool, &input, &image, nullptr);
    if (status == DORA_OK) {
        struct vector_buffer_row* const buffer = &dora_vector_image_buffers(image->buffer_image)[1];
        buffer->owner_kind = VECTOR_BUFFER_OWNER_RETAINED;
        status = dora_identity_claim_acquire(pool, &indices->handle, IDENTITY_KIND_VECTOR, &buffer->owner.retained);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    return status;
}

int dora_vector_init_retain_dictionary(struct dora_vector* vector,
                                       const struct dora_vector* indices,
                                       const struct dora_vector* values) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_shared_pool((const struct dora_vector* const[]){indices, values}, 2, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_init_retain_dictionary_in_pool(vector, pool, indices, values);
}

int dora_vector_init_retain_run_end(struct dora_vector* vector,
                                    const struct dora_vector* values,
                                    const uint64_t* run_lengths) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(values, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return run_end_construct(vector, pool, values, run_lengths, 0, false);
}

int dora_vector_init_retain_constant_in_pool(struct dora_vector* vector,
                                             struct dora_pool pool,
                                             const struct dora_vector* value,
                                             uint64_t row_count) {
    return run_end_construct(vector, pool, value, nullptr, row_count, true);
}

int dora_vector_init_retain_constant(struct dora_vector* vector, const struct dora_vector* value, uint64_t row_count) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(value, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_init_retain_constant_in_pool(vector, pool, value, row_count);
}

static int selection_construct(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_vector* source,
                               const struct dora_vector* indices,
                               bool check_activity,
                               const struct dora_vector* shared_rows,
                               const void* parent_selection) {
    if (!dora_vector_is_empty(vector) || !dora_vector_is_valid(source) || !dora_vector_is_valid(indices)) {
        return EINVAL;
    }
    struct vector_image* source_image = nullptr;
    int status = dora_vector_resolve(source, &source_image);
    struct dora_vector_view index_view = {};
    struct dora_logical_type index_type = {};
    if (status == DORA_OK) {
        status = dora_vector_get_view(indices, &index_view);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_type(indices, &index_type);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (index_type.id != DORA_TYPE_UINT32 && index_type.id != DORA_TYPE_UINT64) {
        return EINVAL;
    }
    struct dora_schema_field field = {};
    status = dora_schema_get_field(&source_image->schema, &field);
    if (status == DORA_OK && field.type.extension_name.byte_count != 0) {
        struct dora_vector storage = {};
        struct dora_vector selected = {};
        status = dora_vector_get_storage(source, &storage);
        if (status == DORA_OK) {
            status = selection_construct(&selected, pool, &storage, indices, check_activity, nullptr, nullptr);
        }
        if (status == DORA_OK) {
            status = dora_vector_init_retain_extension_error(vector,
                                                             pool,
                                                             field.type.extension_name,
                                                             field.type.extension_metadata,
                                                             &selected,
                                                             nullptr,
                                                             nullptr);
        }
        dora_vector_release(&selected);
        return status;
    }
    struct vector_image* image = nullptr;
    if (status == DORA_OK) {
        status = dora_vector_image_share(pool, source, &field, &image, nullptr);
    }
    if (status != DORA_OK) {
        return status;
    }
    struct vector_node_ownership* const owned = &image->ownership;
    const void* const own_selection = owned->view.selection.indices;
    const bool can_share =
        shared_rows != nullptr && own_selection == parent_selection && !owned->requires_effective_validity;
    struct dora_vector rows = {};
    if (can_share) {
        status = selection_adopt_rows(image, shared_rows);
    } else {
        struct dora_allocation physical = {};
        size_t bytes = 0;
        if (ckd_mul(&bytes, index_view.length, sizeof(uint64_t))) {
            status = EOVERFLOW;
        } else {
            status = dora_pool_allocate(pool, bytes, &physical);
        }
        uint64_t* logical_selection = nullptr;
        if (status == DORA_OK && owned->requires_effective_validity) {
            status = dora_vector_defer_validity(image, source, nullptr, index_view.length, &logical_selection);
        }
        struct vector_bound_input bound = {};
        struct scratch_scope scope = {};
        if (status == DORA_OK && index_view.layout != DORA_LAYOUT_FIXED) {
            status = dora_scratch_scope_begin(pool, &scope);
            if (status == DORA_OK) {
                status = dora_vector_bind_input(&scope, indices, nullptr, &bound);
            }
        }
        if (status == DORA_OK && index_view.layout == DORA_LAYOUT_FIXED) {
            switch (index_type.id) {
            case DORA_TYPE_UINT32:
                status = selection_fixed_u32(&index_view,
                                             physical.address,
                                             &owned->view,
                                             logical_selection,
                                             source->rows.length);
                break;
            case DORA_TYPE_UINT64:
                status = selection_fixed_u64(&index_view,
                                             physical.address,
                                             &owned->view,
                                             logical_selection,
                                             source->rows.length);
                break;
            default:
                status = EINVAL;
                break;
            }
        } else {
            for (uint64_t row = 0; status == DORA_OK && row < index_view.length; ++row) {
                uint64_t index = 0;
                bool is_null = false;
                status = bound_index(&bound, row, &index, &is_null);
                if (status != DORA_OK || is_null || index >= source->rows.length) {
                    status = EINVAL;
                    break;
                }
                ((uint64_t*)physical.address)[row] =
                    dora_vector_view_get_physical_row(&owned->view, index) - owned->view.offset;
                if (logical_selection != nullptr) {
                    logical_selection[row] = index;
                }
            }
        }
        dora_scratch_scope_end(&scope);
        if (status == DORA_OK && owned->view.layout == DORA_LAYOUT_STRUCT) {
            status = selection_rows_publish(pool, &physical, index_view.length, &rows);
            if (status == DORA_OK) {
                status = selection_adopt_rows(image, &rows);
            }
        } else if (status == DORA_OK) {
            status = dora_allocation_adopt(&physical, &owned->physical_rows);
            if (status == DORA_OK) {
                owned->owns_physical_rows = true;
                owned->view.selection = (struct dora_vector_selection){
                    .indices = owned->physical_rows.address,
                    .physical_type = DORA_PHYSICAL_TYPE_U64
                };
            }
        }
        dora_allocation_release(&physical);
    }
    const struct dora_vector* const child_rows = can_share ? shared_rows : &rows;
    if (status == DORA_OK && owned->view.layout == DORA_LAYOUT_STRUCT) {
        struct dora_vector* const children = dora_vector_image_children(image);
        for (uint32_t index = 0; status == DORA_OK && index < image->child_vectors.entry_count; ++index) {
            struct dora_vector selected = {};
            status = selection_construct(&selected, pool, &children[index], indices, false, child_rows, own_selection);
            struct dora_handle claim = {};
            if (status == DORA_OK) {
                status = dora_identity_claim_acquire(pool, &selected.handle, IDENTITY_KIND_VECTOR, &claim);
            }
            if (status == DORA_OK) {
                dora_vector_release(&children[index]);
                children[index] = selected;
                children[index].handle = claim;
            }
            dora_vector_release(&selected);
        }
    }
    if (status == DORA_OK) {
        const bool all_null = owned->view.null_count == (int64_t)owned->view.length;
        owned->view.length = index_view.length;
        owned->view.null_count = all_null ? (int64_t)index_view.length : owned->view.null_count == 0 ? 0 : -1;
        status = dora_vector_bind_extension(image, nullptr);
    }
    if (status == DORA_OK && check_activity) {
        const struct vector_input input = {
            .row_count = image->ownership.view.length,
            .should_check_child_activity = true
        };
        dora_pool_callback_enter(pool);
        status = dora_vector_check_activity(image, &input);
        dora_pool_callback_leave(pool);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    dora_vector_release(&rows);
    return status;
}

int dora_vector_init_retain_selection_in_pool(struct dora_vector* vector,
                                              struct dora_pool pool,
                                              const struct dora_vector* source,
                                              const struct dora_vector* indices) {
    return selection_construct(vector, pool, source, indices, true, nullptr, nullptr);
}

int dora_vector_init_retain_selection(struct dora_vector* vector,
                                      const struct dora_vector* source,
                                      const struct dora_vector* indices) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_init_retain_selection_in_pool(vector, pool, source, indices);
}

//--------------------------------------
// Validation and Null
//--------------------------------------

int dora_vector_init_null(struct dora_vector* vector, struct dora_pool pool, uint64_t row_count) {
    const struct dora_schema_field field = {.type.id = DORA_TYPE_NULL, .layout = DORA_LAYOUT_NULL};
    if (row_count > INT64_MAX) {
        return EINVAL;
    }
    const struct vector_input composition = {.field = &field, .row_count = row_count, .null_count = (int64_t)row_count};
    return dora_vector_init_move_input(vector, pool, &composition, nullptr, nullptr);
}

//------------------------------------------------------------------------------
//
// Copies
//
//------------------------------------------------------------------------------

int dora_vector_share(const struct dora_vector* vector, struct dora_pool pool, struct dora_vector* out_shared) {
    struct dora_pool owner = {};
    int status = dora_vector_is_empty(out_shared) ? dora_vector_get_pool(vector, &owner) : EINVAL;
    if (status != DORA_OK) {
        return status;
    }
    if (pool_row_of(owner)->is_draining) {
        return EBUSY;
    }
    dora_pool_callback_enter(owner);
    if (pool_is_same(owner, pool)) {
        status = dora_vector_retain(vector, out_shared);
        dora_pool_callback_leave(owner);
        return status;
    }
    struct vector_image* source_image = nullptr;
    struct dora_schema_field field = {};
    status = dora_vector_resolve(vector, &source_image);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&source_image->schema, &field);
    }
    struct vector_image* image = nullptr;
    if (status == DORA_OK) {
        status = vector_share(pool, vector, &field, nullptr, nullptr, nullptr, &image, nullptr);
    }
    if (status == DORA_OK && field.type.extension_name.byte_count != 0) {
        status = dora_vector_bind_extension(image, nullptr);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, out_shared);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    dora_pool_callback_leave(owner);
    return status;
}
