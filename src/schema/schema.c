// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/schema.h"

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "schema/schema_internal.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t SCHEMA_FIELD_FLAG_MASK = (uint32_t)DORA_SCHEMA_FIELD_FLAG_NONNULLABLE |
                                                   (uint32_t)DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED |
                                                   (uint32_t)DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED;

struct schema_size {
    size_t metadata_count;
    size_t byte_count;
};

static int measure_bytes(struct schema_size* size, struct dora_bytes bytes) {
    if (bytes.data == nullptr && bytes.byte_count != 0) {
        return EINVAL;
    }
    return ckd_add(&size->byte_count, size->byte_count, bytes.byte_count) ? EOVERFLOW : DORA_OK;
}

static int measure_text(struct schema_size* size, const char* text) {
    size_t length = text != nullptr ? strlen(text) : 0;
    return ckd_add(&length, length, 1) || ckd_add(&size->byte_count, size->byte_count, length) ? EOVERFLOW : DORA_OK;
}

static int measure_metadata(struct schema_size* size, const struct dora_metadata_entry* metadata, uint32_t count) {
    if (metadata == nullptr && count != 0) {
        return EINVAL;
    }
    if (ckd_add(&size->metadata_count, size->metadata_count, count)) {
        return EOVERFLOW;
    }
    for (uint32_t index = 0; index < count; ++index) {
        int status = measure_bytes(size, metadata[index].key);
        if (status == DORA_OK) {
            status = measure_bytes(size, metadata[index].value);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int field_check(const struct dora_schema_field* field, enum dora_layout* out_layout) {
    if (field == nullptr || ((uint32_t)field->flags & ~SCHEMA_FIELD_FLAG_MASK) != 0 ||
        ((field->flags & DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED) != 0 && field->layout != DORA_LAYOUT_DICTIONARY) ||
        ((field->flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) != 0 && field->type.id != DORA_TYPE_MAP)) {
        return EINVAL;
    }
    int status = dora_type_check(&field->type);
    if (status == DORA_OK) {
        status = dora_type_resolve_layout(&field->type, field->layout, field->index_type, out_layout);
    }
    if (status != DORA_OK) {
        return status;
    }
    uint32_t expected = 0;
    if (*out_layout == DORA_LAYOUT_DICTIONARY) {
        expected = 1;
    } else if (*out_layout == DORA_LAYOUT_RUN_END) {
        expected = 2;
    } else {
        switch (field->type.id) {
        case DORA_TYPE_LIST:
        case DORA_TYPE_FIXED_LIST:
        case DORA_TYPE_MAP:
            expected = 1;
            break;
        case DORA_TYPE_STRUCT:
            return DORA_OK;
        case DORA_TYPE_UNION:
            return field->child_count <= UNION_MEMBER_CAPACITY ? DORA_OK : EINVAL;
        default:
            break;
        }
    }
    return field->child_count == expected ? DORA_OK : EINVAL;
}

static int measure_field(struct schema_size* size, const struct dora_schema_field* field) {
    enum dora_layout layout = DORA_LAYOUT_DEFAULT;
    int status = field_check(field, &layout);
    if (status == DORA_OK) {
        status = measure_text(size, field->name);
    }
    if (status == DORA_OK && field->type.id == DORA_TYPE_TIMESTAMP &&
        field->type.parameters.timestamp.timezone != nullptr) {
        status = measure_text(size, field->type.parameters.timestamp.timezone);
    }
    if (status == DORA_OK && field->type.id == DORA_TYPE_UNION &&
        ckd_add(&size->byte_count, size->byte_count, UNION_MEMBER_CAPACITY)) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = measure_bytes(size, field->type.extension_name);
    }
    if (status == DORA_OK) {
        status = measure_bytes(size, field->type.extension_metadata);
    }
    return status == DORA_OK ? measure_metadata(size, field->metadata, field->metadata_entry_count) : status;
}

static struct dora_bytes copy_bytes(unsigned char** cursor, struct dora_bytes source) {
    if (source.byte_count == 0) {
        return (struct dora_bytes){};
    }
    const struct dora_bytes result = {.data = *cursor, .byte_count = source.byte_count};
    memcpy(*cursor, source.data, source.byte_count);
    *cursor += source.byte_count;
    return result;
}

static const char* copy_text(unsigned char** cursor, const char* source) {
    const char* const result = (const char*)*cursor;
    const size_t length = source != nullptr ? strlen(source) : 0;
    if (length != 0) {
        memcpy(*cursor, source, length);
    }
    (*cursor)[length] = 0;
    *cursor += length + 1;
    return result;
}

static void copy_field(struct dora_schema_field* destination,
                       const struct dora_schema_field* source,
                       struct dora_metadata_entry** metadata,
                       unsigned char** bytes) {
    *destination = *source;
    destination->name = copy_text(bytes, source->name);
    destination->type.extension_name = copy_bytes(bytes, source->type.extension_name);
    destination->type.extension_metadata = copy_bytes(bytes, source->type.extension_metadata);
    if (source->type.id == DORA_TYPE_TIMESTAMP && source->type.parameters.timestamp.timezone != nullptr) {
        destination->type.parameters.timestamp.timezone = copy_text(bytes, source->type.parameters.timestamp.timezone);
    }
    if (source->type.id == DORA_TYPE_UNION) {
        destination->type.parameters.type_codes = (const int8_t*)*bytes;
        *bytes += UNION_MEMBER_CAPACITY;
    }
    destination->metadata = source->metadata_entry_count == 0 ? nullptr : *metadata;
    for (uint32_t index = 0; index < source->metadata_entry_count; ++index) {
        (*metadata)[index].key = copy_bytes(bytes, source->metadata[index].key);
        (*metadata)[index].value = copy_bytes(bytes, source->metadata[index].value);
    }
    *metadata += source->metadata_entry_count;
}

static int image_allocate(struct dora_pool pool,
                          uint32_t nodes,
                          uint32_t children,
                          const struct schema_size* size,
                          struct schema_image** out_image,
                          struct dora_metadata_entry** out_metadata,
                          unsigned char** out_bytes) {
    size_t node_bytes = 0;
    size_t metadata_bytes = 0;
    size_t child_bytes = 0;
    size_t total = sizeof(struct schema_image);
    if (ckd_mul(&node_bytes, nodes, sizeof(struct schema_node)) ||
        ckd_mul(&metadata_bytes, size->metadata_count, sizeof(struct dora_metadata_entry)) ||
        ckd_mul(&child_bytes, children, sizeof(struct dora_schema)) || ckd_add(&total, total, node_bytes) ||
        ckd_add(&total, total, metadata_bytes) || ckd_add(&total, total, child_bytes) ||
        ckd_add(&total, total, size->byte_count)) {
        return EOVERFLOW;
    }
    struct dora_allocation storage = {};
    const int status = dora_memory_payload_allocate(pool, total, &storage);
    if (status != DORA_OK) {
        return status;
    }
    struct schema_image* const image = storage.address;
    memset(image, 0, total);
    image->storage = storage;
    image->claims.payload = image;
    unsigned char* const metadata = (unsigned char*)(image->nodes + nodes);
    *out_metadata = (struct dora_metadata_entry*)metadata;
    image->children = (struct dora_schema*)(metadata + metadata_bytes);
    *out_bytes = metadata + metadata_bytes + child_bytes;
    image->node_count = nodes;
    image->child_count = children;
    *out_image = image;
    return DORA_OK;
}

static void image_retire(struct schema_image* pending) {
    while (pending != nullptr) {
        struct schema_image* const image = pending;
        pending = image->next_retired;
        for (uint32_t index = 0; index < image->child_count; ++index) {
            void* retired = nullptr;
            dora_identity_claim_release(&image->children[index].handle, IDENTITY_KIND_SCHEMA, &retired);
            if (retired != nullptr) {
                struct schema_image* const child = retired;
                child->next_retired = pending;
                pending = child;
            }
        }
        struct dora_allocation storage = image->storage;
        dora_allocation_release(&storage);
    }
}

static int image_publish(struct schema_image* image, struct dora_schema* schema) {
    struct dora_handle owner = {};
    const int status = dora_identity_slot_publish(image->storage.pool, IDENTITY_KIND_SCHEMA, &image->claims, &owner);
    if (status != DORA_OK) {
        image_retire(image);
        return status;
    }
    *schema = (struct dora_schema){.handle = owner};
    return DORA_OK;
}

static int image_child_field(const struct schema_image* image,
                             const struct dora_schema* child,
                             struct dora_schema_field* out_field) {
    if (child->handle.slot_index == 0) {
        *out_field = image->nodes[child->node_index].field;
        return DORA_OK;
    }
    return dora_schema_get_field(child, out_field);
}

static int union_codes_copy(struct dora_schema_field* field, const struct dora_schema_field* source, uint32_t count) {
    if (count > UNION_MEMBER_CAPACITY) {
        return EINVAL;
    }
    bool seen[UNION_MEMBER_CAPACITY] = {};
    int8_t* const codes = (int8_t*)field->type.parameters.type_codes;
    for (uint32_t index = 0; index < count; ++index) {
        const int8_t code =
            source->type.parameters.type_codes != nullptr ? source->type.parameters.type_codes[index] : (int8_t)index;
        if (code < 0 || seen[(uint8_t)code]) {
            return EINVAL;
        }
        seen[(uint8_t)code] = true;
        codes[index] = code;
    }
    return DORA_OK;
}

static int image_check_node(struct schema_image* image, uint32_t node, const struct dora_schema_field* source) {
    struct dora_schema_field* const field = &image->nodes[node].field;
    const bool dictionary = field->layout == DORA_LAYOUT_DICTIONARY;
    const bool run_end = field->layout == DORA_LAYOUT_RUN_END;
    struct dora_schema_field value = {};
    if (dictionary || run_end) {
        const struct dora_schema* const edge =
            &image->children[image->nodes[node].first_child_index + (run_end ? 1 : 0)];
        int status = image_child_field(image, edge, &value);
        if (status != DORA_OK) {
            return status;
        }
        if (field->type.id == DORA_TYPE_UNION) {
            struct dora_schema located = dora_schema_borrow(edge);
            while (value.layout == DORA_LAYOUT_DICTIONARY || value.layout == DORA_LAYOUT_RUN_END) {
                const uint32_t index = value.layout == DORA_LAYOUT_RUN_END ? 1 : 0;
                if (located.handle.slot_index == 0) {
                    const struct dora_schema* const next =
                        &image->children[image->nodes[located.node_index].first_child_index + index];
                    located = dora_schema_borrow(next);
                    status = image_child_field(image, next, &value);
                } else {
                    struct dora_schema child = {};
                    status = dora_schema_get_physical_child(&located, index, &child);
                    if (status == DORA_OK) {
                        located = child;
                        status = dora_schema_get_field(&located, &value);
                    }
                }
                if (status != DORA_OK) {
                    return status;
                }
            }
        }
        const uint32_t code_count = field->type.id == DORA_TYPE_UNION ? value.child_count : 0;
        if (field->type.id == DORA_TYPE_UNION) {
            status = union_codes_copy(field, source, code_count);
            if (status != DORA_OK) {
                return status;
            }
        }
        if (!dora_type_is_equal(&field->type, &value.type, code_count)) {
            return EINVAL;
        }
    } else if (field->type.id == DORA_TYPE_UNION) {
        const int status = union_codes_copy(field, source, field->child_count);
        if (status != DORA_OK) {
            return status;
        }
    }
    const bool map = !dictionary && !run_end && field->type.id == DORA_TYPE_MAP;
    if (run_end || map) {
        const struct dora_schema* const edge = &image->children[image->nodes[node].first_child_index];
        struct dora_schema_field child = {};
        int status = image_child_field(image, edge, &child);
        if (status != DORA_OK) {
            return status;
        }
        if (run_end && (child.type.id != field->index_type || child.layout != DORA_LAYOUT_FIXED ||
                        child.type.extension_name.byte_count != 0)) {
            return EINVAL;
        }
        if (map &&
            (child.type.id != DORA_TYPE_STRUCT || child.layout != DORA_LAYOUT_STRUCT || child.child_count != 2)) {
            return EINVAL;
        }
        if (edge->handle.slot_index == 0) {
            image->nodes[edge->node_index].field.flags |= DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
        } else if ((child.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
            return EINVAL;
        }
        if (map) {
            struct dora_schema key = {};
            if (edge->handle.slot_index == 0) {
                key = image->children[image->nodes[edge->node_index].first_child_index];
            } else {
                status = dora_schema_get_physical_child(edge, 0, &key);
            }
            if (status == DORA_OK) {
                status = image_child_field(image, &key, &child);
            }
            if (status != DORA_OK) {
                return status;
            }
            if (key.handle.slot_index == 0) {
                image->nodes[key.node_index].field.flags |= DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
            } else if ((child.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
                return EINVAL;
            }
        }
    }
    return DORA_OK;
}

static int
field_image_allocate(struct dora_pool pool, const struct dora_schema_field* field, struct schema_image** out_image) {
    struct schema_size size = {};
    int status = measure_field(&size, field);
    if (status != DORA_OK) {
        return status;
    }
    struct schema_image* image = nullptr;
    struct dora_metadata_entry* metadata = nullptr;
    unsigned char* bytes = nullptr;
    status = image_allocate(pool, 1, field->child_count, &size, &image, &metadata, &bytes);
    if (status != DORA_OK) {
        return status;
    }
    copy_field(&image->nodes[0].field, field, &metadata, &bytes);
    status = dora_type_resolve_layout(&field->type, field->layout, field->index_type, &image->nodes[0].field.layout);
    if (status != DORA_OK) {
        image_retire(image);
        return status;
    }
    *out_image = image;
    return DORA_OK;
}

int dora_schema_field_matches(const struct dora_schema* schema,
                              const struct dora_schema_field* field,
                              bool* out_matches) {
    struct dora_schema_field source = {};
    if (field == nullptr || out_matches == nullptr) {
        return EINVAL;
    }
    int status = dora_schema_get_field(schema, &source);
    enum dora_layout layout = DORA_LAYOUT_DEFAULT;
    if (status == DORA_OK) {
        status = field_check(field, &layout);
    }
    if (status != DORA_OK) {
        return status;
    }
    *out_matches = false;
    if (source.layout != layout || source.index_type != field->index_type || source.flags != field->flags ||
        source.child_count != field->child_count || source.metadata_entry_count != field->metadata_entry_count ||
        strcmp(source.name, field->name != nullptr ? field->name : "") != 0 ||
        (field->metadata_entry_count != 0 && field->metadata == nullptr)) {
        return DORA_OK;
    }
    uint32_t union_children = source.child_count;
    if (source.type.id == DORA_TYPE_UNION &&
        (source.layout == DORA_LAYOUT_DICTIONARY || source.layout == DORA_LAYOUT_RUN_END)) {
        struct dora_schema values = dora_schema_borrow(schema);
        struct dora_schema_field value = source;
        while (value.layout == DORA_LAYOUT_DICTIONARY || value.layout == DORA_LAYOUT_RUN_END) {
            struct dora_schema child = {};
            status = dora_schema_get_physical_child(&values, value.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
            if (status == DORA_OK) {
                values = child;
                status = dora_schema_get_field(&values, &value);
            }
            if (status != DORA_OK) {
                return status;
            }
        }
        union_children = value.child_count;
    }
    if (!dora_type_is_equal(&source.type, &field->type, union_children)) {
        return DORA_OK;
    }
    for (uint32_t index = 0; index < source.metadata_entry_count; ++index) {
        if (!bytes_are_well_formed(field->metadata[index].key) ||
            !bytes_are_well_formed(field->metadata[index].value) ||
            !bytes_are_equal(source.metadata[index].key, field->metadata[index].key) ||
            !bytes_are_equal(source.metadata[index].value, field->metadata[index].value)) {
            return DORA_OK;
        }
    }
    *out_matches = true;
    return DORA_OK;
}

int dora_schema_declarations_equal(const struct dora_schema* left, const struct dora_schema* right, bool* out_equal) {
    const struct schema_image* first = nullptr;
    const struct schema_image* second = nullptr;
    if (out_equal == nullptr) {
        return EINVAL;
    }
    int status = dora_schema_resolve_image(left, &first);
    if (status == DORA_OK) {
        status = dora_schema_resolve_image(right, &second);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (first == second && left->node_index == right->node_index) {
        *out_equal = true;
        return DORA_OK;
    }
    const struct dora_schema_field* const field = &second->nodes[right->node_index].field;
    status = dora_schema_field_matches(left, field, out_equal);
    for (uint32_t index = 0; status == DORA_OK && *out_equal && index < field->child_count; ++index) {
        struct dora_schema first_child = {};
        struct dora_schema second_child = {};
        status = dora_schema_get_physical_child(left, index, &first_child);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(right, index, &second_child);
        }
        if (status == DORA_OK) {
            status = dora_schema_declarations_equal(&first_child, &second_child, out_equal);
        }
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

void dora_schema_release(struct dora_schema* schema) {
    if (schema == nullptr) {
        return;
    }
    struct dora_handle owner = schema->handle;
    *schema = (struct dora_schema){};
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_SCHEMA, &retired);
    image_retire(retired);
}

bool dora_schema_is_valid(const struct dora_schema* schema) {
    const struct schema_image* image = nullptr;
    return dora_schema_resolve_image(schema, &image) == DORA_OK;
}

int dora_schema_retain(const struct dora_schema* schema, struct dora_schema* out_schema) {
    if (!schema_is_empty(out_schema) || !dora_schema_is_valid(schema)) {
        return EINVAL;
    }
    struct dora_handle owner = {};
    const int status = dora_identity_claim_acquire((struct dora_pool){}, &schema->handle, IDENTITY_KIND_SCHEMA, &owner);
    if (status == DORA_OK) {
        if (!schema_is_empty(out_schema)) {
            void* retired = nullptr;
            dora_identity_claim_release(&owner, IDENTITY_KIND_SCHEMA, &retired);
            image_retire(retired);
            return EINVAL;
        }
        *out_schema = (struct dora_schema){.handle = owner, .node_index = schema->node_index};
    }
    return status;
}

static int node_publish(struct dora_schema* schema,
                        struct dora_pool pool,
                        const struct dora_schema_field* field,
                        const struct dora_schema* const children,
                        const struct dora_schema* source) {
    const uint32_t child_count = field->child_count;
    struct schema_image* image = nullptr;
    int status = field_image_allocate(pool, field, &image);
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t index = 0; status == DORA_OK && index < child_count; ++index) {
        struct dora_schema child = {};
        if (children != nullptr) {
            child = children[index];
        } else {
            status = dora_schema_get_physical_child(source, index, &child);
        }
        if (status == DORA_OK) {
            status =
                dora_identity_claim_acquire(pool, &child.handle, IDENTITY_KIND_SCHEMA, &image->children[index].handle);
            image->children[index].node_index = child.node_index;
        }
    }
    if (status == DORA_OK) {
        status = image_check_node(image, 0, field);
    }
    if (status != DORA_OK) {
        image_retire(image);
        return status;
    }
    return image_publish(image, schema);
}

int dora_schema_share(const struct dora_schema* schema, struct dora_pool pool, struct dora_schema* out_shared) {
    struct dora_pool owner = {};
    struct dora_schema_field field = {};
    int status = schema_is_empty(out_shared) ? dora_schema_get_pool(schema, &owner) : EINVAL;
    if (status != DORA_OK) {
        return status;
    }
    if (pool_row_of(owner)->is_draining) {
        return EBUSY;
    }
    dora_pool_callback_enter(owner);
    if (pool_is_same(owner, pool)) {
        status = dora_schema_retain(schema, out_shared);
    } else {
        status = dora_schema_get_field(schema, &field);
        if (status == DORA_OK) {
            status = node_publish(out_shared, pool, &field, nullptr, schema);
        }
    }
    dora_pool_callback_leave(owner);
    return status;
}

int dora_schema_move(struct dora_schema* in_out_source, struct dora_schema* out_destination) {
    if (!schema_is_empty(out_destination) || !dora_schema_is_valid(in_out_source) || !in_out_source->handle.is_owner) {
        return EINVAL;
    }
    *out_destination = *in_out_source;
    *in_out_source = (struct dora_schema){};
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

static int schemas_get_common_pool(const struct dora_schema* schemas, uint32_t count, struct dora_pool* out_pool) {
    struct dora_pool pool = {};
    int status = schemas == nullptr || count == 0 || out_pool == nullptr ? EINVAL : DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        struct dora_pool next = {};
        status = dora_schema_get_pool(&schemas[index], &next);
        if (status == DORA_OK && pool.index == 0) {
            pool = next;
        } else if (status == DORA_OK && !pool_is_same(pool, next)) {
            status = EINVAL;
        }
    }
    if (status == DORA_OK) {
        *out_pool = pool;
    }
    return status;
}

int dora_schema_init_copy(struct dora_schema* schema, struct dora_pool pool, const struct dora_schema_input* input) {
    if (!schema_is_empty(schema) || input == nullptr || (input->fields == nullptr && input->field_count != 0)) {
        return EINVAL;
    }
    if (input->field_count == UINT32_MAX) {
        return EOVERFLOW;
    }
    struct schema_size size = {};
    const struct dora_schema_field root = {
        .type = {.id = DORA_TYPE_STRUCT},
        .layout = DORA_LAYOUT_STRUCT,
        .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
        .metadata = input->metadata,
        .metadata_entry_count = input->metadata_entry_count
    };
    int status = measure_field(&size, &root);
    uint64_t edges = 0;
    for (uint32_t index = 0; status == DORA_OK && index < input->field_count; ++index) {
        status = measure_field(&size, &input->fields[index]);
        edges += input->fields[index].child_count;
        if (edges > input->field_count) {
            status = EINVAL;
        }
    }
    if (status != DORA_OK) {
        return status;
    }
    struct schema_image* image = nullptr;
    struct dora_metadata_entry* metadata = nullptr;
    unsigned char* bytes = nullptr;
    status = image_allocate(pool, input->field_count + 1, input->field_count, &size, &image, &metadata, &bytes);
    if (status != DORA_OK) {
        return status;
    }
    copy_field(&image->nodes[0].field, &root, &metadata, &bytes);
    uint32_t edge_index = 0;
    for (uint32_t index = 1; index < image->node_count; ++index) {
        struct schema_node* const node = &image->nodes[index];
        copy_field(&node->field, &input->fields[index - 1], &metadata, &bytes);
        status = dora_type_resolve_layout(&node->field.type,
                                          node->field.layout,
                                          node->field.index_type,
                                          &node->field.layout);
        if (status != DORA_OK) {
            image_retire(image);
            return status;
        }
        node->first_child_index = edge_index;
        edge_index += node->field.child_count;
    }
    for (uint32_t index = image->node_count; index > 1 && status == DORA_OK;) {
        --index;
        struct schema_node* const node = &image->nodes[index];
        uint32_t next = index + 1;
        for (uint32_t child = 0; child < node->field.child_count; ++child) {
            if (next >= image->node_count) {
                status = EINVAL;
                break;
            }
            image->children[node->first_child_index + child].node_index = next;
            next = image->nodes[next].subtree_end;
        }
        node->subtree_end = next;
        if (status == DORA_OK) {
            status = image_check_node(image, index, &input->fields[index - 1]);
        }
    }
    image->nodes[0].first_child_index = edge_index;
    for (uint32_t next = 1; status == DORA_OK && next < image->node_count;) {
        if (edge_index >= image->child_count) {
            status = EINVAL;
            break;
        }
        image->children[edge_index++].node_index = next;
        ++image->nodes[0].field.child_count;
        next = image->nodes[next].subtree_end;
    }
    if (status != DORA_OK) {
        image_retire(image);
        return status;
    }
    return image_publish(image, schema);
}

int dora_schema_init_retain_node(struct dora_schema* schema,
                                 struct dora_pool pool,
                                 const struct dora_schema_field* field,
                                 const struct dora_schema* children) {
    if (!schema_is_empty(schema) || field == nullptr || (children == nullptr && field->child_count != 0)) {
        return EINVAL;
    }
    const uint32_t child_count = field->child_count;
    for (uint32_t index = 0; index < child_count; ++index) {
        if (!dora_schema_is_valid(&children[index])) {
            return EINVAL;
        }
    }
    return node_publish(schema, pool, field, children, nullptr);
}

int dora_schema_init_retain_struct_in_pool(struct dora_schema* schema,
                                           struct dora_pool pool,
                                           const struct dora_schema* children,
                                           uint32_t child_count) {
    const struct dora_schema_field field = {
        .type = {.id = DORA_TYPE_STRUCT},
        .layout = DORA_LAYOUT_STRUCT,
        .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
        .child_count = child_count
    };
    return dora_schema_init_retain_node(schema, pool, &field, children);
}

int dora_schema_init_retain_struct(struct dora_schema* schema,
                                   const struct dora_schema* children,
                                   uint32_t child_count) {
    struct dora_pool pool = {};
    const int placed = schemas_get_common_pool(children, child_count, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_schema_init_retain_struct_in_pool(schema, pool, children, child_count);
}

int dora_schema_init_retain_field(struct dora_schema* schema,
                                  struct dora_pool pool,
                                  const struct dora_schema* source,
                                  const struct dora_schema_field* field) {
    struct dora_schema_field source_field = {};
    int status = dora_schema_get_field(source, &source_field);
    const uint32_t child_count = source_field.child_count;
    if (status != DORA_OK || field == nullptr || field->child_count != child_count || !schema_is_empty(schema)) {
        return EINVAL;
    }
    bool unchanged = false;
    status = dora_schema_field_matches(source, field, &unchanged);
    if (status != DORA_OK) {
        return status;
    }
    if (unchanged) {
        struct dora_handle claim = {};
        status = dora_identity_claim_acquire(pool, &source->handle, IDENTITY_KIND_SCHEMA, &claim);
        if (status == DORA_OK) {
            *schema = (struct dora_schema){.handle = claim, .node_index = source->node_index};
        }
        return status;
    }
    return node_publish(schema, pool, field, nullptr, source);
}
