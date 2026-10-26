// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/statistics.h"

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "extension/extension_internal.h"
#include "memory/memory_internal.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "types/array_count.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

const struct statistics_descriptor dora_statistics_descriptors[STATISTICS_KIND_COUNT] = {
    {"row_count", "row_count_estimate", DORA_STATISTICS_KIND_ROW_COUNT, DORA_TYPE_INT64},
    {"null_count", "null_count_estimate", DORA_STATISTICS_KIND_NULL_COUNT, DORA_TYPE_INT64},
    {"nan_count", "nan_count_estimate", DORA_STATISTICS_KIND_NAN_COUNT, DORA_TYPE_INT64},
    {"min", "min_estimate", DORA_STATISTICS_KIND_MIN, DORA_TYPE_NULL},
    {"max", "max_estimate", DORA_STATISTICS_KIND_MAX, DORA_TYPE_NULL},
    {"distinct_count", "distinct_count_estimate", DORA_STATISTICS_KIND_DISTINCT_COUNT, DORA_TYPE_INT64},
    {"min_byte_width", "min_byte_width_estimate", DORA_STATISTICS_KIND_MIN_BYTE_WIDTH, DORA_TYPE_INT64},
    {"max_byte_width", "max_byte_width_estimate", DORA_STATISTICS_KIND_MAX_BYTE_WIDTH, DORA_TYPE_INT64},
    {"average_byte_width", "average_byte_width_estimate", DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH, DORA_TYPE_FLOAT64}
};

static int semantic_policy(struct dora_pool pool, const struct dora_schema* schema, uint32_t required) {
    if (required == 0) {
        return DORA_OK;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status == DORA_OK && field.type.id == DORA_TYPE_MAP && (required & DORA_EXTENSION_FLAG_STORAGE_ORDER) != 0) {
        return ENOTSUP;
    }
    if (status == DORA_OK && field.type.extension_name.byte_count != 0) {
        const struct dora_extension_definition* definition = nullptr;
        status = dora_extension_find(pool, field.type.extension_name, &definition);
        if (status == DORA_OK && (definition == nullptr || (definition->flags & required) != required)) {
            status = ENOTSUP;
        }
    }
    for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
        struct dora_schema member = {};
        status = dora_schema_get_physical_child(schema, child, &member);
        if (status == DORA_OK) {
            status = semantic_policy(pool, &member, required);
        }
    }
    return status;
}

static int bound_schema(struct dora_schema* out_schema,
                        struct dora_pool pool,
                        const struct dora_schema* source,
                        const struct dora_schema_field* declaration) {
    struct dora_schema_field field = *declaration;
    field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
    const uint32_t nullable_child = field.layout == DORA_LAYOUT_RUN_END ? 1 : 0;
    const bool rewrite_child = field.layout == DORA_LAYOUT_RUN_END || field.type.id == DORA_TYPE_UNION;
    struct dora_allocation storage = {};
    size_t bytes = 0;
    if (ckd_mul(&bytes, field.child_count, sizeof(struct dora_schema))) {
        return EOVERFLOW;
    }
    int status = dora_memory_payload_allocate(pool, bytes, &storage);
    struct dora_schema* const children = storage.address;
    if (status == DORA_OK && bytes != 0) {
        memset(children, 0, bytes);
    }
    for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
        struct dora_schema original = {};
        status = dora_schema_get_physical_child(source, child, &original);
        if (status == DORA_OK && rewrite_child && child == nullable_child) {
            struct dora_schema_field child_field = {};
            status = dora_schema_get_field(&original, &child_field);
            if (status == DORA_OK) {
                status = bound_schema(&children[child], pool, &original, &child_field);
            }
        } else if (status == DORA_OK) {
            status = dora_schema_retain(&original, &children[child]);
        }
    }
    if (status == DORA_OK) {
        status = dora_schema_node_claim(pool, &field, children, out_schema);
    }
    for (uint32_t child = 0; children != nullptr && child < field.child_count; ++child) {
        dora_schema_release(&children[child]);
    }
    dora_allocation_release(&storage);
    return status;
}

static int group_policy(struct dora_pool pool,
                        const struct dora_schema* target,
                        const struct dora_schema_field* field,
                        uint32_t all) {
    const bool variable = field->type.id == DORA_TYPE_BINARY || field->type.id == DORA_TYPE_UTF8 ||
                          field->type.id == DORA_TYPE_LIST || field->type.id == DORA_TYPE_MAP;
    const bool floating = field->type.id >= DORA_TYPE_FLOAT16 && field->type.id <= DORA_TYPE_FLOAT64;
    if ((!floating && (all & DORA_STATISTICS_KIND_NAN_COUNT) != 0) ||
        (!variable &&
         (all & ((uint32_t)DORA_STATISTICS_KIND_MIN_BYTE_WIDTH | (uint32_t)DORA_STATISTICS_KIND_MAX_BYTE_WIDTH |
                 (uint32_t)DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH)) != 0)) {
        return EINVAL;
    }
    uint32_t required = (all & DORA_STATISTICS_KIND_DISTINCT_COUNT) != 0 ? DORA_EXTENSION_FLAG_STORAGE_EQUALITY : 0;
    if ((all & ((uint32_t)DORA_STATISTICS_KIND_MIN | (uint32_t)DORA_STATISTICS_KIND_MAX)) != 0) {
        required |= DORA_EXTENSION_FLAG_STORAGE_ORDER;
    }
    return semantic_policy(pool, target, required);
}

static int group_schema(struct dora_schema* out_schema,
                        struct dora_pool pool,
                        const struct dora_schema* target,
                        uint16_t requested,
                        uint16_t estimates) {
    struct dora_schema_field target_field = {};
    int status = dora_schema_get_field(target, &target_field);
    if (status != DORA_OK) {
        return status;
    }
    status = group_policy(pool, target, &target_field, (uint32_t)requested | estimates);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_schema children[STATISTICS_GROUP_FIELD_CAPACITY] = {};
    uint32_t count = 0;
    for (uint32_t index = 0; status == DORA_OK && index < ARRAY_COUNT(dora_statistics_descriptors); ++index) {
        const struct statistics_descriptor descriptor = dora_statistics_descriptors[index];
        if ((requested & descriptor.kind) == 0) {
            continue;
        }
        if (descriptor.kind == DORA_STATISTICS_KIND_MIN || descriptor.kind == DORA_STATISTICS_KIND_MAX) {
            struct dora_schema_field field = target_field;
            field.name = descriptor.name;
            field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
            status = bound_schema(&children[count], pool, target, &field);
            ++count;
            if (status == DORA_OK) {
                const struct dora_schema_field exact = {
                    .name = descriptor.kind == DORA_STATISTICS_KIND_MIN ? "is_min_exact" : "is_max_exact",
                    .type.id = DORA_TYPE_BOOLEAN
                };
                status = dora_schema_node_claim(pool, &exact, nullptr, &children[count]);
                ++count;
            }
        } else {
            const struct dora_schema_field field = {.name = descriptor.name, .type.id = descriptor.type};
            status = dora_schema_node_claim(pool, &field, nullptr, &children[count]);
            ++count;
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < ARRAY_COUNT(dora_statistics_descriptors); ++index) {
        const struct statistics_descriptor descriptor = dora_statistics_descriptors[index];
        if ((estimates & descriptor.kind) == 0) {
            continue;
        }
        const char* const name = descriptor.estimate_name;
        if (descriptor.kind == DORA_STATISTICS_KIND_MIN || descriptor.kind == DORA_STATISTICS_KIND_MAX) {
            struct dora_schema_field field = target_field;
            field.name = name;
            field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
            status = bound_schema(&children[count], pool, target, &field);
        } else {
            const struct dora_schema_field field = {.name = name, .type.id = DORA_TYPE_FLOAT64};
            status = dora_schema_node_claim(pool, &field, nullptr, &children[count]);
        }
        ++count;
    }
    const struct dora_schema_field group = {
        .name = target_field.name,
        .type.id = DORA_TYPE_STRUCT,
        .layout = DORA_LAYOUT_STRUCT,
        .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
        .child_count = count
    };
    if (status == DORA_OK) {
        status = dora_schema_node_claim(pool, &group, children, out_schema);
    }
    for (uint32_t index = 0; index < count; ++index) {
        dora_schema_release(&children[index]);
    }
    return status;
}

static int
bound_matches(const struct dora_schema* actual, const struct dora_schema* target, const char* name, bool* out_matches) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(target, &field);
    if (status != DORA_OK) {
        return status;
    }
    if (name != nullptr) {
        field.name = name;
    }
    field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
    status = dora_schema_field_matches(actual, &field, out_matches);
    const uint32_t nullable_child = field.layout == DORA_LAYOUT_RUN_END ? 1 : 0;
    for (uint32_t child = 0; status == DORA_OK && *out_matches && child < field.child_count; ++child) {
        struct dora_schema source_child = {};
        struct dora_schema actual_child = {};
        status = dora_schema_get_physical_child(target, child, &source_child);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(actual, child, &actual_child);
        }
        if (status == DORA_OK && child == nullable_child &&
            (field.layout == DORA_LAYOUT_RUN_END || field.type.id == DORA_TYPE_UNION)) {
            status = bound_matches(&actual_child, &source_child, nullptr, out_matches);
        } else if (status == DORA_OK) {
            status = dora_schema_declarations_equal(&actual_child, &source_child, out_matches);
        }
    }
    return status;
}

static int group_check(struct dora_pool pool, const struct dora_schema* actual, const struct dora_schema* target) {
    struct dora_schema_field source = {};
    struct dora_schema_field group = {};
    int status = dora_schema_get_field(target, &source);
    if (status == DORA_OK) {
        status = dora_schema_get_field(actual, &group);
    }
    bool matches = false;
    const struct dora_schema_field expected = {
        .name = source.name,
        .type.id = DORA_TYPE_STRUCT,
        .layout = DORA_LAYOUT_STRUCT,
        .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
        .child_count = group.child_count
    };
    if (status == DORA_OK) {
        status = dora_schema_field_matches(actual, &expected, &matches);
    }
    if (status != DORA_OK || !matches) {
        return status != DORA_OK ? status : EINVAL;
    }
    uint16_t requested = 0;
    uint16_t estimates = 0;
    for (uint32_t child = 0; status == DORA_OK && child < group.child_count; ++child) {
        struct dora_schema member = {};
        struct dora_schema_field field = {};
        status = dora_schema_get_physical_child(actual, child, &member);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&member, &field);
        }
        if (status != DORA_OK) {
            return status;
        }
        bool known = strcmp(field.name, "is_min_exact") == 0 || strcmp(field.name, "is_max_exact") == 0;
        for (uint32_t kind = 0; kind < ARRAY_COUNT(dora_statistics_descriptors); ++kind) {
            if (strcmp(field.name, dora_statistics_descriptors[kind].name) == 0) {
                requested |= dora_statistics_descriptors[kind].kind;
                known = true;
            }
            if (strcmp(field.name, dora_statistics_descriptors[kind].estimate_name) == 0) {
                estimates |= dora_statistics_descriptors[kind].kind;
                known = true;
            }
        }
        if (!known) {
            return EINVAL;
        }
    }
    status = group_policy(pool, target, &source, (uint32_t)requested | estimates);
    uint32_t child = 0;
    for (uint32_t pass = 0; status == DORA_OK && pass < 2; ++pass) {
        const uint16_t kinds = pass == 0 ? requested : estimates;
        for (uint32_t kind = 0; status == DORA_OK && kind < ARRAY_COUNT(dora_statistics_descriptors); ++kind) {
            const struct statistics_descriptor descriptor = dora_statistics_descriptors[kind];
            if ((kinds & descriptor.kind) == 0) {
                continue;
            }
            const char* const name = pass == 0 ? descriptor.name : descriptor.estimate_name;
            struct dora_schema member = {};
            status = dora_schema_get_physical_child(actual, child++, &member);
            const bool bound =
                descriptor.kind == DORA_STATISTICS_KIND_MIN || descriptor.kind == DORA_STATISTICS_KIND_MAX;
            if (status == DORA_OK && bound) {
                status = bound_matches(&member, target, name, &matches);
            } else if (status == DORA_OK) {
                const struct dora_schema_field field = {
                    .name = name,
                    .type.id = pass == 0 ? descriptor.type : DORA_TYPE_FLOAT64
                };
                status = dora_schema_field_matches(&member, &field, &matches);
            }
            if (status == DORA_OK && !matches) {
                status = EINVAL;
            }
            if (status == DORA_OK && bound && pass == 0) {
                const struct dora_schema_field field = {
                    .name = descriptor.kind == DORA_STATISTICS_KIND_MIN ? "is_min_exact" : "is_max_exact",
                    .type.id = DORA_TYPE_BOOLEAN
                };
                member = (struct dora_schema){};
                status = dora_schema_get_physical_child(actual, child++, &member);
                if (status == DORA_OK) {
                    status = dora_schema_field_matches(&member, &field, &matches);
                }
                if (status == DORA_OK && !matches) {
                    status = EINVAL;
                }
            }
        }
    }
    return status != DORA_OK ? status : child == group.child_count ? DORA_OK : EINVAL;
}

//------------------------------------------------------------------------------
//
// Statistics Attachment
//
//------------------------------------------------------------------------------

static int statistics_publish(struct dora_pool pool,
                              const struct dora_vector* data,
                              const struct dora_schema* schema,
                              const struct dora_vector* statistics,
                              struct dora_error* out_error,
                              struct dora_vector* out_vector) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    struct vector_image* image = nullptr;
    if (status == DORA_OK) {
        status = dora_vector_image_share(pool, data, &field, &image, out_error);
    }
    if (status == DORA_OK) {
        status = dora_vector_set_statistics_claim(image, statistics, VECTOR_STATISTICS_IMPORTED);
    }
    if (status == DORA_OK) {
        status = dora_vector_bind_extension(image, out_error);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, out_vector);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
    }
    return status;
}

int dora_statistics_attach_error(struct dora_pool pool,
                                 const struct dora_vector* data,
                                 const struct dora_vector* statistics,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    if (!dora_vector_is_empty(out_vector) || !dora_vector_is_valid(data) || !dora_vector_is_valid(statistics) ||
        statistics->rows.length != 1) {
        return EINVAL;
    }
    struct dora_schema data_schema = {};
    struct dora_schema actual = {};
    struct dora_vector published = {};
    int status = dora_vector_retain_schema(data, &data_schema);
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(statistics, &actual);
    }
    struct dora_allocation walk = {};
    uint32_t count = 0;
    if (status == DORA_OK) {
        status = dora_schema_walk(pool, &data_schema, &walk, &count);
    }
    const struct dora_schema_field root = {
        .type.id = DORA_TYPE_STRUCT,
        .layout = DORA_LAYOUT_STRUCT,
        .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
        .child_count = count
    };
    bool matches = false;
    if (status == DORA_OK) {
        status = dora_schema_field_matches(&actual, &root, &matches);
    }
    if (status == DORA_OK && !matches) {
        status = EINVAL;
    }
    const struct schema_walk_node* const nodes = walk.address;
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        struct dora_schema group = {};
        status = dora_schema_get_physical_child(&actual, index, &group);
        if (status == DORA_OK) {
            status = group_check(pool, &group, &nodes[index].schema);
        }
    }
    if (status == DORA_OK) {
        status = statistics_publish(pool, data, &data_schema, statistics, out_error, &published);
    }
    dora_schema_release(&actual);
    dora_schema_release(&data_schema);
    dora_allocation_release(&walk);
    return vector_finish_publication(&published, out_vector, status);
}

//------------------------------------------------------------------------------
//
// Schema Construction
//
//------------------------------------------------------------------------------

int dora_statistics_schema_init_in_pool(struct dora_schema* schema,
                                        struct dora_pool pool,
                                        const struct dora_schema* target,
                                        const struct dora_statistics_request* requests,
                                        uint32_t request_count) {
    if (schema == nullptr || schema->handle.slot_index != 0 || schema->handle.generation != 0 ||
        schema->handle.pool_index != 0 || schema->handle.is_owner || schema->handle.reserved != 0 ||
        schema->node_index != 0 || (request_count != 0 && requests == nullptr)) {
        return EINVAL;
    }
    struct dora_schema published = {};
    struct dora_allocation walk = {};
    uint32_t count = 0;
    int status = dora_schema_walk(pool, target, &walk, &count);
    for (uint32_t index = 0; status == DORA_OK && index < request_count; ++index) {
        if (requests[index].schema_node_index >= count ||
            (requests[index].kinds & (uint16_t)~STATISTICS_KIND_MASK) != 0) {
            status = EINVAL;
        }
    }
    size_t bytes = 0;
    struct dora_allocation groups = {};
    if (status == DORA_OK && ckd_mul(&bytes, count, sizeof(struct dora_schema))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &groups);
    }
    struct dora_schema* const schemas = groups.address;
    if (status == DORA_OK) {
        memset(schemas, 0, bytes);
        const struct schema_walk_node* const nodes = walk.address;
        for (uint32_t node = 0; status == DORA_OK && node < count; ++node) {
            uint16_t kinds = 0;
            for (uint32_t index = 0; index < request_count; ++index) {
                if (requests[index].schema_node_index == node) {
                    kinds |= requests[index].kinds;
                }
            }
            status = group_schema(&schemas[node], pool, &nodes[node].schema, kinds, 0);
        }
        if (status == DORA_OK) {
            const struct dora_schema_field root = {
                .type = {.id = DORA_TYPE_STRUCT},
                .layout = DORA_LAYOUT_STRUCT,
                .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
                .child_count = count
            };
            status = dora_schema_node_claim(pool, &root, schemas, &published);
        }
        for (uint32_t node = 0; node < count; ++node) {
            dora_schema_release(&schemas[node]);
        }
    }
    dora_allocation_release(&groups);
    dora_allocation_release(&walk);
    if (status == DORA_OK) {
        status = dora_schema_move(&published, schema);
    }
    dora_schema_release(&published);
    return status;
}

int dora_statistics_schema_init(struct dora_schema* schema,
                                const struct dora_schema* target,
                                const struct dora_statistics_request* requests,
                                uint32_t request_count) {
    struct dora_pool pool = {};
    const int placed = dora_schema_get_pool(target, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_statistics_schema_init_in_pool(schema, pool, target, requests, request_count);
}

//------------------------------------------------------------------------------
//
// Computation
//
//------------------------------------------------------------------------------

int dora_statistics_compute(const struct dora_vector* source,
                            const struct dora_statistics_request* requests,
                            uint32_t request_count,
                            struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result)) {
        return EINVAL;
    }
    struct dora_vector retained = {};
    struct dora_schema schema = {};
    struct dora_vector statistics = {};
    struct statistics_accumulator* accumulator = nullptr;
    struct dora_vector published = {};
    int status = dora_vector_retain(source, &retained);
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(&retained, &schema);
    }
    if (status == DORA_OK) {
        status = dora_statistics_accumulator_open(pool, &schema, requests, request_count, &accumulator);
    }
    if (status == DORA_OK) {
        status = dora_statistics_accumulator_update(accumulator, &retained);
    }
    if (status == DORA_OK) {
        status = dora_statistics_accumulator_finish(accumulator, &statistics, nullptr);
    }
    if (status == DORA_OK) {
        status = statistics_publish(pool, &retained, &schema, &statistics, nullptr, &published);
    }
    dora_vector_release(&statistics);
    dora_statistics_accumulator_release(&accumulator);
    dora_schema_release(&schema);
    dora_vector_release(&retained);
    return vector_finish_publication(&published, out_result, status);
}
