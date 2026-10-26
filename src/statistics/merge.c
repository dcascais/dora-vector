// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
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

static constexpr uint32_t STATISTIC_ESTIMATE_RANK_OFFSET = 20;

//------------------------------------------------------------------------------
//
// Merge
//
//------------------------------------------------------------------------------

struct merge_field {
    struct dora_schema schema;
    struct dora_allocation value;
    long double weighted_sum;
    uint64_t weight;
    uint64_t populations;
    bool was_seen;
    bool is_unknown;
    bool is_exact;
};

static bool statistics_name_is_recognized(const char* name) {
    if (strcmp(name, "is_min_exact") == 0 || strcmp(name, "is_max_exact") == 0) {
        return true;
    }
    for (uint32_t index = 0; index < ARRAY_COUNT(dora_statistics_descriptors); ++index) {
        if (strcmp(name, dora_statistics_descriptors[index].name) == 0 ||
            strcmp(name, dora_statistics_descriptors[index].estimate_name) == 0) {
            return true;
        }
    }
    return false;
}

static int merge_value(struct dora_pool pool,
                       const struct dora_vector* group,
                       const char* name,
                       uint64_t row,
                       struct dora_allocation* out_value) {
    struct dora_vector member = {};
    const int status = dora_vector_find_child(group, name, &member);
    return status == ENOENT    ? DORA_OK
           : status != DORA_OK ? status
                               : dora_statistics_key_serialize_row(pool, &member, row, out_value);
}

static int
merge_count(const struct dora_vector* group, const char* name, uint64_t row, uint64_t* out_count, bool* out_known) {
    struct dora_vector member = {};
    int status = dora_vector_find_child(group, name, &member);
    if (status == ENOENT) {
        return DORA_OK;
    }
    int64_t value = 0;
    bool is_null = true;
    if (status == DORA_OK) {
        status = dora_vector_read_i64(&member, row, &value, &is_null);
    }
    if (status == DORA_OK && !is_null) {
        if (value < 0) {
            return EINVAL;
        }
        *out_count = (uint64_t)value;
        *out_known = true;
    }
    return status;
}

static int merge_exact(const struct dora_vector* group, const char* name, uint64_t row, bool* out_exact) {
    struct dora_vector member = {};
    int status = dora_vector_find_child(group, name, &member);
    if (status == ENOENT) {
        return DORA_OK;
    }
    bool is_null = true;
    bool value = false;
    if (status == DORA_OK) {
        status = dora_vector_read_bool(&member, row, &value, &is_null);
    }
    if (status == DORA_OK) {
        *out_exact = !is_null && value;
    }
    return status;
}

static int merge_collect(struct merge_field fields[STATISTICS_GROUP_FIELD_CAPACITY],
                         uint32_t* count,
                         const struct dora_vector* group) {
    struct dora_schema schema = {};
    struct dora_schema_field field = {};
    int status = dora_vector_retain_schema(group, &schema);
    if (status == DORA_OK) {
        status = dora_schema_get_field(&schema, &field);
    }
    if (status == DORA_OK && field.type.id != DORA_TYPE_STRUCT) {
        status = EINVAL;
    }
    for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
        struct dora_schema member = {};
        struct dora_schema_field declaration = {};
        status = dora_schema_get_child(&schema, child, &member);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&member, &declaration);
        }
        if (status != DORA_OK || !statistics_name_is_recognized(declaration.name)) {
            continue;
        }
        uint32_t index = 0;
        for (; status == DORA_OK && index < *count; ++index) {
            struct dora_schema_field existing = {};
            status = dora_schema_get_field(&fields[index].schema, &existing);
            if (status == DORA_OK && strcmp(existing.name, declaration.name) == 0) {
                break;
            }
        }
        if (status != DORA_OK) {
            break;
        }
        if (index != *count) {
            bool equal = false;
            status = dora_schema_declarations_equal(&fields[index].schema, &member, &equal);
            if (status == DORA_OK && !equal) {
                status = EINVAL;
            }
        } else if (*count == STATISTICS_GROUP_FIELD_CAPACITY) {
            status = EINVAL;
        } else {
            status = dora_schema_retain(&member, &fields[(*count)++].schema);
        }
    }
    dora_schema_release(&schema);
    return status;
}

static int merge_row(struct dora_pool pool,
                     const struct dora_vector* group,
                     uint64_t row,
                     struct merge_field fields[STATISTICS_GROUP_FIELD_CAPACITY],
                     uint32_t count) {
    uint64_t rows = 0;
    uint64_t nulls = 0;
    bool rows_known = false;
    bool nulls_known = false;
    int status = merge_count(group, "row_count", row, &rows, &rows_known);
    if (status == DORA_OK) {
        status = merge_count(group, "null_count", row, &nulls, &nulls_known);
    }
    if (status == DORA_OK && rows_known && nulls_known && nulls > rows) {
        status = EINVAL;
    }
    const bool no_rows = rows_known && rows == 0;
    const bool empty = no_rows || (rows_known && nulls_known && rows == nulls);
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        struct merge_field* const state = &fields[index];
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&state->schema, &field);
        if (status != DORA_OK) {
            break;
        }
        if (strcmp(field.name, "is_min_exact") == 0 || strcmp(field.name, "is_max_exact") == 0) {
            continue;
        }
        const bool minimum = strcmp(field.name, "min") == 0 || strcmp(field.name, "min_estimate") == 0;
        const bool maximum = strcmp(field.name, "max") == 0 || strcmp(field.name, "max_estimate") == 0;
        const bool distinct = strncmp(field.name, "distinct_count", sizeof("distinct_count") - 1) == 0;
        const bool average = strncmp(field.name, "average_byte_width", sizeof("average_byte_width") - 1) == 0;
        const bool min_width = strncmp(field.name, "min_byte_width", sizeof("min_byte_width") - 1) == 0;
        const bool max_width = strncmp(field.name, "max_byte_width", sizeof("max_byte_width") - 1) == 0;
        if (((minimum || maximum || distinct || min_width) && empty) || ((average || max_width) && no_rows)) {
            continue;
        }
        struct dora_allocation incoming = {};
        status = merge_value(pool, group, field.name, row, &incoming);
        const struct statistics_key* const key = incoming.address;
        const bool known = key != nullptr && !key->is_null;
        if (status == DORA_OK && known && !minimum && !maximum &&
            ((key->physical_type == DORA_PHYSICAL_TYPE_I64 && key->scalar.i64 < 0) ||
             (key->physical_type == DORA_PHYSICAL_TYPE_F64 && !(key->scalar.f64 >= 0)))) {
            status = EINVAL;
        }
        bool exact = false;
        if (status == DORA_OK && (minimum || maximum)) {
            status = merge_exact(group, minimum ? "is_min_exact" : "is_max_exact", row, &exact);
        }
        if (!known) {
            state->is_unknown = true;
        }
        if (status == DORA_OK && distinct && state->populations++ != 0) {
            state->is_unknown = true;
        }
        if (status == DORA_OK && known && average) {
            if (!rows_known || key->physical_type != DORA_PHYSICAL_TYPE_F64 ||
                ckd_add(&state->weight, state->weight, rows)) {
                state->is_unknown = true;
            } else {
                state->weighted_sum += (long double)key->scalar.f64 * (long double)rows;
            }
        } else if (status == DORA_OK && known && !state->is_unknown) {
            if (!state->was_seen) {
                state->value = incoming;
                incoming = (struct dora_allocation){};
                state->is_exact = exact;
            } else if (minimum || maximum || min_width || max_width) {
                int order = 0;
                status = dora_statistics_key_compare(key, state->value.address, &order);
                if (status == DORA_OK && (minimum || min_width ? order < 0 : order > 0)) {
                    dora_allocation_release(&state->value);
                    state->value = incoming;
                    incoming = (struct dora_allocation){};
                    state->is_exact = exact;
                } else if (status == DORA_OK && order == 0) {
                    state->is_exact = state->is_exact || exact;
                }
            } else if (!distinct) {
                struct statistics_key* const sum = state->value.address;
                if (sum->physical_type == DORA_PHYSICAL_TYPE_I64) {
                    if (key->scalar.i64 < 0 || ckd_add(&sum->scalar.i64, sum->scalar.i64, key->scalar.i64)) {
                        status = EOVERFLOW;
                    }
                } else if (sum->physical_type == DORA_PHYSICAL_TYPE_F64) {
                    sum->scalar.f64 += key->scalar.f64;
                } else {
                    status = EINVAL;
                }
            }
        }
        state->was_seen = true;
        dora_allocation_release(&incoming);
    }
    return status;
}

static uint32_t merge_rank(const char* name) {
    for (uint32_t index = 0; index < ARRAY_COUNT(dora_statistics_descriptors); ++index) {
        if (strcmp(name, dora_statistics_descriptors[index].name) == 0) {
            return index * 2;
        }
        if ((dora_statistics_descriptors[index].kind == DORA_STATISTICS_KIND_MIN &&
             strcmp(name, "is_min_exact") == 0) ||
            (dora_statistics_descriptors[index].kind == DORA_STATISTICS_KIND_MAX &&
             strcmp(name, "is_max_exact") == 0)) {
            return index * 2 + 1;
        }
        if (strcmp(name, dora_statistics_descriptors[index].estimate_name) == 0) {
            return STATISTIC_ESTIMATE_RANK_OFFSET + index;
        }
    }
    return UINT32_MAX;
}

static int merge_group(struct dora_pool pool,
                       const struct dora_vector* left,
                       const struct dora_vector* right,
                       struct dora_vector* out_group) {
    struct merge_field fields[STATISTICS_GROUP_FIELD_CAPACITY] = {};
    uint32_t count = 0;
    int status = merge_collect(fields, &count, left);
    if (status == DORA_OK && right != nullptr) {
        status = merge_collect(fields, &count, right);
    }
    for (uint32_t index = 1; status == DORA_OK && index < count; ++index) {
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&fields[index].schema, &field);
        const uint32_t rank = status == DORA_OK ? merge_rank(field.name) : 0;
        uint32_t position = index;
        while (status == DORA_OK && position != 0) {
            struct dora_schema_field previous = {};
            status = dora_schema_get_field(&fields[position - 1].schema, &previous);
            if (status != DORA_OK || merge_rank(previous.name) <= rank) {
                break;
            }
            const struct merge_field swap = fields[position - 1];
            fields[position - 1] = fields[position];
            fields[position] = swap;
            --position;
        }
    }
    const struct dora_vector* inputs[] = {left, right};
    for (uint32_t side = 0; status == DORA_OK && side < 2; ++side) {
        if (inputs[side] == nullptr) {
            continue;
        }
        for (uint64_t row = 0; status == DORA_OK && row < inputs[side]->rows.length; ++row) {
            status = merge_row(pool, inputs[side], row, fields, count);
        }
    }
    struct dora_vector children[STATISTICS_GROUP_FIELD_CAPACITY] = {};
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        struct merge_field* const state = &fields[index];
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&state->schema, &field);
        if (status != DORA_OK) {
            break;
        }
        struct statistics_key scalar = {.serialized_byte_count = sizeof(scalar), .type = field.type.id};
        const struct statistics_key* value = state->is_unknown ? nullptr : state->value.address;
        if (field.type.id == DORA_TYPE_BOOLEAN &&
            (strcmp(field.name, "is_min_exact") == 0 || strcmp(field.name, "is_max_exact") == 0)) {
            const char* const bound_name = strcmp(field.name, "is_min_exact") == 0 ? "min" : "max";
            for (uint32_t bound = 0; status == DORA_OK && bound < count; ++bound) {
                struct dora_schema_field declaration = {};
                status = dora_schema_get_field(&fields[bound].schema, &declaration);
                if (status == DORA_OK && strcmp(declaration.name, bound_name) == 0) {
                    scalar.scalar.boolean =
                        !fields[bound].is_unknown && fields[bound].was_seen && fields[bound].is_exact;
                }
            }
            value = &scalar;
        } else if (strncmp(field.name, "average_byte_width", sizeof("average_byte_width") - 1) == 0 &&
                   !state->is_unknown && state->weight != 0) {
            scalar.physical_type = DORA_PHYSICAL_TYPE_F64;
            scalar.scalar.f64 = (double)(state->weighted_sum / (long double)state->weight);
            value = &scalar;
        } else if (!state->was_seen && (strncmp(field.name, "distinct_count", sizeof("distinct_count") - 1) == 0 ||
                                        strncmp(field.name, "max_byte_width", sizeof("max_byte_width") - 1) == 0 ||
                                        strncmp(field.name, "row_count", sizeof("row_count") - 1) == 0 ||
                                        strncmp(field.name, "null_count", sizeof("null_count") - 1) == 0 ||
                                        strncmp(field.name, "nan_count", sizeof("nan_count") - 1) == 0)) {
            scalar.physical_type = field.type.id == DORA_TYPE_FLOAT64 ? DORA_PHYSICAL_TYPE_F64 : DORA_PHYSICAL_TYPE_I64;
            value = &scalar;
        }
        if (status == DORA_OK) {
            status = dora_statistics_key_materialize(pool, &state->schema, value, &children[index], nullptr);
        }
    }
    struct dora_schema schema = {};
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(left, &schema);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(&schema, &field);
    }
    if (status == DORA_OK) {
        struct dora_allocation validity = {};
        field.child_count = count;
        const struct vector_input composition = {
            .field = &field,
            .row_count = 1,
            .allocations = &validity,
            .buffer_count = 1,
            .children = children,
            .child_count = count
        };
        status = dora_vector_init_move_input(out_group, pool, &composition, &validity, nullptr);
    }
    dora_schema_release(&schema);
    for (uint32_t index = 0; index < STATISTICS_GROUP_FIELD_CAPACITY; ++index) {
        dora_vector_release(&children[index]);
        dora_allocation_release(&fields[index].value);
        dora_schema_release(&fields[index].schema);
    }
    return status;
}

int dora_statistics_merge(const struct dora_vector* left,
                          const struct dora_vector* right,
                          struct dora_vector* out_result) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(left, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (!dora_vector_is_empty(out_result) || !dora_vector_is_valid(left) ||
        (right != nullptr && !dora_vector_is_valid(right))) {
        return EINVAL;
    }
    struct dora_vector retained_left = {};
    struct dora_vector retained_right = {};
    struct dora_schema left_schema = {};
    struct dora_schema right_schema = {};
    struct dora_schema_field a = {};
    struct dora_schema_field b = {};
    struct dora_vector published = {};
    int status = dora_vector_retain(left, &retained_left);
    if (status == DORA_OK && right != nullptr) {
        status = dora_vector_retain(right, &retained_right);
    }
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(&retained_left, &left_schema);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(&left_schema, &a);
    }
    if (status == DORA_OK && a.type.id != DORA_TYPE_STRUCT) {
        status = EINVAL;
    }
    if (status == DORA_OK && right != nullptr) {
        status = dora_vector_retain_schema(&retained_right, &right_schema);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&right_schema, &b);
        }
        if (status == DORA_OK && (b.type.id != DORA_TYPE_STRUCT || a.child_count != b.child_count)) {
            status = EINVAL;
        }
    }
    struct dora_allocation storage = {};
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, a.child_count, sizeof(struct dora_vector))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &storage);
    }
    struct dora_vector* const groups = storage.address;
    if (status == DORA_OK && bytes != 0) {
        memset(groups, 0, bytes);
    }
    for (uint32_t node = 0; status == DORA_OK && node < a.child_count; ++node) {
        struct dora_vector group_left = {};
        struct dora_vector group_right = {};
        status = dora_vector_get_child(&retained_left, node, &group_left);
        if (status == DORA_OK && right != nullptr) {
            status = dora_vector_get_child(&retained_right, node, &group_right);
        }
        if (status == DORA_OK) {
            status = merge_group(pool, &group_left, right == nullptr ? nullptr : &group_right, &groups[node]);
        }
    }
    if (status == DORA_OK) {
        struct dora_allocation validity = {};
        const struct vector_input composition = {
            .field = &a,
            .row_count = 1,
            .allocations = &validity,
            .buffer_count = 1,
            .children = groups,
            .child_count = a.child_count
        };
        status = dora_vector_init_move_input(&published, pool, &composition, &validity, nullptr);
    }
    for (uint32_t node = 0; groups != nullptr && node < a.child_count; ++node) {
        dora_vector_release(&groups[node]);
    }
    dora_allocation_release(&storage);
    dora_schema_release(&left_schema);
    dora_schema_release(&right_schema);
    dora_vector_release(&retained_left);
    dora_vector_release(&retained_right);
    return vector_finish_publication(&published, out_result, status);
}
