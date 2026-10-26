// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "numeric/arithmetic.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

struct pruning_count {
    uint64_t value;
    bool is_known;
};

struct pruning_bounds {
    struct statistics_key lower;
    struct statistics_key upper;
};

struct pruning_order {
    int lower;
    int upper;
    bool has_no_nan;
};

struct pruning_predicate {
    uint32_t schema_node_index;
    enum dora_compare_op op;
    bool is_absolute;
};

static int read_count(const struct dora_vector* group, const char* name, struct pruning_count* out_count) {
    struct dora_vector child = {};
    int status = dora_vector_find_child(group, name, &child);
    if (status == ENOENT) {
        return DORA_OK;
    }
    int64_t value = 0;
    bool is_null = true;
    if (status == DORA_OK) {
        status = dora_vector_read_i64(&child, 0, &value, &is_null);
    }
    if (status == DORA_OK && !is_null && value >= 0) {
        *out_count = (struct pruning_count){.value = (uint64_t)value, .is_known = true};
    }
    return status;
}

static int read_bound(struct dora_pool pool,
                      const struct dora_vector* group,
                      const char* name,
                      struct dora_allocation* out_bound) {
    struct dora_vector child = {};
    const int status = dora_vector_find_child(group, name, &child);
    return status == ENOENT    ? DORA_OK
           : status != DORA_OK ? status
                               : dora_statistics_key_serialize_row(pool, &child, 0, out_bound);
}

// Signed integers, unsigned integers, floating-point types, Decimal
static int absolute_key(const struct statistics_key* key, struct statistics_key* out) {
    *out = *key;
    if (!((key->type >= DORA_TYPE_INT8 && key->type <= DORA_TYPE_FLOAT64) || key->type == DORA_TYPE_DECIMAL)) {
        return ENOTSUP;
    }
    switch (key->physical_type) {
// Returns DORA_OK or EOVERFLOW from the enclosing function.
#define ABSOLUTE(TAG, MEMBER)                                                                                          \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        return numeric_abs_checked_##MEMBER(key->scalar.MEMBER, &out->scalar.MEMBER) ? EOVERFLOW : DORA_OK;
// Returns DORA_OK from the enclosing function.
#define ABSOLUTE_FLOAT(TAG, MEMBER)                                                                                    \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        out->scalar.MEMBER = numeric_abs_##MEMBER(key->scalar.MEMBER);                                                 \
        return DORA_OK;
        ABSOLUTE(I8, i8)
        ABSOLUTE(I16, i16)
        ABSOLUTE(I32, i32)
        ABSOLUTE(I64, i64)
        ABSOLUTE(I128, i128)
        ABSOLUTE(I256, i256)
        ABSOLUTE(U8, u8)
        ABSOLUTE(U16, u16)
        ABSOLUTE(U32, u32)
        ABSOLUTE(U64, u64)
        ABSOLUTE(U128, u128)
        ABSOLUTE_FLOAT(F16, f16)
        ABSOLUTE_FLOAT(F32, f32)
        ABSOLUTE_FLOAT(F64, f64)
#undef ABSOLUTE_FLOAT
#undef ABSOLUTE
    default:
        return ENOTSUP;
    }
}

static int absolute_bounds(const struct statistics_key* lower,
                           const struct statistics_key* upper,
                           struct pruning_bounds* out_bounds) {
    struct statistics_key a = {};
    struct statistics_key b = {};
    int status = absolute_key(lower, &a);
    if (status == DORA_OK) {
        status = absolute_key(upper, &b);
    }
    int order = 0;
    if (status == DORA_OK) {
        status = dora_statistics_key_compare(&a, &b, &order);
    }
    if (status != DORA_OK) {
        return status;
    }
    out_bounds->lower = order < 0 ? a : b;
    out_bounds->upper = order > 0 ? a : b;
    struct statistics_key zero = *lower;
    zero.scalar = (union statistics_scalar_value){};
    int lower_sign = 0;
    int upper_sign = 0;
    status = dora_statistics_key_compare(lower, &zero, &lower_sign);
    if (status == DORA_OK) {
        status = dora_statistics_key_compare(upper, &zero, &upper_sign);
    }
    if (status == DORA_OK && lower_sign <= 0 && upper_sign >= 0) {
        out_bounds->lower = zero;
    }
    return status;
}

static enum dora_pruning_result
nan_result(enum dora_compare_op op, struct pruning_count rows, struct pruning_count nulls, struct pruning_count nans) {
    const bool none = nans.is_known && nans.value == 0;
    const bool all = rows.is_known && nulls.is_known && nans.is_known && nulls.value <= rows.value &&
                     nans.value == rows.value - nulls.value;
    switch (op) {
    case DORA_COMPARE_OP_GT:
        return DORA_PRUNING_RESULT_CANNOT_MATCH;
    case DORA_COMPARE_OP_LE:
        return DORA_PRUNING_RESULT_ALL_MATCH;
    case DORA_COMPARE_OP_EQ:
    case DORA_COMPARE_OP_GE:
        return all    ? DORA_PRUNING_RESULT_ALL_MATCH
               : none ? DORA_PRUNING_RESULT_CANNOT_MATCH
                      : DORA_PRUNING_RESULT_MAY_MATCH;
    case DORA_COMPARE_OP_NE:
    case DORA_COMPARE_OP_LT:
        return none  ? DORA_PRUNING_RESULT_ALL_MATCH
               : all ? DORA_PRUNING_RESULT_CANNOT_MATCH
                     : DORA_PRUNING_RESULT_MAY_MATCH;
    }
    return DORA_PRUNING_RESULT_MAY_MATCH;
}

static enum dora_pruning_result bounded_result(enum dora_compare_op op, struct pruning_order orders) {
    const int lower = orders.lower;
    const int upper = orders.upper;
    const bool no_nan = orders.has_no_nan;
    bool all = false;
    bool none = false;
    switch (op) {
    case DORA_COMPARE_OP_GT:
        all = lower > 0;
        none = upper <= 0 && no_nan;
        break;
    case DORA_COMPARE_OP_GE:
        all = lower >= 0;
        none = upper < 0 && no_nan;
        break;
    case DORA_COMPARE_OP_LT:
        all = upper < 0 && no_nan;
        none = lower >= 0;
        break;
    case DORA_COMPARE_OP_LE:
        all = upper <= 0 && no_nan;
        none = lower > 0;
        break;
    case DORA_COMPARE_OP_EQ:
        all = lower == 0 && upper == 0 && no_nan;
        none = lower > 0 || upper < 0;
        break;
    case DORA_COMPARE_OP_NE:
        all = lower > 0 || upper < 0;
        none = lower == 0 && upper == 0 && no_nan;
        break;
    }
    return all    ? DORA_PRUNING_RESULT_ALL_MATCH
           : none ? DORA_PRUNING_RESULT_CANNOT_MATCH
                  : DORA_PRUNING_RESULT_MAY_MATCH;
}

static int pruning_statistics(const struct dora_vector* segment,
                              const struct schema_walk_node* nodes,
                              uint32_t count,
                              struct pruning_predicate predicate,
                              struct dora_vector* out_statistics) {
    int status = dora_vector_get_statistics(segment, out_statistics);
    if (status != ENOTSUP) {
        return status;
    }
    struct vector_image* image = nullptr;
    status = dora_vector_resolve(segment, &image);
    if (status != DORA_OK || image->ownership.statistics.handle.slot_index == 0) {
        return status;
    }
    for (uint32_t cursor = predicate.schema_node_index; cursor != 0;) {
        const uint32_t parent = nodes[cursor].parent_node_index;
        if (parent >= cursor) {
            return DORA_OK;
        }
        struct dora_schema_field field = {};
        status = dora_schema_get_field(&nodes[parent].schema, &field);
        if (status != DORA_OK) {
            return status;
        }
        if (nodes[cursor].child_index >= field.child_count) {
            return DORA_OK;
        }
        switch (field.layout) {
        case DORA_LAYOUT_STRUCT:
        case DORA_LAYOUT_LIST:
        case DORA_LAYOUT_LARGE_LIST:
        case DORA_LAYOUT_LIST_VIEW:
        case DORA_LAYOUT_LARGE_LIST_VIEW:
        case DORA_LAYOUT_FIXED_LIST:
        case DORA_LAYOUT_DENSE_UNION:
        case DORA_LAYOUT_SPARSE_UNION:
        case DORA_LAYOUT_DICTIONARY:
        case DORA_LAYOUT_RUN_END:
            break;
        default:
            return DORA_OK;
        }
        cursor = parent;
    }
    uint32_t groups = 0;
    status = dora_vector_get_child_count(&image->ownership.statistics, &groups);
    if (status == DORA_OK && groups == count) {
        *out_statistics = dora_vector_borrow(&image->ownership.statistics);
    }
    return status;
}

static int prune_segment(struct dora_pool pool,
                         const struct dora_vector* segment,
                         struct pruning_predicate predicate,
                         const struct statistics_key* operand,
                         const struct schema_walk_node* nodes,
                         uint32_t count,
                         enum dora_pruning_result* out_result) {
    const uint32_t node = predicate.schema_node_index;
    const enum dora_compare_op op = predicate.op;
    *out_result = DORA_PRUNING_RESULT_MAY_MATCH;
    if (operand->is_null) {
        *out_result = DORA_PRUNING_RESULT_CANNOT_MATCH;
        return DORA_OK;
    }
    struct dora_vector statistics = {};
    int status = pruning_statistics(segment, nodes, count, predicate, &statistics);
    if (status == ENOTSUP || (status == DORA_OK && statistics.handle.slot_index == 0)) {
        return DORA_OK;
    }
    struct dora_vector group = {};
    if (status == DORA_OK) {
        status = dora_vector_get_child(&statistics, node, &group);
    }
    struct pruning_count rows = {};
    struct pruning_count nulls = {};
    struct pruning_count nans = {};
    if (status == DORA_OK) {
        status = read_count(&group, "row_count", &rows);
    }
    if (status == DORA_OK) {
        status = read_count(&group, "null_count", &nulls);
    }
    if (status == DORA_OK) {
        status = read_count(&group, "nan_count", &nans);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (rows.is_known && nulls.is_known && rows.value == nulls.value) {
        *out_result = DORA_PRUNING_RESULT_CANNOT_MATCH;
        return DORA_OK;
    }
    const bool floating = operand->type >= DORA_TYPE_FLOAT16 && operand->type <= DORA_TYPE_FLOAT64;
    if (floating && !dora_statistics_key_is_nan(operand) && rows.is_known && nulls.is_known && nans.is_known &&
        nulls.value <= rows.value && nans.value == rows.value - nulls.value) {
        switch (op) {
        case DORA_COMPARE_OP_EQ:
        case DORA_COMPARE_OP_LT:
        case DORA_COMPARE_OP_LE:
            *out_result = DORA_PRUNING_RESULT_CANNOT_MATCH;
            break;
        case DORA_COMPARE_OP_NE:
        case DORA_COMPARE_OP_GT:
        case DORA_COMPARE_OP_GE:
            *out_result = DORA_PRUNING_RESULT_ALL_MATCH;
            break;
        }
        return DORA_OK;
    }
    struct dora_allocation lower_storage = {};
    struct dora_allocation upper_storage = {};
    status = read_bound(pool, &group, "min", &lower_storage);
    if (status == DORA_OK) {
        status = read_bound(pool, &group, "max", &upper_storage);
    }
    const struct statistics_key* lower = lower_storage.address;
    const struct statistics_key* upper = upper_storage.address;
    if (status == DORA_OK && !predicate.is_absolute &&
        (operand->type == DORA_TYPE_UTF8 || operand->type == DORA_TYPE_BINARY)) {
        int low = 0, high = 0;
        const bool has_low =
            lower != nullptr && !lower->is_null && dora_statistics_key_compare(lower, operand, &low) == DORA_OK;
        const bool has_high =
            upper != nullptr && !upper->is_null && dora_statistics_key_compare(upper, operand, &high) == DORA_OK;
        if (has_low && low == 0) {
            struct pruning_count width = {};
            const int width_status = read_count(&group, "min_byte_width", &width);
            if (width_status == DORA_OK && width.is_known && width.value > operand->count) {
                low = 1;
            }
        }
        bool none = false, all = false;
        switch (op) {
        case DORA_COMPARE_OP_EQ:
            none = (has_low && low > 0) || (has_high && high < 0);
            all = has_low && has_high && low == 0 && high == 0;
            break;
        case DORA_COMPARE_OP_NE:
            none = has_low && has_high && low == 0 && high == 0;
            all = (has_low && low > 0) || (has_high && high < 0);
            break;
        case DORA_COMPARE_OP_LT:
            none = has_low && low >= 0;
            all = has_high && high < 0;
            break;
        case DORA_COMPARE_OP_LE:
            none = has_low && low > 0;
            all = has_high && high <= 0;
            break;
        case DORA_COMPARE_OP_GT:
            none = has_high && high <= 0;
            all = has_low && low > 0;
            break;
        case DORA_COMPARE_OP_GE:
            none = has_high && high < 0;
            all = has_low && low >= 0;
            break;
        }
        *out_result = none  ? DORA_PRUNING_RESULT_CANNOT_MATCH
                      : all ? DORA_PRUNING_RESULT_ALL_MATCH
                            : DORA_PRUNING_RESULT_MAY_MATCH;
        dora_allocation_release(&lower_storage);
        dora_allocation_release(&upper_storage);
        return DORA_OK;
    }
    struct pruning_bounds absolute = {};
    const bool bounds = lower != nullptr && upper != nullptr && !lower->is_null && !upper->is_null;
    const bool no_nan =
        !floating || (bounds && dora_statistics_key_is_nan(upper)) || (nans.is_known && nans.value == 0);
    bool usable = bounds;
    if (dora_statistics_key_is_nan(operand) && !(bounds && no_nan)) {
        *out_result = nan_result(op, rows, nulls, nans);
        usable = false;
    }
    if (status == DORA_OK && usable && predicate.is_absolute) {
        usable = absolute_bounds(lower, upper, &absolute) == DORA_OK;
        lower = &absolute.lower;
        upper = &absolute.upper;
    }
    if (status == DORA_OK && usable) {
        int lower_order = 0;
        int upper_order = 0;
        const int a = dora_statistics_key_compare(lower, operand, &lower_order);
        const int b = dora_statistics_key_compare(upper, operand, &upper_order);
        if (a == DORA_OK && b == DORA_OK) {
            *out_result = bounded_result(
                op,
                (struct pruning_order){.lower = lower_order, .upper = upper_order, .has_no_nan = no_nan});
        }
    }
    dora_allocation_release(&lower_storage);
    dora_allocation_release(&upper_storage);
    return status;
}

static int pruning_node(const struct dora_vector* segment,
                        const struct schema_walk_node* nodes,
                        uint32_t node,
                        uint32_t depth,
                        struct dora_vector* out_vector) {
    if (depth > STATISTICS_MAX_DEPTH) {
        return EINVAL;
    }
    if (node == 0) {
        *out_vector = dora_vector_borrow(segment);
        return DORA_OK;
    }
    if (nodes[node].parent_node_index >= node) {
        return EINVAL;
    }
    struct dora_vector parent = {};
    int status = pruning_node(segment, nodes, nodes[node].parent_node_index, depth + 1, &parent);
    if (status == DORA_OK) {
        status = dora_vector_get_physical_child(&parent, nodes[node].child_index, out_vector);
    }
    return status;
}

static int
pruning_schema_plain(struct dora_vector* in_out_vector, struct dora_schema_field* out_field, bool* out_ordered) {
    for (uint32_t depth = 0; depth <= STATISTICS_MAX_DEPTH; ++depth) {
        struct vector_image* image = nullptr;
        int status = dora_vector_resolve(in_out_vector, &image);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&image->schema, out_field);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (out_field->type.extension_name.byte_count != 0) {
            const struct dora_extension_definition* const definition = image->ownership.extension;
            if (definition == nullptr || (definition->flags & DORA_EXTENSION_FLAG_STORAGE_ORDER) == 0) {
                *out_ordered = false;
                return DORA_OK;
            }
        }
        if (out_field->type.id == DORA_TYPE_MAP) {
            *out_ordered = false;
            return DORA_OK;
        }
        if (out_field->layout != DORA_LAYOUT_DICTIONARY && out_field->layout != DORA_LAYOUT_RUN_END) {
            return DORA_OK;
        }
        struct dora_vector child = {};
        status =
            dora_vector_get_physical_child(in_out_vector, out_field->layout == DORA_LAYOUT_RUN_END ? 1 : 0, &child);
        if (status != DORA_OK) {
            return status;
        }
        *in_out_vector = child;
    }
    return EINVAL;
}

static int pruning_schema_matches(const struct dora_vector pair[2], uint32_t depth, bool* out_matches) {
    if (depth > STATISTICS_MAX_DEPTH) {
        return EINVAL;
    }
    struct dora_vector left = pair[0];
    struct dora_vector right = pair[1];
    struct dora_schema_field a = {};
    struct dora_schema_field b = {};
    int status = pruning_schema_plain(&left, &a, out_matches);
    if (status == DORA_OK && *out_matches) {
        status = pruning_schema_plain(&right, &b, out_matches);
    }
    if (status != DORA_OK || !*out_matches) {
        return status;
    }
    *out_matches = a.child_count == b.child_count && dora_type_is_equal(&a.type, &b.type, a.child_count);
    for (uint32_t child = 0; *out_matches && child < a.child_count; ++child) {
        struct dora_vector first = {};
        struct dora_vector second = {};
        status = dora_vector_get_physical_child(&left, child, &first);
        if (status == DORA_OK) {
            status = dora_vector_get_physical_child(&right, child, &second);
        }
        if (status == DORA_OK) {
            status = pruning_schema_matches((const struct dora_vector[]){first, second}, depth + 1, out_matches);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static int prune(const struct dora_vector* segments,
                 uint32_t segment_count,
                 struct pruning_predicate predicate,
                 const struct dora_vector* operand,
                 enum dora_pruning_result* out_results) {
    const uint32_t node = predicate.schema_node_index;
    const enum dora_compare_op op = predicate.op;
    struct dora_pool pool = {};
    if ((segment_count != 0 && (segments == nullptr || out_results == nullptr)) ||
        dora_vector_get_pool(operand, &pool) != DORA_OK || operand->rows.length != 1 || op > DORA_COMPARE_OP_GE) {
        return EINVAL;
    }
    struct dora_allocation key = {};
    int status = dora_statistics_key_serialize_row(pool, operand, 0, &key);
    struct dora_allocation results = {};
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, segment_count, &results);
    }
    for (uint32_t index = 0; status == DORA_OK && index < segment_count; ++index) {
        struct dora_schema schema = {};
        struct dora_allocation walk = {};
        uint32_t count = 0;
        status = dora_vector_retain_schema(&segments[index], &schema);
        if (status == DORA_OK) {
            status = dora_schema_walk(pool, &schema, &walk, &count);
        }
        if (status == DORA_OK && node >= count) {
            status = EINVAL;
        }
        enum dora_pruning_result result = DORA_PRUNING_RESULT_MAY_MATCH;
        struct dora_vector target = {};
        bool compatible = true;
        if (status == DORA_OK) {
            status = pruning_node(&segments[index], walk.address, node, 0, &target);
        }
        if (status == DORA_OK) {
            status = pruning_schema_matches((const struct dora_vector[]){target, *operand}, 0, &compatible);
        }
        if (status == DORA_OK && ((const struct statistics_key*)key.address)->is_null) {
            result = DORA_PRUNING_RESULT_CANNOT_MATCH;
        } else if (status == DORA_OK && compatible) {
            status = prune_segment(pool, &segments[index], predicate, key.address, walk.address, count, &result);
        }
        if (status == DORA_OK) {
            ((uint8_t*)results.address)[index] = result;
        }
        dora_allocation_release(&walk);
        dora_schema_release(&schema);
    }
    if (status == DORA_OK && segment_count != 0) {
        memcpy(out_results, results.address, segment_count);
    }
    dora_allocation_release(&results);
    dora_allocation_release(&key);
    return status;
}

//------------------------------------------------------------------------------
//
// Pruning
//
//------------------------------------------------------------------------------

int dora_statistics_prune(const struct dora_vector* segments,
                          uint32_t segment_count,
                          uint32_t schema_node_index,
                          enum dora_compare_op op,
                          const struct dora_vector* operand,
                          enum dora_pruning_result* out_results) {
    return prune(segments,
                 segment_count,
                 (struct pruning_predicate){.schema_node_index = schema_node_index, .op = op},
                 operand,
                 out_results);
}

int dora_statistics_prune_abs(const struct dora_vector* segments,
                              uint32_t segment_count,
                              uint32_t schema_node_index,
                              enum dora_compare_op op,
                              const struct dora_vector* operand,
                              enum dora_pruning_result* out_results) {
    return prune(segments,
                 segment_count,
                 (struct pruning_predicate){.schema_node_index = schema_node_index, .op = op, .is_absolute = true},
                 operand,
                 out_results);
}
