// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/statistics.h"

#include "builder/builder_internal.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "interop/arrow/arrow_internal.h"
#include "numeric/arithmetic.h"
#include "numeric/float.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "status/status_internal.h"
#include "types/array_count.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t ARROW_STATISTIC_CELL_COUNT = 2 * STATISTICS_KIND_COUNT;

struct statistic_name {
    const char* exact_key;
    const char* estimate_key;
};

static const struct statistic_name names[] = {
    {"ARROW:row_count:exact", "ARROW:row_count:approximate"},
    {"ARROW:null_count:exact", "ARROW:null_count:approximate"},
    {"DORA:nan_count:exact", "DORA:nan_count:approximate"},
    {"ARROW:min_value:exact", "ARROW:min_value:approximate"},
    {"ARROW:max_value:exact", "ARROW:max_value:approximate"},
    {"ARROW:distinct_count:exact", "ARROW:distinct_count:approximate"},
    {"DORA:min_byte_width:exact", "DORA:min_byte_width:approximate"},
    {"ARROW:max_byte_width:exact", "ARROW:max_byte_width:approximate"},
    {"ARROW:average_byte_width:exact", "ARROW:average_byte_width:approximate"}
};

struct statistic_budget {
    struct dora_arrow_limits limits;
    struct arrow_traversal_state traversal;
};

static int statistics_charge(struct statistic_budget* budget, uint64_t work, size_t bytes) {
    return budget == nullptr ? DORA_OK : arrow_traversal_charge(&budget->limits, &budget->traversal, work, bytes);
}

struct statistic_cell {
    struct dora_vector value;
    uint32_t member_index;
    bool is_present;
};

struct statistic_preorder {
    uint32_t* native_to_arrow;
    uint32_t* arrow_to_native;
    uint32_t arrow_node_count;
    bool has_record_root;
};

struct statistic_fields {
    struct dora_allocation storage;
    struct statistic_budget* budget;
    uint32_t field_count;
};

struct bound_dictionary {
    struct dora_builder builder;
    struct dora_vector source;
    struct dora_allocation indices_storage;
};

struct bound_context {
    struct dora_pool pool;
    struct statistic_budget* budget;
    struct dora_allocation dictionaries;
    uint32_t dictionary_count;
    bool allows_utf8_as_binary;
    struct dora_error* error;
};

enum bound_phase {
    BOUND_PREPARE,
    BOUND_APPEND,
};

static void fields_storage(struct statistic_fields* fields) {
    struct dora_schema_field* const data = fields->storage.address;
    for (uint32_t index = 0; index < fields->field_count; ++index) {
        if (data[index].type.id == DORA_TYPE_UTF8) {
            data[index].type = (struct dora_logical_type){.id = DORA_TYPE_BINARY};
        }
    }
}

static int field_cost(struct statistic_budget* budget, const struct dora_schema_field* field) {
    return budget == nullptr ? DORA_OK : dora_arrow_field_charge(&budget->limits, &budget->traversal, field);
}

static int fields_append(struct dora_pool pool, struct statistic_fields* fields, struct dora_schema_field field) {
    size_t bytes = 0;
    if (fields->field_count == UINT32_MAX || ckd_mul(&bytes, (size_t)fields->field_count + 1, sizeof(field))) {
        return EOVERFLOW;
    }
    int status = statistics_charge(fields->budget, 1, bytes > fields->storage.capacity_bytes ? bytes : 0);
    if (status == DORA_OK) {
        status = field_cost(fields->budget, &field);
    }
    if (status == DORA_OK && fields->budget != nullptr) {
        if (fields->budget->traversal.node_count >= fields->budget->limits.max_node_count ||
            field.child_count > fields->budget->limits.max_edge_count - fields->budget->traversal.edge_count) {
            status = ERANGE;
        } else {
            ++fields->budget->traversal.node_count;
            fields->budget->traversal.edge_count += field.child_count;
        }
    }
    if (status == DORA_OK) {
        status = dora_allocation_reserve(pool, bytes, fields->field_count * sizeof(field), &fields->storage);
    }
    if (status == DORA_OK) {
        ((struct dora_schema_field*)fields->storage.address)[fields->field_count++] = field;
    }
    return status;
}

static int fields_clone(struct dora_pool pool,
                        struct statistic_fields* fields,
                        const struct dora_schema* schema,
                        const char* name,
                        uint32_t depth) {
    if (fields->budget != nullptr && depth >= fields->budget->limits.max_depth) {
        return ERANGE;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (name != nullptr) {
        field.name = name;
    }
    if (status == DORA_OK) {
        status = fields_append(pool, fields, field);
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_physical_child(schema, index, &child);
        if (status == DORA_OK) {
            status = fields_clone(pool, fields, &child, nullptr, depth + 1);
        }
    }
    return status;
}

static int
node_count(const struct dora_schema* schema, uint32_t* in_out_count, struct statistic_budget* budget, uint32_t depth) {
    if (budget != nullptr && (depth >= budget->limits.max_depth || *in_out_count >= budget->limits.max_node_count)) {
        return ERANGE;
    }
    const int charged = statistics_charge(budget, 1, 0);
    if (charged != DORA_OK) {
        return charged;
    }
    if (*in_out_count == UINT32_MAX) {
        return EOVERFLOW;
    }
    ++*in_out_count;
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_physical_child(schema, index, &child);
        if (status == DORA_OK) {
            status = node_count(&child, in_out_count, budget, depth + 1);
        }
    }
    return status;
}

static int preorder_fill(const struct dora_schema* schema,
                         uint32_t nodes,
                         struct statistic_preorder* preorder,
                         uint32_t* in_out_node,
                         bool hidden) {
    if (*in_out_node >= nodes || preorder->native_to_arrow == nullptr || preorder->arrow_to_native == nullptr) {
        return EINVAL;
    }
    const uint32_t node = (*in_out_node)++;
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    const enum dora_layout layout = field.layout;
    if (layout == DORA_LAYOUT_DICTIONARY && field.child_count != 1) {
        return EINVAL;
    }
    if (node == 0) {
        preorder->has_record_root = field.type.id == DORA_TYPE_STRUCT && layout == DORA_LAYOUT_STRUCT;
    }
    if (hidden) {
        preorder->native_to_arrow[node] = UINT32_MAX;
    } else {
        if (preorder->arrow_node_count >= nodes) {
            return EINVAL;
        }
        const uint32_t column = preorder->arrow_node_count - (preorder->has_record_root && node != 0 ? 1U : 0U);
        if (column > INT32_MAX) {
            return EOVERFLOW;
        }
        preorder->native_to_arrow[node] = preorder->arrow_node_count;
        preorder->arrow_to_native[preorder->arrow_node_count++] = node;
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_physical_child(schema, index, &child);
        if (status == DORA_OK) {
            status = preorder_fill(&child, nodes, preorder, in_out_node, layout == DORA_LAYOUT_DICTIONARY);
        }
    }
    return status;
}

static int cells_allocate(struct dora_pool pool,
                          const struct dora_schema* target,
                          uint32_t nodes,
                          struct dora_allocation* out_storage,
                          struct statistic_preorder* out_preorder,
                          struct statistic_budget* budget) {
    size_t count = 0;
    size_t cell_bytes = 0;
    size_t map_bytes = 0;
    size_t bytes = 0;
    uint64_t work = 0;
    if (ckd_mul(&count, nodes, ARROW_STATISTIC_CELL_COUNT) ||
        ckd_mul(&cell_bytes, count, sizeof(struct statistic_cell)) || ckd_mul(&count, nodes, 2U) ||
        ckd_mul(&map_bytes, count, sizeof(uint32_t)) || ckd_add(&bytes, cell_bytes, map_bytes) ||
        ckd_mul(&work, nodes, 2U)) {
        return EOVERFLOW;
    }
    int status = statistics_charge(budget, work, bytes);
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, out_storage);
    }
    if (status == DORA_OK && bytes != 0) {
        memset(out_storage->address, 0, bytes);
        out_preorder->native_to_arrow = (uint32_t*)((uint8_t*)out_storage->address + cell_bytes);
        out_preorder->arrow_to_native = out_preorder->native_to_arrow + nodes;
    }
    uint32_t node = 0;
    if (status == DORA_OK) {
        status = preorder_fill(target, nodes, out_preorder, &node, false);
    }
    if (status == DORA_OK && node != nodes) {
        status = EINVAL;
    }
    return status;
}

static int schema_same(const struct dora_schema* left, const struct dora_schema* right, bool* out_same) {
    struct dora_schema_field a = {};
    struct dora_schema_field b = {};
    int status = dora_schema_get_field(left, &a);
    if (status == DORA_OK) {
        status = dora_schema_get_field(right, &b);
    }
    if (status == DORA_OK) {
        b.name = a.name;
        status = dora_schema_field_matches(left, &b, out_same);
    }
    for (uint32_t index = 0; status == DORA_OK && *out_same && index < a.child_count; ++index) {
        struct dora_schema x = {};
        struct dora_schema y = {};
        status = dora_schema_get_physical_child(left, index, &x);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(right, index, &y);
        }
        if (status == DORA_OK) {
            status = dora_schema_declarations_equal(&x, &y, out_same);
        }
    }
    return status;
}

static int export_cells(struct dora_pool pool,
                        const struct dora_vector* statistics,
                        uint32_t nodes,
                        const struct statistic_preorder* preorder,
                        struct statistic_cell* cells,
                        struct dora_schema* members,
                        uint32_t* out_member_count) {
    for (uint32_t node = 0; node < nodes; ++node) {
        if (preorder->native_to_arrow[node] == UINT32_MAX) {
            continue;
        }
        struct dora_vector group = {};
        int status = dora_vector_get_child(statistics, node, &group);
        if (status != DORA_OK) {
            return status;
        }
        for (uint32_t entry = 0; entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            const uint32_t kind = entry % STATISTICS_KIND_COUNT;
            const bool estimate = entry >= STATISTICS_KIND_COUNT;
            struct dora_vector value = {};
            status = dora_vector_find_child(&group,
                                            estimate ? dora_statistics_descriptors[kind].estimate_name
                                                     : dora_statistics_descriptors[kind].name,
                                            &value);
            if (status == ENOENT) {
                continue;
            }
            bool is_null = false;
            if (status == DORA_OK) {
                status = dora_vector_is_null(&value, 0, &is_null);
            }
            if (status != DORA_OK) {
                return status;
            }
            if (is_null) {
                continue;
            }
            uint32_t destination = entry;
            if (!estimate && (kind == 3 || kind == 4)) {
                struct dora_vector exact = {};
                bool is_exact = false;
                status = dora_vector_find_child(&group, kind == 3 ? "is_min_exact" : "is_max_exact", &exact);
                if (status == DORA_OK) {
                    status = dora_vector_read_bool(&exact, 0, &is_exact, &is_null);
                }
                if (status == DORA_OK && (is_null || !is_exact)) {
                    destination += STATISTICS_KIND_COUNT;
                }
            }
            if (status != DORA_OK) {
                return status;
            }
            struct statistic_cell* const cell = &cells[node * ARROW_STATISTIC_CELL_COUNT + destination];
            if (cell->is_present) {
                continue;
            }
            struct dora_vector decoded = {};
            uint32_t children = 0;
            bool transport = false;
            status = dora_vector_get_child_count(&value, &children);
            if (status == DORA_OK) {
                status = dora_vector_copy_requires_transport(&value, &transport);
            }
            if (status == DORA_OK) {
                status = children != 0 || transport ? dora_vector_decode_error(&value, pool, &decoded, nullptr)
                                                    : dora_vector_retain(&value, &decoded);
            }
            struct dora_schema schema = {};
            if (status == DORA_OK) {
                status = dora_vector_retain_schema(&decoded, &schema);
            }
            uint32_t member = 0;
            for (; status == DORA_OK && member < *out_member_count; ++member) {
                bool same = false;
                status = schema_same(&schema, &members[member], &same);
                if (same) {
                    break;
                }
            }
            if (status == DORA_OK && member == *out_member_count) {
                if (member == UNION_MEMBER_CAPACITY) {
                    status = ERANGE;
                } else {
                    members[member] = schema;
                    schema = (struct dora_schema){};
                    ++*out_member_count;
                }
            }
            dora_schema_release(&schema);
            if (status == DORA_OK) {
                *cell = (struct statistic_cell){.value = decoded, .member_index = member, .is_present = true};
                decoded = (struct dora_vector){};
            }
            dora_vector_release(&decoded);
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    return DORA_OK;
}

static int scalar_integer(const struct dora_vector* source,
                          enum dora_physical_type physical,
                          int64_t* out_signed,
                          uint64_t* out_unsigned,
                          bool* out_is_unsigned) {
    bool is_null = false;
    *out_is_unsigned = false;
    switch (physical) {
    case DORA_PHYSICAL_TYPE_I8: {
        int8_t value = 0;
        const int status = dora_vector_read_i8(source, 0, &value, &is_null);
        *out_signed = (int)value;
        return status;
    }
    case DORA_PHYSICAL_TYPE_I16: {
        int16_t value = 0;
        const int status = dora_vector_read_i16(source, 0, &value, &is_null);
        *out_signed = value;
        return status;
    }
    case DORA_PHYSICAL_TYPE_I32: {
        int32_t value = 0;
        const int status = dora_vector_read_i32(source, 0, &value, &is_null);
        *out_signed = value;
        return status;
    }
    case DORA_PHYSICAL_TYPE_I64:
        return dora_vector_read_i64(source, 0, out_signed, &is_null);
    case DORA_PHYSICAL_TYPE_U8: {
        uint8_t value = 0;
        const int status = dora_vector_read_u8(source, 0, &value, &is_null);
        *out_unsigned = value;
        *out_is_unsigned = true;
        return status;
    }
    case DORA_PHYSICAL_TYPE_U16: {
        uint16_t value = 0;
        const int status = dora_vector_read_u16(source, 0, &value, &is_null);
        *out_unsigned = value;
        *out_is_unsigned = true;
        return status;
    }
    case DORA_PHYSICAL_TYPE_U32: {
        uint32_t value = 0;
        const int status = dora_vector_read_u32(source, 0, &value, &is_null);
        *out_unsigned = value;
        *out_is_unsigned = true;
        return status;
    }
    case DORA_PHYSICAL_TYPE_U64:
        *out_is_unsigned = true;
        return dora_vector_read_u64(source, 0, out_unsigned, &is_null);
    default:
        return EINVAL;
    }
}

static bool type_is_integer(enum dora_type type) {
    return type >= DORA_TYPE_INT8 && type <= DORA_TYPE_UINT128;
}

static int bound_compatible(const struct dora_schema* target,
                            const struct dora_schema* source,
                            bool byte_storage,
                            struct statistic_budget* budget,
                            uint32_t depth) {
    if (budget != nullptr && depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    struct dora_schema_field a = {};
    struct dora_schema_field b = {};
    int status = statistics_charge(budget, 1, 0);
    if (status == DORA_OK) {
        status = dora_schema_get_field(target, &a);
    }
    if (status == DORA_OK) {
        status = dora_schema_get_field(source, &b);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (a.layout == DORA_LAYOUT_DICTIONARY || a.layout == DORA_LAYOUT_RUN_END) {
        struct dora_schema value = {};
        status = dora_schema_get_physical_child(target, a.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &value);
        return status == DORA_OK ? bound_compatible(&value, source, byte_storage, budget, depth + 1) : status;
    }
    if (b.layout == DORA_LAYOUT_DICTIONARY || b.layout == DORA_LAYOUT_RUN_END) {
        struct dora_schema value = {};
        status = dora_schema_get_physical_child(source, b.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &value);
        return status == DORA_OK ? bound_compatible(target, &value, byte_storage, budget, depth + 1) : status;
    }
    if (byte_storage && a.type.id == DORA_TYPE_BINARY && b.type.id == DORA_TYPE_UTF8 && a.child_count == 0 &&
        b.child_count == 0 && a.type.extension_name.byte_count == 0) {
        return DORA_OK;
    }
    if (!bytes_are_equal(a.type.extension_name, b.type.extension_name) ||
        !bytes_are_equal(a.type.extension_metadata, b.type.extension_metadata)) {
        return EINVAL;
    }
    if (a.type.id != b.type.id) {
        if (a.type.extension_name.byte_count != 0) {
            return EINVAL;
        }
        if (type_is_integer(a.type.id) && type_is_integer(b.type.id)) {
            return DORA_OK;
        }
        if (a.type.id >= DORA_TYPE_FLOAT16 && a.type.id <= DORA_TYPE_FLOAT64 && b.type.id >= DORA_TYPE_FLOAT16 &&
            b.type.id <= DORA_TYPE_FLOAT64) {
            return DORA_OK;
        }
        if ((a.type.id == DORA_TYPE_DATE32 || a.type.id == DORA_TYPE_DATE64 || a.type.id == DORA_TYPE_TIME ||
             a.type.id == DORA_TYPE_TIMESTAMP || a.type.id == DORA_TYPE_DURATION ||
             a.type.id == DORA_TYPE_INTERVAL_YEAR_MONTH) &&
            type_is_integer(b.type.id)) {
            return DORA_OK;
        }
        return EINVAL;
    }
    if (a.child_count != b.child_count) {
        return EINVAL;
    }
    bool same = true;
    switch (a.type.id) {
    case DORA_TYPE_FIXED_BINARY:
        same = a.type.parameters.byte_width == b.type.parameters.byte_width;
        break;
    case DORA_TYPE_FIXED_LIST:
        same = a.type.parameters.list_element_count == b.type.parameters.list_element_count;
        break;
    case DORA_TYPE_TIME:
    case DORA_TYPE_DURATION:
        same = a.type.parameters.unit == b.type.parameters.unit;
        break;
    case DORA_TYPE_TIMESTAMP:
        same = a.type.parameters.timestamp.unit == b.type.parameters.timestamp.unit &&
               strcmp(a.type.parameters.timestamp.timezone == nullptr ? "" : a.type.parameters.timestamp.timezone,
                      b.type.parameters.timestamp.timezone == nullptr ? "" : b.type.parameters.timestamp.timezone) == 0;
        break;
    case DORA_TYPE_DECIMAL:
        same = a.type.parameters.decimal.scale == b.type.parameters.decimal.scale;
        break;
    case DORA_TYPE_UNION:
        same = a.child_count == 0 ||
               memcmp(a.type.parameters.type_codes, b.type.parameters.type_codes, a.child_count) == 0;
        break;
    default:
        break;
    }
    if (!same) {
        return EINVAL;
    }
    for (uint32_t index = 0; status == DORA_OK && index < a.child_count; ++index) {
        struct dora_schema x = {};
        struct dora_schema y = {};
        status = dora_schema_get_physical_child(target, index, &x);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(source, index, &y);
        }
        if (status == DORA_OK) {
            status = bound_compatible(&x, &y, byte_storage, budget, depth + 1);
        }
    }
    return status;
}

static int append_widened(struct dora_builder* builder,
                          const struct dora_schema_field* target,
                          const struct dora_vector* source,
                          const struct dora_vector_view* view,
                          struct statistic_budget* budget,
                          struct dora_error* out_error) {
    if (target->type.extension_name.byte_count != 0) {
        return EINVAL;
    }
    if (target->type.id == DORA_TYPE_FLOAT16 || target->type.id == DORA_TYPE_FLOAT32 ||
        target->type.id == DORA_TYPE_FLOAT64) {
        bool is_null = false;
        double value = 0;
        int status;
        if (view->physical_type == DORA_PHYSICAL_TYPE_F64) {
            status = dora_vector_read_f64(source, 0, &value, &is_null);
        } else if (view->physical_type == DORA_PHYSICAL_TYPE_F32) {
            float single = 0;
            status = dora_vector_read_f32(source, 0, &single, &is_null);
            value = single;
        } else if (view->physical_type == DORA_PHYSICAL_TYPE_F16) {
            struct dora_f16_bits half = {};
            status = dora_vector_read_f16(source, 0, &half, &is_null);
            value = numeric_half_widen(half);
        } else {
            return EINVAL;
        }
        if (status != DORA_OK) {
            return status;
        }
        if (isnan(value)) {
            return EINVAL;
        }
        if (target->type.id == DORA_TYPE_FLOAT64) {
            status = statistics_charge(budget, 0, sizeof(value));
            return status == DORA_OK
                       ? dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_F64, &value, out_error)
                       : status;
        }
        if (isfinite(value) && (value > FLT_MAX || value < -FLT_MAX)) {
            return ERANGE;
        }
        const float single = (float)value;
        if ((double)single != value) {
            return ERANGE;
        }
        if (target->type.id == DORA_TYPE_FLOAT32) {
            status = statistics_charge(budget, 0, sizeof(single));
            return status == DORA_OK
                       ? dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_F32, &single, out_error)
                       : status;
        }
        const struct dora_f16_bits half = numeric_half_narrow(single);
        if (numeric_half_widen(half) != single) {
            return ERANGE;
        }
        status = statistics_charge(budget, 0, sizeof(half));
        return status == DORA_OK ? dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_F16, &half, out_error)
                                 : status;
    }
    struct dora_i256 value = {};
    bool is_null = false;
    int status;
    if (view->physical_type == DORA_PHYSICAL_TYPE_I256) {
        status = dora_vector_read_i256(source, 0, &value, &is_null);
    } else if (view->physical_type == DORA_PHYSICAL_TYPE_I128) {
        struct dora_i128 wide = {};
        status = dora_vector_read_i128(source, 0, &wide, &is_null);
        const uint64_t sign = (wide.limbs[1] >> 63U) != 0 ? UINT64_MAX : 0;
        value = (struct dora_i256){.limbs = {wide.limbs[0], wide.limbs[1], sign, sign}};
    } else if (view->physical_type == DORA_PHYSICAL_TYPE_U128) {
        struct dora_u128 wide = {};
        status = dora_vector_read_u128(source, 0, &wide, &is_null);
        value = (struct dora_i256){.limbs = {wide.limbs[0], wide.limbs[1], 0, 0}};
    } else {
        int64_t signed_value = 0;
        uint64_t unsigned_value = 0;
        bool is_unsigned = false;
        status = scalar_integer(source, view->physical_type, &signed_value, &unsigned_value, &is_unsigned);
        const uint64_t sign = !is_unsigned && signed_value < 0 ? UINT64_MAX : 0;
        value = (struct dora_i256){.limbs = {is_unsigned ? unsigned_value : (uint64_t)signed_value, sign, sign, sign}};
    }
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    if (status == DORA_OK) {
        status = type_physical(&target->type, &physical);
    }
    if (target->type.id == DORA_TYPE_DECIMAL) {
        struct dora_logical_type original = {};
        if (status == DORA_OK) {
            status = dora_vector_get_type(source, &original);
        }
        if (status == DORA_OK && (original.id != DORA_TYPE_DECIMAL ||
                                  original.parameters.decimal.scale != target->type.parameters.decimal.scale)) {
            status = EINVAL;
        }
    }

    union statistic_integer {
        int8_t i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
        struct dora_i128 i128;
        struct dora_i256 i256;
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        struct dora_u128 u128;
    } converted = {};

    if (status == DORA_OK) {
        status = dora_numeric_store_integer(value, physical, &converted);
    }
    const size_t width = physical_width(physical);
    if (status == DORA_OK) {
        status = statistics_charge(budget, 0, width);
    }
    if (status != DORA_OK) {
        return status;
    }
    return dora_builder_append_scalar_error(builder, physical, &converted, out_error);
}

static int
append_index(enum dora_type type, struct dora_builder* builder, uint64_t value, struct dora_error* out_error) {
    switch (type) {
    case DORA_TYPE_INT8: {
        if (value > INT8_MAX) {
            return ERANGE;
        }
        const int8_t index = (int8_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_I8, &index, out_error);
    }
    case DORA_TYPE_INT16: {
        if (value > INT16_MAX) {
            return ERANGE;
        }
        const int16_t index = (int16_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_I16, &index, out_error);
    }
    case DORA_TYPE_INT32: {
        if (value > INT32_MAX) {
            return ERANGE;
        }
        const int32_t index = (int32_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_I32, &index, out_error);
    }
    case DORA_TYPE_INT64: {
        if (value > INT64_MAX) {
            return ERANGE;
        }
        const int64_t index = (int64_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_I64, &index, out_error);
    }
    case DORA_TYPE_UINT8: {
        if (value > UINT8_MAX) {
            return ERANGE;
        }
        const uint8_t index = (uint8_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_U8, &index, out_error);
    }
    case DORA_TYPE_UINT16: {
        if (value > UINT16_MAX) {
            return ERANGE;
        }
        const uint16_t index = (uint16_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_U16, &index, out_error);
    }
    case DORA_TYPE_UINT32: {
        if (value > UINT32_MAX) {
            return ERANGE;
        }
        const uint32_t index = (uint32_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_U32, &index, out_error);
    }
    case DORA_TYPE_UINT64: {
        const uint64_t index = (uint64_t)value;
        return dora_builder_append_scalar_error(builder, DORA_PHYSICAL_TYPE_U64, &index, out_error);
    }
    default:
        return EINVAL;
    }
}

static int bound_index(const struct dora_vector_view* view, uint64_t row, uint64_t* out_index) {
    union bound_index_value {
        int8_t i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
    } value = {};

    const uint64_t physical = dora_vector_view_get_physical_row(view, row);
    size_t offset = 0;
    if (view->byte_width == 0 || view->byte_width > sizeof(value) || view->values.data == nullptr) {
        return EINVAL;
    }
    if (ckd_mul(&offset, physical, view->byte_width)) {
        return EOVERFLOW;
    }
    if (offset > view->values.byte_count || view->byte_width > view->values.byte_count - offset) {
        return EINVAL;
    }
    memcpy(&value, (const uint8_t*)view->values.data + offset, view->byte_width);
    switch (view->physical_type) {
    case DORA_PHYSICAL_TYPE_I8:
        if (value.i8 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i8;
        break;
    case DORA_PHYSICAL_TYPE_I16:
        if (value.i16 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i16;
        break;
    case DORA_PHYSICAL_TYPE_I32:
        if (value.i32 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i32;
        break;
    case DORA_PHYSICAL_TYPE_I64:
        if (value.i64 < 0) {
            return EINVAL;
        }
        *out_index = (uint64_t)value.i64;
        break;
    case DORA_PHYSICAL_TYPE_U8:
        *out_index = value.u8;
        break;
    case DORA_PHYSICAL_TYPE_U16:
        *out_index = value.u16;
        break;
    case DORA_PHYSICAL_TYPE_U32:
        *out_index = value.u32;
        break;
    case DORA_PHYSICAL_TYPE_U64:
        *out_index = value.u64;
        break;
    default:
        return EINVAL;
    }
    return DORA_OK;
}

static int conversion_row_cost(struct statistic_budget* budget,
                               const struct dora_vector* source,
                               uint64_t row,
                               bool copy,
                               uint32_t depth) {
    if (depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    int status = statistics_charge(budget, 1, 0);
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = dora_vector_get_view(source, &view);
    }
    if (status != DORA_OK || row >= view.length) {
        return status == DORA_OK ? EINVAL : status;
    }
    if (dora_vector_view_is_null(&view, row)) {
        return DORA_OK;
    }
    if (view.layout == DORA_LAYOUT_DICTIONARY || view.layout == DORA_LAYOUT_RUN_END ||
        view.layout == DORA_LAYOUT_SPARSE_UNION || view.layout == DORA_LAYOUT_DENSE_UNION) {
        uint32_t member = 0;
        uint64_t index = 0;
        uint32_t width = view.byte_width;
        if (view.layout == DORA_LAYOUT_DICTIONARY) {
            status = bound_index(&view, row, &index);
        } else if (view.layout == DORA_LAYOUT_RUN_END) {
            member = 1;
            index = dora_vector_view_find_run(&view, row);
        } else {
            member = dora_vector_view_get_member(&view, row, &index);
            width = view.layout == DORA_LAYOUT_DENSE_UNION ? sizeof(int8_t) + sizeof(int32_t) : sizeof(int8_t);
        }
        if (status == DORA_OK) {
            status = statistics_charge(budget, width, 0);
        }
        struct dora_vector child = {};
        if (status == DORA_OK) {
            status = dora_vector_get_physical_child(source, member, &child);
        }
        if (status == DORA_OK) {
            status = conversion_row_cost(budget, &child, index, copy, depth + 1);
        }
        return status;
    }
    if (view.type == DORA_TYPE_BINARY || view.type == DORA_TYPE_UTF8 || view.type == DORA_TYPE_FIXED_BINARY) {
        const struct dora_bytes bytes = dora_vector_view_get_bytes(&view, row);
        return statistics_charge(budget, bytes.byte_count, copy ? bytes.byte_count : 0);
    }
    if (view.layout == DORA_LAYOUT_FIXED || view.layout == DORA_LAYOUT_BITS) {
        const uint32_t width = view.layout == DORA_LAYOUT_BITS ? sizeof(bool) : view.byte_width;
        return statistics_charge(budget, width, copy ? width : 0);
    }
    const bool list = view.type == DORA_TYPE_LIST || view.type == DORA_TYPE_FIXED_LIST || view.type == DORA_TYPE_MAP;
    const struct dora_range rows =
        list ? dora_vector_view_get_elements(&view, row) : (struct dora_range){.offset = row, .length = 1};
    uint32_t children = 0;
    if (view.type == DORA_TYPE_MAP) {
        children = 1;
    } else {
        status = dora_vector_get_child_count(source, &children);
    }
    for (uint32_t index = 0; status == DORA_OK && index < children; ++index) {
        struct dora_vector child = {};
        status = view.layout == DORA_LAYOUT_STRUCT ? dora_vector_get_child(source, index, &child)
                                                   : dora_vector_get_physical_child(source, index, &child);
        for (uint64_t value = 0; status == DORA_OK && value < rows.length; ++value) {
            uint64_t child_row = 0;
            if (ckd_add(&child_row, rows.offset, value)) {
                status = EOVERFLOW;
            }
            if (status == DORA_OK) {
                status = conversion_row_cost(budget, &child, child_row, copy, depth + 1);
            }
        }
    }
    return status;
}

static int
conversion_cost(struct statistic_budget* budget, const struct dora_vector* source, bool copy, uint32_t depth) {
    if (budget == nullptr) {
        return DORA_OK;
    }
    for (uint64_t row = 0; row < source->rows.length; ++row) {
        const int status = conversion_row_cost(budget, source, row, copy, depth);
        if (status != DORA_OK) {
            return status;
        }
    }
    return DORA_OK;
}

static void bound_release(struct bound_context* context) {
    struct bound_dictionary* const dictionaries = context->dictionaries.address;
    for (uint32_t index = 0; index < context->dictionary_count; ++index) {
        dora_allocation_release(&dictionaries[index].indices_storage);
    }
    dora_allocation_release(&context->dictionaries);
}

static int bound_dictionary_allocate(struct bound_context* context,
                                     uint64_t rows,
                                     struct dora_allocation* indices,
                                     struct dora_allocation* representatives) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, rows, sizeof(uint64_t))) {
        return EOVERFLOW;
    }
    int status = statistics_charge(context->budget, rows, bytes);
    if (status == DORA_OK) {
        status = dora_pool_allocate(context->pool, bytes, indices);
    }
    if (status == DORA_OK) {
        status = statistics_charge(context->budget, 0, bytes);
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(context->pool, bytes, representatives);
    }
    return status;
}

static int bound_dictionary_find(struct bound_context* context,
                                 const struct dora_vector* source,
                                 const uint64_t* representatives,
                                 uint64_t count,
                                 const struct dora_vector* value,
                                 uint64_t* out_index) {
    *out_index = count;
    bool duplicate = false;
    int status = DORA_OK;
    for (uint64_t previous = 0; status == DORA_OK && !duplicate && previous < count; ++previous) {
        status = statistics_charge(context->budget, 1, 0);
        if (status != DORA_OK) {
            break;
        }
        struct dora_vector earlier = {};
        const uint64_t earlier_row = representatives[previous];
        status = dora_vector_slice(source, (struct dora_range){.offset = earlier_row, .length = 1}, &earlier);
        if (status == DORA_OK) {
            status = conversion_cost(context->budget, &earlier, false, 0);
        }
        if (status == DORA_OK) {
            status = conversion_cost(context->budget, value, false, 0);
        }
        if (status == DORA_OK) {
            status = dora_vector_is_equal(value, &earlier, &duplicate);
        }
        if (duplicate) {
            *out_index = previous;
        }
    }
    return status;
}

static int bound_dictionary_register(struct bound_context* context,
                                     const struct dora_builder* builder,
                                     const struct dora_vector* source,
                                     struct dora_allocation* indices) {
    size_t bytes = 0;
    if (context->dictionary_count == UINT32_MAX ||
        ckd_mul(&bytes, (size_t)context->dictionary_count + 1, sizeof(struct bound_dictionary))) {
        return EOVERFLOW;
    }
    int status = statistics_charge(context->budget, 1, bytes > context->dictionaries.capacity_bytes ? bytes : 0);
    if (status == DORA_OK) {
        status = dora_allocation_reserve(context->pool,
                                         bytes,
                                         context->dictionary_count * sizeof(struct bound_dictionary),
                                         &context->dictionaries);
    }
    if (status == DORA_OK) {
        ((struct bound_dictionary*)context->dictionaries.address)[context->dictionary_count++] =
            (struct bound_dictionary){.builder = *builder, .source = *source, .indices_storage = *indices};
        *indices = (struct dora_allocation){};
    }
    return status;
}

static int bound_convert(struct bound_context* context,
                         struct dora_builder* builder,
                         const struct dora_schema* schema,
                         const struct dora_vector* source,
                         enum bound_phase phase) {
    if (phase == BOUND_PREPARE) {
        struct dora_schema_field field = {};
        int status = statistics_charge(context->budget, 1, 0);
        if (status == DORA_OK) {
            status = dora_schema_get_field(schema, &field);
        }
        if (status != DORA_OK) {
            return status;
        }
        if (field.layout == DORA_LAYOUT_DICTIONARY) {
            struct dora_schema value_schema = {};
            status = dora_schema_get_physical_child(schema, 0, &value_schema);
            struct dora_builder values = {};
            if (status == DORA_OK) {
                status = dora_builder_init_schema_error(&values, context->pool, &value_schema, nullptr, context->error);
            }
            if (status == DORA_OK) {
                status = bound_convert(context, &values, &value_schema, source, BOUND_PREPARE);
            }
            struct dora_allocation indices = {};
            struct dora_allocation representatives = {};
            if (status == DORA_OK) {
                status = bound_dictionary_allocate(context, source->rows.length, &indices, &representatives);
            }
            uint64_t count = 0;
            for (uint64_t row = 0; status == DORA_OK && row < source->rows.length; ++row) {
                struct dora_vector value = {};
                bool is_null = false;
                status = dora_vector_is_null(source, row, &is_null);
                ((uint64_t*)indices.address)[row] = is_null ? UINT64_MAX : count;
                if (status == DORA_OK && !is_null) {
                    status = dora_vector_slice(source, (struct dora_range){.offset = row, .length = 1}, &value);
                    if (status == DORA_OK) {
                        status = bound_dictionary_find(context,
                                                       source,
                                                       representatives.address,
                                                       count,
                                                       &value,
                                                       &((uint64_t*)indices.address)[row]);
                    }
                    if (status == DORA_OK && ((const uint64_t*)indices.address)[row] == count) {
                        status = bound_convert(context, &values, &value_schema, &value, BOUND_APPEND);
                        if (status == DORA_OK) {
                            ((uint64_t*)representatives.address)[count++] = row;
                        }
                    }
                }
            }
            dora_allocation_release(&representatives);
            struct dora_vector dictionary = {};
            if (status == DORA_OK) {
                status = dora_builder_finish_error(&values, &dictionary, context->error);
            }
            if (status == DORA_OK) {
                status = dora_builder_set_dictionary_error(builder, &dictionary, context->error);
            }
            dora_vector_release(&dictionary);
            dora_builder_release(&values);
            if (status == DORA_OK) {
                status = bound_dictionary_register(context, builder, source, &indices);
            }
            dora_allocation_release(&indices);
        } else if (field.layout != DORA_LAYOUT_RUN_END) {
            uint32_t children = 0;
            status = dora_schema_get_child_count(schema, &children);
            for (uint32_t index = 0; status == DORA_OK && index < children; ++index) {
                struct dora_builder child = {};
                struct dora_vector values = {};
                struct dora_schema child_schema = {};
                status = dora_schema_get_child(schema, index, &child_schema);
                if (status == DORA_OK) {
                    status = dora_builder_get_child(builder, index, &child);
                }
                if (status == DORA_OK) {
                    status = dora_vector_get_child(source, index, &values);
                }
                if (status == DORA_OK) {
                    status = bound_convert(context, &child, &child_schema, &values, BOUND_PREPARE);
                }
            }
        }
        return status;
    }
    const struct dora_pool pool = context->pool;
    bool is_null = false;
    int status = statistics_charge(context->budget, 1, 0);
    if (status == DORA_OK) {
        status = dora_vector_is_null(source, 0, &is_null);
    }
    if (status != DORA_OK) {
        return status;
    }
    if (is_null) {
        return dora_builder_append_nulls_error(builder, 1, context->error);
    }
    struct dora_schema_field field = {};
    status = dora_schema_get_field(schema, &field);
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = dora_vector_get_view(source, &view);
    }
    if (status == DORA_OK && field.layout == DORA_LAYOUT_DICTIONARY) {
        const struct bound_dictionary* const dictionaries = context->dictionaries.address;
        const struct bound_dictionary* found = nullptr;
        for (uint32_t index = 0; status == DORA_OK && index < context->dictionary_count; ++index) {
            status = statistics_charge(context->budget, 1, 0);
            if (status == DORA_OK && dictionaries[index].builder.node_index == builder->node_index &&
                dora_handle_is_same(dictionaries[index].builder.handle, builder->handle)) {
                found = &dictionaries[index];
                break;
            }
        }
        if (status == DORA_OK && (found == nullptr || !dora_handle_is_same(found->source.handle, source->handle) ||
                                  source->rows.offset < found->source.rows.offset ||
                                  source->rows.offset - found->source.rows.offset >= found->source.rows.length)) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            const uint64_t index =
                ((const uint64_t*)found->indices_storage.address)[source->rows.offset - found->source.rows.offset];
            status = append_index(field.index_type, builder, index, context->error);
        }
    } else if (status == DORA_OK && field.layout == DORA_LAYOUT_RUN_END) {
        struct dora_schema value_schema = {};
        status = dora_schema_get_physical_child(schema, 1, &value_schema);
        struct dora_builder values = {};
        struct dora_vector value = {};
        if (status == DORA_OK) {
            status = dora_builder_init_schema_error(&values, pool, &value_schema, nullptr, context->error);
        }
        if (status == DORA_OK) {
            status = bound_convert(context, &values, &value_schema, source, BOUND_PREPARE);
        }
        if (status == DORA_OK) {
            status = bound_convert(context, &values, &value_schema, source, BOUND_APPEND);
        }
        if (status == DORA_OK) {
            status = dora_builder_finish_error(&values, &value, context->error);
        }
        if (status == DORA_OK) {
            status = dora_builder_append_vector_error(builder, &value, context->error);
        }
        dora_vector_release(&value);
        dora_builder_release(&values);
    } else if (status == DORA_OK && (field.type.id == DORA_TYPE_STRUCT || field.type.id == DORA_TYPE_LIST ||
                                     field.type.id == DORA_TYPE_FIXED_LIST || field.type.id == DORA_TYPE_UNION)) {
        if (view.type != field.type.id) {
            status = EINVAL;
        }
        uint32_t source_children = 0;
        if (status == DORA_OK) {
            status = dora_vector_get_child_count(source, &source_children);
        }
        if (status == DORA_OK && source_children != field.child_count) {
            status = EINVAL;
        }
        uint64_t destination_offset = 0;
        uint32_t selected = 0;
        uint64_t selected_row = 0;
        struct dora_range elements = {};
        if (status == DORA_OK && field.type.id == DORA_TYPE_UNION) {
            selected = dora_vector_view_get_member(&view, 0, &selected_row);
            if (selected >= field.child_count) {
                status = EINVAL;
            }
        } else if (status == DORA_OK && field.type.id != DORA_TYPE_STRUCT) {
            elements = dora_vector_view_get_elements(&view, 0);
        }
        for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
            if (field.type.id == DORA_TYPE_UNION && index != selected) {
                continue;
            }
            struct dora_builder child = {};
            struct dora_vector values = {};
            struct dora_schema child_schema = {};
            status = dora_schema_get_child(schema, index, &child_schema);
            if (status == DORA_OK) {
                status = dora_builder_get_child(builder, index, &child);
            }
            if (status == DORA_OK) {
                status = dora_vector_get_child(source, index, &values);
            }
            if (status == DORA_OK && index == 0 && field.type.id != DORA_TYPE_STRUCT &&
                field.type.id != DORA_TYPE_UNION) {
                struct dora_builder_metrics metrics = {};
                status = dora_builder_get_metrics(&child, &metrics);
                destination_offset = metrics.row_count;
            }
            const struct dora_range rows = field.type.id == DORA_TYPE_STRUCT ? (struct dora_range){.length = 1}
                                           : field.type.id == DORA_TYPE_UNION
                                               ? (struct dora_range){.offset = selected_row, .length = 1}
                                               : elements;
            for (uint64_t row = 0; status == DORA_OK && row < rows.length; ++row) {
                struct dora_vector scalar = {};
                status =
                    dora_vector_slice(&values, (struct dora_range){.offset = rows.offset + row, .length = 1}, &scalar);
                if (status == DORA_OK) {
                    status = bound_convert(context, &child, &child_schema, &scalar, BOUND_APPEND);
                }
            }
        }
        if (status == DORA_OK) {
            switch (field.type.id) {
            case DORA_TYPE_STRUCT:
                status = builder->node_index == 0 ? dora_builder_end_row_error(builder, context->error)
                                                  : dora_builder_append_container_error(builder,
                                                                                        DORA_TYPE_STRUCT,
                                                                                        (struct dora_range){},
                                                                                        0,
                                                                                        false,
                                                                                        context->error);
                break;
            case DORA_TYPE_UNION:
                status = dora_builder_append_container_error(builder,
                                                             DORA_TYPE_UNION,
                                                             (struct dora_range){},
                                                             field.type.parameters.type_codes[selected],
                                                             false,
                                                             context->error);
                break;
            case DORA_TYPE_FIXED_LIST:
                status = dora_builder_append_container_error(builder,
                                                             DORA_TYPE_FIXED_LIST,
                                                             (struct dora_range){},
                                                             0,
                                                             false,
                                                             context->error);
                break;
            default: {
                const bool list_view =
                    field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
                status = dora_builder_append_container_error(
                    builder,
                    DORA_TYPE_LIST,
                    (struct dora_range){.offset = list_view ? destination_offset : 0, .length = elements.length},
                    0,
                    list_view,
                    context->error);
                break;
            }
            }
        }
    } else if (status == DORA_OK) {
        struct dora_logical_type original = {};
        status = dora_vector_get_type(source, &original);
        bool same = original.id == field.type.id;
        if (same && field.type.id == DORA_TYPE_DECIMAL) {
            same = original.parameters.decimal.bit_width == field.type.parameters.decimal.bit_width &&
                   original.parameters.decimal.precision == field.type.parameters.decimal.precision;
        }
        if (status == DORA_OK && (field.type.id == DORA_TYPE_BINARY || field.type.id == DORA_TYPE_UTF8 ||
                                  field.type.id == DORA_TYPE_FIXED_BINARY)) {
            same = true;
        }
        if (status == DORA_OK) {
            status = conversion_cost(context->budget, source, same, 0);
        }
        if (status == DORA_OK) {
            if (context->allows_utf8_as_binary && field.type.id == DORA_TYPE_BINARY && view.type == DORA_TYPE_UTF8) {
                struct dora_bytes bytes = {};
                bool null = false;
                status = dora_vector_read_bytes(source, 0, &bytes, &null);
                if (status == DORA_OK) {
                    status = dora_builder_append_bytes_error(builder, bytes, context->error);
                }
            } else {
                status = same ? dora_builder_append_vector_error(builder, source, context->error)
                              : append_widened(builder, &field, source, &view, context->budget, context->error);
            }
        }
    }
    return status;
}

static int bound_append(struct bound_context* context, struct dora_builder* builder, const struct dora_vector* source) {
    struct dora_schema target = {};
    struct dora_schema actual = {};
    int status = dora_builder_retain_schema(builder, &target);
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(source, &actual);
    }
    if (status == DORA_OK) {
        status = bound_compatible(&target, &actual, context->allows_utf8_as_binary, context->budget, 0);
    }
    if (status == DORA_OK) {
        status = bound_convert(context, builder, &target, source, BOUND_PREPARE);
    }
    if (status == DORA_OK) {
        status = bound_convert(context, builder, &target, source, BOUND_APPEND);
    }
    dora_schema_release(&actual);
    dora_schema_release(&target);
    return status;
}

static int export_values(struct dora_pool pool,
                         uint32_t nodes,
                         const struct statistic_preorder* preorder,
                         const struct statistic_cell* cells,
                         const struct dora_schema* members,
                         uint32_t member_count,
                         struct dora_vector* out_values,
                         struct dora_schema* out_schema) {
    int8_t codes[UNION_MEMBER_CAPACITY] = {};
    for (uint32_t index = 0; index < member_count; ++index) {
        codes[index] = (int8_t)index;
    }
    struct statistic_fields fields = {};
    int status = fields_append(pool, &fields, (struct dora_schema_field){.name = "column", .type.id = DORA_TYPE_INT32});
    if (status == DORA_OK) {
        status = fields_append(pool,
                               &fields,
                               (struct dora_schema_field){
                                   .name = "statistics",
                                   .type.id = DORA_TYPE_MAP,
                                   .layout = DORA_LAYOUT_LIST,
                                   .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
                                   .child_count = 1
                               });
    }
    if (status == DORA_OK) {
        status = fields_append(pool,
                               &fields,
                               (struct dora_schema_field){
                                   .name = "entries",
                                   .type.id = DORA_TYPE_STRUCT,
                                   .layout = DORA_LAYOUT_STRUCT,
                                   .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
                                   .child_count = 2
                               });
    }
    if (status == DORA_OK) {
        status = fields_append(pool,
                               &fields,
                               (struct dora_schema_field){
                                   .name = "key",
                                   .type.id = DORA_TYPE_UTF8,
                                   .layout = DORA_LAYOUT_DICTIONARY,
                                   .index_type = DORA_TYPE_INT32,
                                   .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
                                   .child_count = 1
                               });
    }
    if (status == DORA_OK) {
        status = fields_append(pool, &fields, (struct dora_schema_field){.type.id = DORA_TYPE_UTF8});
    }
    if (status == DORA_OK) {
        status = fields_append(pool,
                               &fields,
                               (struct dora_schema_field){
                                   .name = "items",
                                   .type.id = DORA_TYPE_UNION,
                                   .type.parameters.type_codes = codes,
                                   .layout = DORA_LAYOUT_DENSE_UNION,
                                   .flags = DORA_SCHEMA_FIELD_FLAG_NONNULLABLE,
                                   .child_count = member_count
                               });
    }
    for (uint32_t index = 0; status == DORA_OK && index < member_count; ++index) {
        status = fields_clone(pool, &fields, &members[index], "value", 0);
    }
    if (status == DORA_OK) {
        status = dora_schema_init_copy(
            out_schema,
            pool,
            &(const struct dora_schema_input){.fields = fields.storage.address, .field_count = fields.field_count});
    }
    for (uint32_t index = 6; index < fields.field_count; ++index) {
        struct dora_schema_field* const field = &((struct dora_schema_field*)fields.storage.address)[index];
        if (field->type.id == DORA_TYPE_UTF8) {
            field->type = (struct dora_logical_type){.id = DORA_TYPE_BINARY};
        }
    }
    struct dora_builder builder = {};
    if (status == DORA_OK) {
        status = dora_builder_init_fields(
            &builder,
            pool,
            &(const struct dora_schema_input){.fields = fields.storage.address, .field_count = fields.field_count},
            nullptr);
    }
    dora_allocation_release(&fields.storage);
    struct dora_builder column = {};
    struct dora_builder map = {};
    struct dora_builder key = {};
    struct dora_builder items = {};
    if (status == DORA_OK) {
        status = dora_builder_get_child(&builder, 0, &column);
    }
    if (status == DORA_OK) {
        status = dora_builder_get_child(&builder, 1, &map);
    }
    if (status == DORA_OK) {
        status = dora_builder_get_child(&map, 0, &key);
    }
    if (status == DORA_OK) {
        status = dora_builder_get_child(&map, 1, &items);
    }
    const char* strings[ARROW_STATISTIC_CELL_COUNT] = {};
    for (uint32_t index = 0; index < STATISTICS_KIND_COUNT; ++index) {
        strings[index] = names[index].exact_key;
        strings[index + STATISTICS_KIND_COUNT] = names[index].estimate_key;
    }
    struct dora_vector dictionary = {};
    if (status == DORA_OK) {
        status = dora_vector_init_copy_cstrings(&dictionary,
                                                pool,
                                                &(const struct dora_logical_type){.id = DORA_TYPE_UTF8},
                                                DORA_LAYOUT_BINARY,
                                                strings,
                                                ARROW_STATISTIC_CELL_COUNT,
                                                DORA_FAILURE_POLICY_ERROR);
    }
    if (status == DORA_OK) {
        status = dora_builder_set_dictionary(&key, &dictionary);
    }
    dora_vector_release(&dictionary);
    for (uint32_t node = 0; status == DORA_OK && node < nodes; ++node) {
        const uint32_t canonical = preorder->native_to_arrow[node];
        if (canonical == UINT32_MAX) {
            continue;
        }
        uint64_t count = 0;
        for (uint32_t entry = 0; status == DORA_OK && entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            const struct statistic_cell* const cell = &cells[node * ARROW_STATISTIC_CELL_COUNT + entry];
            if (!cell->is_present) {
                continue;
            }
            struct dora_builder member = {};
            status = dora_builder_get_child(&items, cell->member_index, &member);
            struct bound_context context = {.pool = pool, .allows_utf8_as_binary = true};
            if (status == DORA_OK) {
                status = bound_append(&context, &member, &cell->value);
            }
            bound_release(&context);
            if (status == DORA_OK) {
                status = dora_builder_append_union(&items, (int8_t)cell->member_index);
            }
            if (status == DORA_OK) {
                status = dora_builder_append_i32(&key, (int32_t)entry);
            }
            ++count;
        }
        if (status == DORA_OK) {
            status = dora_builder_append_map(&map, count);
        }
        if (status == DORA_OK) {
            if (preorder->has_record_root && node == 0) {
                status = dora_builder_append_null(&column);
            } else {
                status = dora_builder_append_i32(&column, (int32_t)(canonical - (preorder->has_record_root ? 1U : 0U)));
            }
        }
        if (status == DORA_OK) {
            status = dora_builder_end_row(&builder);
        }
    }
    if (status == DORA_OK) {
        status = dora_builder_finish(&builder, out_values);
    }
    dora_builder_release(&builder);
    return status;
}

static int import_shape(const struct dora_vector* values,
                        struct dora_vector* out_column,
                        struct dora_vector* out_map,
                        struct dora_vector* out_keys,
                        struct dora_vector* out_items) {
    struct dora_vector_view view = {};
    int status = dora_vector_get_view(values, &view);
    uint32_t children = 0;
    if (status == DORA_OK) {
        status = dora_vector_get_child_count(values, &children);
    }
    if (status == DORA_OK && (view.type != DORA_TYPE_STRUCT || children != 2)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_find_child(values, "column", out_column);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(out_column, &view);
    }
    if (status == DORA_OK && view.type != DORA_TYPE_INT32) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_find_child(values, "statistics", out_map);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(out_map, &view);
    }
    if (status == DORA_OK && view.type != DORA_TYPE_MAP) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child(out_map, 0, out_keys);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(out_keys, &view);
    }
    if (status == DORA_OK && (view.type != DORA_TYPE_UTF8 || view.layout != DORA_LAYOUT_DICTIONARY ||
                              view.physical_type != DORA_PHYSICAL_TYPE_I32)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = dora_vector_get_child(out_map, 1, out_items);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(out_items, &view);
    }
    if (status == DORA_OK && view.layout != DORA_LAYOUT_DENSE_UNION) {
        status = EINVAL;
    }
    return status;
}

static int import_cells(const struct dora_vector* values,
                        const struct statistic_preorder* preorder,
                        struct statistic_cell* cells,
                        struct statistic_budget* budget) {
    struct dora_vector columns = {};
    struct dora_vector maps = {};
    struct dora_vector keys = {};
    struct dora_vector items = {};
    int status = import_shape(values, &columns, &maps, &keys, &items);
    struct dora_vector_view map_view = {};
    struct dora_vector_view item_view = {};
    if (status == DORA_OK) {
        status = dora_vector_get_view(&maps, &map_view);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(&items, &item_view);
    }
    for (uint64_t row = 0; status == DORA_OK && row < values->rows.length; ++row) {
        int32_t column = 0;
        bool is_null = false;
        status = statistics_charge(budget, 1, 0);
        if (status == DORA_OK) {
            status = dora_vector_is_null(values, row, &is_null);
        }
        if (status == DORA_OK && is_null) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            status = dora_vector_read_i32(&columns, row, &column, &is_null);
        }
        uint32_t node = 0;
        if (status == DORA_OK && is_null && !preorder->has_record_root) {
            status = EINVAL;
        }
        if (status == DORA_OK && !is_null) {
            const uint64_t canonical = (uint32_t)column + (preorder->has_record_root ? 1U : 0U);
            if (column < 0 || canonical >= preorder->arrow_node_count) {
                status = EINVAL;
            } else {
                node = preorder->arrow_to_native[canonical];
            }
        }
        if (status == DORA_OK && dora_vector_view_is_null(&map_view, row)) {
            status = EINVAL;
        }
        if (status != DORA_OK) {
            break;
        }
        const struct dora_range entries = dora_vector_view_get_elements(&map_view, row);
        for (uint64_t index = 0; status == DORA_OK && index < entries.length; ++index) {
            const uint64_t entry = entries.offset + index;
            struct dora_bytes key = {};
            status = dora_vector_read_bytes(&keys, entry, &key, &is_null);
            if (status == DORA_OK && is_null) {
                status = EINVAL;
            }
            uint32_t kind = ARROW_STATISTIC_CELL_COUNT;
            for (uint32_t candidate = 0; status == DORA_OK && candidate < ARROW_STATISTIC_CELL_COUNT; ++candidate) {
                const char* const name = candidate < STATISTICS_KIND_COUNT
                                             ? names[candidate].exact_key
                                             : names[candidate - STATISTICS_KIND_COUNT].estimate_key;
                const size_t length = strlen(name);
                if (key.byte_count == length) {
                    status = statistics_charge(budget, key.byte_count, 0);
                }
                if (status == DORA_OK &&
                    bytes_are_equal(key, (struct dora_bytes){.data = (const uint8_t*)name, .byte_count = length})) {
                    kind = candidate;
                    break;
                }
            }
            if (kind == ARROW_STATISTIC_CELL_COUNT) {
                continue;
            }
            struct statistic_cell* const cell = &cells[node * ARROW_STATISTIC_CELL_COUNT + kind];
            if (cell->is_present) {
                status = EINVAL;
                break;
            }
            uint64_t value_row = 0;
            const uint32_t member = dora_vector_view_get_member(&item_view, entry, &value_row);
            struct dora_vector child = {};
            if (status == DORA_OK) {
                status = dora_vector_get_child(&items, member, &child);
            }
            if (status == DORA_OK) {
                status = dora_vector_slice(&child, (struct dora_range){.offset = value_row, .length = 1}, &cell->value);
            }
            if (status == DORA_OK) {
                cell->is_present = true;
            }
        }
    }
    return status;
}

static int bound_schema_cost(struct statistic_budget* budget,
                             const struct dora_schema* schema,
                             struct dora_schema_field field,
                             uint32_t depth) {
    if (depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    field.flags &= (uint8_t)~DORA_SCHEMA_FIELD_FLAG_NONNULLABLE;
    const bool compose = field.layout == DORA_LAYOUT_RUN_END || field.type.id == DORA_TYPE_UNION;
    bool unchanged = false;
    if (!compose) {
        int status = statistics_charge(budget, 1, 0);
        if (status == DORA_OK) {
            status = dora_schema_field_matches(schema, &field, &unchanged);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    if (unchanged) {
        return DORA_OK;
    }
    int status = field_cost(budget, &field);
    if (status == DORA_OK && compose && field.child_count != 0) {
        struct dora_schema child = {};
        status = dora_schema_get_physical_child(schema, field.layout == DORA_LAYOUT_RUN_END ? 1 : 0, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child, &field);
        }
        if (status == DORA_OK) {
            status = bound_schema_cost(budget, &child, field, depth + 1);
        }
    }
    return status;
}

static int prototype_cost(struct statistic_budget* budget,
                          const struct dora_schema* schema,
                          const struct dora_statistics_request* requests,
                          uint32_t* in_out_node,
                          uint32_t depth) {
    if (depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    const uint16_t kinds = requests[(*in_out_node)++].kinds;
    if (status == DORA_OK) {
        status = field_cost(budget, &(const struct dora_schema_field){.name = field.name});
    }
    for (uint32_t index = 0; status == DORA_OK && index < STATISTICS_KIND_COUNT; ++index) {
        if ((kinds & dora_statistics_descriptors[index].kind) == 0) {
            continue;
        }
        if (index == 3 || index == 4) {
            struct dora_schema_field bound = field;
            bound.name = dora_statistics_descriptors[index].name;
            status = bound_schema_cost(budget, schema, bound, depth);
            if (status == DORA_OK) {
                status =
                    field_cost(budget,
                               &(const struct dora_schema_field){.name = index == 3 ? "is_min_exact" : "is_max_exact"});
            }
        } else {
            status =
                field_cost(budget, &(const struct dora_schema_field){.name = dora_statistics_descriptors[index].name});
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_schema child = {};
        status = dora_schema_get_physical_child(schema, index, &child);
        if (status == DORA_OK) {
            status = prototype_cost(budget, &child, requests, in_out_node, depth + 1);
        }
    }
    return status;
}

static int restoration_schema_matches(const struct dora_schema* source,
                                      const struct dora_schema* target,
                                      bool plain,
                                      struct statistic_budget* budget,
                                      uint32_t depth,
                                      bool* out_matches) {
    if (depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    int status = statistics_charge(budget, 1, 0);
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(target, &field);
    }
    if (plain) {
        field.type.extension_name = (struct dora_bytes){};
        field.type.extension_metadata = (struct dora_bytes){};
    }
    if (status == DORA_OK) {
        status = dora_schema_field_matches(source, &field, out_matches);
    }
    for (uint32_t index = 0; status == DORA_OK && *out_matches && index < field.child_count; ++index) {
        struct dora_schema actual = {};
        struct dora_schema expected = {};
        status = dora_schema_get_physical_child(source, index, &actual);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(target, index, &expected);
        }
        const bool storage_child = plain && ((field.layout == DORA_LAYOUT_DICTIONARY && index == 0) ||
                                             (field.layout == DORA_LAYOUT_RUN_END && index == 1));
        if (status == DORA_OK) {
            status = restoration_schema_matches(&actual, &expected, storage_child, budget, depth + 1, out_matches);
        }
    }
    return status;
}

static int restoration_cost(struct statistic_budget* budget,
                            const struct dora_vector* source,
                            const struct dora_schema* target,
                            uint32_t depth) {
    if (depth >= budget->limits.max_depth) {
        return ERANGE;
    }
    int status = statistics_charge(budget, 1, 0);
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(target, &field);
    }
    if (status == DORA_OK && field.type.extension_name.byte_count != 0) {
        struct dora_logical_type type = {};
        struct dora_vector storage = dora_vector_borrow(source);
        status = dora_vector_get_type(source, &type);
        if (status == DORA_OK && type.extension_name.byte_count != 0) {
            storage = (struct dora_vector){};
            status = dora_vector_get_storage(source, &storage);
        }
        struct dora_schema actual = {};
        if (status == DORA_OK) {
            status = dora_vector_retain_schema(&storage, &actual);
        }
        bool unchanged = false;
        if (status == DORA_OK) {
            status = restoration_schema_matches(&actual, target, true, budget, depth, &unchanged);
        }
        if (status == DORA_OK && !unchanged) {
            struct dora_schema_field plain = field;
            plain.type.extension_name = (struct dora_bytes){};
            plain.type.extension_metadata = (struct dora_bytes){};
            status = field_cost(budget, &plain);
        }
        dora_schema_release(&actual);
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_vector child = {};
        struct dora_schema declaration = {};
        status = dora_vector_get_physical_child(source, index, &child);
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(target, index, &declaration);
        }
        if (status == DORA_OK) {
            status = restoration_cost(budget, &child, &declaration, depth + 1);
        }
    }
    return status;
}

static int import_values(struct dora_pool pool,
                         const struct dora_schema* target,
                         uint32_t nodes,
                         const struct statistic_cell* cells,
                         struct dora_vector* out_statistics,
                         struct statistic_budget* budget,
                         struct dora_error* out_error) {
    struct dora_allocation requests_storage = {};
    size_t bytes = 0;
    if (ckd_mul(&bytes, nodes, sizeof(struct dora_statistics_request))) {
        return EOVERFLOW;
    }
    int status = statistics_charge(budget, nodes, bytes);
    if (status == DORA_OK) {
        status = dora_pool_allocate(pool, bytes, &requests_storage);
    }
    struct dora_statistics_request* const requests = requests_storage.address;
    for (uint32_t node = 0; status == DORA_OK && node < nodes; ++node) {
        requests[node] = (struct dora_statistics_request){.schema_node_index = node};
        for (uint32_t entry = 0; entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            if (cells[node * ARROW_STATISTIC_CELL_COUNT + entry].is_present) {
                requests[node].kinds |= dora_statistics_descriptors[entry % STATISTICS_KIND_COUNT].kind;
            }
        }
    }
    uint32_t visited = 0;
    if (status == DORA_OK) {
        status = field_cost(budget, &(const struct dora_schema_field){});
    }
    if (status == DORA_OK) {
        status = prototype_cost(budget, target, requests, &visited, 0);
    }
    struct dora_schema prototype = {};
    if (status == DORA_OK) {
        status = dora_statistics_schema_init_in_pool(&prototype, pool, target, requests, nodes);
    }
    dora_allocation_release(&requests_storage);
    dora_arrow_traversal_rewind(&budget->traversal);
    struct statistic_fields fields = {.budget = budget};
    for (uint32_t node = 0; status == DORA_OK && node < nodes; ++node) {
        struct dora_schema group = {};
        struct dora_schema_field group_field = {};
        status = dora_schema_get_physical_child(&prototype, node, &group);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&group, &group_field);
        }
        group_field.child_count = 0;
        for (uint32_t entry = 0; entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            if (cells[node * ARROW_STATISTIC_CELL_COUNT + entry].is_present) {
                group_field.child_count += entry == 3 || entry == 4 ? 2U : 1U;
            }
        }
        if (status == DORA_OK) {
            status = fields_append(pool, &fields, group_field);
        }
        for (uint32_t entry = 0; status == DORA_OK && entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            if (!cells[node * ARROW_STATISTIC_CELL_COUNT + entry].is_present) {
                continue;
            }
            const uint32_t kind = entry % STATISTICS_KIND_COUNT;
            struct dora_schema member = {};
            status = dora_schema_find_child(&group, dora_statistics_descriptors[kind].name, &member);
            if (status == DORA_OK && entry >= STATISTICS_KIND_COUNT && kind != 3 && kind != 4) {
                status = fields_append(pool,
                                       &fields,
                                       (struct dora_schema_field){
                                           .name = dora_statistics_descriptors[kind].estimate_name,
                                           .type.id = DORA_TYPE_FLOAT64
                                       });
            } else if (status == DORA_OK) {
                status = fields_clone(pool,
                                      &fields,
                                      &member,
                                      entry < STATISTICS_KIND_COUNT ? dora_statistics_descriptors[kind].name
                                                                    : dora_statistics_descriptors[kind].estimate_name,
                                      0);
            }
            if (status == DORA_OK && (entry == 3 || entry == 4)) {
                status = fields_append(pool,
                                       &fields,
                                       (struct dora_schema_field){
                                           .name = entry == 3 ? "is_min_exact" : "is_max_exact",
                                           .type.id = DORA_TYPE_BOOLEAN
                                       });
            }
        }
    }
    struct dora_schema desired = {};
    if (status == DORA_OK) {
        status = field_cost(budget, &(const struct dora_schema_field){});
    }
    if (status == DORA_OK) {
        status = dora_schema_init_copy(
            &desired,
            pool,
            &(const struct dora_schema_input){.fields = fields.storage.address, .field_count = fields.field_count});
    }
    fields_storage(&fields);
    for (uint32_t index = 0; status == DORA_OK && index < fields.field_count; ++index) {
        status = field_cost(budget, &((const struct dora_schema_field*)fields.storage.address)[index]);
    }
    struct dora_builder builder = {};
    if (status == DORA_OK) {
        status = field_cost(budget, &(const struct dora_schema_field){});
    }
    if (status == DORA_OK) {
        status = dora_builder_init_fields_error(
            &builder,
            pool,
            &(const struct dora_schema_input){.fields = fields.storage.address, .field_count = fields.field_count},
            nullptr,
            out_error);
    }
    dora_allocation_release(&fields.storage);
    dora_schema_release(&prototype);
    for (uint32_t node = 0; status == DORA_OK && node < nodes; ++node) {
        struct dora_builder group = {};
        status = dora_builder_get_child(&builder, node, &group);
        for (uint32_t entry = 0; status == DORA_OK && entry < ARROW_STATISTIC_CELL_COUNT; ++entry) {
            const struct statistic_cell* const cell = &cells[node * ARROW_STATISTIC_CELL_COUNT + entry];
            if (!cell->is_present) {
                continue;
            }
            const uint32_t kind = entry % STATISTICS_KIND_COUNT;
            struct dora_builder member = {};
            status =
                dora_builder_find_child(&group,
                                        entry < STATISTICS_KIND_COUNT ? dora_statistics_descriptors[kind].name
                                                                      : dora_statistics_descriptors[kind].estimate_name,
                                        &member);
            struct dora_vector decoded = {};
            const struct dora_vector* input = &cell->value;
            uint32_t children = 0;
            bool transport = false;
            if (status == DORA_OK) {
                status = dora_vector_get_child_count(input, &children);
            }
            if (status == DORA_OK) {
                status = dora_vector_copy_requires_transport(input, &transport);
            }
            if (status == DORA_OK && (children != 0 || transport)) {
                status = conversion_cost(budget, input, true, 0);
                if (status == DORA_OK) {
                    status = dora_vector_decode_error(input, pool, &decoded, out_error);
                }
                if (status == DORA_OK) {
                    input = &decoded;
                }
            }
            bool is_null = false;
            if (status == DORA_OK) {
                status = dora_vector_is_null(input, 0, &is_null);
            }
            if (status == DORA_OK && !is_null && kind != 3 && kind != 4) {
                struct dora_vector_view view = {};
                status = dora_vector_get_view(input, &view);
                if (status == DORA_OK && (entry >= STATISTICS_KIND_COUNT || kind == 8)) {
                    double value = 0;
                    status = view.type == DORA_TYPE_FLOAT64 ? dora_vector_read_f64(input, 0, &value, &is_null) : EINVAL;
                    if (status == DORA_OK && (!isfinite(value) || value < 0)) {
                        status = EINVAL;
                    }
                } else if (status == DORA_OK) {
                    int64_t value = 0;
                    status = view.type == DORA_TYPE_INT64 ? dora_vector_read_i64(input, 0, &value, &is_null) : EINVAL;
                    if (status == DORA_OK && value < 0) {
                        status = EINVAL;
                    }
                }
            }
            if (status == DORA_OK && !is_null && (kind == 3 || kind == 4)) {
                struct dora_vector_view view = {};
                status = dora_vector_get_view(input, &view);
                double value = 0;
                if (status == DORA_OK && view.physical_type == DORA_PHYSICAL_TYPE_F64) {
                    status = dora_vector_read_f64(input, 0, &value, &is_null);
                } else if (status == DORA_OK && view.physical_type == DORA_PHYSICAL_TYPE_F32) {
                    float single = 0;
                    status = dora_vector_read_f32(input, 0, &single, &is_null);
                    value = single;
                } else if (status == DORA_OK && view.physical_type == DORA_PHYSICAL_TYPE_F16) {
                    struct dora_f16_bits half = {};
                    status = dora_vector_read_f16(input, 0, &half, &is_null);
                    value = numeric_half_widen(half);
                }
                if (status == DORA_OK && isnan(value)) {
                    status = EINVAL;
                }
            }
            struct bound_context context =
                {.pool = pool, .budget = budget, .allows_utf8_as_binary = true, .error = out_error};
            if (status == DORA_OK) {
                status = bound_append(&context, &member, input);
            }
            bound_release(&context);
            dora_vector_release(&decoded);
            if (status == DORA_OK && (entry == 3 || entry == 4)) {
                struct dora_builder exact = {};
                status = dora_builder_find_child(&group, entry == 3 ? "is_min_exact" : "is_max_exact", &exact);
                if (status == DORA_OK) {
                    status = is_null ? dora_builder_append_null(&exact) : dora_builder_append_bool(&exact, true);
                }
            }
        }
        if (status == DORA_OK) {
            status = dora_builder_append_struct(&group);
        }
    }
    if (status == DORA_OK) {
        status = dora_builder_end_row(&builder);
    }
    struct dora_vector storage = {};
    if (status == DORA_OK) {
        status = dora_builder_finish_error(&builder, &storage, out_error);
    }
    if (status == DORA_OK) {
        status = restoration_cost(budget, &storage, &desired, 0);
    }
    if (status == DORA_OK) {
        status = dora_vector_init_retain_schema_error(out_statistics,
                                                      pool,
                                                      &storage,
                                                      &desired,
                                                      VECTOR_CONTENT_PROMISED,
                                                      out_error);
    }
    dora_vector_release(&storage);
    dora_schema_release(&desired);
    dora_builder_release(&builder);
    return status;
}

int dora_arrow_export_statistics(struct dora_pool pool,
                                 const struct dora_vector* source,
                                 struct ArrowSchema* out_schema,
                                 struct ArrowArray* out_array) {
    if (out_schema == nullptr || out_schema->release != nullptr || out_array == nullptr ||
        out_array->release != nullptr) {
        return EINVAL;
    }
    struct dora_vector retained = {};
    int status = dora_vector_retain(source, &retained);
    source = &retained;
    struct dora_vector statistics = {};
    struct dora_schema target = {};
    if (status == DORA_OK) {
        status = dora_vector_get_statistics(source, &statistics);
    }
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(source, &target);
    }
    uint32_t nodes = 0;
    if (status == DORA_OK) {
        status = node_count(&target, &nodes, nullptr, 0);
    }
    uint32_t groups = 0;
    if (status == DORA_OK) {
        status = dora_vector_get_child_count(&statistics, &groups);
    }
    if (status == DORA_OK && (groups != nodes || statistics.rows.length != 1)) {
        status = EINVAL;
    }
    struct dora_allocation storage = {};
    struct statistic_preorder preorder = {};
    if (status == DORA_OK) {
        status = cells_allocate(pool, &target, nodes, &storage, &preorder, nullptr);
    }
    struct dora_schema members[UNION_MEMBER_CAPACITY] = {};
    uint32_t member_count = 0;
    if (status == DORA_OK) {
        status = export_cells(pool, &statistics, nodes, &preorder, storage.address, members, &member_count);
    }
    if (status == DORA_OK && member_count == 0) {
        struct dora_schema record = {};
        const struct dora_schema_field field = {.type.id = DORA_TYPE_INT64};
        status =
            dora_schema_init_copy(&record, pool, &(const struct dora_schema_input){.fields = &field, .field_count = 1});
        struct dora_schema child = {};
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(&record, 0, &child);
        }
        if (status == DORA_OK) {
            status = dora_schema_retain(&child, &members[0]);
            member_count = 1;
        }
        dora_schema_release(&record);
    }
    struct dora_vector values = {};
    struct dora_schema schema = {};
    if (status == DORA_OK) {
        status = export_values(pool, nodes, &preorder, storage.address, members, member_count, &values, &schema);
    }
    struct ArrowSchema arrow_schema = {};
    struct ArrowArray arrow_array = {};
    if (status == DORA_OK) {
        status = dora_arrow_export_schema(pool, &schema, &arrow_schema);
    }
    if (status == DORA_OK) {
        status = dora_arrow_export_array_in_pool(pool, &values, &arrow_array);
    }
    if (status == DORA_OK) {
        *out_schema = arrow_schema;
        *out_array = arrow_array;
        arrow_schema = (struct ArrowSchema){};
        arrow_array = (struct ArrowArray){};
    }
    if (arrow_schema.release != nullptr) {
        arrow_schema.release(&arrow_schema);
    }
    if (arrow_array.release != nullptr) {
        arrow_array.release(&arrow_array);
    }
    dora_schema_release(&schema);
    dora_vector_release(&values);
    for (uint32_t index = 0; index < member_count; ++index) {
        dora_schema_release(&members[index]);
    }
    struct statistic_cell* const cells = storage.address;
    for (size_t index = 0; cells != nullptr && index < (size_t)nodes * ARROW_STATISTIC_CELL_COUNT; ++index) {
        dora_vector_release(&cells[index].value);
    }
    dora_allocation_release(&storage);
    dora_schema_release(&target);
    dora_vector_release(&retained);
    return status;
}

int dora_arrow_import_statistics(const struct dora_vector* source,
                                 const struct ArrowSchema* statistics_schema,
                                 const struct ArrowArray* statistics_array,
                                 const struct dora_arrow_limits* limits,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    if (!dora_vector_is_empty(out_vector) || !dora_vector_is_valid(source)) {
        return EINVAL;
    }
    struct dora_vector retained = {};
    struct dora_vector published = {};
    int status = dora_vector_retain(source, &retained);
    source = &retained;
    struct dora_schema target = {};
    struct dora_schema schema = {};
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(source, &target);
    }
    uint32_t nodes = 0;
    struct statistic_budget budget = {.limits = dora_arrow_resolve_limits(limits)};
    if (status == DORA_OK) {
        dora_arrow_traversal_init(pool, &budget.limits, &budget.traversal);
    }
    if (status == DORA_OK) {
        status = node_count(&target, &nodes, &budget, 0);
    }
    if (status == DORA_OK && nodes > budget.limits.max_node_count) {
        status = ERANGE;
    }
    if (status == DORA_OK) {
        status = dora_arrow_import_schema_impl(pool,
                                               statistics_schema,
                                               &budget.limits,
                                               &budget.traversal,
                                               &schema,
                                               out_error);
    }
    struct dora_vector values = {};
    if (status == DORA_OK) {
        status = dora_arrow_import_array_impl(pool,
                                              &schema,
                                              statistics_array,
                                              nullptr,
                                              nullptr,
                                              nullptr,
                                              &budget.limits,
                                              &budget.traversal,
                                              true,
                                              &values,
                                              out_error);
    }
    struct dora_allocation storage = {};
    struct statistic_preorder preorder = {};
    if (status == DORA_OK) {
        status = cells_allocate(pool, &target, nodes, &storage, &preorder, &budget);
    }
    if (status == DORA_OK) {
        status = import_cells(&values, &preorder, storage.address, &budget);
    }
    struct dora_vector statistics = {};
    if (status == DORA_OK) {
        status = import_values(pool, &target, nodes, storage.address, &statistics, &budget, out_error);
    }
    if (status == DORA_OK) {
        status = dora_statistics_attach_error(pool, source, &statistics, &published, out_error);
        if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
            error_message_copy_bounded(out_error, "Arrow statistics attachment failed at root");
        }
    }
    dora_vector_release(&statistics);
    dora_allocation_release(&storage);
    dora_vector_release(&values);
    dora_schema_release(&schema);
    dora_schema_release(&target);
    dora_vector_release(&retained);
    dora_allocation_release(&budget.traversal.stack_storage);
    if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
        error_message_copy_bounded(out_error, "Arrow statistics import failed");
    }
    return vector_finish_publication(&published, out_vector, status);
}

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(ARRAY_COUNT(names) == STATISTICS_KIND_COUNT);
