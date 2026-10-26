// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "hash/hash.h"
#include "memory/memory_internal.h"
#include "numeric/float.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "types/order.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int
key_append(struct dora_pool pool, struct dora_allocation* storage, size_t* size, const void* bytes, size_t count) {
    if (count != 0 && bytes == nullptr) {
        return EINVAL;
    }
    size_t required;
    if (ckd_add(&required, *size, count)) {
        return EOVERFLOW;
    }
    const int status = dora_memory_payload_reserve(pool, required, *size, storage);
    if (status == DORA_OK) {
        if (count != 0) {
            memcpy((uint8_t*)storage->address + *size, bytes, count);
        }
        *size = required;
    }
    return status;
}

int dora_statistics_key_read(struct dora_pool pool,
                             const struct dora_node_topology* nodes,
                             uint32_t node_count,
                             uint32_t node_index,
                             uint64_t row_index,
                             struct dora_allocation* in_out_storage,
                             uint16_t requested_kinds,
                             size_t* in_out_serialized_byte_count,
                             struct statistics_key* out_key,
                             struct dora_bytes* out_bytes,
                             uint32_t depth) {
    if (node_index >= node_count || depth > STATISTICS_MAX_DEPTH || row_index >= nodes[node_index].view.length) {
        return EINVAL;
    }
    const struct dora_vector_view* const view = &nodes[node_index].view;
    const bool is_null = dora_vector_view_is_null(view, row_index);
    if (!is_null && (view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END)) {
        uint64_t selected = 0;
        if (view->layout == DORA_LAYOUT_DICTIONARY) {
            const int status = dora_vector_view_read_index(view, row_index, &selected);
            if (status != DORA_OK) {
                return status;
            }
        } else {
            selected = dora_vector_view_find_run(view, row_index);
        }
        const uint32_t child = statistics_input_child(nodes,
                                                      node_count,
                                                      &nodes[node_index],
                                                      view->layout == DORA_LAYOUT_DICTIONARY ? 0 : 1);
        return dora_statistics_key_read(pool,
                                        nodes,
                                        node_count,
                                        child,
                                        selected,
                                        in_out_storage,
                                        requested_kinds,
                                        in_out_serialized_byte_count,
                                        out_key,
                                        out_bytes,
                                        depth + 1);
    }
    const bool serialize = in_out_storage != nullptr;
    const bool payload = serialize || ((uint32_t)requested_kinds &
                                       ((uint32_t)DORA_STATISTICS_KIND_MIN | (uint32_t)DORA_STATISTICS_KIND_MAX |
                                        (uint32_t)DORA_STATISTICS_KIND_DISTINCT_COUNT)) != 0;
    const bool width = ((uint32_t)requested_kinds & ((uint32_t)DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH |
                                                     (uint32_t)DORA_STATISTICS_KIND_MIN_BYTE_WIDTH |
                                                     (uint32_t)DORA_STATISTICS_KIND_MAX_BYTE_WIDTH)) != 0;
    const size_t start = *in_out_serialized_byte_count;
    struct statistics_key key = {
        .serialized_byte_count = sizeof(key),
        .type = view->type,
        .physical_type = view->physical_type,
        .is_null = is_null,
        .utf8_proof = nodes[node_index].utf8_proof
    };
    if (serialize) {
        const int status = key_append(pool, in_out_storage, in_out_serialized_byte_count, &key, sizeof(key));
        if (status != DORA_OK) {
            *out_key = key;
            return status;
        }
    }
    if (is_null) {
        *out_key = key;
        return DORA_OK;
    }
    if (view->type == DORA_TYPE_BOOLEAN) {
        if (payload) {
            key.scalar.boolean = dora_vector_view_get_bool(view, row_index);
        }
        key.byte_width = 1;
    } else if (view->type == DORA_TYPE_BINARY || view->type == DORA_TYPE_UTF8 || view->type == DORA_TYPE_FIXED_BINARY) {
        if (serialize || width || out_bytes != nullptr) {
            const struct dora_bytes value = dora_vector_view_get_bytes(view, row_index);
            key.count = value.byte_count;
            key.byte_width = value.byte_count;
            if (out_bytes != nullptr) {
                *out_bytes = value;
            }
            if (serialize) {
                int status =
                    key_append(pool, in_out_storage, in_out_serialized_byte_count, value.data, value.byte_count);
                const uint8_t padding[SERIALIZED_KEY_ALIGNMENT_BYTES] = {};
                if (status == DORA_OK) {
                    status = key_append(pool,
                                        in_out_storage,
                                        in_out_serialized_byte_count,
                                        padding,
                                        (SERIALIZED_KEY_ALIGNMENT_BYTES -
                                         *in_out_serialized_byte_count % SERIALIZED_KEY_ALIGNMENT_BYTES) %
                                            SERIALIZED_KEY_ALIGNMENT_BYTES);
                }
                if (status != DORA_OK) {
                    *out_key = key;
                    return status;
                }
            }
        }
    } else if (view->type == DORA_TYPE_STRUCT || view->type == DORA_TYPE_LIST || view->type == DORA_TYPE_FIXED_LIST ||
               view->type == DORA_TYPE_MAP || view->type == DORA_TYPE_UNION) {
        if (serialize || width || view->type == DORA_TYPE_UNION) {
            struct dora_range rows = {.offset = row_index, .length = 1};
            uint32_t first_child = nodes[node_index].first_child_index;
            if (view->type == DORA_TYPE_UNION) {
                key.member_index = dora_vector_view_get_member(view, row_index, &rows.offset);
                first_child = statistics_input_child(nodes, node_count, &nodes[node_index], key.member_index);
                if (first_child >= node_count) {
                    return EINVAL;
                }
            } else if (view->type != DORA_TYPE_STRUCT) {
                rows = dora_vector_view_get_elements(view, row_index);
            }
            for (uint64_t element = 0; element < rows.length; ++element) {
                for (uint32_t child = first_child; child < node_count;
                     child = view->type == DORA_TYPE_UNION ? UINT32_MAX : nodes[child].next_sibling_index) {
                    struct statistics_key value = {};
                    const int status = dora_statistics_key_read(pool,
                                                                nodes,
                                                                node_count,
                                                                child,
                                                                rows.offset + element,
                                                                in_out_storage,
                                                                serialize || width ? requested_kinds : 0,
                                                                in_out_serialized_byte_count,
                                                                &value,
                                                                nullptr,
                                                                depth + 1);
                    if (status != DORA_OK) {
                        *out_key = key;
                        return status;
                    }
                    const bool overflow =
                        ckd_add(&key.byte_width, key.byte_width, value.byte_width) || ckd_add(&key.count, key.count, 1);
                    if (view->type == DORA_TYPE_UNION) {
                        key.is_null = value.is_null;
                    }
                    if (overflow) {
                        *out_key = key;
                        return EOVERFLOW;
                    }
                }
            }
        }
    } else if (view->physical_type != DORA_PHYSICAL_TYPE_NONE) {
        const bool nan =
            (requested_kinds & DORA_STATISTICS_KIND_NAN_COUNT) != 0 &&
            (view->physical_type == DORA_PHYSICAL_TYPE_F16 || view->physical_type == DORA_PHYSICAL_TYPE_F32 ||
             view->physical_type == DORA_PHYSICAL_TYPE_F64);
        if (payload || nan) {
            if (view->byte_width > sizeof(key.scalar)) {
                return EINVAL;
            }
            const uint64_t physical = dora_vector_view_get_physical_row(view, row_index);
            (memcpy)(&key.scalar, (const uint8_t*)view->values.data + physical * view->byte_width, view->byte_width);
        }
        key.byte_width = view->byte_width;
    }
    if (serialize) {
        key.serialized_byte_count = *in_out_serialized_byte_count - start;
        memcpy((uint8_t*)in_out_storage->address + start, &key, sizeof(key));
    }
    *out_key = key;
    return DORA_OK;
}

static int scalar_order(const struct statistics_key* left, const struct statistics_key* right, int* out_order) {
    if (left->type == DORA_TYPE_BOOLEAN) {
        *out_order = type_order_bool(left->scalar.boolean, right->scalar.boolean);
        return DORA_OK;
    }
    switch (left->physical_type) {
// Returns DORA_OK from the enclosing function.
#define SCALAR_ORDER(TAG, MEMBER)                                                                                      \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        *out_order = type_order_##MEMBER(left->scalar.MEMBER, right->scalar.MEMBER);                                   \
        return DORA_OK;
        SCALAR_ORDER(I8, i8)
        SCALAR_ORDER(I16, i16)
        SCALAR_ORDER(I32, i32)
        SCALAR_ORDER(I64, i64)
        SCALAR_ORDER(I128, i128)
        SCALAR_ORDER(I256, i256)
        SCALAR_ORDER(U8, u8)
        SCALAR_ORDER(U16, u16)
        SCALAR_ORDER(U32, u32)
        SCALAR_ORDER(U64, u64)
        SCALAR_ORDER(U128, u128)
        SCALAR_ORDER(F16, f16)
        SCALAR_ORDER(F32, f32)
        SCALAR_ORDER(F64, f64)
        SCALAR_ORDER(INTERVAL_DAY_TIME, interval_day_time)
        SCALAR_ORDER(INTERVAL_MONTH_DAY_NANO, interval_month_day_nano)
#undef SCALAR_ORDER
    default:
        return EINVAL;
    }
}

uint64_t dora_statistics_key_hash(const struct statistics_key* key, uint64_t seed) {
    if (key->is_null) {
        return hash_value_u8(0, seed);
    }
    if (key->type == DORA_TYPE_BOOLEAN) {
        return hash_value_u8(key->scalar.boolean ? 1 : 0, seed);
    }
    if (key->type == DORA_TYPE_BINARY || key->type == DORA_TYPE_UTF8 || key->type == DORA_TYPE_FIXED_BINARY) {
        return hash_value_bytes((struct dora_bytes){.data = (const uint8_t*)(key + 1), .byte_count = key->count}, seed);
    }
    if (key->physical_type != DORA_PHYSICAL_TYPE_NONE) {
        switch (key->physical_type) {
// Returns the scalar hash from the enclosing function.
#define SCALAR_HASH(TAG, MEMBER)                                                                                       \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        return hash_value_##MEMBER(key->scalar.MEMBER, seed);
            SCALAR_HASH(I8, i8)
            SCALAR_HASH(I16, i16)
            SCALAR_HASH(I32, i32)
            SCALAR_HASH(I64, i64)
            SCALAR_HASH(I128, i128)
            SCALAR_HASH(I256, i256)
            SCALAR_HASH(U8, u8)
            SCALAR_HASH(U16, u16)
            SCALAR_HASH(U32, u32)
            SCALAR_HASH(U64, u64)
            SCALAR_HASH(U128, u128)
            SCALAR_HASH(F16, f16)
            SCALAR_HASH(F32, f32)
            SCALAR_HASH(F64, f64)
            SCALAR_HASH(INTERVAL_DAY_TIME, interval_day_time)
            SCALAR_HASH(INTERVAL_MONTH_DAY_NANO, interval_month_day_nano)
#undef SCALAR_HASH
        default:
            break;
        }
    }
    uint64_t hash = hash_value_u64(key->count, hash_value_u32(key->member_index, seed));
    const struct statistics_key* child = key + 1;
    for (uint64_t index = 0; index < key->count; ++index) {
        hash = dora_statistics_key_hash(child, hash);
        child = (const struct statistics_key*)((const uint8_t*)child + child->serialized_byte_count);
    }
    return hash;
}

int dora_statistics_bind_input(struct dora_pool pool,
                               const struct dora_vector* vector,
                               const struct statistics_node_record* expected,
                               uint32_t expected_node_count,
                               struct dora_allocation* out_nodes,
                               uint32_t* out_node_count) {
    struct dora_schema schema = {};
    struct dora_allocation walk = {};
    int status = dora_vector_retain_schema(vector, &schema);
    if (status == DORA_OK) {
        status = dora_schema_walk(pool, &schema, &walk, out_node_count);
    }
    if (status == DORA_OK && expected != nullptr && *out_node_count != expected_node_count) {
        status = EINVAL;
    }
    size_t bytes = 0;
    struct dora_allocation vectors = {};
    if (status == DORA_OK && ckd_mul(&bytes, *out_node_count, sizeof(struct dora_vector))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, &vectors);
    }
    if (status == DORA_OK && ckd_mul(&bytes, *out_node_count, sizeof(struct dora_node_topology))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, bytes, out_nodes);
    }
    const struct schema_walk_node* const topology = walk.address;
    struct dora_vector* const values = vectors.address;
    struct dora_node_topology* const nodes = out_nodes->address;
    for (uint32_t index = 0; status == DORA_OK && index < *out_node_count; ++index) {
        if (expected != nullptr) {
            struct dora_schema_field a = {};
            struct dora_schema_field b = {};
            status = dora_schema_get_field(&expected[index].schema, &a);
            if (status == DORA_OK) {
                status = dora_schema_get_field(&topology[index].schema, &b);
            }
            if (status == DORA_OK &&
                (a.child_count != b.child_count || a.layout != b.layout || a.index_type != b.index_type ||
                 !dora_type_is_equal(&a.type, &b.type, a.child_count) ||
                 topology[index].parent_node_index != expected[index].parent_node_index ||
                 topology[index].child_index != expected[index].child_index)) {
                status = EINVAL;
            }
            if (status != DORA_OK) {
                break;
            }
        }
        values[index] = (struct dora_vector){};
        if (index == 0) {
            values[index] = dora_vector_borrow(vector);
        } else {
            const uint32_t parent = topology[index].parent_node_index;
            status = nodes[parent].view.layout == DORA_LAYOUT_STRUCT
                         ? dora_vector_get_child(&values[parent], topology[index].child_index, &values[index])
                         : dora_vector_get_physical_child(&values[parent], topology[index].child_index, &values[index]);
        }
        if (status == DORA_OK) {
            nodes[index] = (struct dora_node_topology){
                .parent_node_index = topology[index].parent_node_index,
                .child_index = topology[index].child_index
            };
            status = dora_vector_get_view(&values[index], &nodes[index].view);
            struct vector_image* image = nullptr;
            if (status == DORA_OK) {
                status = dora_vector_resolve(&values[index], &image);
            }
            if (status == DORA_OK) {
                nodes[index].utf8_proof = vector_utf8_proof(image);
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_node_topology_link(nodes, *out_node_count);
    }
    dora_allocation_release(&vectors);
    dora_allocation_release(&walk);
    dora_schema_release(&schema);
    if (status != DORA_OK) {
        dora_allocation_release(out_nodes);
    }
    return status;
}

static const struct statistics_key* key_child(const struct statistics_key* key, uint64_t child) {
    if (key == nullptr || key->is_null || child >= key->count) {
        return nullptr;
    }
    const struct statistics_key* next = key + 1;
    while (child-- != 0) {
        next = (const struct statistics_key*)((const uint8_t*)next + next->serialized_byte_count);
    }
    return next;
}

static void store_index(uint32_t width, void* address, uint64_t row, uint64_t value) {
    switch (width) {
    case 1:
        ((uint8_t*)address)[row] = value;
        break;
    case 2:
        ((uint16_t*)address)[row] = value;
        break;
    case 4:
        ((uint32_t*)address)[row] = value;
        break;
    default:
        ((uint64_t*)address)[row] = value;
        break;
    }
}

static uint64_t index_limit(enum dora_type type) {
    switch (type) {
    case DORA_TYPE_INT8:
        return INT8_MAX;
    case DORA_TYPE_INT16:
        return INT16_MAX;
    case DORA_TYPE_INT32:
        return INT32_MAX;
    case DORA_TYPE_INT64:
        return INT64_MAX;
    case DORA_TYPE_UINT8:
        return UINT8_MAX;
    case DORA_TYPE_UINT16:
        return UINT16_MAX;
    case DORA_TYPE_UINT32:
        return UINT32_MAX;
    case DORA_TYPE_UINT64:
        return UINT64_MAX;
    default:
        return 0;
    }
}

int dora_statistics_key_compare(const struct statistics_key* left, const struct statistics_key* right, int* out_order) {
    if (left == nullptr || right == nullptr || out_order == nullptr || left->type != right->type) {
        return EINVAL;
    }
    if (left->is_null || right->is_null) {
        *out_order = (int)left->is_null - (int)right->is_null;
        return DORA_OK;
    }
    if (left->type == DORA_TYPE_BINARY || left->type == DORA_TYPE_UTF8 || left->type == DORA_TYPE_FIXED_BINARY) {
        const size_t common = (size_t)(left->count < right->count ? left->count : right->count);
        const int order = common == 0 ? 0 : memcmp(left + 1, right + 1, common);
        *out_order = order != 0 ? (order > 0 ? 1 : -1) : (left->count > right->count) - (left->count < right->count);
        return DORA_OK;
    }
    if (left->type == DORA_TYPE_BOOLEAN || left->physical_type != DORA_PHYSICAL_TYPE_NONE) {
        if (left->physical_type != right->physical_type) {
            return EINVAL;
        }
        return scalar_order(left, right, out_order);
    }
    if (left->type == DORA_TYPE_UNION && left->member_index != right->member_index) {
        *out_order = left->member_index < right->member_index ? -1 : 1;
        return DORA_OK;
    }
    const struct statistics_key* a = left + 1;
    const struct statistics_key* b = right + 1;
    for (uint64_t index = 0; index < left->count && index < right->count; ++index) {
        int order = 0;
        const int status = dora_statistics_key_compare(a, b, &order);
        if (status != DORA_OK || order != 0) {
            *out_order = order;
            return status;
        }
        a = (const struct statistics_key*)((const uint8_t*)a + a->serialized_byte_count);
        b = (const struct statistics_key*)((const uint8_t*)b + b->serialized_byte_count);
    }
    *out_order = (left->count > right->count) - (left->count < right->count);
    return DORA_OK;
}

static int key_materialize(struct dora_pool pool,
                           const struct dora_schema* schema,
                           const struct statistics_key* const* keys,
                           uint64_t count,
                           uint32_t depth,
                           struct dora_vector* out_vector,
                           struct dora_error* out_error) {
    if (depth > STATISTICS_MAX_DEPTH || count > INT64_MAX) {
        return EINVAL;
    }
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    const bool encoded = field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END;
    const bool bytes = type_is_bytes(field.type.id);
    const bool union_value = !encoded && field.type.id == DORA_TYPE_UNION;
    const bool list = !encoded && (field.type.id == DORA_TYPE_LIST || field.type.id == DORA_TYPE_FIXED_LIST ||
                                   field.type.id == DORA_TYPE_MAP);
    const bool binary_view = field.layout == DORA_LAYOUT_BINARY_VIEW;
    uint32_t buffer_count = field.layout == DORA_LAYOUT_NULL || field.layout == DORA_LAYOUT_RUN_END ? 0
                            : field.layout == DORA_LAYOUT_STRUCT || field.layout == DORA_LAYOUT_FIXED_LIST ||
                                    field.layout == DORA_LAYOUT_SPARSE_UNION
                                ? 1
                                : 2;
    if (field.layout == DORA_LAYOUT_BINARY || field.layout == DORA_LAYOUT_LARGE_BINARY ||
        field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW) {
        buffer_count = 3;
    }
    if (binary_view) {
        buffer_count = 3;
        for (uint64_t row = 0; row < count; ++row) {
            if (keys[row] != nullptr && !keys[row]->is_null && keys[row]->count > VIEW_INLINE_BYTE_COUNT &&
                ckd_add(&buffer_count, buffer_count, 1)) {
                return EOVERFLOW;
            }
        }
    }
    struct dora_allocation buffer_storage = {};
    struct dora_allocation child_storage = {};
    status = statistics_allocate_zero(pool, buffer_count, sizeof(struct dora_allocation), &buffer_storage);
    if (status == DORA_OK) {
        status = statistics_allocate_zero(pool, field.child_count, sizeof(struct dora_vector), &child_storage);
    }
    struct dora_allocation* const buffers = buffer_storage.address;
    struct dora_vector* const children = child_storage.address;
    int64_t null_count = 0;
    const bool nullable = (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0;
    for (uint64_t row = 0; row < count; ++row) {
        null_count += nullable && (keys[row] == nullptr || keys[row]->is_null);
    }
    const bool has_validity = field.layout != DORA_LAYOUT_NULL && field.layout != DORA_LAYOUT_RUN_END && !union_value;
    if (status == DORA_OK && has_validity && null_count != 0) {
        status = statistics_allocate_zero(pool, bitmap_words_for_bits(count), sizeof(uint64_t), &buffers[0]);
        if (status == DORA_OK) {
            for (uint64_t row = 0; row < count; ++row) {
                if (keys[row] != nullptr && !keys[row]->is_null) {
                    bitmap_set(buffers[0].address, row);
                }
            }
        }
    }
    if (field.layout == DORA_LAYOUT_NULL) {
        null_count = count;
    }
    if (status == DORA_OK && !encoded && field.layout == DORA_LAYOUT_BITS) {
        status = statistics_allocate_zero(pool, bitmap_words_for_bits(count), sizeof(uint64_t), &buffers[1]);
        for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
            if (keys[row] != nullptr && !keys[row]->is_null) {
                bitmap_assign(buffers[1].address, row, keys[row]->scalar.boolean);
            }
        }
    } else if (status == DORA_OK && !encoded && field.layout == DORA_LAYOUT_FIXED) {
        enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
        status = type_physical(&field.type, &physical);
        const uint32_t width = bytes ? field.type.parameters.byte_width : physical_width(physical);
        if (status == DORA_OK) {
            status = statistics_allocate_zero(pool, count, width, &buffers[1]);
        }
        for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
            if (keys[row] != nullptr && !keys[row]->is_null && width != 0) {
                const void* source = bytes ? (const void*)(keys[row] + 1) : (const void*)&keys[row]->scalar;
                (memcpy)((uint8_t*)buffers[1].address + row * width, source, width);
            }
        }
    } else if (status == DORA_OK && !encoded && bytes) {
        const uint32_t width = field.layout == DORA_LAYOUT_BINARY ? 4 : 8;
        status = statistics_allocate_zero(pool,
                                          count + (binary_view ? 0 : 1),
                                          binary_view ? sizeof(struct dora_binary_view_entry) : width,
                                          &buffers[1]);
        uint64_t total = 0;
        for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
            if (keys[row] != nullptr && !keys[row]->is_null && ckd_add(&total, total, keys[row]->count)) {
                status = EOVERFLOW;
            }
        }
        if (status == DORA_OK && !binary_view && ((width == 4 && total > INT32_MAX) || total > INT64_MAX)) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK) {
            status = statistics_allocate_zero(pool,
                                              binary_view ? buffer_count - 3 : total,
                                              binary_view ? sizeof(int64_t) : 1,
                                              &buffers[binary_view ? buffer_count - 1 : 2]);
        }
        uint64_t offset = 0;
        uint32_t data_index = 0;
        for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
            const struct statistics_key* const key = keys[row];
            const size_t size = key == nullptr || key->is_null ? 0 : (size_t)key->count;
            if (binary_view) {
                struct dora_binary_view_entry* const entry = &((struct dora_binary_view_entry*)buffers[1].address)[row];
                if (size > INT32_MAX) {
                    status = EOVERFLOW;
                    break;
                }
                entry->byte_count = size;
                if (size <= VIEW_INLINE_BYTE_COUNT) {
                    if (size != 0) {
                        memcpy(entry->payload.inline_bytes, key + 1, size);
                    }
                } else {
                    status = statistics_allocate_zero(pool, size, 1, &buffers[2 + data_index]);
                    if (status == DORA_OK) {
                        memcpy(buffers[2 + data_index].address, key + 1, size);
                        memcpy(entry->payload.reference.prefix, key + 1, 4);
                        entry->payload.reference.buffer_index = data_index;
                        ((int64_t*)buffers[buffer_count - 1].address)[data_index++] = size;
                    }
                }
            } else {
                if (size != 0) {
                    memcpy((uint8_t*)buffers[2].address + offset, key + 1, size);
                }
                offset += size;
                store_index(width, buffers[1].address, row + 1, offset);
            }
        }
    }
    uint64_t child_rows = count;
    struct dora_allocation references = {};
    struct dora_allocation activity = {};
    uint64_t reference_count = count;
    if (status == DORA_OK && list) {
        child_rows = 0;
        for (uint64_t row = 0; row < count; ++row) {
            const uint64_t added = field.type.id == DORA_TYPE_FIXED_LIST ? field.type.parameters.list_element_count
                                   : keys[row] == nullptr || keys[row]->is_null ? 0
                                                                                : keys[row]->count;
            if (ckd_add(&child_rows, child_rows, added)) {
                status = EOVERFLOW;
                break;
            }
        }
        reference_count = child_rows;
        const uint32_t width = field.layout == DORA_LAYOUT_LIST || field.layout == DORA_LAYOUT_LIST_VIEW ? 4 : 8;
        if (field.layout != DORA_LAYOUT_FIXED_LIST && status == DORA_OK) {
            const bool view = field.layout == DORA_LAYOUT_LIST_VIEW || field.layout == DORA_LAYOUT_LARGE_LIST_VIEW;
            if ((width == 4 && child_rows > INT32_MAX) || child_rows > INT64_MAX) {
                status = EOVERFLOW;
            } else {
                status = statistics_allocate_zero(pool, count + (view ? 0 : 1), width, &buffers[1]);
            }
            if (status == DORA_OK && view) {
                status = statistics_allocate_zero(pool, count, width, &buffers[2]);
            }
            uint64_t offset = 0;
            for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
                const uint64_t length = keys[row] == nullptr || keys[row]->is_null ? 0 : keys[row]->count;
                if (view) {
                    store_index(width, buffers[1].address, row, offset);
                    store_index(width, buffers[2].address, row, length);
                }
                offset += length;
                if (!view) {
                    store_index(width, buffers[1].address, row + 1, offset);
                }
            }
        }
    }
    if (status == DORA_OK && field.child_count != 0) {
        status = statistics_allocate_zero(pool, reference_count, sizeof(const struct statistics_key*), &references);
    }
    if (status == DORA_OK && union_value) {
        status = statistics_allocate_zero(pool, count, 1, &buffers[0]);
        if (status == DORA_OK && field.layout == DORA_LAYOUT_DENSE_UNION && count > INT32_MAX) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK && field.layout == DORA_LAYOUT_DENSE_UNION) {
            status = statistics_allocate_zero(pool, count, sizeof(int32_t), &buffers[1]);
        }
        null_count = 0;
    }
    if (status == DORA_OK && encoded) {
        struct dora_logical_type index_type = {.id = field.index_type};
        enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
        status = type_physical(&index_type, &physical);
        const uint32_t width = physical_width(physical);
        if (field.layout == DORA_LAYOUT_DICTIONARY && status == DORA_OK) {
            status = statistics_allocate_zero(pool, count, width, &buffers[1]);
        }
        if (status == DORA_OK && field.layout == DORA_LAYOUT_RUN_END && count > index_limit(field.index_type)) {
            status = EOVERFLOW;
        }
        if (status == DORA_OK && field.layout == DORA_LAYOUT_RUN_END) {
            struct dora_schema end_schema = {};
            status = dora_schema_get_child(schema, 0, &end_schema);
            struct dora_allocation ends = {};
            if (status == DORA_OK) {
                status = statistics_allocate_zero(pool, count, sizeof(struct statistics_key), &ends);
            }
            const struct statistics_key** const refs = (const struct statistics_key**)references.address;
            for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
                struct statistics_key* const key = &((struct statistics_key*)ends.address)[row];
                *key = (struct statistics_key){
                    .serialized_byte_count = sizeof(*key),
                    .type = field.index_type,
                    .physical_type = physical,
                    .byte_width = width
                };
                store_index(width, &key->scalar, 0, row + 1);
                refs[row] = key;
            }
            if (status == DORA_OK) {
                status = key_materialize(pool, &end_schema, refs, count, depth + 1, &children[0], out_error);
            }
            dora_allocation_release(&ends);
            null_count = 0;
        }
        if (status == DORA_OK && field.layout == DORA_LAYOUT_DICTIONARY) {
            const struct statistics_key** const refs = (const struct statistics_key**)references.address;
            struct dora_allocation probes = {};
            size_t capacity = 0;
            if (ckd_mul(&capacity, count, DISTINCT_LOAD_DENOMINATOR)) {
                status = EOVERFLOW;
            } else {
                status = statistics_allocate_zero(pool, capacity, sizeof(struct statistics_value_hash_slot), &probes);
            }
            struct statistics_value_hash_slot* const slots = probes.address;
            for (size_t slot = 0; status == DORA_OK && slot < capacity; ++slot) {
                slots[slot].value_position = UINT64_MAX;
            }
            child_rows = 0;
            for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
                if (keys[row] == nullptr || keys[row]->is_null) {
                    continue;
                }
                const uint64_t hash = dora_statistics_key_hash(keys[row], 0);
                size_t position = (size_t)hash % capacity;
                while (slots[position].value_position != UINT64_MAX) {
                    if (slots[position].hash == hash) {
                        int order = 0;
                        status = dora_statistics_key_compare(keys[row], refs[slots[position].value_position], &order);
                        if (status != DORA_OK || order == 0) {
                            break;
                        }
                    }
                    position = (position + 1) % capacity;
                }
                const uint64_t index =
                    slots[position].value_position == UINT64_MAX ? child_rows : slots[position].value_position;
                if (status != DORA_OK || index > index_limit(field.index_type)) {
                    status = status == DORA_OK ? EOVERFLOW : status;
                    break;
                }
                if (index == child_rows) {
                    refs[child_rows++] = keys[row];
                    slots[position] = (struct statistics_value_hash_slot){.hash = hash, .value_position = index};
                }
                store_index(width, buffers[1].address, row, index);
            }
            dora_allocation_release(&probes);
            if (status == DORA_OK && !nullable && count != 0 && child_rows == 0) {
                for (uint64_t row = 0; row < count; ++row) {
                    if (keys[row] != nullptr) {
                        status = EINVAL;
                        break;
                    }
                }
                if (status == DORA_OK) {
                    refs[0] = nullptr;
                    child_rows = 1;
                }
            }
        }
    }
    for (uint32_t child = 0; status == DORA_OK && child < field.child_count; ++child) {
        if (field.layout == DORA_LAYOUT_RUN_END && child == 0) {
            continue;
        }
        struct dora_schema child_schema = {};
        status = dora_schema_get_physical_child(schema, child, &child_schema);
        const struct statistics_key** const refs = (const struct statistics_key**)references.address;
        uint64_t written = 0;
        if (field.layout == DORA_LAYOUT_DICTIONARY) {
            written = child_rows;
        } else {
            for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
                const struct statistics_key* const key = keys[row];
                if (field.layout == DORA_LAYOUT_RUN_END) {
                    refs[written++] = key;
                } else if (list) {
                    const uint64_t elements = field.type.id == DORA_TYPE_FIXED_LIST
                                                  ? field.type.parameters.list_element_count
                                              : key == nullptr || key->is_null ? 0
                                                                               : key->count;
                    for (uint64_t element = 0; element < elements; ++element) {
                        refs[written++] = key_child(key, element * field.child_count + child);
                    }
                } else if (union_value) {
                    const uint32_t member = key == nullptr ? 0 : key->member_index;
                    if (field.layout == DORA_LAYOUT_SPARSE_UNION || member == child) {
                        if (member == child) {
                            ((int8_t*)buffers[0].address)[row] = (int8_t)field.type.parameters.type_codes[child];
                            if (field.layout == DORA_LAYOUT_DENSE_UNION) {
                                ((int32_t*)buffers[1].address)[row] = (int32_t)written;
                            }
                        }
                        refs[written++] = member == child ? key_child(key, 0) : nullptr;
                    }
                } else {
                    refs[written++] = key_child(key, child);
                }
            }
        }
        if (status == DORA_OK) {
            status = key_materialize(pool, &child_schema, refs, written, depth + 1, &children[child], out_error);
        }
    }
    if (status == DORA_OK && !nullable && field.layout != DORA_LAYOUT_NULL) {
        bool missing = false;
        for (uint64_t row = 0; !missing && row < count; ++row) {
            missing = keys[row] == nullptr;
        }
        if (missing) {
            status = statistics_allocate_zero(pool, bitmap_words_for_bits(count), sizeof(uint64_t), &activity);
            for (uint64_t row = 0; status == DORA_OK && row < count; ++row) {
                bitmap_assign(activity.address, row, keys[row] != nullptr);
            }
        }
    }
    struct dora_allocation effective = {};
    if (status == DORA_OK && field.layout == DORA_LAYOUT_NULL && !nullable && count != 0) {
        for (uint64_t row = 0; row < count; ++row) {
            if (keys[row] != nullptr) {
                status = EINVAL;
                break;
            }
        }
        if (status == DORA_OK) {
            status = statistics_allocate_zero(pool, bitmap_words_for_bits(count), sizeof(uint64_t), &effective);
        }
    }
    if (status == DORA_OK) {
        const struct dora_bytes active_rows = {
            .data = activity.address != nullptr ? activity.address : effective.address,
            .byte_count = bitmap_bytes_for_bits(count)
        };
        struct dora_pool schema_pool = {};
        const bool admitted = dora_schema_get_pool(schema, &schema_pool) == DORA_OK && pool_is_same(schema_pool, pool);
        const struct vector_input input = {
            .field = &field,
            .admitted_schema = admitted ? schema : nullptr,
            .row_count = count,
            .null_count = null_count,
            .allocations = buffers,
            .buffer_count = buffer_count,
            .children = children,
            .child_count = field.child_count,
            .ancestor_activity = active_rows.data == nullptr ? nullptr : &active_rows,
            .effective_validity = effective.address == nullptr ? nullptr : &effective,
            .effective_null_count = null_count
        };
        status = dora_vector_init_move_input(out_vector, pool, &input, buffers, out_error);
    }
    dora_allocation_release(&effective);
    dora_allocation_release(&activity);
    for (uint32_t index = 0; children != nullptr && index < field.child_count; ++index) {
        dora_vector_release(&children[index]);
    }
    for (uint32_t index = 0; buffers != nullptr && index < buffer_count; ++index) {
        dora_allocation_release(&buffers[index]);
    }
    dora_allocation_release(&references);
    dora_allocation_release(&child_storage);
    dora_allocation_release(&buffer_storage);
    return status;
}

bool dora_statistics_key_is_nan(const struct statistics_key* key) {
    if (key == nullptr || key->is_null) {
        return false;
    }
    switch (key->physical_type) {
    case DORA_PHYSICAL_TYPE_F16:
        return isnan(numeric_half_widen(key->scalar.f16));
    case DORA_PHYSICAL_TYPE_F32:
        return isnan(key->scalar.f32);
    case DORA_PHYSICAL_TYPE_F64:
        return isnan(key->scalar.f64);
    default:
        return false;
    }
}

int dora_statistics_key_materialize(struct dora_pool pool,
                                    const struct dora_schema* schema,
                                    const struct statistics_key* key,
                                    struct dora_vector* out_vector,
                                    struct dora_error* out_error) {
    return key_materialize(pool, schema, &key, 1, 0, out_vector, out_error);
}

int dora_statistics_key_serialize_row(struct dora_pool pool,
                                      const struct dora_vector* vector,
                                      uint64_t row,
                                      struct dora_allocation* out_key) {
    if (out_key == nullptr || out_key->address != nullptr || out_key->capacity_bytes != 0) {
        return EINVAL;
    }
    struct dora_vector retained = {};
    int status = dora_vector_retain(vector, &retained);
    struct dora_allocation nodes = {};
    uint32_t count = 0;
    if (status == DORA_OK) {
        status = dora_statistics_bind_input(pool, &retained, nullptr, 0, &nodes, &count);
    }
    size_t size = 0;
    if (status == DORA_OK) {
        struct statistics_key key = {};
        status = dora_statistics_key_read(pool, nodes.address, count, 0, row, out_key, 0, &size, &key, nullptr, 0);
    }
    dora_allocation_release(&nodes);
    dora_vector_release(&retained);
    if (status != DORA_OK) {
        dora_allocation_release(out_key);
    }
    return status;
}
