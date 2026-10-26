// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "kernel/binding/elementwise.h"
#include "kernel/binding/read.h"
#include "kernel/binding/reduce.h"
#include "kernel/binding/request.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"
#include "statistics/statistics_internal.h"
#include "types/order.h"
#include "utf8/input.h"
#include "utf8/utf8.h"
#include "vector/image.h"
#include "vector/node_topology.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t DISTINCT_INITIAL_SLOT_CAPACITY = 8;

static void release_replaced(struct dora_allocation current, struct dora_allocation preserved) {
    if (current.address != preserved.address) {
        dora_allocation_release(&current);
    }
}

static int preserve_reserve(struct dora_pool pool,
                            size_t required,
                            size_t initialized,
                            struct dora_allocation* current,
                            const struct dora_allocation* committed) {
    if (required <= current->capacity_bytes) {
        return DORA_OK;
    }
    if (current->address != committed->address) {
        return dora_memory_payload_reserve(pool, required, initialized, current);
    }
    struct dora_allocation candidate = {};
    int status = dora_memory_payload_allocate_preferred(pool, current->capacity_bytes, required, &candidate);
    if (status == DORA_OK) {
        if (initialized != 0) {
            memcpy(candidate.address, current->address, initialized);
        }
        *current = candidate;
    }
    return status;
}

//--------------------------------------
// Distinct Slots
//--------------------------------------

static int distinct_reserve(struct dora_pool pool,
                            struct statistics_node_accumulator* entry,
                            uint64_t need,
                            const struct statistics_node_accumulator* committed,
                            uint32_t slot_bytes) {
    if (entry->distinct_slot_capacity != 0 && entry->distinct_slot_bytes != slot_bytes) {
        return EINVAL;
    }
    if (entry->distinct_slot_capacity != 0 && need <= entry->distinct_slot_capacity / DISTINCT_LOAD_DENOMINATOR) {
        return DORA_OK;
    }
    size_t capacity = DISTINCT_INITIAL_SLOT_CAPACITY;
    while (capacity / DISTINCT_LOAD_DENOMINATOR < need) {
        if (ckd_mul(&capacity, capacity, DISTINCT_LOAD_DENOMINATOR)) {
            return EOVERFLOW;
        }
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, capacity, slot_bytes)) {
        return EOVERFLOW;
    }
    struct dora_allocation candidate = {};
    const int status = dora_memory_payload_allocate(pool, bytes, &candidate);
    if (status != DORA_OK) {
        return status;
    }
    uint8_t* const slots = candidate.address;
    const struct statistics_value_hash_slot empty = {.value_position = UINT64_MAX};
    for (size_t index = 0; index < capacity; ++index) {
        memset(slots + index * slot_bytes, 0, slot_bytes);
        memcpy(slots + index * slot_bytes, &empty, sizeof(empty));
    }
    const uint8_t* const old = entry->distinct_slots.address;
    for (size_t index = 0; index < entry->distinct_slot_capacity; ++index) {
        struct statistics_value_hash_slot header = {};
        memcpy(&header, old + index * slot_bytes, sizeof(header));
        if (header.value_position == UINT64_MAX) {
            continue;
        }
        size_t position = (size_t)header.hash & (capacity - 1);
        for (;;) {
            struct statistics_value_hash_slot occupant = {};
            memcpy(&occupant, slots + position * slot_bytes, sizeof(occupant));
            if (occupant.value_position == UINT64_MAX) {
                break;
            }
            position = (position + 1) & (capacity - 1);
        }
        memcpy(slots + position * slot_bytes, old + index * slot_bytes, slot_bytes);
    }
    release_replaced(entry->distinct_slots, committed->distinct_slots);
    entry->distinct_slots = candidate;
    entry->distinct_slot_capacity = capacity;
    entry->distinct_slot_bytes = slot_bytes;
    return DORA_OK;
}

static int distinct_grow(struct statistics_node_accumulator* entry, uint64_t additional, uint64_t* out_need) {
    return ckd_add(out_need, entry->distinct_count, additional) || *out_need > INT64_MAX ? EOVERFLOW : DORA_OK;
}

//--------------------------------------
// Typed Slots
//--------------------------------------

static inline uint64_t distinct_element(const struct kernel_packet* packet, uint32_t lane) {
    switch (packet->shape) {
    case KERNEL_SHAPE_GATHERED:
        return packet->value_offset + packet->rows[lane];
    case KERNEL_SHAPE_CONSTANT:
        return packet->value_offset;
    default:
        return packet->value_offset + lane;
    }
}

#define DEFINE_DISTINCT_SLOT(S, T)                                                                                     \
    struct distinct_slot_##S {                                                                                         \
        uint64_t hash;                                                                                                 \
        uint64_t value_position;                                                                                       \
        T value;                                                                                                       \
    };                                                                                                                 \
    static inline bool distinct_slot_insert_##S(void* table,                                                           \
                                                size_t capacity,                                                       \
                                                uint64_t hash,                                                         \
                                                T value,                                                               \
                                                uint64_t index) {                                                      \
        struct distinct_slot_##S* const slots = table;                                                                 \
        for (size_t position = (size_t)hash & (capacity - 1);; position = (position + 1) & (capacity - 1)) {           \
            if (slots[position].value_position == UINT64_MAX) {                                                        \
                slots[position] = (struct distinct_slot_##S){.hash = hash, .value_position = index, .value = value};   \
                return true;                                                                                           \
            }                                                                                                          \
            if (slots[position].hash == hash && type_order_##S(slots[position].value, value) == 0) {                   \
                return false;                                                                                          \
            }                                                                                                          \
        }                                                                                                              \
    }
#define DEFINE_DISTINCT_WINDOW(S, T)                                                                                   \
    static uint64_t distinct_window_##S(void* table,                                                                   \
                                        size_t capacity,                                                               \
                                        const struct kernel_packet* packet,                                            \
                                        const uint64_t* hashes,                                                        \
                                        uint64_t count) {                                                              \
        const uint8_t* const values = packet->values;                                                                  \
        for (uint32_t word = 0; word * 64 < packet->count; ++word) {                                                   \
            const uint64_t tail = bitmap_word_tail_mask(packet->count, word);                                          \
            uint64_t active = packet->validity == nullptr ? tail : packet->validity[word] & tail;                      \
            while (active != 0) {                                                                                      \
                const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                            \
                T value = {};                                                                                          \
                (memcpy)(&value, values + distinct_element(packet, lane) * sizeof(T), sizeof(T));                      \
                count += distinct_slot_insert_##S(table, capacity, hashes[lane], value, count) ? 1 : 0;                \
            }                                                                                                          \
        }                                                                                                              \
        return count;                                                                                                  \
    }
// Carriers: signed integers, unsigned integers, floating-point types, IntervalDayTime, IntervalMonthDayNano
#define DEFINE_DISTINCT_CARRIER(S, T)                                                                                  \
    DEFINE_DISTINCT_SLOT(S, T)                                                                                         \
    DEFINE_DISTINCT_WINDOW(S, T)
DEFINE_DISTINCT_CARRIER(i8, int8_t)
DEFINE_DISTINCT_CARRIER(i16, int16_t)
DEFINE_DISTINCT_CARRIER(i32, int32_t)
DEFINE_DISTINCT_CARRIER(i64, int64_t)
DEFINE_DISTINCT_CARRIER(i128, struct dora_i128)
DEFINE_DISTINCT_CARRIER(i256, struct dora_i256)
DEFINE_DISTINCT_CARRIER(u8, uint8_t)
DEFINE_DISTINCT_CARRIER(u16, uint16_t)
DEFINE_DISTINCT_CARRIER(u32, uint32_t)
DEFINE_DISTINCT_CARRIER(u64, uint64_t)
DEFINE_DISTINCT_CARRIER(u128, struct dora_u128)
DEFINE_DISTINCT_CARRIER(f16, struct dora_f16_bits)
DEFINE_DISTINCT_CARRIER(f32, float)
DEFINE_DISTINCT_CARRIER(f64, double)
DEFINE_DISTINCT_CARRIER(interval_day_time, struct dora_interval_day_time)
DEFINE_DISTINCT_CARRIER(interval_month_day_nano, struct dora_interval_month_day_nano)
DEFINE_DISTINCT_SLOT(bool, bool)
#undef DEFINE_DISTINCT_CARRIER
#undef DEFINE_DISTINCT_WINDOW
#undef DEFINE_DISTINCT_SLOT

static uint64_t distinct_window_boolean(void* table,
                                        size_t capacity,
                                        const struct kernel_packet* packet,
                                        const uint64_t* hashes,
                                        uint64_t count) {
    for (uint32_t word = 0; word * 64 < packet->count; ++word) {
        const uint64_t tail = bitmap_word_tail_mask(packet->count, word);
        uint64_t active = packet->validity == nullptr ? tail : packet->validity[word] & tail;
        while (active != 0) {
            const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);
            const bool value = dora_bitmap_test(packet->values, distinct_element(packet, lane));
            count += distinct_slot_insert_bool(table, capacity, hashes[lane], value, count) ? 1 : 0;
        }
    }
    return count;
}

struct distinct_carrier {
    uint64_t (*window)(void* table,
                       size_t capacity,
                       const struct kernel_packet* packet,
                       const uint64_t* hashes,
                       uint64_t count);
    uint32_t slot_bytes;
};

static struct distinct_carrier distinct_carrier_of(enum dora_type type, enum dora_physical_type physical) {
    if (type == DORA_TYPE_BOOLEAN) {
        return (struct distinct_carrier){distinct_window_boolean, sizeof(struct distinct_slot_bool)};
    }
    switch (physical) {
// Returns a carrier descriptor from the enclosing function.
#define DISTINCT_CARRIER(TAG, S)                                                                                       \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        return (struct distinct_carrier){distinct_window_##S, sizeof(struct distinct_slot_##S)};
        DISTINCT_CARRIER(I8, i8)
        DISTINCT_CARRIER(I16, i16)
        DISTINCT_CARRIER(I32, i32)
        DISTINCT_CARRIER(I64, i64)
        DISTINCT_CARRIER(I128, i128)
        DISTINCT_CARRIER(I256, i256)
        DISTINCT_CARRIER(U8, u8)
        DISTINCT_CARRIER(U16, u16)
        DISTINCT_CARRIER(U32, u32)
        DISTINCT_CARRIER(U64, u64)
        DISTINCT_CARRIER(U128, u128)
        DISTINCT_CARRIER(F16, f16)
        DISTINCT_CARRIER(F32, f32)
        DISTINCT_CARRIER(F64, f64)
        DISTINCT_CARRIER(INTERVAL_DAY_TIME, interval_day_time)
        DISTINCT_CARRIER(INTERVAL_MONTH_DAY_NANO, interval_month_day_nano)
#undef DISTINCT_CARRIER
    default:
        return (struct distinct_carrier){};
    }
}

static bool
distinct_key_insert(void* table, size_t capacity, const struct statistics_key* key, uint64_t hash, uint64_t count) {
    if (key->type == DORA_TYPE_BOOLEAN) {
        return distinct_slot_insert_bool(table, capacity, hash, key->scalar.boolean, count);
    }
    switch (key->physical_type) {
// Returns the insertion result from the enclosing function.
#define DISTINCT_KEY(TAG, S)                                                                                           \
    case DORA_PHYSICAL_TYPE_##TAG:                                                                                     \
        return distinct_slot_insert_##S(table, capacity, hash, key->scalar.S, count);
        DISTINCT_KEY(I8, i8)
        DISTINCT_KEY(I16, i16)
        DISTINCT_KEY(I32, i32)
        DISTINCT_KEY(I64, i64)
        DISTINCT_KEY(I128, i128)
        DISTINCT_KEY(I256, i256)
        DISTINCT_KEY(U8, u8)
        DISTINCT_KEY(U16, u16)
        DISTINCT_KEY(U32, u32)
        DISTINCT_KEY(U64, u64)
        DISTINCT_KEY(U128, u128)
        DISTINCT_KEY(F16, f16)
        DISTINCT_KEY(F32, f32)
        DISTINCT_KEY(F64, f64)
        DISTINCT_KEY(INTERVAL_DAY_TIME, interval_day_time)
        DISTINCT_KEY(INTERVAL_MONTH_DAY_NANO, interval_month_day_nano)
#undef DISTINCT_KEY
    default:
        return false;
    }
}

//--------------------------------------
// Row-Index Slots
//--------------------------------------

static int distinct_insert(struct dora_pool pool,
                           struct statistics_node_accumulator* entry,
                           const struct statistics_node_accumulator* committed,
                           const struct statistics_key* key,
                           uint64_t hash) {
    uint64_t need = 0;
    int status = distinct_grow(entry, 1, &need);
    const struct distinct_carrier carrier =
        key->type == DORA_TYPE_BOOLEAN || key->physical_type != DORA_PHYSICAL_TYPE_NONE
            ? distinct_carrier_of(key->type, key->physical_type)
            : (struct distinct_carrier){};
    if (status == DORA_OK) {
        status = distinct_reserve(pool,
                                  entry,
                                  need,
                                  committed,
                                  carrier.window != nullptr ? carrier.slot_bytes
                                                            : (uint32_t)sizeof(struct statistics_value_hash_slot));
    }
    if (status != DORA_OK) {
        return status;
    }
    const size_t capacity = entry->distinct_slot_capacity;
    if (carrier.window != nullptr) {
        entry->distinct_count +=
            distinct_key_insert(entry->distinct_slots.address, capacity, key, hash, entry->distinct_count) ? 1 : 0;
        return DORA_OK;
    }
    struct statistics_value_hash_slot* const slots = entry->distinct_slots.address;
    size_t position = (size_t)hash & (capacity - 1);
    for (; slots[position].value_position != UINT64_MAX; position = (position + 1) & (capacity - 1)) {
        if (slots[position].hash == hash) {
            const struct statistics_key* const stored =
                (const struct statistics_key*)((const uint8_t*)entry->distinct_values.address +
                                               slots[position].value_position);
            int order = 0;
            status = dora_statistics_key_compare(key, stored, &order);
            if (status != DORA_OK || order == 0) {
                return status;
            }
        }
    }
    size_t required = 0;
    if (ckd_add(&required, entry->distinct_value_bytes, key->serialized_byte_count)) {
        return EOVERFLOW;
    }
    status = preserve_reserve(pool,
                              required,
                              entry->distinct_value_bytes,
                              &entry->distinct_values,
                              &committed->distinct_values);
    if (status != DORA_OK) {
        return status;
    }
    memcpy((uint8_t*)entry->distinct_values.address + entry->distinct_value_bytes, key, key->serialized_byte_count);
    slots[position] = (struct statistics_value_hash_slot){.hash = hash, .value_position = entry->distinct_value_bytes};
    entry->distinct_value_bytes = required;
    ++entry->distinct_count;
    return DORA_OK;
}

static const struct statistics_key* accumulator_bound(const struct statistics_node_accumulator* entry, bool maximum) {
    const struct dora_allocation* const storage = maximum ? &entry->maximum : &entry->minimum;
    const struct statistics_key* const key = maximum ? &entry->maximum_key : &entry->minimum_key;
    return storage->address != nullptr ? storage->address : key->serialized_byte_count != 0 ? key : nullptr;
}

static int bound_update(struct dora_pool pool,
                        struct statistics_node_accumulator* entry,
                        const struct statistics_node_accumulator* committed,
                        const struct statistics_key* key,
                        const struct dora_bytes* bytes,
                        bool maximum) {
    struct dora_allocation* const bound = maximum ? &entry->maximum : &entry->minimum;
    const struct statistics_key* const previous = accumulator_bound(entry, maximum);
    bool replace = previous == nullptr;
    if (previous != nullptr) {
        int status;
        if (bytes != nullptr) {
            const int order = type_order_bytes(
                *bytes,
                (struct dora_bytes){.data = (const uint8_t*)(previous + 1), .byte_count = previous->count});
            replace = (maximum && order > 0) || (!maximum && order < 0);
            status = DORA_OK;
        } else {
            int order = 0;
            status = dora_statistics_key_compare(key, previous, &order);
            replace = (maximum && order > 0) || (!maximum && order < 0);
        }
        if (status != DORA_OK) {
            return status;
        }
    }
    if (!replace) {
        return DORA_OK;
    }
    size_t size = key->serialized_byte_count;
    size_t padding = 0;
    if (bytes != nullptr) {
        if (ckd_add(&size, sizeof(*key), bytes->byte_count)) {
            return EOVERFLOW;
        }
        padding =
            (SERIALIZED_KEY_ALIGNMENT_BYTES - size % SERIALIZED_KEY_ALIGNMENT_BYTES) % SERIALIZED_KEY_ALIGNMENT_BYTES;
        if (ckd_add(&size, size, padding)) {
            return EOVERFLOW;
        }
    }
    const struct dora_allocation preserved = maximum ? committed->maximum : committed->minimum;
    struct statistics_key* const inline_key = maximum ? &entry->maximum_key : &entry->minimum_key;
    if (size == sizeof(*key)) {
        release_replaced(*bound, preserved);
        *bound = (struct dora_allocation){};
        *inline_key = *key;
        return DORA_OK;
    }
    if (bound->address == preserved.address) {
        *bound = (struct dora_allocation){};
    }
    const int status = dora_memory_payload_reserve(pool, size, 0, bound);
    if (status == DORA_OK) {
        if (bytes == nullptr) {
            memcpy(bound->address, key, size);
        } else {
            struct statistics_key* const stored = bound->address;
            *stored = *key;
            stored->serialized_byte_count = size;
            memcpy(stored + 1, bytes->data, bytes->byte_count);
            memset((uint8_t*)(stored + 1) + bytes->byte_count, 0, padding);
        }
        *inline_key = (struct statistics_key){};
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Incremental Accumulation
//
//------------------------------------------------------------------------------

//--------------------------------------
// Value Hashes
//--------------------------------------

struct value_hasher {
    struct kernel_unary_binding binding;
    const struct dora_logical_type* type;
    bool is_bound;
};

static bool statistics_type_is_bytes(enum dora_type type) {
    return type == DORA_TYPE_BINARY || type == DORA_TYPE_UTF8 || type == DORA_TYPE_FIXED_BINARY;
}

static bool statistics_type_is_nested(enum dora_type type) {
    return type == DORA_TYPE_STRUCT || type == DORA_TYPE_LIST || type == DORA_TYPE_FIXED_LIST ||
           type == DORA_TYPE_MAP || type == DORA_TYPE_UNION;
}

static int key_value_hash(struct value_hasher* hasher, const struct statistics_key* key, uint64_t* out_hash) {
    if (statistics_type_is_nested(key->type)) {
        *out_hash = dora_statistics_key_hash(key, 0);
        return DORA_OK;
    }
    if (!hasher->is_bound) {
        const struct kernel_bind_request request = {.input_type = hasher->type};
        const int status = dora_kernel_hash_bind(&request, &hasher->binding);
        if (status != DORA_OK) {
            return status;
        }
        hasher->is_bound = true;
    }
    const uint64_t bits = key->type == DORA_TYPE_BOOLEAN && key->scalar.boolean ? 1 : 0;
    const struct dora_bytes bytes = {.data = (const uint8_t*)(key + 1), .byte_count = key->count};
    const struct kernel_packet packet = {
        .values = key->type == DORA_TYPE_BOOLEAN        ? (const void*)&bits
                  : statistics_type_is_bytes(key->type) ? (const void*)&bytes
                                                        : (const void*)&key->scalar,
        .count = 1,
        .shape = KERNEL_SHAPE_FLAT,
        .placement = {.device = {.type = DORA_DEVICE_TYPE_CPU}, .access = DORA_ACCESS_HOST},
    };
    struct kernel_result result = {.values = out_hash, .row_capacity = 1};
    return hasher->binding.fill(&packet, &hasher->binding, &result);
}

//--------------------------------------
// Windows
//--------------------------------------

struct window_scope {
    struct statistics_node_accumulator* entry;
    const struct statistics_node_accumulator* committed;
    const struct dora_node_topology* nodes;
    const struct dora_vector_view* values;
    struct dora_pool pool;
    uint32_t node_count;
    uint32_t node;
    uint16_t requested;
    bool is_encoded;
};

struct window_bindings {
    struct kernel_reduction_binding nan;
    struct kernel_reduction_binding bounds[2];
    struct kernel_reduction_binding widths;
    struct kernel_unary_binding hash;
    struct kernel_bytes_binding bytes;
};

static constexpr uint16_t STATISTICS_WIDTH_KINDS = (uint32_t)DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH |
                                                   (uint32_t)DORA_STATISTICS_KIND_MIN_BYTE_WIDTH |
                                                   (uint32_t)DORA_STATISTICS_KIND_MAX_BYTE_WIDTH;

static bool view_supports_statistics_window(const struct dora_vector_view* view) {
    if (view->selection.indices != nullptr) {
        return false;
    }
    switch (view->layout) {
    case DORA_LAYOUT_NULL:
    case DORA_LAYOUT_BITS:
    case DORA_LAYOUT_FIXED:
        return true;
    case DORA_LAYOUT_BINARY:
    case DORA_LAYOUT_LARGE_BINARY:
    case DORA_LAYOUT_BINARY_VIEW:
        return statistics_type_is_bytes(view->type);
    default:
        return false;
    }
}

static int window_bind(const struct dora_vector_view* view,
                       const struct dora_logical_type* type,
                       uint16_t requested,
                       struct window_bindings* out_bindings) {
    struct window_bindings bindings = {};
    const struct kernel_bind_request request = {.input_type = type};
    const bool bytes = statistics_type_is_bytes(view->type);
    int status = DORA_OK;
    if (view->layout == DORA_LAYOUT_NULL) {
        *out_bindings = bindings;
        return DORA_OK;
    }
    if (bytes) {
        status = dora_kernel_bytes_bind(view, &bindings.bytes);
    }
    if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_NAN_COUNT) != 0 && type->id >= DORA_TYPE_FLOAT16 &&
        type->id <= DORA_TYPE_FLOAT64) {
        status = dora_kernel_reduction_bind(KERNEL_REDUCTION_NAN_COUNT, &request, &bindings.nan);
    }
    if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_MIN) != 0) {
        status = dora_kernel_reduction_bind(KERNEL_REDUCTION_LOWER_BOUND, &request, &bindings.bounds[0]);
    }
    if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_MAX) != 0) {
        status = dora_kernel_reduction_bind(KERNEL_REDUCTION_UPPER_BOUND, &request, &bindings.bounds[1]);
    }
    if (status == DORA_OK && bytes && (requested & STATISTICS_WIDTH_KINDS) != 0) {
        status = dora_kernel_reduction_bind(KERNEL_REDUCTION_BYTE_WIDTH, &request, &bindings.widths);
    }
    if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_DISTINCT_COUNT) != 0) {
        status = dora_kernel_hash_bind(&request, &bindings.hash);
    }
    if (status == DORA_OK) {
        *out_bindings = bindings;
    }
    return status;
}

static uint64_t window_length(const struct dora_vector_view* view, uint64_t row) {
    switch (view->layout) {
    case DORA_LAYOUT_BINARY: {
        int32_t bounds[2] = {};
        memcpy(bounds, (const uint8_t*)view->offsets.data + row * sizeof(int32_t), sizeof(bounds));
        return bounds[1] - bounds[0];
    }
    case DORA_LAYOUT_LARGE_BINARY: {
        int64_t bounds[2] = {};
        memcpy(bounds, (const uint8_t*)view->offsets.data + row * sizeof(int64_t), sizeof(bounds));
        return bounds[1] - bounds[0];
    }
    case DORA_LAYOUT_BINARY_VIEW: {
        int32_t size = 0;
        memcpy(&size, (const uint8_t*)view->values.data + row * sizeof(struct dora_binary_view_entry), sizeof(size));
        return size;
    }
    default:
        return view->byte_width;
    }
}

static int window_widths(const struct window_scope* scope,
                         const struct window_bindings* bindings,
                         const struct kernel_packet* packet,
                         struct dora_range window,
                         uint32_t valid) {
    struct statistics_node_accumulator* const entry = scope->entry;
    struct kernel_reduction_state state = {};
    dora_kernel_reduction_state_init(&bindings->widths, &state);
    if (!scope->is_encoded && valid == packet->count) {
        const struct dora_vector_view* const view = &scope->nodes[scope->node].view;
        uint64_t total_string_length = 0;
        uint64_t max_string_length = state.maximum_width;
        uint64_t min_string_length = state.minimum_width;
        for (uint64_t row = view->offset + window.offset; row < view->offset + window.offset + window.length; ++row) {
            const uint64_t size = window_length(view, row);
            if (size > max_string_length) {
                max_string_length = size;
            }
            if (size < min_string_length) {
                min_string_length = size;
            }
            total_string_length += size;
        }
        state.total_width.limbs[0] = total_string_length;
        state.maximum_width = max_string_length;
        state.minimum_width = min_string_length;
        state.has_value = valid != 0;
    } else if (valid != 0) {
        const int status = bindings->widths.update(packet, &bindings->widths, &state);
        if (status != DORA_OK) {
            return status;
        }
    }
    const uint64_t low = entry->total_byte_count.limbs[0];
    entry->total_byte_count.limbs[0] += state.total_width.limbs[0];
    const uint64_t carry = (uint64_t)(entry->total_byte_count.limbs[0] < low);
    if (ckd_add(&entry->total_byte_count.limbs[1], entry->total_byte_count.limbs[1], state.total_width.limbs[1]) ||
        ckd_add(&entry->total_byte_count.limbs[1], entry->total_byte_count.limbs[1], carry)) {
        return EOVERFLOW;
    }
    if (state.has_value) {
        entry->minimum_byte_count =
            state.minimum_width < entry->minimum_byte_count ? state.minimum_width : entry->minimum_byte_count;
        entry->maximum_byte_count =
            state.maximum_width > entry->maximum_byte_count ? state.maximum_width : entry->maximum_byte_count;
    }
    return DORA_OK;
}

static int window_bound(const struct window_scope* scope, const struct kernel_reduction_state* state, bool maximum) {
    const struct dora_vector_view* const view = scope->values;
    struct statistics_key key = {
        .serialized_byte_count = sizeof(key),
        .type = view->type,
        .physical_type = view->physical_type,
        .utf8_proof = scope->nodes[scope->node].utf8_proof
    };
    if (statistics_type_is_bytes(view->type)) {
        struct dora_bytes bytes = {};
        memcpy(&bytes, state->value.extremum, sizeof(bytes));
        key.count = bytes.byte_count;
        key.byte_width = bytes.byte_count;
        return bound_update(scope->pool, scope->entry, scope->committed, &key, &bytes, maximum);
    }
    if (view->type == DORA_TYPE_BOOLEAN) {
        key.scalar.boolean = state->value.boolean;
        key.byte_width = 1;
    } else {
        (memcpy)(&key.scalar, state->value.extremum, view->byte_width);
        key.byte_width = view->byte_width;
    }
    return bound_update(scope->pool, scope->entry, scope->committed, &key, nullptr, maximum);
}

struct distinct_pending {
    const struct dora_bytes* descriptors;
    const uint32_t* lanes;
    const uint64_t* offsets;
    uint32_t count;
};

static bool distinct_pending_value_is_equal(struct distinct_pending staged, uint64_t offset, struct dora_bytes value) {
    uint32_t low = 0;
    uint32_t high = staged.count;
    while (low + 1 < high) {
        const uint32_t middle = low + (high - low) / 2;
        if (staged.offsets[middle] <= offset) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const struct dora_bytes pending = staged.descriptors[staged.lanes[low]];
    return pending.byte_count == value.byte_count &&
           (value.byte_count == 0 || memcmp(pending.data, value.data, value.byte_count) == 0);
}

static int
window_distinct_bytes(const struct window_scope* scope, const struct kernel_packet* packet, const uint64_t* hashes) {
    struct statistics_node_accumulator* const entry = scope->entry;
    const struct dora_vector_view* const view = scope->values;
    const struct dora_bytes* const descriptors = packet->values;
    struct statistics_value_hash_slot* const slots = entry->distinct_slots.address;
    const size_t mask = entry->distinct_slot_capacity - 1;
    uint32_t new_lanes[KERNEL_CHUNK_ROW_COUNT];
    uint64_t new_offsets[KERNEL_CHUNK_ROW_COUNT];
    uint32_t new_count = 0;
    size_t pending = 0;
    for (uint32_t word = 0; word * 64 < packet->count; ++word) {
        const uint64_t tail = bitmap_word_tail_mask(packet->count, word);
        uint64_t active = packet->validity == nullptr ? tail : packet->validity[word] & tail;
        while (active != 0) {
            const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);
            const uint64_t hash = hashes[lane];
            const struct dora_bytes value = descriptors[lane];
            for (size_t position = (size_t)hash & mask;; position = (position + 1) & mask) {
                struct statistics_value_hash_slot* const slot = &slots[position];
                if (slot->value_position == UINT64_MAX) {
                    size_t size = 0;
                    if (ckd_add(&size, sizeof(struct statistics_key), value.byte_count) ||
                        ckd_add(&size,
                                size,
                                (SERIALIZED_KEY_ALIGNMENT_BYTES - size % SERIALIZED_KEY_ALIGNMENT_BYTES) %
                                    SERIALIZED_KEY_ALIGNMENT_BYTES) ||
                        ckd_add(&pending, pending, size)) {
                        return EOVERFLOW;
                    }
                    const uint64_t offset = entry->distinct_value_bytes + pending - size;
                    *slot = (struct statistics_value_hash_slot){.hash = hash, .value_position = offset};
                    new_lanes[new_count] = lane;
                    new_offsets[new_count] = offset;
                    ++new_count;
                    break;
                }
                if (slot->hash != hash) {
                    continue;
                }
                bool equal = false;
                if (slot->value_position < entry->distinct_value_bytes) {
                    const struct statistics_key* const stored =
                        (const struct statistics_key*)((const uint8_t*)entry->distinct_values.address +
                                                       slot->value_position);
                    equal = stored->count == value.byte_count &&
                            (value.byte_count == 0 || memcmp(stored + 1, value.data, value.byte_count) == 0);
                } else if (new_count == 0) {
                    return EINVAL;
                } else {
                    equal = distinct_pending_value_is_equal(
                        (struct distinct_pending){
                            .descriptors = descriptors,
                            .lanes = new_lanes,
                            .offsets = new_offsets,
                            .count = new_count
                        },
                        slot->value_position,
                        value);
                }
                if (equal) {
                    break;
                }
            }
        }
    }
    if (new_count == 0) {
        return DORA_OK;
    }
    size_t required = 0;
    if (ckd_add(&required, entry->distinct_value_bytes, pending)) {
        return EOVERFLOW;
    }
    const int status = preserve_reserve(scope->pool,
                                        required,
                                        entry->distinct_value_bytes,
                                        &entry->distinct_values,
                                        &scope->committed->distinct_values);
    if (status != DORA_OK) {
        return status;
    }
    for (uint32_t index = 0; index < new_count; ++index) {
        const struct dora_bytes value = descriptors[new_lanes[index]];
        const uint64_t end = index + 1 < new_count ? new_offsets[index + 1] : required;
        uint8_t* const stored = (uint8_t*)entry->distinct_values.address + new_offsets[index];
        const struct statistics_key key = {
            .serialized_byte_count = end - new_offsets[index],
            .byte_width = value.byte_count,
            .count = value.byte_count,
            .type = view->type,
            .physical_type = view->physical_type,
        };
        memset(stored, 0, key.serialized_byte_count);
        memcpy(stored, &key, sizeof(key));
        if (value.byte_count != 0) {
            memcpy(stored + sizeof(key), value.data, value.byte_count);
        }
    }
    entry->distinct_value_bytes = required;
    entry->distinct_count += new_count;
    return DORA_OK;
}

static int window_distinct(const struct window_scope* scope,
                           const struct window_bindings* bindings,
                           const struct kernel_packet* packet,
                           uint32_t valid) {
    uint64_t hashes[KERNEL_CHUNK_ROW_COUNT];
    struct kernel_result result = {.values = hashes, .row_capacity = packet->count};
    int status = bindings->hash.fill(packet, &bindings->hash, &result);
    struct statistics_node_accumulator* const entry = scope->entry;
    const struct dora_vector_view* const view = scope->values;
    const struct distinct_carrier carrier = statistics_type_is_bytes(view->type)
                                                ? (struct distinct_carrier){}
                                                : distinct_carrier_of(view->type, view->physical_type);
    uint64_t need = 0;
    if (status == DORA_OK) {
        status = distinct_grow(entry, valid, &need);
    }
    if (status == DORA_OK) {
        status = distinct_reserve(scope->pool,
                                  entry,
                                  need,
                                  scope->committed,
                                  carrier.window != nullptr ? carrier.slot_bytes
                                                            : (uint32_t)sizeof(struct statistics_value_hash_slot));
    }
    if (status != DORA_OK) {
        return status;
    }
    if (carrier.window != nullptr) {
        entry->distinct_count = carrier.window(entry->distinct_slots.address,
                                               entry->distinct_slot_capacity,
                                               packet,
                                               hashes,
                                               entry->distinct_count);
        return DORA_OK;
    }
    return window_distinct_bytes(scope, packet, hashes);
}

static int window_packet(const struct window_scope* scope,
                         const struct window_bindings* bindings,
                         const struct kernel_packet* packet,
                         struct dora_range window,
                         uint32_t valid) {
    struct statistics_node_accumulator* const entry = scope->entry;
    if ((scope->requested & DORA_STATISTICS_KIND_NULL_COUNT) != 0) {
        entry->null_count += packet->count - valid;
    }
    int status = DORA_OK;
    if (bindings->widths.update != nullptr) {
        status = window_widths(scope, bindings, packet, window, valid);
    }
    if (status == DORA_OK && bindings->nan.update != nullptr && valid != 0) {
        struct kernel_reduction_state state = {};
        dora_kernel_reduction_state_init(&bindings->nan, &state);
        status = bindings->nan.update(packet, &bindings->nan, &state);
        entry->nan_count += status == DORA_OK ? state.value.count : 0;
    }
    for (uint32_t bound = 0; status == DORA_OK && bound < 2; ++bound) {
        if (bindings->bounds[bound].update == nullptr || valid == 0) {
            continue;
        }
        struct kernel_reduction_state state = {};
        dora_kernel_reduction_state_init(&bindings->bounds[bound], &state);
        status = bindings->bounds[bound].update(packet, &bindings->bounds[bound], &state);
        if (status == DORA_OK && state.has_value) {
            status = window_bound(scope, &state, bound == 1);
        }
    }
    if (status == DORA_OK && bindings->hash.fill != nullptr && valid != 0) {
        status = window_distinct(scope, bindings, packet, valid);
    }
    return status;
}

static int
window_chunk(const struct window_scope* scope, const struct window_bindings* bindings, struct dora_range window) {
    const struct dora_vector_view* const view = &scope->nodes[scope->node].view;
    const uint32_t count = (uint32_t)window.length;
    uint64_t words[KERNEL_CHUNK_WORD_COUNT];
    bitmap_fill_bytes((uint8_t*)words, sizeof words, false);
    struct dora_bytes descriptors[KERNEL_CHUNK_ROW_COUNT];
    struct kernel_packet packet = {
        .values = view->values.data,
        .value_offset = view->offset + window.offset,
        .count = count,
        .shape = KERNEL_SHAPE_FLAT,
        .placement = view->placement,
    };
    uint32_t valid = count;
    if (view->layout == DORA_LAYOUT_NULL) {
        packet.validity = words;
        valid = 0;
    } else if (view->validity.data != nullptr) {
        dora_bitmap_load_words(view->validity.data, view->validity_bit_offset + window.offset, count, words);
        packet.validity = words;
        valid = (uint32_t)dora_bitmap_count_set_words(words, count);
    }
    const bool describe = (bindings->widths.update != nullptr && valid != count) ||
                          bindings->bounds[0].update != nullptr || bindings->bounds[1].update != nullptr ||
                          bindings->hash.fill != nullptr;
    int status = DORA_OK;
    if (bindings->bytes.fill != nullptr && valid != 0 && describe) {
        status = bindings->bytes.fill(&bindings->bytes, window.offset, nullptr, packet.validity, count, descriptors);
        packet.values = descriptors;
        packet.value_offset = 0;
    }
    return status == DORA_OK ? window_packet(scope, bindings, &packet, window, valid) : status;
}

static int
window_update(const struct window_scope* scope, const struct window_bindings* bindings, struct dora_range rows) {
    int status = DORA_OK;
    for (uint64_t first = 0; status == DORA_OK && first < rows.length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range window = {
            .offset = rows.offset + first,
            .length = rows.length - first < KERNEL_CHUNK_ROW_COUNT ? rows.length - first : KERNEL_CHUNK_ROW_COUNT,
        };
        status = window_chunk(scope, bindings, window);
    }
    return status;
}

//--------------------------------------
// Encoded Windows
//--------------------------------------

static bool node_supports_encoded_window(const struct dora_node_topology* nodes,
                                         uint32_t node_count,
                                         uint32_t node,
                                         uint32_t* out_terminal) {
    bool encoded = false;
    for (uint32_t depth = 0; node < node_count && depth <= STATISTICS_MAX_DEPTH; ++depth) {
        const struct dora_vector_view* const view = &nodes[node].view;
        encoded = encoded || view->selection.indices != nullptr;
        if (view->layout != DORA_LAYOUT_DICTIONARY && view->layout != DORA_LAYOUT_RUN_END) {
            *out_terminal = node;
            struct dora_vector_view plain = *view;
            plain.selection = (struct dora_vector_selection){};
            return encoded && view_supports_statistics_window(&plain);
        }
        encoded = true;
        node = statistics_input_child(nodes, node_count, &nodes[node], view->layout == DORA_LAYOUT_DICTIONARY ? 0 : 1);
    }
    return false;
}

static int
encoded_update(const struct window_scope* scope, const struct window_bindings* bindings, struct dora_range rows) {
    struct scratch_scope scratch = {};
    int status = dora_scratch_scope_begin(scope->pool, &scratch);
    struct vector_bound_input input = {};
    if (status == DORA_OK) {
        status = dora_vector_input_from_nodes(&scratch, scope->nodes, scope->node_count, scope->node, &input);
    }
    struct vector_operand operand = {};
    if (status == DORA_OK) {
        status = dora_vector_operand_bind(&scratch, 0, &input, VECTOR_OPERAND_REQUIRE_PACKET, &operand);
    }
    for (uint64_t first = 0; status == DORA_OK && first < rows.length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range window = {
            .offset = rows.offset + first,
            .length = rows.length - first < KERNEL_CHUNK_ROW_COUNT ? rows.length - first : KERNEL_CHUNK_ROW_COUNT,
        };
        struct kernel_packet packet;
        status = dora_vector_operand_prepare(&operand, window, nullptr, &packet);
        const uint32_t valid = status != DORA_OK ? 0
                               : packet.validity == nullptr
                                   ? packet.count
                                   : (uint32_t)dora_bitmap_count_set_words(packet.validity, packet.count);
        if (status == DORA_OK) {
            status = window_packet(scope, bindings, &packet, window, valid);
        }
    }
    dora_scratch_scope_end(&scratch);
    return status;
}

//--------------------------------------
// Ranges
//--------------------------------------

static int range_update(struct statistics_accumulator* accumulator,
                        const struct dora_node_topology* nodes,
                        uint32_t node_count,
                        uint32_t node,
                        struct dora_range rows,
                        bool masked,
                        struct dora_allocation* key_storage,
                        uint32_t depth) {
    if (node >= node_count || depth > STATISTICS_MAX_DEPTH ||
        (!masked && (rows.offset > nodes[node].view.length || rows.length > nodes[node].view.length - rows.offset))) {
        return EINVAL;
    }
    const struct statistics_node_record* const record =
        &((const struct statistics_node_record*)accumulator->nodes.address)[node];
    const uint16_t requested = record->requested_kinds;
    if (rows.length == 0 || (requested == 0 && !record->has_child_demand)) {
        return DORA_OK;
    }
    const struct dora_vector_view* const view = &nodes[node].view;
    struct statistics_node_accumulator* entry = nullptr;
    const struct statistics_node_accumulator* committed = nullptr;
    if (requested != 0) {
        if (record->accumulator_index >= accumulator->value_count || accumulator->values.address == nullptr ||
            !accumulator->has_checkpoint) {
            return EINVAL;
        }
        entry = &((struct statistics_node_accumulator*)accumulator->values.address)[record->accumulator_index];
        committed = &((const struct statistics_node_accumulator*)
                          accumulator->checkpoint_storage.address)[record->accumulator_index];
        if (ckd_add(&entry->row_count, entry->row_count, rows.length) || entry->row_count > INT64_MAX) {
            return EOVERFLOW;
        }
    }
    const bool inspect = entry != nullptr && (requested & ~(uint32_t)DORA_STATISTICS_KIND_ROW_COUNT) != 0;
    if (!inspect && !record->has_child_demand) {
        return DORA_OK;
    }
    if (masked) {
        if (entry != nullptr) {
            if ((requested & DORA_STATISTICS_KIND_NULL_COUNT) != 0) {
                entry->null_count += rows.length;
            }
        }
        if (record->has_child_demand && view->layout == DORA_LAYOUT_STRUCT) {
            int status = DORA_OK;
            for (uint32_t child = nodes[node].first_child_index; status == DORA_OK && child < node_count;
                 child = nodes[child].next_sibling_index) {
                status = range_update(accumulator, nodes, node_count, child, rows, true, key_storage, depth + 1);
            }
            return status;
        }
        return DORA_OK;
    }
    struct dora_schema_field field = {};
    int status = inspect ? dora_schema_get_field(&record->schema, &field) : DORA_OK;
    if (status != DORA_OK) {
        return status;
    }
    struct window_bindings bindings = {};
    const struct window_scope scope = {
        .entry = entry,
        .committed = committed,
        .nodes = nodes,
        .values = view,
        .pool = accumulator->storage.pool,
        .node_count = node_count,
        .node = node,
        .requested = requested,
    };
    if (inspect && !record->has_child_demand && view_supports_statistics_window(view) &&
        window_bind(view, &field.type, requested, &bindings) == DORA_OK) {
        return window_update(&scope, &bindings, rows);
    }
    uint32_t terminal = node;
    if (inspect && !record->has_child_demand && node_supports_encoded_window(nodes, node_count, node, &terminal) &&
        window_bind(&nodes[terminal].view, &field.type, requested, &bindings) == DORA_OK) {
        struct window_scope encoded = scope;
        encoded.values = &nodes[terminal].view;
        encoded.is_encoded = true;
        return encoded_update(&encoded, &bindings, rows);
    }
    const bool payload =
        ((uint32_t)requested & ((uint32_t)DORA_STATISTICS_KIND_MIN | (uint32_t)DORA_STATISTICS_KIND_MAX |
                                (uint32_t)DORA_STATISTICS_KIND_DISTINCT_COUNT)) != 0;
    const bool byte_value = statistics_type_is_bytes(view->type);
    const bool borrowed = payload && byte_value && (requested & DORA_STATISTICS_KIND_DISTINCT_COUNT) == 0;
    const bool serialize = payload && !borrowed && (byte_value || statistics_type_is_nested(view->type));
    const bool width = ((uint32_t)requested & STATISTICS_WIDTH_KINDS) != 0;
    struct value_hasher hasher = {.type = &field.type};
    for (uint64_t offset = 0; inspect && offset < rows.length; ++offset) {
        const uint64_t row = rows.offset + offset;
        size_t size = 0;
        struct statistics_key value = {};
        struct dora_bytes value_bytes = {};
        status = dora_statistics_key_read(accumulator->storage.pool,
                                          nodes,
                                          node_count,
                                          node,
                                          row,
                                          serialize ? key_storage : nullptr,
                                          requested,
                                          &size,
                                          &value,
                                          borrowed ? &value_bytes : nullptr,
                                          depth);
        if (status != DORA_OK) {
            return status;
        }
        const struct statistics_key* const key = serialize ? key_storage->address : &value;
        if ((requested & DORA_STATISTICS_KIND_NULL_COUNT) != 0) {
            entry->null_count += key->is_null ? 1 : 0;
        }
        const bool nan = dora_statistics_key_is_nan(key);
        if ((requested & DORA_STATISTICS_KIND_NAN_COUNT) != 0) {
            entry->nan_count += nan ? 1 : 0;
        }
        if (width) {
            const uint64_t bytes = key->is_null ? 0 : key->byte_width;
            const uint64_t old = entry->total_byte_count.limbs[0];
            entry->total_byte_count.limbs[0] += bytes;
            if (entry->total_byte_count.limbs[0] < old &&
                ckd_add(&entry->total_byte_count.limbs[1], entry->total_byte_count.limbs[1], 1)) {
                return EOVERFLOW;
            }
            if (!key->is_null && bytes < entry->minimum_byte_count) {
                entry->minimum_byte_count = bytes;
            }
            if (bytes > entry->maximum_byte_count) {
                entry->maximum_byte_count = bytes;
            }
        }
        if (payload && !key->is_null) {
            if ((requested & DORA_STATISTICS_KIND_DISTINCT_COUNT) != 0) {
                uint64_t hash = 0;
                status = key_value_hash(&hasher, key, &hash);
                if (status == DORA_OK) {
                    status = distinct_insert(accumulator->storage.pool, entry, committed, key, hash);
                }
            }
            if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_MIN) != 0) {
                status = bound_update(accumulator->storage.pool,
                                      entry,
                                      committed,
                                      key,
                                      borrowed ? &value_bytes : nullptr,
                                      false);
            }
            if (status == DORA_OK && (requested & DORA_STATISTICS_KIND_MAX) != 0) {
                status = bound_update(accumulator->storage.pool,
                                      entry,
                                      committed,
                                      key,
                                      borrowed ? &value_bytes : nullptr,
                                      true);
            }
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    if (!record->has_child_demand || view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END) {
        return DORA_OK;
    }
    const bool list = view->type == DORA_TYPE_LIST || view->type == DORA_TYPE_FIXED_LIST || view->type == DORA_TYPE_MAP;
    struct dora_range pending = {};
    for (uint64_t offset = 0; status == DORA_OK && offset < rows.length;) {
        const uint64_t row = rows.offset + offset;
        const bool is_null = dora_vector_view_is_null(view, row);
        uint64_t end = offset + 1;
        if (view->layout == DORA_LAYOUT_STRUCT) {
            while (end < rows.length && dora_vector_view_is_null(view, rows.offset + end) == is_null) {
                ++end;
            }
            const struct dora_range group = {.offset = row, .length = end - offset};
            for (uint32_t child = nodes[node].first_child_index; status == DORA_OK && child < node_count;
                 child = nodes[child].next_sibling_index) {
                status = range_update(accumulator, nodes, node_count, child, group, is_null, key_storage, depth + 1);
            }
        } else if (!is_null && (view->layout == DORA_LAYOUT_DENSE_UNION || view->layout == DORA_LAYOUT_SPARSE_UNION)) {
            uint64_t child_row = 0;
            const uint32_t member = dora_vector_view_get_member(view, row, &child_row);
            const uint32_t child = statistics_input_child(nodes, node_count, &nodes[node], member);
            status = range_update(accumulator,
                                  nodes,
                                  node_count,
                                  child,
                                  (struct dora_range){.offset = child_row, .length = 1},
                                  false,
                                  key_storage,
                                  depth + 1);
        } else if (!is_null && list) {
            const struct dora_range elements = dora_vector_view_get_elements(view, row);
            if (pending.length != 0 && elements.offset == pending.offset + pending.length) {
                pending.length += elements.length;
            } else {
                for (uint32_t child = nodes[node].first_child_index; status == DORA_OK && child < node_count;
                     child = nodes[child].next_sibling_index) {
                    status =
                        range_update(accumulator, nodes, node_count, child, pending, false, key_storage, depth + 1);
                }
                pending = elements;
            }
        }
        offset = end;
    }
    if (status == DORA_OK && list) {
        for (uint32_t child = nodes[node].first_child_index; status == DORA_OK && child < node_count;
             child = nodes[child].next_sibling_index) {
            status = range_update(accumulator, nodes, node_count, child, pending, false, key_storage, depth + 1);
        }
    }
    return status;
}

int dora_statistics_checkpoint_begin(struct statistics_accumulator* accumulator,
                                     struct statistics_checkpoint* out_checkpoint) {
    if (accumulator == nullptr || out_checkpoint == nullptr || out_checkpoint->values.address != nullptr ||
        out_checkpoint->values.capacity_bytes != 0 || out_checkpoint->row_count != 0) {
        return EINVAL;
    }
    if (accumulator->is_running) {
        return EBUSY;
    }
    accumulator->is_running = true;
    const size_t bytes = (size_t)accumulator->value_count * sizeof(struct statistics_node_accumulator);
    out_checkpoint->values = accumulator->checkpoint_storage;
    if (bytes != 0) {
        memcpy(out_checkpoint->values.address, accumulator->values.address, bytes);
    }
    out_checkpoint->row_count = accumulator->row_count;
    accumulator->has_checkpoint = true;
    return DORA_OK;
}

void dora_statistics_checkpoint_commit(struct statistics_accumulator* accumulator,
                                       struct statistics_checkpoint* checkpoint) {
    if (accumulator == nullptr || checkpoint == nullptr || !accumulator->is_running || !accumulator->has_checkpoint ||
        accumulator->checkpoint_storage.address != checkpoint->values.address) {
        return;
    }
    const struct statistics_node_accumulator* const old = checkpoint->values.address;
    const struct statistics_node_accumulator* const current = accumulator->values.address;
    for (uint32_t index = 0; index < accumulator->value_count; ++index) {
        release_replaced(old[index].minimum, current[index].minimum);
        release_replaced(old[index].maximum, current[index].maximum);
        release_replaced(old[index].distinct_values, current[index].distinct_values);
        release_replaced(old[index].distinct_slots, current[index].distinct_slots);
    }
    accumulator->has_checkpoint = false;
    *checkpoint = (struct statistics_checkpoint){};
    accumulator->is_running = false;
}

void dora_statistics_checkpoint_rollback(struct statistics_accumulator* accumulator,
                                         struct statistics_checkpoint* checkpoint) {
    if (accumulator == nullptr || checkpoint == nullptr || !accumulator->is_running || !accumulator->has_checkpoint ||
        accumulator->checkpoint_storage.address != checkpoint->values.address) {
        return;
    }
    const struct statistics_node_accumulator* const old = checkpoint->values.address;
    struct statistics_node_accumulator* const current = accumulator->values.address;
    for (uint32_t index = 0; index < accumulator->value_count; ++index) {
        release_replaced(current[index].minimum, old[index].minimum);
        release_replaced(current[index].maximum, old[index].maximum);
        release_replaced(current[index].distinct_values, old[index].distinct_values);
        release_replaced(current[index].distinct_slots, old[index].distinct_slots);
        current[index] = old[index];
        const size_t stride = current[index].distinct_slot_bytes;
        const uint64_t limit = stride > sizeof(struct statistics_value_hash_slot) ? current[index].distinct_count
                                                                                  : current[index].distinct_value_bytes;
        uint8_t* const slots = current[index].distinct_slots.address;
        for (size_t slot = 0; slot < current[index].distinct_slot_capacity; ++slot) {
            struct statistics_value_hash_slot header = {};
            memcpy(&header, slots + slot * stride, sizeof(header));
            if (header.value_position >= limit) {
                header = (struct statistics_value_hash_slot){.value_position = UINT64_MAX};
                memcpy(slots + slot * stride, &header, sizeof(header));
            }
        }
    }
    accumulator->row_count = checkpoint->row_count;
    accumulator->has_checkpoint = false;
    *checkpoint = (struct statistics_checkpoint){};
    accumulator->is_running = false;
}

int dora_statistics_accumulator_update_nodes(struct statistics_accumulator* accumulator,
                                             const struct dora_node_topology* nodes,
                                             uint32_t node_count,
                                             uint32_t node,
                                             struct dora_range rows) {
    if (accumulator == nullptr || !accumulator->is_running || nodes == nullptr ||
        node_count != accumulator->node_count || node >= node_count || rows.offset > nodes[node].view.length ||
        rows.length > nodes[node].view.length - rows.offset) {
        return EINVAL;
    }
    struct dora_allocation key_storage = {};
    int status = range_update(accumulator, nodes, node_count, node, rows, false, &key_storage, 0);
    if (status == DORA_OK && node == 0 &&
        (ckd_add(&accumulator->row_count, accumulator->row_count, rows.length) || accumulator->row_count > INT64_MAX)) {
        status = EOVERFLOW;
    }
    dora_allocation_release(&key_storage);
    return status;
}

int dora_statistics_accumulator_update(struct statistics_accumulator* accumulator,
                                       const struct dora_vector* appended_chunk) {
    if (accumulator == nullptr || !dora_vector_is_valid(appended_chunk)) {
        return EINVAL;
    }
    struct statistics_checkpoint checkpoint = {};
    int status = dora_statistics_checkpoint_begin(accumulator, &checkpoint);
    if (status != DORA_OK) {
        return status;
    }
    struct dora_vector retained = {};
    status = dora_vector_retain(appended_chunk, &retained);
    struct dora_allocation nodes = {};
    uint32_t count = 0;
    if (status == DORA_OK) {
        status = dora_statistics_bind_input(accumulator->storage.pool,
                                            &retained,
                                            accumulator->nodes.address,
                                            accumulator->node_count,
                                            &nodes,
                                            &count);
    }
    const struct dora_node_topology* const inputs = nodes.address;
    if (status == DORA_OK) {
        status = dora_statistics_accumulator_update_nodes(accumulator,
                                                          inputs,
                                                          count,
                                                          0,
                                                          (struct dora_range){.length = inputs[0].view.length});
    }
    for (uint32_t node = 1; status == DORA_OK && node < count; ++node) {
        const enum dora_layout layout = inputs[inputs[node].parent_node_index].view.layout;
        if (layout == DORA_LAYOUT_DICTIONARY || layout == DORA_LAYOUT_RUN_END) {
            uint64_t length = inputs[node].view.length;
            if (layout == DORA_LAYOUT_RUN_END && inputs[node].child_index == 1) {
                const uint32_t ends = statistics_input_child(inputs, count, &inputs[inputs[node].parent_node_index], 0);
                if (ends >= count || inputs[ends].view.length > length) {
                    status = EINVAL;
                    break;
                }
                length = inputs[ends].view.length;
            }
            status = dora_statistics_accumulator_update_nodes(accumulator,
                                                              inputs,
                                                              count,
                                                              node,
                                                              (struct dora_range){.length = length});
        }
    }
    dora_allocation_release(&nodes);
    dora_vector_release(&retained);
    if (status == DORA_OK) {
        dora_statistics_checkpoint_commit(accumulator, &checkpoint);
    } else {
        dora_statistics_checkpoint_rollback(accumulator, &checkpoint);
    }
    return status;
}

static enum dora_statistics_kind statistic_kind(const char* name) {
    if (strcmp(name, "row_count") == 0) {
        return DORA_STATISTICS_KIND_ROW_COUNT;
    }
    if (strcmp(name, "null_count") == 0) {
        return DORA_STATISTICS_KIND_NULL_COUNT;
    }
    if (strcmp(name, "nan_count") == 0) {
        return DORA_STATISTICS_KIND_NAN_COUNT;
    }
    if (strcmp(name, "distinct_count") == 0) {
        return DORA_STATISTICS_KIND_DISTINCT_COUNT;
    }
    if (strcmp(name, "min") == 0 || strcmp(name, "is_min_exact") == 0) {
        return DORA_STATISTICS_KIND_MIN;
    }
    if (strcmp(name, "max") == 0 || strcmp(name, "is_max_exact") == 0) {
        return DORA_STATISTICS_KIND_MAX;
    }
    if (strcmp(name, "min_byte_width") == 0) {
        return DORA_STATISTICS_KIND_MIN_BYTE_WIDTH;
    }
    if (strcmp(name, "max_byte_width") == 0) {
        return DORA_STATISTICS_KIND_MAX_BYTE_WIDTH;
    }
    return DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH;
}

static int prepare_bound(const struct statistics_node_accumulator* entry,
                         bool maximum,
                         size_t* out_count,
                         bool* out_available,
                         bool* out_exact) {
    const struct statistics_key* const key = accumulator_bound(entry, maximum);
    *out_available = *out_exact = key != nullptr && !key->is_null;
    *out_count = 0;
    if (!*out_available) {
        return DORA_OK;
    }
    if (key->count > SIZE_MAX) {
        return EOVERFLOW;
    }
    const uint32_t cap = maximum ? entry->maximum_bound_bytes : entry->minimum_bound_bytes;
    const size_t budget = cap == 0 ? (size_t)key->count : cap;
    if (key->type == DORA_TYPE_UTF8) {
        return dora_utf8_measure_bound(
            (struct dora_utf8_input){.bytes = {(const uint8_t*)(key + 1), (size_t)key->count}},
            maximum ? DORA_UTF8_BOUND_UPPER : DORA_UTF8_BOUND_LOWER,
            budget,
            out_count,
            out_available,
            out_exact);
    }
    *out_count = (size_t)key->count;
    if (key->type != DORA_TYPE_BINARY || key->count <= budget) {
        return DORA_OK;
    }
    *out_exact = false;
    *out_count = budget;
    if (maximum) {
        const uint8_t* const bytes = (const uint8_t*)(key + 1);
        while (*out_count != 0 && bytes[*out_count - 1] == UINT8_MAX) {
            --*out_count;
        }
        *out_available = *out_count != 0;
    }
    return DORA_OK;
}

static int truncated_bound(struct dora_pool pool,
                           const struct statistics_node_accumulator* entry,
                           bool maximum,
                           size_t count,
                           bool available,
                           bool exact,
                           struct dora_allocation* out_storage,
                           const struct statistics_key** out_key) {
    const struct statistics_key* const bound = accumulator_bound(entry, maximum);
    *out_key = available ? bound : nullptr;
    if (!available || exact) {
        return DORA_OK;
    }
    const uint32_t cap = maximum ? entry->maximum_bound_bytes : entry->minimum_bound_bytes;
    const size_t capacity = bound->type == DORA_TYPE_UTF8 ? cap : count;
    size_t size = 0;
    if (ckd_add(&size, sizeof(*bound), capacity)) {
        return EOVERFLOW;
    }
    int status = dora_memory_payload_allocate(pool, size, out_storage);
    if (status != DORA_OK) {
        return status;
    }
    struct statistics_key* const key = out_storage->address;
    *key = *bound;
    key->count = count;
    key->byte_width = count;
    key->serialized_byte_count = sizeof(*key) + count;
    uint8_t* const bytes = (uint8_t*)(key + 1);
    const uint8_t* const source = (const uint8_t*)(bound + 1);
    if (bound->type == DORA_TYPE_UTF8) {
        size_t written;
        bool result_available, result_exact;
        status = dora_utf8_make_bound(
            (struct dora_utf8_input){.bytes = {source, (size_t)bound->count}, .proof = bound->utf8_proof},
            maximum ? DORA_UTF8_BOUND_UPPER : DORA_UTF8_BOUND_LOWER,
            (struct dora_mutable_bytes){bytes, capacity},
            &written,
            &result_available,
            &result_exact);
    } else {
        if (count != 0) {
            memcpy(bytes, source, count);
        }
        if (maximum) {
            ++bytes[count - 1];
        }
    }
    if (status == DORA_OK) {
        *out_key = key;
    }
    return status;
}

static int emit_kind(struct dora_pool pool,
                     const struct dora_schema* schema,
                     const struct statistics_node_accumulator* entry,
                     enum dora_statistics_kind kind,
                     const size_t bound_sizes[static 2],
                     const bool bound_available[static 2],
                     const bool bound_exact[static 2],
                     struct dora_vector* out_vector,
                     struct dora_error* out_error) {
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status != DORA_OK) {
        return status;
    }
    const bool bound = kind == DORA_STATISTICS_KIND_MIN || kind == DORA_STATISTICS_KIND_MAX;
    if (bound) {
        const bool maximum = kind == DORA_STATISTICS_KIND_MAX;
        if (strcmp(field.name, "is_min_exact") == 0 || strcmp(field.name, "is_max_exact") == 0) {
            const bool exact = bound_available[maximum] && bound_exact[maximum];
            const struct statistics_key flag =
                {.serialized_byte_count = sizeof(flag), .type = DORA_TYPE_BOOLEAN, .scalar.boolean = exact};
            return dora_statistics_key_materialize(pool, schema, &flag, out_vector, out_error);
        }
        struct dora_allocation storage = {};
        const struct statistics_key* key = nullptr;
        status = truncated_bound(pool,
                                 entry,
                                 maximum,
                                 bound_sizes[maximum],
                                 bound_available[maximum],
                                 bound_exact[maximum],
                                 &storage,
                                 &key);
        if (status == DORA_OK) {
            status = dora_statistics_key_materialize(pool, schema, key, out_vector, out_error);
        }
        dora_allocation_release(&storage);
        return status;
    }
    struct statistics_key key = {
        .serialized_byte_count = sizeof(key),
        .type = field.type.id,
        .physical_type = field.type.id == DORA_TYPE_FLOAT64 ? DORA_PHYSICAL_TYPE_F64 : DORA_PHYSICAL_TYPE_I64
    };
    uint64_t value = 0;
    switch (kind) {
    case DORA_STATISTICS_KIND_ROW_COUNT:
        value = entry->row_count;
        break;
    case DORA_STATISTICS_KIND_NULL_COUNT:
        value = entry->null_count;
        break;
    case DORA_STATISTICS_KIND_DISTINCT_COUNT:
        value = entry->distinct_count;
        break;
    case DORA_STATISTICS_KIND_NAN_COUNT:
        value = entry->nan_count;
        break;
    case DORA_STATISTICS_KIND_MIN_BYTE_WIDTH:
        key.is_null = entry->minimum_byte_count == UINT64_MAX;
        value = key.is_null ? 0 : entry->minimum_byte_count;
        break;
    case DORA_STATISTICS_KIND_MAX_BYTE_WIDTH:
        value = entry->maximum_byte_count;
        break;
    case DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH:
        key.is_null = entry->row_count == 0;
        key.scalar.f64 =
            entry->row_count == 0
                ? 0
                : (ldexp((double)entry->total_byte_count.limbs[1], 64) + (double)entry->total_byte_count.limbs[0]) /
                      (double)entry->row_count;
        break;
    default:
        return EINVAL;
    }
    if (field.type.id != DORA_TYPE_FLOAT64) {
        if (value > INT64_MAX) {
            return EOVERFLOW;
        }
        key.scalar.i64 = value;
    }
    return dora_statistics_key_materialize(pool, schema, &key, out_vector, out_error);
}

int dora_statistics_accumulator_finish(struct statistics_accumulator* accumulator,
                                       struct dora_vector* out_statistics,
                                       struct dora_error* out_error) {
    if (accumulator == nullptr || !dora_vector_is_empty(out_statistics)) {
        return EINVAL;
    }
    if (accumulator->is_running && !accumulator->has_checkpoint) {
        return EBUSY;
    }
    const bool was_running = accumulator->is_running;
    accumulator->is_running = true;
    const struct dora_pool pool = accumulator->storage.pool;
    const struct dora_schema* const schema = &accumulator->output_schema;
    struct dora_allocation group_storage = {};
    struct dora_vector published = {};
    int status = statistics_allocate_zero(pool, accumulator->node_count, sizeof(struct dora_vector), &group_storage);
    struct dora_vector* const groups = group_storage.address;
    const struct statistics_node_record* const nodes = accumulator->nodes.address;
    const struct statistics_node_accumulator* const values = accumulator->values.address;
    for (uint32_t node = 0; status == DORA_OK && node < accumulator->node_count; ++node) {
        struct dora_schema group = {};
        struct dora_schema_field group_field = {};
        status = dora_schema_get_child(schema, node, &group);
        if (status == DORA_OK) {
            status = dora_schema_get_field(&group, &group_field);
        }
        struct dora_vector children[STATISTICS_GROUP_FIELD_CAPACITY] = {};
        size_t bound_sizes[2] = {};
        bool bound_available[2] = {}, bound_exact[2] = {}, bound_prepared[2] = {};
        if (status == DORA_OK && group_field.child_count > STATISTICS_GROUP_FIELD_CAPACITY) {
            status = EINVAL;
        }
        for (uint32_t child = 0; status == DORA_OK && child < group_field.child_count; ++child) {
            struct dora_schema member = {};
            struct dora_schema_field field = {};
            status = dora_schema_get_child(&group, child, &member);
            if (status == DORA_OK) {
                status = dora_schema_get_field(&member, &field);
            }
            if (status != DORA_OK) {
                break;
            }
            const enum dora_statistics_kind kind = statistic_kind(field.name);
            const struct statistics_node_accumulator* const entry = &values[nodes[node].accumulator_index];
            if ((nodes[node].requested_kinds & kind) == 0) {
                status = EINVAL;
            }
            if (status == DORA_OK && (kind == DORA_STATISTICS_KIND_MIN || kind == DORA_STATISTICS_KIND_MAX)) {
                const bool maximum = kind == DORA_STATISTICS_KIND_MAX;
                if (!bound_prepared[maximum]) {
                    status = prepare_bound(entry,
                                           maximum,
                                           &bound_sizes[maximum],
                                           &bound_available[maximum],
                                           &bound_exact[maximum]);
                    bound_prepared[maximum] = status == DORA_OK;
                }
            }
            if (status == DORA_OK) {
                status = emit_kind(pool,
                                   &member,
                                   entry,
                                   kind,
                                   bound_sizes,
                                   bound_available,
                                   bound_exact,
                                   &children[child],
                                   out_error);
            }
        }
        if (status == DORA_OK) {
            struct dora_allocation validity = {};
            const struct vector_input input = {
                .field = &group_field,
                .admitted_schema = &group,
                .row_count = 1,
                .allocations = &validity,
                .buffer_count = 1,
                .children = children,
                .child_count = group_field.child_count
            };
            status = dora_vector_init_move_input(&groups[node], pool, &input, &validity, out_error);
        }
        for (uint32_t child = 0; child < STATISTICS_GROUP_FIELD_CAPACITY; ++child) {
            dora_vector_release(&children[child]);
        }
    }
    struct dora_schema_field field = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(schema, &field);
    }
    if (status == DORA_OK) {
        struct dora_allocation validity = {};
        const struct vector_input input = {
            .field = &field,
            .admitted_schema = schema,
            .row_count = 1,
            .allocations = &validity,
            .buffer_count = 1,
            .children = groups,
            .child_count = accumulator->node_count
        };
        status = dora_vector_init_move_input(&published, pool, &input, &validity, out_error);
    }
    for (uint32_t node = 0; groups != nullptr && node < accumulator->node_count; ++node) {
        dora_vector_release(&groups[node]);
    }
    dora_allocation_release(&group_storage);
    accumulator->is_running = was_running;
    return vector_finish_publication(&published, out_statistics, status);
}

void dora_statistics_accumulator_release(struct statistics_accumulator** in_out_accumulator) {
    if (in_out_accumulator == nullptr || *in_out_accumulator == nullptr) {
        return;
    }
    struct statistics_accumulator* const accumulator = *in_out_accumulator;
    *in_out_accumulator = nullptr;
    struct statistics_node_accumulator* const values = accumulator->values.address;
    for (uint32_t index = 0; index < accumulator->value_count; ++index) {
        dora_allocation_release(&values[index].distinct_slots);
        dora_allocation_release(&values[index].distinct_values);
        dora_allocation_release(&values[index].minimum);
        dora_allocation_release(&values[index].maximum);
    }
    dora_allocation_release(&accumulator->nodes);
    dora_allocation_release(&accumulator->values);
    dora_allocation_release(&accumulator->checkpoint_storage);
    dora_schema_release(&accumulator->output_schema);
    dora_schema_release(&accumulator->schema);
    struct dora_allocation storage = accumulator->storage;
    dora_allocation_release(&storage);
}

int dora_statistics_accumulator_open(struct dora_pool pool,
                                     const struct dora_schema* schema,
                                     const struct dora_statistics_request* requests,
                                     uint32_t request_count,
                                     struct statistics_accumulator** out_accumulator) {
    if (out_accumulator == nullptr || *out_accumulator != nullptr || !dora_schema_is_valid(schema) ||
        (request_count != 0 && requests == nullptr)) {
        return EINVAL;
    }
    struct dora_schema result_schema = {};
    int status = dora_statistics_schema_init_in_pool(&result_schema, pool, schema, requests, request_count);
    if (status != DORA_OK) {
        dora_schema_release(&result_schema);
        return status;
    }
    struct dora_allocation storage = {};
    status = dora_memory_payload_allocate(pool, sizeof(struct statistics_accumulator), &storage);
    if (status != DORA_OK) {
        dora_schema_release(&result_schema);
        return status;
    }
    struct statistics_accumulator* accumulator = storage.address;
    *accumulator = (struct statistics_accumulator){.storage = storage};
    status = dora_schema_move(&result_schema, &accumulator->output_schema);
    dora_schema_release(&result_schema);
    if (status == DORA_OK) {
        status = dora_identity_claim_acquire(pool, &schema->handle, IDENTITY_KIND_SCHEMA, &accumulator->schema.handle);
    }
    accumulator->schema.node_index = schema->node_index;
    struct dora_allocation walk = {};
    if (status == DORA_OK) {
        status = dora_schema_walk(pool, schema, &walk, &accumulator->node_count);
    }
    size_t node_bytes = 0;
    if (status == DORA_OK && ckd_mul(&node_bytes, accumulator->node_count, sizeof(struct statistics_node_record))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, node_bytes, &accumulator->nodes);
    }
    struct statistics_node_record* const nodes = accumulator->nodes.address;
    uint32_t value_count = 0;
    size_t value_bytes = 0;
    if (status == DORA_OK) {
        memset(nodes, 0, node_bytes);
        for (uint32_t index = 0; index < request_count; ++index) {
            nodes[requests[index].schema_node_index].requested_kinds |= requests[index].kinds;
        }
        const struct schema_walk_node* const schemas = walk.address;
        for (uint32_t node = 0; node < accumulator->node_count; ++node) {
            nodes[node].accumulator_index = nodes[node].requested_kinds == 0 ? UINT32_MAX : value_count++;
            nodes[node].schema = schemas[node].schema;
            nodes[node].parent_node_index = schemas[node].parent_node_index;
            nodes[node].child_index = schemas[node].child_index;
        }
        if (ckd_mul(&value_bytes, value_count, sizeof(struct statistics_node_accumulator))) {
            status = EOVERFLOW;
        }
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, value_bytes, &accumulator->values);
    }
    if (status == DORA_OK && value_bytes != 0) {
        status = dora_memory_payload_allocate(pool, value_bytes, &accumulator->checkpoint_storage);
    }
    if (status == DORA_OK) {
        if (value_bytes != 0) {
            memset(accumulator->values.address, 0, value_bytes);
        }
        accumulator->value_count = value_count;
        struct statistics_node_accumulator* const values = accumulator->values.address;
        for (uint32_t index = 0; index < value_count; ++index) {
            values[index].minimum_byte_count = UINT64_MAX;
        }
        for (uint32_t cursor = request_count; cursor != 0;) {
            const struct dora_statistics_request* const request = &requests[--cursor];
            if (request->kinds == 0) {
                continue;
            }
            struct statistics_node_accumulator* const entry =
                &values[nodes[request->schema_node_index].accumulator_index];
            if ((request->kinds & DORA_STATISTICS_KIND_MIN) != 0) {
                entry->minimum_bound_bytes = request->max_bound_bytes;
            }
            if ((request->kinds & DORA_STATISTICS_KIND_MAX) != 0) {
                entry->maximum_bound_bytes = request->max_bound_bytes;
            }
        }
        for (uint32_t cursor = accumulator->node_count; status == DORA_OK && cursor > 1;) {
            const uint32_t node = --cursor;
            if (nodes[node].requested_kinds == 0 && !nodes[node].has_child_demand) {
                continue;
            }
            struct dora_schema_field parent = {};
            status = dora_schema_get_field(&nodes[nodes[node].parent_node_index].schema, &parent);
            if (status == DORA_OK && parent.layout != DORA_LAYOUT_DICTIONARY && parent.layout != DORA_LAYOUT_RUN_END) {
                nodes[nodes[node].parent_node_index].has_child_demand = true;
            }
        }
    }
    dora_allocation_release(&walk);
    if (status != DORA_OK) {
        dora_statistics_accumulator_release(&accumulator);
        return status;
    }
    *out_accumulator = accumulator;
    return DORA_OK;
}
