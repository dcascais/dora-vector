// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Statistics accumulation and publication.

#ifndef DORA_STATISTICS_STATISTICS_INTERNAL_H
#define DORA_STATISTICS_STATISTICS_INTERNAL_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "utf8/input.h"
#include "vector/node_topology.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr uint32_t STATISTICS_MAX_DEPTH = 128;
static constexpr uint32_t STATISTICS_KIND_COUNT = 9;
static constexpr uint32_t STATISTICS_GROUP_FIELD_CAPACITY = 2 * STATISTICS_KIND_COUNT + 2;
static constexpr uint16_t STATISTICS_KIND_MASK =
    (uint32_t)DORA_STATISTICS_KIND_AVERAGE_BYTE_WIDTH | (uint32_t)DORA_STATISTICS_KIND_DISTINCT_COUNT |
    (uint32_t)DORA_STATISTICS_KIND_MAX_BYTE_WIDTH | (uint32_t)DORA_STATISTICS_KIND_MAX |
    (uint32_t)DORA_STATISTICS_KIND_MIN | (uint32_t)DORA_STATISTICS_KIND_NULL_COUNT |
    (uint32_t)DORA_STATISTICS_KIND_ROW_COUNT | (uint32_t)DORA_STATISTICS_KIND_MIN_BYTE_WIDTH |
    (uint32_t)DORA_STATISTICS_KIND_NAN_COUNT;

struct statistics_descriptor {
    const char* name;
    const char* estimate_name;
    enum dora_statistics_kind kind;
    enum dora_type type;
};

extern const struct statistics_descriptor dora_statistics_descriptors[STATISTICS_KIND_COUNT];

//------------------------------------------------------------------------------
//
// Statistic Values
//
//------------------------------------------------------------------------------

union statistics_scalar_value {
    bool boolean;
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
    struct dora_f16_bits f16;
    float f32;
    double f64;
    struct dora_interval_day_time interval_day_time;
    struct dora_interval_month_day_nano interval_month_day_nano;
};

struct statistics_key {
    union statistics_scalar_value scalar;
    size_t serialized_byte_count;
    uint64_t byte_width;
    uint64_t count;
    enum dora_type type;
    enum dora_physical_type physical_type;
    uint32_t member_index;
    bool is_null;
    enum dora_utf8_proof utf8_proof;
};

//------------------------------------------------------------------------------
//
// Distinct Values
//
//------------------------------------------------------------------------------

struct statistics_value_hash_slot {
    uint64_t hash;
    uint64_t value_position;
};

//------------------------------------------------------------------------------
//
// Kind Accumulators
//
//------------------------------------------------------------------------------

struct statistics_node_accumulator {
    struct dora_allocation minimum;
    struct dora_allocation maximum;
    struct statistics_key minimum_key;
    struct statistics_key maximum_key;
    struct dora_allocation distinct_slots;
    struct dora_allocation distinct_values;
    struct dora_u128 total_byte_count;
    uint64_t row_count;
    uint64_t null_count;
    uint64_t distinct_count;
    uint64_t nan_count;
    uint64_t minimum_byte_count;
    uint64_t maximum_byte_count;
    size_t distinct_slot_capacity;
    size_t distinct_value_bytes;
    uint32_t distinct_slot_bytes;
    uint32_t minimum_bound_bytes;
    uint32_t maximum_bound_bytes;
};

struct statistics_node_record {
    struct dora_schema schema;
    uint32_t parent_node_index;
    uint32_t child_index;
    uint32_t accumulator_index;
    uint16_t requested_kinds;
    bool has_child_demand;
};

//------------------------------------------------------------------------------
//
// Accumulator Storage
//
//------------------------------------------------------------------------------

struct statistics_accumulator {
    struct dora_allocation storage;
    struct dora_schema schema;
    struct dora_schema output_schema;
    struct dora_allocation nodes;
    struct dora_allocation values;
    struct dora_allocation checkpoint_storage;
    uint64_t row_count;
    uint32_t node_count;
    uint32_t value_count;
    bool has_checkpoint;
    bool is_running;
};

struct statistics_checkpoint {
    struct dora_allocation values;
    uint64_t row_count;
};

int dora_statistics_checkpoint_begin(struct statistics_accumulator* accumulator,
                                     struct statistics_checkpoint* out_checkpoint);
void dora_statistics_checkpoint_commit(struct statistics_accumulator* accumulator,
                                       struct statistics_checkpoint* checkpoint);
void dora_statistics_checkpoint_rollback(struct statistics_accumulator* accumulator,
                                         struct statistics_checkpoint* checkpoint);
int dora_statistics_accumulator_update_nodes(struct statistics_accumulator* accumulator,
                                             const struct dora_node_topology* nodes,
                                             uint32_t node_count,
                                             uint32_t node,
                                             struct dora_range rows);
int dora_statistics_key_compare(const struct statistics_key* left, const struct statistics_key* right, int* out_order);
int dora_statistics_key_materialize(struct dora_pool pool,
                                    const struct dora_schema* schema,
                                    const struct statistics_key* key,
                                    struct dora_vector* out_vector,
                                    struct dora_error* out_error);
int dora_statistics_key_serialize_row(struct dora_pool pool,
                                      const struct dora_vector* vector,
                                      uint64_t row,
                                      struct dora_allocation* out_key);
bool dora_statistics_key_is_nan(const struct statistics_key* key);

static constexpr size_t SERIALIZED_KEY_ALIGNMENT_BYTES = 8;

static constexpr size_t DISTINCT_LOAD_DENOMINATOR = 2;

static inline uint32_t statistics_input_child(const struct dora_node_topology* nodes,
                                              uint32_t count,
                                              const struct dora_node_topology* parent,
                                              uint32_t child) {
    for (uint32_t index = parent->first_child_index; index < count; index = nodes[index].next_sibling_index) {
        if (nodes[index].child_index == child) {
            return index;
        }
    }
    return UINT32_MAX;
}

static inline int statistics_allocate_zero(struct dora_pool pool,
                                           uint64_t element_count,
                                           size_t element_byte_width,
                                           struct dora_allocation* out_allocation) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, element_count, element_byte_width)) {
        return EOVERFLOW;
    }
    const int status = dora_pool_allocate(pool, bytes, out_allocation);
    if (status == DORA_OK && out_allocation->capacity_bytes != 0) {
        memset(out_allocation->address, 0, out_allocation->capacity_bytes);
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
                             uint32_t depth);

uint64_t dora_statistics_key_hash(const struct statistics_key* key, uint64_t seed);

int dora_statistics_bind_input(struct dora_pool pool,
                               const struct dora_vector* vector,
                               const struct statistics_node_record* expected,
                               uint32_t expected_node_count,
                               struct dora_allocation* out_nodes,
                               uint32_t* out_node_count);

//------------------------------------------------------------------------------
//
// Incremental Accumulation
//
//------------------------------------------------------------------------------

int dora_statistics_accumulator_open(struct dora_pool pool,
                                     const struct dora_schema* schema,
                                     const struct dora_statistics_request* requests,
                                     uint32_t request_count,
                                     struct statistics_accumulator** out_accumulator);

int dora_statistics_accumulator_update(struct statistics_accumulator* accumulator,
                                       const struct dora_vector* appended_chunk);

int dora_statistics_accumulator_finish(struct statistics_accumulator* accumulator,
                                       struct dora_vector* out_statistics,
                                       struct dora_error* out_error);

void dora_statistics_accumulator_release(struct statistics_accumulator** in_out_accumulator);

int dora_statistics_schema_init_in_pool(struct dora_schema* schema,
                                        struct dora_pool pool,
                                        const struct dora_schema* target,
                                        const struct dora_statistics_request* requests,
                                        uint32_t request_count);

//------------------------------------------------------------------------------
//
// Operations with Diagnostics
//
//------------------------------------------------------------------------------

int dora_statistics_attach_error(struct dora_pool pool,
                                 const struct dora_vector* data,
                                 const struct dora_vector* statistics,
                                 struct dora_vector* out_vector,
                                 struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(alignof(struct statistics_key) <= SERIALIZED_KEY_ALIGNMENT_BYTES);

#endif
