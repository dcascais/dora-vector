// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Builder transactions and storage.

#ifndef DORA_BUILDER_STATE_H
#define DORA_BUILDER_STATE_H

#include "dora/vector/builder.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "identity/identity.h"
#include "statistics/statistics_internal.h"
#include "vector/image.h"
#include "vector/node_topology.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Mutable Buffers
//
//------------------------------------------------------------------------------

enum builder_buffer_role : uint8_t {
    BUILDER_BUFFER_VALUE_BITS,
    BUILDER_BUFFER_VALIDITY_BITS,
    BUILDER_BUFFER_FIXED_VALUES,
    BUILDER_BUFFER_OFFSETS32,
    BUILDER_BUFFER_OFFSETS64,
    BUILDER_BUFFER_BINARY_VIEW,
    BUILDER_BUFFER_SIZES32,
    BUILDER_BUFFER_SIZES64,
    BUILDER_BUFFER_TYPE_IDS,
    BUILDER_BUFFER_MEMBER_ROWS,
    BUILDER_BUFFER_BYTES,
    BUILDER_BUFFER_VARIADIC_SIZES,
    BUILDER_BUFFER_CHILD_FRONTIERS
};

struct builder_buffer_state {
    struct dora_allocation allocation;
    size_t next_capacity_bytes;
    uint64_t initialized_unit_count;
    uint32_t unit_bit_width;
    uint32_t next_buffer;
    enum builder_buffer_role role;
};

//------------------------------------------------------------------------------
//
// Row Checkpoints
//
//------------------------------------------------------------------------------

struct builder_row_state {
    enum vector_content_state content_state;
    uint64_t row_count;
    uint64_t closed_row_count;
    uint64_t stable_row_count;
    uint64_t child_row_count;
    uint64_t byte_count;
    uint32_t data_buffer_count;
    uint32_t last_data_buffer;
    int64_t null_count;
    bool has_masked_nulls;
};

struct builder_row_checkpoint {
    struct builder_row_state rows;
    uint64_t first_changed_run;
    struct dora_allocation inactive;
    uint64_t inactive_first;
    uint64_t inactive_end;
    bool is_active;
};

struct builder_buffer_checkpoint {
    struct dora_allocation bound;
    struct dora_allocation saved;
    uint64_t saved_first;
    uint64_t initialized_unit_count;
    uint64_t reached_unit_count;
    bool is_active;
};

//------------------------------------------------------------------------------
//
// Node State
//
//------------------------------------------------------------------------------

struct builder_node_state {
    struct dora_schema schema;
    const struct dora_schema_field* field;
    struct dora_allocation extension_parameters;
    const struct dora_extension_definition* extension;
    uint32_t comparison_flags;
    uint32_t first_buffer;
    uint32_t buffer_count;
    uint32_t frontier_buffer;
    struct dora_vector dictionary;
    struct builder_row_state rows;
    bool is_run_eligible;
};

//------------------------------------------------------------------------------
//
// Mutations and Windows
//
//------------------------------------------------------------------------------

// Retained while the operation is parked.
struct builder_operation_data {
    struct dora_builder target;
    size_t written_byte_count;
    struct statistics_checkpoint statistics;
    uint32_t touched_node_count;
    uint32_t touched_buffer_count;
    uint32_t bound_buffer_count;
};

struct builder_window {
    struct builder_operation_data operation;
    uint64_t row_count;
    uint64_t first_unit;
    size_t byte_count;
    uint32_t buffer_index;
};

static constexpr uint64_t BUILDER_RUN_ROW_LIMIT = 2048;

enum builder_run_kind : uint8_t {
    BUILDER_RUN_VALUES,
    BUILDER_RUN_BITS,
    BUILDER_RUN_BYTES,
    BUILDER_RUN_VIEW,
    BUILDER_RUN_FIXED_BINARY
};

struct builder_content_bound {
    struct dora_i256 bound;
    int status;
};

struct builder_column_run {
    struct builder_row_state* rows;
    struct builder_buffer_checkpoint* values;
    struct builder_buffer_checkpoint* validity;
    struct builder_buffer_checkpoint* bytes;
    void* value_base;
    void* byte_base;
    uint64_t* validity_words;
    struct builder_content_bound content_bound;
    // First row requiring growth or run completion.
    uint64_t limit;
    uint64_t row_limit;
    uint64_t byte_limit;
    uint32_t width_bytes;
    uint32_t offset_bit_width;
    enum dora_physical_type physical;
    enum builder_run_kind kind;
    bool is_open;
    bool should_check_content;
};

// Pending columns commit together; failure or discard restores records.
struct builder_runs {
    struct builder_operation_data operation;
    struct builder_column_run* columns;
    uint64_t records;
};

//------------------------------------------------------------------------------
//
// Builder Lifecycle
//
//------------------------------------------------------------------------------

enum builder_phase : uint8_t {
    BUILDER_PHASE_OPEN,
    BUILDER_PHASE_PREPARED,
    BUILDER_PHASE_SHARED
};

enum builder_entry : uint8_t {
    BUILDER_ENTRY_OPEN,
    BUILDER_ENTRY_ANY_PHASE,
    BUILDER_ENTRY_INSPECT,
    BUILDER_ENTRY_RECLAIM,
    BUILDER_ENTRY_CLEAR,
    BUILDER_ENTRY_PREPARED_ACCESS
};

struct builder_state {
    struct dora_allocation storage;
    struct identity_claim_state claims;
    struct dora_schema schema;
    struct dora_allocation bookkeeping;
    struct dora_allocation buffer_storage;
    struct builder_node_state* nodes;
    struct builder_buffer_state* buffers;
    struct builder_row_checkpoint* mutation_nodes;
    struct builder_buffer_checkpoint* mutation_buffers;
    struct dora_node_topology* topology;
    struct statistics_accumulator* statistics;
    struct builder_window window;
    struct builder_runs runs;
    uint64_t completed_root_row_count;
    // Shared-phase owners and adopted buffer addresses.
    struct dora_allocation shared_vectors;
    struct dora_allocation shared_addresses;
    struct dora_builder_options options;
    size_t capacity_bytes;
    size_t written_byte_count;
    uint32_t node_count;
    uint32_t buffer_count;
    uint32_t buffer_capacity;
    enum builder_phase phase;
    // Owning release deferred until operation exit.
    struct dora_handle release_owner;
    bool has_validation_nodes;
};

struct builder_operation {
    struct builder_operation_data data;
    struct builder_state* state;
    struct identity_slot* slot;
    bool is_content_carried;
};

static inline uint32_t builder_node_buffer_count(const struct builder_node_state* node) {
    return node->field->layout == DORA_LAYOUT_BINARY_VIEW ? node->rows.data_buffer_count + 3 : node->buffer_count;
}

static inline uint32_t
builder_next_buffer(const struct builder_state* state, const struct builder_node_state* node, uint32_t index) {
    if (node->field->layout != DORA_LAYOUT_BINARY_VIEW || index == node->first_buffer) {
        return index + 1;
    }
    if (index == node->first_buffer + 1) {
        return node->first_buffer + (node->rows.data_buffer_count == 0 ? 3U : 2U);
    }
    return index == node->rows.last_data_buffer ? node->first_buffer + 3 : state->buffers[index].next_buffer;
}

//------------------------------------------------------------------------------
//
// State Resolution
//
//------------------------------------------------------------------------------

int dora_builder_resolve_state(const struct dora_builder* builder, struct builder_state** out_state);

// Requires an entered operation without active checkpoints; preserves that state.
int dora_builder_statistics_checkpoint(struct builder_operation* operation,
                                       struct statistics_checkpoint* out_checkpoint);

int dora_builder_activity(struct builder_operation* operation, uint32_t number, struct dora_allocation* out_activity);

//------------------------------------------------------------------------------
//
// Operations
//
//------------------------------------------------------------------------------

// Runs no callbacks.
int dora_builder_operation_capture(const struct dora_builder* receiver, struct builder_operation* out_operation);

// Run commit may release the target; PREPARED_ACCESS resumes its parked window instead.
int dora_builder_operation_enter(struct builder_operation* operation, enum builder_entry entry);

int dora_builder_operation_begin(const struct dora_builder* receiver,
                                 enum builder_entry entry,
                                 struct builder_operation* out_operation);

// Failure leaves the operation parked.
int dora_builder_operation_resume(struct builder_operation_data* parked,
                                  enum builder_phase expected_phase,
                                  struct builder_operation* out_active);

// Clears active before deferred release; successful live-state requests return EINVAL if released.
int dora_builder_operation_leave(struct builder_operation* active, int status, struct builder_state** out_live_state);

// Parks captured data before operation exit.
int dora_builder_operation_park(struct builder_operation* active,
                                struct builder_operation_data* out_parked,
                                int status,
                                struct builder_state** out_live_state);

// Ignores the owner role.
bool dora_builder_operation_addresses(const struct builder_operation* operation, const struct dora_builder* receiver);

void dora_builder_operation_reset_checkpoints(struct builder_operation* operation);

void dora_builder_operation_data_move(struct builder_operation_data* source, struct builder_operation_data* out_data);

// Run completion may release the builder.
int dora_builder_runs_end(struct builder_state* state, bool commit);

// Rolls back without entering the retiring builder.
void dora_builder_run_discard(struct builder_state* state);

void dora_builder_discard_window(struct builder_state* state);

#endif
