// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector binding and publication.

#ifndef DORA_VECTOR_VECTOR_INTERNAL_H
#define DORA_VECTOR_VECTOR_INTERNAL_H

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "hash/keyset.h"
#include "memory/scratch.h"
#include "utf8/input.h"
#include "vector/image.h"

#include <errno.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

int dora_vector_init_retain_validated_in_pool(struct dora_vector* vector,
                                              struct dora_pool pool,
                                              const struct dora_vector* source,
                                              enum dora_failure_policy content_policy,
                                              struct dora_error* out_error);

int dora_vector_defer_validity(struct vector_image* image,
                               const struct dora_vector* source,
                               const struct dora_vector* ancestor,
                               uint64_t row_count,
                               uint64_t** out_selection);

int dora_vector_prepare(struct dora_pool pool,
                        const struct vector_input* input,
                        struct vector_image** out_image,
                        struct dora_error* out_error);

int dora_vector_admit(struct dora_pool pool,
                      const struct dora_schema* schema,
                      struct vector_node_ownership* owned,
                      const struct vector_input* input,
                      struct dora_view_buffer* buffers);

int dora_vector_check_activity(struct vector_image* image, const struct vector_input* input);

int dora_vector_derive_children(struct vector_image** in_out_image,
                                bool are_children_folded,
                                struct dora_error* out_error);

int dora_vector_refresh_schema(struct vector_image* image);

int dora_vector_bind_parameters(const struct dora_extension_definition* definition,
                                struct dora_pool pool,
                                const struct dora_schema* storage,
                                const struct dora_schema_field* field,
                                struct dora_allocation* out_parameters,
                                struct dora_error* out_error);

int dora_vector_bind_extension(struct vector_image* image, struct dora_error* out_error);

int dora_vector_init_move_input(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct vector_input* input,
                                struct dora_allocation* in_out_buffers,
                                struct dora_error* out_error);

int dora_vector_image_share(struct dora_pool pool,
                            const struct dora_vector* source,
                            const struct dora_schema_field* field,
                            struct vector_image** out_image,
                            struct dora_error* out_error);

int dora_vector_init_wrap_input(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct vector_input* input,
                                struct dora_error* out_error);

int dora_vector_init_retain_constant_in_pool(struct dora_vector* vector,
                                             struct dora_pool pool,
                                             const struct dora_vector* value,
                                             uint64_t row_count);

int dora_vector_init_retain_selection_in_pool(struct dora_vector* vector,
                                              struct dora_pool pool,
                                              const struct dora_vector* source,
                                              const struct dora_vector* indices);

int dora_vector_init_retain_dictionary_in_pool(struct dora_vector* vector,
                                               struct dora_pool pool,
                                               const struct dora_vector* indices,
                                               const struct dora_vector* values);

int dora_vector_validity_shareable(const struct dora_vector* source, bool* out_shareable);

int dora_vector_init_share_validity(struct dora_vector* vector,
                                    const struct vector_input* input,
                                    const struct dora_vector* source,
                                    uint32_t shared_index,
                                    struct dora_allocation* in_out_buffers);

int dora_vector_init_share_indices(struct dora_vector* vector,
                                   const struct dora_schema_field* field,
                                   const struct dora_vector* source,
                                   const struct dora_vector* values);

int dora_vector_admit_foreign(struct dora_pool pool,
                              const struct dora_schema* schema,
                              const struct vector_input* input,
                              struct dora_view_buffer* buffers,
                              struct vector_bound_node* in_out_node,
                              struct dora_error* out_error);

int dora_vector_init_copy_bytes_standing(struct dora_vector* vector,
                                         struct dora_pool pool,
                                         const struct dora_logical_type* type,
                                         enum dora_layout layout,
                                         const struct dora_bytes* values,
                                         const bool* validity,
                                         uint64_t row_count,
                                         enum vector_content_state standing);

int dora_vector_init_retain_extension_error(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            struct dora_bytes name,
                                            struct dora_bytes metadata,
                                            const struct dora_vector* storage,
                                            const struct dora_schema* admitted_schema,
                                            struct dora_error* out_error);

int dora_vector_init_retain_schema_error(struct dora_vector* vector,
                                         struct dora_pool pool,
                                         const struct dora_vector* source,
                                         const struct dora_schema* schema,
                                         enum vector_content_state content_state,
                                         struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

bool dora_vector_is_empty(const struct dora_vector* vector);

int dora_vector_resolve(const struct dora_vector* vector, struct vector_image** out_image);

bool dora_vector_buffer_has_owner(const struct vector_image* image, struct dora_view_buffer buffer);

bool dora_vector_owns_host_validity(const struct vector_image* image, struct dora_view_buffer validity);

struct vector_buffer_row* dora_vector_image_buffers(struct vector_image* image);

struct dora_vector* dora_vector_image_children(struct vector_image* image);

int dora_vector_publish(struct vector_image* image, struct dora_vector* out_vector);

void dora_vector_discard(struct vector_image* pending);

// Requires sole ownership of an unshared image; successful reclaim consumes its claim.
int dora_vector_reclaim_buffers(struct dora_vector* in_out_vector,
                                struct dora_allocation* out_buffers,
                                uint32_t buffer_capacity,
                                uint32_t* out_buffer_count);

int dora_vector_set_statistics_claim(struct vector_image* image,
                                     const struct dora_vector* statistics,
                                     enum vector_statistics_provenance provenance);

// Requires count > 0 and valid vectors from one pool.
int dora_vector_get_shared_pool(const struct dora_vector* const* vectors, uint32_t count, struct dora_pool* out_pool);

// Requires count > 0 and valid vectors from one pool.
int dora_vector_get_shared_pool_span(const struct dora_vector* vectors, uint32_t count, struct dora_pool* out_pool);

// Consumes staged after the last callback.
static inline int vector_finish_publication(struct dora_vector* staged, struct dora_vector* out, int status) {
    if (status == DORA_OK) {
        if (dora_vector_is_empty(out)) {
            *out = *staged;
            *staged = (struct dora_vector){};
            return DORA_OK;
        }
        status = EINVAL;
    }
    dora_vector_release(staged);
    return status;
}

// PROMISED skips standalone checks; decodes remain checked.
static inline enum dora_utf8_proof vector_utf8_proof(const struct vector_image* image) {
    if (image == nullptr) {
        return DORA_UTF8_CHECKED;
    }
    for (;;) {
        const struct dora_vector* values = nullptr;
        if (!dora_vector_is_empty(&image->ownership.storage)) {
            values = &image->ownership.storage;
        } else if (image->ownership.view.layout == DORA_LAYOUT_DICTIONARY ||
                   image->ownership.view.layout == DORA_LAYOUT_RUN_END) {
            const struct dora_vector* children =
                (const struct dora_vector*)(image->table_bytes + image->child_vectors.byte_offset);
            values = &children[image->ownership.view.layout == DORA_LAYOUT_DICTIONARY ? 0 : 1];
        }
        if (values == nullptr) {
            return image->node.content_state == VECTOR_CONTENT_VALIDATED_ASCII ? DORA_UTF8_VALIDATED_ASCII
                   : image->node.content_state >= VECTOR_CONTENT_VALIDATED     ? DORA_UTF8_VALIDATED
                                                                               : DORA_UTF8_PROMISED;
        }
        struct vector_image* next = nullptr;
        if (dora_vector_resolve(values, &next) != DORA_OK || next == nullptr) {
            return DORA_UTF8_PROMISED;
        }
        image = next;
    }
}

//------------------------------------------------------------------------------
//
// Reads
//
//------------------------------------------------------------------------------

//--------------------------------------
// Decoded Values
//--------------------------------------

struct decoded_elements {
    struct dora_vector vector;
    struct dora_range rows;
};

struct decoded_member {
    struct dora_vector vector;
    uint64_t row;
    int8_t type_id;
};

union decoded_row_value {
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
    struct dora_bytes bytes;
    struct decoded_elements elements;
    struct decoded_member member;
};

struct decoded_row {
    union decoded_row_value value;
    enum dora_type type;
    enum dora_physical_type physical_type;
    bool is_null;
};

int dora_vector_bind_coordinates(const struct dora_vector* vector, struct dora_vector_view* out_view);

int dora_vector_bind_input(struct scratch_scope* scope,
                           const struct dora_vector* source,
                           struct vector_bound_node* leaf,
                           struct vector_bound_input* out_input);

int dora_vector_bind_children(struct scratch_scope* scope,
                              const struct dora_vector* children,
                              uint32_t child_count,
                              struct vector_bound_node* leaves,
                              struct vector_bound_input* out_input);

uint32_t dora_vector_input_child(uint32_t node, const struct vector_bound_input* input, uint32_t ordinal);

int dora_vector_compare_rows(const struct dora_vector* keys,
                             uint64_t left_row,
                             uint64_t right_row,
                             enum dora_compare_op comparison,
                             bool* out_matches);

int dora_vector_get_physical_child(const struct dora_vector* vector,
                                   uint32_t child_index,
                                   struct dora_vector* out_child);

int dora_vector_get_own_validity(const struct dora_vector* vector,
                                 struct dora_view_buffer* out_validity,
                                 uint64_t* out_bit_offset);

int dora_vector_get_binary_view_sizes(const struct dora_vector* vector, struct dora_view_buffer* out_sizes);

int dora_vector_get_own_null_count(const struct dora_vector* vector, int64_t* out_null_count);

int dora_vector_get_extension(const struct dora_vector* vector,
                              const struct dora_extension_definition** out_definition,
                              const struct dora_allocation** out_parameters);

int dora_vector_get_computed_statistics(const struct dora_vector* vector, struct dora_vector* out_statistics);

int dora_vector_decode_row(const struct dora_vector* vector, uint64_t row, struct decoded_row* out_row);

bool dora_vector_try_get_exact_null_count(const struct dora_vector* vector,
                                          const struct vector_image* image,
                                          uint64_t* out_null_count);

int dora_vector_input_admit(const struct vector_bound_input* input, uint8_t required_flags, bool reject_extensions);

int dora_vector_input_admit_pair(const struct vector_bound_input* left,
                                 const struct vector_bound_input* right,
                                 bool order);

//------------------------------------------------------------------------------
//
// Copies
//
//------------------------------------------------------------------------------

int dora_vector_copy_requires_transport(const struct dora_vector* vector, bool* out_needed);

int dora_vector_copy_foreign(struct dora_pool pool,
                             const struct dora_schema* schema,
                             const struct vector_bound_input* input,
                             uint32_t root_node_index,
                             struct dora_buffer_owner* owner,
                             struct dora_vector* out_vector,
                             struct dora_error* out_error);

int dora_vector_decode_error(const struct dora_vector* vector,
                             struct dora_pool pool,
                             struct dora_vector* out_decoded,
                             struct dora_error* out_error);

int dora_vector_copy_schema_error(const struct dora_vector* vector,
                                  struct dora_pool pool,
                                  const struct dora_schema* schema,
                                  struct dora_vector* out_copy,
                                  struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Validation and Readiness
//
//------------------------------------------------------------------------------

// Requires zero initialization; bound to one caller and key vector.
struct vector_map_scratch {
    struct dora_vector keys;
    struct scratch_scope scope;
    struct dora_keyset set;
    struct vector_bound_input input;
    struct dora_vector_view view;
    bool are_hashes_admitted;
    bool can_bind_directly;
};

// Rejects null keys with EINVAL and duplicates with EILSEQ; sorted keys require strict order.
int dora_vector_check_map_keys(struct dora_pool pool,
                               const struct dora_vector* keys,
                               struct dora_range entries,
                               bool sorted,
                               struct vector_map_scratch* scratch);

int dora_vector_check_map_rows(struct dora_pool pool,
                               const struct dora_vector* keys,
                               const struct dora_vector_view* maps,
                               struct vector_map_scratch* scratch);

int dora_vector_check_map_vector(struct dora_pool pool, const struct dora_vector* map);

void dora_vector_map_scratch_release(struct vector_map_scratch* scratch);

#endif
