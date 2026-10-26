// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Builder mutation and publication.

#ifndef DORA_BUILDER_BUILDER_INTERNAL_H
#define DORA_BUILDER_BUILDER_INTERNAL_H

#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "utf8/input.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Construction
//
//------------------------------------------------------------------------------

int dora_builder_init_fields_error(struct dora_builder* builder,
                                   struct dora_pool pool,
                                   const struct dora_schema_input* input,
                                   const struct dora_builder_options* options,
                                   struct dora_error* out_error);

int dora_builder_init_schema_error(struct dora_builder* builder,
                                   struct dora_pool pool,
                                   const struct dora_schema* schema,
                                   const struct dora_builder_options* options,
                                   struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Navigation and Setup
//
//------------------------------------------------------------------------------

int dora_builder_set_dictionary_error(struct dora_builder* builder,
                                      const struct dora_vector* values,
                                      struct dora_error* out_error);

//------------------------------------------------------------------------------
//
// Appends
//
//------------------------------------------------------------------------------

int dora_builder_append_vector_error(struct dora_builder* builder,
                                     const struct dora_vector* source,
                                     struct dora_error* out_error);

int dora_builder_append_nulls_error(struct dora_builder* builder, uint64_t row_count, struct dora_error* out_error);

int dora_builder_append_scalar_error(struct dora_builder* builder,
                                     enum dora_physical_type physical,
                                     const void* value,
                                     struct dora_error* out_error);

int dora_builder_append_bytes_error(struct dora_builder* builder,
                                    struct dora_bytes value,
                                    struct dora_error* out_error);

int dora_builder_append_container_error(struct dora_builder* builder,
                                        enum dora_type type,
                                        struct dora_range elements,
                                        int8_t type_id,
                                        bool view,
                                        struct dora_error* out_error);

int dora_builder_append_reader_text(struct dora_builder* builder,
                                    const char* text,
                                    size_t byte_count,
                                    enum dora_utf8_proof input_proof);

//------------------------------------------------------------------------------
//
// Completion and Lifecycle
//
//------------------------------------------------------------------------------

int dora_builder_end_row_error(struct dora_builder* builder, struct dora_error* out_error);

// Leaves the builder open and empty; the vector owns the batch payload.
int dora_builder_finish_batch(struct dora_builder* builder, struct dora_vector* out_vector);

int dora_builder_finish_error(struct dora_builder* in_out_builder,
                              struct dora_vector* out_vector,
                              struct dora_error* out_error);

#endif
