// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Extension storage resolution.

#ifndef DORA_EXTENSION_STORAGE_H
#define DORA_EXTENSION_STORAGE_H

#include "dora/vector/handle.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "vector/vector_internal.h"

static inline int extension_resolve_storage_field(struct dora_schema* schema, struct dora_schema_field* out_field) {
    int status = dora_schema_get_field(schema, out_field);
    while (status == DORA_OK &&
           (out_field->layout == DORA_LAYOUT_DICTIONARY || out_field->layout == DORA_LAYOUT_RUN_END)) {
        struct dora_schema child = {};
        status = dora_schema_get_child(schema, out_field->layout == DORA_LAYOUT_DICTIONARY ? 0 : 1, &child);
        if (status == DORA_OK) {
            *schema = child;
            status = dora_schema_get_field(schema, out_field);
        }
    }
    return status;
}

static inline int extension_prepare_storage(struct dora_pool pool,
                                            const struct dora_vector* source,
                                            struct dora_vector* out_storage,
                                            struct dora_vector* out_owned,
                                            struct dora_vector_view* out_view) {
    *out_storage = dora_vector_borrow(source);
    int status = dora_vector_get_view(source, out_view);
    if (status == DORA_OK && (out_view->layout == DORA_LAYOUT_DICTIONARY || out_view->layout == DORA_LAYOUT_RUN_END ||
                              out_view->selection.indices != nullptr)) {
        status = dora_vector_decode_error(source, pool, out_owned, nullptr);
        if (status == DORA_OK) {
            *out_storage = dora_vector_borrow(out_owned);
            status = dora_vector_get_view(out_storage, out_view);
        }
    }
    return status;
}

#endif
