// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Buffered text output and column binding.

#ifndef DORA_IO_TEXT_WRITER_H
#define DORA_IO_TEXT_WRITER_H

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "memory/scratch.h"
#include "types/text.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/vector_internal.h"
#include "view/view_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t TEXT_WRITER_BUFFER_BYTES = 32768;

struct text_writer_output {
    struct dora_byte_sink* sink;
    size_t used_bytes;
    struct dora_allocation buffer;
};

int dora_text_writer_prepare_output(struct text_writer_output* output, struct dora_pool pool);

int dora_text_writer_write(struct text_writer_output* output, const void* bytes, size_t byte_count);
int dora_text_writer_drain(struct text_writer_output* output);

struct text_writer_column {
    union {
        struct dora_vector decoded;
        struct vector_bound_input bound;
        uint32_t root_node_index;
    } ownership;
    struct dora_vector_view view;
    struct dora_schema_field field;
    enum dora_utf8_proof proof;
};

struct text_writer_input {
    struct scratch_scope scope;
    struct dora_vector root;
    struct dora_vector decoded;
    struct dora_schema schema;
    struct dora_schema columns_schema;
    struct dora_vector_view view;
    struct vector_bound_input bound;
    struct text_writer_column* columns;
    struct dora_mutable_bytes format_buffer;
    uint32_t struct_node_index;
    uint32_t column_count;
};

static inline bool text_writer_is_encoded(const struct dora_vector_view* view) {
    return view->layout == DORA_LAYOUT_DICTIONARY || view->layout == DORA_LAYOUT_RUN_END;
}

int dora_text_writer_prepare_input(struct dora_pool pool,
                                   const struct dora_vector* input,
                                   struct text_writer_input* prepared);
int dora_text_writer_bind_input_column(struct text_writer_input* prepared, uint32_t index);
void dora_text_writer_release_input(struct text_writer_input* prepared);

static inline int text_writer_resolve(const struct vector_bound_input* input,
                                      uint32_t node,
                                      uint64_t* row,
                                      const struct dora_vector_view** out_view,
                                      bool* out_null) {
    for (;;) {
        const struct dora_vector_view* const view = &input->nodes[node].view;
        if (*row >= view->length) {
            return EINVAL;
        }
        *out_view = view;
        *out_null = dora_vector_view_is_null(view, *row);
        if (*out_null || !text_writer_is_encoded(view)) {
            return DORA_OK;
        }
        if (view->layout == DORA_LAYOUT_DICTIONARY) {
            const int status = dora_vector_view_read_index(view, *row, row);
            if (status != DORA_OK) {
                return status;
            }
            node = dora_vector_input_child(node, input, 0);
        } else {
            *row = dora_vector_view_find_run(view, *row);
            node = dora_vector_input_child(node, input, 1);
        }
    }
}

static inline int text_writer_format(const struct dora_logical_type* type,
                                     const struct dora_vector_view* view,
                                     uint64_t row,
                                     struct scratch_scope* scope,
                                     struct dora_mutable_bytes* text,
                                     char* inline_text,
                                     size_t inline_capacity,
                                     struct dora_bytes* out_bytes) {
    if (type->id == DORA_TYPE_UTF8) {
        *out_bytes = dora_vector_view_get_bytes(view, row);
        return DORA_OK;
    }
    bool boolean = false;
    const void* value;
    if (type->id == DORA_TYPE_BOOLEAN) {
        boolean = dora_vector_view_get_bool(view, row);
        value = &boolean;
    } else {
        const uint64_t physical = dora_vector_view_get_physical_row(view, row);
        value = (const uint8_t*)view->values.data + physical * view->byte_width;
    }
    char* output = text->data != nullptr ? (char*)text->data : inline_text;
    size_t capacity = text->data != nullptr ? text->byte_count : inline_capacity;
    size_t count = 0;
    int status = dora_type_format_scalar(type, view->physical_type, value, false, output, capacity, &count);
    if (status == ENOBUFS) {
        if (ckd_add(&capacity, count, (size_t)1)) {
            return EOVERFLOW;
        }
        void* address = nullptr;
        status = dora_scratch_allocate_growing(scope, capacity, text->byte_count, &address, &capacity);
        if (status == DORA_OK) {
            *text = (struct dora_mutable_bytes){address, capacity};
        }
        if (status == DORA_OK) {
            output = (char*)text->data;
            status = dora_type_format_scalar(type, view->physical_type, value, false, output, text->byte_count, &count);
        }
    }
    if (status == DORA_OK) {
        *out_bytes = (struct dora_bytes){.data = (const uint8_t*)output, .byte_count = count};
    }
    return status;
}

#endif
