// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "io/text_writer.h"

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "io/text.h"
#include "memory/memory_internal.h"
#include "memory/scratch.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

int dora_text_writer_prepare_output(struct text_writer_output* output, struct dora_pool pool) {
    const int status = dora_memory_payload_allocate(pool, TEXT_WRITER_BUFFER_BYTES, &output->buffer);
    return status == ENOMEM ? DORA_OK : status;
}

int dora_text_writer_drain(struct text_writer_output* output) {
    if (output->used_bytes == 0) {
        return DORA_OK;
    }
    const size_t length = output->used_bytes;
    output->used_bytes = 0;
    return output->sink->write(output->sink, (struct dora_bytes){output->buffer.address, length});
}

int dora_text_writer_write(struct text_writer_output* output, const void* bytes, size_t byte_count) {
    const uint8_t* cursor = bytes;
    while (byte_count != 0) {
        const size_t capacity = output->buffer.capacity_bytes;
        if (byte_count < capacity) {
            const size_t available = capacity - output->used_bytes;
            const size_t copied = byte_count < available ? byte_count : available;
            memcpy((uint8_t*)output->buffer.address + output->used_bytes, cursor, copied);
            output->used_bytes += copied;
            cursor += copied;
            byte_count -= copied;
            if (output->used_bytes < capacity) {
                return DORA_OK;
            }
        }
        const int status = dora_text_writer_drain(output);
        if (status != DORA_OK) {
            return status;
        }
        if (byte_count >= capacity) {
            return output->sink->write(output->sink, (struct dora_bytes){cursor, byte_count});
        }
    }
    return DORA_OK;
}

static uint32_t text_writer_leaf(const struct vector_bound_input* input, uint32_t node) {
    while (text_writer_is_encoded(&input->nodes[node].view)) {
        node = dora_vector_input_child(node, input, input->nodes[node].view.layout == DORA_LAYOUT_RUN_END ? 1 : 0);
    }
    return node;
}

static int text_writer_check_column(const struct dora_schema_field* field,
                                    const struct dora_vector_view* leaf,
                                    uint32_t child_count) {
    if (child_count != 0 || !text_field_is_supported(field, false)) {
        return ENOTSUP;
    }
    return leaf->byte_width > sizeof(struct dora_i256) ? EINVAL : DORA_OK;
}

static int text_writer_bind_column(struct scratch_scope* scope,
                                   const struct dora_vector* vector,
                                   const struct dora_schema* schema,
                                   struct text_writer_column* out_column) {
    const struct dora_pool pool = scope->pool;
    struct text_writer_column column = {};
    struct dora_vector decoded = {};
    struct vector_bound_input bound = {};
    int status = dora_schema_get_field(schema, &column.field);
    if (status == DORA_OK) {
        status = dora_vector_get_view(vector, &column.view);
    }
    const struct dora_vector_view* leaf = &column.view;
    uint32_t child_count = column.field.child_count;
    if (status == DORA_OK && text_writer_is_encoded(&column.view)) {
        bool transport = false;
        status = dora_vector_copy_requires_transport(vector, &transport);
        if (status == DORA_OK && transport) {
            status = dora_vector_decode_error(vector, pool, &decoded, nullptr);
            if (status == DORA_OK) {
                status = dora_vector_get_view(&decoded, &column.view);
            }
            child_count = 0;
        } else if (status == DORA_OK) {
            status = dora_vector_bind_input(scope, vector, nullptr, &bound);
            if (status == DORA_OK) {
                const uint32_t node = text_writer_leaf(&bound, 0);
                leaf = &bound.nodes[node].view;
                child_count = bound.nodes[node].field.child_count;
            }
        }
    }
    if (status == DORA_OK) {
        status = text_writer_check_column(&column.field, leaf, child_count);
    }
    if (status != DORA_OK) {
        dora_vector_release(&decoded);
        return status;
    }
    if (bound.nodes != nullptr) {
        column.ownership.bound = bound;
    } else {
        column.ownership.decoded = decoded;
    }
    struct vector_image* image = nullptr;
    status = dora_vector_resolve(vector, &image);
    if (status != DORA_OK) {
        dora_vector_release(&decoded);
        return status;
    }
    column.proof = vector_utf8_proof(image);
    *out_column = column;
    return DORA_OK;
}

static int text_writer_bind_root_column(const struct vector_bound_input* root,
                                        uint32_t node,
                                        const struct dora_schema* schema,
                                        struct text_writer_column* out_column) {
    const uint32_t leaf = text_writer_leaf(root, node);
    struct dora_schema_field field = {};
    int status = dora_schema_get_field(schema, &field);
    if (status == DORA_OK) {
        status = text_writer_check_column(&field, &root->nodes[leaf].view, root->nodes[leaf].field.child_count);
    }
    if (status == DORA_OK) {
        *out_column = (struct text_writer_column){
            .ownership.root_node_index = node,
            .view = root->nodes[node].view,
            .field = field,
            .proof = vector_utf8_proof(root->nodes[node].image)
        };
    }
    return status;
}

int dora_text_writer_bind_input_column(struct text_writer_input* prepared, uint32_t index) {
    struct text_writer_column* const columns = prepared->columns;
    struct dora_schema field = {};
    if (prepared->bound.nodes != nullptr) {
        const uint32_t node = dora_vector_input_child(prepared->struct_node_index, &prepared->bound, index);
        int status = dora_schema_get_child(&prepared->columns_schema, index, &field);
        if (status == DORA_OK) {
            status = text_writer_bind_root_column(&prepared->bound, node, &field, &columns[index]);
        }
        return status;
    }
    struct dora_vector child = {};
    int status = dora_vector_get_child(&prepared->root, index, &child);
    if (status == DORA_OK) {
        status = dora_schema_get_child(&prepared->schema, index, &field);
    }
    if (status == DORA_OK) {
        status = text_writer_bind_column(&prepared->scope, &child, &field, &columns[index]);
    }
    return status;
}

int dora_text_writer_prepare_input(struct dora_pool pool,
                                   const struct dora_vector* input,
                                   struct text_writer_input* prepared) {
    prepared->root = dora_vector_borrow(input);
    int status = dora_scratch_scope_begin(pool, &prepared->scope);
    if (status == DORA_OK) {
        status = dora_vector_get_view(input, &prepared->view);
    }
    if (status == DORA_OK && text_writer_is_encoded(&prepared->view)) {
        bool transport = false;
        status = dora_vector_copy_requires_transport(input, &transport);
        if (status == DORA_OK && transport) {
            status = dora_vector_decode_error(input, pool, &prepared->decoded, nullptr);
            if (status == DORA_OK) {
                prepared->root = dora_vector_borrow(&prepared->decoded);
                status = dora_vector_get_view(&prepared->root, &prepared->view);
            }
        } else if (status == DORA_OK) {
            status = dora_vector_bind_input(&prepared->scope, input, nullptr, &prepared->bound);
            if (status == DORA_OK) {
                prepared->struct_node_index = text_writer_leaf(&prepared->bound, 0);
            }
        }
    }
    if (status == DORA_OK && (prepared->view.type != DORA_TYPE_STRUCT ||
                              (prepared->bound.nodes != nullptr &&
                               prepared->bound.nodes[prepared->struct_node_index].view.layout != DORA_LAYOUT_STRUCT))) {
        status = ENOTSUP;
    }
    if (status == DORA_OK && prepared->bound.nodes != nullptr) {
        prepared->column_count = prepared->bound.nodes[prepared->struct_node_index].field.child_count;
    } else if (status == DORA_OK) {
        status = dora_vector_get_child_count(&prepared->root, &prepared->column_count);
    }
    if (status == DORA_OK) {
        status = dora_vector_retain_schema(&prepared->root, &prepared->schema);
    }
    prepared->columns_schema = dora_schema_borrow(&prepared->schema);
    if (status == DORA_OK && prepared->bound.nodes != nullptr) {
        uint32_t node = 0;
        while (status == DORA_OK && text_writer_is_encoded(&prepared->bound.nodes[node].view)) {
            const uint32_t ordinal = prepared->bound.nodes[node].view.layout == DORA_LAYOUT_RUN_END ? 1 : 0;
            struct dora_schema child = {};
            status = dora_schema_get_child(&prepared->columns_schema, ordinal, &child);
            prepared->columns_schema = child;
            node = dora_vector_input_child(node, &prepared->bound, ordinal);
        }
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, prepared->column_count, sizeof(struct text_writer_column))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        void* address = nullptr;
        status = dora_scratch_allocate(&prepared->scope, bytes, &address);
        prepared->columns = address;
    }
    if (status == DORA_OK && bytes != 0) {
        memset(prepared->columns, 0, bytes);
    }
    return status;
}

static void text_writer_release_column(struct text_writer_column* column) {
    if (!text_writer_is_encoded(&column->view)) {
        dora_vector_release(&column->ownership.decoded);
    }
}

void dora_text_writer_release_input(struct text_writer_input* prepared) {
    struct text_writer_column* const columns = prepared->columns;
    if (columns != nullptr && prepared->bound.nodes == nullptr) {
        for (uint32_t index = 0; index < prepared->column_count; ++index) {
            text_writer_release_column(&columns[index]);
        }
    }
    dora_schema_release(&prepared->schema);
    dora_vector_release(&prepared->decoded);
    dora_scratch_scope_end(&prepared->scope);
}
