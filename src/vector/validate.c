// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/binding/validate.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "kernel/packet.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "memory/scratch.h"
#include "schema/schema_internal.h"
#include "types/array_count.h"
#include "utf8/batch.h"
#include "utf8/input.h"
#include "vector/image.h"
#include "vector/operand.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct validation {
    struct dora_pool pool;
    struct dora_error* error;
    const struct dora_schema_field* declaration;
    bool should_null_invalid;
    const char* path;
};

static void validation_error_at(const struct validation* context, const char* name, const char* reason, uint64_t row) {
    struct dora_error* error = context->error;
    if (context->path != nullptr && context->path[0] != 0) {
        name = context->path;
    }
    if (error == nullptr || error->message[0] != 0) {
        return;
    }
    name = name != nullptr && name[0] != 0 ? name : "<unnamed>";
    const struct dora_bytes parts[] = {
        {(const uint8_t*)"Field '", 7},
        {(const uint8_t*)name, strlen(name)},
        {(const uint8_t*)"': ", 3},
        {(const uint8_t*)reason, strlen(reason)}
    };
    size_t written = 0;
    for (size_t part = 0; part < ARRAY_COUNT(parts); ++part) {
        const size_t remaining = sizeof error->message - 1 - written;
        const size_t count = parts[part].byte_count < remaining ? parts[part].byte_count : remaining;
        memcpy(error->message + written, parts[part].data, count);
        written += count;
    }
    if (row != UINT64_MAX) {
        static const char marker[] = " at row ";
        const size_t remaining = sizeof error->message - 1 - written;
        const size_t count = sizeof marker - 1 < remaining ? sizeof marker - 1 : remaining;
        memcpy(error->message + written, marker, count);
        written += count;
        char digits[20];
        size_t digit_count = 0;
        do {
            digits[digit_count++] = (char)('0' + row % 10);
            row /= 10;
        } while (row != 0);
        while (digit_count != 0 && written < sizeof(error->message) - 1) {
            error->message[written++] = digits[--digit_count];
        }
    }
    error->message[written] = 0;
}

static void validation_error(const struct validation* context, const char* name, const char* reason) {
    validation_error_at(context, name, reason, UINT64_MAX);
}

static int validation_bitmap(struct dora_pool pool, uint64_t length, struct dora_allocation* out) {
    size_t bytes = 0;
    if (ckd_mul(&bytes, bitmap_words_for_bits(length), sizeof(uint64_t))) {
        return EOVERFLOW;
    }
    const int status = dora_pool_allocate(pool, bytes, out);
    if (status == DORA_OK && bytes != 0) {
        bitmap_fill_bytes(out->address, bytes, true);
    }
    return status;
}

static int validation_count(const struct dora_vector* source,
                            const struct vector_image* image,
                            const struct dora_vector_view* view,
                            int64_t* out_count) {
    *out_count = view->null_count;
    const struct vector_node_ownership* const own = &image->ownership;
    if (own->own_validity.data == nullptr || own->own_null_count < 0 || view->selection.indices != nullptr ||
        source->rows.offset != 0 || source->rows.length != own->view.length) {
        return DORA_OK;
    }
    const uint64_t count =
        view->length - dora_bitmap_count_set_bytes(own->own_validity.data, own->own_validity_bit_offset, view->length);
    *out_count = (int64_t)count;
    return count == (uint64_t)own->own_null_count ? DORA_OK : EILSEQ;
}

struct validation_rows {
    const struct validation* context;
    const struct dora_vector* source;
    const struct dora_schema_field* field;
    const struct dora_vector_view* view;
    struct dora_allocation* validity;
    bool should_null_invalid;
    bool* in_out_is_all_ascii;
};

struct validation_failure {
    uint64_t row;
    int status;
};

static int validation_fault(const struct validation_rows* rows, struct validation_failure failure) {
    if (!rows->should_null_invalid || (failure.status != EILSEQ && failure.status != ERANGE)) {
        validation_error_at(rows->context, rows->field->name, "value content validation failed", failure.row);
        return failure.status;
    }
    if ((rows->field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
        return EINVAL;
    }
    if (rows->validity->address == nullptr) {
        const int status = validation_bitmap(rows->context->pool, rows->view->length, rows->validity);
        if (status != DORA_OK) {
            return status;
        }
    }
    bitmap_clear(rows->validity->address, failure.row);
    return DORA_OK;
}

static int validation_content(const struct validation_rows* rows) {
    const struct dora_vector_view* const view = rows->view;
    struct kernel_validate_binding binding = {};
    if (view->length == 0 || !kernel_validate_has_content(view, &rows->field->type) ||
        dora_kernel_validate_content_bind(view, &rows->field->type, &binding) != DORA_OK) {
        return DORA_OK;
    }
    int status = DORA_OK;
    uint64_t first = 0;
    while (status == DORA_OK && first < view->length) {
        const uint32_t count =
            view->length - first < KERNEL_CHUNK_ROW_COUNT ? (uint32_t)(view->length - first) : KERNEL_CHUNK_ROW_COUNT;
        uint64_t bad_row = 0;
        status = binding.check(&binding, first, count, &bad_row);
        first += count;
        if (status != DORA_OK) {
            status = validation_fault(rows, (struct validation_failure){.row = bad_row, .status = status});
            first = bad_row + 1;
        }
    }
    return status;
}

static int validation_text_classify(const struct validation_rows* rows,
                                    struct dora_range chunk,
                                    const struct dora_utf8_batch* batch) {
    bool ascii = false;
    uint64_t invalid[KERNEL_CHUNK_WORD_COUNT];
    uint32_t invalid_count;
    int status = dora_utf8_classify_batch(batch, invalid, KERNEL_CHUNK_WORD_COUNT, &invalid_count, &ascii);
    *rows->in_out_is_all_ascii &= ascii;
    if (status != EILSEQ) {
        return status;
    }
    for (uint32_t word = 0; word < bitmap_words_for_bits(chunk.length); ++word) {
        uint64_t bad = invalid[word];
        while (bad != 0) {
            const uint64_t lane = (uint64_t)word * 64 + dora_bitmap_word_pop_lowest_set(&bad);
            status = validation_fault(rows, (struct validation_failure){.row = chunk.offset + lane, .status = EILSEQ});
            if (status != DORA_OK) {
                return status;
            }
        }
    }
    return DORA_OK;
}

static int validation_text_batch(const struct validation_rows* rows,
                                 struct dora_range chunk,
                                 const struct dora_utf8_batch* batch) {
    if (rows->should_null_invalid) {
        return validation_text_classify(rows, chunk, batch);
    }
    bool ascii = false;
    uint32_t first_bad = 0;
    const int status = dora_utf8_validate_batch(batch, &ascii, rows->context->error != nullptr ? &first_bad : nullptr);
    *rows->in_out_is_all_ascii &= ascii;
    return status == DORA_OK || rows->context->error == nullptr
               ? status
               : validation_fault(rows, (struct validation_failure){chunk.offset + first_bad, status});
}

static int validation_offsets(const struct validation_rows* rows, struct dora_range chunk) {
    const struct dora_vector_view* view = rows->view;
    if (view->selection.indices != nullptr ||
        (view->layout != DORA_LAYOUT_BINARY && view->layout != DORA_LAYOUT_LARGE_BINARY)) {
        return ENOTSUP;
    }
    const size_t width = view->layout == DORA_LAYOUT_BINARY ? 4 : 8;
    size_t origin;
    uint64_t first;
    if (ckd_add(&first, view->offset, chunk.offset) || ckd_mul(&origin, first, width)) {
        return EOVERFLOW;
    }
    if (origin > view->offsets.byte_count || view->data_buffer_count == 0) {
        return EINVAL;
    }
    const struct dora_utf8_batch batch = {
        .row_count = (uint32_t)chunk.length,
        .physical_value_count = (size_t)chunk.length,
        .layout = width == 4 ? DORA_UTF8_OFFSETS_I32 : DORA_UTF8_OFFSETS_I64,
        .eligible_rows =
            view->validity.data == nullptr
                ? (struct dora_utf8_mask){}
                : (struct
                   dora_utf8_mask){{view->validity.data, view->validity.byte_count}, view->validity_bit_offset + chunk.offset},
        .rows = {
            {view->data_buffers[0].data, view->data_buffers[0].byte_count},
            {(const uint8_t*)view->offsets.data + origin, view->offsets.byte_count - origin}
        }
    };
    return validation_text_batch(rows, chunk, &batch);
}

static int
validation_native(const struct validation_rows* rows, struct vector_operand* operand, struct dora_range chunk) {
    struct dora_utf8_batch batch = {};
    const int status = dora_vector_operand_prepare_utf8(operand, chunk, nullptr, &batch);
    batch.proof = DORA_UTF8_CHECKED;
    uint64_t eligible[KERNEL_CHUNK_WORD_COUNT] = {};
    if (status == DORA_OK) {
        for (uint32_t row = 0; row < chunk.length; ++row) {
            bitmap_assign(eligible, row, !dora_vector_view_is_null(rows->view, chunk.offset + row));
        }
        batch.eligible_rows = (struct dora_utf8_mask){{(const uint8_t*)eligible, sizeof eligible}, 0};
    }
    return status == DORA_OK ? validation_text_batch(rows, chunk, &batch) : status;
}

static int validation_utf8(const struct validation_rows* rows) {
    const struct dora_vector_view* const view = rows->view;
    if (rows->field->type.id != DORA_TYPE_UTF8 || view->layout == DORA_LAYOUT_DICTIONARY ||
        view->layout == DORA_LAYOUT_RUN_END) {
        return DORA_OK;
    }
    if (view->selection.indices == nullptr &&
        (view->layout == DORA_LAYOUT_BINARY || view->layout == DORA_LAYOUT_LARGE_BINARY)) {
        for (uint64_t first = 0; first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
            const struct dora_range chunk = {
                first,
                view->length - first < KERNEL_CHUNK_ROW_COUNT ? view->length - first : KERNEL_CHUNK_ROW_COUNT
            };
            const int status = validation_offsets(rows, chunk);
            if (status != DORA_OK) {
                return status;
            }
        }
        return DORA_OK;
    }
    struct vector_bound_node leaf;
    struct vector_bound_input input = {};
    struct vector_operand operand = {};
    struct scratch_scope scope = {};
    bool bound = false;
    int status = dora_scratch_scope_begin(rows->context->pool, &scope);
    for (uint64_t first = 0; status == DORA_OK && first < view->length; first += KERNEL_CHUNK_ROW_COUNT) {
        const struct dora_range chunk = {
            .offset = first,
            .length = view->length - first < KERNEL_CHUNK_ROW_COUNT ? view->length - first : KERNEL_CHUNK_ROW_COUNT,
        };
        status = validation_offsets(rows, chunk);
        if (status != ENOTSUP) {
            continue;
        }
        status = DORA_OK;
        if (!bound) {
            status = dora_vector_bind_input(&scope, rows->source, &leaf, &input);
            if (status == DORA_OK) {
                status = dora_vector_operand_bind(&scope, 0, &input, VECTOR_OPERAND_REQUIRE_TEXT, &operand);
            }
            bound = true;
        }
        if (status == DORA_OK) {
            status = validation_native(rows, &operand, chunk);
        }
    }
    dora_scratch_scope_end(&scope);
    return status;
}

static int validation_decimal(const struct validation_rows* rows) {
    const struct dora_vector_view* const view = rows->view;
    if (rows->field->type.id != DORA_TYPE_DECIMAL || view->layout != DORA_LAYOUT_FIXED || view->length == 0) {
        return DORA_OK;
    }
    struct kernel_validate_binding binding = {};
    int status = dora_kernel_validate_bind(view, &rows->field->type, nullptr, &binding);
    uint64_t first = 0;
    while (status == DORA_OK && first < view->length) {
        const uint32_t count =
            view->length - first < KERNEL_CHUNK_ROW_COUNT ? (uint32_t)(view->length - first) : KERNEL_CHUNK_ROW_COUNT;
        uint64_t bad_row = 0;
        status = binding.check(&binding, first, count, &bad_row);
        first += count;
        if (status != DORA_OK) {
            status = validation_fault(
                rows,
                (struct validation_failure){.row = bad_row, .status = status == EOVERFLOW ? ERANGE : status});
            first = bad_row + 1;
        }
    }
    return status;
}

static int validation_map(const struct validation_rows* rows) {
    const struct dora_schema_field* field = rows->field;
    const struct dora_vector_view* view = rows->view;
    const struct dora_vector* source = rows->source;
    const struct dora_pool pool = rows->context->pool;
    if (field->type.id != DORA_TYPE_MAP || view->layout == DORA_LAYOUT_DICTIONARY ||
        view->layout == DORA_LAYOUT_RUN_END) {
        return DORA_OK;
    }
    struct dora_vector keys = {};
    int status = dora_vector_get_child(source, 0, &keys);
    struct vector_map_scratch scratch = {};
    const bool sorted = (field->flags & DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED) != 0;
    if (status == DORA_OK && !sorted && dora_vector_check_map_rows(pool, &keys, view, &scratch) == DORA_OK) {
        dora_vector_map_scratch_release(&scratch);
        return DORA_OK;
    }
    for (uint64_t row = 0; status == DORA_OK && row < view->length; ++row) {
        if (dora_vector_view_is_null(view, row)) {
            continue;
        }
        status = dora_vector_check_map_keys(pool, &keys, dora_vector_view_get_elements(view, row), sorted, &scratch);
        if (status != DORA_OK) {
            status = validation_fault(rows, (struct validation_failure){row, status});
        }
    }
    dora_vector_map_scratch_release(&scratch);
    return status;
}

static int validation_mask(struct vector_image* image, const uint64_t* validity) {
    struct vector_node_ownership* const owned = &image->ownership;
    const struct dora_vector_view source = owned->view;
    uint64_t extent = 0;
    for (uint64_t row = 0; row < source.length; ++row) {
        const uint64_t index = dora_vector_view_get_physical_row(&source, row) - source.offset;
        if (index >= extent) {
            extent = index + 1;
        }
    }
    size_t bytes = 0;
    size_t total = 0;
    if (ckd_mul(&bytes, bitmap_words_for_bits(extent), sizeof(uint64_t)) || ckd_mul(&total, bytes, 2)) {
        return EOVERFLOW;
    }
    struct dora_allocation allocation = {};
    int status = dora_memory_payload_allocate(image->allocation.pool, total, &allocation);
    if (status != DORA_OK) {
        return status;
    }
    if (total != 0) {
        bitmap_fill_bytes(allocation.address, total, true);
    }
    uint64_t* const own_bits = allocation.address;
    uint64_t* const effective_bits = total == 0 ? nullptr : (uint64_t*)((uint8_t*)allocation.address + bytes);
    int64_t own_count = 0;
    int64_t effective_count = 0;
    for (uint64_t row = 0; total != 0 && row < source.length; ++row) {
        const uint64_t index = dora_vector_view_get_physical_row(&source, row) - source.offset;
        const bool invalid = !dora_bitmap_test((const uint8_t*)validity, row);
        const bool own_valid =
            !invalid && (owned->own_null_count == 0 || owned->own_validity.data == nullptr ||
                         dora_bitmap_test(owned->own_validity.data, owned->own_validity_bit_offset + index));
        const bool effective_valid = !invalid && !dora_vector_view_is_null(&source, row);
        bitmap_assign(own_bits, index, own_valid);
        bitmap_assign(effective_bits, index, effective_valid);
        own_count += own_valid ? 0 : 1;
        effective_count += effective_valid ? 0 : 1;
    }
    dora_allocation_release(&owned->effective_validity);
    owned->effective_validity = allocation;
    owned->own_validity =
        (struct dora_view_buffer){.data = own_bits, .byte_count = bytes, .readable_byte_count = bytes};
    owned->own_validity_bit_offset = 0;
    owned->own_null_count = own_count;
    owned->view.validity =
        (struct dora_view_buffer){.data = effective_bits, .byte_count = bytes, .readable_byte_count = bytes};
    owned->view.validity_bit_offset = 0;
    owned->view.null_count = effective_count;
    return DORA_OK;
}

static int
validation_publish(struct vector_image* image, const struct dora_schema_field* field, struct dora_vector* out) {
    if (image->child_vectors.entry_count != 0) {
        const int status = dora_vector_refresh_schema(image);
        if (status != DORA_OK) {
            dora_vector_discard(image);
            return status;
        }
    }
    if (image->node.content_state < VECTOR_CONTENT_VALIDATED) {
        image->node.content_state = VECTOR_CONTENT_VALIDATED;
    }
    struct dora_vector candidate = {};
    int status = dora_vector_publish(image, &candidate);
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return status;
    }
    if ((field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
        for (uint64_t row = 0; status == DORA_OK && row < candidate.rows.length; ++row) {
            bool was_null = false;
            status = dora_vector_is_null(&image->ownership.backing, row, &was_null);
            bool is_null = false;
            if (status == DORA_OK && !was_null) {
                status = dora_vector_is_null(&candidate, row, &is_null);
            }
            if (status == DORA_OK && is_null) {
                status = EINVAL;
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_vector_move(&candidate, out);
    }
    dora_vector_release(&candidate);
    return status;
}

static int validation_null_rows(const struct validation* context,
                                const struct dora_vector* source,
                                const uint64_t* validity,
                                struct dora_vector* out) {
    struct vector_image* source_image = nullptr;
    int status = dora_vector_resolve(source, &source_image);
    struct dora_schema_field field = {};
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(&source_image->schema, &field);
    }
    if (status == DORA_OK) {
        status = dora_vector_get_view(source, &view);
    }
    if (status != DORA_OK) {
        return status;
    }
    if ((field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
        for (uint64_t row = 0; row < view.length; ++row) {
            bool is_null = false;
            status = dora_vector_is_null(source, row, &is_null);
            if (status != DORA_OK || (!is_null && !dora_bitmap_test((const uint8_t*)validity, row))) {
                return status != DORA_OK ? status : EINVAL;
            }
        }
    }
    struct vector_image* image = nullptr;
    if (field.type.extension_name.byte_count != 0) {
        struct dora_vector storage = {};
        struct dora_vector masked = {};
        status = dora_vector_get_storage(source, &storage);
        if (status == DORA_OK) {
            status = validation_null_rows(context, &storage, validity, &masked);
        }
        if (status == DORA_OK) {
            status = dora_vector_image_share(context->pool, &masked, &field, &image, context->error);
        }
        if (status == DORA_OK) {
            status = dora_vector_bind_extension(image, context->error);
        }
        dora_vector_release(&masked);
    } else {
        status = dora_vector_image_share(context->pool, source, &field, &image, context->error);
        const bool union_layout = field.layout == DORA_LAYOUT_DENSE_UNION || field.layout == DORA_LAYOUT_SPARSE_UNION;
        if (status == DORA_OK && (union_layout || field.layout == DORA_LAYOUT_RUN_END)) {
            struct dora_vector* const children = dora_vector_image_children(image);
            for (uint32_t index = union_layout ? 0 : 1; status == DORA_OK && index < field.child_count; ++index) {
                struct dora_allocation child_validity = {};
                bool changed = false;
                for (uint64_t row = 0; status == DORA_OK && row < view.length; ++row) {
                    if (dora_bitmap_test((const uint8_t*)validity, row) || dora_vector_view_is_null(&view, row)) {
                        continue;
                    }
                    uint64_t child_row = 0;
                    const uint32_t child_index = union_layout ? dora_vector_view_get_member(&view, row, &child_row) : 1;
                    if (!union_layout) {
                        child_row = dora_vector_view_find_run(&view, row);
                    }
                    if (child_index != index) {
                        continue;
                    }
                    if (!changed) {
                        status = validation_bitmap(context->pool, children[index].rows.length, &child_validity);
                        changed = status == DORA_OK;
                    }
                    if (status == DORA_OK) {
                        bitmap_clear(child_validity.address, child_row);
                    }
                }
                struct dora_vector masked = {};
                if (status == DORA_OK && changed) {
                    status = validation_null_rows(context, &children[index], child_validity.address, &masked);
                }
                if (status == DORA_OK && changed) {
                    status = dora_identity_claim_transfer(context->pool, &masked.handle, IDENTITY_KIND_VECTOR);
                    if (status == DORA_OK) {
                        dora_vector_release(&children[index]);
                        children[index] = masked;
                        masked = (struct dora_vector){};
                    }
                }
                dora_vector_release(&masked);
                dora_allocation_release(&child_validity);
            }
        } else if (status == DORA_OK && field.layout != DORA_LAYOUT_NULL) {
            status = validation_mask(image, validity);
            if (status == DORA_OK) {
                status = dora_vector_derive_children(&image, false, context->error);
            }
        }
    }
    if (status == DORA_OK) {
        status = validation_publish(image, &field, out);
        image = nullptr;
    }
    dora_vector_discard(image);
    return status;
}

static int validation_extension(const struct validation* context,
                                const struct dora_vector* storage,
                                const struct dora_schema_field* field,
                                const struct vector_image* source_image,
                                uint64_t row_count,
                                struct dora_vector* out) {
    struct vector_image* image = nullptr;
    int status = DORA_OK;
    if (out != nullptr) {
        status = dora_vector_image_share(context->pool, storage, field, &image, context->error);
        if (status == DORA_OK) {
            status = dora_vector_bind_extension(image, context->error);
        }
        if (status == DORA_OK && image->ownership.extension == nullptr) {
            status = ENOTSUP;
        }
    }
    const struct vector_image* const bound = out == nullptr ? source_image : image;
    const bool encoded = field->layout == DORA_LAYOUT_DICTIONARY || field->layout == DORA_LAYOUT_RUN_END;
    struct dora_vector_view storage_view = {};
    if (status == DORA_OK) {
        status = dora_vector_get_view(storage, &storage_view);
    }
    const bool all_null = storage_view.null_count == (int64_t)storage_view.length;
    struct dora_allocation validity = {};
    if (status == DORA_OK && !encoded && !all_null && bound->ownership.extension->validate != nullptr) {
        status = validation_bitmap(context->pool, row_count, &validity);
        for (uint64_t row = 0; status == DORA_OK && row < row_count; ++row) {
            bool is_null = false;
            status = dora_vector_is_null(storage, row, &is_null);
            if (status == DORA_OK) {
                bitmap_assign(validity.address, row, !is_null);
            }
        }
        struct dora_error hook_error = {};
        bool did_validate = false;
        if (status == DORA_OK) {
            const struct dora_extension_definition* const definition = bound->ownership.extension;
            // Validation callbacks may reenter either pool.
            dora_pool_callback_enter(context->pool);
            dora_pool_callback_enter(bound->allocation.pool);
            did_validate = true;
            status = definition->validate(definition,
                                          context->pool,
                                          &bound->ownership.extension_parameters,
                                          storage,
                                          validity.address,
                                          context->error != nullptr ? &hook_error : nullptr);
            dora_pool_callback_leave(bound->allocation.pool);
            dora_pool_callback_leave(context->pool);
            hook_error.message[sizeof(hook_error.message) - 1] = 0;
        }
        const bool null_failure = did_validate && validity.address != nullptr && out != nullptr &&
                                  context->should_null_invalid && (status == EILSEQ || status == ERANGE);
        uint64_t first_bad = UINT64_MAX;
        if (did_validate && (status == EILSEQ || status == ERANGE) && validity.address != nullptr) {
            for (uint64_t row = 0; row < row_count; ++row) {
                if (!dora_vector_view_is_null(&storage_view, row) && !dora_bitmap_test(validity.address, row)) {
                    first_bad = row;
                    break;
                }
            }
        }
        if (status != DORA_OK && !null_failure) {
            validation_error_at(context,
                                field->name,
                                hook_error.message[0] != 0 ? hook_error.message : "extension validation failed",
                                first_bad);
        }
        if (null_failure) {
            if ((field->flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0) {
                validation_error_at(context,
                                    field->name,
                                    hook_error.message[0] != 0 ? hook_error.message : "extension validation failed",
                                    first_bad);
                status = EINVAL;
            } else {
                struct dora_vector masked = {};
                status = validation_null_rows(context, storage, validity.address, &masked);
                if (status == DORA_OK) {
                    dora_vector_discard(image);
                    image = nullptr;
                    status = dora_vector_image_share(context->pool, &masked, field, &image, context->error);
                }
                if (status == DORA_OK) {
                    status = dora_vector_bind_extension(image, context->error);
                }
                dora_vector_release(&masked);
            }
        }
    }
    if (status == DORA_OK && out != nullptr) {
        status = validation_publish(image, field, out);
        image = nullptr;
    }
    dora_vector_discard(image);
    dora_allocation_release(&validity);
    return status;
}

static int validation_recount(struct vector_image* image, int64_t null_count) {
    struct vector_node_ownership* const own = &image->ownership;
    struct dora_vector_view* const view = &own->view;
    own->own_null_count = null_count;
    if (view->validity.data == nullptr ||
        (view->validity.data == own->own_validity.data && view->validity_bit_offset == own->own_validity_bit_offset)) {
        view->validity = own->own_validity;
        view->validity_bit_offset = own->own_validity_bit_offset;
        view->null_count = null_count;
        return DORA_OK;
    }
    const uint64_t words = bitmap_words_for_bits(view->length);
    uint64_t valid_count = 0;
    bool changed = false;
    for (uint64_t word = 0; word < words; ++word) {
        const uint64_t effective = dora_vector_view_get_validity_word(view, word);
        const uint64_t original = dora_bitmap_get_word(own->own_validity.data,
                                                       own->own_validity_bit_offset + word * 64,
                                                       own->own_validity_bit_offset + view->length);
        changed = changed || (effective & original) != effective;
        valid_count += dora_bitmap_word_count_ones(effective & original);
    }
    if (changed) {
        struct dora_allocation mask = {};
        const int status = validation_bitmap(image->allocation.pool, view->length, &mask);
        if (status != DORA_OK) {
            return status;
        }
        uint64_t* const bits = mask.address;
        for (uint64_t word = 0; word < words; ++word) {
            bits[word] = dora_vector_view_get_validity_word(view, word) &
                         dora_bitmap_get_word(own->own_validity.data,
                                              own->own_validity_bit_offset + word * 64,
                                              own->own_validity_bit_offset + view->length);
        }
        dora_allocation_release(&own->effective_validity);
        own->effective_validity = mask;
        view->validity = (struct dora_view_buffer){
            .data = mask.address,
            .byte_count = bitmap_bytes_for_bits(view->length),
            .readable_byte_count = mask.capacity_bytes
        };
        view->validity_bit_offset = 0;
    }
    view->null_count = (int64_t)(view->length - valid_count);
    return DORA_OK;
}

static int validation_node(const struct validation* parent, const struct dora_vector* source, struct dora_vector* out) {
    struct validation local = *parent;
    char path[256];
    const struct validation* context = &local;
    struct vector_image* source_image = nullptr;
    int status = dora_vector_resolve(source, &source_image);
    struct dora_schema_field field = {};
    struct dora_vector_view view = {};
    if (status == DORA_OK) {
        status = dora_schema_get_field(&source_image->schema, &field);
    }
    if (status == DORA_OK && context->declaration != nullptr) {
        field.name = context->declaration->name;
        field.flags = context->declaration->flags;
        field.metadata = context->declaration->metadata;
        field.metadata_entry_count = context->declaration->metadata_entry_count;
    }
    if (status == DORA_OK && source_image->node.content_state >= VECTOR_CONTENT_VALIDATED) {
        return out == nullptr ? DORA_OK : dora_vector_retain(source, out);
    }
    if (status == DORA_OK) {
        if (context->error != nullptr && (local.path != nullptr || field.child_count != 0)) {
            size_t used = local.path != nullptr ? strlen(local.path) : 0;
            if (used != 0) {
                memcpy(path, local.path, used);
            }
            if (used != 0 && used < sizeof path - 1) {
                path[used++] = '.';
            }
            const char* name = field.name != nullptr && field.name[0] != 0 ? field.name : "<unnamed>";
            while (*name != 0 && used < sizeof path - 1) {
                path[used++] = *name++;
            }
            path[used] = 0;
            local.path = path;
        }
        status = dora_vector_get_view(source, &view);
        if (status == DORA_OK && view.selection.indices == nullptr && view.layout != DORA_LAYOUT_DICTIONARY &&
            view.layout != DORA_LAYOUT_RUN_END) {
            const struct vector_node_ownership* const own = &source_image->ownership;
            view.validity = own->own_validity;
            view.validity_bit_offset = own->own_validity_bit_offset + source->rows.offset;
            view.null_count =
                own->own_null_count != 0 && (source->rows.offset != 0 || source->rows.length != own->view.length)
                    ? -1
                    : own->own_null_count;
            if (status == DORA_OK && view.null_count < 0) {
                view.null_count = view.validity.data == nullptr
                                      ? 0
                                      : (int64_t)(view.length - dora_bitmap_count_set_bytes(view.validity.data,
                                                                                            view.validity_bit_offset,
                                                                                            view.length));
            }
            if (status == DORA_OK && view.null_count == 0) {
                view.validity = (struct dora_view_buffer){};
            }
        }
    }
    int64_t recounted = -1;
    bool count_changed = false;
    if (status == DORA_OK) {
        status = validation_count(source, source_image, &view, &recounted);
        count_changed = status == EILSEQ;
        if (count_changed && out != nullptr && context->should_null_invalid) {
            status = (field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) != 0 && recounted != 0 ? EINVAL : DORA_OK;
            view.null_count = recounted;
            view.validity = source_image->ownership.own_validity;
            view.validity_bit_offset = source_image->ownership.own_validity_bit_offset;
        }
    }
    if (status != DORA_OK) {
        validation_error(context, field.name, "declared validity count does not match its bitmap");
        return status;
    }
    if (field.type.extension_name.byte_count != 0) {
        source_image = nullptr;
        status = dora_vector_resolve(source, &source_image);
        struct dora_vector storage = {};
        if (status == DORA_OK) {
            status = dora_vector_get_storage(source, &storage);
        }
        const bool encoded = field.layout == DORA_LAYOUT_DICTIONARY || field.layout == DORA_LAYOUT_RUN_END;
        if (out == nullptr) {
            if (source_image->ownership.extension == nullptr) {
                validation_error(context, field.name, "extension content validation failed");
                return ENOTSUP;
            }
            if (encoded) {
                for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
                    struct dora_vector child = {};
                    status = dora_vector_get_physical_child(source, index, &child);
                    if (status == DORA_OK) {
                        struct validation child_context = *context;
                        child_context.declaration = nullptr;
                        status = validation_node(&child_context, &child, nullptr);
                    }
                }
                if (status != DORA_OK) {
                    validation_error(context, field.name, "extension content validation failed");
                }
                return status;
            }
            if (status == DORA_OK) {
                status = validation_node(context, &storage, nullptr);
            }
        }
        struct dora_vector validated = {};
        if (out != nullptr && status == DORA_OK && encoded) {
            struct vector_image* plain = nullptr;
            struct vector_image* storage_image = nullptr;
            struct dora_schema_field storage_field = {};
            status = dora_vector_resolve(&storage, &storage_image);
            if (status == DORA_OK) {
                status = dora_schema_get_field(&storage_image->schema, &storage_field);
            }
            if (status == DORA_OK) {
                status = dora_vector_image_share(context->pool, &storage, &storage_field, &plain, context->error);
            }
            for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
                struct dora_vector child = {};
                struct dora_vector checked = {};
                status = dora_vector_get_physical_child(source, index, &child);
                if (status == DORA_OK) {
                    struct validation child_context = *context;
                    child_context.declaration = nullptr;
                    status = validation_node(&child_context, &child, &checked);
                }
                struct dora_vector child_storage = dora_vector_borrow(&checked);
                if (status == DORA_OK && index == (field.layout == DORA_LAYOUT_DICTIONARY ? 0U : 1U)) {
                    child_storage = (struct dora_vector){};
                    status = dora_vector_get_storage(&checked, &child_storage);
                }
                struct dora_handle claim = {};
                if (status == DORA_OK) {
                    status =
                        dora_identity_claim_acquire(context->pool, &child_storage.handle, IDENTITY_KIND_VECTOR, &claim);
                }
                if (status == DORA_OK) {
                    struct dora_vector* const children = dora_vector_image_children(plain);
                    dora_vector_release(&children[index]);
                    children[index] = child_storage;
                    children[index].handle = claim;
                }
                dora_vector_release(&checked);
            }
            if (status == DORA_OK) {
                status = validation_publish(plain, &storage_field, &validated);
                plain = nullptr;
            }
            dora_vector_discard(plain);
            storage = dora_vector_borrow(&validated);
        } else if (out != nullptr && status == DORA_OK) {
            status = validation_node(context, &storage, &validated);
            storage = dora_vector_borrow(&validated);
        }
        if (status == DORA_OK) {
            status = validation_extension(context, &storage, &field, source_image, source->rows.length, out);
        }
        dora_vector_release(&validated);
        if (status != DORA_OK) {
            validation_error(context, field.name, "extension content validation failed");
        }
        return status;
    }
    struct dora_allocation validity = {};
    bool all_ascii = true;
    const bool all_null = view.null_count == (int64_t)view.length;
    const struct validation_rows rows = {
        .context = context,
        .source = source,
        .field = &field,
        .view = &view,
        .validity = &validity,
        .in_out_is_all_ascii = &all_ascii,
        .should_null_invalid = out != nullptr && context->should_null_invalid
    };
    status = all_null ? DORA_OK : validation_content(&rows);
    if (status == DORA_OK && !all_null) {
        status = validation_utf8(&rows);
    }
    if (status == DORA_OK && !all_null) {
        status = validation_decimal(&rows);
    }
    struct vector_image* image = nullptr;
    if (status == DORA_OK && out != nullptr) {
        status = dora_vector_image_share(context->pool, source, &field, &image, context->error);
        if (status == DORA_OK && count_changed) {
            status = validation_recount(image, recounted);
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < field.child_count; ++index) {
        struct dora_vector child = {};
        struct dora_vector validated = {};
        status = dora_vector_get_physical_child(source, index, &child);
        struct validation child_context = *context;
        struct dora_schema child_schema = {};
        struct dora_schema_field child_field = {};
        if (status == DORA_OK) {
            status = dora_schema_get_physical_child(&source_image->schema, index, &child_schema);
        }
        if (status == DORA_OK) {
            status = dora_schema_get_field(&child_schema, &child_field);
        }
        child_context.declaration = &child_field;
        if (status == DORA_OK) {
            status = validation_node(&child_context, &child, out == nullptr ? nullptr : &validated);
        }
        if (status == DORA_OK && out != nullptr) {
            status = dora_identity_claim_transfer(context->pool, &validated.handle, IDENTITY_KIND_VECTOR);
            if (status == DORA_OK) {
                struct dora_vector* const children = dora_vector_image_children(image);
                dora_vector_release(&children[index]);
                children[index] = validated;
                validated = (struct dora_vector){};
            }
        }
        dora_vector_release(&validated);
    }
    if (status == DORA_OK && !all_null) {
        status = validation_map(&rows);
    }
    if (status == DORA_OK && image != nullptr) {
        if (validity.address != nullptr) {
            status = validation_mask(image, validity.address);
        }
        if (status == DORA_OK) {
            image->node.content_state = field.type.id == DORA_TYPE_UTF8 && all_ascii ? VECTOR_CONTENT_VALIDATED_ASCII
                                                                                     : VECTOR_CONTENT_VALIDATED;
            status = validation_publish(image, &field, out);
            image = nullptr;
        }
    }
    dora_vector_discard(image);
    dora_allocation_release(&validity);
    if (status != DORA_OK) {
        validation_error(context, field.name, "value content validation failed");
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Validated Publication
//
//------------------------------------------------------------------------------

int dora_vector_init_retain_validated_in_pool(struct dora_vector* vector,
                                              struct dora_pool pool,
                                              const struct dora_vector* source,
                                              enum dora_failure_policy content_policy,
                                              struct dora_error* out_error) {
    if (content_policy > DORA_FAILURE_POLICY_NULL || !dora_vector_is_empty(vector) || !dora_vector_is_valid(source) ||
        pool_row_of(pool) == nullptr) {
        return EINVAL;
    }
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    const struct validation context =
        {.pool = pool, .error = out_error, .should_null_invalid = content_policy == DORA_FAILURE_POLICY_NULL};
    struct dora_vector published = {};
    int status = validation_node(&context, source, &published);
    status = vector_finish_publication(&published, vector, status);
    if (status == DORA_OK && out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    return status;
}

int dora_vector_init_retain_validated(struct dora_vector* vector,
                                      const struct dora_vector* source,
                                      enum dora_failure_policy content_policy,
                                      struct dora_error* out_error) {
    struct dora_pool pool = {};
    const int placed = dora_vector_get_pool(source, &pool);
    if (placed != DORA_OK) {
        return placed;
    }
    return dora_vector_init_retain_validated_in_pool(vector, pool, source, content_policy, out_error);
}

//------------------------------------------------------------------------------
//
// Content Validation
//
//------------------------------------------------------------------------------

int dora_vector_validate(const struct dora_vector* vector, struct dora_error* out_error) {
    struct vector_image* image = nullptr;
    const int status = dora_vector_resolve(vector, &image);
    if (status != DORA_OK) {
        return status;
    }
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    const struct validation context = {.pool = image->allocation.pool, .error = out_error};
    return validation_node(&context, vector, nullptr);
}
