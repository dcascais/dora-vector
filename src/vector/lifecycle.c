// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

static bool buffer_allocation_contains(const struct dora_allocation* allocation, struct dora_view_buffer buffer) {
    const uintptr_t base = (uintptr_t)allocation->address;
    const uintptr_t address = (uintptr_t)buffer.data;
    return allocation->address != nullptr && address >= base && address - base <= allocation->capacity_bytes &&
           buffer.byte_count <= allocation->capacity_bytes - (address - base);
}

static void queue_dependency(struct dora_handle* owner, struct vector_image** pending) {
    void* retired = nullptr;
    dora_identity_claim_release(owner, IDENTITY_KIND_VECTOR, &retired);
    if (retired != nullptr) {
        struct vector_image* const image = retired;
        image->next_retired = *pending;
        *pending = image;
    }
}

bool dora_vector_is_empty(const struct dora_vector* vector) {
    return vector != nullptr && vector->handle.slot_index == 0 && vector->handle.generation == 0 &&
           vector->handle.pool_index == 0 && !vector->handle.is_owner && vector->handle.reserved == 0 &&
           vector->rows.offset == 0 && vector->rows.length == 0;
}

struct vector_buffer_row* dora_vector_image_buffers(struct vector_image* image) {
    return (struct vector_buffer_row*)(image->table_bytes + image->buffers.byte_offset);
}

struct dora_vector* dora_vector_image_children(struct vector_image* image) {
    return (struct dora_vector*)(image->table_bytes + image->child_vectors.byte_offset);
}

int dora_vector_resolve(const struct dora_vector* vector, struct vector_image** out_image) {
    const struct identity_slot* slot = nullptr;
    if (vector == nullptr || out_image == nullptr ||
        dora_identity_resolve(&vector->handle, IDENTITY_KIND_VECTOR, &slot) != DORA_OK) {
        return EINVAL;
    }
    struct vector_image* const image = slot->claims->payload;
    if (vector->rows.offset > image->ownership.view.length ||
        vector->rows.length > image->ownership.view.length - vector->rows.offset) {
        return EINVAL;
    }
    *out_image = image;
    return DORA_OK;
}

bool dora_vector_owns_host_validity(const struct vector_image* image, struct dora_view_buffer validity) {
    if (validity.data == nullptr) {
        return false;
    }
    const uintptr_t address = (uintptr_t)validity.data;
    const struct dora_allocation* const allocations[2] = {
        &image->ownership.effective_validity,
        &image->ownership.selection_storage
    };
    for (uint32_t index = 0; index < 2; ++index) {
        const struct dora_allocation* const allocation = allocations[index];
        const uintptr_t base = (uintptr_t)allocation->address;
        if (allocation->address != nullptr && address >= base) {
            const uintptr_t offset = address - base;
            if (offset <= allocation->capacity_bytes && validity.byte_count <= allocation->capacity_bytes - offset) {
                return true;
            }
        }
    }
    return false;
}

bool dora_vector_buffer_has_owner(const struct vector_image* image, struct dora_view_buffer buffer) {
    if (dora_vector_owns_host_validity(image, buffer)) {
        return true;
    }
    const struct vector_image* const backing = image->buffer_image;
    if (backing->wrapped_owner != nullptr && backing->wrapped_owner->release != nullptr) {
        return true;
    }
    const struct vector_buffer_row* const rows =
        (const struct vector_buffer_row*)(backing->table_bytes + backing->buffers.byte_offset);
    for (uint32_t index = 0; index < backing->buffers.entry_count; ++index) {
        const struct vector_buffer_row* const row = &rows[index];
        if (row->owner_kind == VECTOR_BUFFER_OWNER_ALLOCATION &&
            buffer_allocation_contains(&row->owner.allocation, buffer)) {
            return true;
        }
        if (row->owner_kind == VECTOR_BUFFER_OWNER_RETAINED) {
            const struct identity_slot* owner = nullptr;
            if (dora_identity_resolve(&row->owner.retained, IDENTITY_KIND_VECTOR, &owner) == DORA_OK &&
                dora_vector_buffer_has_owner(owner->claims->payload, buffer)) {
                return true;
            }
        }
    }
    if (image->ownership.backing.handle.slot_index != 0) {
        struct vector_image* provider = nullptr;
        if (dora_vector_resolve(&image->ownership.backing, &provider) == DORA_OK) {
            return dora_vector_buffer_has_owner(provider, buffer);
        }
    }
    return false;
}

void dora_vector_discard(struct vector_image* pending) {
    while (pending != nullptr) {
        struct vector_image* const image = pending;
        pending = image->next_retired;
        const struct dora_pool pool = image->allocation.pool;
        dora_pool_callback_enter(pool);
        struct vector_node_ownership* const owned = &image->ownership;
        struct vector_buffer_row* const buffers = dora_vector_image_buffers(image);
        struct dora_vector* const children = dora_vector_image_children(image);
        dora_allocation_release(&owned->extension_parameters);
        dora_allocation_release(&owned->effective_validity);
        dora_allocation_release(&owned->selection_storage);
        for (uint32_t index = 0; index < image->buffers.entry_count; ++index) {
            if (buffers[index].owner_kind == VECTOR_BUFFER_OWNER_ALLOCATION) {
                dora_allocation_release(&buffers[index].owner.allocation);
            }
        }
        if (image->wrapped_owner != nullptr && image->wrapped_owner->release != nullptr) {
            image->wrapped_owner->release(image->wrapped_owner);
        }
        dora_schema_release(&image->schema);
        queue_dependency(&owned->backing.handle, &pending);
        queue_dependency(&owned->storage.handle, &pending);
        if (owned->owns_physical_rows) {
            dora_allocation_release(&owned->physical_rows);
        } else {
            queue_dependency(&owned->selection_source.handle, &pending);
        }
        queue_dependency(&owned->statistics.handle, &pending);
        if (owned->deferred_validity != nullptr) {
            struct vector_deferred_validity* const deferred = owned->deferred_validity;
            queue_dependency(&deferred->inputs[0].handle, &pending);
            queue_dependency(&deferred->inputs[1].handle, &pending);
            struct dora_allocation allocation = deferred->allocation;
            dora_allocation_release(&allocation);
        }
        for (uint32_t index = 0; index < image->buffers.entry_count; ++index) {
            if (buffers[index].owner_kind == VECTOR_BUFFER_OWNER_RETAINED) {
                queue_dependency(&buffers[index].owner.retained, &pending);
            }
        }
        for (uint32_t index = 0; index < image->child_vectors.entry_count; ++index) {
            queue_dependency(&children[index].handle, &pending);
        }
        struct dora_allocation allocation = image->allocation;
        dora_allocation_release(&allocation);
        dora_pool_callback_leave(pool);
    }
}

int dora_vector_publish(struct vector_image* image, struct dora_vector* out_vector) {
    if (image == nullptr || !dora_vector_is_empty(out_vector)) {
        return EINVAL;
    }
    enum vector_content_state state = image->node.content_state;
    const bool encoded =
        image->ownership.view.layout == DORA_LAYOUT_DICTIONARY || image->ownership.view.layout == DORA_LAYOUT_RUN_END;
    if (encoded && image->ownership.view.type == DORA_TYPE_UTF8 && state >= VECTOR_CONTENT_VALIDATED) {
        state = VECTOR_CONTENT_VALIDATED_ASCII;
    }
    for (uint32_t index = 0; index < image->child_vectors.entry_count; ++index) {
        struct vector_image* child = nullptr;
        const int resolved = dora_vector_resolve(&dora_vector_image_children(image)[index], &child);
        if (resolved != DORA_OK) {
            return resolved;
        }
        enum vector_content_state child_state = child->node.content_state;
        if (image->ownership.view.layout == DORA_LAYOUT_RUN_END && index == 0 &&
            child_state >= VECTOR_CONTENT_VALIDATED) {
            child_state = VECTOR_CONTENT_VALIDATED_ASCII;
        }
        if (child_state < state) {
            state = child_state;
        }
    }
    if (!dora_vector_is_empty(&image->ownership.storage)) {
        struct vector_image* storage = nullptr;
        const int resolved = dora_vector_resolve(&image->ownership.storage, &storage);
        if (resolved != DORA_OK) {
            return resolved;
        }
        if (storage->node.content_state < state) {
            state = storage->node.content_state;
        }
    }
    if (image->ownership.view.type != DORA_TYPE_UTF8 && state > VECTOR_CONTENT_VALIDATED) {
        state = VECTOR_CONTENT_VALIDATED;
    }
    image->node.content_state = state;
    struct dora_allocation parameters = {};
    int status = dora_allocation_adopt(&image->ownership.extension_parameters, &parameters);
    if (status != DORA_OK) {
        return status;
    }
    image->ownership.extension_parameters = parameters;
    struct dora_handle owner = {};
    status = dora_identity_slot_publish(image->allocation.pool, IDENTITY_KIND_VECTOR, &image->claims, &owner);
    if (status != DORA_OK) {
        return status;
    }
    if (!dora_vector_is_empty(out_vector)) {
        void* retired = nullptr;
        dora_identity_claim_release(&owner, IDENTITY_KIND_VECTOR, &retired);
        return EINVAL;
    }
    *out_vector = (struct dora_vector){.handle = owner, .rows.length = image->ownership.view.length};
    return DORA_OK;
}

void dora_vector_release(struct dora_vector* vector) {
    if (vector == nullptr) {
        return;
    }
    struct dora_handle owner = vector->handle;
    *vector = (struct dora_vector){};
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_VECTOR, &retired);
    dora_vector_discard(retired);
}

int dora_vector_reclaim_buffers(struct dora_vector* in_out_vector,
                                struct dora_allocation* out_buffers,
                                uint32_t buffer_capacity,
                                uint32_t* out_buffer_count) {
    struct vector_image* image = nullptr;
    if (out_buffers == nullptr || out_buffer_count == nullptr || in_out_vector == nullptr ||
        !in_out_vector->handle.is_owner || dora_vector_resolve(in_out_vector, &image) != DORA_OK) {
        return EINVAL;
    }
    uint32_t claims = 1;
    if (image->ownership.storage.handle.slot_index != 0) {
        if (image->claims.claim_count != 1) {
            return EBUSY;
        }
        struct vector_image* storage = nullptr;
        struct vector_image* backing = nullptr;
        if (dora_vector_resolve(&image->ownership.storage, &storage) != DORA_OK ||
            dora_vector_resolve(&image->ownership.backing, &backing) != DORA_OK || image->buffer_image != storage ||
            backing != storage) {
            return EINVAL;
        }
        image = storage;
        claims = 2;
    }
    if (image->buffer_image != image || image->ownership.backing.handle.slot_index != 0 ||
        image->ownership.storage.handle.slot_index != 0 || image->buffers.entry_count > buffer_capacity) {
        return EINVAL;
    }
    if (image->claims.claim_count != claims) {
        return EBUSY;
    }
    struct vector_buffer_row* const rows = dora_vector_image_buffers(image);
    for (uint32_t index = 0; index < image->buffers.entry_count; ++index) {
        out_buffers[index] = (struct dora_allocation){};
        if (rows[index].owner_kind == VECTOR_BUFFER_OWNER_ALLOCATION) {
            out_buffers[index] = rows[index].owner.allocation;
            rows[index].owner.allocation = (struct dora_allocation){};
        }
    }
    *out_buffer_count = image->buffers.entry_count;
    dora_vector_release(in_out_vector);
    return DORA_OK;
}

bool dora_vector_is_valid(const struct dora_vector* vector) {
    struct vector_image* image = nullptr;
    return dora_vector_resolve(vector, &image) == DORA_OK;
}

int dora_vector_retain(const struct dora_vector* vector, struct dora_vector* out_vector) {
    if (!dora_vector_is_empty(out_vector) || !dora_vector_is_valid(vector)) {
        return EINVAL;
    }
    struct dora_handle owner = {};
    const int status = dora_identity_claim_acquire((struct dora_pool){}, &vector->handle, IDENTITY_KIND_VECTOR, &owner);
    if (status == DORA_OK) {
        if (!dora_vector_is_empty(out_vector)) {
            void* retired = nullptr;
            dora_identity_claim_release(&owner, IDENTITY_KIND_VECTOR, &retired);
            dora_vector_discard(retired);
            return EINVAL;
        }
        *out_vector = *vector;
        out_vector->handle = owner;
    }
    return status;
}

int dora_vector_move(struct dora_vector* in_out_source, struct dora_vector* out_destination) {
    if (!dora_vector_is_empty(out_destination) || !dora_vector_is_valid(in_out_source) ||
        !in_out_source->handle.is_owner) {
        return EINVAL;
    }
    *out_destination = *in_out_source;
    *in_out_source = (struct dora_vector){};
    return DORA_OK;
}

//--------------------------------------
// Statistics Attachment
//--------------------------------------

int dora_vector_set_statistics_claim(struct vector_image* image,
                                     const struct dora_vector* statistics,
                                     enum vector_statistics_provenance provenance) {
    if (image == nullptr || !dora_vector_is_valid(statistics) || !dora_vector_is_empty(&image->ownership.statistics)) {
        return EINVAL;
    }
    struct dora_handle owner = {};
    const int status =
        dora_identity_claim_acquire(image->allocation.pool, &statistics->handle, IDENTITY_KIND_VECTOR, &owner);
    if (status == DORA_OK) {
        image->ownership.statistics = *statistics;
        image->ownership.statistics.handle = owner;
        image->ownership.statistics_provenance = provenance;
    }
    return status;
}

//--------------------------------------
// Placement
//--------------------------------------

int dora_vector_get_pool(const struct dora_vector* vector, struct dora_pool* out_pool) {
    struct vector_image* image = nullptr;
    if (out_pool == nullptr || dora_vector_resolve(vector, &image) != DORA_OK) {
        return EINVAL;
    }
    *out_pool = image->allocation.pool;
    return DORA_OK;
}

static int shared_pool_join(const struct dora_vector* vector, struct dora_pool* in_out_pool) {
    struct dora_pool pool = {};
    const int status = dora_vector_get_pool(vector, &pool);
    if (status != DORA_OK) {
        return status;
    }
    if (in_out_pool->index == 0) {
        *in_out_pool = pool;
        return DORA_OK;
    }
    return pool_is_same(*in_out_pool, pool) ? DORA_OK : EINVAL;
}

int dora_vector_get_shared_pool(const struct dora_vector* const* vectors, uint32_t count, struct dora_pool* out_pool) {
    struct dora_pool pool = {};
    int status = vectors == nullptr || count == 0 || out_pool == nullptr ? EINVAL : DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        status = shared_pool_join(vectors[index], &pool);
    }
    if (status == DORA_OK) {
        *out_pool = pool;
    }
    return status;
}

int dora_vector_get_shared_pool_span(const struct dora_vector* vectors, uint32_t count, struct dora_pool* out_pool) {
    struct dora_pool pool = {};
    int status = vectors == nullptr || count == 0 || out_pool == nullptr ? EINVAL : DORA_OK;
    for (uint32_t index = 0; status == DORA_OK && index < count; ++index) {
        status = shared_pool_join(&vectors[index], &pool);
    }
    if (status == DORA_OK) {
        *out_pool = pool;
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Validation and Readiness
//
//------------------------------------------------------------------------------

int dora_vector_settle(const struct dora_vector* vector) {
    const struct dora_vector* current = vector;
    for (;;) {
        struct vector_image* image = nullptr;
        int status = dora_vector_resolve(current, &image);
        if (status != DORA_OK) {
            return status;
        }
        // Only readiness callbacks require pool work; ordinary readers leave the pool untouched.
        const struct dora_readiness* const readiness = &image->ownership.view.readiness;
        const struct dora_pool pool = readiness->event != nullptr ? image->allocation.pool : (struct dora_pool){};
        dora_pool_callback_enter(pool);
        if (readiness->event != nullptr) {
            status = readiness->settle != nullptr ? readiness->settle(readiness) : EINVAL;
        }
        struct dora_vector* const children = dora_vector_image_children(image);
        for (uint32_t index = 0; status == DORA_OK && index < image->child_vectors.entry_count; ++index) {
            status = dora_vector_settle(&children[index]);
        }
        if (status == DORA_OK && image->ownership.storage.handle.slot_index != 0) {
            status = dora_vector_settle(&image->ownership.storage);
        }
        if (status == DORA_OK && image->ownership.backing.handle.slot_index != 0) {
            if (image->ownership.deferred_validity == nullptr) {
                current = &image->ownership.backing;
                dora_pool_callback_leave(pool);
                continue;
            }
            status = dora_vector_settle(&image->ownership.backing);
        }
        if (image->ownership.deferred_validity != nullptr) {
            for (uint32_t index = 0; status == DORA_OK && index < 2; ++index) {
                if (image->ownership.deferred_validity->inputs[index].handle.slot_index != 0) {
                    status = dora_vector_settle(&image->ownership.deferred_validity->inputs[index]);
                }
            }
        }
        dora_pool_callback_leave(pool);
        return status;
    }
}
