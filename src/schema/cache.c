// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "memory/registry.h"
#include "schema/schema_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

size_t dora_schema_cache_bytes(struct dora_pool pool) {
    const struct pool_row* const row = pool_row_of(pool);
    size_t bytes = 0;
    for (const struct schema_image* image = row == nullptr ? nullptr : row->cached_schemas; image != nullptr;
         image = image->next_cached) {
        bytes += image->storage.capacity_bytes;
    }
    return bytes;
}

uint32_t dora_schema_cache_count(struct dora_pool pool) {
    const struct pool_row* const row = pool_row_of(pool);
    uint32_t count = 0;
    for (const struct schema_image* image = row == nullptr ? nullptr : row->cached_schemas; image != nullptr;
         image = image->next_cached) {
        ++count;
    }
    return count;
}

uint32_t dora_schema_cache_slot_count(struct dora_pool pool) {
    const struct pool_row* const row = pool_row_of(pool);
    uint32_t count = 0;
    for (const struct schema_image* image = row == nullptr ? nullptr : row->cached_schemas; image != nullptr;
         image = image->next_cached) {
        count += 1 + image->child_count;
    }
    return count;
}

static void cached_schemas_release(struct dora_pool pool, struct schema_image* image) {
    dora_pool_callback_enter(pool);
    while (image != nullptr) {
        struct schema_image* const next = image->next_cached;
        struct dora_schema owner = image->cached_owner;
        image->cached_owner = (struct dora_schema){};
        image->next_cached = nullptr;
        dora_schema_release(&owner);
        image = next;
    }
    dora_pool_callback_leave(pool);
}

bool dora_schema_cache_evict(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return false;
    }
    bool evicted = false;
    // Reentrant callbacks must not extend the eviction bound.
    const uint32_t initial_count = dora_schema_cache_count(pool);
    for (uint32_t remaining = initial_count; remaining != 0; --remaining) {
        struct schema_image* retired = nullptr;
        struct schema_image** tail = &retired;
        for (struct schema_image** link = &row->cached_schemas; *link != nullptr;) {
            struct schema_image* const image = *link;
            if (image->claims.claim_count == 1) {
                *link = image->next_cached;
                image->next_cached = nullptr;
                *tail = image;
                tail = &image->next_cached;
            } else {
                link = &image->next_cached;
            }
        }
        if (retired == nullptr) {
            return evicted;
        }
        evicted = true;
        cached_schemas_release(pool, retired);
    }
    return evicted;
}

void dora_schema_cache_drain(struct dora_pool pool) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr) {
        return;
    }
    struct schema_image* const image = row->cached_schemas;
    row->cached_schemas = nullptr;
    cached_schemas_release(pool, image);
}

// The extra claim prevents eviction during directory growth.
static int cached_image_claim(struct dora_pool pool, struct schema_image* image, struct dora_schema* out_schema) {
    ++image->claims.claim_count;
    const int status =
        dora_identity_claim_acquire(pool, &image->cached_owner.handle, IDENTITY_KIND_SCHEMA, &out_schema->handle);
    --image->claims.claim_count;
    return status;
}

int dora_schema_node_claim(struct dora_pool pool,
                           const struct dora_schema_field* field,
                           const struct dora_schema* children,
                           struct dora_schema* out_schema) {
    struct pool_row* const row = pool_row_of(pool);
    if (row == nullptr || field == nullptr || (field->child_count != 0 && children == nullptr) ||
        !schema_is_empty(out_schema)) {
        return EINVAL;
    }
    struct dora_schema candidate = {};
    int status = DORA_OK;
    for (;;) {
        for (struct schema_image* image = row->cached_schemas; image != nullptr; image = image->next_cached) {
            bool matches = false;
            status = dora_schema_field_matches(&image->cached_owner, field, &matches);
            for (uint32_t child = 0; status == DORA_OK && matches && child < field->child_count; ++child) {
                struct dora_schema cached = {};
                status = dora_schema_get_physical_child(&image->cached_owner, child, &cached);
                if (status == DORA_OK) {
                    status = dora_schema_declarations_equal(&cached, &children[child], &matches);
                }
            }
            if (status != DORA_OK || !matches) {
                if (status != DORA_OK) {
                    break;
                }
                continue;
            }
            status = cached_image_claim(pool, image, out_schema);
            dora_schema_release(&candidate);
            return status;
        }
        if (status != DORA_OK) {
            break;
        }
        if (candidate.handle.slot_index == 0) {
            struct dora_allocation storage = {};
            size_t bytes = 0;
            status = ckd_mul(&bytes, field->child_count, sizeof(struct dora_schema)) ? EOVERFLOW : DORA_OK;
            if (status == DORA_OK && bytes != 0) {
                status = dora_memory_payload_allocate(pool, bytes, &storage);
            }
            struct dora_schema* const cached = storage.address;
            if (cached != nullptr) {
                memset(cached, 0, bytes);
            }
            for (uint32_t index = 0; status == DORA_OK && index < field->child_count; ++index) {
                struct dora_schema_field child_field = {};
                status = dora_schema_get_field(&children[index], &child_field);
                struct dora_allocation child_storage = {};
                size_t child_bytes = 0;
                if (status == DORA_OK && ckd_mul(&child_bytes, child_field.child_count, sizeof(struct dora_schema))) {
                    status = EOVERFLOW;
                }
                if (status == DORA_OK && child_bytes != 0) {
                    status = dora_memory_payload_allocate(pool, child_bytes, &child_storage);
                }
                struct dora_schema* const grandchildren = child_storage.address;
                if (grandchildren != nullptr) {
                    memset(grandchildren, 0, child_bytes);
                }
                for (uint32_t child_index = 0; status == DORA_OK && child_index < child_field.child_count;
                     ++child_index) {
                    status = dora_schema_get_physical_child(&children[index], child_index, &grandchildren[child_index]);
                }
                if (status == DORA_OK) {
                    status = dora_schema_node_claim(pool, &child_field, grandchildren, &cached[index]);
                }
                dora_allocation_release(&child_storage);
            }
            if (status == DORA_OK) {
                status = dora_schema_init_retain_node(&candidate, pool, field, cached);
            }
            for (uint32_t index = 0; cached != nullptr && index < field->child_count; ++index) {
                dora_schema_release(&cached[index]);
            }
            dora_allocation_release(&storage);
            if (status != DORA_OK) {
                break;
            }
            continue;
        }
        const struct schema_image* resolved = nullptr;
        status = dora_schema_resolve_image(&candidate, &resolved);
        if (status == DORA_OK) {
            status = dora_identity_claim_transfer(pool, &candidate.handle, IDENTITY_KIND_SCHEMA);
        }
        if (status == DORA_OK) {
            struct schema_image* const image = (struct schema_image*)resolved;
            image->cached_owner = candidate;
            image->next_cached = row->cached_schemas;
            row->cached_schemas = image;
            candidate = (struct dora_schema){};
            status = cached_image_claim(pool, image, out_schema);
        }
        break;
    }
    dora_schema_release(&candidate);
    return status;
}
