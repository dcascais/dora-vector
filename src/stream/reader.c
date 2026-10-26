// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/vector.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "stream/stream_internal.h"
#include "vector/vector_internal.h"

#include <errno.h>

struct reader_state {
    struct dora_allocation storage;
    struct identity_claim_state claims;
    struct dora_vector_source* source;
    struct dora_error error;
    int failure;
    bool is_end;
    bool is_release_pending;
};

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

static void source_release(struct dora_vector_source* source) {
    if (source != nullptr && source->release != nullptr) {
        source->release(source);
    }
}

static void reader_discard(struct reader_state* state) {
    if (state != nullptr) {
        const struct dora_pool pool = state->storage.pool;
        dora_pool_callback_enter(pool);
        struct dora_vector_source* const source = state->source;
        state->source = nullptr;
        source_release(source);
        struct dora_allocation storage = state->storage;
        dora_allocation_release(&storage);
        dora_pool_callback_leave(pool);
    }
}

static bool reader_is_empty(const struct dora_reader* reader) {
    return reader != nullptr && handle_is_empty(reader->handle);
}

int dora_reader_move(struct dora_reader* in_out_source, struct dora_reader* out_destination) {
    if (in_out_source == nullptr || !in_out_source->handle.is_owner || !reader_is_empty(out_destination)) {
        return EINVAL;
    }
    struct identity_slot* slot = nullptr;
    const int status = dora_identity_slot_enter(&in_out_source->handle, IDENTITY_KIND_READER, &slot);
    if (status == DORA_OK) {
        *out_destination = *in_out_source;
        *in_out_source = (struct dora_reader){};
        dora_identity_slot_leave(&out_destination->handle, slot);
    }
    return status;
}

void dora_reader_release(struct dora_reader* reader) {
    if (reader == nullptr) {
        return;
    }
    struct dora_handle owner = reader->handle;
    *reader = (struct dora_reader){};
    const struct identity_slot* slot = nullptr;
    if (owner.is_owner && dora_identity_resolve(&owner, IDENTITY_KIND_READER, &slot) == DORA_OK && slot->is_running) {
        struct reader_state* const state = slot->claims->payload;
        state->is_release_pending = true;
        return;
    }
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_READER, &retired);
    reader_discard(retired);
}

static void reader_leave(struct identity_slot* slot, struct dora_handle owner) {
    const struct reader_state* const state = slot->claims->payload;
    const bool release_pending = state->is_release_pending;
    dora_identity_slot_leave(&owner, slot);
    if (release_pending) {
        owner.is_owner = true;
        struct dora_reader reader = {.handle = owner};
        dora_reader_release(&reader);
    }
}

int dora_reader_init(struct dora_reader* reader, struct dora_pool pool, struct dora_vector_source* source) {
    if (!reader_is_empty(reader) || source == nullptr || source->next == nullptr || source->release == nullptr) {
        source_release(source);
        return EINVAL;
    }
    struct dora_allocation storage = {};
    int status = dora_memory_payload_allocate(pool, sizeof(struct reader_state), &storage);
    if (status != DORA_OK) {
        source_release(source);
        return status;
    }
    struct reader_state* const state = storage.address;
    *state = (struct reader_state){.storage = storage, .source = source};
    state->claims.payload = state;
    struct dora_reader candidate = {};
    status = dora_identity_slot_publish(pool, IDENTITY_KIND_READER, &state->claims, &candidate.handle);
    if (status != DORA_OK) {
        reader_discard(state);
    } else if (!reader_is_empty(reader)) {
        status = EINVAL;
        dora_reader_release(&candidate);
    } else {
        *reader = candidate;
    }
    return status;
}

bool dora_reader_is_valid(const struct dora_reader* reader) {
    const struct identity_slot* slot = nullptr;
    return reader != nullptr && dora_identity_resolve(&reader->handle, IDENTITY_KIND_READER, &slot) == DORA_OK;
}

int dora_reader_get_pool(const struct dora_reader* reader, struct dora_pool* out_pool) {
    const struct identity_slot* slot = nullptr;
    if (reader == nullptr || out_pool == nullptr ||
        dora_identity_resolve(&reader->handle, IDENTITY_KIND_READER, &slot) != DORA_OK) {
        return EINVAL;
    }
    *out_pool = ((const struct reader_state*)slot->claims->payload)->storage.pool;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Reading
//
//------------------------------------------------------------------------------

int dora_reader_next(struct dora_reader* reader,
                     struct dora_vector* out_vector,
                     struct dora_stream_result* out_result,
                     struct dora_error* out_error) {
    if (reader == nullptr || !dora_vector_is_empty(out_vector) || out_result == nullptr) {
        return EINVAL;
    }
    const struct dora_handle owner = reader->handle;
    struct identity_slot* slot = nullptr;
    int status = dora_identity_slot_enter(&owner, IDENTITY_KIND_READER, &slot);
    if (status != DORA_OK) {
        return status;
    }
    struct reader_state* const state = slot->claims->payload;
    struct dora_vector output = {};
    struct dora_stream_result result = {.status = DORA_STREAM_STATUS_END};
    struct dora_error error = {};
    if (state->failure != DORA_OK) {
        status = state->failure;
        error = state->error;
    } else if (!state->is_end) {
        status = state->source->next(state->source, state->storage.pool, &output, &result, &error);
        if (status == DORA_OK && (result.status > DORA_STREAM_STATUS_END ||
                                  (result.status == DORA_STREAM_STATUS_READY &&
                                   (!output.handle.is_owner || !dora_vector_is_valid(&output))) ||
                                  (result.status != DORA_STREAM_STATUS_READY && !dora_vector_is_empty(&output)))) {
            status = EINVAL;
        }
        if (status == DORA_OK && result.status == DORA_STREAM_STATUS_READY) {
            status = dora_identity_claim_transfer(state->storage.pool, &output.handle, IDENTITY_KIND_VECTOR);
        }
        if (status != DORA_OK) {
            state->failure = status;
            state->error = error;
        } else {
            state->is_end = result.status == DORA_STREAM_STATUS_END;
        }
    }
    if (status != DORA_OK) {
        dora_vector_release(&output);
    }
    const bool release_pending = state->is_release_pending;
    reader_leave(slot, owner);
    if (status == DORA_OK && result.status == DORA_STREAM_STATUS_READY) {
        status = dora_identity_claim_transfer((struct dora_pool){}, &output.handle, IDENTITY_KIND_VECTOR);
    }
    if (status == DORA_OK && !dora_vector_is_empty(out_vector)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        *out_vector = output;
        *out_result = result;
        return DORA_OK;
    } else if (!release_pending) {
        state->failure = status;
        state->error = error;
    }
    dora_vector_release(&output);
    if (out_error != nullptr && status != DORA_OK) {
        *out_error = error;
    }
    return status;
}
