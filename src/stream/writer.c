// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"

#include <errno.h>

struct writer_state {
    struct dora_allocation storage;
    struct identity_claim_state claims;
    struct dora_vector_sink* sink;
    struct dora_vector retry_input;
    struct dora_error error;
    int failure;
    bool is_input_closed;
    bool is_finished;
    bool is_release_pending;
};

static void sink_release(struct dora_vector_sink* sink) {
    if (sink != nullptr && sink->release != nullptr) {
        sink->release(sink);
    }
}

static void writer_discard(struct writer_state* state) {
    if (state != nullptr) {
        const struct dora_pool pool = state->storage.pool;
        dora_pool_callback_enter(pool);
        struct dora_vector_sink* const sink = state->sink;
        state->sink = nullptr;
        sink_release(sink);
        struct dora_allocation storage = state->storage;
        dora_allocation_release(&storage);
        dora_pool_callback_leave(pool);
    }
}

static bool writer_is_empty(const struct dora_writer* writer) {
    return writer != nullptr && handle_is_empty(writer->handle);
}

static bool inputs_are_same(const struct dora_vector* first, const struct dora_vector* second) {
    return dora_handle_is_same(first->handle, second->handle) && first->rows.offset == second->rows.offset &&
           first->rows.length == second->rows.length;
}

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

int dora_writer_init(struct dora_writer* writer, struct dora_pool pool, struct dora_vector_sink* sink) {
    if (!writer_is_empty(writer) || sink == nullptr || sink->push == nullptr || sink->finish == nullptr ||
        sink->release == nullptr) {
        sink_release(sink);
        return EINVAL;
    }
    struct dora_allocation storage = {};
    int status = dora_memory_payload_allocate(pool, sizeof(struct writer_state), &storage);
    if (status != DORA_OK) {
        sink_release(sink);
        return status;
    }
    struct writer_state* const state = storage.address;
    *state = (struct writer_state){.storage = storage, .sink = sink};
    state->claims.payload = state;
    status = dora_identity_slot_publish(pool, IDENTITY_KIND_WRITER, &state->claims, &writer->handle);
    if (status != DORA_OK) {
        writer_discard(state);
    }
    return status;
}

int dora_writer_move(struct dora_writer* in_out_source, struct dora_writer* out_destination) {
    if (in_out_source == nullptr || !in_out_source->handle.is_owner || !writer_is_empty(out_destination)) {
        return EINVAL;
    }
    struct identity_slot* slot = nullptr;
    const int status = dora_identity_slot_enter(&in_out_source->handle, IDENTITY_KIND_WRITER, &slot);
    if (status == DORA_OK) {
        *out_destination = *in_out_source;
        *in_out_source = (struct dora_writer){};
        dora_identity_slot_leave(&out_destination->handle, slot);
    }
    return status;
}

void dora_writer_release(struct dora_writer* writer) {
    if (writer == nullptr) {
        return;
    }
    struct dora_handle owner = writer->handle;
    *writer = (struct dora_writer){};
    const struct identity_slot* slot = nullptr;
    if (owner.is_owner && dora_identity_resolve(&owner, IDENTITY_KIND_WRITER, &slot) == DORA_OK && slot->is_running) {
        struct writer_state* const state = slot->claims->payload;
        state->is_release_pending = true;
        return;
    }
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_WRITER, &retired);
    writer_discard(retired);
}

static void writer_leave(struct identity_slot* slot, struct dora_handle owner) {
    const struct writer_state* const state = slot->claims->payload;
    const bool release_pending = state->is_release_pending;
    dora_identity_slot_leave(&owner, slot);
    if (release_pending) {
        owner.is_owner = true;
        struct dora_writer writer = {.handle = owner};
        dora_writer_release(&writer);
    }
}

bool dora_writer_is_valid(const struct dora_writer* writer) {
    const struct identity_slot* slot = nullptr;
    return writer != nullptr && dora_identity_resolve(&writer->handle, IDENTITY_KIND_WRITER, &slot) == DORA_OK;
}

//------------------------------------------------------------------------------
//
// Writing
//
//------------------------------------------------------------------------------

int dora_writer_push(struct dora_writer* writer,
                     const struct dora_vector* input,
                     struct dora_stream_result* out_result,
                     struct dora_error* out_error) {
    if (writer == nullptr || out_result == nullptr || !dora_vector_is_valid(input)) {
        return EINVAL;
    }
    const struct dora_handle owner = writer->handle;
    struct identity_slot* slot = nullptr;
    int status = dora_identity_slot_enter(&owner, IDENTITY_KIND_WRITER, &slot);
    if (status != DORA_OK) {
        return status;
    }
    struct writer_state* const state = slot->claims->payload;
    bool called = false;
    struct dora_stream_result result = {};
    struct dora_error error = {};
    if (state->failure != DORA_OK) {
        status = state->failure;
        error = state->error;
    } else if (state->is_input_closed ||
               (state->retry_input.handle.slot_index != 0 && !inputs_are_same(input, &state->retry_input))) {
        status = EINVAL;
    } else {
        const struct dora_vector borrowed = dora_vector_borrow(input);
        state->retry_input = borrowed;
        called = true;
        status = state->sink->push(state->sink, &borrowed, &result, &error);
        if (status == DORA_OK && result.status > DORA_STREAM_STATUS_END) {
            status = EINVAL;
        }
    }
    if (called) {
        if (status != DORA_OK) {
            state->failure = status;
            state->error = error;
        }
        if (status != DORA_OK || result.status != DORA_STREAM_STATUS_BLOCKED) {
            state->retry_input = (struct dora_vector){};
        }
        if (status == DORA_OK) {
            result.is_input_consumed = result.status != DORA_STREAM_STATUS_BLOCKED;
            state->is_input_closed = result.status == DORA_STREAM_STATUS_END;
            *out_result = result;
        }
    }
    if (out_error != nullptr && status != DORA_OK) {
        *out_error = error;
    }
    writer_leave(slot, owner);
    return status;
}

int dora_writer_finish(struct dora_writer* writer,
                       struct dora_stream_result* out_result,
                       struct dora_error* out_error) {
    if (writer == nullptr || out_result == nullptr) {
        return EINVAL;
    }
    const struct dora_handle owner = writer->handle;
    struct identity_slot* slot = nullptr;
    int status = dora_identity_slot_enter(&owner, IDENTITY_KIND_WRITER, &slot);
    if (status != DORA_OK) {
        return status;
    }
    struct writer_state* const state = slot->claims->payload;
    struct dora_stream_result result = {.status = DORA_STREAM_STATUS_END};
    struct dora_error error = {};
    if (state->failure != DORA_OK) {
        status = state->failure;
        error = state->error;
    } else if (state->retry_input.handle.slot_index != 0) {
        status = EBUSY;
    } else if (!state->is_finished) {
        state->is_input_closed = true;
        status = state->sink->finish(state->sink, &result, &error);
        if (status == DORA_OK && result.status > DORA_STREAM_STATUS_END) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            state->is_finished = result.status != DORA_STREAM_STATUS_BLOCKED;
        } else {
            state->failure = status;
            state->error = error;
        }
    }
    if (status == DORA_OK) {
        *out_result = result;
    } else if (out_error != nullptr) {
        *out_error = error;
    }
    writer_leave(slot, owner);
    return status;
}
