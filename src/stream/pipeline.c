// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/vector.h"
#include "identity/identity.h"
#include "memory/memory_internal.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

struct pipeline_stage {
    struct dora_vector_transform* transform;
    struct dora_vector pending;
    bool is_end;
};

struct pipeline_state {
    struct dora_allocation storage;
    struct identity_claim_state claims;
    struct dora_reader reader;
    struct dora_writer writer;
    struct dora_pipeline_observation observation;
    struct dora_error error;
    uint32_t transform_count;
    bool is_input_ended;
    bool is_release_pending;
    struct pipeline_stage stages[];
};

static bool pipeline_is_empty(const struct dora_pipeline* pipeline) {
    return pipeline != nullptr && handle_is_empty(pipeline->handle);
}

static bool pipeline_is_terminal(const struct pipeline_state* state) {
    return state->observation.progress == DORA_PIPELINE_PROGRESS_FINISHED ||
           state->observation.progress == DORA_PIPELINE_PROGRESS_CANCELLED ||
           state->observation.progress == DORA_PIPELINE_PROGRESS_FAILED;
}

static void transforms_release(struct dora_vector_transform* const* transforms, uint32_t count) {
    for (uint32_t index = 0; transforms != nullptr && index < count; ++index) {
        bool duplicate = false;
        for (uint32_t prior = 0; prior < index; ++prior) {
            duplicate = duplicate || transforms[prior] == transforms[index];
        }
        if (!duplicate && transforms[index] != nullptr && transforms[index]->release != nullptr) {
            transforms[index]->release(transforms[index]);
        }
    }
}

static void pipeline_cleanup(struct pipeline_state* state) {
    for (uint32_t index = 0; index < state->transform_count; ++index) {
        struct dora_vector_transform* const transform = state->stages[index].transform;
        state->stages[index].transform = nullptr;
        if (transform != nullptr) {
            transform->release(transform);
        }
    }
    for (uint32_t index = 0; index <= state->transform_count; ++index) {
        dora_vector_release(&state->stages[index].pending);
    }
    dora_reader_release(&state->reader);
    dora_writer_release(&state->writer);
}

static void pipeline_discard(struct pipeline_state* state) {
    if (state != nullptr) {
        const struct dora_pool pool = state->storage.pool;
        dora_pool_callback_enter(pool);
        pipeline_cleanup(state);
        struct dora_allocation storage = state->storage;
        dora_allocation_release(&storage);
        dora_pool_callback_leave(pool);
    }
}

static int pipeline_enter(const struct dora_pipeline* pipeline,
                          struct pipeline_state** out_state,
                          struct identity_slot** out_slot) {
    if (pipeline == nullptr) {
        return EINVAL;
    }
    const int status = dora_identity_slot_enter(&pipeline->handle, IDENTITY_KIND_PIPELINE, out_slot);
    if (status == DORA_OK) {
        *out_state = (*out_slot)->claims->payload;
    }
    return status;
}

static void pipeline_observe(struct pipeline_state* state,
                             enum dora_pipeline_progress progress,
                             enum dora_pipeline_origin origin,
                             int status,
                             struct dora_readiness readiness) {
    state->observation = (struct dora_pipeline_observation){
        .progress = progress,
        .origin = origin,
        .status = status,
        .readiness = readiness
    };
}

static void pipeline_terminal(struct pipeline_state* state,
                              enum dora_pipeline_progress progress,
                              enum dora_pipeline_origin origin,
                              int status) {
    pipeline_observe(state, progress, origin, status, (struct dora_readiness){});
    for (uint32_t index = 0; index <= state->transform_count; ++index) {
        dora_vector_release(&state->stages[index].pending);
    }
}

static int execution_owner(const struct dora_handle* handle, enum identity_kind kind) {
    const struct identity_slot* slot = nullptr;
    if (!handle->is_owner || dora_identity_resolve(handle, kind, &slot) != DORA_OK) {
        return EINVAL;
    }
    return slot->is_running ? EBUSY : DORA_OK;
}

static void pipeline_stop_upstream(struct pipeline_state* state, uint32_t last) {
    state->is_input_ended = true;
    for (uint32_t index = 0; index <= last; ++index) {
        dora_vector_release(&state->stages[index].pending);
        state->stages[index].is_end = true;
    }
}

static int pipeline_transform(struct pipeline_state* state, uint32_t index, bool is_end, struct dora_error* error) {
    struct pipeline_stage* const stage = &state->stages[index];
    const struct dora_vector input = dora_vector_borrow(&stage->pending);
    const bool has_input = !dora_vector_is_empty(&input);
    struct dora_vector output = {};
    struct dora_stream_result result = {};
    int status = stage->transform->apply(stage->transform,
                                         state->storage.pool,
                                         has_input ? &input : nullptr,
                                         is_end,
                                         &output,
                                         &result,
                                         error);
    bool contract = false;
    if (status == DORA_OK) {
        const bool present = !dora_vector_is_empty(&output);
        contract =
            result.status > DORA_STREAM_STATUS_END ||
            (present && (!output.handle.is_owner || !dora_vector_is_valid(&output))) ||
            (result.status == DORA_STREAM_STATUS_BLOCKED && (present || result.is_input_consumed)) ||
            (result.status == DORA_STREAM_STATUS_READY && !present && (!result.is_input_consumed || !has_input)) ||
            (present && input.handle.slot_index != 0 && dora_handle_is_same(input.handle, output.handle));
        if (contract) {
            status = EINVAL;
        } else if (present) {
            status = dora_identity_claim_transfer(state->storage.pool, &output.handle, IDENTITY_KIND_VECTOR);
        }
    }
    if (status != DORA_OK) {
        if (input.handle.slot_index != 0 && dora_handle_is_same(input.handle, output.handle)) {
            output.handle.is_owner = false;
        }
        dora_vector_release(&output);
        state->error = *error;
        pipeline_terminal(state,
                          DORA_PIPELINE_PROGRESS_FAILED,
                          contract ? DORA_PIPELINE_ORIGIN_CONTRACT : DORA_PIPELINE_ORIGIN_TRANSFORM,
                          status);
        return status;
    }
    state->stages[index + 1].pending = output;
    if (result.status == DORA_STREAM_STATUS_END) {
        pipeline_stop_upstream(state, index);
    } else if (result.is_input_consumed) {
        dora_vector_release(&stage->pending);
    }
    pipeline_observe(state,
                     result.status == DORA_STREAM_STATUS_BLOCKED ? DORA_PIPELINE_PROGRESS_BLOCKED
                                                                 : DORA_PIPELINE_PROGRESS_ADVANCED,
                     DORA_PIPELINE_ORIGIN_TRANSFORM,
                     DORA_OK,
                     result.readiness);
    return DORA_OK;
}

static int pipeline_drive(struct pipeline_state* state, struct dora_error* error) {
    if (pipeline_is_terminal(state)) {
        *error = state->error;
        return state->observation.status;
    }
    const uint32_t count = state->transform_count;
    for (uint32_t cursor = count + 1; cursor != 0;) {
        const uint32_t index = --cursor;
        if (state->stages[index].pending.handle.slot_index == 0) {
            continue;
        }
        if (index < count) {
            const bool is_end =
                (index == 0 && state->is_input_ended) || (index != 0 && state->stages[index - 1].is_end);
            return pipeline_transform(state, index, is_end, error);
        }
        if (state->writer.handle.slot_index == 0) {
            pipeline_observe(state,
                             DORA_PIPELINE_PROGRESS_HAS_OUTPUT,
                             DORA_PIPELINE_ORIGIN_CALLER,
                             DORA_OK,
                             (struct dora_readiness){});
            return DORA_OK;
        }
        struct dora_stream_result result = {};
        const int status = dora_writer_push(&state->writer, &state->stages[count].pending, &result, error);
        if (status != DORA_OK) {
            state->error = *error;
            pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FAILED, DORA_PIPELINE_ORIGIN_WRITER, status);
            return status;
        }
        if (result.status != DORA_STREAM_STATUS_BLOCKED) {
            dora_vector_release(&state->stages[count].pending);
        }
        if (result.status == DORA_STREAM_STATUS_END) {
            pipeline_stop_upstream(state, count);
        }
        pipeline_observe(state,
                         result.status == DORA_STREAM_STATUS_BLOCKED ? DORA_PIPELINE_PROGRESS_BLOCKED
                                                                     : DORA_PIPELINE_PROGRESS_ADVANCED,
                         DORA_PIPELINE_ORIGIN_WRITER,
                         DORA_OK,
                         result.readiness);
        return DORA_OK;
    }
    if (state->is_input_ended) {
        for (uint32_t index = 0; !state->stages[count].is_end && index < count; ++index) {
            if (!state->stages[index].is_end) {
                return pipeline_transform(state, index, true, error);
            }
        }
        if (state->writer.handle.slot_index != 0) {
            struct dora_stream_result result = {};
            const int status = dora_writer_finish(&state->writer, &result, error);
            if (status != DORA_OK) {
                state->error = *error;
                pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FAILED, DORA_PIPELINE_ORIGIN_WRITER, status);
                return status;
            }
            if (result.status == DORA_STREAM_STATUS_BLOCKED) {
                pipeline_observe(state,
                                 DORA_PIPELINE_PROGRESS_BLOCKED,
                                 DORA_PIPELINE_ORIGIN_WRITER,
                                 DORA_OK,
                                 result.readiness);
                return DORA_OK;
            }
        }
        pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FINISHED, DORA_PIPELINE_ORIGIN_NONE, DORA_OK);
        return DORA_OK;
    }
    if (state->reader.handle.slot_index == 0) {
        pipeline_observe(state,
                         DORA_PIPELINE_PROGRESS_NEEDS_INPUT,
                         DORA_PIPELINE_ORIGIN_CALLER,
                         DORA_OK,
                         (struct dora_readiness){});
        return DORA_OK;
    }
    struct dora_stream_result result = {};
    struct dora_vector output = {};
    int status = dora_reader_next(&state->reader, &output, &result, error);
    if (status == DORA_OK && output.handle.slot_index != 0) {
        status = dora_identity_claim_transfer(state->storage.pool, &output.handle, IDENTITY_KIND_VECTOR);
    }
    if (status != DORA_OK) {
        dora_vector_release(&output);
        state->error = *error;
        pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FAILED, DORA_PIPELINE_ORIGIN_READER, status);
        return status;
    }
    state->stages[0].pending = output;
    state->is_input_ended = result.status == DORA_STREAM_STATUS_END;
    pipeline_observe(state,
                     result.status == DORA_STREAM_STATUS_BLOCKED ? DORA_PIPELINE_PROGRESS_BLOCKED
                                                                 : DORA_PIPELINE_PROGRESS_ADVANCED,
                     DORA_PIPELINE_ORIGIN_READER,
                     DORA_OK,
                     result.readiness);
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Lifecycle
//
//------------------------------------------------------------------------------

int dora_pipeline_init(struct dora_pipeline* pipeline,
                       struct dora_pool pool,
                       struct dora_reader* in_out_reader,
                       struct dora_vector_transform* const* transforms,
                       uint32_t transform_count,
                       struct dora_writer* in_out_writer) {
    const bool has_reader = in_out_reader != nullptr && in_out_reader->handle.slot_index != 0;
    const bool has_writer = in_out_writer != nullptr && in_out_writer->handle.slot_index != 0;
    if (!pipeline_is_empty(pipeline) || transform_count == UINT32_MAX ||
        (transform_count != 0 && transforms == nullptr)) {
        transforms_release(transforms, transform_count);
        return EINVAL;
    }
    if (has_reader) {
        const int status = execution_owner(&in_out_reader->handle, IDENTITY_KIND_READER);
        if (status != DORA_OK) {
            transforms_release(transforms, transform_count);
            return status;
        }
    }
    if (has_writer) {
        const int status = execution_owner(&in_out_writer->handle, IDENTITY_KIND_WRITER);
        if (status != DORA_OK) {
            transforms_release(transforms, transform_count);
            return status;
        }
    }
    for (uint32_t index = 0; index < transform_count; ++index) {
        if (transforms[index] == nullptr || transforms[index]->apply == nullptr ||
            transforms[index]->release == nullptr) {
            transforms_release(transforms, transform_count);
            return EINVAL;
        }
        for (uint32_t prior = 0; prior < index; ++prior) {
            if (transforms[prior] == transforms[index]) {
                transforms_release(transforms, transform_count);
                return EINVAL;
            }
        }
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, (size_t)transform_count + 1, sizeof(struct pipeline_stage)) ||
        ckd_add(&bytes, bytes, sizeof(struct pipeline_state))) {
        transforms_release(transforms, transform_count);
        return EOVERFLOW;
    }
    struct dora_allocation storage = {};
    int status = dora_memory_payload_allocate(pool, bytes, &storage);
    if (status != DORA_OK) {
        transforms_release(transforms, transform_count);
        return status;
    }
    struct pipeline_state* const state = storage.address;
    *state = (struct pipeline_state){.storage = storage, .transform_count = transform_count};
    state->claims.payload = state;
    for (uint32_t index = 0; index <= transform_count; ++index) {
        state->stages[index] = (struct pipeline_stage){};
        if (index < transform_count) {
            state->stages[index].transform = transforms[index];
        }
    }
    struct dora_pipeline candidate = {};
    bool reader_transferred = false;
    bool writer_transferred = false;
    if (has_reader) {
        status = dora_identity_claim_transfer(pool, &in_out_reader->handle, IDENTITY_KIND_READER);
        reader_transferred = status == DORA_OK;
    }
    if (status == DORA_OK && has_writer) {
        status = dora_identity_claim_transfer(pool, &in_out_writer->handle, IDENTITY_KIND_WRITER);
        writer_transferred = status == DORA_OK;
    }
    if (status == DORA_OK) {
        status = dora_identity_slot_publish(pool, IDENTITY_KIND_PIPELINE, &state->claims, &candidate.handle);
    }
    if (status != DORA_OK) {
        if (reader_transferred) {
            const int rollback =
                dora_identity_claim_transfer((struct dora_pool){}, &in_out_reader->handle, IDENTITY_KIND_READER);
            if (rollback != DORA_OK) {
                status = rollback;
            }
        }
        if (writer_transferred) {
            const int rollback =
                dora_identity_claim_transfer((struct dora_pool){}, &in_out_writer->handle, IDENTITY_KIND_WRITER);
            if (rollback != DORA_OK) {
                status = rollback;
            }
        }
        pipeline_discard(state);
        return status;
    }
    if (has_reader) {
        state->reader = *in_out_reader;
        *in_out_reader = (struct dora_reader){};
    }
    if (has_writer) {
        state->writer = *in_out_writer;
        *in_out_writer = (struct dora_writer){};
    }
    *pipeline = candidate;
    return DORA_OK;
}

int dora_pipeline_move(struct dora_pipeline* in_out_source, struct dora_pipeline* out_destination) {
    if (in_out_source == nullptr || !in_out_source->handle.is_owner || !pipeline_is_empty(out_destination)) {
        return EINVAL;
    }
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    const int status = pipeline_enter(in_out_source, &state, &slot);
    if (status == DORA_OK) {
        *out_destination = *in_out_source;
        *in_out_source = (struct dora_pipeline){};
        dora_identity_slot_leave(&out_destination->handle, slot);
    }
    return status;
}

void dora_pipeline_release(struct dora_pipeline* pipeline) {
    if (pipeline == nullptr) {
        return;
    }
    struct dora_handle owner = pipeline->handle;
    *pipeline = (struct dora_pipeline){};
    const struct identity_slot* slot = nullptr;
    if (owner.is_owner && dora_identity_resolve(&owner, IDENTITY_KIND_PIPELINE, &slot) == DORA_OK && slot->is_running) {
        struct pipeline_state* const state = slot->claims->payload;
        state->is_release_pending = true;
        return;
    }
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_PIPELINE, &retired);
    pipeline_discard(retired);
}

static void pipeline_leave(struct identity_slot* slot, struct dora_handle owner) {
    const struct pipeline_state* const state = slot->claims->payload;
    const bool release_pending = state->is_release_pending;
    dora_identity_slot_leave(&owner, slot);
    if (release_pending) {
        owner.is_owner = true;
        struct dora_pipeline pipeline = {.handle = owner};
        dora_pipeline_release(&pipeline);
    }
}

int dora_pipeline_cancel(struct dora_pipeline* pipeline) {
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    const int status = pipeline_enter(pipeline, &state, &slot);
    if (status == DORA_OK) {
        const struct dora_handle owner = pipeline->handle;
        if (!pipeline_is_terminal(state)) {
            pipeline_terminal(state, DORA_PIPELINE_PROGRESS_CANCELLED, DORA_PIPELINE_ORIGIN_NONE, DORA_OK);
        }
        pipeline_leave(slot, owner);
    }
    return status;
}

int dora_pipeline_fail(struct dora_pipeline* pipeline, int status) {
    if (status == DORA_OK) {
        return EINVAL;
    }
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    const int result = pipeline_enter(pipeline, &state, &slot);
    if (result == DORA_OK) {
        const struct dora_handle owner = pipeline->handle;
        if (!pipeline_is_terminal(state)) {
            pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FAILED, DORA_PIPELINE_ORIGIN_CALLER, status);
        }
        pipeline_leave(slot, owner);
    }
    return result;
}

bool dora_pipeline_is_valid(const struct dora_pipeline* pipeline) {
    const struct identity_slot* slot = nullptr;
    return pipeline != nullptr && dora_identity_resolve(&pipeline->handle, IDENTITY_KIND_PIPELINE, &slot) == DORA_OK;
}

//------------------------------------------------------------------------------
//
// Input and Output
//
//------------------------------------------------------------------------------

int dora_pipeline_supply(struct dora_pipeline* pipeline, struct dora_vector* in_out_vector) {
    if (in_out_vector == nullptr || !in_out_vector->handle.is_owner || !dora_vector_is_valid(in_out_vector)) {
        return EINVAL;
    }
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    int status = pipeline_enter(pipeline, &state, &slot);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_handle owner = pipeline->handle;
    if (pipeline_is_terminal(state) || state->is_input_ended || state->reader.handle.slot_index != 0) {
        status = EINVAL;
    } else {
        for (uint32_t index = 0; index <= state->transform_count; ++index) {
            if (state->stages[index].pending.handle.slot_index != 0) {
                status = EBUSY;
            }
        }
    }
    if (status == DORA_OK) {
        status = dora_identity_claim_transfer(state->storage.pool, &in_out_vector->handle, IDENTITY_KIND_VECTOR);
    }
    if (status == DORA_OK) {
        state->stages[0].pending = *in_out_vector;
        *in_out_vector = (struct dora_vector){};
        pipeline_observe(state,
                         DORA_PIPELINE_PROGRESS_ADVANCED,
                         DORA_PIPELINE_ORIGIN_CALLER,
                         DORA_OK,
                         (struct dora_readiness){});
    }
    pipeline_leave(slot, owner);
    return status;
}

int dora_pipeline_end_input(struct dora_pipeline* pipeline) {
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    int status = pipeline_enter(pipeline, &state, &slot);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_handle owner = pipeline->handle;
    if (pipeline_is_terminal(state) || state->reader.handle.slot_index != 0) {
        status = EINVAL;
    } else {
        state->is_input_ended = true;
    }
    pipeline_leave(slot, owner);
    return status;
}

int dora_pipeline_take(struct dora_pipeline* pipeline, struct dora_vector* out_vector) {
    if (!dora_vector_is_empty(out_vector)) {
        return EINVAL;
    }
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    int status = pipeline_enter(pipeline, &state, &slot);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_handle owner = pipeline->handle;
    struct dora_vector* const output = &state->stages[state->transform_count].pending;
    if (pipeline_is_terminal(state) || state->writer.handle.slot_index != 0) {
        status = EINVAL;
    } else if (output->handle.slot_index == 0) {
        status = EAGAIN;
    } else {
        status = dora_identity_claim_transfer((struct dora_pool){}, &output->handle, IDENTITY_KIND_VECTOR);
        if (status == DORA_OK) {
            *out_vector = *output;
            *output = (struct dora_vector){};
            pipeline_observe(state,
                             DORA_PIPELINE_PROGRESS_ADVANCED,
                             DORA_PIPELINE_ORIGIN_CALLER,
                             DORA_OK,
                             (struct dora_readiness){});
        }
    }
    pipeline_leave(slot, owner);
    return status;
}

//------------------------------------------------------------------------------
//
// Execution and Observation
//
//------------------------------------------------------------------------------

int dora_pipeline_step(struct dora_pipeline* pipeline,
                       struct dora_pipeline_observation* out_observation,
                       struct dora_error* out_error) {
    if (out_observation == nullptr) {
        return EINVAL;
    }
    struct pipeline_state* state = nullptr;
    struct identity_slot* slot = nullptr;
    int status = pipeline_enter(pipeline, &state, &slot);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_handle owner = pipeline->handle;
    struct dora_error error = {};
    status = pipeline_drive(state, &error);
    if (status == DORA_OK && state->observation.progress == DORA_PIPELINE_PROGRESS_ADVANCED &&
        state->writer.handle.slot_index == 0) {
        if (state->stages[state->transform_count].pending.handle.slot_index != 0) {
            pipeline_observe(state,
                             DORA_PIPELINE_PROGRESS_HAS_OUTPUT,
                             DORA_PIPELINE_ORIGIN_CALLER,
                             DORA_OK,
                             (struct dora_readiness){});
        } else if (state->transform_count == 0 && state->is_input_ended) {
            pipeline_terminal(state, DORA_PIPELINE_PROGRESS_FINISHED, DORA_PIPELINE_ORIGIN_NONE, DORA_OK);
        }
    }
    *out_observation = state->observation;
    if (status != DORA_OK && out_error != nullptr) {
        *out_error = error;
    }
    pipeline_leave(slot, owner);
    return status;
}

int dora_pipeline_get_observation(const struct dora_pipeline* pipeline,
                                  struct dora_pipeline_observation* out_observation) {
    const struct identity_slot* slot = nullptr;
    if (pipeline == nullptr || out_observation == nullptr ||
        dora_identity_resolve(&pipeline->handle, IDENTITY_KIND_PIPELINE, &slot) != DORA_OK) {
        return EINVAL;
    }
    if (slot->is_running) {
        return EBUSY;
    }
    const struct pipeline_state* const state = slot->claims->payload;
    *out_observation = state->observation;
    return DORA_OK;
}
