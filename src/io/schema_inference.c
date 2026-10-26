// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/io/schema_inference.h"

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "identity/identity.h"
#include "io/csv/csv_inference.h"
#include "io/json/json_inference.h"
#include "io/schema_inference_internal.h"
#include "io/text_reader.h"
#include "memory/memory_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

bool dora_text_schema_inference_is_empty(const struct dora_schema_inference* inference) {
    return inference != nullptr && handle_is_empty(inference->handle);
}

static bool text_outputs_are_empty(const struct dora_schema* schema, const struct dora_reader* reader) {
    return schema != nullptr && reader != nullptr && handle_is_empty(schema->handle) && schema->node_index == 0 &&
           handle_is_empty(reader->handle);
}

static bool text_ranges_overlap(const void* first, size_t first_size, const void* second, size_t second_size) {
    const uintptr_t left = (uintptr_t)first;
    const uintptr_t right = (uintptr_t)second;
    if (left <= right) {
        return right - left < first_size;
    }
    return left - right < second_size;
}

static void text_schema_inference_clear(struct text_schema_inference_state* state) {
    if (state->format == TEXT_FORMAT_CSV) {
        dora_csv_schema_inference_cleanup(state);
    } else {
        dora_json_schema_inference_cleanup(state);
    }
}

static void text_schema_inference_discard(struct text_schema_inference_state* state) {
    const struct dora_pool pool = state->allocation.pool;
    dora_pool_callback_enter(pool);
    text_schema_inference_clear(state);
    struct dora_allocation allocation = state->allocation;
    dora_allocation_release(&allocation);
    dora_pool_callback_leave(pool);
}

int dora_schema_inference_move(struct dora_schema_inference* in_out_source,
                               struct dora_schema_inference* out_destination) {
    if (in_out_source == nullptr || !in_out_source->handle.is_owner ||
        !dora_text_schema_inference_is_empty(out_destination) ||
        text_ranges_overlap(in_out_source, sizeof(*in_out_source), out_destination, sizeof(*out_destination))) {
        return EINVAL;
    }
    struct identity_slot* slot = nullptr;
    const int status = dora_identity_slot_enter(&in_out_source->handle, IDENTITY_KIND_SCHEMA_INFERENCE, &slot);
    if (status != DORA_OK) {
        return status;
    }
    const struct dora_handle owner = in_out_source->handle;
    *out_destination = *in_out_source;
    *in_out_source = (struct dora_schema_inference){};
    dora_identity_slot_leave(&owner, slot);
    return DORA_OK;
}

void dora_schema_inference_release(struct dora_schema_inference* inference) {
    if (inference == nullptr) {
        return;
    }
    struct dora_handle owner = inference->handle;
    *inference = (struct dora_schema_inference){};
    const struct identity_slot* slot = nullptr;
    if (owner.is_owner && dora_identity_resolve(&owner, IDENTITY_KIND_SCHEMA_INFERENCE, &slot) == DORA_OK &&
        slot->is_running) {
        struct text_schema_inference_state* const state = slot->claims->payload;
        state->is_release_pending = true;
        return;
    }
    void* retired = nullptr;
    dora_identity_claim_release(&owner, IDENTITY_KIND_SCHEMA_INFERENCE, &retired);
    if (retired != nullptr) {
        text_schema_inference_discard(retired);
    }
}

int dora_text_schema_inference_publish(struct dora_schema_inference* inference,
                                       struct text_schema_inference_state* state,
                                       struct text_input* text,
                                       struct dora_byte_source* input,
                                       const struct dora_schema_inference_options* options) {
    const struct dora_pool pool = state->allocation.pool;
    int status =
        dora_text_schema_inference_is_empty(inference) && input != nullptr && input->read != nullptr ? DORA_OK : EINVAL;
    if (status == DORA_OK) {
        text->input = input;
        text->should_retain = true;
        text->replay_byte_limit = options != nullptr ? options->max_replay_bytes : 0;
        state->sample_row_limit =
            options != nullptr && options->sample_row_limit != 0 ? options->sample_row_limit : UINT64_C(20480);
        state->claims.payload = state;
    }
    struct dora_schema_inference temporary = {};
    if (status == DORA_OK) {
        status = dora_identity_slot_publish(pool, IDENTITY_KIND_SCHEMA_INFERENCE, &state->claims, &temporary.handle);
    }
    if (status == DORA_OK && !dora_text_schema_inference_is_empty(inference)) {
        status = EINVAL;
    }
    if (status != DORA_OK) {
        if (temporary.handle.is_owner) {
            dora_schema_inference_release(&temporary);
        } else {
            text_schema_inference_discard(state);
        }
        return status;
    }
    *inference = temporary;
    return DORA_OK;
}

static void text_schema_inference_leave(struct identity_slot* slot, struct dora_handle owner) {
    const struct text_schema_inference_state* const state = slot->claims->payload;
    const bool pending = state->is_release_pending;
    dora_identity_slot_leave(&owner, slot);
    if (pending) {
        owner.is_owner = true;
        struct dora_schema_inference temporary = {.handle = owner};
        dora_schema_inference_release(&temporary);
    }
}

bool dora_schema_inference_is_valid(const struct dora_schema_inference* inference) {
    const struct identity_slot* slot = nullptr;
    return inference != nullptr &&
           dora_identity_resolve(&inference->handle, IDENTITY_KIND_SCHEMA_INFERENCE, &slot) == DORA_OK;
}

int dora_schema_inference_next(struct dora_schema_inference* inference,
                               struct dora_schema* out_schema,
                               struct dora_reader* out_reader,
                               struct dora_stream_result* out_result,
                               struct dora_error* out_error) {
    if (inference == nullptr || !inference->handle.is_owner || !text_outputs_are_empty(out_schema, out_reader) ||
        out_result == nullptr ||
        text_ranges_overlap(out_schema, sizeof(*out_schema), out_reader, sizeof(*out_reader)) ||
        text_ranges_overlap(inference, sizeof(*inference), out_schema, sizeof(*out_schema)) ||
        text_ranges_overlap(inference, sizeof(*inference), out_reader, sizeof(*out_reader)) ||
        text_ranges_overlap(out_result, sizeof(*out_result), inference, sizeof(*inference)) ||
        text_ranges_overlap(out_result, sizeof(*out_result), out_schema, sizeof(*out_schema)) ||
        text_ranges_overlap(out_result, sizeof(*out_result), out_reader, sizeof(*out_reader)) ||
        (out_error != nullptr &&
         (text_ranges_overlap(out_error, sizeof(*out_error), inference, sizeof(*inference)) ||
          text_ranges_overlap(out_error, sizeof(*out_error), out_schema, sizeof(*out_schema)) ||
          text_ranges_overlap(out_error, sizeof(*out_error), out_reader, sizeof(*out_reader)) ||
          text_ranges_overlap(out_error, sizeof(*out_error), out_result, sizeof(*out_result))))) {
        return EINVAL;
    }
    const struct dora_handle owner = inference->handle;
    struct identity_slot* slot = nullptr;
    int status = dora_identity_slot_enter(&owner, IDENTITY_KIND_SCHEMA_INFERENCE, &slot);
    if (status != DORA_OK) {
        return status;
    }
    struct text_schema_inference_state* const state = slot->claims->payload;
    struct dora_schema schema = {};
    struct dora_reader reader = {};
    struct dora_stream_result flow = {.status = DORA_STREAM_STATUS_END};
    if (state->phase == TEXT_SCHEMA_INFERENCE_FAILED) {
        status = state->failure;
    } else if (state->phase == TEXT_SCHEMA_INFERENCE_SCAN) {
        struct text_scan_result scan = {};
        status = state->format == TEXT_FORMAT_CSV ? dora_csv_schema_inference_scan(state, &scan)
                                                  : dora_json_schema_inference_scan(state, &scan);
        if (status == DORA_OK && scan.is_ready) {
            status = state->format == TEXT_FORMAT_CSV ? dora_csv_schema_inference_handoff(state, &schema, &reader)
                                                      : dora_json_schema_inference_handoff(state, &schema, &reader);
            if (status == DORA_OK) {
                text_schema_inference_clear(state);
                if (!text_outputs_are_empty(out_schema, out_reader)) {
                    status = EINVAL;
                } else {
                    state->phase = TEXT_SCHEMA_INFERENCE_HANDED_OFF;
                    flow.status = DORA_STREAM_STATUS_READY;
                }
            }
        } else if (status == DORA_OK && scan.is_blocked) {
            flow.status = DORA_STREAM_STATUS_BLOCKED;
        } else if (status == DORA_OK) {
            status = EINVAL;
        }
    }
    if (status != DORA_OK) {
        dora_schema_release(&schema);
        dora_reader_release(&reader);
        if (state->phase != TEXT_SCHEMA_INFERENCE_FAILED) {
            state->failure = status;
            state->phase = TEXT_SCHEMA_INFERENCE_FAILED;
            memcpy(state->error.message, "Text discovery failed", sizeof("Text discovery failed"));
            text_schema_inference_clear(state);
        }
        if (out_error != nullptr) {
            *out_error = state->error;
        }
    }
    text_schema_inference_leave(slot, owner);
    if (status == DORA_OK && flow.status == DORA_STREAM_STATUS_READY) {
        if (text_outputs_are_empty(out_schema, out_reader)) {
            *out_schema = schema;
            *out_reader = reader;
        } else {
            status = EINVAL;
            dora_schema_release(&schema);
            dora_reader_release(&reader);
        }
    }
    if (status == DORA_OK) {
        *out_result = flow;
    }
    return status;
}
