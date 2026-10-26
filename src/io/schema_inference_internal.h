// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Text schema inference state.

#ifndef DORA_IO_SCHEMA_INFERENCE_INTERNAL_H
#define DORA_IO_SCHEMA_INFERENCE_INTERNAL_H

#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/schema_inference.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "identity/identity.h"
#include "io/text_reader.h"

#include <stdint.h>

enum text_format : uint8_t {
    TEXT_FORMAT_CSV,
    TEXT_FORMAT_JSON
};

enum text_schema_inference_phase : uint8_t {
    TEXT_SCHEMA_INFERENCE_SCAN,
    TEXT_SCHEMA_INFERENCE_HANDED_OFF,
    TEXT_SCHEMA_INFERENCE_FAILED
};

struct text_scan_result {
    bool is_ready;
    bool is_blocked;
};

struct text_schema_inference_state {
    struct dora_allocation allocation;
    struct identity_claim_state claims;
    struct dora_error error;
    uint64_t row_count;
    uint64_t sample_row_limit;
    int failure;
    enum text_format format;
    enum text_schema_inference_phase phase;
    bool is_release_pending;
};

bool dora_text_schema_inference_is_empty(const struct dora_schema_inference* inference);
// Consumes state on every return; never reads input.
int dora_text_schema_inference_publish(struct dora_schema_inference* inference,
                                       struct text_schema_inference_state* state,
                                       struct text_input* text,
                                       struct dora_byte_source* input,
                                       const struct dora_schema_inference_options* options);
#endif
