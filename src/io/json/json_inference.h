// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// JSON schema inference.

#ifndef DORA_IO_JSON_JSON_INFERENCE_H
#define DORA_IO_JSON_JSON_INFERENCE_H

#include "dora/vector/handle.h"
#include "io/schema_inference_internal.h"

int dora_json_schema_inference_scan(struct text_schema_inference_state* inference_state,
                                    struct text_scan_result* out_result);
int dora_json_schema_inference_handoff(struct text_schema_inference_state* inference_state,
                                       struct dora_schema* out_schema,
                                       struct dora_reader* out_reader);
void dora_json_schema_inference_cleanup(struct text_schema_inference_state* inference_state);

#endif
