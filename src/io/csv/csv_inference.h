// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// CSV schema inference.

#ifndef DORA_IO_CSV_CSV_INFERENCE_H
#define DORA_IO_CSV_CSV_INFERENCE_H

#include "dora/vector/handle.h"
#include "io/schema_inference_internal.h"

int dora_csv_schema_inference_scan(struct text_schema_inference_state* inference_state,
                                   struct text_scan_result* out_result);
int dora_csv_schema_inference_handoff(struct text_schema_inference_state* inference_state,
                                      struct dora_schema* out_schema,
                                      struct dora_reader* out_reader);
void dora_csv_schema_inference_cleanup(struct text_schema_inference_state* inference_state);

#endif
