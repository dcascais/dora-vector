// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// JSON option admission.

#ifndef DORA_IO_JSON_JSON_INTERNAL_H
#define DORA_IO_JSON_JSON_INTERNAL_H

#include "dora/vector/io/json.h"

int dora_json_options_copy(const struct dora_json_options* options, struct dora_json_options* out_options);

#endif
