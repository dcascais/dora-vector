// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/io/json.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "io/json/json_internal.h"
#include "io/text.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

int dora_json_options_copy(const struct dora_json_options* options, struct dora_json_options* out_options) {
    struct dora_json_options copy = options != nullptr ? *options : (struct dora_json_options){};
    if (copy.format > DORA_JSON_FORMAT_ARRAY ||
        (copy.content_policy == DORA_FAILURE_POLICY_WRAP || copy.content_policy > DORA_FAILURE_POLICY_TRUST) ||
        copy.row_capacity > INT64_MAX) {
        return EINVAL;
    }
    copy.row_capacity = copy.row_capacity != 0 ? copy.row_capacity : TEXT_DEFAULT_ROW_CAPACITY;
    copy.max_record_bytes = copy.max_record_bytes != 0 ? copy.max_record_bytes : (size_t)16 * 1024 * 1024;
    *out_options = copy;
    return DORA_OK;
}
