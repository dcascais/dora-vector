// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/io/csv.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "io/csv/csv_internal.h"
#include "io/text.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

int dora_csv_options_copy(const struct dora_csv_options* options, struct dora_csv_options* out_options) {
    struct dora_csv_options value = options != nullptr ? *options : (struct dora_csv_options){};
    value.delimiter = value.delimiter != 0 ? value.delimiter : ',';
    value.quote = value.quote != 0 ? value.quote : '"';
    value.row_capacity = value.row_capacity != 0 ? value.row_capacity : TEXT_DEFAULT_ROW_CAPACITY;
    if (value.delimiter == value.quote || value.delimiter == '\r' || value.delimiter == '\n' || value.quote == '\r' ||
        value.quote == '\n' || value.escape == value.delimiter || value.escape == '\r' || value.escape == '\n' ||
        value.row_capacity > INT64_MAX ||
        (value.content_policy == DORA_FAILURE_POLICY_WRAP || value.content_policy > DORA_FAILURE_POLICY_TRUST) ||
        (value.null_token.byte_count != 0 && value.null_token.data == nullptr)) {
        return EINVAL;
    }
    for (size_t index = 0; index < value.null_token.byte_count; ++index) {
        const uint8_t byte = value.null_token.data[index];
        if (byte == value.delimiter || byte == value.quote || byte == '\r' || byte == '\n' ||
            (value.escape != 0 && byte == value.escape)) {
            return EINVAL;
        }
    }
    *out_options = value;
    return DORA_OK;
}
