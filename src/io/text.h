// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Text format admission.

#ifndef DORA_IO_TEXT_H
#define DORA_IO_TEXT_H

#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "status/status_internal.h"

#include <stdint.h>
#include <string.h>

static constexpr uint64_t TEXT_DEFAULT_ROW_CAPACITY = 2048;

static inline void text_error_set(struct dora_error* error, const char* message) {
    if (error != nullptr) {
        *error = (struct dora_error){};
        if (message != nullptr) {
            error_message_copy_bounded(error, message);
        }
    }
}

static inline bool text_field_is_supported(const struct dora_schema_field* field, bool allow_json_extension) {
    return !(field->type.extension_name.byte_count != 0 || field->type.id >= DORA_TYPE_LIST ||
             field->type.id == DORA_TYPE_BINARY || field->type.id == DORA_TYPE_FIXED_BINARY) ||
           (allow_json_extension && field->type.id == DORA_TYPE_UTF8 && field->type.extension_name.byte_count == 10 &&
            memcmp(field->type.extension_name.data, "arrow.json", 10) == 0);
}

#endif
