// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Bounded diagnostic messages.

#ifndef DORA_STATUS_STATUS_INTERNAL_H
#define DORA_STATUS_STATUS_INTERNAL_H

#include "dora/vector/status.h"

#include <stddef.h>
#include <string.h>

// Preserves the final byte; shorter messages are padded with NUL.
static inline void error_message_copy_bounded(struct dora_error* error, const char* message) {
    constexpr size_t bound = sizeof(error->message) - 1;
    const char* const terminator = memchr(message, '\0', bound);
    const size_t length = terminator == nullptr ? bound : (size_t)(terminator - message);
    memcpy(error->message, message, length);
    memset(error->message + length, '\0', bound - length);
}

#endif
