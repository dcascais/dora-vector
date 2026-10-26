// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Numeric failure flags.

#ifndef DORA_NUMERIC_FAILURE_H
#define DORA_NUMERIC_FAILURE_H

#include <stdint.h>

enum numeric_failure : uint8_t {
    NUMERIC_FAILURE_OVERFLOW = 1,
    NUMERIC_FAILURE_DIVIDE_BY_ZERO = 2,
    NUMERIC_FAILURE_INVALID_VALUE = 4,
    NUMERIC_FAILURE_RANGE = 8
};

#endif
