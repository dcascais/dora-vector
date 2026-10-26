// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 validity proofs.

#ifndef DORA_UTF8_INPUT_H
#define DORA_UTF8_INPUT_H

#include "dora/vector/types.h"

#include <stdint.h>

enum dora_utf8_proof : uint8_t {
    DORA_UTF8_CHECKED,
    DORA_UTF8_PROMISED,
    DORA_UTF8_VALIDATED,
    DORA_UTF8_VALIDATED_ASCII
};

struct dora_utf8_input {
    struct dora_bytes bytes;
    enum dora_utf8_proof proof;
};

#endif
