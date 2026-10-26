// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Status and diagnostics.

#ifndef DORA_VECTOR_STATUS_H
#define DORA_VECTOR_STATUS_H

//------------------------------------------------------------------------------
//
// Status
//
//------------------------------------------------------------------------------

static constexpr int DORA_OK = 0;

//------------------------------------------------------------------------------
//
// Error
//
//------------------------------------------------------------------------------

struct dora_error {
    /// NUL-terminated diagnostic.
    char message[128];
};

#endif
