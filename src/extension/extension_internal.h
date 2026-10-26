// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Pool extension registrations.

#ifndef DORA_EXTENSION_EXTENSION_INTERNAL_H
#define DORA_EXTENSION_EXTENSION_INTERNAL_H

#include "dora/vector/extension.h"
#include "dora/vector/handle.h"
#include "dora/vector/types.h"

//------------------------------------------------------------------------------
//
// Extension Registry
//
//------------------------------------------------------------------------------

struct extension_registration {
    const struct dora_extension_definition* definition;
    struct extension_registration* next;
};

//--------------------------------------
// Definition Lookup
//--------------------------------------

int dora_extension_find(struct dora_pool pool,
                        struct dora_bytes name,
                        const struct dora_extension_definition** out_definition);

//--------------------------------------
// Registration Lifecycle
//--------------------------------------

void dora_extension_registrations_close(struct dora_pool pool);

#endif
