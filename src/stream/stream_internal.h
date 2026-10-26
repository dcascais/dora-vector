// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Stream pool resolution.

#ifndef DORA_STREAM_STREAM_INTERNAL_H
#define DORA_STREAM_STREAM_INTERNAL_H

#include "dora/vector/handle.h"

int dora_reader_get_pool(const struct dora_reader* reader, struct dora_pool* out_pool);

#endif
