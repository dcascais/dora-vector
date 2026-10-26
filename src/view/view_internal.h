// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector view access.

#ifndef DORA_VIEW_VIEW_INTERNAL_H
#define DORA_VIEW_VIEW_INTERNAL_H

#include "dora/vector/view.h"

#include <stdint.h>

int dora_vector_view_read_index(const struct dora_vector_view* view, uint64_t row, uint64_t* out_index);

#endif
