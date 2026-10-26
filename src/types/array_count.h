// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Array-only element counts.

#ifndef DORA_TYPES_ARRAY_COUNT_H
#define DORA_TYPES_ARRAY_COUNT_H

static constexpr int ARRAY_COUNT_REQUIRES_AN_ARRAY = -1;

// Rejects pointers and adjusted array parameters.
#define ARRAY_COUNT(A)                                                                                                 \
    (0 * sizeof(char[_Generic((typeof_unqual(A)*)nullptr,                                                              \
                         typeof((A) + 0)*: ARRAY_COUNT_REQUIRES_AN_ARRAY,                                              \
                         default: 1)]) +                                                                               \
     sizeof(A) / sizeof((A)[0]))

#endif
