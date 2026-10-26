// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// CSV option admission.

#ifndef DORA_IO_CSV_CSV_INTERNAL_H
#define DORA_IO_CSV_CSV_INTERNAL_H

#include "dora/vector/io/csv.h"

int dora_csv_options_copy(const struct dora_csv_options* options, struct dora_csv_options* out_options);

#endif
