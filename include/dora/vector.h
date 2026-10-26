// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Vector API.

#ifndef DORA_VECTOR_H
#define DORA_VECTOR_H

// IWYU pragma: begin_exports

//------------------------------------------------------------------------------
//
// API
//
//------------------------------------------------------------------------------

#include "dora/vector/bitmap.h"
#include "dora/vector/builder.h"
#include "dora/vector/compute.h"
#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/statistics.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"

//------------------------------------------------------------------------------
//
// Extensions
//
//------------------------------------------------------------------------------

#include "dora/vector/extension.h"

//------------------------------------------------------------------------------
//
// IO
//
//------------------------------------------------------------------------------

#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/io/file.h"
#include "dora/vector/io/json.h"
#include "dora/vector/io/schema_inference.h"

// IWYU pragma: end_exports

#endif
