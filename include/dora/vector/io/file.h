// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// FILE-backed byte sources and sinks.

#ifndef DORA_VECTOR_IO_FILE_H
#define DORA_VECTOR_IO_FILE_H

#include "dora/vector/io/byte.h"

#include <stdio.h>

//------------------------------------------------------------------------------
//
// File Implementations
//
//------------------------------------------------------------------------------

//--------------------------------------
// Source
//--------------------------------------

struct dora_byte_source_file {
    /// Source callbacks.
    struct dora_byte_source source;
    /// Borrowed file; release does not close it.
    FILE* file;
};

int dora_byte_source_file_init(struct dora_byte_source_file* source, FILE* file);

//--------------------------------------
// Sink
//--------------------------------------

struct dora_byte_sink_file {
    /// Sink callbacks.
    struct dora_byte_sink sink;
    /// Borrowed file; release does not close it.
    FILE* file;
};

int dora_byte_sink_file_init(struct dora_byte_sink_file* sink, FILE* file);

#endif
