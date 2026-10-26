// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/io/file.h"

#include "dora/vector/io/byte.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

//------------------------------------------------------------------------------
//
// File Callbacks
//
//------------------------------------------------------------------------------

static int byte_file_read(struct dora_byte_source* source,
                          uint8_t* out_bytes,
                          size_t capacity_bytes,
                          size_t* out_byte_count,
                          bool* out_is_end) {
    if (source == nullptr || out_byte_count == nullptr || out_is_end == nullptr ||
        (capacity_bytes != 0 && out_bytes == nullptr)) {
        return EINVAL;
    }
    const struct dora_byte_source_file* const state = (const struct dora_byte_source_file*)source;
    if (state->file == nullptr) {
        return EINVAL;
    }
    const size_t count = capacity_bytes == 0 ? 0 : fread(out_bytes, 1, capacity_bytes, state->file);
    if (ferror(state->file) != 0) {
        return EIO;
    }
    *out_byte_count = count;
    *out_is_end = feof(state->file) != 0;
    return DORA_OK;
}

static void byte_file_source_release(struct dora_byte_source* source) {
    if (source != nullptr) {
        *(struct dora_byte_source_file*)source = (struct dora_byte_source_file){};
    }
}

static int byte_file_write(struct dora_byte_sink* sink, struct dora_bytes bytes) {
    if (sink == nullptr || (bytes.byte_count != 0 && bytes.data == nullptr)) {
        return EINVAL;
    }
    const struct dora_byte_sink_file* const state = (const struct dora_byte_sink_file*)sink;
    if (state->file == nullptr) {
        return EINVAL;
    }
    if (bytes.byte_count == 0) {
        return DORA_OK;
    }
    return fwrite(bytes.data, 1, bytes.byte_count, state->file) == bytes.byte_count ? DORA_OK : EIO;
}

static int byte_file_flush(struct dora_byte_sink* sink) {
    if (sink == nullptr || ((const struct dora_byte_sink_file*)sink)->file == nullptr) {
        return EINVAL;
    }
    return fflush(((const struct dora_byte_sink_file*)sink)->file) == EOF ? EIO : DORA_OK;
}

static void byte_file_sink_release(struct dora_byte_sink* sink) {
    if (sink != nullptr) {
        *(struct dora_byte_sink_file*)sink = (struct dora_byte_sink_file){};
    }
}

//------------------------------------------------------------------------------
//
// File Transport
//
//------------------------------------------------------------------------------

int dora_byte_source_file_init(struct dora_byte_source_file* source, FILE* file) {
    if (file == nullptr || source == nullptr || source->source.read != nullptr || source->source.read_at != nullptr ||
        source->source.get_byte_count != nullptr || source->source.release != nullptr || source->file != nullptr) {
        return EINVAL;
    }
    *source = (struct dora_byte_source_file){
        .source = {.read = byte_file_read, .release = byte_file_source_release},
        .file = file
    };
    return DORA_OK;
}

int dora_byte_sink_file_init(struct dora_byte_sink_file* sink, FILE* file) {
    if (file == nullptr || sink == nullptr || sink->sink.write != nullptr || sink->sink.flush != nullptr ||
        sink->sink.release != nullptr || sink->file != nullptr) {
        return EINVAL;
    }
    *sink = (struct dora_byte_sink_file){
        .sink = {.write = byte_file_write, .flush = byte_file_flush, .release = byte_file_sink_release},
        .file = file
    };
    return DORA_OK;
}
