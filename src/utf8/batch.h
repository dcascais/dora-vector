// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 batch validation and output.

#ifndef DORA_UTF8_BATCH_H
#define DORA_UTF8_BATCH_H

#include "dora/vector/types.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <stddef.h>
#include <stdint.h>

enum dora_utf8_layout : uint8_t {
    DORA_UTF8_OFFSETS_I32,
    DORA_UTF8_OFFSETS_I64,
    DORA_UTF8_BINARY_VIEW,
    DORA_UTF8_DESCRIPTORS,
    DORA_UTF8_CONSTANT
};

struct dora_utf8_mask {
    struct dora_bytes bytes;
    uint64_t bit_offset;
};

struct dora_utf8_offset_data {
    struct dora_bytes payload;
    struct dora_bytes offsets;
};

struct dora_utf8_view_data {
    struct dora_bytes records;
    const struct dora_bytes* buffers;
    size_t buffer_count;
};

struct dora_utf8_batch {
    size_t physical_value_count;
    struct dora_bytes row_indices;
    struct dora_utf8_mask eligible_rows;

    union {
        struct dora_utf8_offset_data rows;
        struct dora_utf8_view_data views;
        const struct dora_bytes* descriptors;
        struct dora_bytes constant;
    };

    uint32_t row_count;
    uint8_t index_width_bytes;
    enum dora_utf8_proof proof;
    enum dora_utf8_layout layout;
};

enum dora_utf8_output_layout : uint8_t {
    DORA_UTF8_PACKED_I32,
    DORA_UTF8_PACKED_I64
};

struct dora_utf8_packed_output {
    struct dora_mutable_bytes bytes;
    struct dora_mutable_bytes offset_bytes;
    size_t byte_offset;
};

struct dora_utf8_batch_output {
    struct dora_utf8_packed_output packed;
    enum dora_utf8_output_layout layout;
};

// Reports logical lanes, including duplicates; checks geometry before content.
int dora_utf8_validate_batch(const struct dora_utf8_batch* batch, bool* out_all_ascii, uint32_t* out_first_bad_row);
int dora_utf8_classify_batch(const struct dora_utf8_batch* batch,
                             uint64_t* invalid,
                             size_t word_capacity,
                             uint32_t* out_invalid_count,
                             bool* out_all_ascii);
int dora_utf8_count_batch(const struct dora_utf8_batch* batch, size_t* counts, uint32_t* out_first_bad_row);
int dora_utf8_locate_batch(const struct dora_utf8_batch* batch,
                           size_t start_scalar_index,
                           size_t scalar_count,
                           struct dora_range* out_byte_ranges,
                           uint32_t* out_first_bad_row);
int dora_utf8_find_batch(const struct dora_utf8_batch* batch,
                         struct dora_utf8_input needle,
                         size_t* offsets,
                         uint64_t* found,
                         size_t word_capacity,
                         uint32_t* out_first_bad_row);
int dora_utf8_measure_batch(const struct dora_utf8_batch* batch,
                            size_t* out_lengths,
                            size_t* out_length,
                            uint32_t* out_first_bad_row);
// Requires byte_bound to cover eligible payload; writes are uncommitted.
int dora_utf8_copy_batch_sized(const struct dora_utf8_batch* batch,
                               size_t byte_bound,
                               const struct dora_utf8_batch_output* output,
                               uint32_t* out_first_bad_row,
                               bool* out_all_ascii);
int dora_utf8_measure_concat(const struct dora_utf8_batch* batches,
                             size_t input_count,
                             struct dora_utf8_mask requested,
                             size_t* out_lengths,
                             size_t* out_length,
                             uint32_t* out_first_bad_row);
int dora_utf8_concat(const struct dora_utf8_batch* batches,
                     size_t input_count,
                     struct dora_utf8_mask requested,
                     enum dora_utf8_write_policy policy,
                     const struct dora_utf8_batch_output* output,
                     uint32_t* out_first_bad_row);

#endif
