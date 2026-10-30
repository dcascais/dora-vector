// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Deterministic benchmark fixtures.

#ifndef DORA_BENCHMARKS_BENCH_DATA_H
#define DORA_BENCHMARKS_BENCH_DATA_H

#include "bench.h"
#include "dora/vector/handle.h"
#include "dora/vector/io/byte.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/memory.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Columns
//
//------------------------------------------------------------------------------

//--------------------------------------
// Text Columns
//--------------------------------------

static constexpr size_t BENCH_UTF8_SLOT_BYTES = 32;

int bench_utf8(struct bench* bench,
               const struct bench_args* args,
               uint64_t seed,
               enum dora_layout layout,
               struct dora_vector* out_vector);

//--------------------------------------
// Generated Columns
//--------------------------------------

static constexpr uint32_t BENCH_INTEGER_LIMIT = 10'000'000;
static constexpr uint32_t BENCH_INTEGER_MIDPOINT = (BENCH_INTEGER_LIMIT + 1) / 2;
static constexpr uint32_t BENCH_FLOAT_NUMERATOR = 8;
static constexpr uint32_t BENCH_FLOAT_DENOMINATOR = 10;
static constexpr double BENCH_FLOAT_MIDPOINT =
    (double)(BENCH_INTEGER_MIDPOINT * BENCH_FLOAT_NUMERATOR) / BENCH_FLOAT_DENOMINATOR;
static constexpr size_t BENCH_FORMATTED_VALUE_BYTES = 128;

int bench_random(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector);
int bench_unchecked_utf8(struct bench* bench,
                         const struct bench_args* args,
                         uint64_t seed,
                         struct dora_vector* out_vector);
int bench_invalid_utf8(struct bench* bench,
                       const struct bench_args* args,
                       uint64_t seed,
                       struct dora_vector* out_vector);
int bench_formatted(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector);

//------------------------------------------------------------------------------
//
// Encoded Columns
//
//------------------------------------------------------------------------------

static constexpr uint32_t BENCH_DICTIONARY_CARDINALITY = 100;

int bench_dictionary(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector);
int bench_decoded_dictionary(struct bench* bench,
                             const struct bench_args* args,
                             uint64_t seed,
                             struct dora_vector* out_vector);
int bench_constant(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* out_vector);
int bench_take_indices(struct bench* bench,
                       const struct bench_args* args,
                       uint64_t seed,
                       struct dora_vector* out_vector);

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

struct bench_range {
    _BitInt(128) minimum;
    _BitInt(128) maximum;
};

int bench_integer_range(enum dora_type type, struct bench_range* out_range);

struct bench_outlier {
    uint64_t from_row;
    _BitInt(128) value;
};

int bench_with_bounds(struct bench* bench,
                      const struct bench_args* args,
                      uint64_t seed,
                      const struct bench_outlier* outlier,
                      struct dora_vector* out_vector);
int bench_exact_bounds(const struct dora_vector* vector, struct bench_range* out_bounds);

//------------------------------------------------------------------------------
//
// Batches
//
//------------------------------------------------------------------------------

struct bench_batch_source {
    struct dora_vector_source source;
    const struct dora_vector* batches;
    size_t count;
    size_t next;
};

void bench_batch_source_init(struct bench_batch_source* source, const struct dora_vector* batches, size_t count);
int bench_batches(struct bench* bench, size_t count, struct dora_vector* out_batches);
int bench_collect(struct dora_reader* reader, size_t capacity, struct dora_vector* out_batches, size_t* out_count);

//------------------------------------------------------------------------------
//
// Documents
//
//------------------------------------------------------------------------------

static constexpr struct dora_csv_options BENCH_CSV_OPTIONS = {.has_header = true, .row_capacity = BENCH_BATCH_ROWS};

size_t bench_line_end(const char* text, size_t byte_count, uint64_t lines);

enum bench_text_columns : uint8_t {
    BENCH_TEXT_COLUMNS_MIXED,
    BENCH_TEXT_COLUMNS_NUMERIC
};

struct bench_text_args {
    uint64_t rows;
    enum bench_text_format format;
    enum bench_text_columns columns;
};

int bench_text(struct bench* bench,
               const struct bench_text_args* args,
               struct dora_allocation* out_text,
               size_t* out_byte_count);
int bench_text_schema(struct bench* bench, enum bench_text_columns columns, struct dora_schema* out_schema);
void bench_release_byte_source(struct dora_byte_source* source);

struct bench_text_row {
    uint32_t id;
    double value;
    char text[BENCH_UTF8_SLOT_BYTES + 1];
    bool flag;
    int64_t ts_microseconds;
};

void bench_text_rows(uint64_t count, struct bench_text_row* out_rows);

//------------------------------------------------------------------------------
//
// Views
//
//------------------------------------------------------------------------------

const void* bench_view_values(const struct dora_vector_view* view, size_t* out_width);

//------------------------------------------------------------------------------
//
// Accounting
//
//------------------------------------------------------------------------------

int bench_input_bytes(const struct dora_vector* vector, uint64_t* out_bytes);

//------------------------------------------------------------------------------
//
// Oracles
//
//------------------------------------------------------------------------------

int bench_read_number(const struct dora_vector* source, uint64_t row, double* out_value);
int bench_read_integer(const struct dora_vector* vector, uint64_t row, struct dora_i128* out_value, bool* out_is_null);
_BitInt(128) bench_i128_value(struct dora_i128 value);
int bench_temporal_text(const struct dora_logical_type* type,
                        int64_t ticks,
                        char out_text[BENCH_FORMATTED_VALUE_BYTES],
                        size_t* out_byte_count);
bool bench_bytes_equal(struct dora_bytes left, struct dora_bytes right);
int bench_vectors_equal(const struct dora_vector* expected, const struct dora_vector* actual);
int bench_batches_equal(const struct dora_vector* expected,
                        size_t expected_count,
                        const struct dora_vector* actual,
                        size_t actual_count);

#endif
