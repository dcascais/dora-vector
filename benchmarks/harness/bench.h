// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Benchmark timing, statistics and reporting.

#ifndef DORA_BENCHMARKS_BENCH_H
#define DORA_BENCHMARKS_BENCH_H

#include "dora/vector/handle.h"
#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

struct bench_statistics {
    double min;
    double mean;
    double max;
    double variation;
};

int bench_describe(const double* values, size_t count, struct bench_statistics* out_statistics);
uint64_t bench_calibrate(uint64_t iterations, double elapsed, double min_seconds);

//------------------------------------------------------------------------------
//
// Formatting
//
//------------------------------------------------------------------------------

int bench_format_status(int length, size_t capacity);
int bench_format_time(double nanoseconds, size_t capacity, char* out_text);
int bench_format_rate(double rate, bool binary, size_t capacity, char* out_text);
int bench_escape(const char* source, size_t capacity, char* out_text);

//------------------------------------------------------------------------------
//
// Platform
//
//------------------------------------------------------------------------------

int bench_read_clock(uint64_t* out_nanoseconds);

//------------------------------------------------------------------------------
//
// Measurement
//
//------------------------------------------------------------------------------

enum bench_mode : uint8_t {
    BENCH_MODE_CALIBRATED,
    BENCH_MODE_EXACT,
    BENCH_MODE_SMOKE,
    BENCH_MODE_LIST
};

struct bench {
    struct dora_pool pool;
    size_t pool_granule_bytes;
    double min_seconds;
    size_t repetitions;
    uint64_t iterations;
    uint64_t epoch_iterations;
    double* wall_nanoseconds;
    uint64_t* faults;
    uint64_t* context_switches;
    uint64_t items_per_iteration;
    uint64_t bytes_per_iteration;
    int status;
    const char* error_message;
    enum bench_mode mode;
    bool measured;
};

//--------------------------------------
// Status
//--------------------------------------

void bench_check(struct bench* bench, int status);

//--------------------------------------
// Runs
//--------------------------------------

struct bench_operation {
    void (*run)(void* context, size_t count, void* out_slots, int* out_statuses);
    void (*release)(void* slots, size_t count);
    size_t slot_bytes;
};

void bench_measure(struct bench* bench, const struct bench_operation* operation, void* context);
void bench_release_vectors(void* slots, size_t count);

#define BENCH_DEFINE_OPERATION(name, slot_type, compute_function, release_function)                                    \
    static_assert(alignof(slot_type) <= alignof(max_align_t));                                                         \
    static void bench_detail_run_##name(void* bench_detail_context,                                                    \
                                        size_t bench_detail_count,                                                     \
                                        void* bench_detail_slots,                                                      \
                                        int* bench_detail_statuses) {                                                  \
        slot_type* bench_detail_typed_slots = bench_detail_slots;                                                      \
        for (size_t bench_detail_slot = 0; bench_detail_slot < bench_detail_count; ++bench_detail_slot) {              \
            bench_detail_statuses[bench_detail_slot] =                                                                 \
                compute_function(bench_detail_context, &bench_detail_typed_slots[bench_detail_slot]);                  \
        }                                                                                                              \
    }                                                                                                                  \
    static const struct bench_operation name =                                                                         \
        {.run = bench_detail_run_##name, .release = release_function, .slot_bytes = sizeof(slot_type)};

#define BENCH_DEFINE_STATUS_OPERATION(name, compute_function)                                                          \
    static void bench_detail_run_##name(void* bench_detail_context,                                                    \
                                        size_t bench_detail_count,                                                     \
                                        void*,                                                                         \
                                        int* bench_detail_statuses) {                                                  \
        for (size_t bench_detail_slot = 0; bench_detail_slot < bench_detail_count; ++bench_detail_slot) {              \
            bench_detail_statuses[bench_detail_slot] = compute_function(bench_detail_context);                         \
        }                                                                                                              \
    }                                                                                                                  \
    static const struct bench_operation name = {.run = bench_detail_run_##name};

//--------------------------------------
// Counters
//--------------------------------------

void bench_set_items(struct bench* bench, uint64_t items_per_iteration);
void bench_set_bytes(struct bench* bench, uint64_t bytes_per_iteration);

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

static constexpr uint64_t BENCH_ROWS = 65'536;
static constexpr uint64_t BENCH_BATCH_ROWS = 8192;
static constexpr size_t BENCH_BATCH_COUNT = 8;

enum bench_alphabet : uint8_t {
    BENCH_ALPHABET_ASCII,
    BENCH_ALPHABET_UNICODE
};

enum bench_text_format : uint8_t {
    BENCH_TEXT_FORMAT_CSV,
    BENCH_TEXT_FORMAT_NDJSON
};

struct bench_args {
    uint64_t rows;
    struct dora_logical_type type;
    double null_fraction;
    double selectivity;
    uint8_t variant;
    enum bench_alphabet alphabet;
    enum bench_text_format format;
    struct dora_logical_type target;
    size_t byte_count;
};

struct bench_case {
    const char* name;
    void (*function)(struct bench* bench, const struct bench_args* args);
    struct bench_args args;
};

//------------------------------------------------------------------------------
//
// Execution
//
//------------------------------------------------------------------------------

static constexpr size_t BENCH_POOL_GRANULE_BYTES = 64;

int bench_main(int argc, char** argv, const struct bench_case* cases, size_t case_count);

//------------------------------------------------------------------------------
//
// Compile-time checks
//
//------------------------------------------------------------------------------

static_assert(BENCH_ROWS == BENCH_BATCH_COUNT * BENCH_BATCH_ROWS);

#endif
