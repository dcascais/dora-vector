// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"
#include "bench_data.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "types/array_count.h"

#include <errno.h>
#include <math.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

static int statistics_checks() {
    const double odd[] = {5, 1, 3};
    const double one[] = {7};
    const double zeros[] = {0, 0};
    const double invalid[] = {1, NAN};
    const double negative[] = {1, -1};
    struct bench_statistics statistics = {};
    int status = bench_describe(odd, ARRAY_COUNT(odd), &statistics);
    if (status != 0 || statistics.min != 1 || statistics.mean != 3 || statistics.max != 5 ||
        fabs(statistics.variation - 2.0 / 3) > 1e-14) {
        return status != 0 ? status : EDOM;
    }
    status = bench_describe(one, ARRAY_COUNT(one), &statistics);
    if (status != 0 || statistics.min != 7 || statistics.max != 7 || statistics.variation != 0) {
        return status != 0 ? status : EDOM;
    }
    status = bench_describe(zeros, ARRAY_COUNT(zeros), &statistics);
    if (status != 0 || statistics.mean != 0 || statistics.variation != 0) {
        return status != 0 ? status : EDOM;
    }
    if (bench_describe(invalid, ARRAY_COUNT(invalid), &statistics) != EDOM ||
        bench_describe(negative, ARRAY_COUNT(negative), &statistics) != EDOM ||
        bench_describe(one, 0, &statistics) != EDOM) {
        return EDOM;
    }
    if (bench_calibrate(1, 0, 0.1) != 10 || bench_calibrate(10, 0.009, 0.1) != 100 ||
        bench_calibrate(10, 0.02, 0.1) != 70 || bench_calibrate(10, 0.023, 0.1) != 61 ||
        bench_calibrate(10, 0.1, 0.1) != 10 || bench_calibrate(999'999'999, 0.099, 0.1) != 1'000'000'000 ||
        bench_calibrate(1'000'000'000, 0, 0.1) != 1'000'000'000) {
        return EDOM;
    }
    return 0;
}

//------------------------------------------------------------------------------
//
// Formatting
//
//------------------------------------------------------------------------------

static int formatting_checks() {
    char text[128];
    const double times[] = {12, 1250, 1'250'000, 1'250'000'000};
    const char* expected[] = {"12.000 ns", "1.250 us", "1.250 ms", "1.250 s"};
    for (size_t index = 0; index < ARRAY_COUNT(times); ++index) {
        const int status = bench_format_time(times[index], sizeof(text), text);
        if (status != 0 || strcmp(text, expected[index]) != 0) {
            return status != 0 ? status : EDOM;
        }
    }
    if (bench_format_rate(3.59e9, false, sizeof(text), text) != 0 || strcmp(text, "3.59 G/s") != 0 ||
        bench_format_rate(54.8 * 1'073'741'824, true, sizeof(text), text) != 0 || strcmp(text, "54.8 GiB/s") != 0 ||
        bench_format_rate(512, true, sizeof(text), text) != 0 || strcmp(text, "512 B/s") != 0 ||
        bench_format_rate(1010 * 1'048'576.0, true, sizeof(text), text) != 0 || strcmp(text, "0.986 GiB/s") != 0 ||
        bench_format_rate(999.7e6, false, sizeof(text), text) != 0 || strcmp(text, "1 G/s") != 0 ||
        bench_format_rate(1023, true, sizeof(text), text) != 0 || strcmp(text, "0.999 KiB/s") != 0 ||
        bench_format_time(NAN, sizeof(text), text) != EDOM || bench_format_time(1, 2, text) != ENOSPC ||
        bench_escape("a\"\\\n\t\r\001", sizeof(text), text) != 0 ||
        strcmp(text, "\"a\\\"\\\\\\u000a\\u0009\\u000d\\u0001\"") != 0 || bench_escape("a", 3, text) != ENOSPC ||
        bench_escape("", 3, text) != 0 || strcmp(text, "\"\"") != 0) {
        return EDOM;
    }
    return 0;
}

//------------------------------------------------------------------------------
//
// Generators
//
//------------------------------------------------------------------------------

static int seeded_null_count(struct bench* bench, const struct bench_args* args, uint64_t seed, uint64_t* out_count) {
    const struct bench_args mask_args =
        {.rows = args->rows, .type.id = DORA_TYPE_BOOLEAN, .selectivity = args->null_fraction};
    struct dora_vector mask = {};
    struct dora_vector_view view;
    int status = bench_random(bench, &mask_args, seed, &mask);
    if (status == 0) {
        status = dora_vector_get_view(&mask, &view);
    }
    uint64_t count = 0;
    for (uint64_t row = 0; status == 0 && row < view.length; ++row) {
        count += dora_vector_view_get_bool(&view, row) ? 1 : 0;
    }
    dora_vector_release(&mask);
    if (status == 0) {
        *out_count = count;
    }
    return status;
}

static int
inspect_generated(struct bench* bench, const struct bench_args* args, uint64_t seed, struct dora_vector* vector) {
    uint64_t rows = 0;
    uint64_t bytes = 0;
    struct dora_vector_view view;
    int status = dora_vector_validate(vector, nullptr);
    if (status == 0) {
        status = dora_vector_get_length(vector, &rows);
    }
    if (status == 0) {
        status = dora_vector_get_view(vector, &view);
    }
    if (status == 0) {
        status = bench_input_bytes(vector, &bytes);
    }
    if (status == 0 &&
        (rows != args->rows || bytes == 0 || (args->null_fraction == 0 && view.validity.data != nullptr))) {
        status = EDOM;
    }
    uint64_t null_count = 0;
    if (status == 0 && args->null_fraction > 0) {
        status = seeded_null_count(bench, args, seed, &null_count);
    }
    uint64_t actual_null_count = 0;
    if (status == 0) {
        status = dora_vector_get_null_count(vector, &actual_null_count);
    }
    return status != 0 ? status : actual_null_count == null_count ? 0 : EDOM;
}

static int check_coefficients(const struct bench_args* args, const struct dora_vector* vector) {
    if (args->type.id != DORA_TYPE_DECIMAL) {
        return 0;
    }
    _BitInt(128) limit = 1;
    for (int32_t digit = 0; digit < args->type.parameters.decimal.precision; ++digit) {
        limit *= 10;
    }
    int status = 0;
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        struct dora_i128 value = {};
        bool is_null = false;
        status = dora_vector_read_i128(vector, row, &value, &is_null);
        const _BitInt(128) coefficient = bench_i128_value(value);
        if (status == 0 && !is_null && (coefficient < 0 || coefficient >= limit)) {
            status = EDOM;
        }
    }
    return status;
}

static int check_int16_domain(const struct bench_args* args, const struct dora_vector* vector) {
    if (args->type.id != DORA_TYPE_INT16) {
        return 0;
    }
    int status = 0;
    for (uint64_t row = 0; status == 0 && row < args->rows; ++row) {
        int16_t value = 0;
        bool is_null = false;
        status = dora_vector_read_i16(vector, row, &value, &is_null);
        if (status == 0 && !is_null && value < 1) {
            status = EDOM;
        }
    }
    return status;
}

static int check_fixed_binary(const struct bench_args* args, const struct dora_vector* vector) {
    if (args->type.id != DORA_TYPE_FIXED_BINARY) {
        return 0;
    }
    struct dora_vector_view view;
    const int status = dora_vector_get_view(vector, &view);
    if (status != 0 || view.layout != DORA_LAYOUT_FIXED || view.byte_width != args->type.parameters.byte_width) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t left = 0; left < view.length; ++left) {
        if (dora_vector_view_is_null(&view, left)) {
            continue;
        }
        const struct dora_bytes value = dora_vector_view_get_bytes(&view, left);
        if (value.byte_count != view.byte_width) {
            return EDOM;
        }
        for (uint64_t right = left + 1; right < view.length; ++right) {
            if (bench_bytes_equal(value, dora_vector_view_get_bytes(&view, right))) {
                return EDOM;
            }
        }
    }
    return 0;
}

static constexpr uint8_t CAPITAL_E_ACUTE[] = {0xc3, 0x89};

static int check_unicode(const struct dora_vector* vector) {
    struct dora_vector_view view;
    const int status = dora_vector_get_view(vector, &view);
    for (uint64_t row = 0; status == 0 && row < view.length; ++row) {
        const struct dora_bytes text = dora_vector_view_get_bytes(&view, row);
        for (size_t offset = 0; offset + sizeof(CAPITAL_E_ACUTE) <= text.byte_count; ++offset) {
            if (memcmp(text.data + offset, CAPITAL_E_ACUTE, sizeof(CAPITAL_E_ACUTE)) == 0) {
                return 0;
            }
        }
    }
    return status != 0 ? status : EDOM;
}

static int check_dictionary(const struct dora_vector* vector) {
    struct dora_vector domain = {};
    struct dora_vector_view view;
    int status = dora_vector_get_child(vector, 0, &domain);
    if (status == 0) {
        status = dora_vector_get_view(&domain, &view);
    }
    if (status == 0 && view.length != BENCH_DICTIONARY_CARDINALITY) {
        status = EDOM;
    }
    for (uint64_t left = 0; status == 0 && left < view.length; ++left) {
        for (uint64_t right = left + 1; status == 0 && right < view.length; ++right) {
            if (bench_bytes_equal(dora_vector_view_get_bytes(&view, left), dora_vector_view_get_bytes(&view, right))) {
                status = EDOM;
            }
        }
    }
    dora_vector_release(&domain);
    return status;
}

static int check_take_indices(const struct bench_args* args, const struct dora_vector* vector) {
    struct dora_vector_view view;
    const int status = dora_vector_get_view(vector, &view);
    const uint32_t* indices =
        status == 0 && view.type == DORA_TYPE_UINT32 ? dora_vector_view_get_values_u32(&view) : nullptr;
    if (indices == nullptr) {
        return status != 0 ? status : EDOM;
    }
    for (uint64_t row = 0; row < view.length; ++row) {
        if (indices[row] >= args->rows) {
            return EDOM;
        }
    }
    return 0;
}

static int check_with_bounds_rows(const struct dora_vector* source,
                                  const struct dora_vector* with_bounds,
                                  const struct bench_outlier* outlier) {
    uint64_t rows = 0;
    int status = dora_vector_get_length(source, &rows);
    bool is_placed = outlier == nullptr;
    struct bench_range expected_bounds = {};
    uint64_t valid_count = 0;
    for (uint64_t row = 0; status == 0 && row < rows; ++row) {
        struct dora_i128 expected_value = {};
        struct dora_i128 actual_value = {};
        bool expected_null = false;
        bool actual_null = false;
        status = bench_read_integer(source, row, &expected_value, &expected_null);
        if (status == 0) {
            status = bench_read_integer(with_bounds, row, &actual_value, &actual_null);
        }
        _BitInt(128) expected = bench_i128_value(expected_value);
        if (status == 0 && !expected_null && !is_placed && row >= outlier->from_row) {
            expected = outlier->value;
            is_placed = true;
        }
        if (status == 0 &&
            (expected_null != actual_null || (!expected_null && bench_i128_value(actual_value) != expected))) {
            status = EDOM;
        }
        if (status == 0 && !expected_null) {
            const bool is_first = valid_count == 0;
            expected_bounds.minimum =
                is_first || expected < expected_bounds.minimum ? expected : expected_bounds.minimum;
            expected_bounds.maximum =
                is_first || expected > expected_bounds.maximum ? expected : expected_bounds.maximum;
            ++valid_count;
        }
    }
    struct bench_range bounds = {};
    if (status == 0) {
        status = bench_exact_bounds(with_bounds, &bounds);
    }
    if (status == 0 &&
        (!is_placed || bounds.minimum != expected_bounds.minimum || bounds.maximum != expected_bounds.maximum)) {
        status = EDOM;
    }
    return status;
}

static constexpr uint64_t CHECK_ROWS = 129;

static int with_bounds_checks(struct bench* bench) {
    const enum dora_type types[] = {DORA_TYPE_INT32, DORA_TYPE_INT64, DORA_TYPE_UINT32, DORA_TYPE_UINT64};
    const struct bench_outlier outlier = {.from_row = 1, .value = INT32_C(1) << 30};
    const struct bench_outlier* placements[] = {nullptr, &outlier};
    int status = 0;
    for (size_t index = 0; status == 0 && index < ARRAY_COUNT(types); ++index) {
        for (int nulls = 0; status == 0 && nulls < 2; ++nulls) {
            for (size_t placement = 0; status == 0 && placement < ARRAY_COUNT(placements); ++placement) {
                const struct bench_args args =
                    {.rows = CHECK_ROWS, .type.id = types[index], .null_fraction = nulls * 0.5};
                struct dora_vector source = {};
                struct dora_vector with_bounds = {};
                struct bench_range bounds = {};
                uint64_t source_bytes = 0;
                uint64_t with_bounds_bytes = 0;
                status = bench_random(bench, &args, 1, &source);
                if (status == 0) {
                    status = bench_with_bounds(bench, &args, 1, placements[placement], &with_bounds);
                }
                if (status == 0) {
                    status = inspect_generated(bench, &args, 1, &with_bounds);
                }
                if (status == 0) {
                    status = bench_input_bytes(&source, &source_bytes);
                }
                if (status == 0) {
                    status = bench_input_bytes(&with_bounds, &with_bounds_bytes);
                }
                if (status == 0 && (with_bounds_bytes != source_bytes || bench_exact_bounds(&source, &bounds) == 0)) {
                    status = EDOM;
                }
                if (status == 0) {
                    status = check_with_bounds_rows(&source, &with_bounds, placements[placement]);
                }
                dora_vector_release(&with_bounds);
                dora_vector_release(&source);
            }
        }
    }
    return status;
}

static int generator_checks() {
    struct bench bench = {};
    int status = dora_pool_init(&bench.pool, SIZE_MAX, BENCH_POOL_GRANULE_BYTES);
    if (status != 0) {
        return status;
    }

    const struct dora_logical_type types[] = {
        {.id = DORA_TYPE_INT16},
        {.id = DORA_TYPE_INT32},
        {.id = DORA_TYPE_INT64},
        {.id = DORA_TYPE_UINT32},
        {.id = DORA_TYPE_UINT64},
        {.id = DORA_TYPE_FLOAT32},
        {.id = DORA_TYPE_FLOAT64},
        {.id = DORA_TYPE_BOOLEAN},
        {.id = DORA_TYPE_UTF8},
        {.id = DORA_TYPE_FIXED_BINARY, .parameters.byte_width = 16},
        {.id = DORA_TYPE_DATE32},
        {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr}},
        {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_SECOND, .timezone = "UTC"}},
        {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_NANOSECOND, .timezone = "UTC"}},
        {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 20, .scale = 2, .bit_width = 128}},
        {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 19, .scale = 4, .bit_width = 128}},
        {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 19, .scale = 10, .bit_width = 128}},
        {.id = DORA_TYPE_DECIMAL, .parameters.decimal = {.precision = 38, .scale = 8, .bit_width = 128}},
        {.id = DORA_TYPE_LIST},
        {.id = DORA_TYPE_STRUCT}
    };

    for (size_t index = 0; status == 0 && index < ARRAY_COUNT(types); ++index) {
        for (int nulls = 0; status == 0 && nulls < 2; ++nulls) {
            struct bench_args args =
                {.rows = CHECK_ROWS, .type = types[index], .null_fraction = nulls * 0.5, .selectivity = 0.5};
            struct dora_vector vector = {};
            struct dora_vector duplicate = {};
            status = bench_random(&bench, &args, 1, &vector);
            if (status == 0) {
                status = inspect_generated(&bench, &args, 1, &vector);
                struct dora_logical_type actual = {};
                if (status == 0) {
                    status = dora_vector_get_type(&vector, &actual);
                }
                if (status == 0 && actual.id == DORA_TYPE_TIMESTAMP) {
                    const char* expected_zone = args.type.parameters.timestamp.timezone;
                    const char* actual_zone = actual.parameters.timestamp.timezone;
                    if (actual.parameters.timestamp.unit != args.type.parameters.timestamp.unit ||
                        ((expected_zone == nullptr) != (actual_zone == nullptr)) ||
                        (expected_zone != nullptr && strcmp(expected_zone, actual_zone) != 0)) {
                        status = EDOM;
                    }
                }
                if (status == 0 && actual.id == DORA_TYPE_DECIMAL &&
                    (actual.parameters.decimal.precision != args.type.parameters.decimal.precision ||
                     actual.parameters.decimal.scale != args.type.parameters.decimal.scale ||
                     actual.parameters.decimal.bit_width != args.type.parameters.decimal.bit_width)) {
                    status = EDOM;
                }
            }
            if (status == 0) {
                status = check_coefficients(&args, &vector);
            }
            if (status == 0) {
                status = check_int16_domain(&args, &vector);
            }
            if (status == 0) {
                status = check_fixed_binary(&args, &vector);
            }
            if (status == 0) {
                status = bench_random(&bench, &args, 1, &duplicate);
            }
            bool equal = false;
            if (status == 0) {
                status = dora_vector_is_equal(&vector, &duplicate, &equal);
                if (status == 0 && !equal) {
                    status = EDOM;
                }
            }
            dora_vector_release(&duplicate);
            dora_vector_release(&vector);
        }
    }
    const struct bench_args args = {.rows = CHECK_ROWS, .type.id = DORA_TYPE_INT64};
    const struct bench_args unicode_args =
        {.rows = CHECK_ROWS, .type.id = DORA_TYPE_INT64, .alphabet = BENCH_ALPHABET_UNICODE};
    struct dora_vector generated[5] = {};
    if (status == 0) {
        status = bench_utf8(&bench, &unicode_args, 1, DORA_LAYOUT_BINARY, &generated[0]);
    }
    if (status == 0) {
        status = bench_utf8(&bench, &args, 1, DORA_LAYOUT_BINARY_VIEW, &generated[1]);
    }
    if (status == 0) {
        status = bench_dictionary(&bench, &args, 1, &generated[2]);
    }
    if (status == 0) {
        status = bench_constant(&bench, &args, 1, &generated[3]);
    }
    if (status == 0) {
        status = bench_take_indices(&bench, &args, 1, &generated[4]);
    }
    if (status == 0) {
        status = check_unicode(&generated[0]);
    }
    if (status == 0) {
        status = check_dictionary(&generated[2]);
    }
    if (status == 0) {
        status = check_take_indices(&args, &generated[4]);
    }
    for (size_t index = 0; index < ARRAY_COUNT(generated); ++index) {
        if (status == 0) {
            status = inspect_generated(&bench, &args, 1, &generated[index]);
        }
        dora_vector_release(&generated[index]);
    }

    const struct dora_logical_type text_types[] = {
        {.id = DORA_TYPE_INT64},
        {.id = DORA_TYPE_FLOAT64},
        {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_MICROSECOND, .timezone = nullptr}}
    };

    for (size_t index = 0; status == 0 && index < ARRAY_COUNT(text_types); ++index) {
        struct bench_args text_args = {.rows = CHECK_ROWS, .type = text_types[index], .null_fraction = 0.5};
        struct dora_vector vector = {};
        status = bench_formatted(&bench, &text_args, 1, &vector);
        if (status == 0) {
            status = inspect_generated(&bench, &text_args, 1, &vector);
        }
        dora_vector_release(&vector);
    }
    const struct bench_text_args documents[] = {
        {.rows = CHECK_ROWS, .format = BENCH_TEXT_FORMAT_CSV, .columns = BENCH_TEXT_COLUMNS_MIXED},
        {.rows = CHECK_ROWS, .format = BENCH_TEXT_FORMAT_NDJSON, .columns = BENCH_TEXT_COLUMNS_MIXED},
        {.rows = CHECK_ROWS, .format = BENCH_TEXT_FORMAT_CSV, .columns = BENCH_TEXT_COLUMNS_NUMERIC},
        {.rows = CHECK_ROWS, .format = BENCH_TEXT_FORMAT_NDJSON, .columns = BENCH_TEXT_COLUMNS_NUMERIC}
    };
    for (size_t index = 0; status == 0 && index < ARRAY_COUNT(documents); ++index) {
        struct dora_allocation text = {};
        size_t bytes = 0;
        status = bench_text(&bench, &documents[index], &text, &bytes);
        if (status == 0 && (bytes == 0 || bytes > text.capacity_bytes)) {
            status = EDOM;
        }
        dora_allocation_release(&text);
    }
    if (status == 0) {
        status = with_bounds_checks(&bench);
    }
    const int release_status = dora_pool_release(&bench.pool);
    return status == 0 ? release_status : status;
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

struct lifecycle_state {
    struct dora_pool pool;
    uint64_t operations;
    size_t largest_count;
    size_t run_count;
    size_t run_capacity;
    size_t* counts;
    int status;
    bool is_reused;
};

static void record_run(struct lifecycle_state* state, size_t count) {
    if (state->run_count == state->run_capacity) {
        size_t capacity = state->run_capacity == 0 ? 64 : state->run_capacity;
        size_t bytes;
        if (ckd_mul(&capacity, capacity, 2) || ckd_mul(&bytes, capacity, sizeof(*state->counts))) {
            state->status = EOVERFLOW;
            return;
        }
        size_t* counts = realloc(state->counts, bytes);
        if (counts == nullptr) {
            state->status = ENOMEM;
            return;
        }
        state->counts = counts;
        state->run_capacity = capacity;
    }
    state->counts[state->run_count++] = count;
}

static void run_lifecycle(void* context, size_t count, void* out_slots, int* out_statuses) {
    struct lifecycle_state* state = context;
    struct dora_vector* results = out_slots;
    const struct dora_logical_type type = {.id = DORA_TYPE_UINT64};
    state->largest_count = count > state->largest_count ? count : state->largest_count;
    record_run(state, count);
    for (size_t slot = 0; slot < count; ++slot) {
        state->is_reused = state->is_reused || dora_vector_is_valid(&results[slot]);
        out_statuses[slot] =
            dora_vector_init_copy_u64(&results[slot], state->pool, &type, &state->operations, nullptr, 1);
        ++state->operations;
    }
}

static bool has_fixed_schedule(const struct lifecycle_state* state, const struct bench* bench) {
    size_t end = state->run_count;
    size_t last_start = 0;
    size_t length = 0;
    for (size_t group = 0; group < bench->repetitions; ++group) {
        uint64_t total = 0;
        size_t start = end;
        while (start > 0 && total < bench->iterations) {
            total += state->counts[--start];
        }
        if (total != bench->iterations || (group != 0 && end - start != length)) {
            return false;
        }
        if (group == 0) {
            last_start = start;
            length = end - start;
        }
        for (size_t run = 0; run < length; ++run) {
            if (state->counts[start + run] != state->counts[last_start + run] ||
                (run + 1 < length && state->counts[start + run] != bench->epoch_iterations)) {
                return false;
            }
        }
        end = start;
    }
    return true;
}

static bool has_single_run(const struct lifecycle_state* state, const struct bench* bench) {
    const bool is_untimed = bench->wall_nanoseconds == nullptr && bench->faults == nullptr &&
                            bench->context_switches == nullptr && bench->epoch_iterations == 0;
    return is_untimed && state->run_count == 1 && state->largest_count == 1 && state->operations == 1;
}

static bool
has_expected_runs(const struct lifecycle_state* state, const struct bench* bench, const struct bench_args* args) {
    if (bench->mode == BENCH_MODE_SMOKE) {
        return has_single_run(state, bench);
    }
    const uint64_t timed = bench->repetitions * bench->iterations;
    const bool has_totals =
        bench->mode != BENCH_MODE_EXACT || (timed == args->rows && state->operations == timed + bench->iterations);
    return has_totals && state->largest_count <= bench->epoch_iterations && has_fixed_schedule(state, bench);
}

static void lifecycle(struct bench* bench, const struct bench_args* args) {
    struct lifecycle_state state = {.pool = bench->pool};
    bench_measure(bench,
                  &(const struct bench_operation){
                      .run = run_lifecycle,
                      .release = bench_release_vectors,
                      .slot_bytes = sizeof(struct dora_vector)
                  },
                  &state);
    bench_check(bench, state.status);
    bench_check(bench, !state.is_reused && has_expected_runs(&state, bench, args) ? 0 : EDOM);
    free(state.counts);
    bench_set_items(bench, 1);
}

static void run_empty(void*, size_t count, void*, int* out_statuses) {
    for (size_t slot = 0; slot < count; ++slot) {
        out_statuses[slot] = 0;
    }
}

static constexpr uint64_t WORK_STORES = 64;

static volatile uint64_t work_sink;

static void run_work(void*, size_t count, void*, int* out_statuses) {
    for (size_t slot = 0; slot < count; ++slot) {
        for (uint64_t store = 0; store < WORK_STORES; ++store) {
            work_sink = store;
        }
        out_statuses[slot] = 0;
    }
}

static void run_failure(void* context, size_t count, void*, int* out_statuses) {
    bool* is_failing = context;
    for (size_t slot = 0; slot < count; ++slot) {
        out_statuses[slot] = !*is_failing ? 0 : slot == 0 ? EDOM : EIO;
    }
    *is_failing = true;
}

static void failure(struct bench* bench, const struct bench_args*) {
    bool is_failing = false;
    bench_measure(bench, &(const struct bench_operation){.run = run_failure}, &is_failing);
    bench_check(bench, EINVAL);
}

static void leak(struct bench* bench, const struct bench_args*) {
    const int64_t value = 1;
    struct dora_vector vector = {};
    bench_check(bench,
                dora_vector_init_copy_i64(&vector,
                                          bench->pool,
                                          &(const struct dora_logical_type){.id = DORA_TYPE_INT64},
                                          &value,
                                          nullptr,
                                          1));
}

enum counter_variant : uint8_t {
    VARIANT_COUNTER_NONE,
    VARIANT_COUNTER_BYTES
};

static void counters(struct bench* bench, const struct bench_args* args) {
    bench_measure(bench, &(const struct bench_operation){.run = run_work}, nullptr);
    if (args->rows != 0) {
        bench_set_items(bench, args->rows);
    }
    if (args->variant == VARIANT_COUNTER_BYTES) {
        bench_set_bytes(bench, 8);
    }
}

enum variant : uint8_t {
    VARIANT_ZERO,
    VARIANT_POSITIVE,
    VARIANT_MIXED_ZERO,
    VARIANT_NEGATIVE,
    VARIANT_NAN,
    VARIANT_INFINITE
};

static void summary_samples(struct bench* bench, const struct bench_args* args) {
    bench_measure(bench, &(const struct bench_operation){.run = run_empty}, nullptr);
    const double first_samples[] = {
        [VARIANT_ZERO] = 0,
        [VARIANT_POSITIVE] = 1,
        [VARIANT_MIXED_ZERO] = 0,
        [VARIANT_NEGATIVE] = -1,
        [VARIANT_NAN] = NAN,
        [VARIANT_INFINITE] = INFINITY
    };
    const double later_sample = args->variant == VARIANT_ZERO ? 0 : 1;
    for (size_t index = 0; index < bench->repetitions; ++index) {
        bench->wall_nanoseconds[index] = index == 0 ? first_samples[args->variant] : later_sample;
    }
    bench_set_items(bench, 1);
    bench_set_bytes(bench, 8);
}

static constexpr uint64_t CONTRACT_NANOSECONDS = 20'000;

static int wait_contract() {
    uint64_t start = 0;
    int status = bench_read_clock(&start);
    uint64_t now = start;
    while (status == 0 && now - start < CONTRACT_NANOSECONDS) {
        status = bench_read_clock(&now);
    }
    return status;
}

static void run_contract(void*, size_t count, void*, int* out_statuses) {
    for (size_t slot = 0; slot < count; ++slot) {
        out_statuses[slot] = wait_contract();
    }
}

static void release_contract(void*, size_t) {
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (wait_contract() != 0 || page_bytes <= 0) {
        abort();
    }
    void* page = mmap(nullptr, (size_t)page_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        abort();
    }
    volatile uint8_t* byte = page;
    *byte = 1;
    if (munmap(page, (size_t)page_bytes) != 0) {
        abort();
    }
}

static constexpr double CONTRACT_MAX_RATIO = 1.1;

static void timing_contract(struct bench* bench, const struct bench_args*) {
    bench_measure(bench, &(const struct bench_operation){.run = run_contract, .release = release_contract}, nullptr);
    struct bench_statistics statistics = {};
    bench_check(bench, bench_describe(bench->wall_nanoseconds, bench->repetitions, &statistics));
    uint64_t faults = 0;
    for (size_t index = 0; index < bench->repetitions; ++index) {
        faults += bench->faults[index];
    }
    const double span = (double)CONTRACT_NANOSECONDS;
    bench_check(bench,
                statistics.min >= span && statistics.min <= CONTRACT_MAX_RATIO * span &&
                        faults >= bench->repetitions * bench->iterations
                    ? 0
                    : EDOM);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--samples") == 0) {
        const struct bench_case sample_cases[] = {
            {"zero", summary_samples, {}},
            {"positive", summary_samples, {.variant = VARIANT_POSITIVE}},
            {"mixed_zero", summary_samples, {.variant = VARIANT_MIXED_ZERO}},
            {"negative", summary_samples, {.variant = VARIANT_NEGATIVE}},
            {"nan", summary_samples, {.variant = VARIANT_NAN}},
            {"infinite", summary_samples, {.variant = VARIANT_INFINITE}}
        };
        argv[1] = argv[0];
        return bench_main(argc - 1, argv + 1, sample_cases, ARRAY_COUNT(sample_cases));
    }
    if (argc > 1 && strcmp(argv[1], "--counters") == 0) {
        const struct bench_case counter_cases[] = {
            {"none", counters, {}},
            {"items", counters, {.rows = 1}},
            {"bytes", counters, {.variant = VARIANT_COUNTER_BYTES}},
            {"both", counters, {.rows = 1, .variant = VARIANT_COUNTER_BYTES}}
        };
        argv[1] = argv[0];
        return bench_main(argc - 1, argv + 1, counter_cases, ARRAY_COUNT(counter_cases));
    }
    if (argc > 1 && (strcmp(argv[1], "--failure") == 0 || strcmp(argv[1], "--leak") == 0)) {
        const struct bench_case error_cases[] = {{"error", strcmp(argv[1], "--failure") == 0 ? failure : leak, {}}};
        argv[1] = argv[0];
        return bench_main(argc - 1, argv + 1, error_cases, ARRAY_COUNT(error_cases));
    }
    int status = statistics_checks();
    if (status == 0) {
        status = formatting_checks();
    }
    if (status == 0) {
        status = generator_checks();
    }
    if (status != 0) {
        return 1;
    }
    const struct bench_case zero_cases[] = {{"zero", summary_samples, {}}};
    char* zero_arguments[] = {argv[0], "--min-time", "2x"};
    if (bench_main(ARRAY_COUNT(zero_arguments), zero_arguments, zero_cases, ARRAY_COUNT(zero_cases)) != 1) {
        return 1;
    }
    const struct bench_case exact_cases[] = {{"lifecycle", lifecycle, {.rows = 34}}};
    char* exact_arguments[] = {argv[0], "--min-time", "17x", "--repetitions", "2"};
    char* exact_fixed_arguments[] = {argv[0], "--min-time", "17x", "--repetitions", "2", "--pool", "fixed"};
    const struct bench_case lifecycle_cases[] = {{"lifecycle", lifecycle, {}}};
    char* calibrated_arguments[] = {argv[0], "--min-time", "0.01", "--repetitions", "3"};
    char* calibrated_fixed_arguments[] = {argv[0], "--min-time", "0.01", "--repetitions", "3", "--pool", "fixed"};
    char* smoke_arguments[] = {argv[0], "--smoke"};
    char* smoke_fixed_arguments[] = {argv[0], "--smoke", "--pool", "fixed"};
    const struct bench_case contract_cases[] = {{"timing_contract", timing_contract, {}}};
    char* contract_arguments[] = {argv[0], "--min-time", "8x", "--repetitions", "5"};
    return bench_main(ARRAY_COUNT(exact_arguments), exact_arguments, exact_cases, ARRAY_COUNT(exact_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(exact_fixed_arguments),
                              exact_fixed_arguments,
                              exact_cases,
                              ARRAY_COUNT(exact_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(calibrated_arguments),
                              calibrated_arguments,
                              lifecycle_cases,
                              ARRAY_COUNT(lifecycle_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(calibrated_fixed_arguments),
                              calibrated_fixed_arguments,
                              lifecycle_cases,
                              ARRAY_COUNT(lifecycle_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(smoke_arguments),
                              smoke_arguments,
                              lifecycle_cases,
                              ARRAY_COUNT(lifecycle_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(smoke_fixed_arguments),
                              smoke_fixed_arguments,
                              lifecycle_cases,
                              ARRAY_COUNT(lifecycle_cases)) != 0 ||
                   bench_main(ARRAY_COUNT(contract_arguments),
                              contract_arguments,
                              contract_cases,
                              ARRAY_COUNT(contract_cases)) != 0
               ? 1
               : 0;
}
