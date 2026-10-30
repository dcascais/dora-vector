// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bench.h"

#include "bench_config.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/vector.h"
#include "types/array_count.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <regex.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#elifdef __linux__
#include <dirent.h>
#include <limits.h>
#else
#error "dora-vector benchmarks support macOS and Linux"
#endif

//------------------------------------------------------------------------------
//
// Statistics
//
//------------------------------------------------------------------------------

int bench_describe(const double* values, size_t count, struct bench_statistics* out_statistics) {
    if (count == 0) {
        return EDOM;
    }
    struct bench_statistics statistics = {.min = values[0], .max = values[0]};
    double squares = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index]) || values[index] < 0) {
            return EDOM;
        }
        statistics.min = values[index] < statistics.min ? values[index] : statistics.min;
        statistics.max = values[index] > statistics.max ? values[index] : statistics.max;
        const double delta = values[index] - statistics.mean;
        statistics.mean += delta / (double)(index + 1);
        squares += delta * (values[index] - statistics.mean);
    }
    if (count > 1 && statistics.mean != 0) {
        statistics.variation = sqrt(squares / (double)(count - 1)) / statistics.mean;
    }
    if (!isfinite(statistics.mean) || !isfinite(statistics.variation)) {
        return EDOM;
    }
    *out_statistics = statistics;
    return 0;
}

static constexpr uint64_t CALIBRATION_MAX_ITERATIONS = 1'000'000'000;

uint64_t bench_calibrate(uint64_t iterations, double elapsed, double min_seconds) {
    if (iterations >= CALIBRATION_MAX_ITERATIONS || elapsed >= min_seconds) {
        return iterations;
    }
    const double next =
        elapsed < min_seconds * 0.1 ? (double)iterations * 10 : (double)iterations * min_seconds * 1.4 / elapsed;
    if (next >= (double)CALIBRATION_MAX_ITERATIONS) {
        return CALIBRATION_MAX_ITERATIONS;
    }
    const uint64_t grown = (uint64_t)lround(next);
    return grown > iterations ? grown : iterations + 1;
}

//------------------------------------------------------------------------------
//
// Formatting
//
//------------------------------------------------------------------------------

int bench_format_status(int length, size_t capacity) {
    return length < 0 ? EIO : (size_t)length >= capacity ? ENOSPC : 0;
}

int bench_format_time(double nanoseconds, size_t capacity, char* out_text) {
    if (!isfinite(nanoseconds) || nanoseconds < 0) {
        return EDOM;
    }
    const char* units[] = {"ns", "us", "ms", "s"};
    size_t unit = 0;
    while (nanoseconds >= 1000 && unit + 1 < ARRAY_COUNT(units)) {
        nanoseconds /= 1000;
        ++unit;
    }
    return bench_format_status(snprintf(out_text, capacity, "%.3f %s", nanoseconds, units[unit]), capacity);
}

static constexpr double RATE_SCALE_THRESHOLD = 999.5;

int bench_format_rate(double rate, bool binary, size_t capacity, char* out_text) {
    if (!isfinite(rate) || rate < 0) {
        return EDOM;
    }
    const char* decimal_units[] = {"", "k", "M", "G", "T", "P", "E"};
    const char* binary_units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    const double scale = binary ? 1024 : 1000;
    size_t unit = 0;
    while (rate >= RATE_SCALE_THRESHOLD && unit + 1 < ARRAY_COUNT(decimal_units)) {
        rate /= scale;
        ++unit;
    }
    return bench_format_status(
        snprintf(out_text, capacity, "%.3g %s/s", rate, binary ? binary_units[unit] : decimal_units[unit]),
        capacity);
}

int bench_escape(const char* source, size_t capacity, char* out_text) {
    if (capacity < 3) {
        return ENOSPC;
    }
    size_t length = 0;
    out_text[length++] = '"';
    for (const unsigned char* cursor = (const unsigned char*)source; *cursor != 0; ++cursor) {
        char escaped[7];
        size_t count = 1;
        escaped[0] = (char)*cursor;
        if (*cursor == '"' || *cursor == '\\') {
            escaped[0] = '\\';
            escaped[1] = (char)*cursor;
            count = 2;
        } else if (*cursor < 0x20) {
            const int status =
                bench_format_status(snprintf(escaped, sizeof(escaped), "\\u%04x", (unsigned)*cursor), sizeof(escaped));
            if (status != 0) {
                return status;
            }
            count = 6;
        }
        if (count > capacity - length - 2) {
            return ENOSPC;
        }
        memcpy(out_text + length, escaped, count);
        length += count;
    }
    out_text[length++] = '"';
    out_text[length] = 0;
    return 0;
}

//------------------------------------------------------------------------------
//
// Platform
//
//------------------------------------------------------------------------------

static constexpr size_t GOVERNOR_BYTES = 32;

struct platform {
    long performance_cpus;
    long efficiency_cpus;
    char governor[GOVERNOR_BYTES];
};

#ifdef __APPLE__

//--------------------------------------
// macOS
//--------------------------------------

static constexpr clockid_t WALL_CLOCK = CLOCK_UPTIME_RAW;
static constexpr char WALL_CLOCK_NAME[] = "uptime-raw";

int bench_read_clock(uint64_t* out_nanoseconds) {
    const uint64_t nanoseconds = clock_gettime_nsec_np(WALL_CLOCK);
    if (nanoseconds == 0) {
        return errno;
    }
    *out_nanoseconds = nanoseconds;
    return 0;
}

static int read_sysctl(const char* name, long* out_value) {
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
        return errno;
    }
    *out_value = value;
    return 0;
}

static int prepare_platform(struct platform* out_platform) {
    struct platform platform = {};
    long levels = 0;
    int status = read_sysctl("hw.nperflevels", &levels);
    status = status == ENOENT ? 0 : status;
    if (status == 0 && levels >= 2) {
        status = read_sysctl("hw.perflevel0.logicalcpu", &platform.performance_cpus);
    }
    if (status == 0 && levels >= 2) {
        status = read_sysctl("hw.perflevel1.logicalcpu", &platform.efficiency_cpus);
    }
    if (status == 0) {
        *out_platform = platform;
    }
    return status;
}

#elifdef __linux__

//--------------------------------------
// Linux
//--------------------------------------

static constexpr clockid_t WALL_CLOCK = CLOCK_MONOTONIC;
static constexpr char WALL_CLOCK_NAME[] = "monotonic";

int bench_read_clock(uint64_t* out_nanoseconds) {
    struct timespec time;
    if (clock_gettime(WALL_CLOCK, &time) != 0) {
        return errno;
    }
    *out_nanoseconds = (uint64_t)time.tv_sec * 1'000'000'000 + (uint64_t)time.tv_nsec;
    return 0;
}

static constexpr char CPUFREQ_DIRECTORY[] = "/sys/devices/system/cpu/cpufreq";
static constexpr char CPUFREQ_POLICY_PREFIX[] = "policy";

static int read_governor(const char* policy, size_t capacity, char* out_governor) {
    char path[PATH_MAX];
    const int length = snprintf(path, sizeof(path), "%s/%s/scaling_governor", CPUFREQ_DIRECTORY, policy);
    int status = bench_format_status(length, sizeof(path));
    FILE* file = status == 0 ? fopen(path, "r") : nullptr;
    if (status == 0 && file == nullptr) {
        status = errno;
    }
    if (status == 0) {
        errno = 0;
        if (fgets(out_governor, (int)capacity, file) == nullptr) {
            status = ferror(file) && errno != 0 ? errno : EIO;
        }
    }
    if (file != nullptr && fclose(file) != 0 && status == 0) {
        status = EIO;
    }
    char* newline = status == 0 ? strchr(out_governor, '\n') : nullptr;
    if (newline != nullptr) {
        *newline = 0;
    }
    return status;
}

static int read_governors(DIR* directory, struct platform* in_out_platform) {
    char governor[GOVERNOR_BYTES];
    for (;;) {
        errno = 0;
        const struct dirent* entry = readdir(directory);
        if (entry == nullptr) {
            return errno;
        }
        if (strncmp(entry->d_name, CPUFREQ_POLICY_PREFIX, sizeof(CPUFREQ_POLICY_PREFIX) - 1) != 0) {
            continue;
        }
        const int status = read_governor(entry->d_name, sizeof(governor), governor);
        if (status == ENOENT || status == EBUSY) {
            continue;
        }
        if (status != 0) {
            return status;
        }
        if (in_out_platform->governor[0] == 0) {
            memcpy(in_out_platform->governor, governor, strlen(governor) + 1);
        } else if (strcmp(in_out_platform->governor, governor) != 0) {
            memcpy(in_out_platform->governor, "mixed", sizeof("mixed"));
        }
    }
}

static int prepare_platform(struct platform* out_platform) {
    struct platform platform = {};
    DIR* directory = opendir(CPUFREQ_DIRECTORY);
    int status = directory != nullptr ? read_governors(directory, &platform) : errno == ENOENT ? 0 : errno;
    if (directory != nullptr && closedir(directory) != 0 && status == 0) {
        status = errno;
    }
    if (status == 0) {
        *out_platform = platform;
    }
    return status;
}

#endif

//------------------------------------------------------------------------------
//
// Measurement
//
//------------------------------------------------------------------------------

//--------------------------------------
// Status
//--------------------------------------

void bench_check(struct bench* bench, int status) {
    if (status != 0 && bench->status == 0) {
        bench->status = status;
        bench->error_message = strerror(status);
    }
}

//--------------------------------------
// Runs
//--------------------------------------

struct usage {
    uint64_t faults;
    uint64_t context_switches;
};

static int read_usage(struct usage* out_usage) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return errno;
    }
    *out_usage = (struct usage){
        .faults = (uint64_t)usage.ru_minflt + (uint64_t)usage.ru_majflt,
        .context_switches = (uint64_t)usage.ru_nvcsw + (uint64_t)usage.ru_nivcsw
    };
    return 0;
}

struct epoch {
    const struct bench_operation* operation;
    void* context;
    void* slots;
    int* statuses;
    size_t capacity;
    size_t limit;
};

static int resize_epoch(struct epoch* epoch, size_t capacity) {
    const size_t slot_bytes = epoch->operation->slot_bytes;
    void* slots = slot_bytes == 0 ? nullptr : calloc(capacity, slot_bytes);
    int* statuses = calloc(capacity, sizeof(*statuses));
    if (statuses == nullptr || (slot_bytes != 0 && slots == nullptr)) {
        free(statuses);
        free(slots);
        return ENOMEM;
    }
    free(epoch->statuses);
    free(epoch->slots);
    epoch->slots = slots;
    epoch->statuses = statuses;
    epoch->capacity = capacity;
    return 0;
}

static int finish_epoch(const struct epoch* epoch, size_t count) {
    if (epoch->operation->release != nullptr) {
        epoch->operation->release(epoch->slots, count);
    }
    if (epoch->slots != nullptr) {
        memset(epoch->slots, 0, count * epoch->operation->slot_bytes);
    }
    int status = 0;
    for (size_t index = 0; status == 0 && index < count; ++index) {
        status = epoch->statuses[index];
    }
    return status;
}

static int time_epoch(const struct epoch* epoch, size_t count, uint64_t* out_nanoseconds) {
    void (*const run)(void*, size_t, void*, int*) = epoch->operation->run;
    void* const context = epoch->context;
    void* const slots = epoch->slots;
    int* const statuses = epoch->statuses;
    uint64_t start = 0;
    uint64_t end = 0;
    const int start_status = bench_read_clock(&start);
    run(context, count, slots, statuses);
    const int end_status = bench_read_clock(&end);
    const int run_status = finish_epoch(epoch, count);
    const int status = start_status != 0 ? start_status : end_status != 0 ? end_status : run_status;
    if (status == 0) {
        *out_nanoseconds = end - start;
    }
    return status;
}

static constexpr uint64_t EPOCH_MIN_NANOSECONDS = 4'000;

static int time_iterations(struct epoch* epoch, uint64_t iterations, uint64_t* out_nanoseconds) {
    uint64_t total = 0;
    int status = 0;
    for (uint64_t done = 0; status == 0 && done < iterations;) {
        const size_t count = iterations - done < epoch->capacity ? (size_t)(iterations - done) : epoch->capacity;
        uint64_t elapsed = 0;
        status = time_epoch(epoch, count, &elapsed);
        done += count;
        total += elapsed;
        if (status == 0 && count == epoch->capacity && count < epoch->limit && elapsed < EPOCH_MIN_NANOSECONDS) {
            status = resize_epoch(epoch, count > epoch->limit / 2 ? epoch->limit : count * 2);
        }
    }
    if (status == 0) {
        *out_nanoseconds = total;
    }
    return status;
}

static int time_repetition(struct bench* bench, struct epoch* epoch, size_t repetition) {
    struct usage before = {};
    struct usage after = {};
    uint64_t elapsed = 0;
    int status = read_usage(&before);
    if (status == 0) {
        status = time_iterations(epoch, bench->iterations, &elapsed);
    }
    if (status == 0) {
        status = read_usage(&after);
    }
    if (status == 0) {
        bench->wall_nanoseconds[repetition] = (double)elapsed / (double)bench->iterations;
        bench->faults[repetition] = after.faults - before.faults;
        bench->context_switches[repetition] = after.context_switches - before.context_switches;
    }
    return status;
}

static int time_samples(struct bench* bench, struct epoch* epoch) {
    bool is_calibrating = bench->mode == BENCH_MODE_CALIBRATED;
    const uint64_t limit = is_calibrating ? CALIBRATION_MAX_ITERATIONS : bench->iterations;
    epoch->limit = limit < SIZE_MAX ? (size_t)limit : SIZE_MAX;
    int status = 0;
    do {
        uint64_t elapsed = 0;
        status = time_iterations(epoch, bench->iterations, &elapsed);
        if (status == 0 && is_calibrating) {
            const uint64_t next = bench_calibrate(bench->iterations, (double)elapsed * 1e-9, bench->min_seconds);
            is_calibrating = next != bench->iterations;
            bench->iterations = next;
        }
    } while (status == 0 && is_calibrating);
    epoch->limit = epoch->capacity;
    bench->epoch_iterations = epoch->capacity < bench->iterations ? epoch->capacity : bench->iterations;
    for (size_t repetition = 0; status == 0 && repetition < bench->repetitions; ++repetition) {
        status = time_repetition(bench, epoch, repetition);
    }
    return status;
}

void bench_measure(struct bench* bench, const struct bench_operation* operation, void* context) {
    if (bench->status != 0) {
        return;
    }
    if (bench->measured) {
        bench_check(bench, EINVAL);
        return;
    }
    bench->measured = true;
    struct epoch epoch = {.operation = operation, .context = context};
    int status = resize_epoch(&epoch, 1);
    if (status == 0 && bench->mode == BENCH_MODE_SMOKE) {
        operation->run(context, 1, epoch.slots, epoch.statuses);
        status = finish_epoch(&epoch, 1);
    } else if (status == 0) {
        status = time_samples(bench, &epoch);
    }
    bench_check(bench, status);
    free(epoch.statuses);
    free(epoch.slots);
}

void bench_release_vectors(void* slots, size_t count) {
    struct dora_vector* vectors = slots;
    for (size_t index = 0; index < count; ++index) {
        dora_vector_release(&vectors[index]);
    }
}

//--------------------------------------
// Counters
//--------------------------------------

void bench_set_items(struct bench* bench, uint64_t items_per_iteration) {
    bench->items_per_iteration = items_per_iteration;
}

void bench_set_bytes(struct bench* bench, uint64_t bytes_per_iteration) {
    bench->bytes_per_iteration = bytes_per_iteration;
}

//------------------------------------------------------------------------------
//
// Execution
//
//------------------------------------------------------------------------------

//--------------------------------------
// Options
//--------------------------------------

struct options {
    double min_seconds;
    uint64_t iterations;
    size_t repetitions;
    size_t fixed_bytes;
    size_t granule_bytes;
    enum bench_mode mode;
    bool fixed;
    bool help;
    const char* filter;
    const char* json;
};

static int write_usage(FILE* stream) {
    return fputs("Usage: benchmark [--list] [--filter REGEX] [--min-time S|Nx]\n"
                 "                 [--repetitions N] [--json PATH] [--pool host|fixed[:BYTES]] [--smoke] [--help]\n",
                 stream) < 0
               ? EIO
               : 0;
}

static int parse_positive_integer(const char* text, char suffix, uint64_t* out_value) {
    if (text[0] < '0' || text[0] > '9') {
        return EINVAL;
    }
    errno = 0;
    char* end;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end[0] != suffix || (suffix != 0 && end[1] != 0) || value == 0) {
        return EINVAL;
    }
    *out_value = value;
    return 0;
}

static int parse_minimum_time(const char* text, struct options* options, bool* out_is_exact) {
    const size_t length = strlen(text);
    *out_is_exact = length > 1 && text[length - 1] == 'x';
    if (*out_is_exact) {
        return parse_positive_integer(text, 'x', &options->iterations);
    }
    errno = 0;
    char* end;
    const double seconds = strtod(text, &end);
    if (errno != 0 || end == text || *end != 0 || !isfinite(seconds) || seconds <= 0) {
        return EINVAL;
    }
    options->min_seconds = seconds;
    options->iterations = 1;
    return 0;
}

static int parse_options(int argc, char** argv, struct options* options) {
    bool is_exact = false;
    bool has_timed_option = false;
    bool has_pool_option = false;
    for (int index = 1; index < argc; ++index) {
        const char* option = argv[index];
        if (strcmp(option, "--smoke") == 0 || strcmp(option, "--list") == 0) {
            const enum bench_mode mode = strcmp(option, "--list") == 0 ? BENCH_MODE_LIST : BENCH_MODE_SMOKE;
            if (options->mode != BENCH_MODE_CALIBRATED && options->mode != mode) {
                return EINVAL;
            }
            options->mode = mode;
        } else if (strcmp(option, "--help") == 0) {
            options->help = true;
        } else {
            if (index + 1 >= argc) {
                return EINVAL;
            }
            const char* value = argv[++index];
            if (strcmp(option, "--filter") == 0) {
                options->filter = value;
            } else if (strcmp(option, "--json") == 0) {
                has_timed_option = true;
                options->json = value;
            } else if (strcmp(option, "--min-time") == 0) {
                has_timed_option = true;
                const int status = parse_minimum_time(value, options, &is_exact);
                if (status != 0) {
                    return status;
                }
            } else if (strcmp(option, "--repetitions") == 0) {
                has_timed_option = true;
                uint64_t repetitions;
                const int status = parse_positive_integer(value, 0, &repetitions);
                if (status != 0 || repetitions > SIZE_MAX / sizeof(double)) {
                    return EINVAL;
                }
                options->repetitions = (size_t)repetitions;
            } else if (strcmp(option, "--pool") == 0) {
                has_pool_option = true;
                options->fixed = false;
                if (strcmp(value, "fixed") == 0 || strncmp(value, "fixed:", sizeof("fixed:") - 1) == 0) {
                    options->fixed = true;
                    if (value[sizeof("fixed") - 1] == ':') {
                        uint64_t bytes;
                        const int status = parse_positive_integer(value + sizeof("fixed:") - 1, 0, &bytes);
                        if (status != 0 || bytes > SIZE_MAX) {
                            return EINVAL;
                        }
                        options->fixed_bytes = (size_t)bytes;
                    }
                } else if (strcmp(value, "host") != 0) {
                    return EINVAL;
                }
            } else {
                return EINVAL;
            }
        }
    }
    if (options->mode != BENCH_MODE_CALIBRATED) {
        return has_timed_option || (options->mode == BENCH_MODE_LIST && has_pool_option) ? EINVAL : 0;
    }
    if (is_exact) {
        options->mode = BENCH_MODE_EXACT;
    }
    return 0;
}

//--------------------------------------
// Pools
//--------------------------------------

static int init_region(void** region, size_t byte_count, size_t* out_granule_bytes) {
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (page_bytes <= 0) {
        return EIO;
    }
    *region = malloc(byte_count);
    if (*region == nullptr) {
        return ENOMEM;
    }
    volatile uint8_t* bytes = *region;
    for (size_t offset = 0; offset < byte_count; offset += (size_t)page_bytes) {
        bytes[offset] = 0;
    }
    bytes[byte_count - 1] = 0;
    struct dora_pool pool = {};
    size_t granule_bytes = BENCH_POOL_GRANULE_BYTES;
    int status = dora_pool_init_fixed_region(&pool, *region, byte_count, granule_bytes);
    while (status == ERANGE) {
        granule_bytes *= 2;
        status = dora_pool_init_fixed_region(&pool, *region, byte_count, granule_bytes);
    }
    if (status == 0) {
        status = dora_pool_release(&pool);
    }
    if (status == 0) {
        *out_granule_bytes = granule_bytes;
    } else {
        free(*region);
        *region = nullptr;
    }
    return status;
}

static int init_pool(struct dora_pool* pool, const struct options* options, void* region) {
    return options->fixed ? dora_pool_init_fixed_region(pool, region, options->fixed_bytes, options->granule_bytes)
                          : dora_pool_init(pool, SIZE_MAX, options->granule_bytes);
}

//--------------------------------------
// Warm-up
//--------------------------------------

static constexpr uint32_t WARM_UP_CHUNK_STEPS = 65'536;
static constexpr uint64_t WARM_UP_MULTIPLIER = 6'364'136'223'846'793'005U;
static constexpr uint64_t WARM_UP_INCREMENT = 1'442'695'040'888'963'407U;
static constexpr uint64_t WARM_UP_HOLD_NANOSECONDS = 250'000'000;
static constexpr uint64_t WARM_UP_LIMIT_NANOSECONDS = 2'000'000'000;

static int warm_up() {
    volatile uint64_t state = 0;
    uint64_t start = 0;
    int status = bench_read_clock(&start);
    uint64_t now = start;
    uint64_t improved = start;
    uint64_t best = UINT64_MAX;
    while (status == 0 && now - improved < WARM_UP_HOLD_NANOSECONDS && now - start < WARM_UP_LIMIT_NANOSECONDS) {
        const uint64_t chunk_start = now;
        uint64_t value = state;
        for (uint32_t step = 0; step < WARM_UP_CHUNK_STEPS; ++step) {
            value = value * WARM_UP_MULTIPLIER + WARM_UP_INCREMENT;
        }
        state = value;
        status = bench_read_clock(&now);
        const uint64_t elapsed = now - chunk_start;
        if (status == 0 && (best == UINT64_MAX || elapsed * 100 < best * 99)) {
            best = elapsed;
            improved = now;
        }
    }
    return status;
}

//--------------------------------------
// Context
//--------------------------------------

static int write_string(const char* value, FILE* stream) {
    size_t capacity;
    if (ckd_mul(&capacity, strlen(value), 6) || ckd_add(&capacity, capacity, 3)) {
        return EOVERFLOW;
    }
    char* escaped = malloc(capacity);
    if (escaped == nullptr) {
        return ENOMEM;
    }
    int status = bench_escape(value, capacity, escaped);
    if (status == 0 && fputs(escaped, stream) < 0) {
        status = EIO;
    }
    free(escaped);
    return status;
}

static constexpr uint64_t CLOCK_PROBE_NANOSECONDS = 100'000;

static int measure_clock(uint64_t* out_resolution, double* out_read) {
    struct timespec resolution;
    if (clock_getres(WALL_CLOCK, &resolution) != 0) {
        return errno;
    }
    uint64_t first = 0;
    int status = bench_read_clock(&first);
    uint64_t last = first;
    uint64_t reads = 0;
    while (status == 0 && last - first < CLOCK_PROBE_NANOSECONDS) {
        status = bench_read_clock(&last);
        ++reads;
    }
    if (status == 0) {
        *out_resolution = (uint64_t)resolution.tv_sec * 1'000'000'000 + (uint64_t)resolution.tv_nsec;
        *out_read = (double)(last - first) / (double)reads;
    }
    return status;
}

struct context {
    char date[64];
    char host[256];
    long cpus;
    long page_bytes;
    uint64_t clock_resolution;
    double clock_read;
    struct platform platform;
};

static int init_context(struct context* context) {
    const time_t now = time(nullptr);
    struct tm calendar;
    if (now == (time_t)-1 || localtime_r(&now, &calendar) == nullptr ||
        strftime(context->date, sizeof(context->date), "%Y-%m-%dT%H:%M:%S%z", &calendar) == 0) {
        return EIO;
    }
    if (gethostname(context->host, sizeof(context->host)) != 0) {
        return errno;
    }
    context->host[sizeof(context->host) - 1] = 0;
    context->cpus = sysconf(_SC_NPROCESSORS_ONLN);
    context->page_bytes = sysconf(_SC_PAGESIZE);
    if (context->cpus <= 0 || context->page_bytes <= 0) {
        return EIO;
    }
    const int status = prepare_platform(&context->platform);
    return status == 0 ? measure_clock(&context->clock_resolution, &context->clock_read) : status;
}

static int write_console_context(const struct context* context, const char* executable, const struct options* options) {
    const struct platform* platform = &context->platform;
    if (printf("%s\nRunning %s\nhost: %s  compiler: %s  build: %s  c_flags: \"%s\"  library_options: \"%s\"\ncpus: %ld",
               context->date,
               executable,
               context->host,
               BENCH_COMPILER,
               BENCH_BUILD_CONFIG,
               BENCH_C_FLAGS,
               BENCH_LIBRARY_OPTIONS,
               context->cpus) < 0) {
        return EIO;
    }
    if (platform->performance_cpus != 0 &&
        printf(" (%ldP+%ldE)", platform->performance_cpus, platform->efficiency_cpus) < 0) {
        return EIO;
    }
    if (printf("  page: %ld B", context->page_bytes) < 0) {
        return EIO;
    }
    if (platform->governor[0] != 0 && printf("  governor: %s", platform->governor) < 0) {
        return EIO;
    }
    if (printf("\nclock: %s  resolution: %" PRIu64 " ns  read: %.1f ns\npool: %s  granule: %zu B",
               WALL_CLOCK_NAME,
               context->clock_resolution,
               context->clock_read,
               options->fixed ? "fixed" : "host",
               options->granule_bytes) < 0) {
        return EIO;
    }
    if (options->fixed && printf("  bytes: %zu B", options->fixed_bytes) < 0) {
        return EIO;
    }
    return puts("") < 0 ? EIO : 0;
}

static int
write_context(const struct context* context, const char* executable, const struct options* options, FILE* stream) {
    const struct platform* platform = &context->platform;
    const char* keys[] = {"date", "host_name", "executable", "compiler", "build_config", "c_flags", "library_options"};
    const char* values[] = {
        context->date,
        context->host,
        executable,
        BENCH_COMPILER,
        BENCH_BUILD_CONFIG,
        BENCH_C_FLAGS,
        BENCH_LIBRARY_OPTIONS
    };
    if (fputs("{\"context\":{", stream) < 0) {
        return EIO;
    }
    for (size_t index = 0; index < ARRAY_COUNT(keys); ++index) {
        if (fprintf(stream, "%s\"%s\":", index == 0 ? "" : ",", keys[index]) < 0) {
            return EIO;
        }
        const int status = write_string(values[index], stream);
        if (status != 0) {
            return status;
        }
    }
    if (fprintf(stream, ",\"num_cpus\":%ld", context->cpus) < 0) {
        return EIO;
    }
    if (platform->performance_cpus != 0 && fprintf(stream,
                                                   ",\"performance_cpus\":%ld,\"efficiency_cpus\":%ld",
                                                   platform->performance_cpus,
                                                   platform->efficiency_cpus) < 0) {
        return EIO;
    }
    if (fprintf(stream,
                ",\"page_bytes\":%ld,\"clock\":\"%s\",\"clock_resolution_ns\":%" PRIu64 ",\"clock_read_ns\":%.17g",
                context->page_bytes,
                WALL_CLOCK_NAME,
                context->clock_resolution,
                context->clock_read) < 0) {
        return EIO;
    }
    if (platform->governor[0] != 0) {
        if (fputs(",\"cpu_governor\":", stream) < 0) {
            return EIO;
        }
        const int status = write_string(platform->governor, stream);
        if (status != 0) {
            return status;
        }
    }
    if (fprintf(stream,
                ",\"pool\":\"%s\",\"pool_granule_bytes\":%zu",
                options->fixed ? "fixed" : "host",
                options->granule_bytes) < 0) {
        return EIO;
    }
    if (options->fixed && fprintf(stream, ",\"pool_bytes\":%zu", options->fixed_bytes) < 0) {
        return EIO;
    }
    return fputs("},\"benchmarks\":[\n", stream) < 0 ? EIO : 0;
}

//--------------------------------------
// Summaries
//--------------------------------------

struct summary {
    struct bench_statistics statistics;
    double items;
    double bytes;
    uint64_t faults;
    uint64_t context_switches;
};

static constexpr char SPAN_BELOW_RESOLUTION[] = "timed span below the clock resolution; raise N in --min-time Nx";

static struct summary summarize_samples(struct bench* bench) {
    struct summary summary = {};
    if (bench->status != 0) {
        return summary;
    }
    const int status = bench_describe(bench->wall_nanoseconds, bench->repetitions, &summary.statistics);
    if (status != 0) {
        bench_check(bench, status);
        return (struct summary){};
    }
    if (summary.statistics.min == 0) {
        bench->status = ERANGE;
        bench->error_message = SPAN_BELOW_RESOLUTION;
        return (struct summary){};
    }
    const double seconds = summary.statistics.min * 1e-9;
    if (bench->items_per_iteration != 0) {
        summary.items = (double)bench->items_per_iteration / seconds;
    }
    if (bench->bytes_per_iteration != 0) {
        summary.bytes = (double)bench->bytes_per_iteration / seconds;
    }
    if (!isfinite(summary.items) || !isfinite(summary.bytes)) {
        bench_check(bench, EDOM);
        return (struct summary){};
    }
    for (size_t index = 0; index < bench->repetitions; ++index) {
        summary.faults += bench->faults[index];
        summary.context_switches += bench->context_switches[index];
    }
    return summary;
}

//--------------------------------------
// Output
//--------------------------------------

static constexpr char NAME_HEADER[] = "Benchmark";

static int measure_name_width(const struct bench_case* cases, size_t case_count) {
    size_t width = sizeof(NAME_HEADER) - 1;
    for (size_t index = 0; index < case_count; ++index) {
        const size_t length = strlen(cases[index].name);
        width = length > width ? length : width;
    }
    return (int)width;
}

static int write_console_header(int name_width) {
    return printf("%-*s %13s %13s %13s %10s %12s %12s %12s %8s %12s\n",
                  name_width,
                  NAME_HEADER,
                  "Min",
                  "Mean",
                  "Max",
                  "Coef. var.",
                  "Items/s",
                  "Bytes/s",
                  "Ctx switches",
                  "Faults",
                  "Iterations") < 0
               ? EIO
               : 0;
}

static int write_console_status(const struct bench* bench, const char* name, int name_width) {
    const int length = bench->status != 0 ? printf("%-*s ERROR: %s\n", name_width, name, bench->error_message)
                                          : printf("%-*s ok\n", name_width, name);
    return length < 0 ? EIO : 0;
}

static constexpr char RATE_ABSENT[] = "-";

static int
write_console_row(const struct bench* bench, const char* name, int name_width, const struct summary* summary) {
    if (bench->status != 0) {
        return write_console_status(bench, name, name_width);
    }
    char min[64];
    char mean[64];
    char max[64];
    char bytes[64];
    char items[64];
    int status = bench_format_time(summary->statistics.min, sizeof(min), min);
    if (status == 0) {
        status = bench_format_time(summary->statistics.mean, sizeof(mean), mean);
    }
    if (status == 0) {
        status = bench_format_time(summary->statistics.max, sizeof(max), max);
    }
    if (status == 0 && summary->bytes > 0) {
        status = bench_format_rate(summary->bytes, true, sizeof(bytes), bytes);
    }
    if (status == 0 && summary->items > 0) {
        status = bench_format_rate(summary->items, false, sizeof(items), items);
    }
    if (status != 0) {
        return status;
    }
    return printf("%-*s %13s %13s %13s %9.2f%% %12s %12s %12" PRIu64 " %8" PRIu64 " %12" PRIu64 "\n",
                  name_width,
                  name,
                  min,
                  mean,
                  max,
                  summary->statistics.variation * 100,
                  summary->items > 0 ? items : RATE_ABSENT,
                  summary->bytes > 0 ? bytes : RATE_ABSENT,
                  summary->context_switches,
                  summary->faults,
                  bench->iterations) < 0
               ? EIO
               : 0;
}

static int write_samples(const struct bench* bench, FILE* stream) {
    if (fputs(",\"samples\":[", stream) < 0) {
        return EIO;
    }
    for (size_t index = 0; index < bench->repetitions; ++index) {
        if (fprintf(stream,
                    "%s{\"time_ns\":%.17g,\"context_switches\":%" PRIu64 ",\"faults\":%" PRIu64 "}",
                    index == 0 ? "" : ",",
                    bench->wall_nanoseconds[index],
                    bench->context_switches[index],
                    bench->faults[index]) < 0) {
            return EIO;
        }
    }
    return fputs("]", stream) < 0 ? EIO : 0;
}

static int write_json_row(const struct bench* bench, const char* name, const struct summary* summary, FILE* stream) {
    if (fputs("{\"name\":", stream) < 0) {
        return EIO;
    }
    int status = write_string(name, stream);
    if (status != 0) {
        return status;
    }
    if (bench->status != 0) {
        if (fputs(",\"error_message\":", stream) < 0) {
            return EIO;
        }
        status = write_string(bench->error_message, stream);
        return status != 0 ? status : fputs("}", stream) < 0 ? EIO : 0;
    }
    if (fprintf(stream,
                ",\"min_ns\":%.17g,\"mean_ns\":%.17g,\"max_ns\":%.17g,\"coefficient_of_variation\":%.17g",
                summary->statistics.min,
                summary->statistics.mean,
                summary->statistics.max,
                summary->statistics.variation) < 0) {
        return EIO;
    }
    if (summary->items > 0 && fprintf(stream, ",\"items_per_second\":%.17g", summary->items) < 0) {
        return EIO;
    }
    if (summary->bytes > 0 && fprintf(stream, ",\"bytes_per_second\":%.17g", summary->bytes) < 0) {
        return EIO;
    }
    if (fprintf(stream,
                ",\"iterations\":%" PRIu64 ",\"repetitions\":%zu,\"epoch_iterations\":%" PRIu64,
                bench->iterations,
                bench->repetitions,
                bench->epoch_iterations) < 0) {
        return EIO;
    }
    status = write_samples(bench, stream);
    return status != 0 ? status : fputs("}", stream) < 0 ? EIO : 0;
}

//--------------------------------------
// Runs
//--------------------------------------

static int init_timing(struct bench* bench, const struct options* options) {
    bench->min_seconds = options->min_seconds;
    bench->iterations = options->iterations;
    bench->repetitions = options->repetitions;
    bench->wall_nanoseconds = calloc(bench->repetitions, sizeof(*bench->wall_nanoseconds));
    bench->faults = calloc(bench->repetitions, sizeof(*bench->faults));
    bench->context_switches = calloc(bench->repetitions, sizeof(*bench->context_switches));
    return bench->wall_nanoseconds == nullptr || bench->faults == nullptr || bench->context_switches == nullptr ? ENOMEM
                                                                                                                : 0;
}

static int run_case(const struct bench_case* entry,
                    const struct options* options,
                    int name_width,
                    FILE* json,
                    void** in_out_region) {
    struct bench bench = {.pool_granule_bytes = options->granule_bytes, .mode = options->mode};
    if (bench.mode != BENCH_MODE_SMOKE) {
        bench_check(&bench, init_timing(&bench, options));
    }
    if (bench.status == 0) {
        bench_check(&bench, init_pool(&bench.pool, options, *in_out_region));
    }
    if (bench.status == 0) {
        entry->function(&bench, &entry->args);
    }
    if (dora_pool_is_valid(bench.pool)) {
        const int status = dora_pool_release(&bench.pool);
        bench_check(&bench, status);
    }
    if (dora_pool_is_valid(bench.pool)) {
        *in_out_region = nullptr;
    }
    bench_check(&bench, bench.measured ? 0 : EINVAL);
    if (bench.mode == BENCH_MODE_SMOKE) {
        bench_check(&bench, write_console_status(&bench, entry->name, name_width));
    } else {
        const struct summary summary = summarize_samples(&bench);
        bench_check(&bench, write_console_row(&bench, entry->name, name_width, &summary));
        if (json != nullptr) {
            bench_check(&bench, write_json_row(&bench, entry->name, &summary, json));
        }
    }
    free(bench.context_switches);
    free(bench.faults);
    free(bench.wall_nanoseconds);
    return bench.status;
}

static int select_case(const regex_t* filter, const char* name, bool* out_is_selected) {
    const int match = filter != nullptr ? regexec(filter, name, 0, nullptr, 0) : 0;
    if (match != 0 && match != REG_NOMATCH) {
        return EINVAL;
    }
    *out_is_selected = match == 0;
    return 0;
}

static constexpr double DEFAULT_MIN_SECONDS = 0.1;
static constexpr size_t DEFAULT_REPETITIONS = 5;
static constexpr size_t DEFAULT_FIXED_REGION_BYTES = (size_t)256 * 1024 * 1024;

int bench_main(int argc, char** argv, const struct bench_case* cases, size_t case_count) {
    struct options options = {
        .min_seconds = DEFAULT_MIN_SECONDS,
        .iterations = 1,
        .repetitions = DEFAULT_REPETITIONS,
        .fixed_bytes = DEFAULT_FIXED_REGION_BYTES,
        .granule_bytes = BENCH_POOL_GRANULE_BYTES
    };
    if (parse_options(argc, argv, &options) != 0) {
        return write_usage(stderr) == 0 ? 2 : 1;
    }
    if (options.help) {
        return write_usage(stdout) == 0 ? 0 : 1;
    }
    regex_t filter;
    if (options.filter != nullptr && regcomp(&filter, options.filter, REG_EXTENDED | REG_NOSUB) != 0) {
        return write_usage(stderr) == 0 ? 2 : 1;
    }
    const regex_t* selection = options.filter != nullptr ? &filter : nullptr;
    bool has_match = selection == nullptr;
    int status = 0;
    for (size_t index = 0; status == 0 && !has_match && index < case_count; ++index) {
        status = select_case(selection, cases[index].name, &has_match);
    }
    if (status == 0 && !has_match) {
        regfree(&filter);
        return fprintf(stderr, "benchmark: no case matches --filter %s\n", options.filter) < 0 ? 1 : 2;
    }
    FILE* json = nullptr;
    void* region = nullptr;
    const int name_width = measure_name_width(cases, case_count);
    if (status == 0 && options.mode != BENCH_MODE_LIST && options.fixed) {
        status = init_region(&region, options.fixed_bytes, &options.granule_bytes);
    }
    if (status == 0 && options.mode != BENCH_MODE_LIST && options.mode != BENCH_MODE_SMOKE) {
        struct context context = {};
        if (options.json != nullptr) {
            json = fopen(options.json, "w");
            status = json == nullptr ? errno : 0;
        }
        if (status == 0) {
            status = warm_up();
        }
        if (status == 0) {
            status = init_context(&context);
        }
        if (status == 0) {
            status = write_console_context(&context, argv[0], &options);
        }
        if (status == 0) {
            status = write_console_header(name_width);
        }
        if (status == 0 && json != nullptr) {
            status = write_context(&context, argv[0], &options, json);
        }
    }
    bool first = true;
    bool failed = false;
    for (size_t index = 0; status == 0 && index < case_count; ++index) {
        bool is_selected = false;
        status = select_case(selection, cases[index].name, &is_selected);
        if (status != 0 || !is_selected) {
            continue;
        }
        if (options.mode == BENCH_MODE_LIST) {
            if (puts(cases[index].name) < 0) {
                status = EIO;
            }
        } else {
            if (json != nullptr && !first && fputs(",\n", json) < 0) {
                status = EIO;
                break;
            }
            if (run_case(&cases[index], &options, name_width, json, &region) != 0) {
                failed = true;
            }
            if (options.fixed && region == nullptr) {
                status = EBUSY;
            }
            first = false;
        }
    }
    if (json != nullptr) {
        if (fputs("\n]}\n", json) < 0 && status == 0) {
            status = EIO;
        }
        if (fclose(json) != 0 && status == 0) {
            status = EIO;
        }
    }
    free(region);
    if (options.filter != nullptr) {
        regfree(&filter);
    }
    if (fflush(stdout) != 0 && status == 0) {
        status = EIO;
    }
    if (status != 0 && fprintf(stderr, "benchmark: %s\n", strerror(status)) < 0) {
        return 1;
    }
    return status != 0 || failed ? 1 : 0;
}
