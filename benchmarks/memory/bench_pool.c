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
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Fixtures
//
//------------------------------------------------------------------------------

enum variant : uint8_t {
    VARIANT_ALLOCATE,
    VARIANT_TOUCH,
    VARIANT_GROW
};

struct inputs {
    struct dora_pool pool;
    size_t granule_bytes;
    size_t byte_count;
    struct dora_vector source;
    const uint8_t* data;
};

static struct inputs prepare(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs =
        {.pool = bench->pool, .granule_bytes = bench->pool_granule_bytes, .byte_count = args->byte_count};
    if (args->variant == VARIANT_GROW) {
        const struct bench_args data_args = {
            .rows = (args->byte_count + sizeof(uint64_t) - 1) / sizeof(uint64_t),
            .type.id = DORA_TYPE_UINT64
        };
        struct dora_vector_view view = {};
        bench_check(bench, bench_random(bench, &data_args, 1, &inputs.source));
        bench_check(bench, dora_vector_get_view(&inputs.source, &view));
        inputs.data = (const uint8_t*)dora_vector_view_get_values_u64(&view);
    }
    return inputs;
}

//------------------------------------------------------------------------------
//
// Checks
//
//------------------------------------------------------------------------------

static int check_allocation(const struct inputs* inputs) {
    struct dora_allocation allocation = {};
    int status = dora_pool_allocate(inputs->pool, inputs->byte_count, &allocation);
    if (status == 0 && (allocation.capacity_bytes < inputs->byte_count ||
                        (uintptr_t)allocation.address % inputs->granule_bytes != 0)) {
        status = EDOM;
    }
    dora_allocation_release(&allocation);
    return status;
}

static constexpr size_t GROW_BYTES = (size_t)16 * 1024 * 1024;

static int check_grow(const struct inputs* inputs, const struct dora_allocation* allocation) {
    if (allocation->capacity_bytes < GROW_BYTES) {
        return EDOM;
    }
    const uint8_t* bytes = allocation->address;
    for (size_t offset = 0; offset < GROW_BYTES; offset += inputs->byte_count) {
        const size_t remaining = GROW_BYTES - offset;
        const size_t count = inputs->byte_count < remaining ? inputs->byte_count : remaining;
        if (memcmp(bytes + offset, inputs->data, count) != 0) {
            return EDOM;
        }
    }
    return 0;
}

static void finish(struct bench* bench, const struct bench_args* args, struct inputs* inputs) {
    bench_set_items(bench, args->variant == VARIANT_GROW ? 0 : 1);
    bench_set_bytes(bench,
                    args->variant == VARIANT_TOUCH  ? args->byte_count
                    : args->variant == VARIANT_GROW ? GROW_BYTES
                                                    : 0);
    dora_vector_release(&inputs->source);
}

//------------------------------------------------------------------------------
//
// Cases
//
//------------------------------------------------------------------------------

//--------------------------------------
// Allocation
//--------------------------------------

static int compute_allocate_release(const struct inputs* inputs) {
    struct dora_allocation allocation = {};
    const int status = dora_pool_allocate(inputs->pool, inputs->byte_count, &allocation);
    dora_allocation_release(&allocation);
    return status;
}

BENCH_DEFINE_STATUS_OPERATION(allocate_release_operation, compute_allocate_release)

static void allocate_release(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        bench_check(bench, check_allocation(&inputs));
    }
    bench_measure(bench, &allocate_release_operation, &inputs);
    finish(bench, args, &inputs);
}

static constexpr size_t TOUCH_STRIDE_BYTES = 64;

static int compute_allocate_touch_release(const struct inputs* inputs, uint64_t* out_slot) {
    struct dora_allocation allocation = {};
    const int status = dora_pool_allocate(inputs->pool, inputs->byte_count, &allocation);
    const uint8_t* bytes = allocation.address;
    uint64_t total = 0;
    for (size_t offset = 0; offset < allocation.capacity_bytes; offset += TOUCH_STRIDE_BYTES) {
        total += bytes[offset];
    }
    *out_slot = total;
    dora_allocation_release(&allocation);
    return status;
}

BENCH_DEFINE_OPERATION(allocate_touch_release_operation, uint64_t, compute_allocate_touch_release, nullptr)

static void allocate_touch_release(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        bench_check(bench, check_allocation(&inputs));
    }
    bench_measure(bench, &allocate_touch_release_operation, &inputs);
    finish(bench, args, &inputs);
}

//--------------------------------------
// Growth
//--------------------------------------

static void release_allocations(void* slots, size_t count) {
    struct dora_allocation* allocations = slots;
    for (size_t slot = 0; slot < count; ++slot) {
        dora_allocation_release(&allocations[slot]);
    }
}

static int compute_grow(const struct inputs* inputs, struct dora_allocation* out_slot) {
    const struct dora_pool pool = inputs->pool;
    const size_t byte_count = inputs->byte_count;
    const uint8_t* data = inputs->data;
    size_t length = 0;
    int status = 0;
    while (status == 0 && length < GROW_BYTES) {
        const size_t remaining = GROW_BYTES - length;
        const size_t count = byte_count < remaining ? byte_count : remaining;
        status = dora_allocation_reserve(pool, length + count, length, out_slot);
        if (status == 0) {
            memcpy((uint8_t*)out_slot->address + length, data, count);
            length += count;
        }
    }
    return status;
}

BENCH_DEFINE_OPERATION(grow_operation, struct dora_allocation, compute_grow, release_allocations)

static void grow(struct bench* bench, const struct bench_args* args) {
    struct inputs inputs = prepare(bench, args);
    if (bench->status == 0) {
        struct dora_allocation allocation = {};
        bench_check(bench, inputs.data == nullptr ? EDOM : compute_grow(&inputs, &allocation));
        bench_check(bench, check_grow(&inputs, &allocation));
        release_allocations(&allocation, 1);
    }
    bench_measure(bench, &grow_operation, &inputs);
    finish(bench, args, &inputs);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
    const struct bench_case cases[] = {
        {"allocate_release/size:4096", allocate_release, {.byte_count = 4096}},
        {"allocate_release/size:65536", allocate_release, {.byte_count = 65'536}},
        {"allocate_release/size:1048576", allocate_release, {.byte_count = 1'048'576}},
        {"allocate_release/size:16777216", allocate_release, {.byte_count = 16'777'216}},
        {"allocate_touch_release/size:4096", allocate_touch_release, {.byte_count = 4096, .variant = VARIANT_TOUCH}},
        {"allocate_touch_release/size:65536", allocate_touch_release, {.byte_count = 65'536, .variant = VARIANT_TOUCH}},
        {"allocate_touch_release/size:1048576",
         allocate_touch_release,
         {.byte_count = 1'048'576, .variant = VARIANT_TOUCH}},
        {"allocate_touch_release/size:16777216",
         allocate_touch_release,
         {.byte_count = 16'777'216, .variant = VARIANT_TOUCH}},
        {"grow/write:8", grow, {.byte_count = 8, .variant = VARIANT_GROW}},
        {"grow/write:700", grow, {.byte_count = 700, .variant = VARIANT_GROW}},
        {"grow/write:1500000", grow, {.byte_count = 1'500'000, .variant = VARIANT_GROW}}
    };
    return bench_main(argc, argv, cases, ARRAY_COUNT(cases));
}
