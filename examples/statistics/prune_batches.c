// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/statistics.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const int32_t values[2][3] = {{10, 20, 30}, {70, 100, 0}};
    const bool validity[2][3] = {{true, true, true}, {true, true, false}};
    const int32_t minimum = 50;
    constexpr uint32_t bounds = DORA_STATISTICS_KIND_MIN | DORA_STATISTICS_KIND_MAX;
    const struct dora_statistics_request request = {
        .schema_node_index = 0,
        .kinds = bounds | DORA_STATISTICS_KIND_NULL_COUNT
    };
    struct dora_pool pool = {};
    struct dora_vector measured[2] = {};
    struct dora_vector operand = {};
    struct dora_vector threshold = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    for (uint32_t index = 0; status == 0 && index < 2; ++index) {
        struct dora_vector batch = {};
        status = dora_vector_init_copy_i32(&batch, pool, &i32, values[index], validity[index], 3);
        if (status == 0) {
            status = dora_statistics_compute(&batch, &request, 1, &measured[index]);
        }
        dora_vector_release(&batch);
    }
    if (status == 0) {
        status = dora_vector_init_copy_i32(&operand, pool, &i32, &minimum, nullptr, 1);
    }
    if (status == 0) {
        status = dora_vector_init_retain_constant(&threshold, &operand, 3);
    }

    enum dora_pruning_result verdicts[2] = {};

    if (status == 0) {
        status = dora_statistics_prune(measured, 2, 0, DORA_COMPARE_OP_GE, &operand, verdicts);
    }
    for (uint32_t index = 0; status == 0 && index < 2; ++index) {
        if (verdicts[index] == DORA_PRUNING_RESULT_CANNOT_MATCH) {
            if (printf("batch=%" PRIu32 " skipped\n", index) < 0) {
                status = EIO;
            }
            continue;
        }
        // ALL_MATCH excludes null rows.
        struct dora_vector selection = {};
        uint64_t matches = 0;
        status = dora_compute_select(&measured[index], DORA_COMPARE_OP_GE, &threshold, &selection);
        if (status == 0) {
            status = dora_vector_get_length(&selection, &matches);
        }
        if (status == 0 && printf("batch=%" PRIu32 " matches=%" PRIu64 "\n", index, matches) < 0) {
            status = EIO;
        }
        dora_vector_release(&selection);
    }
    dora_vector_release(&threshold);
    dora_vector_release(&operand);
    dora_vector_release(&measured[1]);
    dora_vector_release(&measured[0]);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "prune_batches failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
