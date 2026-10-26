// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type f64 = {.id = DORA_TYPE_FLOAT64};
    const double distances[] = {12.0, 30.0, 0.0, 15.0};
    const bool valid[] = {true, true, false, true};
    const double durations[] = {0.5, 2.0, 1.0, 0.5};
    struct dora_pool pool = {};
    struct dora_vector km = {};
    struct dora_vector hours = {};
    struct dora_vector speed = {};
    struct dora_vector sum = {};
    struct dora_vector count = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_f64(&km, pool, &f64, distances, valid, 4);
    }
    if (status == 0) {
        status = dora_vector_init_copy_f64(&hours, pool, &f64, durations, nullptr, 4);
    }
    if (status == 0) {
        status = dora_compute_divide(&km, &hours, DORA_FAILURE_POLICY_ERROR, &speed);
    }
    if (status == 0) {
        status = dora_compute_sum(&speed, &sum);
    }
    if (status == 0) {
        status = dora_compute_count(&speed, &count);
    }
    double total = 0.0;
    int64_t observed = 0;
    bool is_null = false;
    if (status == 0) {
        status = dora_vector_read_i64(&count, 0, &observed, &is_null);
    }
    if (status == 0 && observed > 0) {
        status = dora_vector_read_f64(&sum, 0, &total, &is_null);
        if (status == 0 && is_null) {
            status = EINVAL;
        }
    }
    if (status == 0) {
        const int printed = observed == 0
                                ? printf("mean_km_per_hour=null\n")
                                : printf("observed=%" PRId64 " mean_km_per_hour=%.2f\n", observed, total / observed);
        if (printed < 0) {
            status = EIO;
        }
    }
    dora_vector_release(&count);
    dora_vector_release(&sum);
    dora_vector_release(&speed);
    dora_vector_release(&hours);
    dora_vector_release(&km);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "derive_column failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
