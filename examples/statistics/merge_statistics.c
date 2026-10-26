// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/statistics.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int read_statistic(const struct dora_vector* statistics, const char* name, int64_t* out_value) {
    struct dora_vector group = {};
    struct dora_vector member = {};
    int status = dora_vector_get_child(statistics, 0, &group);
    if (status == 0) {
        status = dora_vector_find_child(&group, name, &member);
    }
    bool is_null = false;
    if (status == 0) {
        status = dora_vector_read_i64(&member, 0, out_value, &is_null);
    }
    return status != 0 ? status : is_null ? EINVAL : 0;
}

int main() {
    const struct dora_logical_type i64 = {.id = DORA_TYPE_INT64};
    const int64_t values[2][3] = {{18, 0, 24}, {12, 30, 0}};
    const bool valid[2][3] = {{true, false, true}, {true, true, false}};
    constexpr uint32_t bounds = DORA_STATISTICS_KIND_MIN | DORA_STATISTICS_KIND_MAX;
    const struct dora_statistics_request request = {
        .schema_node_index = 0,
        .kinds = bounds | DORA_STATISTICS_KIND_ROW_COUNT | DORA_STATISTICS_KIND_NULL_COUNT
    };
    struct dora_pool pool = {};
    struct dora_vector accumulated = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    for (uint32_t index = 0; status == 0 && index < 2; ++index) {
        struct dora_vector source = {};
        struct dora_vector measured = {};
        struct dora_vector statistics = {};
        struct dora_vector merged = {};
        status = dora_vector_init_copy_i64(&source, pool, &i64, values[index], valid[index], 3);
        if (status == 0) {
            status = dora_statistics_compute(&source, &request, 1, &measured);
        }
        if (status == 0) {
            status = dora_vector_get_statistics(&measured, &statistics);
        }
        if (status == 0) {
            status = index == 0 ? dora_vector_retain(&statistics, &merged)
                                : dora_statistics_merge(&accumulated, &statistics, &merged);
        }
        if (status == 0) {
            dora_vector_release(&accumulated);
            status = dora_vector_move(&merged, &accumulated);
        }
        dora_vector_release(&merged);
        dora_vector_release(&measured);
        dora_vector_release(&source);
    }
    int64_t rows = 0;
    int64_t nulls = 0;
    int64_t minimum = 0;
    int64_t maximum = 0;
    if (status == 0) {
        status = read_statistic(&accumulated, "row_count", &rows);
    }
    if (status == 0) {
        status = read_statistic(&accumulated, "null_count", &nulls);
    }
    if (status == 0) {
        status = read_statistic(&accumulated, "min", &minimum);
    }
    if (status == 0) {
        status = read_statistic(&accumulated, "max", &maximum);
    }
    if (status == 0 &&
        printf("rows=%" PRId64 " nulls=%" PRId64 " min=%" PRId64 " max=%" PRId64 "\n", rows, nulls, minimum, maximum) <
            0) {
        status = EIO;
    }
    dora_vector_release(&accumulated);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "merge_statistics failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
