// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

struct readings_source {
    struct dora_vector_source source;
    uint32_t next_batch;
};

static int next_readings(struct dora_vector_source* source,
                         struct dora_pool pool,
                         struct dora_vector* out_batch,
                         struct dora_stream_result* out_result,
                         struct dora_error*) {
    struct readings_source* readings = (struct readings_source*)source;
    if (readings->next_batch == 3) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
        return 0;
    }
    const int32_t values[] = {readings->next_batch * 10, readings->next_batch * 10 + 1};
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const int status = dora_vector_init_copy_i32(out_batch, pool, &i32, values, nullptr, 2);
    if (status == 0) {
        ++readings->next_batch;
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY};
    }
    return status;
}

static void release_readings(struct dora_vector_source* source) {
    *source = (struct dora_vector_source){};
}

int main() {
    struct readings_source source = {.source = {.next = next_readings, .release = release_readings}};
    struct dora_pool pool = {};
    struct dora_reader reader = {};
    struct dora_vector batch = {};
    struct dora_error error = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        // source must outlive reader.
        status = dora_reader_init(&reader, pool, &source.source);
    }
    while (status == 0) {
        struct dora_stream_result flow = {};
        status = dora_reader_next(&reader, &batch, &flow, &error);
        if (status != 0 || flow.status == DORA_STREAM_STATUS_END) {
            break;
        }
        if (flow.status != DORA_STREAM_STATUS_READY) {
            status = EAGAIN;
            break;
        }
        uint64_t rows = 0;
        status = dora_vector_get_length(&batch, &rows);
        for (uint64_t row = 0; status == 0 && row < rows; ++row) {
            int32_t value = 0;
            bool is_null = false;
            status = dora_vector_read_i32(&batch, row, &value, &is_null);
            if (status == 0 && is_null) {
                status = EINVAL;
            }
            if (status == 0 && printf("reading=%" PRId32 "\n", value) < 0) {
                status = EIO;
            }
        }
        dora_vector_release(&batch);
    }
    dora_vector_release(&batch);
    dora_reader_release(&reader);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "batch_reader failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
