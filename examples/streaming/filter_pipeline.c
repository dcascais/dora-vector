// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/compute.h"
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

static int drop_nulls(struct dora_vector_transform*,
                      struct dora_pool,
                      const struct dora_vector* input,
                      bool is_end_of_input,
                      struct dora_vector* out_vector,
                      struct dora_stream_result* out_result,
                      struct dora_error*) {
    // End-of-input may accompany batch input.
    if (input == nullptr) {
        if (!is_end_of_input) {
            return EINVAL;
        }
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
        return 0;
    }
    struct dora_vector selection = {};
    int status = dora_compute_select_not_null(input, &selection);
    if (status == 0) {
        status = dora_vector_init_retain_selection(out_vector, input, &selection);
    }
    if (status == 0) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY, .is_input_consumed = true};
    }
    dora_vector_release(&selection);
    return status;
}

static void release_transform(struct dora_vector_transform* transform) {
    *transform = (struct dora_vector_transform){};
}

int main() {
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const int32_t readings[] = {18, 0, 24};
    const bool valid[] = {true, false, true};
    struct dora_vector_transform transform = {.apply = drop_nulls, .release = release_transform};
    struct dora_vector_transform* const transforms[] = {&transform};
    struct dora_pool pool = {};
    struct dora_pipeline pipeline = {};
    struct dora_vector incoming = {};
    struct dora_vector outgoing = {};
    struct dora_error error = {};
    bool supplied = false;
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&incoming, pool, &i32, readings, valid, 3);
    }
    if (status == 0) {
        status = dora_pipeline_init(&pipeline, pool, nullptr, transforms, 1, nullptr);
    }
    while (status == 0) {
        struct dora_pipeline_observation observation = {};
        status = dora_pipeline_step(&pipeline, &observation, &error);
        if (status != 0 || observation.progress == DORA_PIPELINE_PROGRESS_FINISHED) {
            break;
        }
        switch (observation.progress) {
        case DORA_PIPELINE_PROGRESS_NEEDS_INPUT:
            if (supplied) {
                status = EINVAL;
                break;
            }
            // Success moves incoming into pipeline.
            status = dora_pipeline_supply(&pipeline, &incoming);
            if (status == 0) {
                supplied = true;
                status = dora_pipeline_end_input(&pipeline);
            }
            break;
        case DORA_PIPELINE_PROGRESS_HAS_OUTPUT: {
            // Success moves pipeline output into outgoing.
            status = dora_pipeline_take(&pipeline, &outgoing);
            uint64_t rows = 0;
            if (status == 0) {
                status = dora_vector_get_length(&outgoing, &rows);
            }
            for (uint64_t row = 0; status == 0 && row < rows; ++row) {
                int32_t value = 0;
                bool is_null = false;
                status = dora_vector_read_i32(&outgoing, row, &value, &is_null);
                if (status == 0 && is_null) {
                    status = EINVAL;
                }
                if (status == 0 && printf("reading=%" PRId32 "\n", value) < 0) {
                    status = EIO;
                }
            }
            dora_vector_release(&outgoing);
            break;
        }
        case DORA_PIPELINE_PROGRESS_ADVANCED:
            break;
        case DORA_PIPELINE_PROGRESS_BLOCKED:
            status = EAGAIN;
            break;
        default:
            status = EIO;
            break;
        }
    }
    dora_vector_release(&outgoing);
    dora_vector_release(&incoming);
    dora_pipeline_release(&pipeline);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "filter_pipeline failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
