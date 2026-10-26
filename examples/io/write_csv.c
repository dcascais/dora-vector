// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/io/csv.h"
#include "dora/vector/io/file.h"
#include "dora/vector/memory.h"
#include "dora/vector/status.h"
#include "dora/vector/stream.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type i32 = {.id = DORA_TYPE_INT32};
    const struct dora_logical_type utf8 = {.id = DORA_TYPE_UTF8};
    const int32_t ids[] = {101, 102, 103};
    const struct dora_bytes notes[] = {DORA_BYTES_LITERAL("ready"), DORA_BYTES_LITERAL("fragile, glass"), {}};
    const bool valid[] = {true, true, false};
    const char* const names[] = {"id", "note"};
    struct dora_pool pool = {};
    struct dora_vector columns[2] = {};
    struct dora_vector batch = {};
    struct dora_byte_sink_file output = {};
    struct dora_writer writer = {};
    struct dora_stream_result flow = {};
    struct dora_error error = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_i32(&columns[0], pool, &i32, ids, nullptr, 3);
    }
    if (status == 0) {
        status = dora_vector_init_copy_bytes(&columns[1],
                                             pool,
                                             &utf8,
                                             DORA_LAYOUT_BINARY,
                                             notes,
                                             valid,
                                             3,
                                             DORA_FAILURE_POLICY_ERROR);
    }
    if (status == 0) {
        status = dora_vector_init_retain_struct(&batch, columns, names, 2, nullptr, 3);
    }
    if (status == 0) {
        status = dora_byte_sink_file_init(&output, stdout);
    }
    if (status == 0) {
        status = dora_csv_writer_init(
            &writer,
            pool,
            &output.sink,
            &(const struct dora_csv_options){.has_header = true, .null_token = DORA_BYTES_LITERAL("NULL")},
            &error);
    }
    if (status == 0) {
        status = dora_writer_push(&writer, &batch, &flow, &error);
    }
    if (status == 0 && flow.status != DORA_STREAM_STATUS_READY) {
        status = EAGAIN;
    }
    if (status == 0) {
        status = dora_writer_finish(&writer, &flow, &error);
    }
    if (status == 0 && flow.status != DORA_STREAM_STATUS_END) {
        status = EAGAIN;
    }
    dora_writer_release(&writer);
    if (output.sink.release != nullptr) {
        output.sink.release(&output.sink);
    }
    dora_vector_release(&batch);
    dora_vector_release(&columns[1]);
    dora_vector_release(&columns[0]);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "write_csv failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
