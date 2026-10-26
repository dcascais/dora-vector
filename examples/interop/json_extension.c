// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const struct dora_logical_type utf8 = {.id = DORA_TYPE_UTF8};
    const struct dora_bytes documents[] = {DORA_BYTES_LITERAL("{\"priority\":2}"), DORA_BYTES_LITERAL("[1,2,3]")};
    struct dora_pool pool = {};
    struct dora_vector storage = {};
    struct dora_vector documents_column = {};
    struct dora_schema schema = {};
    struct ArrowSchema arrow_schema = {};
    struct ArrowArray arrow_array = {};
    int status = dora_pool_init(&pool, SIZE_MAX, 64);
    if (status == 0) {
        status = dora_vector_init_copy_bytes(&storage,
                                             pool,
                                             &utf8,
                                             DORA_LAYOUT_BINARY,
                                             documents,
                                             nullptr,
                                             2,
                                             DORA_FAILURE_POLICY_ERROR);
    }
    if (status == 0) {
        status = dora_vector_init_retain_extension(&documents_column,
                                                   (struct dora_bytes)DORA_BYTES_LITERAL("arrow.json"),
                                                   (struct dora_bytes){},
                                                   &storage);
    }
    if (status == 0) {
        status = dora_vector_retain_schema(&documents_column, &schema);
    }
    if (status == 0) {
        status = dora_arrow_export_schema(pool, &schema, &arrow_schema);
    }
    if (status == 0) {
        status = dora_arrow_export_array(&documents_column, &arrow_array);
    }
    struct dora_schema_field field = {};
    if (status == 0) {
        status = dora_schema_get_field(&schema, &field);
    }
    if (status == 0 && printf("extension=") < 0) {
        status = EIO;
    }
    if (status == 0) {
        const struct dora_bytes name = field.type.extension_name;
        if ((name.byte_count != 0 && fwrite(name.data, 1, name.byte_count, stdout) != name.byte_count) ||
            printf(" storage=%s rows=%" PRId64 "\n", arrow_schema.format, arrow_array.length) < 0) {
            status = EIO;
        }
    }
    if (arrow_array.release != nullptr) {
        arrow_array.release(&arrow_array);
    }
    if (arrow_schema.release != nullptr) {
        arrow_schema.release(&arrow_schema);
    }
    dora_schema_release(&schema);
    dora_vector_release(&documents_column);
    dora_vector_release(&storage);

    const int pool_status = dora_pool_release(&pool);
    if (status == 0) {
        status = pool_status;
    }
    if (status != 0) {
        if (fprintf(stderr, "json_extension failed: %d\n", status) < 0) {
            return EXIT_FAILURE;
        }
    }
    return status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
