// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Text column inference.

#ifndef DORA_IO_INFERRED_COLUMNS_H
#define DORA_IO_INFERRED_COLUMNS_H

#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/types.h"
#include "utf8/input.h"

#include <stddef.h>
#include <stdint.h>

enum text_infer_kind : uint8_t {
    TEXT_INFER_NULL,
    TEXT_INFER_BOOL,
    TEXT_INFER_I64,
    TEXT_INFER_I128,
    TEXT_INFER_I256,
    TEXT_INFER_F64,
    TEXT_INFER_DATE,
    TEXT_INFER_TIME,
    TEXT_INFER_TIMESTAMP,
    TEXT_INFER_TIMESTAMP_UTC,
    TEXT_INFER_UTF8,
    TEXT_INFER_JSON
};

enum text_value_kind : uint8_t {
    TEXT_VALUE_CSV,
    TEXT_VALUE_NULL,
    TEXT_VALUE_BOOL,
    TEXT_VALUE_NUMBER,
    TEXT_VALUE_STRING,
    TEXT_VALUE_OPAQUE
};

struct text_infer_value {
    struct dora_bytes bytes;
    struct dora_bytes raw;
    enum text_value_kind kind;
    bool is_null;
    enum dora_utf8_proof proof;
    enum dora_utf8_proof raw_proof;
};

struct text_name {
    size_t byte_offset;
    size_t byte_count;
};

struct text_infer_column {
    struct text_name original;
    struct text_name published;
    uint64_t collision_count;
    enum text_infer_kind kind;
    bool is_last_integer_negative;
    bool has_offset;
    bool was_seen;
};

struct text_columns {
    struct dora_allocation column_storage;
    struct dora_allocation name_storage;
    size_t name_byte_count;
    uint32_t column_count;
};

int dora_text_columns_add(struct text_columns* columns, struct dora_pool pool, struct dora_bytes name);
void dora_text_columns_release(struct text_columns* columns);
struct dora_bytes dora_text_column_name(const struct text_columns* columns, uint32_t index);
uint32_t dora_text_column_find(const struct text_columns* columns, struct dora_bytes name);
void dora_text_columns_unsee(struct text_columns* columns);
int dora_text_infer_csv(struct dora_bytes value, struct text_infer_column* column);
void dora_text_infer_json(const struct text_infer_value* value, struct text_infer_column* column);
int dora_text_columns_schema(struct text_columns* columns,
                             struct dora_pool pool,
                             bool json,
                             bool header,
                             struct dora_schema* out_schema);
int dora_text_inferred_append(struct dora_builder* builder,
                              enum text_infer_kind kind,
                              struct text_infer_value value,
                              enum dora_failure_policy policy);

#endif
