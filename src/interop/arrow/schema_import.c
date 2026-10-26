// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "interop/arrow/arrow_internal.h"
#include "schema/schema_internal.h"
#include "status/status_internal.h"
#include "types/array_count.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct arrow_format_info {
    union dora_type_parameters parameters;
    uint32_t type_id_count;
    enum dora_type type_kind;
    enum dora_layout layout_kind;
};

const struct arrow_simple_format dora_arrow_simple_formats[ARROW_SIMPLE_FORMAT_COUNT] = {
    {DORA_BYTES_LITERAL("n"), DORA_TYPE_NULL, DORA_LAYOUT_NULL, 0},
    {DORA_BYTES_LITERAL("b"), DORA_TYPE_BOOLEAN, DORA_LAYOUT_BITS, 0},
    {DORA_BYTES_LITERAL("c"), DORA_TYPE_INT8, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("C"), DORA_TYPE_UINT8, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("s"), DORA_TYPE_INT16, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("S"), DORA_TYPE_UINT16, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("i"), DORA_TYPE_INT32, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("I"), DORA_TYPE_UINT32, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("l"), DORA_TYPE_INT64, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("L"), DORA_TYPE_UINT64, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("e"), DORA_TYPE_FLOAT16, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("f"), DORA_TYPE_FLOAT32, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("g"), DORA_TYPE_FLOAT64, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("z"), DORA_TYPE_BINARY, DORA_LAYOUT_BINARY, 0},
    {DORA_BYTES_LITERAL("Z"), DORA_TYPE_BINARY, DORA_LAYOUT_LARGE_BINARY, 0},
    {DORA_BYTES_LITERAL("vz"), DORA_TYPE_BINARY, DORA_LAYOUT_BINARY_VIEW, 0},
    {DORA_BYTES_LITERAL("u"), DORA_TYPE_UTF8, DORA_LAYOUT_BINARY, 0},
    {DORA_BYTES_LITERAL("U"), DORA_TYPE_UTF8, DORA_LAYOUT_LARGE_BINARY, 0},
    {DORA_BYTES_LITERAL("vu"), DORA_TYPE_UTF8, DORA_LAYOUT_BINARY_VIEW, 0},
    {DORA_BYTES_LITERAL("tdD"), DORA_TYPE_DATE32, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("tdm"), DORA_TYPE_DATE64, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("tts"), DORA_TYPE_TIME, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_SECOND},
    {DORA_BYTES_LITERAL("ttm"), DORA_TYPE_TIME, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_MILLISECOND},
    {DORA_BYTES_LITERAL("ttu"), DORA_TYPE_TIME, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_MICROSECOND},
    {DORA_BYTES_LITERAL("ttn"), DORA_TYPE_TIME, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_NANOSECOND},
    {DORA_BYTES_LITERAL("tDs"), DORA_TYPE_DURATION, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_SECOND},
    {DORA_BYTES_LITERAL("tDm"), DORA_TYPE_DURATION, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_MILLISECOND},
    {DORA_BYTES_LITERAL("tDu"), DORA_TYPE_DURATION, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_MICROSECOND},
    {DORA_BYTES_LITERAL("tDn"), DORA_TYPE_DURATION, DORA_LAYOUT_FIXED, DORA_TIME_UNIT_NANOSECOND},
    {DORA_BYTES_LITERAL("tiM"), DORA_TYPE_INTERVAL_YEAR_MONTH, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("tiD"), DORA_TYPE_INTERVAL_DAY_TIME, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("tin"), DORA_TYPE_INTERVAL_MONTH_DAY_NANO, DORA_LAYOUT_FIXED, 0},
    {DORA_BYTES_LITERAL("+l"), DORA_TYPE_LIST, DORA_LAYOUT_LIST, 0},
    {DORA_BYTES_LITERAL("+L"), DORA_TYPE_LIST, DORA_LAYOUT_LARGE_LIST, 0},
    {DORA_BYTES_LITERAL("+vl"), DORA_TYPE_LIST, DORA_LAYOUT_LIST_VIEW, 0},
    {DORA_BYTES_LITERAL("+vL"), DORA_TYPE_LIST, DORA_LAYOUT_LARGE_LIST_VIEW, 0},
    {DORA_BYTES_LITERAL("+s"), DORA_TYPE_STRUCT, DORA_LAYOUT_STRUCT, 0},
    {DORA_BYTES_LITERAL("+m"), DORA_TYPE_MAP, DORA_LAYOUT_LIST, 0}
};

struct integer_bounds {
    int64_t minimum;
    int64_t maximum;
};

static bool parse_integer(const char** cursor, struct integer_bounds range, int64_t* out_value) {
    const char* position = *cursor;
    const bool negative = *position == '-';
    if (negative) {
        ++position;
    }
    if (*position < '0' || *position > '9') {
        return false;
    }
    int64_t value = 0;
    while (*position >= '0' && *position <= '9') {
        const int64_t digit = *position++ - '0';
        if (ckd_mul(&value, value, 10) || ckd_add(&value, value, negative ? -digit : digit)) {
            return false;
        }
    }
    if (value < range.minimum || value > range.maximum) {
        return false;
    }
    *cursor = position;
    *out_value = value;
    return true;
}

//------------------------------------------------------------------------------
//
// Format Parsing
//
//------------------------------------------------------------------------------

static int parse_format(const char* format, int8_t* out_type_ids, struct arrow_format_info* out_info) {
    if (format == nullptr || out_info == nullptr) {
        return EINVAL;
    }
    *out_info = (struct arrow_format_info){};
    for (uint32_t index = 0; index < ARRAY_COUNT(dora_arrow_simple_formats); ++index) {
        const struct arrow_simple_format* const entry = &dora_arrow_simple_formats[index];
        if (strcmp(format, (const char*)entry->format.data) == 0) {
            *out_info = (struct arrow_format_info){.type_kind = entry->type_kind, .layout_kind = entry->layout_kind};
            if (entry->type_kind == DORA_TYPE_TIME || entry->type_kind == DORA_TYPE_DURATION) {
                out_info->parameters.unit = (enum dora_time_unit)entry->parameter;
            }
            return DORA_OK;
        }
    }
    if (strncmp(format, "d:", 2) == 0) {
        const char* cursor = format + 2;
        int64_t precision = 0;
        int64_t scale = 0;
        int64_t width = 128;
        if (!parse_integer(&cursor,
                           (struct integer_bounds){.minimum = 1, .maximum = DECIMAL256_MAX_PRECISION},
                           &precision) ||
            *cursor != ',') {
            return EINVAL;
        }
        ++cursor;
        if (!parse_integer(&cursor, (struct integer_bounds){.minimum = INT32_MIN, .maximum = INT32_MAX}, &scale)) {
            return EINVAL;
        }
        if (*cursor == ',') {
            ++cursor;
            if (!parse_integer(&cursor, (struct integer_bounds){.maximum = 256}, &width)) {
                return EINVAL;
            }
        }
        if (*cursor != '\0' || (width != 32 && width != 64 && width != 128 && width != 256) ||
            precision > (width == 32    ? DECIMAL32_MAX_PRECISION
                         : width == 64  ? DECIMAL64_MAX_PRECISION
                         : width == 128 ? DECIMAL128_MAX_PRECISION
                                        : DECIMAL256_MAX_PRECISION)) {
            return EINVAL;
        }
        *out_info = (struct arrow_format_info){
            .type_kind = DORA_TYPE_DECIMAL,
            .layout_kind = DORA_LAYOUT_FIXED,
            .parameters
                .decimal = {.precision = (int32_t)precision, .scale = (int32_t)scale, .bit_width = (uint32_t)width}
        };
        return DORA_OK;
    }
    if (strncmp(format, "w:", 2) == 0 || strncmp(format, "+w:", 3) == 0) {
        const bool list = format[0] == '+';
        const char* cursor = format + (list ? 3 : 2);
        int64_t width = 0;
        if (!parse_integer(&cursor, (struct integer_bounds){.maximum = INT32_MAX}, &width) || *cursor != '\0') {
            return EINVAL;
        }
        out_info->type_kind = list ? DORA_TYPE_FIXED_LIST : DORA_TYPE_FIXED_BINARY;
        out_info->layout_kind = list ? DORA_LAYOUT_FIXED_LIST : DORA_LAYOUT_FIXED;
        if (list) {
            out_info->parameters.list_element_count = (uint32_t)width;
        } else {
            out_info->parameters.byte_width = (uint32_t)width;
        }
        return DORA_OK;
    }
    if (strncmp(format, "ts", 2) == 0 && format[2] != '\0' && format[3] == ':') {
        const char* unit = strchr("smun", format[2]);
        if (unit == nullptr) {
            return EINVAL;
        }
        *out_info = (struct arrow_format_info){
            .type_kind = DORA_TYPE_TIMESTAMP,
            .layout_kind = DORA_LAYOUT_FIXED,
            .parameters.timestamp = {.unit = (enum dora_time_unit)(unit - "smun"), .timezone = format + 4}
        };
        return DORA_OK;
    }
    if (strncmp(format, "+ud:", 4) == 0 || strncmp(format, "+us:", 4) == 0) {
        const char* cursor = format + 4;
        uint32_t count = 0;
        bool seen[UNION_MEMBER_CAPACITY] = {};
        while (*cursor != '\0') {
            int64_t type_id = 0;
            if (count == UNION_MEMBER_CAPACITY ||
                !parse_integer(&cursor, (struct integer_bounds){.maximum = INT8_MAX}, &type_id) || seen[type_id]) {
                return EINVAL;
            }
            seen[type_id] = true;
            if (out_type_ids != nullptr) {
                out_type_ids[count] = (int8_t)type_id;
            }
            ++count;
            if (*cursor == ',') {
                if (*++cursor == '\0') {
                    return EINVAL;
                }
            } else if (*cursor != '\0') {
                return EINVAL;
            }
        }
        *out_info = (struct arrow_format_info){
            .type_kind = DORA_TYPE_UNION,
            .layout_kind = format[2] == 'd' ? DORA_LAYOUT_DENSE_UNION : DORA_LAYOUT_SPARSE_UNION,
            .parameters.type_codes = out_type_ids,
            .type_id_count = count
        };
        return DORA_OK;
    }
    if (strcmp(format, "+r") == 0) {
        out_info->layout_kind = DORA_LAYOUT_RUN_END;
        return DORA_OK;
    }
    return ENOTSUP;
}

//------------------------------------------------------------------------------
//
// Metadata Admission
//
//------------------------------------------------------------------------------

static int metadata_admit(const char* metadata,
                          struct dora_metadata_entry* out_entries,
                          uint32_t* out_count,
                          struct dora_metadata_entry* out_extension,
                          bool* out_has_extension,
                          uint64_t work_limit,
                          uint64_t output_limit,
                          uint64_t* out_work,
                          uint64_t* out_output) {
    if (out_count == nullptr || out_extension == nullptr || out_has_extension == nullptr || out_work == nullptr ||
        out_output == nullptr) {
        return EINVAL;
    }
    *out_work = 0;
    *out_output = 0;
    *out_count = 0;
    *out_has_extension = false;
    *out_extension = (struct dora_metadata_entry){};
    if (metadata == nullptr) {
        return DORA_OK;
    }
    const uint8_t* cursor = (const uint8_t*)metadata;
    if (work_limit < sizeof(int32_t)) {
        return ERANGE;
    }
    int32_t count = 0;
    memcpy(&count, cursor, sizeof(count));
    if (count < 0) {
        return EINVAL;
    }
    cursor += sizeof(count);
    *out_work = sizeof(count);
    bool has_metadata = false;
    for (int32_t index = 0; index < count; ++index) {
        struct dora_bytes pair[2] = {};
        for (uint32_t part = 0; part < 2; ++part) {
            if (ckd_add(out_work, *out_work, sizeof(int32_t))) {
                return EOVERFLOW;
            }
            if (*out_work > work_limit) {
                return ERANGE;
            }
            int32_t length = 0;
            memcpy(&length, cursor, sizeof(length));
            if (length < 0) {
                return EINVAL;
            }
            cursor += sizeof(length);
            if (ckd_add(out_work, *out_work, (uint32_t)length) || ckd_add(out_output, *out_output, (uint32_t)length)) {
                return EOVERFLOW;
            }
            if (*out_work > work_limit || *out_output > output_limit) {
                return ERANGE;
            }
            pair[part] = (struct dora_bytes){.data = cursor, .byte_count = length};
            cursor += length;
        }
        if (bytes_are_equal(pair[0], (struct dora_bytes)DORA_BYTES_LITERAL("ARROW:extension:name"))) {
            if (*out_has_extension || pair[1].byte_count == 0) {
                return EINVAL;
            }
            *out_has_extension = true;
            out_extension->key = pair[1];
        } else if (bytes_are_equal(pair[0], (struct dora_bytes)DORA_BYTES_LITERAL("ARROW:extension:metadata"))) {
            if (has_metadata) {
                return EINVAL;
            }
            has_metadata = true;
            out_extension->value = pair[1];
        } else {
            if (out_entries != nullptr) {
                out_entries[*out_count] = (struct dora_metadata_entry){.key = pair[0], .value = pair[1]};
            }
            ++*out_count;
        }
    }
    return has_metadata && !*out_has_extension ? EINVAL : DORA_OK;
}

bool dora_arrow_is_extension_metadata_key(struct dora_bytes key) {
    const struct dora_bytes name = DORA_BYTES_LITERAL("ARROW:extension:name");
    const struct dora_bytes metadata = DORA_BYTES_LITERAL("ARROW:extension:metadata");
    return bytes_are_equal(key, name) || bytes_are_equal(key, metadata);
}

struct schema_import_node {
    struct dora_schema_field field;
    struct dora_allocation metadata;
    int8_t type_codes[UNION_MEMBER_CAPACITY];
    uint32_t union_owner_node_index;
};

struct schema_import_context {
    struct dora_pool pool;
    struct dora_arrow_limits limits;
    struct arrow_traversal_state* traversal;
    struct dora_allocation nodes_storage;
    struct schema_import_node* nodes;
    uint32_t node_count;
    uint32_t node_capacity;
    bool should_emit;
};

static int schema_text_charge(struct schema_import_context* context, const char* text) {
    if (text == nullptr) {
        return DORA_OK;
    }
    const uint64_t remaining = context->limits.max_coordinate_work_count - context->traversal->coordinate_work_count;
    const size_t bound = remaining > SIZE_MAX ? SIZE_MAX : (size_t)remaining;
    size_t length = 0;
    while (length < bound && text[length] != '\0') {
        ++length;
    }
    const uint64_t work = length == bound ? bound : (uint64_t)length + 1;
    const int status = arrow_traversal_charge(&context->limits, context->traversal, work, 0);
    return status != DORA_OK || length != bound ? status
                                                : arrow_traversal_charge(&context->limits, context->traversal, 1, 0);
}

static int schema_enter(struct schema_import_context* context, const struct ArrowSchema* source, uint32_t depth) {
    if (source == nullptr || source->release == nullptr || source->format == nullptr || source->n_children < 0 ||
        (uint64_t)source->n_children > UINT32_MAX || (source->n_children != 0 && source->children == nullptr) ||
        ((uint64_t)source->flags & ~UINT64_C(7)) != 0) {
        return EINVAL;
    }
    if (depth >= context->limits.max_depth || context->traversal->node_count >= context->limits.max_node_count) {
        return ERANGE;
    }
    const int stack_status = dora_arrow_traversal_push(context->traversal, source, depth);
    if (stack_status != DORA_OK) {
        return stack_status;
    }
    ++context->traversal->node_count;
    uint32_t edges = (uint32_t)source->n_children;
    if ((source->dictionary != nullptr && ckd_add(&edges, edges, 1)) ||
        ckd_add(&context->traversal->edge_count, context->traversal->edge_count, edges)) {
        return EOVERFLOW;
    }
    if (context->traversal->edge_count > context->limits.max_edge_count) {
        return ERANGE;
    }
    if (context->should_emit) {
        return DORA_OK;
    }
    int status = schema_text_charge(context, source->format);
    if (status == DORA_OK) {
        status = schema_text_charge(context, source->name);
    }
    return status;
}

static int schema_measure(struct schema_import_context* context, const struct ArrowSchema* source, uint32_t depth) {
    int status = schema_enter(context, source, depth);
    uint32_t count = 0;
    struct dora_metadata_entry extension = {};
    bool has_extension = false;
    uint64_t work = 0;
    uint64_t output = 0;
    if (status == DORA_OK) {
        status = metadata_admit(source->metadata,
                                nullptr,
                                &count,
                                &extension,
                                &has_extension,
                                context->limits.max_coordinate_work_count - context->traversal->coordinate_work_count,
                                UINT64_MAX,
                                &work,
                                &output);
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context->limits, context->traversal, work, 0);
    }
    if (status == DORA_OK) {
        ++context->node_count;
    }
    for (uint32_t index = 0; status == DORA_OK && index < (uint32_t)source->n_children; ++index) {
        status = schema_measure(context, source->children[index], depth + 1);
    }
    if (status == DORA_OK && source->dictionary != nullptr) {
        status = schema_measure(context, source->dictionary, depth + 1);
    }
    return status;
}

int dora_arrow_field_charge(const struct dora_arrow_limits* limits,
                            struct arrow_traversal_state* state,
                            const struct dora_schema_field* field) {
    size_t bytes = 0;
    const size_t name = field->name == nullptr ? 0 : strlen(field->name);
    if (ckd_add(&bytes, name, 1) || ckd_add(&bytes, bytes, field->type.extension_name.byte_count) ||
        ckd_add(&bytes, bytes, field->type.extension_metadata.byte_count)) {
        return EOVERFLOW;
    }
    if (field->type.id == DORA_TYPE_TIMESTAMP && field->type.parameters.timestamp.timezone != nullptr &&
        (ckd_add(&bytes, bytes, strlen(field->type.parameters.timestamp.timezone)) || ckd_add(&bytes, bytes, 1))) {
        return EOVERFLOW;
    }
    if (field->type.id == DORA_TYPE_UNION && ckd_add(&bytes, bytes, UNION_MEMBER_CAPACITY)) {
        return EOVERFLOW;
    }
    for (uint32_t index = 0; index < field->metadata_entry_count; ++index) {
        if (ckd_add(&bytes, bytes, field->metadata[index].key.byte_count) ||
            ckd_add(&bytes, bytes, field->metadata[index].value.byte_count)) {
            return EOVERFLOW;
        }
    }
    return arrow_traversal_charge(limits, state, 0, bytes);
}

static void opaque_integer(struct dora_schema_field* field) {
    if (field->type.id != DORA_TYPE_FIXED_BINARY ||
        !bytes_are_equal(field->type.extension_name, (struct dora_bytes)DORA_BYTES_LITERAL("arrow.opaque"))) {
        return;
    }
    enum dora_type type = DORA_TYPE_NULL;
    const struct dora_bytes metadata = field->type.extension_metadata;
    if (field->type.parameters.byte_width == 16 &&
        bytes_are_equal(metadata,
                        (struct dora_bytes)DORA_BYTES_LITERAL("{\"type_name\":\"int128\",\"vendor_name\":\"dora\"}"))) {
        type = DORA_TYPE_INT128;
    }
    if (field->type.parameters.byte_width == 16 &&
        bytes_are_equal(
            metadata,
            (struct dora_bytes)DORA_BYTES_LITERAL("{\"type_name\":\"uint128\",\"vendor_name\":\"dora\"}"))) {
        type = DORA_TYPE_UINT128;
    }
    if (field->type.parameters.byte_width == 32 &&
        bytes_are_equal(metadata,
                        (struct dora_bytes)DORA_BYTES_LITERAL("{\"type_name\":\"int256\",\"vendor_name\":\"dora\"}"))) {
        type = DORA_TYPE_INT256;
    }
    if (type != DORA_TYPE_NULL) {
        field->type = (struct dora_logical_type){.id = type};
    }
}

static int schema_import(struct schema_import_context* context,
                         const struct ArrowSchema* source,
                         uint32_t depth,
                         uint32_t* out_node) {
    int status = schema_enter(context, source, depth);
    if (status != DORA_OK) {
        return status;
    }
    struct schema_import_node node = {};
    struct arrow_format_info format = {};
    status = parse_format(source->format, node.type_codes, &format);
    if (status != DORA_OK) {
        return status;
    }
    const bool dictionary = source->dictionary != nullptr;
    const bool run_end = format.layout_kind == DORA_LAYOUT_RUN_END;
    const bool map = !dictionary && format.type_kind == DORA_TYPE_MAP;
    if ((dictionary &&
         (source->n_children != 0 || format.type_kind < DORA_TYPE_INT8 || format.type_kind > DORA_TYPE_UINT64 ||
          format.type_kind == DORA_TYPE_INT128 || format.type_kind == DORA_TYPE_INT256)) ||
        (run_end && source->n_children != 2)) {
        return EINVAL;
    }
    if (map && source->n_children != 1) {
        return EINVAL;
    }
    uint64_t work = 0;
    uint64_t output = 0;
    struct dora_metadata_entry extension = {};
    bool has_extension = false;
    const uint64_t work_left = context->limits.max_coordinate_work_count - context->traversal->coordinate_work_count;
    const uint64_t output_left = UINT64_MAX;
    status = metadata_admit(source->metadata,
                            nullptr,
                            &node.field.metadata_entry_count,
                            &extension,
                            &has_extension,
                            work_left,
                            output_left,
                            &work,
                            &output);
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context->limits, context->traversal, work, 0);
    }
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, node.field.metadata_entry_count, sizeof(struct dora_metadata_entry))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context->limits, context->traversal, 0, bytes);
    }
    if (status == DORA_OK) {
        status = dora_pool_allocate(context->pool, bytes, &node.metadata);
    }
    if (status == DORA_OK && node.field.metadata_entry_count != 0) {
        status = metadata_admit(source->metadata,
                                node.metadata.address,
                                &node.field.metadata_entry_count,
                                &extension,
                                &has_extension,
                                context->limits.max_coordinate_work_count - context->traversal->coordinate_work_count,
                                output_left,
                                &work,
                                &output);
        if (status == DORA_OK) {
            status = arrow_traversal_charge(&context->limits, context->traversal, work, 0);
        }
    }
    node.field.name = source->name;
    node.field.metadata = node.metadata.address;
    node.field.type = (struct dora_logical_type){
        .id = format.type_kind,
        .parameters = format.parameters,
        .extension_name = extension.key,
        .extension_metadata = extension.value
    };
    node.field.layout = dictionary ? DORA_LAYOUT_DICTIONARY : format.layout_kind;
    node.field.index_type = dictionary ? format.type_kind : DORA_TYPE_NULL;
    node.field.flags =
        ((uint64_t)source->flags & (uint64_t)ARROW_FLAG_NULLABLE) == 0 ? DORA_SCHEMA_FIELD_FLAG_NONNULLABLE : 0;
    if (((uint64_t)source->flags & (uint64_t)ARROW_FLAG_DICTIONARY_ORDERED) != 0) {
        node.field.flags |= DORA_SCHEMA_FIELD_FLAG_DICTIONARY_ORDERED;
    }
    if (((uint64_t)source->flags & (uint64_t)ARROW_FLAG_MAP_KEYS_SORTED) != 0) {
        node.field.flags |= DORA_SCHEMA_FIELD_FLAG_MAP_KEYS_SORTED;
    }
    node.field.child_count = dictionary ? 1 : (uint32_t)source->n_children;
    if (status == DORA_OK && !dictionary && !run_end && format.type_kind == DORA_TYPE_UNION &&
        format.type_id_count != node.field.child_count) {
        status = EINVAL;
    }
    const uint32_t index = context->node_count;
    node.union_owner_node_index = index;
    if (status == DORA_OK && index >= context->node_capacity) {
        status = EINVAL;
    }
    if (status != DORA_OK) {
        dora_allocation_release(&node.metadata);
        return status;
    }
    context->nodes[context->node_count++] = node;
    uint32_t value_index = UINT32_MAX;
    uint32_t end_index = UINT32_MAX;
    for (uint32_t child = 0; status == DORA_OK && child < node.field.child_count; ++child) {
        uint32_t current = 0;
        status = schema_import(context, dictionary ? source->dictionary : source->children[child], depth + 1, &current);
        if (dictionary || child == 1) {
            value_index = current;
        }
        if (child == 0) {
            end_index = current;
        }
    }
    struct schema_import_node* const nodes = context->nodes;
    if (status == DORA_OK && map) {
        const struct dora_schema_field entries = nodes[end_index].field;
        if (entries.type.id != DORA_TYPE_STRUCT || entries.layout != DORA_LAYOUT_STRUCT || entries.child_count != 2 ||
            (entries.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0 ||
            (nodes[end_index + 1].field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0) {
            return EINVAL;
        }
    }
    if (status == DORA_OK && (dictionary || run_end)) {
        if (run_end &&
            (nodes[end_index].field.type.id < DORA_TYPE_INT16 || nodes[end_index].field.type.id > DORA_TYPE_INT64 ||
             nodes[end_index].field.layout != DORA_LAYOUT_FIXED ||
             (nodes[end_index].field.flags & DORA_SCHEMA_FIELD_FLAG_NONNULLABLE) == 0)) {
            return EINVAL;
        }
        nodes[index].field.type = nodes[value_index].field.type;
        nodes[index].union_owner_node_index = nodes[value_index].union_owner_node_index;
        if (run_end) {
            nodes[index].field.index_type = nodes[end_index].field.type.id;
        }
        if (has_extension) {
            uint32_t current = value_index;
            for (;;) {
                nodes[current].field.type.extension_name = extension.key;
                nodes[current].field.type.extension_metadata = extension.value;
                if (nodes[current].field.layout != DORA_LAYOUT_DICTIONARY &&
                    nodes[current].field.layout != DORA_LAYOUT_RUN_END) {
                    break;
                }
                ++current;
                if (nodes[current - 1].field.layout == DORA_LAYOUT_RUN_END) {
                    ++current;
                }
            }
            opaque_integer(&nodes[current].field);
            const struct dora_logical_type type = nodes[current].field.type;
            current = value_index;
            for (;;) {
                nodes[current].field.type = type;
                if (nodes[current].field.layout != DORA_LAYOUT_DICTIONARY &&
                    nodes[current].field.layout != DORA_LAYOUT_RUN_END) {
                    break;
                }
                ++current;
                if (nodes[current - 1].field.layout == DORA_LAYOUT_RUN_END) {
                    ++current;
                }
            }
            nodes[index].field.type = type;
        }
    } else if (status == DORA_OK) {
        opaque_integer(&nodes[index].field);
    }
    if (status == DORA_OK) {
        *out_node = index;
    }
    return status;
}

//------------------------------------------------------------------------------
//
// Schema Transfer
//
//------------------------------------------------------------------------------

int dora_arrow_import_schema_impl(struct dora_pool pool,
                                  const struct ArrowSchema* source,
                                  const struct dora_arrow_limits* limits,
                                  struct arrow_traversal_state* state,
                                  struct dora_schema* out_schema,
                                  struct dora_error* out_error) {
    if (!schema_is_empty(out_schema)) {
        return EINVAL;
    }
    struct schema_import_node inline_nodes[8];
    struct schema_import_context context = {.pool = pool, .limits = *limits, .traversal = state, .nodes = inline_nodes};
    dora_arrow_traversal_rewind(state);
    int status = schema_measure(&context, source, 0);
    context.node_capacity = context.node_count;
    context.node_count = 0;
    context.should_emit = true;
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, context.node_capacity, sizeof(struct schema_import_node))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(limits, state, 0, bytes);
    }
    if (status == DORA_OK && context.node_capacity > 8) {
        status = dora_pool_allocate(pool, bytes, &context.nodes_storage);
        context.nodes = context.nodes_storage.address;
    }
    dora_arrow_traversal_rewind(state);
    uint32_t root = 0;
    if (status == DORA_OK) {
        status = schema_import(&context, source, 0, &root);
    }
    if (status == DORA_OK && context.node_count != context.node_capacity) {
        status = EINVAL;
    }
    struct dora_schema_field inline_fields[8];
    struct dora_allocation field_storage = {};
    if (status == DORA_OK && ckd_mul(&bytes, context.node_count, sizeof(struct dora_schema_field))) {
        status = EOVERFLOW;
    }
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&context.limits, context.traversal, 0, bytes);
    }
    if (status == DORA_OK && context.node_count > 8) {
        status = dora_pool_allocate(pool, bytes, &field_storage);
    }
    struct schema_import_node* const nodes = context.nodes;
    struct dora_schema_field* const fields = context.node_count > 8 ? field_storage.address : inline_fields;
    for (uint32_t index = 0; status == DORA_OK && index < context.node_count; ++index) {
        fields[index] = nodes[index].field;
        if (fields[index].type.id == DORA_TYPE_UNION) {
            fields[index].type.parameters.type_codes = nodes[nodes[index].union_owner_node_index].type_codes;
        }
    }
    for (uint32_t index = 0; status == DORA_OK && index < context.node_count; ++index) {
        status = dora_arrow_field_charge(limits, state, &fields[index]);
    }
    struct dora_schema record = {};
    if (status == DORA_OK) {
        status = dora_arrow_field_charge(limits, state, &(const struct dora_schema_field){});
    }
    if (status == DORA_OK) {
        const struct dora_schema_input input = {.fields = fields, .field_count = context.node_count};
        status = context.node_count == 1 ? dora_schema_init_retain_node(&record, pool, fields, nullptr)
                                         : dora_schema_init_copy(&record, pool, &input);
    }
    struct dora_schema child = {};
    if (status == DORA_OK && context.node_count != 1) {
        status = dora_schema_get_child(&record, 0, &child);
    }
    if (status == DORA_OK) {
        status =
            context.node_count == 1 ? dora_schema_move(&record, out_schema) : dora_schema_retain(&child, out_schema);
    }
    dora_schema_release(&record);
    for (uint32_t index = 0; index < context.node_count; ++index) {
        dora_allocation_release(&nodes[index].metadata);
    }
    dora_allocation_release(&field_storage);
    dora_allocation_release(&context.nodes_storage);
    if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
        error_message_copy_bounded(out_error, "Invalid Arrow schema or exceeded import limit");
    }
    return status;
}

int dora_arrow_import_schema(struct dora_pool pool,
                             const struct ArrowSchema* source,
                             const struct dora_arrow_limits* limits,
                             struct dora_schema* out_schema,
                             struct dora_error* out_error) {
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    if (!schema_is_empty(out_schema)) {
        return EINVAL;
    }
    struct dora_schema candidate = {};
    const struct dora_arrow_limits resolved = dora_arrow_resolve_limits(limits);
    struct arrow_traversal_state state = {};
    dora_arrow_traversal_init(pool, &resolved, &state);
    int status = dora_arrow_import_schema_impl(pool, source, &resolved, &state, &candidate, out_error);
    dora_allocation_release(&state.stack_storage);
    if (status == DORA_OK) {
        status = dora_schema_move(&candidate, out_schema);
    }
    dora_schema_release(&candidate);
    return status;
}
