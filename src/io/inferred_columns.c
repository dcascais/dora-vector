// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "io/inferred_columns.h"

#include "builder/builder_internal.h"
#include "dora/vector/builder.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "memory/memory_internal.h"
#include "numeric/arithmetic.h"
#include "numeric/limbs.h"
#include "temporal/temporal.h"
#include "types/text.h"
#include "utf8/ascii.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stdint.h>
#include <string.h>

union text_scalar {
    bool boolean;
    int32_t i32;
    int64_t i64;
    struct dora_i128 i128;
    struct dora_i256 i256;
    double f64;
};

static int text_csv_bool(struct dora_bytes value, bool strict, bool* out_value) {
    bool result = false;
    if (value.byte_count == 1) {
        if (utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("t"), true) ||
            (!strict && (utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("y"), true) ||
                         utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("1"), true)))) {
            result = true;
        } else if (!utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("f"), true) &&
                   (strict || (!utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("n"), true) &&
                               !utf8_bytes_are_equal(value, (struct dora_bytes)DORA_BYTES_LITERAL("0"), true)))) {
            return EINVAL;
        }
    } else {
        static constexpr char words[][8] = {"no", "yes", "true", "false"};
        if (value.byte_count < 2 || value.byte_count > 5) {
            return EINVAL;
        }
        const struct dora_bytes word = {(const uint8_t*)words[value.byte_count - 2], value.byte_count};
        if (!utf8_bytes_are_equal(value, word, true)) {
            return EINVAL;
        }
        result = value.byte_count == 3 || value.byte_count == 4;
    }
    *out_value = result;
    return DORA_OK;
}

static const struct dora_logical_type* text_type(enum text_infer_kind kind) {
    static const struct dora_logical_type types[] = {
        [TEXT_INFER_NULL] = {.id = DORA_TYPE_NULL},
        [TEXT_INFER_BOOL] = {.id = DORA_TYPE_BOOLEAN},
        [TEXT_INFER_I64] = {.id = DORA_TYPE_INT64},
        [TEXT_INFER_I128] = {.id = DORA_TYPE_INT128},
        [TEXT_INFER_I256] = {.id = DORA_TYPE_INT256},
        [TEXT_INFER_F64] = {.id = DORA_TYPE_FLOAT64},
        [TEXT_INFER_DATE] = {.id = DORA_TYPE_DATE32},
        [TEXT_INFER_TIME] = {.id = DORA_TYPE_TIME, .parameters.unit = DORA_TIME_UNIT_NANOSECOND},
        [TEXT_INFER_TIMESTAMP] =
            {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_NANOSECOND}},
        [TEXT_INFER_TIMESTAMP_UTC] =
            {.id = DORA_TYPE_TIMESTAMP, .parameters.timestamp = {.unit = DORA_TIME_UNIT_NANOSECOND, .timezone = "UTC"}},
        [TEXT_INFER_UTF8] = {.id = DORA_TYPE_UTF8},
        [TEXT_INFER_JSON] = {.id = DORA_TYPE_UTF8, .extension_name = DORA_BYTES_LITERAL("arrow.json")}
    };
    return &types[kind];
}

static constexpr uint8_t text_radices[UINT8_MAX + 1] = {['x'] = 16, ['X'] = 16, ['b'] = 2, ['B'] = 2};

static bool text_is_integer_space(uint8_t value) {
    static constexpr bool spaces[UINT8_MAX + 1] =
        {[' '] = true, ['\t'] = true, ['\r'] = true, ['\n'] = true, ['\v'] = true, ['\f'] = true};
    return spaces[value];
}

static int text_parse_temporal(enum text_infer_kind kind,
                               struct dora_bytes bytes,
                               union text_scalar* out_value,
                               bool* out_has_offset) {
    const struct dora_logical_type* const type = text_type(kind);
    const enum dora_physical_type physical = kind == TEXT_INFER_DATE ? DORA_PHYSICAL_TYPE_I32 : DORA_PHYSICAL_TYPE_I64;
    const int status = dora_temporal_parse((const char*)bytes.data,
                                           bytes.byte_count,
                                           type->id,
                                           physical,
                                           kind == TEXT_INFER_DATE ? DORA_TIME_UNIT_SECOND : DORA_TIME_UNIT_NANOSECOND,
                                           kind == TEXT_INFER_TIMESTAMP_UTC,
                                           DORA_ROUNDING_MODE_EXACT,
                                           TEMPORAL_PARSE_READER,
                                           out_has_offset,
                                           out_value);
    return status == EOVERFLOW ? ERANGE : status;
}

static int
text_parse(enum text_infer_kind kind, struct dora_bytes bytes, union text_scalar* out_value, bool* out_has_offset) {
    if (kind >= TEXT_INFER_DATE) {
        return text_parse_temporal(kind, bytes, out_value, out_has_offset);
    }

    static const enum dora_physical_type physicals[] = {
        [TEXT_INFER_I64 - TEXT_INFER_I64] = DORA_PHYSICAL_TYPE_I64,
        [TEXT_INFER_I128 - TEXT_INFER_I64] = DORA_PHYSICAL_TYPE_I128,
        [TEXT_INFER_I256 - TEXT_INFER_I64] = DORA_PHYSICAL_TYPE_I256,
        [TEXT_INFER_F64 - TEXT_INFER_I64] = DORA_PHYSICAL_TYPE_F64
    };
    const struct dora_logical_type* const type = text_type(kind);
    const enum dora_physical_type physical = physicals[kind - TEXT_INFER_I64];
    if (kind <= TEXT_INFER_F64) {
        size_t position = 0;
        while (position < bytes.byte_count && text_is_integer_space(bytes.data[position])) {
            ++position;
        }
        if (kind <= TEXT_INFER_I256 && bytes.byte_count - position > 2 && bytes.data[position] == '0' &&
            text_radices[bytes.data[position + 1]] != 0) {
            static constexpr uint8_t digits[UINT8_MAX + 1] = {
                ['0'] = 1,  ['1'] = 2,  ['2'] = 3,  ['3'] = 4,  ['4'] = 5,  ['5'] = 6,  ['6'] = 7,  ['7'] = 8,
                ['8'] = 9,  ['9'] = 10, ['a'] = 11, ['b'] = 12, ['c'] = 13, ['d'] = 14, ['e'] = 15, ['f'] = 16,
                ['A'] = 11, ['B'] = 12, ['C'] = 13, ['D'] = 14, ['E'] = 15, ['F'] = 16
            };
            const uint32_t radix = text_radices[bytes.data[position + 1]];
            struct dora_i256 value = {};
            position += 2;
            for (; position < bytes.byte_count; ++position) {
                const uint32_t digit = (uint32_t)digits[bytes.data[position]] - 1U;
                if (digit >= radix) {
                    return EINVAL;
                }
                if (position + 1 < bytes.byte_count && bytes.data[position + 1] == '_') {
                    ++position;
                    if (position + 1 == bytes.byte_count) {
                        return EINVAL;
                    }
                }
                const int status = dora_limbs_mul_u32((struct limbs_span){.values = value.limbs, .count = 4},
                                                      (struct limbs_affine_u32){.factor = radix, .addend = digit});
                if (status != DORA_OK) {
                    return ERANGE;
                }
            }
            return value.limbs[3] > INT64_MAX ? ERANGE : dora_numeric_store_integer(value, physical, out_value);
        }
        if (position != 0) {
            bytes.data += position;
            bytes.byte_count -= position;
        }
        while (bytes.byte_count != 0 && text_is_integer_space(bytes.data[bytes.byte_count - 1])) {
            --bytes.byte_count;
        }
    }
    if (kind == TEXT_INFER_F64) {
        const int status = dora_type_parse_double((const char*)bytes.data, bytes.byte_count, &out_value->f64);
        return status == ERANGE ? DORA_OK : status;
    }
    const int status = kind == TEXT_INFER_I64 ? dora_type_parse_i64(type,
                                                                    physical,
                                                                    (const char*)bytes.data,
                                                                    bytes.byte_count,
                                                                    DORA_ROUNDING_MODE_EXACT,
                                                                    nullptr,
                                                                    out_value)
                                              : dora_type_parse_exact(type,
                                                                      physical,
                                                                      (const char*)bytes.data,
                                                                      bytes.byte_count,
                                                                      DORA_ROUNDING_MODE_EXACT,
                                                                      nullptr,
                                                                      out_value);
    return status == EOVERFLOW ? ERANGE : status == EILSEQ ? EINVAL : status;
}

static size_t text_digits_end(struct dora_bytes value, size_t position) {
    while (position < value.byte_count && value.data[position] >= '0' && value.data[position] <= '9') {
        ++position;
    }
    return position;
}

static enum text_infer_kind text_number_kind(struct dora_bytes value, bool* out_bignum) {
    *out_bignum = true;
    size_t position = 0;
    while (position < value.byte_count && text_is_integer_space(value.data[position])) {
        ++position;
    }
    if (position == value.byte_count || value.data[position] == '+') {
        return TEXT_INFER_UTF8;
    }
    if (value.byte_count - position > 1 && value.data[position] == '0') {
        const uint8_t next = value.data[position + 1];
        if (text_radices[next] != 0) {
            *out_bignum = false;
            return TEXT_INFER_I64;
        }
        if (next >= '0' && next <= '9') {
            return TEXT_INFER_UTF8;
        }
    }
    if (value.data[position] == '-') {
        ++position;
        *out_bignum = value.byte_count - position < 2 || value.data[position] != '0' ||
                      value.data[position + 1] < '0' || value.data[position + 1] > '9';
    }
    const size_t digits = position;
    position = text_digits_end(value, position);
    if (position < value.byte_count &&
        (value.data[position] == '.' ||
         (position != digits && (value.data[position] == 'e' || value.data[position] == 'E')))) {
        return text_is_integer_space(value.data[value.byte_count - 1]) ? TEXT_INFER_UTF8 : TEXT_INFER_F64;
    }
    if (position == digits) {
        return TEXT_INFER_UTF8;
    }
    while (position < value.byte_count && text_is_integer_space(value.data[position])) {
        ++position;
    }
    return position == value.byte_count ? TEXT_INFER_I64 : TEXT_INFER_UTF8;
}

static enum text_infer_kind text_integer_kind(struct dora_bytes value, enum text_infer_kind minimum) {
    union text_scalar scalar;
    if (minimum <= TEXT_INFER_I64 && text_parse(TEXT_INFER_I64, value, &scalar, nullptr) == DORA_OK) {
        return TEXT_INFER_I64;
    }
    if (text_parse(TEXT_INFER_I256, value, &scalar, nullptr) != DORA_OK) {
        return TEXT_INFER_NULL;
    }
    if (minimum == TEXT_INFER_I256) {
        return minimum;
    }
    const uint64_t extension = scalar.i256.limbs[1] > INT64_MAX ? UINT64_MAX : 0;
    if (scalar.i256.limbs[2] != extension || scalar.i256.limbs[3] != extension) {
        return TEXT_INFER_I256;
    }
    if (minimum == TEXT_INFER_I128) {
        return minimum;
    }
    const uint64_t narrow_extension = scalar.i256.limbs[0] > INT64_MAX ? UINT64_MAX : 0;
    return scalar.i256.limbs[1] == narrow_extension ? TEXT_INFER_I64 : TEXT_INFER_I128;
}

static enum text_infer_kind
text_string_kind(struct dora_bytes value, enum text_infer_kind current, bool* out_has_offset) {
    union text_scalar scalar;
    if ((current == TEXT_INFER_NULL || current == TEXT_INFER_TIME) &&
        text_parse_temporal(TEXT_INFER_TIME, value, &scalar, nullptr) == DORA_OK) {
        return TEXT_INFER_TIME;
    }
    if (current == TEXT_INFER_TIME) {
        return TEXT_INFER_UTF8;
    }
    if ((current == TEXT_INFER_NULL || current == TEXT_INFER_DATE) &&
        text_parse_temporal(TEXT_INFER_DATE, value, &scalar, nullptr) == DORA_OK) {
        return TEXT_INFER_DATE;
    }
    if (text_parse_temporal(TEXT_INFER_TIMESTAMP_UTC, value, &scalar, out_has_offset) == DORA_OK) {
        return *out_has_offset ? TEXT_INFER_TIMESTAMP_UTC : TEXT_INFER_TIMESTAMP;
    }
    return TEXT_INFER_UTF8;
}

int dora_text_infer_csv(struct dora_bytes value, struct text_infer_column* column) {
    const int status = dora_utf8_validate_span(value, nullptr);
    if (status != DORA_OK || column->kind == TEXT_INFER_UTF8) {
        return status;
    }
    union text_scalar scalar;
    enum text_infer_kind kind = column->kind;
    if (kind == TEXT_INFER_NULL || kind == TEXT_INFER_BOOL) {
        if (text_csv_bool(value, true, &scalar.boolean) == DORA_OK) {
            column->kind = TEXT_INFER_BOOL;
            return DORA_OK;
        }
        if (kind == TEXT_INFER_BOOL) {
            column->kind = TEXT_INFER_UTF8;
            return DORA_OK;
        }
    }
    if (kind <= TEXT_INFER_F64) {
        bool bignum = false;
        const enum text_infer_kind number = text_number_kind(value, &bignum);
        if (number == TEXT_INFER_I64 && kind != TEXT_INFER_F64) {
            kind = text_integer_kind(value, kind);
            if (!bignum && (kind == TEXT_INFER_NULL || kind == TEXT_INFER_I256)) {
                kind = TEXT_INFER_F64;
            }
            if (kind == TEXT_INFER_NULL) {
                kind = TEXT_INFER_UTF8;
            }
        } else if (number != TEXT_INFER_UTF8) {
            kind = TEXT_INFER_F64;
        } else if (kind != TEXT_INFER_NULL) {
            kind = TEXT_INFER_UTF8;
        }
    }
    if (kind == TEXT_INFER_F64 && (text_is_integer_space(value.data[value.byte_count - 1]) ||
                                   text_parse(TEXT_INFER_F64, value, &scalar, nullptr) != DORA_OK)) {
        kind = TEXT_INFER_UTF8;
    }
    if (kind == TEXT_INFER_NULL || (kind >= TEXT_INFER_DATE && kind <= TEXT_INFER_TIMESTAMP_UTC)) {
        bool has_offset = false;
        const enum text_infer_kind candidate = text_string_kind(value, kind, &has_offset);
        kind = kind == TEXT_INFER_TIMESTAMP_UTC && candidate == TEXT_INFER_TIMESTAMP ? kind
               : (kind == TEXT_INFER_DATE || kind == TEXT_INFER_TIME || kind == TEXT_INFER_TIMESTAMP_UTC) &&
                       candidate != kind
                   ? TEXT_INFER_UTF8
                   : candidate;
    }
    column->kind = kind;
    return DORA_OK;
}

static bool text_integer_fits(struct dora_bytes digits, struct dora_bytes limit) {
    if (digits.byte_count != limit.byte_count) {
        return digits.byte_count < limit.byte_count;
    }
    for (size_t position = 0; position < digits.byte_count; ++position) {
        if (digits.data[position] != limit.data[position]) {
            return digits.data[position] < limit.data[position];
        }
    }
    return true;
}

void dora_text_infer_json(const struct text_infer_value* value, struct text_infer_column* column) {
    if (value->is_null || column->kind == TEXT_INFER_JSON) {
        return;
    }
    enum text_infer_kind kind = column->kind;
    if (value->kind == TEXT_VALUE_BOOL && (kind == TEXT_INFER_NULL || kind == TEXT_INFER_BOOL)) {
        kind = TEXT_INFER_BOOL;
    } else if (value->kind == TEXT_VALUE_STRING) {
        if (kind == TEXT_INFER_NULL || (kind >= TEXT_INFER_DATE && kind <= TEXT_INFER_TIMESTAMP_UTC)) {
            bool has_offset = false;
            const enum text_infer_kind candidate = text_string_kind(value->bytes, kind, &has_offset);
            kind = kind >= TEXT_INFER_DATE && kind != TEXT_INFER_TIME && candidate >= TEXT_INFER_TIMESTAMP &&
                           candidate <= TEXT_INFER_TIMESTAMP_UTC && column->has_offset != has_offset
                       ? TEXT_INFER_UTF8
                       : candidate;
            column->has_offset = has_offset;
        } else if (kind != TEXT_INFER_UTF8) {
            kind = TEXT_INFER_JSON;
        }
    } else if (value->kind == TEXT_VALUE_NUMBER && kind <= TEXT_INFER_F64 && kind != TEXT_INFER_BOOL) {
        if (kind != TEXT_INFER_F64) {
            enum text_infer_kind number = TEXT_INFER_F64;
            const bool negative = value->bytes.byte_count != 0 && value->bytes.data[0] == '-';
            const size_t sign = negative ? 1 : 0;
            if (text_digits_end(value->bytes, sign) == value->bytes.byte_count) {
                const struct dora_bytes digits = {value->bytes.data + sign, value->bytes.byte_count - sign};
                if (text_integer_fits(digits,
                                      negative ? (struct dora_bytes)DORA_BYTES_LITERAL("9223372036854775808")
                                               : (struct dora_bytes)DORA_BYTES_LITERAL("9223372036854775807"))) {
                    number = TEXT_INFER_I64;
                } else if (!negative &&
                           text_integer_fits(digits, (struct dora_bytes)DORA_BYTES_LITERAL("18446744073709551615"))) {
                    number = TEXT_INFER_I128;
                }
            }
            if (number == TEXT_INFER_I64 && kind == TEXT_INFER_I64 && column->is_last_integer_negative != negative) {
                number = TEXT_INFER_I128;
            }
            column->is_last_integer_negative = negative;
            kind = kind > number ? kind : number;
        }
    } else {
        kind = TEXT_INFER_JSON;
    }
    column->kind = kind;
}

static int text_name_append(struct text_columns* columns,
                            struct dora_pool pool,
                            struct dora_bytes name,
                            struct text_name* out_name) {
    if (name.byte_count != 0 && name.data == nullptr) {
        return EINVAL;
    }
    size_t required = 0;
    if (ckd_add(&required, columns->name_byte_count, name.byte_count) || ckd_add(&required, required, (size_t)1)) {
        return EOVERFLOW;
    }
    const uintptr_t address = (uintptr_t)name.data;
    const uintptr_t base = (uintptr_t)columns->name_storage.address;
    const bool internal =
        columns->name_storage.address != nullptr && address >= base && address - base < columns->name_byte_count;
    const size_t offset = internal ? (size_t)(address - base) : 0;
    if (internal && name.byte_count > columns->name_byte_count - offset) {
        return EINVAL;
    }
    const int status = dora_memory_payload_reserve(pool, required, columns->name_byte_count, &columns->name_storage);
    if (status != DORA_OK) {
        return status;
    }
    uint8_t* const bytes = columns->name_storage.address;
    if (name.byte_count != 0) {
        size_t written = 0;
        const int moved =
            utf8_move_bytes((struct dora_bytes){internal ? bytes + offset : name.data, name.byte_count},
                            (struct dora_mutable_bytes){bytes + columns->name_byte_count, name.byte_count},
                            &written);
        if (moved != 0) {
            return moved;
        }
    }
    bytes[required - 1] = 0;
    *out_name = (struct text_name){columns->name_byte_count, name.byte_count};
    columns->name_byte_count = required;
    return DORA_OK;
}

int dora_text_columns_add(struct text_columns* columns, struct dora_pool pool, struct dora_bytes name) {
    if (columns->column_count >= UINT32_MAX - 1) {
        return EOVERFLOW;
    }
    size_t bytes = 0;
    size_t used = 0;
    if (ckd_mul(&bytes, (size_t)columns->column_count + 1, sizeof(struct text_infer_column)) ||
        ckd_mul(&used, (size_t)columns->column_count, sizeof(struct text_infer_column))) {
        return EOVERFLOW;
    }
    int status = dora_memory_payload_reserve(pool, bytes, used, &columns->column_storage);
    struct text_name copied = {};
    if (status == DORA_OK) {
        status = text_name_append(columns, pool, name, &copied);
    }
    if (status == DORA_OK) {
        struct text_infer_column* const fields = columns->column_storage.address;
        fields[columns->column_count++] = (struct text_infer_column){.original = copied};
    }
    return status;
}

void dora_text_columns_release(struct text_columns* columns) {
    dora_allocation_release(&columns->column_storage);
    dora_allocation_release(&columns->name_storage);
    *columns = (struct text_columns){};
}

static struct dora_bytes text_name_bytes(const struct text_columns* columns, struct text_name name) {
    return (struct dora_bytes){(const uint8_t*)columns->name_storage.address + name.byte_offset, name.byte_count};
}

struct dora_bytes dora_text_column_name(const struct text_columns* columns, uint32_t index) {
    const struct text_infer_column* const fields = columns->column_storage.address;
    return text_name_bytes(columns, fields[index].original);
}

uint32_t dora_text_column_find(const struct text_columns* columns, struct dora_bytes name) {
    for (uint32_t index = 0; index < columns->column_count; ++index) {
        const struct dora_bytes current = dora_text_column_name(columns, index);
        if (utf8_bytes_are_equal(current, name, false)) {
            return index;
        }
    }
    return UINT32_MAX;
}

void dora_text_columns_unsee(struct text_columns* columns) {
    struct text_infer_column* const fields = columns->column_storage.address;
    for (uint32_t index = 0; index < columns->column_count; ++index) {
        fields[index].was_seen = false;
    }
}

static bool text_is_blank(struct dora_bytes value) {
    for (size_t index = 0; index < value.byte_count; ++index) {
        if (!dora_utf8_ascii_is_text_space((char)value.data[index])) {
            return false;
        }
    }
    return true;
}

static int text_trim_name(struct dora_bytes value, struct dora_bytes* out_trimmed) {
    struct dora_range range = {};
    int status = dora_utf8_trim_range((struct dora_utf8_input){value, DORA_UTF8_CHECKED}, nullptr, &range);
    if (status == 0) {
        *out_trimmed = range.length == 0 ? (struct dora_bytes){}
                                         : (struct dora_bytes){value.data + range.offset, (size_t)range.length};
    }
    return status;
}

static int text_public_names(struct text_columns* columns, struct dora_pool pool, bool json, bool header) {
    for (uint32_t index = 0; index < columns->column_count; ++index) {
        struct text_infer_column* const fields = columns->column_storage.address;
        struct dora_bytes name = dora_text_column_name(columns, index);
        char generated[32] = "column";
        if (!json && (!header || text_is_blank(name))) {
            const size_t width = dora_type_unsigned_length((uint64_t)columns->column_count - 1);
            const size_t digits = dora_type_unsigned_length(index);
            memset(generated + 6, '0', width - digits);
            dora_type_format_unsigned(index, generated + 6 + width);
            name = (struct dora_bytes){(const uint8_t*)generated, 6 + width};
        }
        if (!json) {
            const int status = text_trim_name(name, &name);
            if (status != DORA_OK) {
                return status;
            }
        }
        struct text_name current = {};
        int status = text_name_append(columns, pool, name, &current);
        if (status != DORA_OK) {
            return status;
        }
        while (true) {
            uint32_t previous = 0;
            for (; previous < index; ++previous) {
                if (utf8_bytes_are_equal(text_name_bytes(columns, current),
                                         text_name_bytes(columns, fields[previous].published),
                                         true)) {
                    break;
                }
            }
            if (previous == index) {
                break;
            }
            if (ckd_add(&fields[previous].collision_count, fields[previous].collision_count, UINT64_C(1))) {
                return EOVERFLOW;
            }
            const uint64_t count = fields[previous].collision_count;
            const size_t digits = dora_type_unsigned_length(count);
            size_t suffix_size = 0;
            size_t required = 0;
            if (ckd_add(&suffix_size, current.byte_count, digits) || ckd_add(&suffix_size, suffix_size, (size_t)1) ||
                ckd_add(&required, columns->name_byte_count, suffix_size) || ckd_add(&required, required, (size_t)1)) {
                return EOVERFLOW;
            }
            status = dora_memory_payload_reserve(pool, required, columns->name_byte_count, &columns->name_storage);
            if (status != DORA_OK) {
                return status;
            }
            uint8_t* const bytes = columns->name_storage.address;
            const size_t start = columns->name_byte_count;
            size_t written = 0;
            status = utf8_move_bytes((struct dora_bytes){bytes + current.byte_offset, current.byte_count},
                                     (struct dora_mutable_bytes){bytes + start, current.byte_count},
                                     &written);
            if (status != 0) {
                return status;
            }
            bytes[start + current.byte_count] = '_';
            dora_type_format_unsigned(count, (char*)bytes + start + suffix_size);
            bytes[required - 1] = 0;
            current = (struct text_name){start, suffix_size};
            columns->name_byte_count = required;
        }
        fields[index].published = current;
    }
    return DORA_OK;
}

int dora_text_columns_schema(struct text_columns* columns,
                             struct dora_pool pool,
                             bool json,
                             bool header,
                             struct dora_schema* out_schema) {
    int status = text_public_names(columns, pool, json, header);
    size_t bytes = 0;
    if (status == DORA_OK && ckd_mul(&bytes, (size_t)columns->column_count, sizeof(struct dora_schema_field))) {
        status = EOVERFLOW;
    }
    struct dora_allocation declarations = {};
    if (status == DORA_OK && columns->column_count != 0) {
        status = dora_memory_payload_allocate(pool, bytes, &declarations);
    }
    struct dora_schema_field* const fields = declarations.address;
    struct text_infer_column* const inferred = columns->column_storage.address;
    for (uint32_t index = 0; status == DORA_OK && index < columns->column_count; ++index) {
        if (!json && inferred[index].kind == TEXT_INFER_NULL) {
            inferred[index].kind = TEXT_INFER_UTF8;
        }
        fields[index] = (struct dora_schema_field){
            .name = (const char*)columns->name_storage.address + inferred[index].published.byte_offset,
            .type = *text_type(inferred[index].kind)
        };
    }
    if (status == DORA_OK) {
        status =
            dora_schema_init_copy(out_schema,
                                  pool,
                                  &(struct dora_schema_input){.fields = fields, .field_count = columns->column_count});
    }
    dora_allocation_release(&declarations);
    return status;
}

static bool text_kind_accepts(enum text_infer_kind kind, enum text_value_kind token) {
    if (token == TEXT_VALUE_CSV || kind == TEXT_INFER_JSON) {
        return true;
    }
    if (kind == TEXT_INFER_BOOL) {
        return token == TEXT_VALUE_BOOL;
    }
    if (kind >= TEXT_INFER_I64 && kind <= TEXT_INFER_F64) {
        return token == TEXT_VALUE_NUMBER;
    }
    return kind != TEXT_INFER_NULL && token == TEXT_VALUE_STRING;
}

int dora_text_inferred_append(struct dora_builder* builder,
                              enum text_infer_kind kind,
                              struct text_infer_value value,
                              enum dora_failure_policy policy) {
    if (value.is_null) {
        return dora_builder_append_null(builder);
    }
    if (value.kind == TEXT_VALUE_CSV && kind == TEXT_INFER_UTF8) {
        bool ascii = false;
        const int valid = policy == DORA_FAILURE_POLICY_TRUST ? DORA_OK : dora_utf8_validate_span(value.bytes, &ascii);
        if (valid != DORA_OK) {
            return valid == EILSEQ && policy == DORA_FAILURE_POLICY_NULL ? dora_builder_append_null(builder) : valid;
        }
        value.proof = policy == DORA_FAILURE_POLICY_TRUST ? DORA_UTF8_PROMISED
                      : ascii                             ? DORA_UTF8_VALIDATED_ASCII
                                                          : DORA_UTF8_VALIDATED;
    }
    int status = text_kind_accepts(kind, value.kind) && kind != TEXT_INFER_NULL ? DORA_OK : EINVAL;
    union text_scalar scalar = {};
    if (status == DORA_OK && kind == TEXT_INFER_BOOL) {
        status = text_csv_bool(value.bytes, value.kind != TEXT_VALUE_CSV, &scalar.boolean);
    } else if (status == DORA_OK && kind >= TEXT_INFER_I64 && kind <= TEXT_INFER_TIMESTAMP_UTC) {
        status = text_parse(kind, value.bytes, &scalar, nullptr);
    }
    if (status != DORA_OK) {
        return (status == EINVAL || status == ERANGE) && policy == DORA_FAILURE_POLICY_NULL
                   ? dora_builder_append_null(builder)
                   : status;
    }
    switch (kind) {
    case TEXT_INFER_NULL:
        return dora_builder_append_null(builder);
    case TEXT_INFER_BOOL:
        return dora_builder_append_bool(builder, scalar.boolean);
    case TEXT_INFER_I64:
    case TEXT_INFER_TIME:
    case TEXT_INFER_TIMESTAMP:
    case TEXT_INFER_TIMESTAMP_UTC:
        return dora_builder_append_i64(builder, scalar.i64);
    case TEXT_INFER_I128:
        return dora_builder_append_i128(builder, scalar.i128);
    case TEXT_INFER_I256:
        return dora_builder_append_i256(builder, scalar.i256);
    case TEXT_INFER_F64:
        return dora_builder_append_f64(builder, scalar.f64);
    case TEXT_INFER_DATE:
        return dora_builder_append_i32(builder, scalar.i32);
    case TEXT_INFER_UTF8:
        return dora_builder_append_reader_text(builder,
                                               (const char*)value.bytes.data,
                                               value.bytes.byte_count,
                                               value.proof);
    case TEXT_INFER_JSON:
        return dora_builder_append_reader_text(builder,
                                               (const char*)value.raw.data,
                                               value.raw.byte_count,
                                               value.raw_proof);
    }
    return EINVAL;
}
