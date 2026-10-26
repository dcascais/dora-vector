// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "temporal/temporal.h"

#include "dora/vector/types.h"
#include "numeric/arithmetic.h"
#include "utf8/ascii.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static constexpr size_t FORMAT_TEMPORAL_INTEGER_DIGIT_CAPACITY = 20;

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Temporal
//--------------------------------------

static uint32_t temporal_month_days(int32_t year, uint32_t month) {
    static const uint8_t days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) {
        return 0;
    }
    return days[month - 1] + (uint32_t)(month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));
}

static int
temporal_parse_digits(const char* bytes, size_t length, size_t* position, size_t count, uint32_t* out_value) {
    if (length - *position < count) {
        return EINVAL;
    }
    uint32_t value = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t digit = (uint32_t)(uint8_t)bytes[*position + i] - '0';
        if (digit > 9) {
            return EINVAL;
        }
        value = value * 10 + digit;
    }
    *position += count;
    *out_value = value;
    return 0;
}

static constexpr uint32_t CIVIL_ERA_SHIFT = 82U;
static constexpr uint32_t CIVIL_EPOCH_OFFSET = 719'468U + 146'097U * CIVIL_ERA_SHIFT;
static constexpr uint32_t CIVIL_YEAR_OFFSET = 400U * CIVIL_ERA_SHIFT;
static constexpr int64_t CIVIL_RATA_DIE_MIN = -12'699'422;
static constexpr int64_t CIVIL_RATA_DIE_MAX = 1'061'042'401;
static constexpr int32_t CIVIL_YEAR_MIN = -32'800;
static constexpr int32_t CIVIL_YEAR_MAX = 2'906'945;

struct temporal_civil_date {
    int32_t year;
    uint32_t month;
    uint32_t day;
};

static int64_t temporal_unit_factor(enum dora_time_unit unit) {
    static const int64_t factors[4] = {1, 1000, 1'000'000, 1'000'000'000};
    return unit <= DORA_TIME_UNIT_NANOSECOND ? factors[unit] : 0;
}

static int64_t temporal_days_fast(struct temporal_civil_date date) {
    const uint32_t jan_feb_adjust = date.month <= 2U ? 1U : 0U;
    const uint32_t shifted_year = ((uint32_t)date.year + CIVIL_YEAR_OFFSET) - jan_feb_adjust;
    const uint32_t month_from_march = jan_feb_adjust != 0 ? date.month + 12U : date.month;
    const uint32_t day_of_month_zero_based = date.day - 1U;
    const uint32_t century = shifted_year / 100U;
    const uint32_t year_days = 1461U * shifted_year / 4U - century + century / 4U;
    const uint32_t month_days = (979U * month_from_march - 2919U) / 32U;
    const uint32_t shifted_rata_die = year_days + month_days + day_of_month_zero_based;
    return (int32_t)(shifted_rata_die - CIVIL_EPOCH_OFFSET);
}

static struct temporal_civil_date temporal_civil_fast(int32_t day_number) {
    const uint32_t shifted_day = (uint32_t)day_number + CIVIL_EPOCH_OFFSET;
    const uint32_t century_numerator = 4U * shifted_day + 3U;
    const uint32_t century = century_numerator / 146'097U;
    const uint32_t day_within_century = century_numerator % 146'097U / 4U;
    const uint32_t year_numerator = 4U * day_within_century + 3U;
    const uint64_t year_product = UINT64_C(2'939'745) * year_numerator;
    const uint32_t year_within_century = (uint32_t)(year_product >> 32U);
    const uint32_t day_within_year = (uint32_t)(year_product & UINT64_C(0xFFFF'FFFF)) / 2'939'745U / 4U;
    const uint32_t year_within_era = 100U * century + year_within_century;
    const uint32_t month_numerator = 2141U * day_within_year + 197'913U;
    const uint32_t month_from_march = month_numerator / 65'536U;
    const uint32_t day_of_month_zero_based = month_numerator % 65'536U / 2141U;
    const uint32_t jan_feb_adjust = day_within_year >= 306U ? 1U : 0U;
    return (struct temporal_civil_date){
        .year = (int32_t)(year_within_era - CIVIL_YEAR_OFFSET) + (int32_t)jan_feb_adjust,
        .month = jan_feb_adjust != 0 ? month_from_march - 12U : month_from_march,
        .day = day_of_month_zero_based + 1U,
    };
}

int dora_temporal_days_from_civil(int32_t year, uint32_t month, uint32_t day, int64_t* out_days) {
    if (out_days == nullptr || day < 1 || day > temporal_month_days(year, month)) {
        return ERANGE;
    }
    if (year > CIVIL_YEAR_MIN && year < CIVIL_YEAR_MAX) {
        *out_days = temporal_days_fast((struct temporal_civil_date){.year = year, .month = month, .day = day});
        return 0;
    }
    int64_t adjusted_year = (int64_t)year - (month <= 2);
    int64_t era = dora_numeric_floor_div(adjusted_year, 400);
    int64_t year_of_era = adjusted_year - era * 400;
    int64_t adjusted_month = month > 2 ? (int64_t)month - 3 : (int64_t)month + 9;
    int64_t day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    int64_t day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    int64_t era_days = 0;
    int64_t result = 0;
    if (ckd_mul(&era_days, era, 146'097) || ckd_add(&result, era_days, day_of_era - 719'468)) {
        return EOVERFLOW;
    }
    *out_days = result;
    return 0;
}

int dora_temporal_civil_from_days(int64_t days, int32_t* out_year, uint32_t* out_month, uint32_t* out_day) {
    if (out_year == nullptr || out_month == nullptr || out_day == nullptr) {
        return EINVAL;
    }
    if (days >= CIVIL_RATA_DIE_MIN && days <= CIVIL_RATA_DIE_MAX) {
        const struct temporal_civil_date date = temporal_civil_fast((int32_t)days);
        *out_year = date.year;
        *out_month = date.month;
        *out_day = date.day;
        return 0;
    }
    int64_t shifted = 0;
    if (ckd_add(&shifted, days, 719'468)) {
        return EOVERFLOW;
    }
    int64_t era = dora_numeric_floor_div(shifted, 146'097);
    int64_t day_of_era = shifted % 146'097;
    if (day_of_era < 0) {
        day_of_era += 146'097;
    }
    int64_t year_of_era = (day_of_era - day_of_era / 1460 + day_of_era / 36'524 - day_of_era / 146'096) / 365;
    int64_t year = year_of_era + era * 400;
    int64_t day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    int64_t month_prime = (5 * day_of_year + 2) / 153;
    int64_t day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    int64_t month = month_prime + (month_prime < 10 ? 3 : -9);
    year += month <= 2;
    if (year < INT32_MIN || year > INT32_MAX) {
        return ERANGE;
    }
    *out_year = (int32_t)year;
    *out_month = (uint32_t)month;
    *out_day = (uint32_t)day;
    return 0;
}

static int temporal_parse_date(const char* bytes, size_t length, size_t* position, int64_t* out_days) {
    size_t p = *position;
    bool signed_year = p < length && (bytes[p] == '+' || bytes[p] == '-');
    bool negative = signed_year && bytes[p++] == '-';
    size_t year_start = p;
    int64_t year = 0;
    while (p < length && bytes[p] >= '0' && bytes[p] <= '9') {
        if (year > INT32_MAX / 10) {
            return ERANGE;
        }
        year = year * 10 + (bytes[p++] - '0');
        if (year > (negative ? -(int64_t)INT32_MIN : INT32_MAX)) {
            return ERANGE;
        }
    }
    if (p - year_start < (signed_year ? 5 : 4) || (!signed_year && p - year_start != 4) || p >= length ||
        bytes[p] != '-') {
        return EINVAL;
    }
    ++p;
    uint32_t month = 0;
    uint32_t day = 0;
    if (temporal_parse_digits(bytes, length, &p, 2, &month) != 0 || p >= length || bytes[p] != '-') {
        return EINVAL;
    }
    ++p;
    if (temporal_parse_digits(bytes, length, &p, 2, &day) != 0) {
        return EINVAL;
    }
    *position = p;
    return dora_temporal_days_from_civil((int32_t)(negative ? -year : year), month, day, out_days);
}

static int temporal_parse_time(const char* bytes,
                               size_t length,
                               size_t* position,
                               enum dora_time_unit unit,
                               enum dora_rounding_mode mode,
                               int64_t* out_ticks) {
    size_t p = *position;
    uint32_t hour = 0;
    uint32_t minute = 0;
    uint32_t second = 0;
    if (temporal_parse_digits(bytes, length, &p, 2, &hour) != 0 || p >= length || bytes[p] != ':') {
        return EINVAL;
    }
    ++p;
    if (temporal_parse_digits(bytes, length, &p, 2, &minute) != 0 || p >= length || bytes[p] != ':') {
        return EINVAL;
    }
    ++p;
    if (temporal_parse_digits(bytes, length, &p, 2, &second) != 0) {
        return EINVAL;
    }
    if (hour >= 24 || minute >= 60 || second >= 60) {
        return ERANGE;
    }
    unsigned precision = unit == DORA_TIME_UNIT_SECOND        ? 0
                         : unit == DORA_TIME_UNIT_MILLISECOND ? 3
                         : unit == DORA_TIME_UNIT_MICROSECOND ? 6
                                                              : 9;
    int64_t fraction = 0;
    size_t digits = 0;
    uint32_t first_discarded = 0;
    bool sticky = false;
    if (p < length && bytes[p] == '.') {
        ++p;
        size_t start = p;
        while (p < length && bytes[p] >= '0' && bytes[p] <= '9') {
            uint32_t digit = (uint32_t)(bytes[p++] - '0');
            if (digits < precision) {
                fraction = fraction * 10 + digit;
            } else if (digits == precision) {
                first_discarded = digit;
            } else {
                sticky = sticky || digit != 0;
            }
            ++digits;
        }
        if (p == start) {
            return EINVAL;
        }
    }
    while (digits < precision) {
        fraction *= 10;
        ++digits;
    }
    if (first_discarded != 0 || sticky) {
        if (mode == DORA_ROUNDING_MODE_EXACT) {
            return ERANGE;
        }
        if ((mode == DORA_ROUNDING_MODE_HALF_EVEN &&
             (first_discarded > 5 || (first_discarded == 5 && (sticky || (fraction % 2) != 0)))) ||
            (mode == DORA_ROUNDING_MODE_HALF_AWAY && first_discarded >= 5)) {
            ++fraction;
        }
    }
    int64_t factor = temporal_unit_factor(unit);
    int64_t whole = ((int64_t)hour * 3600 + (int64_t)minute * 60 + second) * factor;
    *out_ticks = whole + fraction;
    *position = p;
    return 0;
}

int dora_temporal_parse(const char* bytes,
                        size_t byte_count,
                        enum dora_type kind,
                        enum dora_physical_type scalar,
                        enum dora_time_unit unit,
                        bool is_instant,
                        enum dora_rounding_mode mode,
                        enum temporal_parse_mode grammar,
                        bool* out_has_offset,
                        void* out_value) {
    if (bytes == nullptr || out_value == nullptr || byte_count == 0 || unit > DORA_TIME_UNIT_NANOSECOND ||
        mode > DORA_ROUNDING_MODE_HALF_AWAY || grammar > TEMPORAL_PARSE_READER ||
        (kind == DORA_TYPE_DATE32 && scalar != DORA_PHYSICAL_TYPE_I32) ||
        (kind == DORA_TYPE_DATE64 && scalar != DORA_PHYSICAL_TYPE_I64)) {
        return EINVAL;
    }
    if (out_has_offset != nullptr) {
        *out_has_offset = false;
    }
    size_t p = 0;
    while (p < byte_count && dora_utf8_ascii_is_text_space(bytes[p])) {
        ++p;
    }
    size_t end = byte_count;
    while (end > p && dora_utf8_ascii_is_text_space(bytes[end - 1])) {
        --end;
    }
    int64_t ticks = 0;
    if (kind == DORA_TYPE_DURATION) {
        bool negative = p < end && bytes[p] == '-';
        if (p < end && (bytes[p] == '-' || bytes[p] == '+')) {
            ++p;
        }
        uint64_t magnitude = 0;
        size_t start = p;
        uint64_t limit = negative ? (1ULL << 63U) : INT64_MAX;
        for (; p < end && bytes[p] >= '0' && bytes[p] <= '9'; ++p) {
            uint32_t digit = (uint32_t)(bytes[p] - '0');
            if (magnitude > (limit - digit) / 10) {
                return ERANGE;
            }
            magnitude = magnitude * 10 + digit;
        }
        if (p == start || p != end) {
            return EINVAL;
        }
        ticks = negative ? (magnitude == (1ULL << 63U) ? INT64_MIN : -(int64_t)magnitude) : (int64_t)magnitude;
        goto store;
    }
    if ((kind == DORA_TYPE_DATE32 || kind == DORA_TYPE_DATE64) || kind == DORA_TYPE_TIMESTAMP) {
        int64_t days = 0;
        int status = temporal_parse_date(bytes, end, &p, &days);
        if (status != 0) {
            return status;
        }
        if (kind != DORA_TYPE_TIMESTAMP) {
            if (p != end) {
                return EINVAL;
            }
            int64_t factor = scalar == DORA_PHYSICAL_TYPE_I64 ? 86'400'000 : 1;
            if (ckd_mul(&ticks, days, factor)) {
                return EOVERFLOW;
            }
            goto store;
        }
        int64_t time_ticks = 0;
        if (p != end || grammar == TEMPORAL_PARSE_CAST) {
            if (p >= end || (bytes[p] != 'T' && (grammar != TEMPORAL_PARSE_READER || bytes[p] != ' '))) {
                return EINVAL;
            }
            ++p;
            status = temporal_parse_time(bytes, end, &p, unit, mode, &time_ticks);
            if (status != 0) {
                return status;
            }
        }
        int64_t offset_seconds = 0;
        bool has_offset = p < end;
        if (has_offset) {
            if (bytes[p] == 'Z' && p + 1 == end) {
                ++p;
            } else if (bytes[p] == '+' || bytes[p] == '-') {
                bool west = bytes[p] == '-';
                ++p;
                uint32_t hours = 0;
                uint32_t minutes = 0;
                if (temporal_parse_digits(bytes, end, &p, 2, &hours) != 0) {
                    return EINVAL;
                }
                if (p < end || grammar == TEMPORAL_PARSE_CAST) {
                    if (p < end && bytes[p] == ':') {
                        ++p;
                    } else if (grammar == TEMPORAL_PARSE_CAST) {
                        return EINVAL;
                    }
                    if (temporal_parse_digits(bytes, end, &p, 2, &minutes) != 0) {
                        return EINVAL;
                    }
                }
                if (hours > 23 || minutes > 59) {
                    return EINVAL;
                }
                if (is_instant) {
                    offset_seconds = (int64_t)(hours * 3600 + minutes * 60) * (west ? -1 : 1);
                }
            } else {
                return EINVAL;
            }
        }
        if (p != end || (grammar == TEMPORAL_PARSE_CAST && has_offset != is_instant)) {
            return EINVAL;
        }
        if (out_has_offset != nullptr) {
            *out_has_offset = has_offset;
        }
        int64_t factor = temporal_unit_factor(unit);
        int64_t day_ticks = 0;
        const int64_t offset_ticks = offset_seconds * factor;
        if (ckd_mul(&day_ticks, days, 86'400 * factor) || ckd_add(&ticks, day_ticks, time_ticks) ||
            ckd_sub(&ticks, ticks, offset_ticks)) {
            return EOVERFLOW;
        }
        goto store;
    }
    if (kind == DORA_TYPE_TIME) {
        int status = temporal_parse_time(bytes, end, &p, unit, mode, &ticks);
        if (status != 0) {
            return status;
        }
        if (p != end) {
            return EINVAL;
        }
        if (ticks >= 86'400 * temporal_unit_factor(unit)) {
            return ERANGE;
        }
        goto store;
    }
    return EINVAL;
store:
    if (scalar == DORA_PHYSICAL_TYPE_I32) {
        if (ticks < INT32_MIN || ticks > INT32_MAX) {
            return ERANGE;
        }
        const int32_t narrow = (int32_t)ticks;
        memcpy(out_value, &narrow, sizeof(narrow));
    } else {
        memcpy(out_value, &ticks, sizeof(ticks));
    }
    return 0;
}

int dora_temporal_parse_type(const struct dora_logical_type* type,
                             enum dora_physical_type scalar,
                             const char* bytes,
                             size_t byte_count,
                             enum dora_rounding_mode mode,
                             enum temporal_parse_mode grammar,
                             bool* out_has_offset,
                             void* out_value) {
    if (type == nullptr) {
        return EINVAL;
    }
    const enum dora_type kind = type->id;
    const bool timestamp = kind == DORA_TYPE_TIMESTAMP;
    const enum dora_time_unit unit = timestamp ? type->parameters.timestamp.unit
                                     : kind == DORA_TYPE_DATE32 || kind == DORA_TYPE_DATE64 ? DORA_TIME_UNIT_SECOND
                                                                                            : type->parameters.unit;
    const bool instant =
        timestamp && type->parameters.timestamp.timezone != nullptr && type->parameters.timestamp.timezone[0] != '\0';
    return dora_temporal_parse(bytes,
                               byte_count,
                               kind,
                               scalar,
                               unit,
                               instant,
                               mode,
                               grammar,
                               out_has_offset,
                               out_value);
}

int dora_temporal_append_byte(struct temporal_text* text, char byte) {
    if (text->byte_count >= sizeof(text->bytes) - 1) {
        return EOVERFLOW;
    }
    text->bytes[text->byte_count++] = byte;
    return 0;
}

static int temporal_append_unsigned(uint64_t value, struct temporal_text* text, unsigned minimum_digits) {
    char reversed[FORMAT_TEMPORAL_INTEGER_DIGIT_CAPACITY] = {};
    size_t digits = 0;
    do {
        reversed[digits++] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    size_t width = digits > minimum_digits ? digits : minimum_digits;
    if (width > sizeof(text->bytes) - 1 - text->byte_count) {
        return EOVERFLOW;
    }
    for (size_t zeros = width - digits; zeros != 0; --zeros) {
        text->bytes[text->byte_count++] = '0';
    }
    while (digits != 0) {
        text->bytes[text->byte_count++] = reversed[--digits];
    }
    return 0;
}

int dora_temporal_append_signed(struct temporal_text* text,
                                int64_t value,
                                bool should_write_plus_sign,
                                unsigned minimum_digits) {
    bool negative = value < 0;
    uint64_t magnitude = negative ? UINT64_C(0) - (uint64_t)value : (uint64_t)value;
    if (negative || should_write_plus_sign) {
        int status = dora_temporal_append_byte(text, negative ? '-' : '+');
        if (status != 0) {
            return status;
        }
    }
    return temporal_append_unsigned(magnitude, text, minimum_digits);
}

int dora_temporal_format(int64_t ticks,
                         enum dora_type kind,
                         enum dora_physical_type scalar,
                         enum dora_time_unit unit,
                         bool is_instant,
                         char* out_bytes,
                         size_t capacity_bytes,
                         size_t* out_byte_count) {
    if (out_byte_count == nullptr || unit > DORA_TIME_UNIT_NANOSECOND) {
        return EINVAL;
    }
    if (kind != DORA_TYPE_DURATION && kind != DORA_TYPE_DATE32 && kind != DORA_TYPE_DATE64 && kind != DORA_TYPE_TIME &&
        kind != DORA_TYPE_TIMESTAMP) {
        return EINVAL;
    }
    struct temporal_text text = {};
    if (kind == DORA_TYPE_DURATION) {
        int status = dora_temporal_append_signed(&text, ticks, false, 1);
        if (status != 0) {
            return status;
        }
    } else {
        int64_t factor = temporal_unit_factor(unit);
        int64_t day_ticks = 86'400 * factor;
        if (kind == DORA_TYPE_TIME && (ticks < 0 || ticks >= day_ticks)) {
            return ERANGE;
        }
        int64_t days =
            (kind == DORA_TYPE_DATE32 || kind == DORA_TYPE_DATE64) && scalar == DORA_PHYSICAL_TYPE_I32 ? ticks
            : (kind == DORA_TYPE_DATE32 || kind == DORA_TYPE_DATE64) ? dora_numeric_floor_div(ticks, 86'400'000)
                                                                     : dora_numeric_floor_div(ticks, day_ticks);
        int64_t within_day = 0;
        if (kind != DORA_TYPE_DATE32) {
            within_day = ticks % day_ticks;
            if (within_day < 0) {
                within_day += day_ticks;
            }
        }
        if (kind == DORA_TYPE_DATE32 || kind == DORA_TYPE_DATE64 || kind == DORA_TYPE_TIMESTAMP) {
            int32_t year = 0;
            uint32_t month = 0;
            uint32_t day = 0;
            int status = dora_temporal_civil_from_days(days, &year, &month, &day);
            if (status != 0) {
                return status;
            }
            bool extended_year = year < 0 || year > 9999;
            if (dora_temporal_append_signed(&text, year, extended_year, extended_year ? 5 : 4) != 0 ||
                dora_temporal_append_byte(&text, '-') != 0 || temporal_append_unsigned(month, &text, 2) != 0 ||
                dora_temporal_append_byte(&text, '-') != 0 || temporal_append_unsigned(day, &text, 2) != 0) {
                return EOVERFLOW;
            }
        }
        if (kind == DORA_TYPE_TIME || kind == DORA_TYPE_TIMESTAMP) {
            if (kind == DORA_TYPE_TIMESTAMP && dora_temporal_append_byte(&text, 'T') != 0) {
                return EOVERFLOW;
            }
            uint32_t hour = (uint32_t)(within_day / (3600 * factor));
            uint32_t minute = (uint32_t)((within_day / (60 * factor)) % 60);
            uint32_t second = (uint32_t)((within_day / factor) % 60);
            uint64_t fraction = (uint64_t)(within_day % factor);
            unsigned precision = unit == DORA_TIME_UNIT_SECOND        ? 0
                                 : unit == DORA_TIME_UNIT_MILLISECOND ? 3
                                 : unit == DORA_TIME_UNIT_MICROSECOND ? 6
                                                                      : 9;
            if (temporal_append_unsigned(hour, &text, 2) != 0 || dora_temporal_append_byte(&text, ':') != 0 ||
                temporal_append_unsigned(minute, &text, 2) != 0 || dora_temporal_append_byte(&text, ':') != 0 ||
                temporal_append_unsigned(second, &text, 2) != 0) {
                return EOVERFLOW;
            }
            if (precision != 0 && (dora_temporal_append_byte(&text, '.') != 0 ||
                                   temporal_append_unsigned(fraction, &text, precision) != 0)) {
                return EOVERFLOW;
            }
            if (kind == DORA_TYPE_TIMESTAMP && is_instant && dora_temporal_append_byte(&text, 'Z') != 0) {
                return EOVERFLOW;
            }
        }
    }
    text.bytes[text.byte_count] = '\0';
    *out_byte_count = text.byte_count;
    if (out_bytes == nullptr) {
        return 0;
    }
    if (capacity_bytes <= text.byte_count) {
        return ENOBUFS;
    }
    memcpy(out_bytes, text.bytes, text.byte_count + 1);
    return 0;
}
