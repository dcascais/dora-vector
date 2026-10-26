// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Temporal parsing and formatting.

#ifndef DORA_TEMPORAL_TEMPORAL_H
#define DORA_TEMPORAL_TEMPORAL_H

#include "dora/vector/types.h"

#include <stddef.h>
#include <stdint.h>

static constexpr size_t TEMPORAL_TEXT_CAPACITY = 96;

//------------------------------------------------------------------------------
//
// Conversion
//
//------------------------------------------------------------------------------

//--------------------------------------
// Temporal
//--------------------------------------

int dora_temporal_days_from_civil(int32_t year, uint32_t month, uint32_t day, int64_t* out_days);
int dora_temporal_civil_from_days(int64_t days, int32_t* out_year, uint32_t* out_month, uint32_t* out_day);

enum temporal_parse_mode : uint8_t {
    TEMPORAL_PARSE_CAST,
    TEMPORAL_PARSE_READER
};

// Date32, Date64, Time, Timestamp, Duration
int dora_temporal_parse(const char* bytes,
                        size_t byte_count,
                        enum dora_type kind,
                        enum dora_physical_type scalar,
                        enum dora_time_unit unit,
                        bool is_instant,
                        enum dora_rounding_mode mode,
                        enum temporal_parse_mode grammar,
                        bool* out_has_offset,
                        void* out_value);

// Date32, Date64, Time, Timestamp, Duration
int dora_temporal_parse_type(const struct dora_logical_type* type,
                             enum dora_physical_type scalar,
                             const char* bytes,
                             size_t byte_count,
                             enum dora_rounding_mode mode,
                             enum temporal_parse_mode grammar,
                             bool* out_has_offset,
                             void* out_value);
// Date32, Date64, Time, Timestamp, Duration
int dora_temporal_format(int64_t ticks,
                         enum dora_type kind,
                         enum dora_physical_type scalar,
                         enum dora_time_unit unit,
                         bool is_instant,
                         char* out_bytes,
                         size_t capacity_bytes,
                         size_t* out_byte_count);

//--------------------------------------
// Temporal Text
//--------------------------------------

struct temporal_text {
    char bytes[TEMPORAL_TEXT_CAPACITY];
    size_t byte_count;
};

int dora_temporal_append_byte(struct temporal_text* text, char byte);
int dora_temporal_append_signed(struct temporal_text* text,
                                int64_t value,
                                bool should_write_plus_sign,
                                unsigned minimum_digits);

#endif
