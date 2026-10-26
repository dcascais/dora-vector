// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Interval normalization and ordering.

#ifndef DORA_TEMPORAL_INTERVAL_H
#define DORA_TEMPORAL_INTERVAL_H

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Intervals
//
//------------------------------------------------------------------------------

struct temporal_interval_parts {
    int64_t months;
    int64_t days;
    int64_t nanoseconds;
};

static inline struct temporal_interval_parts temporal_interval_normalize(struct temporal_interval_parts value) {
    constexpr int64_t nanoseconds_per_day = 86'400'000'000'000;
    int64_t carry_days = value.nanoseconds / nanoseconds_per_day;
    value.nanoseconds %= nanoseconds_per_day;
    if (value.nanoseconds < 0) {
        --carry_days;
        value.nanoseconds += nanoseconds_per_day;
    }
    value.days += carry_days;
    int64_t carry_months = value.days / 30;
    value.days %= 30;
    if (value.days < 0) {
        --carry_months;
        value.days += 30;
    }
    value.months += carry_months;
    return value;
}

static inline int temporal_interval_order(struct temporal_interval_parts left, struct temporal_interval_parts right) {
    if (left.months != right.months) {
        return (left.months > right.months) - (left.months < right.months);
    }
    if (left.days != right.days) {
        return (left.days > right.days) - (left.days < right.days);
    }
    return (left.nanoseconds > right.nanoseconds) - (left.nanoseconds < right.nanoseconds);
}

#endif
