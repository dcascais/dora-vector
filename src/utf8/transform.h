// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 transformations.

#ifndef DORA_UTF8_TRANSFORM_H
#define DORA_UTF8_TRANSFORM_H

#include "dora/vector/types.h"
#include "types/types_internal.h"
#include "utf8/ascii.h"
#include "utf8/input.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum dora_utf8_case : uint8_t {
    DORA_UTF8_LOWER,
    DORA_UTF8_UPPER
};

struct dora_utf8_join_state {
    size_t byte_count;
    bool has_value;
};

// Requires valid UTF-8.
static inline int utf8_case_capacity(size_t input_bytes, size_t* out_bytes) {
    return ckd_add(out_bytes, input_bytes, input_bytes / 2) ? EOVERFLOW : 0;
}

int dora_utf8_case_map(struct dora_utf8_input input,
                       enum dora_utf8_case mapping,
                       enum dora_utf8_emit_mode mode,
                       struct dora_mutable_bytes output,
                       size_t* out_byte_count);
uint32_t dora_utf8_case_map_scalar(uint32_t scalar, enum dora_utf8_case mapping);

// Output requires input.bytes.byte_count bytes.
static inline int utf8_reverse(struct dora_utf8_input input,
                               enum dora_utf8_emit_mode mode,
                               struct dora_mutable_bytes output,
                               size_t* out_byte_count) {
    if (!utf8_input_is_valid(input) || out_byte_count == nullptr || mode > DORA_UTF8_EMIT ||
        (mode == DORA_UTF8_EMIT && !utf8_output_is_valid(output))) {
        return EINVAL;
    }
    int status = utf8_check_input(input);
    if (status != 0) {
        return status;
    }
    const struct dora_bytes source = input.bytes;
    const bool ascii = input.proof == DORA_UTF8_VALIDATED_ASCII || utf8_span_is_ascii(source.data, source.byte_count);
    if (!ascii && input.proof == DORA_UTF8_PROMISED) {
        status = dora_utf8_validate_span(source, nullptr);
        if (status != 0) {
            return status;
        }
    }
    if (mode == DORA_UTF8_EMIT) {
        if (source.byte_count > output.byte_count) {
            return ENOBUFS;
        }
        const size_t size = source.byte_count;
        if (ascii) {
            for (size_t i = 0; i < size; ++i) {
                output.data[size - 1 - i] = source.data[i];
            }
        } else {
            for (size_t i = 0; i < size;) {
                const uint8_t lead = source.data[i];
                const size_t width = 1U + (size_t)(lead >= 0xc0) + (size_t)(lead >= 0xe0) + (size_t)(lead >= 0xf0);
                if (width > size - i) {
                    return EILSEQ;
                }
                memcpy(output.data + size - i - width, source.data + i, width);
                i += width;
            }
        }
    }
    *out_byte_count = source.byte_count;
    return 0;
}

static inline int
utf8_replace_append(struct dora_bytes piece, struct dora_mutable_bytes output, bool write, size_t* in_out_size) {
    size_t end;
    if (ckd_add(&end, *in_out_size, piece.byte_count) || end > INT64_MAX) {
        return EOVERFLOW;
    }
    if (write && end > output.byte_count) {
        return ENOBUFS;
    }
    if (write && piece.byte_count != 0) {
        utf8_copy_validated_bytes(piece, output.data + *in_out_size);
    }
    *in_out_size = end;
    return 0;
}

static inline int utf8_replace(struct dora_utf8_input input,
                               struct dora_utf8_input needle,
                               struct dora_utf8_input replacement,
                               enum dora_utf8_emit_mode mode,
                               struct dora_mutable_bytes output,
                               size_t* out_size) {
    if (!utf8_input_is_valid(input) || !utf8_input_is_valid(needle) || !utf8_input_is_valid(replacement) ||
        out_size == nullptr || mode > DORA_UTF8_EMIT || (mode == DORA_UTF8_EMIT && !utf8_output_is_valid(output))) {
        return EINVAL;
    }
    int checked = utf8_check_input(input);
    if (checked == 0) {
        checked = utf8_check_input(needle);
    }
    if (checked == 0) {
        checked = utf8_check_input(replacement);
    }
    if (checked != 0) {
        return checked;
    }
    const struct dora_bytes source = input.bytes;
    const bool write = mode == DORA_UTF8_EMIT;

    if (needle.bytes.byte_count == replacement.bytes.byte_count) {
        if (source.byte_count > INT64_MAX) {
            return EOVERFLOW;
        }
        if (!write) {
            *out_size = source.byte_count;
            return 0;
        }
    }
    if (needle.bytes.byte_count == 1 && replacement.bytes.byte_count == 1) {
        if (source.byte_count > output.byte_count) {
            return ENOBUFS;
        }
        for (size_t i = 0; i < source.byte_count; ++i) {
            output.data[i] = source.data[i] == needle.bytes.data[0] ? replacement.bytes.data[0] : source.data[i];
        }
        *out_size = source.byte_count;
        return 0;
    }

    size_t start = 0, size = 0;
    int status = 0;
    for (;;) {
        size_t position = source.byte_count;
        bool found = false;
        if (needle.bytes.byte_count != 0) {
            position = dora_utf8_find_valid(source, needle.bytes, start);
            found = position < source.byte_count;
        }
        if (status == 0) {
            const struct dora_bytes prefix = {source.data == nullptr ? nullptr : source.data + start, position - start};
            status = utf8_replace_append(prefix, output, write, &size);
        }
        if (status != 0 || !found) {
            break;
        }
        status = utf8_replace_append(replacement.bytes, output, write, &size);
        if (status != 0) {
            break;
        }
        start = position + needle.bytes.byte_count;
    }
    if (status == 0) {
        *out_size = size;
    }
    return status;
}

static inline int utf8_join_append(struct dora_utf8_input value,
                                   struct dora_utf8_input separator,
                                   enum dora_utf8_emit_mode mode,
                                   struct dora_mutable_bytes output,
                                   struct dora_utf8_join_state* in_out_state) {
    if (!utf8_input_is_valid(value) || !utf8_input_is_valid(separator) || in_out_state == nullptr ||
        mode > DORA_UTF8_EMIT || (mode == DORA_UTF8_EMIT && !utf8_output_is_valid(output))) {
        return EINVAL;
    }
    int status = utf8_check_input(value);
    if (status == 0 && in_out_state->has_value) {
        status = utf8_check_input(separator);
    }
    if (status != 0) {
        return status;
    }
    const size_t separator_size = in_out_state->has_value ? separator.bytes.byte_count : 0;
    size_t end;
    if (ckd_add(&end, in_out_state->byte_count, separator_size) || ckd_add(&end, end, value.bytes.byte_count) ||
        end > INT64_MAX) {
        return EOVERFLOW;
    }
    if (mode == DORA_UTF8_EMIT) {
        if (end > output.byte_count) {
            return ENOBUFS;
        }
        size_t cursor = in_out_state->byte_count;
        if (separator_size != 0) {
            utf8_copy_validated_bytes(separator.bytes, output.data + cursor);
            cursor += separator_size;
        }
        if (value.bytes.byte_count != 0) {
            utf8_copy_validated_bytes(value.bytes, output.data + cursor);
        }
    }
    *in_out_state = (struct dora_utf8_join_state){end, true};
    return 0;
}

struct dora_utf8_padding {
    struct dora_bytes bytes;
    uint64_t target_scalar_count;
    size_t pattern_scalar_count;
    enum dora_utf8_proof proof;
    bool is_left_padding;
};

struct dora_utf8_padding_plan {
    struct dora_bytes source_bytes;
    struct dora_bytes remainder_bytes;
    uint64_t pattern_repetition_count;
    size_t byte_count;
};

static inline int utf8_padding_plan(struct dora_utf8_input input,
                                    const struct dora_utf8_padding* padding,
                                    struct dora_utf8_padding_plan* out_plan) {
    size_t count = 0;
    const struct dora_bytes source = input.bytes;
    int status = dora_utf8_count(input, &count);
    if (status != 0) {
        return status;
    }
    if (input.proof == DORA_UTF8_CHECKED) {
        input.proof = DORA_UTF8_VALIDATED;
    }
    struct dora_utf8_padding_plan plan = {.source_bytes = source};
    if (padding->target_scalar_count < count) {
        struct dora_range prefix = {};
        status = dora_utf8_locate((struct dora_utf8_input){source, input.proof},
                                  0,
                                  (size_t)padding->target_scalar_count,
                                  &prefix);
        if (status != 0) {
            return status;
        }
        if (prefix.offset != 0 || prefix.length > source.byte_count) {
            return EILSEQ;
        }
        plan.source_bytes.byte_count = prefix.length;
    } else if (padding->target_scalar_count > count) {
        if (padding->pattern_scalar_count == 0) {
            return padding->bytes.byte_count == 0 ? EINVAL : EILSEQ;
        }
        const uint64_t missing = padding->target_scalar_count - count;
        plan.pattern_repetition_count = missing / padding->pattern_scalar_count;
        const size_t remaining = (size_t)(missing % padding->pattern_scalar_count);
        struct dora_range prefix = {};
        status = dora_utf8_locate((struct dora_utf8_input){padding->bytes, padding->proof}, 0, remaining, &prefix);
        if (status != 0) {
            return status;
        }
        if (prefix.offset != 0 || prefix.length > padding->bytes.byte_count) {
            return EILSEQ;
        }
        plan.remainder_bytes = (struct dora_bytes){padding->bytes.data, (size_t)prefix.length};
    }
    if (ckd_mul(&plan.byte_count, plan.pattern_repetition_count, padding->bytes.byte_count) ||
        ckd_add(&plan.byte_count, plan.byte_count, plan.remainder_bytes.byte_count) ||
        ckd_add(&plan.byte_count, plan.byte_count, plan.source_bytes.byte_count) || plan.byte_count > INT64_MAX) {
        return EOVERFLOW;
    }
    *out_plan = plan;
    return 0;
}

static inline void utf8_padding_write(const struct dora_utf8_padding_plan* plan,
                                      const struct dora_utf8_padding* padding,
                                      uint8_t* output) {
    size_t cursor = 0;
    if (!padding->is_left_padding && plan->source_bytes.byte_count != 0) {
        utf8_copy_validated_bytes(plan->source_bytes, output);
        cursor = plan->source_bytes.byte_count;
    }
    if (padding->bytes.byte_count == 1 && plan->pattern_repetition_count != 0) {
        memset(output + cursor, padding->bytes.data[0], (size_t)plan->pattern_repetition_count);
        cursor += (size_t)plan->pattern_repetition_count;
    } else {
        for (uint64_t cycle = 0; cycle < plan->pattern_repetition_count; ++cycle) {
            utf8_copy_validated_bytes(padding->bytes, output + cursor);
            cursor += padding->bytes.byte_count;
        }
    }
    if (plan->remainder_bytes.byte_count != 0) {
        utf8_copy_validated_bytes(plan->remainder_bytes, output + cursor);
        cursor += plan->remainder_bytes.byte_count;
    }
    if (padding->is_left_padding && plan->source_bytes.byte_count != 0) {
        utf8_copy_validated_bytes(plan->source_bytes, output + cursor);
    }
}

static inline int utf8_pad(struct dora_utf8_input input,
                           const struct dora_utf8_padding* padding,
                           enum dora_utf8_emit_mode mode,
                           struct dora_mutable_bytes output,
                           size_t* out_byte_count) {
    if (!utf8_input_is_valid(input) || padding == nullptr || !bytes_are_well_formed(padding->bytes) ||
        out_byte_count == nullptr || mode > DORA_UTF8_EMIT ||
        (mode == DORA_UTF8_EMIT && !utf8_output_is_valid(output))) {
        return EINVAL;
    }
    struct dora_utf8_padding_plan plan = {};
    int status = utf8_padding_plan(input, padding, &plan);
    if (status == 0 && mode == DORA_UTF8_EMIT) {
        if (plan.byte_count > output.byte_count) {
            return ENOBUFS;
        }
        if (plan.byte_count != 0) {
            if (output.data == nullptr) {
                return EINVAL;
            }
            utf8_padding_write(&plan, padding, output.data);
        }
    }
    if (status == 0) {
        *out_byte_count = plan.byte_count;
    }
    return status;
}

#endif
