// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "utf8/like.h"

#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "utf8/ascii.h"
#include "utf8/encoding.h"
#include "utf8/input.h"
#include "utf8/transform.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

struct utf8_like_escape {
    uint32_t scalar;
    bool has_scalar;
};

static int utf8_like_escape_read(struct dora_bytes bytes, struct utf8_like_escape* out_escape) {
    const int status = dora_utf8_validate_span(bytes, nullptr);
    if (status != DORA_OK) {
        return status;
    }
    if (bytes.byte_count > 1) {
        return EINVAL;
    }
    *out_escape = (struct utf8_like_escape){
        .scalar = bytes.byte_count == 0 ? 0 : bytes.data[0],
        .has_scalar = bytes.byte_count != 0
    };
    return DORA_OK;
}

static int utf8_like_scalar(struct dora_bytes bytes, bool insensitive, uint32_t* out_scalar, size_t* out_consumed) {
    if (bytes.byte_count == 0) {
        return EINVAL;
    }
    const int status = utf8_read_scalar(bytes, out_scalar, out_consumed);
    if (status != DORA_OK) {
        return status;
    }
    if (insensitive) {
        *out_scalar = dora_utf8_case_map_scalar(*out_scalar, DORA_UTF8_LOWER);
    }
    return DORA_OK;
}

static int utf8_like_tokens(struct dora_bytes bytes,
                            struct utf8_like_escape escape,
                            bool insensitive,
                            size_t capacity,
                            struct dora_utf8_like_token* tokens,
                            struct dora_utf8_like_program* pattern) {
    size_t position = 0, scalars = 0;
    bool quoted = false;
    while (position < bytes.byte_count) {
        uint32_t scalar = 0;
        size_t consumed = 0;
        const int status = utf8_like_scalar((struct dora_bytes){bytes.data + position, bytes.byte_count - position},
                                            insensitive,
                                            &scalar,
                                            &consumed);
        if (status != DORA_OK) {
            return status;
        }
        if (scalars == capacity) {
            return ENOBUFS;
        }
        ++scalars;
        position += consumed;
        if (!quoted && escape.has_scalar && scalar == escape.scalar) {
            quoted = true;
            continue;
        }
        if (pattern->token_count == capacity) {
            return EINVAL;
        }
        tokens[pattern->token_count++] = (struct dora_utf8_like_token){
            .scalar = scalar,
            .kind = quoted          ? DORA_UTF8_LIKE_LITERAL
                    : scalar == '%' ? DORA_UTF8_LIKE_PERCENT
                    : scalar == '_' ? DORA_UTF8_LIKE_ANY
                                    : DORA_UTF8_LIKE_LITERAL
        };
        quoted = false;
    }
    if (quoted) {
        if (pattern->token_count == capacity) {
            return EINVAL;
        }
        tokens[pattern->token_count++] = (struct dora_utf8_like_token){.kind = DORA_UTF8_LIKE_TERMINAL_ESCAPE};
    }
    return DORA_OK;
}

static inline bool utf8_like_simple_matches(struct dora_bytes source, const struct dora_utf8_like_program* pattern) {
    const struct dora_bytes literal = pattern->literal;
    if (literal.byte_count > source.byte_count ||
        (pattern->plan == DORA_UTF8_LIKE_EXACT && literal.byte_count != source.byte_count)) {
        return false;
    }
    if (literal.byte_count == 0) {
        return true;
    }
    if (pattern->plan != DORA_UTF8_LIKE_INFIX) {
        const size_t offset = pattern->plan == DORA_UTF8_LIKE_SUFFIX ? source.byte_count - literal.byte_count : 0;
        return utf8_bytes_are_equal((struct dora_bytes){source.data + offset, literal.byte_count},
                                    literal,
                                    pattern->is_case_insensitive);
    }
    if (!pattern->is_case_insensitive) {
        return dora_utf8_find_valid(source, literal, 0) != source.byte_count;
    }
    for (size_t i = 0; i <= source.byte_count - literal.byte_count; ++i) {
        if (utf8_bytes_are_equal((struct dora_bytes){source.data + i, literal.byte_count}, literal, true)) {
            return true;
        }
    }
    return false;
}

int dora_utf8_like_match(struct dora_utf8_input input, const struct dora_utf8_like_program* program, bool* out_match) {
    if (!utf8_input_is_valid(input) || program == nullptr || out_match == nullptr ||
        (program->token_count != 0 && program->tokens == nullptr)) {
        return EINVAL;
    }
    const struct dora_bytes source = input.bytes;
    const bool insensitive = program->is_case_insensitive;
    *out_match = false;
    int status = utf8_check_input(input);
    if (status != DORA_OK) {
        return status;
    }
    if (program->plan != DORA_UTF8_LIKE_GENERAL &&
        (!insensitive || (program->is_literal_ascii && (input.proof == DORA_UTF8_VALIDATED_ASCII ||
                                                        utf8_span_is_ascii(source.data, source.byte_count))))) {
        *out_match = utf8_like_simple_matches(source, program);
        return DORA_OK;
    }
    size_t position = 0, token = 0, restart = 0, percent = SIZE_MAX;
    bool backtrack = false;
    while (position < source.byte_count) {
        struct dora_utf8_like_token current = {};
        if (!backtrack && token < program->token_count) {
            current = program->tokens[token];
            if (current.kind == DORA_UTF8_LIKE_TERMINAL_ESCAPE) {
                return EINVAL;
            }
            if (current.kind == DORA_UTF8_LIKE_PERCENT) {
                percent = token++;
                restart = position;
                continue;
            }
        } else {
            backtrack = true;
        }
        if (backtrack && (percent == SIZE_MAX || restart >= source.byte_count)) {
            return DORA_OK;
        }
        const size_t begin = backtrack ? restart : position;
        uint32_t scalar = 0;
        size_t consumed = 0;
        status = utf8_like_scalar((struct dora_bytes){source.data + begin, source.byte_count - begin},
                                  insensitive && !backtrack,
                                  &scalar,
                                  &consumed);
        if (status != DORA_OK) {
            return status;
        }
        if (backtrack) {
            restart += consumed;
            position = restart;
            token = percent + 1;
            backtrack = false;
        } else if (current.kind == DORA_UTF8_LIKE_ANY || current.scalar == scalar) {
            position += consumed;
            ++token;
        } else {
            backtrack = true;
        }
    }
    while (token < program->token_count && program->tokens[token].kind == DORA_UTF8_LIKE_PERCENT) {
        ++token;
    }
    if (token < program->token_count && program->tokens[token].kind == DORA_UTF8_LIKE_TERMINAL_ESCAPE) {
        return EINVAL;
    }
    *out_match = token == program->token_count;
    return DORA_OK;
}

int dora_utf8_like_workspace_size(struct dora_utf8_input pattern, size_t* out_workspace_bytes) {
    if (!utf8_input_is_valid(pattern) || out_workspace_bytes == nullptr) {
        return EINVAL;
    }
    size_t count = 0, bytes = 0;
    int status = dora_utf8_count(pattern, &count);
    if (status != 0) {
        return status;
    }
    if (ckd_mul(&bytes, count, sizeof(struct dora_utf8_like_token) + 4)) {
        return EOVERFLOW;
    }
    *out_workspace_bytes = bytes;
    return 0;
}

static void utf8_like_classify(struct dora_utf8_like_program* program, struct dora_mutable_bytes workspace) {
    size_t begin = 0, end = program->token_count;
    while (begin < end && program->tokens[begin].kind == DORA_UTF8_LIKE_PERCENT) {
        ++begin;
    }
    while (end > begin && program->tokens[end - 1].kind == DORA_UTF8_LIKE_PERCENT) {
        --end;
    }
    size_t bytes = 0;
    bool ascii = true;
    for (size_t i = begin; i < end; ++i) {
        if (program->tokens[i].kind != DORA_UTF8_LIKE_LITERAL) {
            return;
        }
        const uint32_t scalar = program->tokens[i].scalar;
        const size_t width = scalar < 0b1000'0000 ? 1 : scalar < 0x800 ? 2 : scalar < 0x1'0000 ? 3 : 4;
        if (ckd_add(&bytes, bytes, width)) {
            return;
        }
        ascii = ascii && scalar < 0b1000'0000;
    }
    size_t offset = 0;
    if (ckd_mul(&offset, program->token_count, sizeof(struct dora_utf8_like_token)) || offset > workspace.byte_count ||
        bytes > workspace.byte_count - offset) {
        return;
    }
    uint8_t* const literal = bytes == 0 ? nullptr : workspace.data + offset;
    size_t cursor = 0;
    for (size_t i = begin; i < end; ++i) {
        size_t width = 0;
        const int status = utf8_scalar_bytes(program->tokens[i].scalar, literal + cursor, &width);
        if (status != 0) {
            return;
        }
        cursor += width;
    }
    program->literal = (struct dora_bytes){literal, bytes};
    program->is_literal_ascii = ascii;
    program->plan = begin != 0 ? (end != program->token_count || begin == program->token_count ? DORA_UTF8_LIKE_INFIX
                                                                                               : DORA_UTF8_LIKE_SUFFIX)
                               : (end != program->token_count ? DORA_UTF8_LIKE_PREFIX : DORA_UTF8_LIKE_EXACT);
}

int dora_utf8_like_compile(struct dora_utf8_input pattern,
                           struct dora_utf8_input escape,
                           bool case_insensitive,
                           struct dora_mutable_bytes workspace,
                           struct dora_utf8_like_program* out_program) {
    if (!utf8_input_is_valid(pattern) || !utf8_input_is_valid(escape) || !utf8_output_is_valid(workspace) ||
        (uintptr_t)workspace.data % alignof(struct dora_utf8_like_token) != 0 || out_program == nullptr) {
        return EINVAL;
    }
    struct utf8_like_escape escaped = {};
    int status = utf8_like_escape_read(escape.bytes, &escaped);
    if (status == 0) {
        status = utf8_check_input(pattern);
    }
    if (status == 0 && case_insensitive && escaped.has_scalar) {
        escaped.scalar = dora_utf8_case_map_scalar(escaped.scalar, DORA_UTF8_LOWER);
    }
    struct dora_utf8_like_program result = {
        .tokens = (const struct dora_utf8_like_token*)workspace.data,
        .is_case_insensitive = case_insensitive
    };
    if (status == 0) {
        status = utf8_like_tokens(pattern.bytes,
                                  escaped,
                                  case_insensitive,
                                  workspace.byte_count / sizeof(struct dora_utf8_like_token),
                                  (struct dora_utf8_like_token*)workspace.data,
                                  &result);
    }
    if (status == 0) {
        utf8_like_classify(&result, workspace);
        *out_program = result;
    }
    return status;
}
