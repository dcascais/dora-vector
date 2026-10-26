// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// UTF-8 LIKE matching.

#ifndef DORA_UTF8_LIKE_H
#define DORA_UTF8_LIKE_H

#include "dora/vector/types.h"
#include "utf8/input.h"

#include <stddef.h>
#include <stdint.h>

enum dora_utf8_like_kind : uint8_t {
    DORA_UTF8_LIKE_LITERAL,
    DORA_UTF8_LIKE_ANY,
    DORA_UTF8_LIKE_PERCENT,
    DORA_UTF8_LIKE_TERMINAL_ESCAPE
};

struct dora_utf8_like_token {
    uint32_t scalar;
    enum dora_utf8_like_kind kind;
};

enum dora_utf8_like_plan : uint8_t {
    DORA_UTF8_LIKE_GENERAL,
    DORA_UTF8_LIKE_EXACT,
    DORA_UTF8_LIKE_PREFIX,
    DORA_UTF8_LIKE_SUFFIX,
    DORA_UTF8_LIKE_INFIX
};

struct dora_utf8_like_program {
    const struct dora_utf8_like_token* tokens;
    size_t token_count;
    bool is_case_insensitive;
    bool is_literal_ascii;
    enum dora_utf8_like_plan plan;
    struct dora_bytes literal;
};

int dora_utf8_like_workspace_size(struct dora_utf8_input pattern, size_t* out_workspace_bytes);
int dora_utf8_like_compile(struct dora_utf8_input pattern,
                           struct dora_utf8_input escape,
                           bool case_insensitive,
                           struct dora_mutable_bytes workspace,
                           struct dora_utf8_like_program* out_program);
int dora_utf8_like_match(struct dora_utf8_input input, const struct dora_utf8_like_program* program, bool* out_match);

#endif
