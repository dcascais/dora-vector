// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "kernel/decode.h"

#include "bitmap/bitmap_internal.h"
#include "dora/vector/bitmap.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/view.h"
#include "kernel/binding/read.h"
#include "types/load.h"
#include "types/types_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

//------------------------------------------------------------------------------
//
// Lanes and Coordinates
//
//------------------------------------------------------------------------------

struct decode_word {
    uint32_t first;
    uint32_t lanes;
    uint64_t mask;
};

struct decode_span {
    uint64_t* rows;
    uint64_t* validity;
    uint64_t first_row;
    uint32_t count;
};

static inline struct decode_word decode_word_of(uint32_t count, uint32_t word) {
    const uint32_t remaining = count - word * 64;
    const uint32_t lanes = remaining < 64 ? remaining : 64;
    return (struct decode_word){.first = word * 64, .lanes = lanes, .mask = bitmap_mask_low(lanes)};
}

static inline uint64_t decode_physical(const struct dora_vector_view* view, uint64_t row) {
    return view->selection.indices != nullptr ? row : view->offset + row;
}

static inline bool decode_is_null(const struct dora_vector_view* view, uint64_t physical) {
    return view->validity.data != nullptr &&
           !dora_bitmap_test(view->validity.data, view->validity_bit_offset + (physical - view->offset));
}

//------------------------------------------------------------------------------
//
// Decode
//
//------------------------------------------------------------------------------

//--------------------------------------
// Dictionary Indices
//--------------------------------------

#define DICTIONARY_SPLIT(LOAD, OUT, OUT_T)                                                                             \
    do {                                                                                                               \
        const uint64_t origin = view->offset + span.first_row;                                                         \
        const uint64_t validity_origin = view->validity_bit_offset + span.first_row;                                   \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(span.count);                                       \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            const struct decode_word lanes = decode_word_of(span.count, word);                                         \
            uint64_t valid = span.validity[word] & lanes.mask;                                                         \
            if (view->validity.data != nullptr) {                                                                      \
                valid &= dora_bitmap_load_window(view->validity.data, validity_origin + lanes.first, lanes.lanes);     \
            }                                                                                                          \
            span.validity[word] = valid;                                                                               \
            OUT_T* const out = (OUT) + lanes.first;                                                                    \
            if (valid == lanes.mask) {                                                                                 \
                for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {                                                  \
                    out[lane] = (OUT_T)LOAD(data, origin + lanes.first + lane);                                        \
                }                                                                                                      \
                continue;                                                                                              \
            }                                                                                                          \
            for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {                                                      \
                out[lane] = 0;                                                                                         \
            }                                                                                                          \
            while (valid != 0) {                                                                                       \
                const uint32_t lane = dora_bitmap_word_pop_lowest_set(&valid);                                         \
                out[lane] = (OUT_T)LOAD(data, origin + lanes.first + lane);                                            \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

// Dictionary indices: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#define DEFINE_DICTIONARY_DECODE(S, T, P)                                                                              \
    static inline uint64_t dictionary_index_##S(const uint8_t* data, uint64_t physical) {                              \
        return (uint64_t)type_load_##S(data, physical);                                                                \
    }                                                                                                                  \
    static void dictionary_resolve_##S(struct kernel_decode_binding* in_out_binding,                                   \
                                       uint64_t first_row,                                                             \
                                       uint32_t count,                                                                 \
                                       uint64_t* in_out_rows,                                                          \
                                       uint64_t* in_out_validity) {                                                    \
        const struct decode_span span =                                                                                \
            {.rows = in_out_rows, .validity = in_out_validity, .first_row = first_row, .count = count};                \
        const struct dora_vector_view* const view = in_out_binding->view;                                              \
        const uint8_t* const data = view->values.data;                                                                 \
        if (span.first_row != UINT64_MAX) {                                                                            \
            DICTIONARY_SPLIT(dictionary_index_##S, span.rows, uint64_t);                                               \
            return;                                                                                                    \
        }                                                                                                              \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(span.count);                                       \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            uint64_t active = span.validity[word];                                                                     \
            while (active != 0) {                                                                                      \
                const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                            \
                const uint64_t physical = decode_physical(view, span.rows[lane]);                                      \
                if (decode_is_null(view, physical)) {                                                                  \
                    bitmap_clear(span.validity, lane);                                                                 \
                    continue;                                                                                          \
                }                                                                                                      \
                span.rows[lane] = dictionary_index_##S(data, physical);                                                \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    static void dictionary_narrow_##S(const struct kernel_decode_binding* binding,                                     \
                                      uint64_t first_row,                                                              \
                                      uint32_t count,                                                                  \
                                      uint64_t* in_out_validity,                                                       \
                                      uint32_t* out_map) {                                                             \
        const struct decode_span span = {.validity = in_out_validity, .first_row = first_row, .count = count};         \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const uint8_t* const data = view->values.data;                                                                 \
        DICTIONARY_SPLIT(dictionary_index_##S, out_map, uint32_t);                                                     \
    }
SIGNED_INDEX_CARRIERS(DEFINE_DICTIONARY_DECODE)
UNSIGNED_INDEX_CARRIERS(DEFINE_DICTIONARY_DECODE)
#undef DEFINE_DICTIONARY_DECODE
#undef DICTIONARY_SPLIT

//--------------------------------------
// Run Ends
//--------------------------------------

struct run_end_cursor {
    const uint8_t* ends;
    uint64_t run_count;
    uint64_t run;
    uint64_t start;
    uint64_t end;
};

// Run ends: Int16, Int32, Int64
#define DEFINE_RUN_END_RESOLVE(S, T)                                                                                   \
    static uint64_t run_end_load_##S(const uint8_t* ends, uint64_t run) {                                              \
        return (uint64_t)type_load_##S(ends, run);                                                                     \
    }                                                                                                                  \
    static void run_end_seek_##S(struct run_end_cursor* in_out_cursor, uint64_t physical) {                            \
        const bool cached = in_out_cursor->end != 0;                                                                   \
        if (cached && physical >= in_out_cursor->start && physical < in_out_cursor->end) {                             \
            return;                                                                                                    \
        }                                                                                                              \
        const bool forward = cached && physical >= in_out_cursor->end;                                                 \
        if (forward && in_out_cursor->run + 1 < in_out_cursor->run_count) {                                            \
            const uint64_t next_end = run_end_load_##S(in_out_cursor->ends, in_out_cursor->run + 1);                   \
            if (physical < next_end) {                                                                                 \
                in_out_cursor->start = in_out_cursor->end;                                                             \
                in_out_cursor->end = next_end;                                                                         \
                ++in_out_cursor->run;                                                                                  \
                return;                                                                                                \
            }                                                                                                          \
        }                                                                                                              \
        uint64_t low = forward ? in_out_cursor->run + 1 : 0;                                                           \
        uint64_t high = cached && !forward ? in_out_cursor->run : in_out_cursor->run_count;                            \
        while (low < high) {                                                                                           \
            const uint64_t middle = low + (high - low) / 2;                                                            \
            if (run_end_load_##S(in_out_cursor->ends, middle) <= physical) {                                           \
                low = middle + 1;                                                                                      \
            } else {                                                                                                   \
                high = middle;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        in_out_cursor->run = low;                                                                                      \
        in_out_cursor->start = low == 0 ? 0 : run_end_load_##S(in_out_cursor->ends, low - 1);                          \
        in_out_cursor->end = run_end_load_##S(in_out_cursor->ends, low);                                               \
    }                                                                                                                  \
    static void run_end_resolve_##S(struct kernel_decode_binding* in_out_binding,                                      \
                                    uint64_t first_row,                                                                \
                                    uint32_t count,                                                                    \
                                    uint64_t* in_out_rows,                                                             \
                                    uint64_t* in_out_validity) {                                                       \
        const struct decode_span span =                                                                                \
            {.rows = in_out_rows, .validity = in_out_validity, .first_row = first_row, .count = count};                \
        const struct dora_vector_view* const view = in_out_binding->view;                                              \
        const uint64_t element_count = view->values.byte_count / sizeof(T);                                            \
        struct run_end_cursor cursor = {                                                                               \
            .ends = (const uint8_t*)view->values.data + view->run_end_element_offset * sizeof(T),                      \
            .run_count =                                                                                               \
                element_count > view->run_end_element_offset ? element_count - view->run_end_element_offset : 0,       \
            .run = in_out_binding->run_cursor,                                                                         \
            .end = in_out_binding->run_end_cache,                                                                      \
        };                                                                                                             \
        if (cursor.end != 0 && cursor.run != 0) {                                                                      \
            cursor.start = run_end_load_##S(cursor.ends, cursor.run - 1);                                              \
        }                                                                                                              \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(span.count);                                       \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            uint64_t active = span.validity[word];                                                                     \
            while (active != 0) {                                                                                      \
                const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                            \
                const uint64_t row = span.first_row == UINT64_MAX ? span.rows[lane] : span.first_row + lane;           \
                const uint64_t physical = decode_physical(view, row);                                                  \
                if (decode_is_null(view, physical)) {                                                                  \
                    bitmap_clear(span.validity, lane);                                                                 \
                    continue;                                                                                          \
                }                                                                                                      \
                run_end_seek_##S(&cursor, physical);                                                                   \
                span.rows[lane] = cursor.run;                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        in_out_binding->run_cursor = cursor.run;                                                                       \
        in_out_binding->run_end_cache = cursor.end;                                                                    \
    }
DEFINE_RUN_END_RESOLVE(i16, int16_t)
DEFINE_RUN_END_RESOLVE(i32, int32_t)
DEFINE_RUN_END_RESOLVE(i64, int64_t)
#undef DEFINE_RUN_END_RESOLVE

//--------------------------------------
// Selections
//--------------------------------------

#define SELECTION_SPLIT(LOAD, OUT, OUT_T, ADDEND)                                                                      \
    do {                                                                                                               \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(span.count);                                       \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            const struct decode_word lanes = decode_word_of(span.count, word);                                         \
            uint64_t active = span.validity[word] & lanes.mask;                                                        \
            OUT_T* const out = (OUT) + lanes.first;                                                                    \
            const uint64_t origin = span.first_row + lanes.first;                                                      \
            if (active == lanes.mask) {                                                                                \
                for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {                                                  \
                    out[lane] = (OUT_T)((ADDEND) + LOAD(indices, origin + lane));                                      \
                }                                                                                                      \
                continue;                                                                                              \
            }                                                                                                          \
            for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {                                                      \
                out[lane] = 0;                                                                                         \
            }                                                                                                          \
            while (active != 0) {                                                                                      \
                const uint32_t lane = dora_bitmap_word_pop_lowest_set(&active);                                        \
                out[lane] = (OUT_T)((ADDEND) + LOAD(indices, origin + lane));                                          \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

// Selection indices: UInt32, UInt64
#define DEFINE_SELECTION_DECODE(S, T)                                                                                  \
    static inline uint64_t selection_index_##S(const uint8_t* indices, uint64_t logical) {                             \
        return (uint64_t)type_load_##S(indices, logical);                                                              \
    }                                                                                                                  \
    static void selection_resolve_##S(struct kernel_decode_binding* in_out_binding,                                    \
                                      uint64_t first_row,                                                              \
                                      uint32_t count,                                                                  \
                                      uint64_t* in_out_rows,                                                           \
                                      uint64_t* in_out_validity) {                                                     \
        const struct decode_span span =                                                                                \
            {.rows = in_out_rows, .validity = in_out_validity, .first_row = first_row, .count = count};                \
        const struct dora_vector_view* const view = in_out_binding->view;                                              \
        const uint8_t* const indices = view->selection.indices;                                                        \
        if (span.first_row != UINT64_MAX) {                                                                            \
            SELECTION_SPLIT(selection_index_##S, span.rows, uint64_t, view->offset);                                   \
            return;                                                                                                    \
        }                                                                                                              \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(span.count);                                       \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            uint64_t active = span.validity[word];                                                                     \
            while (active != 0) {                                                                                      \
                const uint32_t lane = word * 64 + dora_bitmap_word_pop_lowest_set(&active);                            \
                span.rows[lane] = view->offset + selection_index_##S(indices, span.rows[lane]);                        \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    static void selection_narrow_##S(const struct kernel_decode_binding* binding,                                      \
                                     uint64_t first_row,                                                               \
                                     uint32_t count,                                                                   \
                                     uint64_t* in_out_validity,                                                        \
                                     uint32_t* out_map) {                                                              \
        const struct decode_span span = {.validity = in_out_validity, .first_row = first_row, .count = count};         \
        const uint8_t* const indices = binding->view->selection.indices;                                               \
        SELECTION_SPLIT(selection_index_##S, out_map, uint32_t, 0);                                                    \
    }
DEFINE_SELECTION_DECODE(u32, uint32_t)
DEFINE_SELECTION_DECODE(u64, uint64_t)
#undef DEFINE_SELECTION_DECODE
#undef SELECTION_SPLIT

//--------------------------------------
// Binding
//--------------------------------------

struct decode_entry {
    typeof(&dictionary_resolve_i8) resolve;
    typeof(&dictionary_narrow_i8) narrow;
};

static const struct decode_entry decode_entries[KERNEL_DECODE_SELECTION + 1][DORA_PHYSICAL_TYPE_U64 + 1] = {
    [KERNEL_DECODE_DICTIONARY] =
        {
            [DORA_PHYSICAL_TYPE_I8] = {dictionary_resolve_i8, dictionary_narrow_i8},
            [DORA_PHYSICAL_TYPE_I16] = {dictionary_resolve_i16, dictionary_narrow_i16},
            [DORA_PHYSICAL_TYPE_I32] = {dictionary_resolve_i32, dictionary_narrow_i32},
            [DORA_PHYSICAL_TYPE_I64] = {dictionary_resolve_i64, dictionary_narrow_i64},
            [DORA_PHYSICAL_TYPE_U8] = {dictionary_resolve_u8, dictionary_narrow_u8},
            [DORA_PHYSICAL_TYPE_U16] = {dictionary_resolve_u16, dictionary_narrow_u16},
            [DORA_PHYSICAL_TYPE_U32] = {dictionary_resolve_u32, dictionary_narrow_u32},
            [DORA_PHYSICAL_TYPE_U64] = {dictionary_resolve_u64, dictionary_narrow_u64},
        },
    [KERNEL_DECODE_RUN_END] =
        {
            [DORA_PHYSICAL_TYPE_I16] = {run_end_resolve_i16, nullptr},
            [DORA_PHYSICAL_TYPE_I32] = {run_end_resolve_i32, nullptr},
            [DORA_PHYSICAL_TYPE_I64] = {run_end_resolve_i64, nullptr},
        },
    [KERNEL_DECODE_SELECTION] = {
        [DORA_PHYSICAL_TYPE_U32] = {selection_resolve_u32, selection_narrow_u32},
        [DORA_PHYSICAL_TYPE_U64] = {selection_resolve_u64, selection_narrow_u64},
    },
};

int dora_kernel_decode_bind(const struct dora_vector_view* view, struct kernel_decode_binding* out_binding) {
    if (view == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    struct kernel_decode_binding binding = {.view = view, .index_physical = view->physical_type};
    if (view->selection.indices != nullptr) {
        binding.kind = KERNEL_DECODE_SELECTION;
        binding.index_physical = view->selection.physical_type;
    } else if (view->layout == DORA_LAYOUT_DICTIONARY) {
        binding.kind = KERNEL_DECODE_DICTIONARY;
    } else if (view->layout == DORA_LAYOUT_RUN_END) {
        binding.kind = KERNEL_DECODE_RUN_END;
    } else {
        return ENOTSUP;
    }
    if (binding.index_physical > DORA_PHYSICAL_TYPE_U64 ||
        decode_entries[binding.kind][binding.index_physical].resolve == nullptr) {
        return EINVAL;
    }
    binding.resolve = decode_entries[binding.kind][binding.index_physical].resolve;
    binding.narrow = decode_entries[binding.kind][binding.index_physical].narrow;
    *out_binding = binding;
    return DORA_OK;
}

//------------------------------------------------------------------------------
//
// Narrowing
//
//------------------------------------------------------------------------------

void dora_kernel_narrow_rows(const uint64_t* rows,
                             uint64_t base,
                             uint32_t* restrict out_map,
                             const uint64_t* validity,
                             uint32_t count) {
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    for (uint32_t word = 0; word < word_count; ++word) {
        const struct decode_word lanes = decode_word_of(count, word);
        uint64_t active = validity[word] & lanes.mask;
        const uint64_t* const in = rows + lanes.first;
        uint32_t* const out = out_map + lanes.first;
        if (active == lanes.mask) {
            for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {
                out[lane] = (uint32_t)(in[lane] - base);
            }
            continue;
        }
        for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {
            out[lane] = 0;
        }
        while (active != 0) {
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&active);
            out[lane] = (uint32_t)(in[lane] - base);
        }
    }
}

void dora_kernel_gather_u64(const uint64_t* values,
                            const uint32_t* map,
                            const uint64_t* validity,
                            uint32_t count,
                            uint64_t* out_values,
                            uint64_t null_value) {
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    for (uint32_t word = 0; word < word_count; ++word) {
        const struct decode_word lanes = decode_word_of(count, word);
        uint64_t active = validity[word] & lanes.mask;
        const uint32_t* const in = map + lanes.first;
        uint64_t* const out = out_values + lanes.first;
        if (active == lanes.mask) {
            for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {
                out[lane] = values[in[lane]];
            }
            continue;
        }
        for (uint32_t lane = 0; lane < lanes.lanes; ++lane) {
            out[lane] = null_value;
        }
        while (active != 0) {
            const uint32_t lane = dora_bitmap_word_pop_lowest_set(&active);
            out[lane] = values[in[lane]];
        }
    }
}

int dora_kernel_run_end_last(const struct dora_vector_view* ends, int64_t* out_end) {
    const uint64_t physical = dora_vector_view_get_physical_row(ends, ends->length - 1);
    const uint8_t* const values = ends->values.data;
    switch (ends->physical_type) {
    case DORA_PHYSICAL_TYPE_I16: {
        *out_end = type_load_i16(values, physical);
        return DORA_OK;
    }
    case DORA_PHYSICAL_TYPE_I32: {
        *out_end = type_load_i32(values, physical);
        return DORA_OK;
    }
    case DORA_PHYSICAL_TYPE_I64: {
        *out_end = type_load_i64(values, physical);
        return DORA_OK;
    }
    default:
        return EINVAL;
    }
}

//------------------------------------------------------------------------------
//
// Gather
//
//------------------------------------------------------------------------------

#define DEFINE_GATHER(W)                                                                                               \
    static void gather_##W(const struct kernel_gather_binding* binding,                                                \
                           const uint64_t* rows,                                                                       \
                           void* out_values,                                                                           \
                           const uint64_t* validity,                                                                   \
                           uint32_t count) {                                                                           \
        const struct dora_vector_view* const view = binding->view;                                                     \
        const uint8_t* const values =                                                                                  \
            (const uint8_t*)view->values.data + (view->selection.indices != nullptr ? 0 : view->offset) * (W);         \
        uint8_t* const out = out_values;                                                                               \
        const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);                                            \
        for (uint32_t word = 0; word < word_count; ++word) {                                                           \
            const struct decode_word lanes = decode_word_of(count, word);                                              \
            uint64_t active = validity[word] & lanes.mask;                                                             \
            if (active == lanes.mask) {                                                                                \
                for (uint32_t lane = lanes.first; lane < lanes.first + lanes.lanes; ++lane) {                          \
                    (memcpy)(out + (size_t)lane * (W), values + rows[lane] * (W), (W));                                \
                }                                                                                                      \
                continue;                                                                                              \
            }                                                                                                          \
            while (active != 0) {                                                                                      \
                const uint32_t lane = lanes.first + dora_bitmap_word_pop_lowest_set(&active);                          \
                (memcpy)(out + (size_t)lane * (W), values + rows[lane] * (W), (W));                                    \
            }                                                                                                          \
        }                                                                                                              \
    }
DEFINE_GATHER(1)
DEFINE_GATHER(2)
DEFINE_GATHER(4)
DEFINE_GATHER(8)
DEFINE_GATHER(16)
DEFINE_GATHER(32)
#undef DEFINE_GATHER

static void gather_bits(const struct kernel_gather_binding* binding,
                        const uint64_t* rows,
                        void* out_values,
                        const uint64_t* validity,
                        uint32_t count) {
    const struct dora_vector_view* const view = binding->view;
    dora_bitmap_gather_words_u64(out_values,
                                 view->values.data,
                                 view->selection.indices != nullptr ? 0 : view->offset,
                                 rows,
                                 validity,
                                 count);
}

void dora_kernel_gather_validity(const struct dora_vector_view* view,
                                 const uint64_t* rows,
                                 const uint64_t* active,
                                 uint32_t count,
                                 uint64_t* out_words) {
    dora_bitmap_gather_words_u64(out_words,
                                 view->validity.data,
                                 view->validity_bit_offset - (view->selection.indices != nullptr ? view->offset : 0),
                                 rows,
                                 active,
                                 count);
}

void dora_kernel_gather_map_validity(const struct dora_vector_view* view,
                                     const uint32_t* map,
                                     const uint64_t* active,
                                     uint32_t count,
                                     uint64_t* out_words) {
    dora_bitmap_gather_words_u32(out_words, view->validity.data, view->validity_bit_offset, map, active, count);
}

void dora_kernel_load_validity(const struct dora_vector_view* view,
                               uint64_t first_row,
                               uint32_t count,
                               uint64_t* out_words) {
    if (view->validity.data != nullptr) {
        dora_bitmap_load_words(view->validity.data, view->validity_bit_offset + first_row, count, out_words);
        return;
    }
    const uint32_t word_count = (uint32_t)bitmap_words_for_bits(count);
    for (uint32_t word = 0; word < word_count; ++word) {
        out_words[word] = decode_word_of(count, word).mask;
    }
}

static typeof(&gather_1) gather_by_width(uint32_t width) {
    switch (width) {
    case 1:
        return gather_1;
    case 2:
        return gather_2;
    case 4:
        return gather_4;
    case 8:
        return gather_8;
    case 16:
        return gather_16;
    case 32:
        return gather_32;
    default:
        return nullptr;
    }
}

static bool gather_can_use_u32_map(const struct dora_vector_view* view, bool bits) {
    static constexpr uint64_t map_rows = (uint64_t)UINT32_MAX + 1;
    if (view->selection.indices == nullptr) {
        return view->length <= map_rows;
    }
    if (bits) {
        return view->values.byte_count <= bitmap_bytes_for_bits(map_rows);
    }
    return view->byte_width == 0 || view->values.byte_count / view->byte_width <= map_rows;
}

int dora_kernel_gather_bind(const struct dora_vector_view* view, struct kernel_gather_binding* out_binding) {
    if (view == nullptr || out_binding == nullptr) {
        return EINVAL;
    }
    if (view->layout != DORA_LAYOUT_FIXED && view->layout != DORA_LAYOUT_BITS) {
        return ENOTSUP;
    }
    const bool bits = view->layout == DORA_LAYOUT_BITS;
    *out_binding = (struct kernel_gather_binding){
        .gather = bits ? gather_bits : gather_by_width(view->byte_width),
        .view = view,
        .width = bits ? 0 : view->byte_width,
        .can_use_u32_map = gather_can_use_u32_map(view, bits),
    };
    return DORA_OK;
}
