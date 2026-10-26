// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "bitmap/bitmap_internal.h"
#include "dora/vector/handle.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "memory/memory_internal.h"
#include "types/types_internal.h"
#include "vector/image.h"
#include "vector/vector_internal.h"

#include <errno.h>
#include <stdckdint.h>
#include <stddef.h>
#include <stdint.h>

static int owner_failure(struct dora_buffer_owner* owner, int status) {
    if (owner != nullptr && owner->release != nullptr) {
        owner->release(owner);
    }
    return status;
}

int dora_vector_init_wrap_input(struct dora_vector* vector,
                                struct dora_pool pool,
                                const struct vector_input* input,
                                struct dora_error* out_error) {
    if (!dora_vector_is_empty(vector)) {
        return owner_failure(input->owner, EINVAL);
    }
    struct vector_image* image = nullptr;
    struct dora_vector published = {};
    int status = dora_vector_prepare(pool, input, &image, out_error);
    if (status == DORA_OK) {
        status = dora_vector_publish(image, &published);
    }
    if (status != DORA_OK) {
        dora_vector_discard(image);
        return owner_failure(input->owner, status);
    }
    image->buffer_image->wrapped_owner = input->owner;
    if (!dora_vector_is_empty(vector)) {
        dora_vector_release(&published);
        return EINVAL;
    }
    *vector = published;
    return DORA_OK;
}

static int validity_bytes(const uint8_t* validity, struct dora_range bits, struct dora_bytes* out_bytes) {
    uint64_t endpoint = 0;
    if (ckd_add(&endpoint, bits.offset, bits.length)) {
        return EINVAL;
    }
    const uint64_t bytes = validity != nullptr ? bitmap_bytes_for_bits(endpoint) : 0;
    if (bytes > SIZE_MAX) {
        return EOVERFLOW;
    }
    *out_bytes = (struct dora_bytes){.data = validity, .byte_count = (size_t)bytes};
    return DORA_OK;
}

static int wrap_sized(struct dora_vector* vector,
                      struct dora_pool pool,
                      const struct dora_logical_type* type,
                      const void* values,
                      uint32_t width,
                      const uint8_t* validity,
                      struct dora_range rows,
                      enum dora_physical_type expected,
                      struct dora_buffer_owner* owner,
                      struct dora_error* out_error) {
    uint64_t bytes = 0;
    if (rows.length > INT64_MAX || ckd_mul(&bytes, rows.length, width)) {
        return owner_failure(owner, EINVAL);
    }
    if (bytes > SIZE_MAX) {
        return owner_failure(owner, EOVERFLOW);
    }
    enum dora_physical_type physical = DORA_PHYSICAL_TYPE_NONE;
    int status = type_physical(type, &physical);
    if (status != DORA_OK || physical != expected ||
        (expected == DORA_PHYSICAL_TYPE_NONE && (type == nullptr || type->id != DORA_TYPE_FIXED_BINARY))) {
        return owner_failure(owner, EINVAL);
    }
    struct dora_bytes buffers[2] = {{}, {.data = values, .byte_count = bytes}};
    status = validity_bytes(validity, rows, &buffers[0]);
    if (status != DORA_OK) {
        return owner_failure(owner, status);
    }
    const struct dora_schema_field field = {.type = *type, .layout = DORA_LAYOUT_FIXED};
    const struct vector_input input = {
        .field = &field,
        .row_count = rows.length,
        .null_count = -1,
        .validity_bit_offset = rows.offset,
        .buffers = buffers,
        .buffer_count = 2,
        .owner = owner,
        .should_scan_geometry = true
    };
    return dora_vector_init_wrap_input(vector, pool, &input, out_error);
}

static int wrap_binary(struct dora_vector* vector,
                       struct dora_pool pool,
                       const struct dora_logical_type* type,
                       enum dora_layout layout,
                       const void* offsets,
                       struct dora_bytes data,
                       const uint8_t* validity,
                       struct dora_range rows,
                       struct dora_buffer_owner* owner,
                       struct dora_error* out_error) {
    if (type == nullptr || (type->id != DORA_TYPE_BINARY && type->id != DORA_TYPE_UTF8) || rows.length > INT64_MAX) {
        return owner_failure(owner, EINVAL);
    }
    size_t bytes = 0;
    if (ckd_mul(&bytes, rows.length + 1, layout == DORA_LAYOUT_BINARY ? sizeof(int32_t) : sizeof(int64_t))) {
        return owner_failure(owner, EOVERFLOW);
    }
    struct dora_bytes buffers[3] = {{}, {.data = offsets, .byte_count = bytes}, data};
    const int status = validity_bytes(validity, rows, &buffers[0]);
    if (status != DORA_OK) {
        return owner_failure(owner, status);
    }
    const struct dora_schema_field field = {.type = *type, .layout = layout};
    const struct vector_input input = {
        .field = &field,
        .row_count = rows.length,
        .null_count = -1,
        .validity_bit_offset = rows.offset,
        .buffers = buffers,
        .buffer_count = 3,
        .owner = owner,
        .should_scan_geometry = true
    };
    return dora_vector_init_wrap_input(vector, pool, &input, out_error);
}

//------------------------------------------------------------------------------
//
// Construction by Wrap
//
//------------------------------------------------------------------------------

//--------------------------------------
// Boolean
//--------------------------------------

int dora_vector_init_wrap_packed_bool(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const struct dora_logical_type* type,
                                      const uint8_t* bits,
                                      uint64_t bit_offset,
                                      const uint8_t* validity,
                                      uint64_t validity_bit_offset,
                                      uint64_t row_count,
                                      struct dora_buffer_owner* owner,
                                      struct dora_error* out_error) {
    if (type == nullptr || type->id != DORA_TYPE_BOOLEAN) {
        return owner_failure(owner, EINVAL);
    }
    struct dora_bytes buffers[2] = {};
    int status =
        validity_bytes(validity, (struct dora_range){.offset = validity_bit_offset, .length = row_count}, &buffers[0]);
    if (status == DORA_OK) {
        status = validity_bytes(bits, (struct dora_range){.offset = bit_offset, .length = row_count}, &buffers[1]);
    }
    if (status != DORA_OK) {
        return owner_failure(owner, status);
    }
    const struct dora_schema_field field = {.type = *type, .layout = DORA_LAYOUT_BITS};
    const struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = -1,
        .offset = bit_offset,
        .validity_bit_offset = validity_bit_offset,
        .buffers = buffers,
        .buffer_count = 2,
        .owner = owner,
        .should_scan_geometry = true
    };
    return dora_vector_init_wrap_input(vector, pool, &input, out_error);
}

//--------------------------------------
// Signed Integers
//--------------------------------------

int dora_vector_init_wrap_i8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const int8_t* values,
                             const uint8_t* validity,
                             uint64_t validity_bit_offset,
                             uint64_t row_count,
                             struct dora_buffer_owner* owner,
                             struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(int8_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I8,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_i16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int16_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(int16_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I16,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_i32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int32_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(int32_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I32,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_i64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const int64_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(int64_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I64,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_i128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i128* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_i128),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I128,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_i256(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_i256* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_i256),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_I256,
                      owner,
                      out_error);
}

//--------------------------------------
// Unsigned Integers
//--------------------------------------

int dora_vector_init_wrap_u8(struct dora_vector* vector,
                             struct dora_pool pool,
                             const struct dora_logical_type* type,
                             const uint8_t* values,
                             const uint8_t* validity,
                             uint64_t validity_bit_offset,
                             uint64_t row_count,
                             struct dora_buffer_owner* owner,
                             struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(uint8_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_U8,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_u16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint16_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(uint16_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_U16,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_u32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint32_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(uint32_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_U32,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_u64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const uint64_t* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(uint64_t),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_U64,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_u128(struct dora_vector* vector,
                               struct dora_pool pool,
                               const struct dora_logical_type* type,
                               const struct dora_u128* values,
                               const uint8_t* validity,
                               uint64_t validity_bit_offset,
                               uint64_t row_count,
                               struct dora_buffer_owner* owner,
                               struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_u128),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_U128,
                      owner,
                      out_error);
}

//--------------------------------------
// Floating Point
//--------------------------------------

int dora_vector_init_wrap_f16(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const struct dora_f16_bits* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_f16_bits),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_F16,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_f32(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const float* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(float),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_F32,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_f64(struct dora_vector* vector,
                              struct dora_pool pool,
                              const struct dora_logical_type* type,
                              const double* values,
                              const uint8_t* validity,
                              uint64_t validity_bit_offset,
                              uint64_t row_count,
                              struct dora_buffer_owner* owner,
                              struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(double),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_F64,
                      owner,
                      out_error);
}

//--------------------------------------
// Intervals and Bytes
//--------------------------------------

int dora_vector_init_wrap_interval_day_time(struct dora_vector* vector,
                                            struct dora_pool pool,
                                            const struct dora_logical_type* type,
                                            const struct dora_interval_day_time* values,
                                            const uint8_t* validity,
                                            uint64_t validity_bit_offset,
                                            uint64_t row_count,
                                            struct dora_buffer_owner* owner,
                                            struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_interval_day_time),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_INTERVAL_DAY_TIME,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_interval_month_day_nano(struct dora_vector* vector,
                                                  struct dora_pool pool,
                                                  const struct dora_logical_type* type,
                                                  const struct dora_interval_month_day_nano* values,
                                                  const uint8_t* validity,
                                                  uint64_t validity_bit_offset,
                                                  uint64_t row_count,
                                                  struct dora_buffer_owner* owner,
                                                  struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      sizeof(struct dora_interval_month_day_nano),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_INTERVAL_MONTH_DAY_NANO,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_fixed_binary(struct dora_vector* vector,
                                       struct dora_pool pool,
                                       const struct dora_logical_type* type,
                                       const uint8_t* values,
                                       const uint8_t* validity,
                                       uint64_t validity_bit_offset,
                                       uint64_t row_count,
                                       struct dora_buffer_owner* owner,
                                       struct dora_error* out_error) {
    return wrap_sized(vector,
                      pool,
                      type,
                      values,
                      (type != nullptr ? type->parameters.byte_width : 0),
                      validity,
                      (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                      DORA_PHYSICAL_TYPE_NONE,
                      owner,
                      out_error);
}

int dora_vector_init_wrap_binary(struct dora_vector* vector,
                                 struct dora_pool pool,
                                 const struct dora_logical_type* type,
                                 const int32_t* offsets,
                                 struct dora_bytes data,
                                 const uint8_t* validity,
                                 uint64_t validity_bit_offset,
                                 uint64_t row_count,
                                 struct dora_buffer_owner* owner,
                                 struct dora_error* out_error) {
    return wrap_binary(vector,
                       pool,
                       type,
                       DORA_LAYOUT_BINARY,
                       offsets,
                       data,
                       validity,
                       (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                       owner,
                       out_error);
}

int dora_vector_init_wrap_large_binary(struct dora_vector* vector,
                                       struct dora_pool pool,
                                       const struct dora_logical_type* type,
                                       const int64_t* offsets,
                                       struct dora_bytes data,
                                       const uint8_t* validity,
                                       uint64_t validity_bit_offset,
                                       uint64_t row_count,
                                       struct dora_buffer_owner* owner,
                                       struct dora_error* out_error) {
    return wrap_binary(vector,
                       pool,
                       type,
                       DORA_LAYOUT_LARGE_BINARY,
                       offsets,
                       data,
                       validity,
                       (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                       owner,
                       out_error);
}

int dora_vector_init_wrap_binary_view(struct dora_vector* vector,
                                      struct dora_pool pool,
                                      const struct dora_logical_type* type,
                                      const struct dora_binary_view_entry* views,
                                      const struct dora_bytes* data_buffers,
                                      uint32_t data_buffer_count,
                                      const uint8_t* validity,
                                      uint64_t validity_bit_offset,
                                      uint64_t row_count,
                                      struct dora_buffer_owner* owner,
                                      struct dora_error* out_error) {
    if (!dora_vector_is_empty(vector) || type == nullptr ||
        (type->id != DORA_TYPE_BINARY && type->id != DORA_TYPE_UTF8) || row_count > INT64_MAX ||
        (data_buffer_count != 0 && data_buffers == nullptr)) {
        return owner_failure(owner, EINVAL);
    }
    uint32_t count = 0;
    size_t bytes = 0;
    size_t view_bytes = 0;
    size_t size_bytes = 0;
    if (ckd_add(&count, data_buffer_count, 3) || ckd_mul(&bytes, count, sizeof(struct dora_bytes)) ||
        ckd_mul(&size_bytes, data_buffer_count, sizeof(int64_t)) ||
        ckd_mul(&view_bytes, row_count, sizeof(struct dora_binary_view_entry))) {
        return owner_failure(owner, EOVERFLOW);
    }
    struct dora_allocation workspace = {};
    int status = dora_memory_payload_allocate(pool, bytes, &workspace);
    if (status != DORA_OK) {
        return owner_failure(owner, status);
    }
    struct dora_bytes* const buffers = workspace.address;
    struct dora_allocation sizes = {};
    status = dora_pool_allocate(pool, size_bytes, &sizes);
    if (status == DORA_OK) {
        status = validity_bytes(validity,
                                (struct dora_range){.offset = validity_bit_offset, .length = row_count},
                                &buffers[0]);
    }
    buffers[1] = (struct dora_bytes){.data = (const uint8_t*)views, .byte_count = view_bytes};
    for (uint32_t index = 0; status == DORA_OK && index < data_buffer_count; ++index) {
        if (data_buffers[index].byte_count > INT64_MAX) {
            status = EOVERFLOW;
            break;
        }
        buffers[index + 2] = data_buffers[index];
        ((int64_t*)sizes.address)[index] = (int64_t)data_buffers[index].byte_count;
    }
    buffers[count - 1] = (struct dora_bytes){.data = sizes.address, .byte_count = size_bytes};
    const struct dora_schema_field field = {.type = *type, .layout = DORA_LAYOUT_BINARY_VIEW};
    const struct vector_input input = {
        .field = &field,
        .row_count = row_count,
        .null_count = -1,
        .validity_bit_offset = validity_bit_offset,
        .buffers = buffers,
        .data_buffer_sizes = sizes.address,
        .buffer_count = count,
        .owner = owner,
        .should_scan_geometry = true
    };
    struct vector_image* image = nullptr;
    struct dora_vector published = {};
    if (status == DORA_OK) {
        status = dora_vector_prepare(pool, &input, &image, out_error);
    }
    if (status == DORA_OK) {
        status =
            dora_allocation_adopt(&sizes, &dora_vector_image_buffers(image->buffer_image)[count - 1].owner.allocation);
    }
    if (status == DORA_OK) {
        status = dora_vector_publish(image, &published);
    }
    if (status == DORA_OK) {
        image->buffer_image->wrapped_owner = owner;
    } else {
        dora_vector_discard(image);
        status = owner_failure(owner, status);
    }
    dora_allocation_release(&sizes);
    dora_allocation_release(&workspace);
    status = vector_finish_publication(&published, vector, status);
    return status;
}
