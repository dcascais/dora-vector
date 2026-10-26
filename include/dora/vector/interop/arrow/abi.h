// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#ifndef DORA_VECTOR_INTEROP_ARROW_ABI_H
#define DORA_VECTOR_INTEROP_ARROW_ABI_H

#include <stdint.h>

//------------------------------------------------------------------------------
//
// C Data Interface
//
//------------------------------------------------------------------------------

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

//--------------------------------------
// Schema
//--------------------------------------

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;

    void (*release)(struct ArrowSchema*);

    void* private_data;
};

//--------------------------------------
// Array
//--------------------------------------

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;

    void (*release)(struct ArrowArray*);

    void* private_data;
};

#endif

//------------------------------------------------------------------------------
//
// C Stream Interface
//
//------------------------------------------------------------------------------

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

//--------------------------------------
// Array Stream
//--------------------------------------

struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream*, struct ArrowSchema* out);
    int (*get_next)(struct ArrowArrayStream*, struct ArrowArray* out);
    const char* (*get_last_error)(struct ArrowArrayStream*);

    void (*release)(struct ArrowArrayStream*);

    void* private_data;
};

#endif

//------------------------------------------------------------------------------
//
// C Device Data Interface
//
//------------------------------------------------------------------------------

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

//--------------------------------------
// Device Type
//--------------------------------------

typedef int32_t ArrowDeviceType;

#define ARROW_DEVICE_CPU 1
#define ARROW_DEVICE_CUDA 2
#define ARROW_DEVICE_CUDA_HOST 3
#define ARROW_DEVICE_OPENCL 4
#define ARROW_DEVICE_VULKAN 7
#define ARROW_DEVICE_METAL 8
#define ARROW_DEVICE_VPI 9
#define ARROW_DEVICE_ROCM 10
#define ARROW_DEVICE_ROCM_HOST 11
#define ARROW_DEVICE_EXT_DEV 12
#define ARROW_DEVICE_CUDA_MANAGED 13
#define ARROW_DEVICE_ONEAPI 14
#define ARROW_DEVICE_WEBGPU 15
#define ARROW_DEVICE_HEXAGON 16

//--------------------------------------
// Device Array
//--------------------------------------

struct ArrowDeviceArray {
    struct ArrowArray array;
    int64_t device_id;
    ArrowDeviceType device_type;
    void* sync_event;
    int64_t reserved[3];
};

#endif

//------------------------------------------------------------------------------
//
// C Device Data Stream Interface
//
//------------------------------------------------------------------------------

#ifndef ARROW_C_DEVICE_STREAM_INTERFACE
#define ARROW_C_DEVICE_STREAM_INTERFACE

//--------------------------------------
// Device Array Stream
//--------------------------------------

struct ArrowDeviceArrayStream {
    ArrowDeviceType device_type;

    int (*get_schema)(struct ArrowDeviceArrayStream*, struct ArrowSchema*);

    int (*get_next)(struct ArrowDeviceArrayStream*, struct ArrowDeviceArray*);

    const char* (*get_last_error)(struct ArrowDeviceArrayStream*);

    void (*release)(struct ArrowDeviceArrayStream*);

    void* private_data;
};

#endif

#endif
