// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Device placement and readiness.

#ifndef DORA_VECTOR_DEVICE_H
#define DORA_VECTOR_DEVICE_H

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Placement and Readiness
//
//------------------------------------------------------------------------------

struct dora_device_region {
    /// Buffer base address.
    const void* data;
    /// Start within data, in bytes.
    uint64_t byte_offset;
    /// Region length in bytes.
    uint64_t byte_count;
};

enum dora_access : uint8_t {
    DORA_ACCESS_NONE = 0,
    DORA_ACCESS_HOST = 1,
    DORA_ACCESS_DEVICE = 2,
    DORA_ACCESS_HOST_AND_DEVICE = 3
};

enum dora_device_type : int32_t {
    DORA_DEVICE_TYPE_CPU = 1
};

struct dora_device {
    /// Device kind.
    enum dora_device_type type;
    /// Device ordinal.
    int64_t id;
};

struct dora_placement {
    /// Storage device.
    struct dora_device device;
    /// Permitted access paths.
    enum dora_access access;
};

struct dora_readiness {
    /// Synchronization token.
    void* event;
    /// Waits for readiness; required when event is nonnull.
    int (*settle)(const struct dora_readiness* readiness);
};

#endif
