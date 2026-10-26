// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

#include "dora/vector/stream.h"

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/interop/arrow/abi.h"
#include "dora/vector/interop/arrow/arrow.h"
#include "dora/vector/memory.h"
#include "dora/vector/schema.h"
#include "dora/vector/status.h"
#include "dora/vector/types.h"
#include "dora/vector/vector.h"
#include "dora/vector/view.h"
#include "identity/identity.h"
#include "interop/arrow/arrow_internal.h"
#include "memory/memory_internal.h"
#include "schema/schema_internal.h"
#include "status/status_internal.h"
#include "stream/stream_internal.h"
#include "utf8/utf8.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

struct imported_stream {
    struct dora_vector_source source;
    struct dora_allocation storage;
    struct dora_schema schema;
    struct ArrowArrayStream stream;
    struct ArrowDeviceArrayStream device_stream;
    struct dora_arrow_device_access* access;
    struct dora_arrow_limits limits;
    bool is_device_stream;
};

struct exported_stream {
    struct dora_allocation storage;
    struct dora_schema schema;
    struct dora_reader reader;
    struct dora_vector pending;
    struct dora_error error;
    ArrowDeviceType device_type;
    int failure;
    bool is_end;
    bool is_busy;
    bool is_release_pending;
};

static void import_release(struct dora_vector_source* source) {
    struct imported_stream* const state = (struct imported_stream*)source;
    if (state->stream.release != nullptr) {
        state->stream.release(&state->stream);
    }
    if (state->device_stream.release != nullptr) {
        state->device_stream.release(&state->device_stream);
    }
    dora_schema_release(&state->schema);
    struct dora_allocation storage = state->storage;
    dora_allocation_release(&storage);
}

static void
import_error(struct ArrowArrayStream* host, struct ArrowDeviceArrayStream* device, struct dora_error* out_error) {
    if (out_error == nullptr) {
        return;
    }
    const char* message = nullptr;
    if (device != nullptr) {
        if (device->get_last_error != nullptr) {
            message = device->get_last_error(device);
        }
    } else if (host != nullptr && host->get_last_error != nullptr) {
        message = host->get_last_error(host);
    }
    size_t byte_count;
    enum dora_utf8_prefix_stop stop;
    const int status = dora_utf8_copy_cstr_prefix(
        message,
        (struct dora_mutable_bytes){(uint8_t*)out_error->message, sizeof(out_error->message)},
        &byte_count,
        &stop);
    if (status != 0) {
        out_error->message[0] = '\0';
    }
}

static int import_next(struct dora_vector_source* source,
                       struct dora_pool pool,
                       struct dora_vector* out_vector,
                       struct dora_stream_result* out_result,
                       struct dora_error* out_error) {
    struct imported_stream* const state = (struct imported_stream*)source;
    struct ArrowDeviceArray array = {};
    int status = state->is_device_stream ? state->device_stream.get_next(&state->device_stream, &array)
                                         : state->stream.get_next(&state->stream, &array.array);
    if (status != DORA_OK) {
        import_error(state->is_device_stream ? nullptr : &state->stream,
                     state->is_device_stream ? &state->device_stream : nullptr,
                     out_error);
    }
    if (status == DORA_OK && array.array.release == nullptr) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_END};
        return DORA_OK;
    }
    if (status == DORA_OK && state->is_device_stream && array.device_type != state->device_stream.device_type) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        status = state->is_device_stream ? dora_arrow_import_move_device_array(pool,
                                                                               &state->schema,
                                                                               &array,
                                                                               state->access,
                                                                               &state->limits,
                                                                               out_vector,
                                                                               out_error)
                                         : dora_arrow_import_move_array(pool,
                                                                        &state->schema,
                                                                        &array.array,
                                                                        &state->limits,
                                                                        out_vector,
                                                                        out_error);
    }
    if (array.array.release != nullptr) {
        array.array.release(&array.array);
    }
    if (status == DORA_OK) {
        *out_result = (struct dora_stream_result){.status = DORA_STREAM_STATUS_READY};
    }
    return status;
}

static int import_stream(struct dora_pool pool,
                         struct ArrowArrayStream* host,
                         struct ArrowDeviceArrayStream* device,
                         struct dora_arrow_device_access* access,
                         const struct dora_arrow_limits* limits,
                         struct dora_reader* out_reader,
                         struct dora_error* out_error) {
    if (out_error != nullptr) {
        *out_error = (struct dora_error){};
    }
    if (out_reader == nullptr || !handle_is_empty(out_reader->handle) || (host == nullptr && device == nullptr) ||
        (host != nullptr && (host->release == nullptr || host->get_schema == nullptr || host->get_next == nullptr)) ||
        (device != nullptr && (device->release == nullptr || device->get_schema == nullptr ||
                               device->get_next == nullptr || access == nullptr || access->describe == nullptr))) {
        return EINVAL;
    }
    // Foreign callbacks require import-pool reset and release to stay blocked.
    dora_pool_callback_enter(pool);
    struct ArrowSchema foreign_schema = {};
    int status =
        device != nullptr ? device->get_schema(device, &foreign_schema) : host->get_schema(host, &foreign_schema);
    if (status != DORA_OK) {
        import_error(host, device, out_error);
    }
    const struct dora_arrow_limits resolved = dora_arrow_resolve_limits(limits);
    struct arrow_traversal_state traversal = {};
    struct dora_schema schema = {};
    if (status == DORA_OK) {
        dora_arrow_traversal_init(pool, &resolved, &traversal);
    }
    if (status == DORA_OK) {
        status = dora_arrow_import_schema_impl(pool, &foreign_schema, &resolved, &traversal, &schema, out_error);
    }
    if (foreign_schema.release != nullptr) {
        foreign_schema.release(&foreign_schema);
    }
    dora_pool_callback_leave(pool);
    struct dora_allocation storage = {};
    if (status == DORA_OK) {
        status = arrow_traversal_charge(&resolved, &traversal, 0, sizeof(struct imported_stream));
    }
    if (status == DORA_OK) {
        status = dora_memory_payload_allocate(pool, sizeof(struct imported_stream), &storage);
    }
    struct imported_stream* const state = storage.address;
    struct dora_reader candidate = {};
    if (status == DORA_OK) {
        *state = (struct imported_stream){
            .source = {.next = import_next, .release = import_release},
            .storage = storage,
            .schema = schema,
            .access = access,
            .limits = dora_arrow_resolve_limits(limits),
            .is_device_stream = device != nullptr
        };
        schema = (struct dora_schema){};
        status = dora_reader_init(&candidate, pool, &state->source);
    }
    dora_schema_release(&schema);
    dora_allocation_release(&traversal.stack_storage);
    if (status == DORA_OK &&
        (out_reader->handle.slot_index != 0 || out_reader->handle.generation != 0 ||
         out_reader->handle.pool_index != 0 || out_reader->handle.is_owner || out_reader->handle.reserved != 0)) {
        status = EINVAL;
    }
    if (status == DORA_OK) {
        if (device != nullptr) {
            state->device_stream = *device;
            device->release = nullptr;
        } else {
            state->stream = *host;
            host->release = nullptr;
        }
        *out_reader = candidate;
    } else {
        dora_reader_release(&candidate);
    }
    if (status != DORA_OK && out_error != nullptr && out_error->message[0] == '\0') {
        error_message_copy_bounded(out_error, "Arrow stream initialization failed");
    }
    return status;
}

static void export_discard(struct exported_stream* state) {
    dora_vector_release(&state->pending);
    dora_reader_release(&state->reader);
    dora_schema_release(&state->schema);
    struct dora_allocation storage = state->storage;
    dora_allocation_release(&storage);
}

static int export_leave(struct exported_stream* state, int status) {
    state->is_busy = false;
    if (state->is_release_pending) {
        export_discard(state);
    }
    return status;
}

static int export_read(struct exported_stream* state) {
    if (state->failure != DORA_OK) {
        return state->failure;
    }
    if (state->is_end || dora_vector_is_valid(&state->pending)) {
        return DORA_OK;
    }
    int status;
    for (;;) {
        struct dora_stream_result result = {};
        status = dora_reader_next(&state->reader, &state->pending, &result, &state->error);
        if (status != DORA_OK) {
            break;
        }
        if (result.status == DORA_STREAM_STATUS_END) {
            state->is_end = true;
            break;
        }
        if (result.status == DORA_STREAM_STATUS_READY) {
            struct dora_schema schema = {};
            status = dora_vector_retain_schema(&state->pending, &schema);
            if (status == DORA_OK) {
                bool matches = false;
                status = dora_schema_declarations_equal(&state->schema, &schema, &matches);
                if (status == DORA_OK && !matches) {
                    status = EINVAL;
                }
            }
            dora_schema_release(&schema);
            break;
        }
        if (result.readiness.settle == nullptr) {
            status = ENOTSUP;
            break;
        }
        status = result.readiness.settle(&result.readiness);
        if (status != DORA_OK) {
            break;
        }
    }
    if (status != DORA_OK) {
        dora_vector_release(&state->pending);
        state->failure = status;
        if (state->error.message[0] == '\0') {
            error_message_copy_bounded(&state->error, "Arrow stream read failed");
        }
    }
    return status;
}

static int export_schema(struct exported_stream* state, struct ArrowSchema* out_schema) {
    if (state->is_busy) {
        return EBUSY;
    }
    state->is_busy = true;
    const int status = dora_arrow_export_schema(state->storage.pool, &state->schema, out_schema);
    if (status != DORA_OK) {
        error_message_copy_bounded(&state->error, "Arrow stream schema export failed");
    }
    return export_leave(state, status);
}

static int host_schema(struct ArrowArrayStream* stream, struct ArrowSchema* out_schema) {
    return stream == nullptr || stream->release == nullptr ? EINVAL : export_schema(stream->private_data, out_schema);
}

static int device_schema(struct ArrowDeviceArrayStream* stream, struct ArrowSchema* out_schema) {
    return stream == nullptr || stream->release == nullptr ? EINVAL : export_schema(stream->private_data, out_schema);
}

static int host_next(struct ArrowArrayStream* stream, struct ArrowArray* out_array) {
    if (stream == nullptr || stream->release == nullptr || out_array == nullptr || out_array->release != nullptr) {
        return EINVAL;
    }
    struct exported_stream* const state = stream->private_data;
    if (state->is_busy) {
        return EBUSY;
    }
    state->is_busy = true;
    int status = export_read(state);
    if (status == DORA_OK && state->is_end) {
        *out_array = (struct ArrowArray){};
    }
    if (status == DORA_OK && !state->is_end) {
        status = dora_arrow_export_array_in_pool(state->storage.pool, &state->pending, out_array);
        if (status == DORA_OK) {
            dora_vector_release(&state->pending);
        }
    }
    if (status != DORA_OK) {
        state->failure = status;
        dora_vector_release(&state->pending);
        if (state->error.message[0] == '\0') {
            error_message_copy_bounded(&state->error, "Arrow stream array export failed");
        }
    }
    return export_leave(state, status);
}

static int device_next(struct ArrowDeviceArrayStream* stream, struct ArrowDeviceArray* out_array) {
    if (stream == nullptr || stream->release == nullptr || out_array == nullptr ||
        out_array->array.release != nullptr) {
        return EINVAL;
    }
    struct exported_stream* const state = stream->private_data;
    if (state->is_busy) {
        return EBUSY;
    }
    state->is_busy = true;
    int status = export_read(state);
    if (status == DORA_OK && state->is_end) {
        *out_array = (struct ArrowDeviceArray){};
    }
    if (status == DORA_OK && !state->is_end) {
        struct ArrowDeviceArray candidate = {};
        status = dora_arrow_export_device_array_in_pool(state->storage.pool, &state->pending, &candidate);
        if (status == DORA_OK && candidate.device_type != state->device_type) {
            status = EINVAL;
        }
        if (status == DORA_OK) {
            *out_array = candidate;
            candidate = (struct ArrowDeviceArray){};
        }
        if (candidate.array.release != nullptr) {
            candidate.array.release(&candidate.array);
        }
        dora_vector_release(&state->pending);
    }
    if (status != DORA_OK) {
        state->failure = status;
        if (state->error.message[0] == '\0') {
            error_message_copy_bounded(&state->error, "Arrow device stream export failed");
        }
    }
    return export_leave(state, status);
}

static const char* host_error(struct ArrowArrayStream* stream) {
    if (stream == nullptr || stream->release == nullptr) {
        return nullptr;
    }
    const struct exported_stream* const state = stream->private_data;
    return state->error.message[0] == '\0' ? nullptr : state->error.message;
}

static const char* device_error(struct ArrowDeviceArrayStream* stream) {
    if (stream == nullptr || stream->release == nullptr) {
        return nullptr;
    }
    const struct exported_stream* const state = stream->private_data;
    return state->error.message[0] == '\0' ? nullptr : state->error.message;
}

static void host_release(struct ArrowArrayStream* stream) {
    if (stream == nullptr || stream->release == nullptr) {
        return;
    }
    struct exported_stream* const state = stream->private_data;
    stream->release = nullptr;
    if (state->is_busy) {
        state->is_release_pending = true;
    } else {
        export_discard(state);
    }
}

static void device_release(struct ArrowDeviceArrayStream* stream) {
    if (stream == nullptr || stream->release == nullptr) {
        return;
    }
    struct exported_stream* const state = stream->private_data;
    stream->release = nullptr;
    if (state->is_busy) {
        state->is_release_pending = true;
    } else {
        export_discard(state);
    }
}

static int
export_stream(const struct dora_schema* schema, struct dora_reader* reader, struct exported_stream** out_state) {
    struct dora_pool pool = {};
    if (!dora_schema_is_valid(schema) || dora_reader_get_pool(reader, &pool) != DORA_OK || !reader->handle.is_owner) {
        return EINVAL;
    }
    struct dora_allocation storage = {};
    int status = dora_pool_allocate(pool, sizeof(struct exported_stream), &storage);
    struct exported_stream* const state = storage.address;
    if (status == DORA_OK) {
        *state = (struct exported_stream){.storage = storage, .device_type = ARROW_DEVICE_CPU};
        // The export claim pins the schema until stream release.
        struct dora_handle owner = {};
        status = dora_identity_claim_acquire(pool, &schema->handle, IDENTITY_KIND_SCHEMA, &owner);
        if (status == DORA_OK) {
            state->schema = (struct dora_schema){.handle = owner, .node_index = schema->node_index};
        }
    }
    if (status == DORA_OK) {
        status = dora_reader_move(reader, &state->reader);
    }
    if (status == DORA_OK) {
        *out_state = state;
    } else if (state != nullptr) {
        export_discard(state);
    }
    return status;
}

int dora_arrow_import_stream(struct dora_pool pool,
                             struct ArrowArrayStream* in_out_stream,
                             const struct dora_arrow_limits* limits,
                             struct dora_reader* out_reader,
                             struct dora_error* out_error) {
    return import_stream(pool, in_out_stream, nullptr, nullptr, limits, out_reader, out_error);
}

int dora_arrow_import_device_stream(struct dora_pool pool,
                                    struct ArrowDeviceArrayStream* in_out_stream,
                                    struct dora_arrow_device_access* access,
                                    const struct dora_arrow_limits* limits,
                                    struct dora_reader* out_reader,
                                    struct dora_error* out_error) {
    return import_stream(pool, nullptr, in_out_stream, access, limits, out_reader, out_error);
}

int dora_arrow_export_stream(const struct dora_schema* schema,
                             struct dora_reader* in_out_reader,
                             struct ArrowArrayStream* out_stream) {
    if (out_stream == nullptr || out_stream->release != nullptr) {
        return EINVAL;
    }
    struct exported_stream* state = nullptr;
    const int status = export_stream(schema, in_out_reader, &state);
    if (status == DORA_OK) {
        *out_stream = (struct ArrowArrayStream){
            .get_schema = host_schema,
            .get_next = host_next,
            .get_last_error = host_error,
            .release = host_release,
            .private_data = state
        };
    }
    return status;
}

int dora_arrow_export_device_stream(const struct dora_schema* schema,
                                    struct dora_reader* in_out_reader,
                                    struct ArrowDeviceArrayStream* out_stream) {
    if (out_stream == nullptr || out_stream->release != nullptr) {
        return EINVAL;
    }
    struct exported_stream* state = nullptr;
    const int status = export_stream(schema, in_out_reader, &state);
    if (status != DORA_OK) {
        return status;
    }
    int read_status = export_read(state);
    if (read_status == DORA_OK && !state->is_end) {
        struct dora_vector_view view = {};
        read_status = dora_arrow_device_view(&state->pending, &view);
        if (read_status == DORA_OK) {
            state->device_type = view.placement.device.type;
        }
    }
    if (read_status != DORA_OK) {
        state->failure = read_status;
        dora_vector_release(&state->pending);
    }
    *out_stream = (struct ArrowDeviceArrayStream){
        .device_type = state->device_type,
        .get_schema = device_schema,
        .get_next = device_next,
        .get_last_error = device_error,
        .release = device_release,
        .private_data = state
    };
    return DORA_OK;
}
