// SPDX-FileCopyrightText: 2026 Daniel Cascais
// SPDX-License-Identifier: MIT

/// Readers, writers and pipelines.

#ifndef DORA_VECTOR_STREAM_H
#define DORA_VECTOR_STREAM_H

#include "dora/vector/device.h"
#include "dora/vector/handle.h"
#include "dora/vector/status.h"

#include <stdint.h>

//------------------------------------------------------------------------------
//
// Streams
//
//------------------------------------------------------------------------------

enum dora_stream_status : uint8_t {
    DORA_STREAM_STATUS_READY,
    DORA_STREAM_STATUS_BLOCKED,
    DORA_STREAM_STATUS_END
};

struct dora_stream_result {
    /// Stream state.
    enum dora_stream_status status;
    /// Whether the operation consumed its input.
    bool is_input_consumed;
    /// Readiness when blocked.
    struct dora_readiness readiness;
};

//------------------------------------------------------------------------------
//
// Reader
//
//------------------------------------------------------------------------------

//--------------------------------------
// Source Capabilities
//--------------------------------------

/// Pool reset and release return EBUSY during active reader callbacks.
struct dora_vector_source {
    /// Publishes the next batch and stream state.
    int (*next)(struct dora_vector_source* source,
                struct dora_pool pool,
                struct dora_vector* out_vector,
                struct dora_stream_result* out_result,
                struct dora_error* out_error);
    /// Frees source state.
    void (*release)(struct dora_vector_source* source);
};

//--------------------------------------
// Construction
//--------------------------------------

/// Calls source->release on failure or reader release, if present.
int dora_reader_init(struct dora_reader* reader, struct dora_pool pool, struct dora_vector_source* source);

//--------------------------------------
// Reading
//--------------------------------------

/// Publishes owning batches that survive later reads and reader release.
int dora_reader_next(struct dora_reader* reader,
                     struct dora_vector* out_vector,
                     struct dora_stream_result* out_result,
                     struct dora_error* out_error);

//--------------------------------------
// Lifecycle
//--------------------------------------

int dora_reader_move(struct dora_reader* in_out_source, struct dora_reader* out_destination);

void dora_reader_release(struct dora_reader* reader);

bool dora_reader_is_valid(const struct dora_reader* reader);

//------------------------------------------------------------------------------
//
// Writer
//
//------------------------------------------------------------------------------

//--------------------------------------
// Sink Capabilities
//--------------------------------------

/// Pool reset and release return EBUSY during active writer callbacks.
struct dora_vector_sink {
    /// Consumes input rows and reports progress.
    int (*push)(struct dora_vector_sink* sink,
                const struct dora_vector* input,
                struct dora_stream_result* out_result,
                struct dora_error* out_error);
    /// Completes output and reports progress.
    int (*finish)(struct dora_vector_sink* sink, struct dora_stream_result* out_result, struct dora_error* out_error);
    /// Frees sink state.
    void (*release)(struct dora_vector_sink* sink);
};

//--------------------------------------
// Construction
//--------------------------------------

/// Calls sink->release on failure or writer release, if present.
int dora_writer_init(struct dora_writer* writer, struct dora_pool pool, struct dora_vector_sink* sink);

//--------------------------------------
// Writing
//--------------------------------------

/// Reports input consumption; output writes have no rollback.
int dora_writer_push(struct dora_writer* writer,
                     const struct dora_vector* input,
                     struct dora_stream_result* out_result,
                     struct dora_error* out_error);

/// Closes input; retry if BLOCKED; release never calls sink->finish.
int dora_writer_finish(struct dora_writer* writer, struct dora_stream_result* out_result, struct dora_error* out_error);

//--------------------------------------
// Lifecycle
//--------------------------------------

int dora_writer_move(struct dora_writer* in_out_source, struct dora_writer* out_destination);

void dora_writer_release(struct dora_writer* writer);

bool dora_writer_is_valid(const struct dora_writer* writer);

//------------------------------------------------------------------------------
//
// Pipeline
//
//------------------------------------------------------------------------------

//--------------------------------------
// Transform Capabilities
//--------------------------------------

/// Pool reset and release return EBUSY during active pipeline callbacks.
struct dora_vector_transform {
    /// Transforms borrowed input into an owning output.
    int (*apply)(struct dora_vector_transform* transform,
                 struct dora_pool pool,
                 const struct dora_vector* input,
                 bool is_end_of_input,
                 struct dora_vector* out_vector,
                 struct dora_stream_result* out_result,
                 struct dora_error* out_error);
    /// Frees transform state.
    void (*release)(struct dora_vector_transform* transform);
};

//--------------------------------------
// Progress Observations
//--------------------------------------

enum dora_pipeline_progress : uint8_t {
    DORA_PIPELINE_PROGRESS_ADVANCED,
    DORA_PIPELINE_PROGRESS_NEEDS_INPUT,
    DORA_PIPELINE_PROGRESS_HAS_OUTPUT,
    DORA_PIPELINE_PROGRESS_BLOCKED,
    DORA_PIPELINE_PROGRESS_FINISHED,
    DORA_PIPELINE_PROGRESS_CANCELLED,
    DORA_PIPELINE_PROGRESS_FAILED
};

enum dora_pipeline_origin : uint8_t {
    DORA_PIPELINE_ORIGIN_NONE,
    DORA_PIPELINE_ORIGIN_READER,
    DORA_PIPELINE_ORIGIN_TRANSFORM,
    DORA_PIPELINE_ORIGIN_WRITER,
    DORA_PIPELINE_ORIGIN_CALLER,
    DORA_PIPELINE_ORIGIN_CONTRACT
};

struct dora_pipeline_observation {
    /// Pipeline state.
    enum dora_pipeline_progress progress;
    /// Origin of the observation.
    enum dora_pipeline_origin origin;
    /// Operation status.
    int status;
    /// Readiness when blocked.
    struct dora_readiness readiness;
};

//--------------------------------------
// Construction
//--------------------------------------

/// Calls transform release callbacks on failure or pipeline release, if present; moves reader and writer on success.
int dora_pipeline_init(struct dora_pipeline* pipeline,
                       struct dora_pool pool,
                       struct dora_reader* in_out_reader,
                       struct dora_vector_transform* const* transforms,
                       uint32_t transform_count,
                       struct dora_writer* in_out_writer);

//--------------------------------------
// Execution
//--------------------------------------

int dora_pipeline_supply(struct dora_pipeline* pipeline, struct dora_vector* in_out_vector);

int dora_pipeline_end_input(struct dora_pipeline* pipeline);

int dora_pipeline_take(struct dora_pipeline* pipeline, struct dora_vector* out_vector);

int dora_pipeline_step(struct dora_pipeline* pipeline,
                       struct dora_pipeline_observation* out_observation,
                       struct dora_error* out_error);

int dora_pipeline_get_observation(const struct dora_pipeline* pipeline,
                                  struct dora_pipeline_observation* out_observation);

int dora_pipeline_cancel(struct dora_pipeline* pipeline);

int dora_pipeline_fail(struct dora_pipeline* pipeline, int status);

//--------------------------------------
// Lifecycle
//--------------------------------------

int dora_pipeline_move(struct dora_pipeline* in_out_source, struct dora_pipeline* out_destination);

void dora_pipeline_release(struct dora_pipeline* pipeline);

bool dora_pipeline_is_valid(const struct dora_pipeline* pipeline);

#endif
