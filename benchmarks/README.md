# Benchmarks

Timed benchmarks of dora-vector operations, one executable per feature set.

## Requirements

- CMake 4.4+
- Clang, AppleClang, or GCC with C23 support
- clang-tidy 23+
- Ninja or Make
- Python 3

## Run

```sh
cmake --workflow --preset benchmarks
ctest --preset benchmarks-check
build-benchmarks/benchmarks/dora_bench_compute_arithmetic --filter '^add/'
```

- The workflow builds, checks, and times every case, and writes
  `build-benchmarks/benchmarks/results/<config>/<executable>.json`.
- `benchmarks-check` runs the checks only.

## Options

- `--list`: prints case names.
- `--filter REGEX`: runs cases matching a POSIX extended regular expression.
- `--repetitions N`: repetitions per case. Default: 5.
- `--min-time S`: minimum seconds of one calibration run. Default: 0.1.
- `--min-time Nx`: exactly N iterations per repetition, after one untimed
  warm-up repetition.
- `--json PATH`: writes results to `PATH`.
- `--pool host|fixed[:BYTES]`: pool backend. Default: `host`; `fixed`:
  256 MiB.
- `--smoke`: runs and checks each case once, untimed.

## Measurement

- Each case checks its result once, untimed, then times every repetition.
- Each epoch times K iterations between two clock reads; K doubles while an
  epoch takes under 4 µs.
- **Timed:** the operation's calls.
- **Untimed:** fixtures, checks, calibration, warm-up, releasing results,
  and checking statuses. Pool allocate-release cases and the pipeline's
  per-batch intermediates time their release.
- A case fails on an error status or a 0 ns repetition.

## Results

- **`Min`, `Mean`, `Max`:** wall time per iteration over the repetitions.
  `Min` sets the rates and `compare.py`.
- **`Coef. var.`:** standard deviation over mean of the repetition times.
- **`Items/s`, `Bytes/s`:** per-iteration counts over `Min`.
- **`Ctx switches`, `Faults`:** totals over the repetitions, untimed work
  included, from `getrusage`.

The JSON file is the run's machine-readable record and the input to
`compare.py`.

Items are input rows and bytes are all input buffers, except:

| Cases | Items | Bytes |
| --- | --- | --- |
| Arrow `export`, `export_import/int64`, `slice` | 1 call | |
| Stream drain, Arrow `stream/export_import` | Batches | |
| Inference | Sampled records | Header and sampled records |
| `count` | Rows | |
| String `scan`, `read` | Rows | |
| `filter` | Rows | Mask |
| `take` | Rows | Indices and selected rows |
| Builders | Rows | Values written |
| Text reads, CSV pipeline | Rows | Input text |
| Text writes | Rows | Input batches |
| `allocate_release` | 1 call | |
| `allocate_touch_release` | 1 call | Allocation size |
| `grow` | | 16 MiB |

## Compare

```sh
python3 benchmarks/harness/compare.py BASE NEW --threshold 0.05
```

- `BASE` and `NEW` are JSON files or directories; cases match by executable
  and name, and each side's minimum is compared.
- Prints regressions above the threshold, build-context differences, and failed
  rows; exits 2 on invalid input.
- Results vary between processes; pass several runs per side.

## Catalog

Each set builds the executable `dora_bench_<set>`. In case names, `nulls:F` is
the null fraction, `policy:trust` builds skip UTF-8 validation, and
`bounds:proven` and `bounds:unproven` inputs carry exact min/max statistics
that do or do not let the operation skip overflow checks.

- `vector_build`: per-value and bulk builds.
- `vector_read`: view scans, per-value reads, decode, encode, slice, equality.
- `vector_validate`: content validation.
- `compute_aggregate`: sum, min, max, count.
- `compute_arithmetic`: add, multiply, divide.
- `compute_cast`: numeric, temporal, text, and decimal casts.
- `compute_compare`: comparisons and boolean logic.
- `compute_filter`: filter and take.
- `compute_hash`: hashing.
- `compute_string`: string functions.
- `io_text`: CSV and NDJSON read, write, and inference.
- `execution_stream`: stream drain and a CSV filter-sum pipeline.
- `interop_arrow`: Arrow C export and import.
- `memory_pool`: allocation and growth.

## Definition templates

```c
BENCH_DEFINE_OPERATION(name, slot_type, compute_function, release_function)
BENCH_DEFINE_STATUS_OPERATION(name, compute_function)
```

Each defines `static const struct bench_operation name` at file scope; invoke
without a trailing semicolon. `bench_detail_` names are reserved.

- `compute_function`: `int (const T* context, slot_type* slot)`, or
  `int (const T* context)` for status operations; called once per iteration
  with the `bench_measure` context.
- `slot_type`: alignment at most `alignof(max_align_t)`; slots start zeroed.
- `release_function`: `void (void* slots, size_t count)` or `nullptr`; untimed.

Undefine each local `BENCH_<SET>_DEFINE_*` template after its last use.
