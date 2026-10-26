# Dora Vector examples

Self-contained examples showing column construction,
computation, file I/O, batch processing, statistics, and Arrow data exchange.

## Build and run

Requires the [build prerequisites](../README.md#getting-started).
Commands run from the repository root:

```sh
cmake --preset default -DDORA_BUILD_EXAMPLES=ON
cmake --build --preset default --target run_dora_example_nullable_column
```

Examples are disabled by default. Each source has a `dora_example_<name>` build
target and a `run_dora_example_<name>` run target, where `<name>` is the filename
without `.c`. Aggregate targets build or run all examples:

```sh
cmake --build --preset default --target dora_examples
cmake --build --preset default --target run_dora_examples
```

Build targets do not execute examples. Cross builds require
`CMAKE_CROSSCOMPILING_EMULATOR` for run targets. Each executable links to
`dora::vector`; no Arrow, DuckDB, Velox, or Polars installation is required.

## Column construction

| Source | Input and result | Output |
| --- | --- | --- |
| [Nullable column](build/nullable_column.c) | Appends Int32 values and validity flags. A null validity pointer marks all rows valid. | `18`, `null`, `21`, `24`, `26`, with row labels |
| [Record batch](build/record_batch.c) | Appends rows with named Int32 and UTF-8 fields. Child handles borrow batch storage. | `101,Santiago`; `102,Lisbon`; `103,null` |
| [String dictionary](build/string_dictionary.c) | Constructs a dictionary vector from labels and integer codes. | `paid`, `pending`, `paid`, `null` |
| [List column](build/list_column.c) | Constructs lists from flattened items, row lengths, and validity flags. Empty and null lists remain distinct. | `[7, 8]`, `[]`, `null`, `[9]` |

## Column operations

| Source | Input and result | Output |
| --- | --- | --- |
| [Filter rows](compute/filter_rows.c) | Selects records by a threshold and projects order IDs. Null comparison operands do not match. | `order_id=102`; `order_id=104` |
| [Derive a column](compute/derive_column.c) | Computes speeds from distances and positive durations, then their arithmetic mean. Null speeds are excluded. | `observed=3 mean_km_per_hour=23.00` |
| [Cast values](compute/cast_values.c) | Narrows Int32 to Int8 with `DORA_FAILURE_POLICY_NULL`. Out-of-range values and input nulls produce nulls. | `42`, `null`, `-7`, `null` |
| [Retain a slice](ownership/retain_slice.c) | Retains shared storage and copies independent storage. Both owners remain valid after the source is released. | `retained=200 copied=200`; `retained=300 copied=300` |

## File I/O

| Source | Input | Output with the supplied data |
| --- | --- | --- |
| [CSV summary](io/csv_summary.c) | Header followed by `origin` (UTF-8), `destination` (UTF-8), and `count` (Int64), in that order. | `routes=5366 observed=5366 flights=7009725` |
| [JSON summary](io/json_summary.c) | JSON array of objects with the field names and types declared in the source. | `cars=406 observed_mpg=398 mean_mpg=23.51` |
| [CSV output](io/write_csv.c) | Three records defined in the source. | Header `id,note`, a quoted `fragile, glass` field, and a `NULL` field |

The CSV and JSON run targets supply copies of the [data files](data/).
`cars.json` is the StatLib cars dataset (1982), as distributed by
[vega-datasets](https://github.com/vega/vega-datasets).
Both readers accept a file path and process 128-row batches through end-of-input.
Input files must match the declared schema.

```sh
cmake --build --preset default --target run_dora_example_csv_summary
cmake --build --preset default --target run_dora_example_json_summary

./build/examples/dora_example_csv_summary "/path with spaces/routes.csv"
./build/examples/dora_example_json_summary "/path with spaces/cars.json"
./build/examples/dora_example_write_csv > /tmp/orders.csv
```

The executable paths use the `default` preset's build directory. Relative input
paths are resolved from the caller's working directory. CSV output goes to
standard output. File readers return failure for unreadable or malformed input.

CSV totals must fit Int64 within each batch and across batches; overflow produces
failure. JSON means use floating-point arithmetic and exclude null values.
Empty and all-null inputs produce `mean_mpg=null`.

## Batch processing and Arrow data exchange

| Source | Input and result | Output |
| --- | --- | --- |
| [Batch reader](streaming/batch_reader.c) | Publishes owned batches through a synchronous reader callback. | Readings `0`, `1`, `10`, `11`, `20`, `21` |
| [Filter pipeline](streaming/filter_pipeline.c) | Removes nulls from caller-supplied batches. Supply and take transfer ownership. | `reading=18`; `reading=24` |
| [Arrow array](interop/arrow_array.c) | Exports a timestamp column through the Arrow C Data Interface and imports it by moving ownership. | `Arrow format=tsu:UTC rows=3`, then two microsecond timestamps and a null |
| [Arrow stream](interop/arrow_stream.c) | Transfers a record-batch reader through the Arrow C Stream Interface. | `received=10`; `received=20`; `received=30` |
| [JSON extension](interop/json_extension.c) | Exports UTF-8 documents with `arrow.json` extension metadata. | `extension=arrow.json storage=u rows=2` |

The Arrow examples exchange data between Dora producers and consumers in one
process. Import limits bound schema depth, node and edge counts, coordinate
work, and output bytes.

Transforms must mark consumed input explicitly. `is_end_of_input` can accompany
a final non-null input. The examples use synchronous sources and return `EAGAIN`
on `BLOCKED`. Retrying a blocked operation requires caller-managed readiness.

## Batch statistics

| Source | Input and result | Output |
| --- | --- | --- |
| [Prune batches](statistics/prune_batches.c) | Excludes batches whose bounds cannot satisfy the predicate. | `batch=0 skipped`; `batch=1 matches=2` |
| [Merge statistics](statistics/merge_statistics.c) | Accumulates row counts, null counts, and bounds across released input batches. | `rows=6 nulls=2 min=12 max=30` |

Statistics computation scans the input. Pruning reads stored statistics.
`ALL_MATCH` applies only to non-null values. The merge example requires at least
one observed value for min/max; its helper rejects null statistics.

## Ownership and errors

- Owning handles require explicit release. C struct assignment does not retain
  ownership. Pools must outlive their owners, including exported arrays and streams.
- Child vectors, slices, views, and byte spans borrow storage. Their owners must
  remain alive. Retaining a vector extends the storage lifetime.
- Arrow base structures require release through their live callbacks. Successful
  move import transfers array ownership and clears the source release callback.
- Null values have unspecified payloads. Returned byte spans need not be
  NUL-terminated.
- Programs report errors on standard error and return `EXIT_FAILURE`. Initialized
  owners are released before the pool.
