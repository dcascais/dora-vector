# dora-vector

dora-vector is a C23 library for columnar data and vectorized
in-memory computation.

It provides explicit memory ownership, CPU kernels, and batch processing with
no third-party runtime dependencies. Execution is single-threaded, with no
internal synchronization; the caller controls threading and asynchronous work.

## Features

- **Data:** scalar and nested types, dictionary and run-end encoding, packed
  validity bitmaps, and extensions including fixed- and variable-shape tensors.
- **Memory:** pointer-free handles, typed buffer views, borrowed slices, retained
  sharing, and pools with byte limits, fixed regions, and custom allocators.
- **Construction:** row appends, bulk appends, and direct buffer writes.
- **Compute:** arithmetic, comparisons, ordering, casts, hashing, selection,
  aggregates, statistics, sequence generation, and Unicode text processing.
- **Streaming:** CSV and JSON readers and writers, schema inference, byte sources
  and sinks, and caller-driven batch pipelines.
- **Interchange:** Apache Arrow C Data, C Stream, and C Device adapters with
  explicit ownership and buffer sharing for compatible representations.

## Example

Add two Int32 vectors elementwise to produce `[7, 6, 8]`:

```c
#include <dora/vector.h>
#include <stdlib.h>

int main() {
    size_t pool_limit_bytes = 256UL * 1024;
    size_t pool_alignment_bytes = 64;
    struct dora_pool pool = {};

    uint64_t count = 3;
    int32_t values_a[] = {3, 5, 2};
    int32_t values_b[] = {4, 1, 6};

    struct dora_logical_type type = {.id = DORA_TYPE_INT32};
    struct dora_vector vector_a = {};
    struct dora_vector vector_b = {};
    struct dora_vector sum = {};

    int status = dora_pool_init(&pool, pool_limit_bytes, pool_alignment_bytes);
    if (status != DORA_OK) {
        return EXIT_FAILURE;
    }

    status = dora_vector_init_copy_i32(&vector_a, pool, &type, values_a, nullptr, count);
    if (status == DORA_OK) {
        status = dora_vector_init_copy_i32(&vector_b, pool, &type, values_b, nullptr, count);
    }
    if (status == DORA_OK) {
        status = dora_compute_add(&vector_a, &vector_b, DORA_FAILURE_POLICY_ERROR, &sum);
    }

    dora_vector_release(&sum);
    dora_vector_release(&vector_b);
    dora_vector_release(&vector_a);
    int release_status = dora_pool_release(&pool);

    return status == DORA_OK && release_status == DORA_OK ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

## Getting started

The presets require CMake 4.4 or newer, Ninja, and a C23 compiler.
Examples also require clang-tidy.
Select the compiler through `CC`, `CMAKE_C_COMPILER`, or a toolchain file when
configuring a fresh build directory.

```sh
cmake --preset release
cmake --build --preset release
cmake --install build-release --prefix /path/to/install
```

Examples are opt-in; enable with `-DDORA_BUILD_EXAMPLES=ON`.

Building a build directory's `dora_clangd_select` target writes that build's
generated clangd configuration to the source root's `.clangd`, selecting the
build's compilation database for clangd.

In your project's `CMakeLists.txt`:

```cmake
find_package(dora_vector CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE dora::vector)
```

Configure your project with `-DCMAKE_PREFIX_PATH=/path/to/install`.

Or use a source checkout:

```cmake
add_subdirectory(dora-vector)
target_link_libraries(your_target PRIVATE dora::vector)
```

## Documentation

- [Public API headers](include/dora/vector/) and [umbrella header](include/dora/vector.h)
- [Architecture](docs/ARCHITECTURE.md)
- [Arrow adapter API](include/dora/vector/interop/arrow/arrow.h)
- [Examples](examples/README.md)

## Contributing

Open issues with a minimal reproducer for bugs or a use case for feature requests.
Please do not submit pull requests.
