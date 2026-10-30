#-------------------------------------------------------------------------------
#
# Harness
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Configurations
#---------------------------------------

function(dora_configure_benchmarks configuration)
    string(TOUPPER "${configuration}" configuration_upper)
    string(STRIP "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${configuration_upper}}" flags)
    set(library_options "DORA_KERNEL_BACKEND=${DORA_KERNEL_BACKEND}")
    foreach(definition IN LISTS dora_private_definition_names)
        if(${definition})
            list(APPEND library_options "${definition}=1")
        else()
            list(APPEND library_options "${definition}=0")
        endif()
    endforeach()
    list(JOIN library_options " " library_options)
    dora_quote_string(bench_build_config "${configuration}")
    dora_quote_string(bench_compiler "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
    dora_quote_string(bench_c_flags "${flags}")
    dora_quote_string(bench_library_options "${library_options}")
    configure_file(harness/bench_config.h.in
        "${CMAKE_CURRENT_BINARY_DIR}/harness/${configuration}/bench_config.h" @ONLY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/results/${configuration}")
endfunction()

get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(multi_config)
    foreach(configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
        dora_configure_benchmarks("${configuration}")
    endforeach()
else()
    dora_configure_benchmarks("${CMAKE_BUILD_TYPE}")
endif()

#---------------------------------------
# Library
#---------------------------------------

add_library(dora_bench_harness STATIC harness/bench.c harness/bench_data.c)
target_link_libraries(dora_bench_harness PUBLIC dora::vector PRIVATE m)
target_include_directories(dora_bench_harness PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/harness"
    PRIVATE "${dora_private_include_directory}" "${CMAKE_CURRENT_BINARY_DIR}/harness/$<CONFIG>")
if(NOT APPLE)
    target_compile_definitions(dora_bench_harness PRIVATE _POSIX_C_SOURCE=200809L)
endif()

#-------------------------------------------------------------------------------
#
# Benchmark registration
#
#-------------------------------------------------------------------------------

function(dora_add_benchmark category name)
    cmake_parse_arguments(PARSE_ARGN bench "" "" "SOURCES")
    if(DEFINED bench_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unrecognized arguments for ${category}.${name}: ${bench_UNPARSED_ARGUMENTS}")
    endif()
    if(DEFINED bench_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for ${category}.${name}: ${bench_KEYWORDS_MISSING_VALUES}")
    endif()
    if(NOT bench_SOURCES)
        message(FATAL_ERROR "Benchmark ${category}.${name} requires SOURCES")
    endif()
    set(target "dora_bench_${category}_${name}")
    add_executable(${target} $<TARGET_OBJECTS:dora_vector> $<TARGET_OBJECTS:dora_bench_harness> ${bench_SOURCES})
    target_link_libraries(${target} PRIVATE dora::vector dora_bench_harness)
    target_include_directories(${target} PRIVATE "${dora_private_include_directory}")
    if(BUILD_TESTING)
        add_test(NAME "dora.bench.${category}.${name}.smoke" COMMAND ${target} --smoke)
        set_tests_properties("dora.bench.${category}.${name}.smoke" PROPERTIES LABELS "benchmark")
        add_test(NAME "dora.bench.${category}.${name}" COMMAND ${target}
            --json "${CMAKE_CURRENT_BINARY_DIR}/results/$<CONFIG>/${target}.json")
        set_tests_properties("dora.bench.${category}.${name}"
            PROPERTIES LABELS "benchmark;performance" RUN_SERIAL TRUE)
    endif()
endfunction()

#-------------------------------------------------------------------------------
#
# Benchmarks
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Vector
#---------------------------------------

dora_add_benchmark(vector build
    SOURCES vector/bench_build.c
)

dora_add_benchmark(vector read
    SOURCES vector/bench_read.c
)

dora_add_benchmark(vector validate
    SOURCES vector/bench_validate.c
)

#---------------------------------------
# Compute
#---------------------------------------

dora_add_benchmark(compute aggregate
    SOURCES compute/bench_aggregate.c
)

dora_add_benchmark(compute arithmetic
    SOURCES compute/bench_arithmetic.c
)

dora_add_benchmark(compute cast
    SOURCES compute/bench_cast.c
)

dora_add_benchmark(compute compare
    SOURCES compute/bench_compare.c
)

dora_add_benchmark(compute filter
    SOURCES compute/bench_filter.c
)

dora_add_benchmark(compute hash
    SOURCES compute/bench_hash.c
)

dora_add_benchmark(compute string
    SOURCES compute/bench_string.c
)

#---------------------------------------
# IO
#---------------------------------------

dora_add_benchmark(io text
    SOURCES io/bench_text.c
)

#---------------------------------------
# Execution
#---------------------------------------

dora_add_benchmark(execution stream
    SOURCES execution/bench_stream.c
)

#---------------------------------------
# Interop
#---------------------------------------

dora_add_benchmark(interop arrow
    SOURCES interop/bench_arrow.c
)

#---------------------------------------
# Memory
#---------------------------------------

dora_add_benchmark(memory pool
    SOURCES memory/bench_pool.c
)

#-------------------------------------------------------------------------------
#
# Harness checks
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Native checks
#---------------------------------------

if(BUILD_TESTING)
    add_executable(dora_bench_harness_checks harness/checks/test_harness.c)
    target_link_libraries(dora_bench_harness_checks PRIVATE dora_bench_harness)
    target_include_directories(dora_bench_harness_checks PRIVATE "${dora_private_include_directory}")
    if(NOT APPLE)
        target_compile_definitions(dora_bench_harness_checks PRIVATE _DEFAULT_SOURCE)
    endif()
    add_test(NAME dora.bench.harness COMMAND dora_bench_harness_checks)
    set_tests_properties(dora.bench.harness PROPERTIES LABELS "benchmark")
endif()

#---------------------------------------
# Script checks
#---------------------------------------

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME dora.bench.compare
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/harness/checks/test_compare.py")
    set_tests_properties(dora.bench.compare PROPERTIES LABELS "benchmark")
    add_test(NAME dora.bench.cli
        COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/harness/checks/test_cli.py"
            $<TARGET_FILE:dora_bench_compute_arithmetic> $<TARGET_FILE:dora_bench_harness_checks>)
    set_tests_properties(dora.bench.cli PROPERTIES LABELS "benchmark")
endif()
