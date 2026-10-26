#-------------------------------------------------------------------------------
#
# Project initialization
#
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 4.4)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    if(NOT DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
        set(CACHE{CMAKE_EXPORT_COMPILE_COMMANDS} TYPE BOOL
            HELP "Generate the compilation database"
            VALUE ON)
    endif()
endif()

project(dora_vector VERSION 0.2.0 LANGUAGES C)

#---------------------------------------
# Compiler requirements
#---------------------------------------

if(DEFINED DORA_EXPECTED_C_COMPILER_ID)
    if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "${DORA_EXPECTED_C_COMPILER_ID}")
        message(FATAL_ERROR
            "Expected ${DORA_EXPECTED_C_COMPILER_ID}, detected ${CMAKE_C_COMPILER_ID}. "
            "Select the required C compiler in a fresh build directory.")
    endif()
endif()

if(DEFINED DORA_EXPECTED_C_COMPILER_VERSION_MAJOR)
    string(REGEX MATCH "^[0-9]+" dora_c_compiler_version_major "${CMAKE_C_COMPILER_VERSION}")
    if(NOT "${dora_c_compiler_version_major}" STREQUAL "${DORA_EXPECTED_C_COMPILER_VERSION_MAJOR}")
        message(FATAL_ERROR
            "Expected C compiler major version ${DORA_EXPECTED_C_COMPILER_VERSION_MAJOR}, detected ${CMAKE_C_COMPILER_VERSION}. "
            "Select the required C compiler in a fresh build directory.")
    endif()
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

#-------------------------------------------------------------------------------
#
# Build configuration
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Developer targets
#---------------------------------------

option(DORA_BUILD_EXAMPLES "Build the examples" OFF)

set(dora_developer_build OFF)
if(DORA_BUILD_EXAMPLES)
    set(dora_developer_build ON)
endif()

#---------------------------------------
# Implementation options
#---------------------------------------

set(CACHE{DORA_KERNEL_BACKEND} TYPE STRING
    HELP "Select the kernel backend"
    VALUE cpu)
option(DORA_KERNEL_USE_NEON "Enable NEON kernel implementations" ON)

#-------------------------------------------------------------------------------
#
# Compiler policy
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Compiler diagnostics
#---------------------------------------

if(dora_developer_build)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

set(dora_translation_unit_warnings -Wunused-macros)
if(CMAKE_C_COMPILER_ID MATCHES "^(Clang|AppleClang|GNU)$")
    add_compile_options(-Wall -Wextra -Wpedantic ${dora_translation_unit_warnings})
elseif(dora_developer_build)
    message(FATAL_ERROR "No strict developer warning policy for ${CMAKE_C_COMPILER_ID}")
endif()

#---------------------------------------
# Static analysis
#---------------------------------------

if(dora_developer_build)
    if(NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
        message(FATAL_ERROR "Dora developer targets require a Ninja or Makefile generator for clang-tidy")
    endif()
    if(NOT DEFINED CMAKE_C_CLANG_TIDY)
        cmake_path(GET CMAKE_C_COMPILER PARENT_PATH dora_c_compiler_directory)
        find_program(DORA_CLANG_TIDY_EXECUTABLE NAMES clang-tidy
            HINTS "${dora_c_compiler_directory}"
            OPTIONAL NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        find_program(DORA_CLANG_TIDY_EXECUTABLE NAMES clang-tidy
            REQUIRED NO_CMAKE_FIND_ROOT_PATH)
        set(CMAKE_C_CLANG_TIDY
            "${DORA_CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")
        execute_process(
            COMMAND ${CMAKE_C_CLANG_TIDY} --verify-config
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMAND_ERROR_IS_FATAL ANY)
    endif()
endif()

if(dora_developer_build AND NOT CMAKE_C_CLANG_TIDY)
    message(FATAL_ERROR "Dora examples require native clang-tidy integration")
endif()

#-------------------------------------------------------------------------------
#
# Library
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Backend sources
#---------------------------------------

set(DORA_KERNEL_BACKEND_SOURCES_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kernel/${DORA_KERNEL_BACKEND}/sources.cmake")
if(NOT EXISTS "${DORA_KERNEL_BACKEND_SOURCES_FILE}")
    message(FATAL_ERROR "Unknown kernel backend '${DORA_KERNEL_BACKEND}': ${DORA_KERNEL_BACKEND_SOURCES_FILE} is missing")
endif()
include("${DORA_KERNEL_BACKEND_SOURCES_FILE}")

#---------------------------------------
# Implementation
#---------------------------------------

add_library(dora_vector STATIC
    src/bitmap/bitmap.c
    src/builder/append.c
    src/builder/builder.c
    src/compute/arithmetic.c
    src/compute/boolean.c
    src/compute/bytes.c
    src/compute/bytes_source.c
    src/compute/cast.c
    src/compute/compare.c
    src/compute/generate.c
    src/compute/hash.c
    src/compute/output.c
    src/compute/peel.c
    src/compute/reduce.c
    src/compute/reduce_join.c
    src/compute/select.c
    src/compute/utf8_binary.c
    src/compute/utf8_like.c
    src/compute/utf8_list.c
    src/compute/utf8_literal.c
    src/compute/utf8_scalar.c
    src/compute/utf8_transform.c
    src/extension/bool8.c
    src/extension/extension.c
    src/extension/json.c
    src/extension/opaque.c
    src/extension/tensor.c
    src/extension/timestamp_with_offset.c
    src/extension/uuid.c
    src/extension/variant.c
    src/hash/keyset.c
    src/identity/claim.c
    src/identity/directory.c
    src/interop/arrow/array_import.c
    src/interop/arrow/export.c
    src/interop/arrow/schema_import.c
    src/interop/arrow/statistics.c
    src/interop/arrow/stream.c
    src/interop/arrow/traversal.c
    src/io/byte.c
    src/io/csv/options.c
    src/io/csv/reader.c
    src/io/csv/writer.c
    src/io/file.c
    src/io/inferred_columns.c
    src/io/json/options.c
    src/io/json/reader.c
    src/io/json/writer.c
    src/io/schema_inference.c
    src/io/text_reader.c
    src/io/text_writer.c
    src/json/json.c
    src/memory/allocation.c
    src/memory/allocator.c
    src/memory/pool.c
    src/memory/registry.c
    src/memory/scratch.c
    src/numeric/arithmetic.c
    src/numeric/decimal.c
    src/numeric/decimal_float.c
    src/numeric/limbs.c
    src/schema/access.c
    src/schema/cache.c
    src/schema/schema.c
    src/schema/walk.c
    src/statistics/accumulate.c
    src/statistics/bounds.c
    src/statistics/key.c
    src/statistics/merge.c
    src/statistics/prune.c
    src/statistics/statistics.c
    src/stream/pipeline.c
    src/stream/reader.c
    src/stream/writer.c
    src/temporal/temporal.c
    src/types/text.c
    src/types/types.c
    src/utf8/batch.c
    src/utf8/case.c
    src/utf8/encoding.c
    src/utf8/like.c
    src/utf8/span.c
    src/vector/admit.c
    src/vector/bind.c
    src/vector/copy.c
    src/vector/equality.c
    src/vector/init.c
    src/vector/key.c
    src/vector/lifecycle.c
    src/vector/node_topology.c
    src/vector/operand.c
    src/vector/read.c
    src/vector/validate.c
    src/vector/wrap.c
    src/view/view.c
)
add_library(dora::vector ALIAS dora_vector)
set_target_properties(dora_vector PROPERTIES EXPORT_NAME vector)
target_sources(dora_vector PRIVATE ${DORA_KERNEL_BACKEND_SOURCES})

#---------------------------------------
# Public headers
#---------------------------------------

target_sources(dora_vector PUBLIC
    FILE_SET public_headers TYPE HEADERS
    BASE_DIRS include
    FILES
        include/dora/vector/bitmap.h
        include/dora/vector/builder.h
        include/dora/vector/compute.h
        include/dora/vector/device.h
        include/dora/vector/extension.h
        include/dora/vector/handle.h
        include/dora/vector/interop/arrow/arrow.h
        include/dora/vector/io/byte.h
        include/dora/vector/io/csv.h
        include/dora/vector/io/file.h
        include/dora/vector/io/json.h
        include/dora/vector/io/schema_inference.h
        include/dora/vector/memory.h
        include/dora/vector/schema.h
        include/dora/vector/statistics.h
        include/dora/vector/status.h
        include/dora/vector/stream.h
        include/dora/vector/types.h
        include/dora/vector/vector.h
        include/dora/vector/view.h
        include/dora/vector.h
    FILE_SET arrow_abi_headers TYPE HEADERS
    BASE_DIRS include
    FILES
        include/dora/vector/interop/arrow/abi.h
)

#---------------------------------------
# Private headers
#---------------------------------------

set(dora_private_include_directory "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_sources(dora_vector PRIVATE
    FILE_SET private_headers TYPE HEADERS
    BASE_DIRS "${dora_private_include_directory}"
    FILES
        src/bitmap/bitmap_internal.h
        src/builder/builder_internal.h
        src/builder/state.h
        src/compute/bytes_source.h
        src/compute/compute_internal.h
        src/compute/output.h
        src/compute/peel.h
        src/compute/proof.h
        src/extension/extension_internal.h
        src/extension/storage.h
        src/extension/variant.h
        src/hash/hash.h
        src/hash/keyset.h
        src/identity/identity.h
        src/interop/arrow/arrow_internal.h
        src/io/csv/csv_inference.h
        src/io/csv/csv_internal.h
        src/io/inferred_columns.h
        src/io/json/json_inference.h
        src/io/json/json_internal.h
        src/io/schema_inference_internal.h
        src/io/text.h
        src/io/text_reader.h
        src/io/text_writer.h
        src/json/json.h
        src/kernel/binding/elementwise.h
        src/kernel/binding/read.h
        src/kernel/binding/reduce.h
        src/kernel/binding/request.h
        src/kernel/binding/validate.h
        src/kernel/cpu/packet_bits.h
        src/kernel/decode.h
        src/kernel/generate.h
        src/kernel/hash_cache.h
        src/kernel/packet.h
        src/kernel/selection.h
        src/memory/memory_internal.h
        src/memory/registry.h
        src/memory/scratch.h
        src/numeric/arithmetic.h
        src/numeric/decimal.h
        src/numeric/decimal_float.h
        src/numeric/failure.h
        src/numeric/float.h
        src/numeric/limbs.h
        src/schema/schema_internal.h
        src/statistics/bounds.h
        src/statistics/statistics_internal.h
        src/status/status_internal.h
        src/stream/stream_internal.h
        src/temporal/interval.h
        src/temporal/temporal.h
        src/types/array_count.h
        src/types/load.h
        src/types/order.h
        src/types/text.h
        src/types/types_internal.h
        src/utf8/ascii.h
        src/utf8/batch.h
        src/utf8/encoding.h
        src/utf8/input.h
        src/utf8/like.h
        src/utf8/span.h
        src/utf8/transform.h
        src/utf8/utf8.h
        src/vector/image.h
        src/vector/key.h
        src/vector/node_topology.h
        src/vector/operand.h
        src/vector/vector_internal.h
        src/view/view_internal.h
)

#---------------------------------------
# Target requirements
#---------------------------------------

target_compile_features(dora_vector PUBLIC c_std_23)
set(dora_private_definition_names DORA_KERNEL_USE_NEON)
foreach(definition IN LISTS dora_private_definition_names)
    target_compile_definitions(dora_vector PRIVATE "${definition}=$<BOOL:${${definition}}>")
endforeach()

target_link_libraries(dora_vector PRIVATE m)

#---------------------------------------
# Source-specific options
#---------------------------------------

set_source_files_properties(src/kernel/cpu/arithmetic.c
    PROPERTIES COMPILE_OPTIONS "$<$<C_COMPILER_ID:Clang,AppleClang,GNU>:-ffp-contract=off>")

#---------------------------------------
# Header verification
#---------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    set_target_properties(dora_vector PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS ON
        INTERFACE_HEADER_SETS_TO_VERIFY public_headers
        VERIFY_PRIVATE_HEADER_SETS ON
        PRIVATE_HEADER_SETS_TO_VERIFY private_headers)
endif()

#-------------------------------------------------------------------------------
#
# Installation
#
#-------------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/dora_vectorConfigVersion.cmake"
        COMPATIBILITY SameMinorVersion)
    configure_package_config_file(
        cmake/dora_vector_config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/dora_vectorConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dora_vector")
    install(FILES LICENSE NOTICE
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/dora_vector")
    install(TARGETS dora_vector
        EXPORT dora_vector_targets
        FILE_SET public_headers
        FILE_SET arrow_abi_headers)
    install(EXPORT dora_vector_targets
        NAMESPACE dora::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dora_vector")
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/dora_vectorConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/dora_vectorConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dora_vector")
endif()

#-------------------------------------------------------------------------------
#
# Developer targets
#
#-------------------------------------------------------------------------------

if(DORA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#-------------------------------------------------------------------------------
#
# Tooling metadata
#
#-------------------------------------------------------------------------------

#---------------------------------------
# Build metadata and clangd
#---------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    cmake_file_api(
        QUERY
        API_VERSION 1
        CODEMODEL 2
        CACHE 2
        TOOLCHAINS 1)

    set(dora_clangd_binary_directory "${PROJECT_BINARY_DIR}")
    set(dora_clangd_public_flags "-I${dora_private_include_directory}")
    foreach(definition IN LISTS dora_private_definition_names)
        list(APPEND dora_clangd_public_flags "-D${definition}=*")
    endforeach()
    foreach(variable IN ITEMS dora_clangd_binary_directory dora_translation_unit_warnings dora_clangd_public_flags)
        set(escaped_values)
        foreach(value IN LISTS ${variable})
            string(REPLACE "\\" "\\\\" value "${value}")
            string(REPLACE "\"" "\\\"" value "${value}")
            string(REPLACE "\n" "\\n" value "${value}")
            string(REPLACE "\r" "\\r" value "${value}")
            string(REPLACE "\t" "\\t" value "${value}")
            list(APPEND escaped_values "\"${value}\"")
        endforeach()
        list(JOIN escaped_values ", " ${variable}_yaml)
    endforeach()
    file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/dora-clangd.yaml" CONTENT [=[
CompileFlags:
  CompilationDatabase: @dora_clangd_binary_directory_yaml@
---
If:
  PathMatch: '.*\.h'
CompileFlags:
  Remove: [@dora_translation_unit_warnings_yaml@]
---
If:
  PathMatch: 'include/.*\.h'
CompileFlags:
  Remove: [@dora_clangd_public_flags_yaml@]
]=] @ONLY)
    file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/dora-clangd-select.cmake" CONTENT [=[
if(NOT "@CMAKE_EXPORT_COMPILE_COMMANDS@" OR NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/compile_commands.json")
    message(FATAL_ERROR "dora_clangd_select requires compile_commands.json; configure with CMAKE_EXPORT_COMPILE_COMMANDS=ON")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_LIST_DIR}/dora-clangd.yaml" "${DORA_CLANGD_SOURCE_DIR}/.clangd"
    COMMAND_ERROR_IS_FATAL ANY)
]=] @ONLY)
    add_custom_target(dora_clangd_select
        COMMAND "${CMAKE_COMMAND}" "-DDORA_CLANGD_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            -P "${PROJECT_BINARY_DIR}/dora-clangd-select.cmake"
        VERBATIM)
endif()
