#-------------------------------------------------------------------------------
#
# Example targets
#
#-------------------------------------------------------------------------------

configure_file(data/cars.json data/cars.json COPYONLY)
configure_file(data/flights.csv data/flights.csv COPYONLY)

add_custom_target(dora_examples)
if(NOT CMAKE_CROSSCOMPILING OR CMAKE_CROSSCOMPILING_EMULATOR)
    add_custom_target(run_dora_examples)
endif()

function(dora_add_example source)
    cmake_path(GET source STEM name)
    set(target "dora_example_${name}")
    add_executable("${target}" "${source}")
    target_link_libraries("${target}" PRIVATE dora::vector)
    set_target_properties("${target}" PROPERTIES
        DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_dependencies(dora_examples "${target}")

    if(TARGET run_dora_examples)
        add_custom_target("run_${target}"
            COMMAND "${target}" ${ARGN}
            VERBATIM
            USES_TERMINAL)
        add_dependencies(run_dora_examples "run_${target}")
    endif()
endfunction()

#-------------------------------------------------------------------------------
#
# Examples
#
#-------------------------------------------------------------------------------

dora_add_example(build/nullable_column.c)
dora_add_example(build/record_batch.c)
dora_add_example(build/string_dictionary.c)
dora_add_example(build/list_column.c)

dora_add_example(compute/filter_rows.c)
dora_add_example(compute/derive_column.c)
dora_add_example(compute/cast_values.c)

dora_add_example(ownership/retain_slice.c)

dora_add_example(io/csv_summary.c "${CMAKE_CURRENT_BINARY_DIR}/data/flights.csv")
dora_add_example(io/json_summary.c "${CMAKE_CURRENT_BINARY_DIR}/data/cars.json")
dora_add_example(io/write_csv.c)

dora_add_example(streaming/batch_reader.c)
dora_add_example(streaming/filter_pipeline.c)

dora_add_example(interop/arrow_array.c)
dora_add_example(interop/arrow_stream.c)
dora_add_example(interop/json_extension.c)

dora_add_example(statistics/prune_batches.c)
dora_add_example(statistics/merge_statistics.c)
