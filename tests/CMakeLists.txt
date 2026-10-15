add_executable(tinyinfer_tests scaffold_test.cpp)
target_link_libraries(tinyinfer_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tests COMMAND tinyinfer_tests)

add_executable(tinyinfer_phase0_tests phase0_test.cpp)
target_link_libraries(tinyinfer_phase0_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_phase0_tests COMMAND tinyinfer_phase0_tests)

add_executable(tinyinfer_tensor_index_tests tensor_index_test.cpp)
target_link_libraries(tinyinfer_tensor_index_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_index_tests COMMAND tinyinfer_tensor_index_tests)

add_executable(tinyinfer_tensor_value_tests tensor_value_test.cpp)
target_link_libraries(tinyinfer_tensor_value_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_value_tests COMMAND tinyinfer_tensor_value_tests)

add_executable(tinyinfer_tensor_reshape_tests tensor_reshape_test.cpp)
target_link_libraries(tinyinfer_tensor_reshape_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_reshape_tests COMMAND tinyinfer_tensor_reshape_tests)

add_executable(tinyinfer_tensor_transpose_tests tensor_transpose_test.cpp)
target_link_libraries(tinyinfer_tensor_transpose_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_transpose_tests COMMAND tinyinfer_tensor_transpose_tests)

add_executable(tinyinfer_tensor_contiguous_tests tensor_contiguous_test.cpp)
target_link_libraries(tinyinfer_tensor_contiguous_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_contiguous_tests COMMAND tinyinfer_tensor_contiguous_tests)

add_executable(tinyinfer_tensor_dtype_tests tensor_dtype_test.cpp)
target_link_libraries(tinyinfer_tensor_dtype_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_dtype_tests COMMAND tinyinfer_tensor_dtype_tests)

add_executable(tinyinfer_allocator_tests allocator_test.cpp)
target_link_libraries(tinyinfer_allocator_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_allocator_tests COMMAND tinyinfer_allocator_tests)

add_executable(tinyinfer_arena_allocator_tests arena_allocator_test.cpp)
target_link_libraries(tinyinfer_arena_allocator_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_arena_allocator_tests COMMAND tinyinfer_arena_allocator_tests)

add_executable(tinyinfer_buffer_storage_tests buffer_storage_test.cpp)
target_link_libraries(tinyinfer_buffer_storage_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_buffer_storage_tests COMMAND tinyinfer_buffer_storage_tests)

add_executable(tinyinfer_tensor_view_tests tensor_view_test.cpp)
target_link_libraries(tinyinfer_tensor_view_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_view_tests COMMAND tinyinfer_tensor_view_tests)

add_executable(tinyinfer_tensor_slice_tests tensor_slice_test.cpp)
target_link_libraries(tinyinfer_tensor_slice_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_slice_tests COMMAND tinyinfer_tensor_slice_tests)

add_executable(tinyinfer_tensor_layout_tests tensor_layout_test.cpp)
target_link_libraries(tinyinfer_tensor_layout_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_layout_tests COMMAND tinyinfer_tensor_layout_tests)

add_executable(tinyinfer_tensor_iterator_tests tensor_iterator_test.cpp)
target_link_libraries(tinyinfer_tensor_iterator_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_tensor_iterator_tests COMMAND tinyinfer_tensor_iterator_tests)

add_executable(tinyinfer_kernel_runner_tests kernel_runner_test.cpp)
target_link_libraries(tinyinfer_kernel_runner_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_kernel_runner_tests COMMAND tinyinfer_kernel_runner_tests)

add_executable(tinyinfer_elementwise_ops_tests elementwise_ops_test.cpp)
target_link_libraries(tinyinfer_elementwise_ops_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_elementwise_ops_tests COMMAND tinyinfer_elementwise_ops_tests)

add_executable(tinyinfer_reduction_normalization_tests reduction_normalization_test.cpp)
target_link_libraries(tinyinfer_reduction_normalization_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_reduction_normalization_tests COMMAND tinyinfer_reduction_normalization_tests)

add_executable(tinyinfer_matmul_kernel_tests matmul_kernel_test.cpp)
target_link_libraries(tinyinfer_matmul_kernel_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_matmul_kernel_tests COMMAND tinyinfer_matmul_kernel_tests)

add_executable(tinyinfer_graph_data_model_tests graph_data_model_test.cpp)
target_link_libraries(tinyinfer_graph_data_model_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_data_model_tests COMMAND tinyinfer_graph_data_model_tests)

add_executable(tinyinfer_graph_analysis_tests graph_analysis_test.cpp)
target_link_libraries(tinyinfer_graph_analysis_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_analysis_tests COMMAND tinyinfer_graph_analysis_tests)

add_executable(tinyinfer_operator_schema_tests operator_schema_test.cpp)
target_link_libraries(tinyinfer_operator_schema_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_operator_schema_tests COMMAND tinyinfer_operator_schema_tests)

add_executable(tinyinfer_operator_coverage_tests operator_coverage_test.cpp)
target_link_libraries(tinyinfer_operator_coverage_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_operator_coverage_tests COMMAND tinyinfer_operator_coverage_tests)

add_executable(tinyinfer_graph_validation_tests graph_validation_test.cpp)
target_link_libraries(tinyinfer_graph_validation_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_validation_tests COMMAND tinyinfer_graph_validation_tests)

add_executable(tinyinfer_graph_optimizer_tests graph_optimizer_test.cpp)
target_link_libraries(tinyinfer_graph_optimizer_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_optimizer_tests COMMAND tinyinfer_graph_optimizer_tests)

add_executable(tinyinfer_graph_optimization_passes_tests graph_optimization_passes_test.cpp)
target_link_libraries(tinyinfer_graph_optimization_passes_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_optimization_passes_tests COMMAND tinyinfer_graph_optimization_passes_tests)

add_executable(tinyinfer_matmul_add_canonicalization_tests matmul_add_canonicalization_test.cpp)
target_link_libraries(tinyinfer_matmul_add_canonicalization_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_matmul_add_canonicalization_tests COMMAND tinyinfer_matmul_add_canonicalization_tests)

add_executable(tinyinfer_gemm_activation_fusion_tests gemm_activation_fusion_test.cpp)
target_link_libraries(tinyinfer_gemm_activation_fusion_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_gemm_activation_fusion_tests COMMAND tinyinfer_gemm_activation_fusion_tests)

add_executable(tinyinfer_execution_context_tests execution_context_test.cpp)
target_link_libraries(tinyinfer_execution_context_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_execution_context_tests COMMAND tinyinfer_execution_context_tests)

add_executable(tinyinfer_memory_plan_tests memory_plan_test.cpp)
target_link_libraries(tinyinfer_memory_plan_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_memory_plan_tests COMMAND tinyinfer_memory_plan_tests)

add_executable(tinyinfer_cpu_operator_registry_tests cpu_operator_registry_test.cpp)
target_link_libraries(tinyinfer_cpu_operator_registry_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_cpu_operator_registry_tests COMMAND tinyinfer_cpu_operator_registry_tests)

add_executable(tinyinfer_graph_mlp_tests graph_mlp_test.cpp)
target_link_libraries(tinyinfer_graph_mlp_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_graph_mlp_tests COMMAND tinyinfer_graph_mlp_tests)

add_executable(tinyinfer_onnx_importer_tests onnx_importer_test.cpp)
target_link_libraries(tinyinfer_onnx_importer_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_onnx_importer_tests COMMAND tinyinfer_onnx_importer_tests)

add_executable(tinyinfer_onnx_compatibility_tests onnx_compatibility_test.cpp)
target_link_libraries(tinyinfer_onnx_compatibility_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_onnx_compatibility_tests COMMAND tinyinfer_onnx_compatibility_tests)

add_executable(tinyinfer_onnx_operator_registry_tests onnx_operator_registry_test.cpp)
target_link_libraries(tinyinfer_onnx_operator_registry_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_onnx_operator_registry_tests COMMAND tinyinfer_onnx_operator_registry_tests)

add_executable(tinyinfer_protobuf_model_parser_tests protobuf_model_parser_test.cpp)
target_link_libraries(tinyinfer_protobuf_model_parser_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_protobuf_model_parser_tests COMMAND tinyinfer_protobuf_model_parser_tests)

add_executable(tinyinfer_onnx_gemm_fixture_tests onnx_gemm_fixture_test.cpp)
target_link_libraries(tinyinfer_onnx_gemm_fixture_tests PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_onnx_gemm_fixture_tests PRIVATE
    TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME tinyinfer_onnx_gemm_fixture_tests COMMAND tinyinfer_onnx_gemm_fixture_tests)

add_executable(tinyinfer_onnx_rl_mlp_fixture_tests onnx_rl_mlp_fixture_test.cpp)
target_link_libraries(tinyinfer_onnx_rl_mlp_fixture_tests PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_onnx_rl_mlp_fixture_tests PRIVATE
    TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME tinyinfer_onnx_rl_mlp_fixture_tests COMMAND tinyinfer_onnx_rl_mlp_fixture_tests)

add_executable(tinyinfer_gemm_tests gemm_test.cpp)
target_link_libraries(tinyinfer_gemm_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_gemm_tests COMMAND tinyinfer_gemm_tests)

add_executable(tinyinfer_cpu_execution_plan_tests cpu_execution_plan_test.cpp)
target_link_libraries(tinyinfer_cpu_execution_plan_tests PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_cpu_execution_plan_tests PRIVATE
    TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME tinyinfer_cpu_execution_plan_tests COMMAND tinyinfer_cpu_execution_plan_tests)

add_executable(tinyinfer_gemm_kernel_fusion_tests gemm_kernel_fusion_test.cpp)
target_link_libraries(tinyinfer_gemm_kernel_fusion_tests PRIVATE TinyInfer::tinyinfer)
target_include_directories(tinyinfer_gemm_kernel_fusion_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME tinyinfer_gemm_kernel_fusion_tests COMMAND tinyinfer_gemm_kernel_fusion_tests)

add_executable(tinyinfer_device_runtime_tests device_runtime_test.cpp)
target_link_libraries(tinyinfer_device_runtime_tests PRIVATE TinyInfer::tinyinfer)
add_test(NAME tinyinfer_device_runtime_tests COMMAND tinyinfer_device_runtime_tests)

add_executable(tinyinfer_backend_capability_tests backend_capability_test.cpp)
target_link_libraries(tinyinfer_backend_capability_tests PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_backend_capability_tests PRIVATE
    TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME tinyinfer_backend_capability_tests COMMAND tinyinfer_backend_capability_tests)

add_executable(tinyinfer_cuda_contract_tests cuda_contract_test.cpp)
target_link_libraries(tinyinfer_cuda_contract_tests PRIVATE TinyInfer::cuda_contract)
target_compile_definitions(tinyinfer_cuda_contract_tests PRIVATE TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME tinyinfer_cuda_contract_tests COMMAND tinyinfer_cuda_contract_tests)

if(TINYINFER_ENABLE_METAL)
    foreach(kind runtime kernel execution_plan)
        add_executable(tinyinfer_metal_${kind}_tests metal_${kind}_test.cpp)
        target_link_libraries(tinyinfer_metal_${kind}_tests PRIVATE TinyInfer::metal)
        target_compile_definitions(tinyinfer_metal_${kind}_tests PRIVATE
            TINYINFER_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        if(TINYINFER_METAL_REQUIRE_DEVICE)
            add_test(NAME tinyinfer_metal_${kind}_tests COMMAND tinyinfer_metal_${kind}_tests --require-device)
        else()
            add_test(NAME tinyinfer_metal_${kind}_tests COMMAND tinyinfer_metal_${kind}_tests)
        endif()
        set_tests_properties(tinyinfer_metal_${kind}_tests PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    endforeach()
endif()

# Existing tests also use assert(). Keep those checks active in Release gates.
get_property(TINYINFER_TEST_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
foreach(TINYINFER_TEST_TARGET IN LISTS TINYINFER_TEST_TARGETS)
    if(MSVC)
        target_compile_options(${TINYINFER_TEST_TARGET} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${TINYINFER_TEST_TARGET} PRIVATE -UNDEBUG)
    endif()
endforeach()

# Optional Python example integration; no Python dependency for core-only builds.
if(TARGET tinyinfer_serving_worker)
    find_package(Python3 3.7 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME tinyinfer_serving_scheduler_tests
            COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
                "TINYINFER_SERVING_WORKER=$<TARGET_FILE:tinyinfer_serving_worker>"
                ${Python3_EXECUTABLE} -m unittest discover
                -s ${PROJECT_SOURCE_DIR}/examples/serving_scheduler -v)
        set_tests_properties(tinyinfer_serving_scheduler_tests PROPERTIES TIMEOUT 180)
    endif()
endif()
