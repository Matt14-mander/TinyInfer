cmake_minimum_required(VERSION 3.20)
project(TinyInfer VERSION 0.3.0 LANGUAGES CXX)

option(TINYINFER_BUILD_EXAMPLES "Build TinyInfer examples" ON)
option(TINYINFER_BUILD_TESTS "Build TinyInfer tests" ON)
option(TINYINFER_BUILD_BENCHMARKS "Build TinyInfer benchmarks" OFF)
option(TINYINFER_BUILD_MINIPREFIX "Expose experimental MiniPrefix contract target (no engine yet)" OFF)
option(TINYINFER_ENABLE_METAL "Build optional macOS Metal backend" OFF)
option(TINYINFER_METAL_REQUIRE_DEVICE "Fail Metal tests if no GPU is visible" OFF)
option(TINYINFER_ENABLE_NATIVE_ARCH "Enable host-specific CPU instructions" OFF)
option(TINYINFER_ENABLE_CUDA "Build native CUDA resources and toolchain probe" OFF)
option(TINYINFER_CUDA_REQUIRE_DEVICE "Fail native CUDA tests without a device" OFF)

add_library(tinyinfer
    src/core/memory/allocator.cpp
    src/core/memory/arena_allocator.cpp
    src/core/memory/buffer.cpp
    src/core/memory/storage.cpp
    src/core/tensor_layout.cpp
    src/core/tensor_iterator.cpp
    src/core/reduction_iterator.cpp
    src/core/tensor.cpp
    src/ops/basic_ops.cpp
    src/ops/gemm.cpp
    src/ops/matmul.cpp
    src/ops/operator_schema.cpp
    src/graph/graph.cpp
    src/graph/graph_analysis.cpp
    src/graph/optimizer/optimization_result.cpp
    src/graph/optimizer/graph_rewriter.cpp
    src/graph/optimizer/gemm_activation_fusion_pass.cpp
    src/graph/optimizer/matmul_add_canonicalization_pass.cpp
    src/graph/optimizer/constant_folding_pass.cpp
    src/graph/optimizer/dead_code_elimination_pass.cpp
    src/graph/optimizer/no_op_pass.cpp
    src/graph/optimizer/pass_manager.cpp
    src/model/model.cpp
    src/model/onnx/import_diagnostic.cpp
    src/model/onnx/operator_registry.cpp
    src/model/onnx/onnx_importer.cpp
    src/model/onnx/protobuf_model_parser.cpp
    src/runtime/memory_plan.cpp
    src/runtime/execution_context.cpp
    src/runtime/executor.cpp
    src/runtime/device/device_runtime.cpp
    src/backend/capability.cpp
    src/backend/cpu/cpu_backend.cpp
    src/backend/cpu/execution_plan.cpp
    src/backend/cpu/gemm_epilogue.cpp
    src/backend/cpu/operator_registry.cpp
    src/backend/cpu/matmul_kernel.cpp
)
add_library(TinyInfer::tinyinfer ALIAS tinyinfer)
target_compile_features(tinyinfer PUBLIC cxx_std_17)
target_include_directories(tinyinfer PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(tinyinfer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(tinyinfer PRIVATE /W4)
else()
    target_compile_options(tinyinfer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TINYINFER_ENABLE_NATIVE_ARCH)
    if(MSVC)
        target_compile_options(tinyinfer PRIVATE /arch:AVX2)
    else()
        target_compile_options(tinyinfer PRIVATE -march=native)
    endif()
endif()

# SDK-free preparation metadata; this is not an executable CUDA backend.
add_library(tinyinfer_cuda_contract src/backend/cuda/operator_contract.cpp)
add_library(TinyInfer::cuda_contract ALIAS tinyinfer_cuda_contract)
target_link_libraries(tinyinfer_cuda_contract PUBLIC TinyInfer::tinyinfer)
target_compile_features(tinyinfer_cuda_contract PUBLIC cxx_std_17)

if(TINYINFER_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    add_library(tinyinfer_cuda src/backend/cuda/cuda_runtime.cpp src/backend/cuda/kernels.cu
        src/backend/cuda/execution_plan.cpp)
    add_library(TinyInfer::cuda ALIAS tinyinfer_cuda)
    target_link_libraries(tinyinfer_cuda PUBLIC TinyInfer::tinyinfer PRIVATE tinyinfer_cuda_contract CUDA::cudart CUDA::cublas)
    set_target_properties(tinyinfer_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(tinyinfer_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_compile_features(tinyinfer_cuda PUBLIC cxx_std_17)
    if(MSVC)
        target_compile_options(tinyinfer_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        target_compile_options(tinyinfer_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
    endif()
endif()

if(TINYINFER_ENABLE_METAL)
    if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        message(FATAL_ERROR "TINYINFER_ENABLE_METAL requires macOS")
    endif()
    enable_language(OBJCXX)
    add_library(tinyinfer_metal src/backend/metal/metal_runtime.mm src/backend/metal/execution_plan.cpp)
    add_library(TinyInfer::metal ALIAS tinyinfer_metal)
    target_link_libraries(tinyinfer_metal PUBLIC TinyInfer::tinyinfer PRIVATE "-framework Foundation" "-framework Metal")
    target_compile_features(tinyinfer_metal PUBLIC cxx_std_17)
    set_target_properties(tinyinfer_metal PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES)
    target_compile_options(tinyinfer_metal PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc> -Wall -Wextra -Wpedantic)
endif()

if(TINYINFER_BUILD_MINIPREFIX)
    add_subdirectory(miniprefix)
endif()

if(TINYINFER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(TINYINFER_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
if(TINYINFER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
