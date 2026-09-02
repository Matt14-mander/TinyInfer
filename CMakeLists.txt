cmake_minimum_required(VERSION 3.20)
project(TinyInfer VERSION 0.1.0 LANGUAGES CXX)

option(TINYINFER_BUILD_EXAMPLES "Build TinyInfer examples" ON)
option(TINYINFER_BUILD_TESTS "Build TinyInfer tests" ON)
option(TINYINFER_BUILD_BENCHMARKS "Build TinyInfer benchmarks" OFF)

add_library(tinyinfer
    src/core/tensor.cpp
    src/graph/graph.cpp
    src/runtime/executor.cpp
    src/backend/cpu/cpu_backend.cpp
)
add_library(TinyInfer::tinyinfer ALIAS tinyinfer)
target_compile_features(tinyinfer PUBLIC cxx_std_17)
target_include_directories(tinyinfer PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

if(MSVC)
    target_compile_options(tinyinfer PRIVATE /W4)
else()
    target_compile_options(tinyinfer PRIVATE -Wall -Wextra -Wpedantic)
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
