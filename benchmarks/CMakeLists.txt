add_executable(tinyinfer_matmul_benchmark matmul_benchmark.cpp)
target_link_libraries(tinyinfer_matmul_benchmark PRIVATE TinyInfer::tinyinfer)

add_executable(tinyinfer_model_benchmark model_benchmark.cpp)
target_link_libraries(tinyinfer_model_benchmark PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_model_benchmark PRIVATE
    TINYINFER_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    TINYINFER_BUILD_TYPE="$<CONFIG>")

add_executable(tinyinfer_graph_optimization_benchmark graph_optimization_benchmark.cpp)
target_link_libraries(tinyinfer_graph_optimization_benchmark PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_graph_optimization_benchmark PRIVATE
    TINYINFER_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    TINYINFER_BUILD_TYPE="$<CONFIG>")

add_executable(tinyinfer_gemm_activation_fusion_benchmark
    gemm_activation_fusion_benchmark.cpp)
target_link_libraries(tinyinfer_gemm_activation_fusion_benchmark
    PRIVATE TinyInfer::tinyinfer)
target_compile_definitions(tinyinfer_gemm_activation_fusion_benchmark PRIVATE
    TINYINFER_BUILD_TYPE="$<CONFIG>")
