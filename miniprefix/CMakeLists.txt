# Public value contracts remain usable without CUDA or Python dependencies.
add_library(miniprefix_contracts INTERFACE)
add_library(MiniPrefix::contracts ALIAS miniprefix_contracts)
target_compile_features(miniprefix_contracts INTERFACE cxx_std_17)
target_include_directories(miniprefix_contracts INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_library(miniprefix_kv src/kv/paged_kv.cpp src/kv/prefix_cache.cpp src/kv/kv_manager.cpp)
add_library(MiniPrefix::kv ALIAS miniprefix_kv)
target_link_libraries(miniprefix_kv PUBLIC MiniPrefix::contracts)

add_library(miniprefix_engine src/model/model.cpp src/backend/cpu/llama_runner.cpp src/engine/engine.cpp)
add_library(MiniPrefix::engine ALIAS miniprefix_engine)
target_link_libraries(miniprefix_engine PUBLIC MiniPrefix::kv TinyInfer::tinyinfer)
add_executable(miniprefix_worker apps/worker.cpp)
target_link_libraries(miniprefix_worker PRIVATE MiniPrefix::engine)

if(TINYINFER_ENABLE_CUDA)
    add_library(miniprefix_cuda src/backend/cuda/llama_runner.cu src/backend/cuda/resources.cpp)
    target_compile_definitions(miniprefix_cuda PUBLIC MINIPREFIX_ENABLE_CUDA=1)
    target_link_libraries(miniprefix_cuda PUBLIC MiniPrefix::engine PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(miniprefix_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(miniprefix_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    add_executable(miniprefix_cuda_worker apps/worker.cpp)
    target_link_libraries(miniprefix_cuda_worker PRIVATE miniprefix_cuda)
endif()
