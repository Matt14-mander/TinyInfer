add_executable(tinyinfer_quickstart quickstart.cpp)
target_link_libraries(tinyinfer_quickstart PRIVATE TinyInfer::tinyinfer)

add_executable(tinyinfer_phase0_mlp phase0_mlp.cpp)
target_link_libraries(tinyinfer_phase0_mlp PRIVATE TinyInfer::tinyinfer)

add_executable(tinyinfer_phase0_graph_mlp phase0_graph_mlp.cpp)
target_link_libraries(tinyinfer_phase0_graph_mlp PRIVATE TinyInfer::tinyinfer)

if(TINYINFER_ENABLE_METAL)
    add_executable(tinyinfer_metal_model metal_model.cpp)
    target_link_libraries(tinyinfer_metal_model PRIVATE TinyInfer::metal)
    add_executable(tinyinfer_metal_ops metal_ops.cpp)
    target_link_libraries(tinyinfer_metal_ops PRIVATE TinyInfer::metal)
endif()

add_executable(tinyinfer_cuda_contract_inspect cuda_contract.cpp)
target_link_libraries(tinyinfer_cuda_contract_inspect PRIVATE TinyInfer::cuda_contract)

if(TINYINFER_ENABLE_CUDA)
    add_executable(tinyinfer_cuda_model cuda_model.cpp)
    target_link_libraries(tinyinfer_cuda_model PRIVATE TinyInfer::cuda)
endif()

add_executable(tinyinfer_serving_worker serving_scheduler/tinyinfer_worker.cpp)
target_link_libraries(tinyinfer_serving_worker PRIVATE TinyInfer::tinyinfer)

if(TINYINFER_ENABLE_CUDA)
    add_library(tinyinfer_serving_paged_cuda serving_scheduler/native/cuda_paged_attention.cu)
    target_link_libraries(tinyinfer_serving_paged_cuda PUBLIC TinyInfer::tinyinfer PRIVATE CUDA::cudart)
    set_target_properties(tinyinfer_serving_paged_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(tinyinfer_serving_paged_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    add_executable(tinyinfer_serving_cuda_worker serving_scheduler/tinyinfer_worker.cpp)
    target_compile_definitions(tinyinfer_serving_cuda_worker PRIVATE TINYINFER_SERVING_CUDA=1)
    target_link_libraries(tinyinfer_serving_cuda_worker PRIVATE tinyinfer_serving_paged_cuda)
endif()
