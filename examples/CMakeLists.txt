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
