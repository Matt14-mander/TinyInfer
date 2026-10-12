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
