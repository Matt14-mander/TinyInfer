# Architecture scaffold only. No runnable model/engine/server is claimed here.
# Keep public contracts usable without CUDA or Python dependencies.
add_library(miniprefix_contracts INTERFACE)
add_library(MiniPrefix::contracts ALIAS miniprefix_contracts)
target_compile_features(miniprefix_contracts INTERFACE cxx_std_17)
target_include_directories(miniprefix_contracts INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Add actual targets only alongside working implementations and acceptance:
# miniprefix_engine -> TinyInfer::tinyinfer
# miniprefix_cuda   -> TinyInfer::cuda (TINYINFER_ENABLE_CUDA only)
# miniprefix_cli    -> miniprefix_engine
