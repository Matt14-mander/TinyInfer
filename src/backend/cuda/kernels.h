#pragma once
#include <cuda_runtime_api.h>

namespace tinyinfer::cuda::native {
enum class Kernel { Add, Relu, Tanh, Softmax, Epilogue };
struct Params {
    int count{}, rank{}, dims[4]{}, a_strides[4]{}, b_strides[4]{};
    int columns{}, bias_rows{1}, bias_columns{1};
    int inner{1}, axis_size{1}, groups{};
    bool zero_k{}, has_bias{}, relu{};
    float alpha{1}, beta{1};
};
// Immediate launch errors are returned; asynchronous errors surface on completion.
cudaError_t launch(Kernel, const float*, const float*, const float*, float*, Params, cudaStream_t);
}
