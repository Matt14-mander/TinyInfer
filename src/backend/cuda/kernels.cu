#include "kernels.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

namespace tinyinfer::cuda::native {
namespace {
__global__ void elementwise(Kernel kind, const float* a, const float* b,
                            const float* bias, float* output, Params p) {
    const auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<unsigned>(p.count)) return;
    float value;
    if (kind == Kernel::Add) {
        int remaining = static_cast<int>(i), ai = 0, bi = 0;
        for (int d = p.rank - 1; d >= 0; --d) {
            const int coordinate = remaining % p.dims[d]; remaining /= p.dims[d];
            ai += coordinate * p.a_strides[d]; bi += coordinate * p.b_strides[d];
        }
        value = a[ai] + b[bi];
    } else if (kind == Kernel::Relu) value = a[i] > 0 ? a[i] : 0.0F;
    else if (kind == Kernel::Tanh) value = tanhf(a[i]);
    else {
        value = p.zero_k ? 0.0F : output[i];
        // CPU matmul accumulates from +0; cuBLAS may return -0 for a
        // degenerate product. Canonicalize the product before user coefficients.
        if (value == 0.0F) value = 0.0F;
        // Preserve the CPU identity branch and separate multiply/add rounding.
        if (p.has_bias) {
            const int row = static_cast<int>(i) / p.columns, col = static_cast<int>(i) % p.columns;
            const int index = (p.bias_rows == 1 ? 0 : row) * p.bias_columns + (p.bias_columns == 1 ? 0 : col);
            value = p.alpha * value + p.beta * bias[index];
        } else if (p.alpha != 1.0F) value = p.alpha * value;
        if (p.relu) value = value > 0 ? value : 0.0F;
    }
    output[i] = value;
}
__global__ void softmax(const float* a, float* output, Params p) {
    const auto group = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= static_cast<unsigned>(p.groups)) return;
    const int base = (static_cast<int>(group) / p.inner) * p.axis_size * p.inner + static_cast<int>(group) % p.inner;
    float maximum = -CUDART_INF_F;
    for (int j = 0; j < p.axis_size; ++j) {
        const float x = a[base + j * p.inner]; if (x > maximum) maximum = x;
    }
    float total = 0;
    for (int j = 0; j < p.axis_size; ++j) total += expf(a[base + j * p.inner] - maximum);
    for (int j = 0; j < p.axis_size; ++j) output[base + j * p.inner] = expf(a[base + j * p.inner] - maximum) / total;
}
}
cudaError_t launch(Kernel kind, const float* a, const float* b, const float* bias,
                   float* output, Params p, cudaStream_t stream) {
    const unsigned count = static_cast<unsigned>(kind == Kernel::Softmax ? p.groups : p.count);
    if (!count) return cudaSuccess;
    if (kind == Kernel::Softmax) softmax<<<(count + 255U) / 256U, 256, 0, stream>>>(a, output, p);
    else elementwise<<<(count + 255U) / 256U, 256, 0, stream>>>(kind, a, b, bias, output, p);
    return cudaGetLastError();
}
}
