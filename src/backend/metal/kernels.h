#pragma once
namespace tinyinfer::detail {
inline constexpr const char* metal_kernel_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct Params {
    uint count, rank, m, k, n, trans_a, trans_b, has_bias, relu;
    uint dims[4], a_strides[4], b_strides[4];
    uint bias_rows, bias_cols, inner, axis_size, groups;
    float alpha, beta;
};
uint offset(uint index, constant Params& p, constant uint* strides) {
    uint result = 0;
    for (uint dim = p.rank; dim > 0; --dim) {
        uint coordinate = index % p.dims[dim - 1];
        index /= p.dims[dim - 1];
        result += coordinate * strides[dim - 1];
    }
    return result;
}
kernel void add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = a[offset(i, p, p.a_strides)] + b[offset(i, p, p.b_strides)];
}
kernel void relu_f32(device const float* a [[buffer(0)]], device float* out [[buffer(2)]],
    constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
kernel void tanh_f32(device const float* a [[buffer(0)]], device float* out [[buffer(2)]],
    constant Params& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) out[i] = tanh(a[i]);
}
kernel void gemm_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], constant Params& p [[buffer(3)]],
    device const float* bias [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    uint row = i / p.n, col = i % p.n;
    float sum = 0.0f;
    for (uint kk = 0; kk < p.k; ++kk) {
        uint ai = p.trans_a ? kk * p.m + row : row * p.k + kk;
        uint bi = p.trans_b ? col * p.k + kk : kk * p.n + col;
        sum += a[ai] * b[bi];
    }
    // No-bias identity skips multiplication exactly as the CPU helper does.
    if (p.has_bias) {
        uint ci = (p.bias_rows == 1 ? 0 : row) * p.bias_cols + (p.bias_cols == 1 ? 0 : col);
        sum = p.alpha * sum + p.beta * bias[ci];
    } else if (p.alpha != 1.0f || p.relu) sum = p.alpha * sum;
    out[i] = p.relu ? (sum > 0.0f ? sum : 0.0f) : sum;
}
kernel void softmax_f32(device const float* a [[buffer(0)]], device float* out [[buffer(2)]],
    constant Params& p [[buffer(3)]], uint group [[thread_position_in_grid]]) {
    if (group >= p.groups) return;
    uint outer = group / p.inner, inner = group % p.inner;
    uint base = outer * p.axis_size * p.inner + inner;
    float maximum = -INFINITY;
    for (uint j = 0; j < p.axis_size; ++j) {
        float value = a[base + j * p.inner];
        if (value > maximum) maximum = value;
    }
    float total = 0.0f;
    for (uint j = 0; j < p.axis_size; ++j) total += exp(a[base + j * p.inner] - maximum);
    for (uint j = 0; j < p.axis_size; ++j)
        out[base + j * p.inner] = exp(a[base + j * p.inner] - maximum) / total;
}
)MSL";
} // namespace tinyinfer::detail
