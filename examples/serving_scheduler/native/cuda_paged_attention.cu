#include "cuda_paged_attention.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <string>

namespace serving {
namespace {
void check(cudaError_t result, const char* action) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(result));
}
struct DeviceArray {
    void* pointer = nullptr;
    std::size_t capacity = 0;
    ~DeviceArray() { if (pointer) cudaFree(pointer); }
    void ensure(std::size_t bytes) {
        if (bytes <= capacity) return;
        void* next = nullptr;check(cudaMalloc(&next, bytes), "paged cudaMalloc");
        if (pointer) {
            const auto status = cudaFree(pointer);
            if (status != cudaSuccess) { cudaFree(next);check(status, "paged cudaFree"); }
        }
        pointer = next;capacity = bytes;
    }
};

// One warp per query; lanes 0..15 own output dimensions. Every lane participates
// in the shuffle reductions, including the unused dimension lanes.
__global__ void paged_attention_kernel(const float* keys, const float* values,
    const float* queries, const int* tables, const int* lengths, int stride,
    int block_size, float* output) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int length = lengths[row];
    const float query = lane < 16 ? queries[row * 16 + lane] : 0.0F;
    float maximum = -CUDART_INF_F, denominator = 0.0F, numerator = 0.0F;
    for (int token = 0; token < length; ++token) {
        const int block = tables[std::size_t(row) * stride + token / block_size];
        const std::size_t offset = (std::size_t(block) * block_size + token % block_size) * 16;
        float score = lane < 16 ? query * keys[offset + lane] : 0.0F;
        for (int step = 16; step > 0; step /= 2) score += __shfl_down_sync(0xffffffff, score, step);
        score = __shfl_sync(0xffffffff, score, 0) * 0.25F;
        const float next_max = fmaxf(maximum, score);
        const float correction = expf(maximum - next_max), weight = expf(score - next_max);
        denominator = denominator * correction + weight;
        if (lane < 16) numerator = numerator * correction + weight * values[offset + lane];
        maximum = next_max;
    }
    if (lane < 16) output[row * 16 + lane] = numerator / denominator;
}
} // namespace

struct CudaPagedAttention::Impl {
    cudaStream_t stream = nullptr;
    DeviceArray keys, values, queries, tables, lengths, output;
    std::size_t launches = 0, uploaded = 0;
    Impl() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "paged stream create"); }
    ~Impl() { if (stream) { cudaStreamSynchronize(stream);cudaStreamDestroy(stream); } }
};
CudaPagedAttention::CudaPagedAttention() : impl_(std::make_unique<Impl>()) {}
CudaPagedAttention::~CudaPagedAttention() = default;
std::size_t CudaPagedAttention::launches() const { return impl_->launches; }
std::size_t CudaPagedAttention::uploaded_bytes() const { return impl_->uploaded; }
void CudaPagedAttention::reset_statistics() { impl_->launches = impl_->uploaded = 0; }

tinyinfer::Tensor CudaPagedAttention::run(const PagedKVCache& pool, const tinyinfer::Tensor& queries,
    const std::vector<int>& ids, const std::vector<int>& lengths) {
    if (ids.empty() || ids.size() != lengths.size() || queries.shape() != tinyinfer::Shape({static_cast<std::int64_t>(ids.size()),16}) ||
        !queries.is_contiguous() || queries.dtype() != tinyinfer::DataType::Float32)
        throw std::invalid_argument("invalid CUDA paged batch");
    int stride = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (lengths[i] < 1 || lengths[i] > 4096 ||
            std::size_t(lengths[i]) > pool.table(ids[i]).size() * pool.block_size())
            throw std::invalid_argument("invalid CUDA context length");
        stride = std::max(stride, (lengths[i] + pool.block_size() - 1) / pool.block_size());
    }
    std::vector<int> tables(ids.size() * stride, -1);
    for (std::size_t row = 0; row < ids.size(); ++row)
        std::copy_n(pool.table(ids[row]).begin(), (lengths[row] + pool.block_size() - 1) / pool.block_size(), tables.begin() + row * stride);
    auto& s = *impl_;
    const auto bytes = pool.pool_bytes() / 2;
    s.keys.ensure(bytes);s.values.ensure(bytes);s.queries.ensure(queries.size_bytes());
    s.tables.ensure(tables.size() * sizeof(int));s.lengths.ensure(lengths.size() * sizeof(int));s.output.ensure(queries.size_bytes());
    auto upload = [&](DeviceArray& target, const void* source, std::size_t count) {
        check(cudaMemcpyAsync(target.pointer, source, count, cudaMemcpyHostToDevice, s.stream), "paged H2D");s.uploaded += count;
    };
    tinyinfer::Tensor result(queries.shape());
    // Synchronize on failure too, so host table/query buffers outlive any queued
    // transfers. The exception propagates to the worker; no CPU fallback.
    try {
        upload(s.keys, pool.keys_data(), bytes);upload(s.values, pool.values_data(), bytes);
        upload(s.queries, queries.data<float>(), queries.size_bytes());
        upload(s.tables, tables.data(), tables.size() * sizeof(int));upload(s.lengths, lengths.data(), lengths.size() * sizeof(int));
        paged_attention_kernel<<<static_cast<unsigned>(ids.size()),32,0,s.stream>>>(
            static_cast<float*>(s.keys.pointer), static_cast<float*>(s.values.pointer), static_cast<float*>(s.queries.pointer),
            static_cast<int*>(s.tables.pointer), static_cast<int*>(s.lengths.pointer), stride, pool.block_size(), static_cast<float*>(s.output.pointer));
        check(cudaGetLastError(), "paged kernel launch");++s.launches;
        check(cudaMemcpyAsync(result.data<float>(), s.output.pointer, result.size_bytes(), cudaMemcpyDeviceToHost, s.stream), "paged D2H");
        check(cudaStreamSynchronize(s.stream), "paged stream completion");
    } catch (...) { cudaStreamSynchronize(s.stream);throw; }
    for (std::size_t i = 0; i < result.numel(); ++i)
        if (!std::isfinite(result.at(i))) throw std::runtime_error("nonfinite CUDA paged output");
    return result;
}
} // namespace serving
