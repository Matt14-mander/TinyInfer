#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace miniprefix {
struct RunnerStats;
struct ModelConfig;
namespace cuda_detail {
void check(cudaError_t status,const char* what);
void blas(cublasStatus_t status);

// Internal grow-only allocation. Growth discards contents, as before; callers
// synchronize outstanding use before growth or destruction. No implicit copies.
struct Allocation {
    float* data=nullptr;
    std::size_t bytes=0;
    Allocation()=default;
    Allocation(const Allocation&)=delete;
    Allocation& operator=(const Allocation&)=delete;
    ~Allocation();
    void ensure(std::size_t size,RunnerStats& stats);
};

// Explicit initialization keeps device creation after runner config validation.
// reset/destruction are best-effort cleanup; checked completion remains explicit.
class StreamBlas {
    cudaStream_t stream_=nullptr;
    cublasHandle_t handle_=nullptr;
public:
    StreamBlas()=default;
    StreamBlas(const StreamBlas&)=delete;
    StreamBlas& operator=(const StreamBlas&)=delete;
    ~StreamBlas();
    void initialize();
    cudaStream_t stream() const;
    cublasHandle_t handle() const;
    void synchronize(const char* what) const;
    void synchronize_noexcept() const noexcept;
    void reset() noexcept;
};

// Model temporaries only: excludes weights, KV payload and device block tables.
struct Workspace {
    Allocation x,norm,q,k,v,attention,projection,gate,up,logits,tokens;
    void ensure(int rows,const ModelConfig& config,RunnerStats& stats);
    std::size_t bytes() const;
};
} // namespace cuda_detail
} // namespace miniprefix
