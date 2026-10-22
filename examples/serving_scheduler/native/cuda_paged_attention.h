#pragma once
#include "paged_kv_cache.h"
#include <memory>

namespace serving {
// Optional hybrid CPU-model/GPU-attention baseline. One ordered stream and
// persistent device KV buffers; each call mirrors the host pool, then launches
// all query rows together. No CPU replay after a CUDA failure.
class CudaPagedAttention {
public:
    CudaPagedAttention();
    ~CudaPagedAttention();
    CudaPagedAttention(const CudaPagedAttention&) = delete;
    CudaPagedAttention& operator=(const CudaPagedAttention&) = delete;
    tinyinfer::Tensor run(const PagedKVCache&, const tinyinfer::Tensor& queries,
                         const std::vector<int>& ids, const std::vector<int>& lengths);
    std::size_t launches() const;
    std::size_t uploaded_bytes() const;
    void reset_statistics();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace serving
