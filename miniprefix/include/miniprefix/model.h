#pragma once
#include "request.h"
#include "kv_manager.h"
#include "tinyinfer/core/tensor.h"
#include <map>
#include <memory>

namespace miniprefix {
struct ModelConfig {
    int layers=0, hidden=0, intermediate=0, heads=0, kv_heads=0, vocabulary=0, context=0, eos=0;
    float rms_epsilon=0, rope_theta=0;
    bool tied=true;
    int head_dimension() const { return hidden / heads; }
    int kv_width() const { return kv_heads * head_dimension(); }
    void validate() const;
};
struct ModelWeights {
    ModelConfig config;
    std::map<std::string,tinyinfer::Tensor> tensors;
    static std::shared_ptr<ModelWeights> load(const std::string& path);
    const tinyinfer::Tensor& get(const std::string& name) const;
};
struct RunnerStats {
    std::size_t live_sequences=0, live_kv_bytes=0, peak_kv_bytes=0, forward_tokens=0;
    std::size_t device_weight_bytes=0, weight_uploaded_bytes=0, token_uploaded_bytes=0;
    std::size_t kv_uploaded_bytes=0, logits_downloaded_bytes=0, device_allocations=0;
    std::size_t device_workspace_bytes=0;
    std::size_t kv_block_tokens=0,kv_total_blocks=0,kv_live_blocks=0,kv_peak_blocks=0;
    std::size_t kv_reserved_bytes=0,block_table_uploaded_bytes=0,block_table_device_bytes=0;
    std::size_t prefix_cache_capacity=0,kv_cached_blocks=0,kv_cache_only_blocks=0;
    std::size_t prefix_hits=0,prefix_misses=0,prefix_reused_tokens=0,prefix_evictions=0;
    // Preserve the original field order/layout and map common metadata once.
    void update_kv_stats(const KvStats& stats);
};
class ModelRunner {
public:
    virtual ~ModelRunner()=default;
    virtual const ModelConfig& config() const=0;
    // Fixed per-sequence token limit of this runner, possibly below model context.
    // This is independent of the shared KV pool's available page count.
    virtual int context_limit() const=0;
    // Unbounded/reference runners may use the default; bounded runners must reserve.
    virtual KvAdmission reserve_request(const RequestId&,std::size_t){return KvAdmission::Ready;}
    virtual void release_reservation(const RequestId&) noexcept{}
    virtual void create(const RequestId&)=0;
    virtual std::size_t reuse_prefix(const RequestId&,const std::vector<TokenId>&){return 0;}
    virtual void cache_prefix(const RequestId&){}
    virtual void clear_prefix_cache(){}
    virtual std::vector<float> append(const RequestId&,const std::vector<TokenId>&)=0;
    virtual void release(const RequestId&)=0;
    virtual void clear()=0;
    virtual RunnerStats stats() const=0;
};
std::unique_ptr<ModelRunner> make_cpu_runner(std::shared_ptr<ModelWeights>,int context,KvCacheConfig kv={});
#ifdef MINIPREFIX_ENABLE_CUDA
std::unique_ptr<ModelRunner> make_cuda_runner(std::shared_ptr<ModelWeights>,int context,KvCacheConfig kv={});
#endif
} // namespace miniprefix
