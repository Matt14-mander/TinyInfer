#pragma once
#include "request.h"
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
};
class ModelRunner {
public:
    virtual ~ModelRunner()=default;
    virtual const ModelConfig& config() const=0;
    virtual void create(const RequestId&)=0;
    virtual std::vector<float> append(const RequestId&,const std::vector<TokenId>&)=0;
    virtual void release(const RequestId&)=0;
    virtual void clear()=0;
    virtual RunnerStats stats() const=0;
};
std::unique_ptr<ModelRunner> make_cpu_runner(std::shared_ptr<ModelWeights>,int context);
#ifdef MINIPREFIX_ENABLE_CUDA
std::unique_ptr<ModelRunner> make_cuda_runner(std::shared_ptr<ModelWeights>,int context);
#endif
} // namespace miniprefix
