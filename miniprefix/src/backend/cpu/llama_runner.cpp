#include "miniprefix/model.h"
#include "tinyinfer/backend/cpu/matmul_kernel.h"
#include "tinyinfer/ops/basic_ops.h"
#include <algorithm>
#include <cmath>

namespace miniprefix {
namespace {
using tinyinfer::Tensor;
Tensor rms(const Tensor& x,const Tensor& weight,float eps){
    Tensor out(x.shape());const auto width=x.shape()[1];
    for(std::int64_t r=0;r<x.shape()[0];++r){float sum=0;for(int d=0;d<width;++d){float v=x.at(r*width+d);sum+=v*v;}
        const float scale=1/std::sqrt(sum/float(width)+eps);
        for(int d=0;d<width;++d)out.at(r*width+d)=x.at(r*width+d)*scale*weight.at(d);}
    return out;
}
void rope(Tensor& x,int heads,int dim,int start,float theta){
    for(int row=0;row<x.shape()[0];++row)for(int h=0;h<heads;++h)for(int d=0;d<dim/2;++d){
        const float angle=float(start+row)/std::pow(theta,float(2*d)/float(dim));
        const float cs=std::cos(angle),sn=std::sin(angle);const auto i=(row*heads+h)*dim+d,j=i+dim/2;
        const float a=x.at(i),b=x.at(j);x.at(i)=a*cs-b*sn;x.at(j)=b*cs+a*sn;}
}
class CpuRunner final:public ModelRunner {
    std::shared_ptr<ModelWeights> weights_;int context_;
    KvBlockPool pool_;
    std::vector<float> keys_,values_;
    std::map<std::string,std::unique_ptr<tinyinfer::cpu::PackedMatMulRhs>> packed_;
    std::map<RequestId,PagedSequence> sequences_;RunnerStats stats_;
    Tensor linear(const Tensor& x,const std::string& name){
        auto& packed=packed_[name];if(!packed){
            if(name=="lm_head.weight"&&config().tied)packed=std::make_unique<tinyinfer::cpu::PackedMatMulRhs>(weights_->get("model.embed_tokens.weight"),true);
            else packed=std::make_unique<tinyinfer::cpu::PackedMatMulRhs>(weights_->get(name));}
        Tensor result({x.shape()[0],static_cast<std::int64_t>(packed->columns())});
        tinyinfer::cpu::matmul_packed_simd(x,*packed,result);return result;
    }
public:
    CpuRunner(std::shared_ptr<ModelWeights> w,int context,KvCacheConfig kv):weights_(std::move(w)),context_(context),
        pool_(kv,config().layers,config().kv_width()){
        if(context<1||context>config().context)throw std::invalid_argument("invalid runner context");
        keys_.resize(pool_.block_elements()*kv.max_blocks);values_.resize(keys_.size());}
    const ModelConfig& config() const override{return weights_->config;}
    void create(const RequestId& id) override{
        if(sequences_.count(id))throw std::invalid_argument("sequence already exists");
        sequences_.emplace(id,PagedSequence{});
    }
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        auto& s=sequences_.at(id);const auto& c=config();const int n=static_cast<int>(tokens.size()),dim=c.head_dimension(),kv=c.kv_width();
        if(n<1||n>context_-s.used)throw std::invalid_argument("context overflow/empty input");
        Tensor x({n,c.hidden});const auto& embedding=weights_->get("model.embed_tokens.weight");
        for(int row=0;row<n;++row){if(tokens[row]<0||tokens[row]>=c.vocabulary)throw std::invalid_argument("invalid token");
            std::copy_n(embedding.data<float>()+std::size_t(tokens[row])*c.hidden,c.hidden,x.data<float>()+row*c.hidden);}
        const auto original_blocks=s.blocks.size();pool_.reserve(s,s.used+n);
        try{
        for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
            auto normalized=rms(x,weights_->get(p+"input_layernorm.weight"),c.rms_epsilon);
            auto q=linear(normalized,p+"self_attn.q_proj.weight"),k=linear(normalized,p+"self_attn.k_proj.weight"),v=linear(normalized,p+"self_attn.v_proj.weight");
            rope(q,c.heads,dim,s.used,c.rope_theta);rope(k,c.kv_heads,dim,s.used,c.rope_theta);
            for(int row=0;row<n;++row){const auto offset=pool_.offset(s,l,s.used+row,c.layers,kv);
                std::copy_n(k.data<float>()+row*kv,kv,keys_.data()+offset);
                std::copy_n(v.data<float>()+row*kv,kv,values_.data()+offset);}
            Tensor contexts({n,c.hidden});
            for(int row=0;row<n;++row)for(int h=0;h<c.heads;++h){
                const int length=s.used+row+1,kh=h/(c.heads/c.kv_heads);std::vector<float> scores(length);float maximum=-INFINITY;
                for(int t=0;t<length;++t){const auto offset=pool_.offset(s,l,t,c.layers,kv)+kh*dim;
                    float score=0;for(int d=0;d<dim;++d)score+=q.at((row*c.heads+h)*dim+d)*keys_[offset+d];
                    scores[t]=score/std::sqrt(float(dim));maximum=std::max(maximum,scores[t]);}
                float denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
                for(int d=0;d<dim;++d){float total=0;for(int t=0;t<length;++t)total+=(scores[t]/denominator)*values_[pool_.offset(s,l,t,c.layers,kv)+kh*dim+d];
                    contexts.at((row*c.heads+h)*dim+d)=total;}
            }
            x=tinyinfer::ops::add(x,linear(contexts,p+"self_attn.o_proj.weight"));
            normalized=rms(x,weights_->get(p+"post_attention_layernorm.weight"),c.rms_epsilon);
            auto gate=linear(normalized,p+"mlp.gate_proj.weight"),up=linear(normalized,p+"mlp.up_proj.weight");
            for(std::size_t i=0;i<gate.numel();++i){const float a=gate.at(i);gate.at(i)=(a/(1+std::exp(-a)))*up.at(i);}
            x=tinyinfer::ops::add(x,linear(gate,p+"mlp.down_proj.weight"));
        }
        auto normalized=rms(x.narrow(0,n-1,1),weights_->get("model.norm.weight"),c.rms_epsilon);
        auto logits=linear(normalized,"lm_head.weight");std::vector<float> result(logits.data<float>(),logits.data<float>()+c.vocabulary);
        for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");
        s.used+=n;stats_.forward_tokens+=n;return result;
        }catch(...){pool_.rollback(s,original_blocks);throw;}
    }
    void release(const RequestId& id) override{
        auto it=sequences_.find(id);if(it==sequences_.end())throw std::invalid_argument("unknown sequence");
        pool_.release(it->second);sequences_.erase(it);}
    void clear() override{for(auto& item:sequences_)pool_.release(item.second);sequences_.clear();}
    RunnerStats stats() const override{auto result=stats_;result.live_sequences=sequences_.size();
        result.kv_block_tokens=pool_.config().block_tokens;result.kv_total_blocks=pool_.config().max_blocks;
        result.kv_live_blocks=pool_.live_blocks();result.kv_peak_blocks=pool_.peak_blocks();
        result.kv_reserved_bytes=pool_.reserved_bytes();result.live_kv_bytes=pool_.live_blocks()*pool_.block_bytes();
        result.peak_kv_bytes=pool_.peak_blocks()*pool_.block_bytes();return result;}
};
}
std::unique_ptr<ModelRunner> make_cpu_runner(std::shared_ptr<ModelWeights> weights,int context,KvCacheConfig kv){return std::make_unique<CpuRunner>(std::move(weights),context,kv);}
} // namespace miniprefix
