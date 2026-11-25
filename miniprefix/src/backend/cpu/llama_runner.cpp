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
    std::shared_ptr<ModelWeights> weights_;
    KVManager kv_;
    CostStats costs_;bool cost_timing_=false;
    std::vector<float> keys_,values_;
    std::map<std::string,std::unique_ptr<tinyinfer::cpu::PackedMatMulRhs>> packed_;
    RunnerStats stats_;
    Tensor linear(const Tensor& x,const std::string& name){
        auto& packed=packed_[name];if(!packed){
            if(name=="lm_head.weight"&&config().tied)packed=std::make_unique<tinyinfer::cpu::PackedMatMulRhs>(weights_->get("model.embed_tokens.weight"),true);
            else packed=std::make_unique<tinyinfer::cpu::PackedMatMulRhs>(weights_->get(name));}
        Tensor result({x.shape()[0],static_cast<std::int64_t>(packed->columns())});
        tinyinfer::cpu::matmul_packed_simd(x,*packed,result);return result;
    }
    struct Qkv {Tensor q,k,v;};
    Qkv project_qkv(const Tensor& input,const std::string& layer){
        auto normalized=rms(input,weights_->get(layer+"input_layernorm.weight"),config().rms_epsilon);
        return {linear(normalized,layer+"self_attn.q_proj.weight"),
                linear(normalized,layer+"self_attn.k_proj.weight"),
                linear(normalized,layer+"self_attn.v_proj.weight")};
    }
    Tensor finish_layer(Tensor& x,const Tensor& contexts,const std::string& p){
        const auto& c=config();
        x=tinyinfer::ops::add(x,linear(contexts,p+"self_attn.o_proj.weight"));
        auto normalized=rms(x,weights_->get(p+"post_attention_layernorm.weight"),c.rms_epsilon);
        auto gate=linear(normalized,p+"mlp.gate_proj.weight"),up=linear(normalized,p+"mlp.up_proj.weight");
        for(std::size_t i=0;i<gate.numel();++i){const float a=gate.at(i);gate.at(i)=(a/(1+std::exp(-a)))*up.at(i);}
        return tinyinfer::ops::add(x,linear(gate,p+"mlp.down_proj.weight"));
    }
    template<class Key,class Value>
    void attention(const float* q,float* out,int length,Key key,Value value) const{
        const auto& c=config();
        const int dim=c.head_dimension();
        std::vector<float> scores(length);
        for(int h=0;h<c.heads;++h){
            const int base=h/(c.heads/c.kv_heads)*dim;
            float maximum=-INFINITY;
            for(int t=0;t<length;++t){
                const float* k=key(t)+base;
                float score=0;
                for(int d=0;d<dim;++d)score+=q[h*dim+d]*k[d];
                scores[t]=score/std::sqrt(float(dim));
                maximum=std::max(maximum,scores[t]);
            }
            float denominator=0;
            for(float& score:scores){score=std::exp(score-maximum);denominator+=score;}
            std::fill_n(out+h*dim,dim,0.0f);
            for(int t=0;t<length;++t){
                const float* v=value(t)+base;
                float probability=scores[t]/denominator;
                for(int d=0;d<dim;++d)out[h*dim+d]+=probability*v[d];
            }
        }
    }
public:
    CpuRunner(std::shared_ptr<ModelWeights> w,int context,KvCacheConfig kv):weights_(std::move(w)),
        kv_(kv,config().layers,config().kv_width(),context,config().vocabulary){
        if(context<1||context>config().context)throw std::invalid_argument("invalid runner context");
        keys_.resize(kv_.reserved_bytes()/sizeof(float)/2);values_.resize(keys_.size());}
    const ModelConfig& config() const override{return weights_->config;}
    CostStats cost_stats() const override{
        auto result=kv_.cost_stats();result.cow_pages=costs_.cow_pages;result.cow_bytes=costs_.cow_bytes;
        result.cow_copy_ns=costs_.cow_copy_ns;result.cow_timed_copies=costs_.cow_timed_copies;return result;
    }
    void set_cost_timing(bool enabled) override{cost_timing_=enabled;kv_.set_cost_timing(enabled);}
    int context_limit() const override{return kv_.context_limit();}
    KvAdmission reserve_request(const RequestId& id,std::size_t tokens) override{return kv_.reserve_request(id,tokens);}
    void release_reservation(const RequestId& id) noexcept override{kv_.release_reservation(id);}
    void create(const RequestId& id) override{
        kv_.create(id);
    }
    void fork_sequence(const RequestId& source,const RequestId& target) override{kv_.fork_sequence(source,target);}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt) override{
        return kv_.reuse_prefix(id,prompt);
    }
    void cache_prefix(const RequestId& id) override{kv_.cache_prefix(id);}
    void clear_prefix_cache() override{kv_.clear_prefix_cache();}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        auto append=kv_.prepare_append(id,tokens);
        const auto& s=append.sequence();
        copy_tail(append);
        const auto& c=config();const int n=static_cast<int>(tokens.size()),dim=c.head_dimension(),kv=c.kv_width();
        Tensor x({n,c.hidden});const auto& embedding=weights_->get("model.embed_tokens.weight");
        for(int row=0;row<n;++row){
            std::copy_n(embedding.data<float>()+std::size_t(tokens[row])*c.hidden,c.hidden,x.data<float>()+row*c.hidden);}
        // Resolve layer/token offsets once; resource ownership stays in KVManager.
        std::vector<std::size_t> offsets(s.used+n);
        for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
            for(int t=0;t<s.used+n;++t)offsets[t]=kv_.offset(s,l,t);
            auto [q,k,v]=project_qkv(x,p);
            rope(q,c.heads,dim,s.used,c.rope_theta);rope(k,c.kv_heads,dim,s.used,c.rope_theta);
            for(int row=0;row<n;++row){const auto offset=offsets[s.used+row];
                std::copy_n(k.data<float>()+row*kv,kv,keys_.data()+offset);
                std::copy_n(v.data<float>()+row*kv,kv,values_.data()+offset);}
            Tensor contexts({n,c.hidden});
            for(int row=0;row<n;++row){
                attention(q.data<float>()+row*c.hidden,contexts.data<float>()+row*c.hidden,s.used+row+1,
                    [&](int t){return keys_.data()+offsets[t];},
                    [&](int t){return values_.data()+offsets[t];});
            }
            x=finish_layer(x,contexts,p);
        }
        auto normalized=rms(x.narrow(0,n-1,1),weights_->get("model.norm.weight"),c.rms_epsilon);
        auto logits=linear(normalized,"lm_head.weight");std::vector<float> result(logits.data<float>(),logits.data<float>()+c.vocabulary);
        for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");
        append.commit();return result;
    }
    DecodeBatch prepare_decode_batch(const std::vector<DecodeRow>& rows) override{
        const auto& c=config();
        const auto n=static_cast<std::int64_t>(rows.size());
        const int kv=c.kv_width(),dim=c.head_dimension();
        Tensor x({n,c.hidden});
        const auto* embedding=weights_->get("model.embed_tokens.weight").data<float>();
        std::vector<const PagedSequence*> histories;
        for(std::size_t i=0;i<rows.size();++i){
            const auto& [id,token]=rows[i];
            if(token<0||token>=c.vocabulary)throw std::invalid_argument("invalid decode token");
            histories.push_back(&kv_.sequence(id));
            std::copy_n(embedding+std::size_t(token)*c.hidden,c.hidden,x.data<float>()+i*c.hidden);
        }
        // Only new KV lives in scratch. The engine commits each row in scheduler
        // order after computation, so COW, publication and release stay ordered.
        std::vector<Tensor> keys,values;
        for(int layer=0;layer<c.layers;++layer){
            const auto p="model.layers."+std::to_string(layer)+".";
            auto [q,k,v]=project_qkv(x,p);
            Tensor contexts({n,c.hidden});
            for(std::size_t i=0;i<rows.size();++i){
                const auto& history=*histories[i];
                auto qr=q.narrow(0,i,1),kr=k.narrow(0,i,1);
                rope(qr,c.heads,dim,history.used,c.rope_theta);
                rope(kr,c.kv_heads,dim,history.used,c.rope_theta);
                std::vector<std::size_t> offsets(history.used);
                for(int t=0;t<history.used;++t)offsets[t]=kv_.offset(history,layer,t);
                attention(qr.data<float>(),contexts.data<float>()+i*c.hidden,history.used+1,
                    [&](int t){return t==history.used?kr.data<float>():keys_.data()+offsets[t];},
                    [&](int t){return t==history.used?v.data<float>()+i*kv:values_.data()+offsets[t];});
            }
            keys.push_back(std::move(k));values.push_back(std::move(v));
            x=finish_layer(x,contexts,p);
        }
        auto logits=linear(rms(x,weights_->get("model.norm.weight"),c.rms_epsilon),"lm_head.weight");
        return [this,rows,keys=std::move(keys),values=std::move(values),logits=std::move(logits),kv]
               (std::size_t index){
            const auto& [id,token]=rows.at(index);
            auto append=kv_.prepare_append(id,{token});
            copy_tail(append);
            const auto& s=append.sequence();
            const float* start=logits.data<float>()+index*config().vocabulary;
            std::vector<float> result(start,start+config().vocabulary);
            for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");
            for(int layer=0;layer<config().layers;++layer){
                auto offset=kv_.offset(s,layer,s.used);
                std::copy_n(keys[layer].data<float>()+index*kv,kv,keys_.data()+offset);
                std::copy_n(values[layer].data<float>()+index*kv,kv,values_.data()+offset);
            }
            append.commit();return result;
        };
    }
private:
    void copy_tail(KVManager::Append& append){
        if(append.tail_copy())++costs_.cow_pages;
        {CostTimer timer(cost_timing_&&bool(append.tail_copy()),costs_.cow_copy_ns);
        append.copy_shared_tail([&](std::size_t destination,std::size_t source,std::size_t count){
            std::copy_n(keys_.data()+source,count,keys_.data()+destination);
            std::copy_n(values_.data()+source,count,values_.data()+destination);
            costs_.cow_bytes+=2*count*sizeof(float);
        });}
        if(cost_timing_&&append.tail_copy())++costs_.cow_timed_copies;
    }
    void release(const RequestId& id) override{
        kv_.release(id);}
    void clear() override{kv_.clear();}
    RunnerStats stats() const override{auto result=stats_;result.update_kv_stats(kv_.stats());return result;}
};
}
std::unique_ptr<ModelRunner> make_cpu_runner(std::shared_ptr<ModelWeights> weights,int context,KvCacheConfig kv){return std::make_unique<CpuRunner>(std::move(weights),context,kv);}
} // namespace miniprefix
