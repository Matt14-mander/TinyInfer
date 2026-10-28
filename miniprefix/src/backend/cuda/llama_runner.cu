#include "miniprefix/model.h"
#include "resources.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace miniprefix {
namespace {
using cuda_detail::check;
using cuda_detail::blas;
using cuda_detail::Allocation;
__global__ void embedding_kernel(float* x,const float* weights,const int* tokens,int rows,int width){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width)x[i]=weights[std::size_t(tokens[i/width])*width+i%width];}
__global__ void rms_kernel(const float* x,const float* weight,float* out,int width,float eps){
    __shared__ float sums[256];const int row=blockIdx.x,lane=threadIdx.x;float sum=0;
    for(int d=lane;d<width;d+=256){float v=x[row*width+d];sum+=v*v;}sums[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)sums[lane]+=sums[lane+step];__syncthreads();}
    const float scale=1.0F/sqrtf(sums[0]/width+eps);
    for(int d=lane;d<width;d+=256)out[row*width+d]=x[row*width+d]*scale*weight[d];}
__global__ void rope_kernel(float* x,int count,int heads,int dim,int start,float theta){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*heads*(dim/2))return;
    int d=i%(dim/2),headrow=i/(dim/2),row=headrow/heads;
    float angle=float(start+row)/powf(theta,float(2*d)/dim);float cs=cosf(angle),sn=sinf(angle);
    int a=headrow*dim+d,b=a+dim/2;float va=x[a],vb=x[b];x[a]=va*cs-vb*sn;x[b]=vb*cs+va*sn;}
__global__ void append_kernel(const float* k,const float* v,float* keys,float* values,const int* table,
                              int n,int width,int start,int layer,int layers,int block_tokens){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*width){const int token=start+i/width;
        const auto offset=((std::size_t(table[token/block_tokens])*layers+layer)*block_tokens+token%block_tokens)*width+i%width;
        keys[offset]=k[i];values[offset]=v[i];}}
__global__ void attention_kernel(const float* q,const float* k,const float* v,float* out,const int* table,
                                 int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads),length=start+row+1;
    __shared__ float reduction[256];
    float maximum=-CUDART_INF_F,denom=0,numerator=0;
    const float query=lane<dim?q[(row*heads+head)*dim+lane]:0;
    for(int t=0;t<length;++t){
        const std::size_t offset=(((std::size_t(table[t/block_tokens])*layers+layer)*block_tokens+t%block_tokens)*kvheads+kh)*dim;
        reduction[lane]=lane<dim?query*k[offset+lane]:0;__syncthreads();
        for(int step=128;step;step/=2){if(lane<step)reduction[lane]+=reduction[lane+step];__syncthreads();}
        float score=reduction[0]/sqrtf(float(dim));float next=fmaxf(maximum,score),scale=expf(maximum-next),weight=expf(score-next);
        denom=denom*scale+weight;if(lane<dim)numerator=numerator*scale+weight*v[offset+lane];maximum=next;
        __syncthreads();
    }
    if(lane<dim)out[(row*heads+head)*dim+lane]=numerator/denom;
}
__global__ void add_kernel(float* x,const float* y,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]+=y[i];}
__global__ void swiglu_kernel(float* gate,const float* up,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=gate[i];gate[i]=(v/(1+expf(-v)))*up[i];}}
class CudaRunner final:public ModelRunner {
    std::shared_ptr<ModelWeights> weights_;RunnerStats stats_;
    KVManager kv_;
    Allocation keys_,values_;
    cuda_detail::StreamBlas execution_;
    std::map<std::string,std::unique_ptr<Allocation>> parameters_;
    std::map<RequestId,std::unique_ptr<Allocation>> block_tables_;
    cuda_detail::Workspace workspace_;
    float* parameter(const std::string& name){return parameters_.at(name)->data;}
    void linear(const Allocation& x,const std::string& name,Allocation& out,int rows,int inner,int columns){
        const float alpha=1,beta=0;
        blas(cublasSgemm(execution_.handle(),CUBLAS_OP_N,CUBLAS_OP_N,columns,rows,inner,&alpha,parameter(name),columns,x.data,inner,&beta,out.data,columns));}
    void upload_parameter(const std::string& name,const tinyinfer::Tensor& input){
        auto data=input;data.contiguous();auto memory=std::make_unique<Allocation>();memory->ensure(data.size_bytes(),stats_);
        check(cudaMemcpyAsync(memory->data,data.data<float>(),data.size_bytes(),cudaMemcpyHostToDevice,execution_.stream()),"upload weights");
        execution_.synchronize("weight upload completion");
        stats_.weight_uploaded_bytes+=data.size_bytes();stats_.device_weight_bytes+=data.size_bytes();parameters_.emplace(name,std::move(memory));}
public:
    CudaRunner(std::shared_ptr<ModelWeights> weights,int context,KvCacheConfig kv):weights_(std::move(weights)),
        kv_(kv,config().layers,config().kv_width(),context,config().vocabulary){
        if(context<1||context>config().context)throw std::invalid_argument("invalid runner context");
        execution_.initialize();
        try{
            for(const auto& p:weights_->tensors)upload_parameter(p.first,p.second);
            if(config().tied){auto head=weights_->get("model.embed_tokens.weight");head.transpose(0,1);upload_parameter("lm_head.weight",head);}
            keys_.ensure(kv_.reserved_bytes()/2,stats_);values_.ensure(kv_.reserved_bytes()/2,stats_);
        }catch(...){execution_.reset();throw;}
    }
    ~CudaRunner() override{execution_.synchronize_noexcept();block_tables_.clear();parameters_.clear();execution_.reset();}
    const ModelConfig& config() const override{return weights_->config;}
    int context_limit() const override{return kv_.context_limit();}
    void create(const RequestId& id) override{
        if(kv_.contains(id))throw std::invalid_argument("sequence already exists");
        auto table=std::make_unique<Allocation>();table->ensure(kv_.max_sequence_blocks()*sizeof(int),stats_);
        kv_.create(id);
        try{block_tables_.emplace(id,std::move(table));}catch(...){kv_.release(id);throw;}}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt) override{
        const auto used=kv_.reuse_prefix(id,prompt);const auto& state=kv_.sequence(id);
        if(used){const auto bytes=state.blocks.size()*sizeof(int);
            try{check(cudaMemcpyAsync(block_tables_.at(id)->data,state.blocks.data(),bytes,cudaMemcpyHostToDevice,execution_.stream()),"upload reused block table");
                execution_.synchronize("prefix table completion");stats_.block_table_uploaded_bytes+=bytes;
            }catch(...){execution_.synchronize_noexcept();kv_.reset_sequence(id);throw;}}
        return used;
    }
    void cache_prefix(const RequestId& id) override{kv_.cache_prefix(id);}
    void clear_prefix_cache() override{execution_.synchronize("clear prefix completion");kv_.clear_prefix_cache();}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        auto append=kv_.prepare_append(id,tokens);const auto& state=append.sequence();
        const auto& c=config();const int n=static_cast<int>(tokens.size());
        std::vector<float> result(c.vocabulary);
        const auto original_blocks=append.original_blocks();
        try{
            if(state.blocks.size()!=original_blocks){const auto bytes=(state.blocks.size()-original_blocks)*sizeof(int);
                check(cudaMemcpyAsync(reinterpret_cast<int*>(block_tables_.at(id)->data)+original_blocks,state.blocks.data()+original_blocks,
                                      bytes,cudaMemcpyHostToDevice,execution_.stream()),"upload block table entries");
                stats_.block_table_uploaded_bytes+=bytes;}
            workspace_.ensure(n,c,stats_);check(cudaMemcpyAsync(workspace_.tokens.data,tokens.data(),tokens.size()*sizeof(TokenId),cudaMemcpyHostToDevice,execution_.stream()),"upload input tokens");
            stats_.token_uploaded_bytes+=tokens.size()*sizeof(TokenId);
            embedding_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,parameter("model.embed_tokens.weight"),reinterpret_cast<int*>(workspace_.tokens.data),n,c.hidden);
            for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
                rms_kernel<<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"input_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
                linear(workspace_.norm,p+"self_attn.q_proj.weight",workspace_.q,n,c.hidden,c.hidden);
                linear(workspace_.norm,p+"self_attn.k_proj.weight",workspace_.k,n,c.hidden,c.kv_width());linear(workspace_.norm,p+"self_attn.v_proj.weight",workspace_.v,n,c.hidden,c.kv_width());
                rope_kernel<<<(n*c.heads*c.head_dimension()/2+255)/256,256,0,execution_.stream()>>>(workspace_.q.data,n,c.heads,c.head_dimension(),state.used,c.rope_theta);
                rope_kernel<<<(n*c.kv_width()/2+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,n,c.kv_heads,c.head_dimension(),state.used,c.rope_theta);
                const auto* table=reinterpret_cast<const int*>(block_tables_.at(id)->data);
                append_kernel<<<(n*c.kv_width()+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,workspace_.v.data,keys_.data,values_.data,table,
                    n,c.kv_width(),state.used,l,c.layers,kv_.config().block_tokens);
                attention_kernel<<<dim3(n,c.heads),256,0,execution_.stream()>>>(workspace_.q.data,keys_.data,values_.data,workspace_.attention.data,table,
                    c.heads,c.kv_heads,c.head_dimension(),state.used,l,c.layers,kv_.config().block_tokens);
                linear(workspace_.attention,p+"self_attn.o_proj.weight",workspace_.projection,n,c.hidden,c.hidden);
                add_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,workspace_.projection.data,n*c.hidden);
                rms_kernel<<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"post_attention_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
                linear(workspace_.norm,p+"mlp.gate_proj.weight",workspace_.gate,n,c.hidden,c.intermediate);linear(workspace_.norm,p+"mlp.up_proj.weight",workspace_.up,n,c.hidden,c.intermediate);
                swiglu_kernel<<<(n*c.intermediate+255)/256,256,0,execution_.stream()>>>(workspace_.gate.data,workspace_.up.data,n*c.intermediate);
                linear(workspace_.gate,p+"mlp.down_proj.weight",workspace_.projection,n,c.intermediate,c.hidden);
                add_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,workspace_.projection.data,n*c.hidden);
            }
            rms_kernel<<<1,256,0,execution_.stream()>>>(workspace_.x.data+std::size_t(n-1)*c.hidden,parameter("model.norm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
            linear(workspace_.norm,"lm_head.weight",workspace_.logits,1,c.hidden,c.vocabulary);
            check(cudaGetLastError(),"model kernel launch");
            check(cudaMemcpyAsync(result.data(),workspace_.logits.data,result.size()*4,cudaMemcpyDeviceToHost,execution_.stream()),"download logits");
            execution_.synchronize("model completion");
            for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite CUDA logits");
        }catch(...){execution_.synchronize_noexcept();append.rollback();throw;}
        append.commit();stats_.logits_downloaded_bytes+=result.size()*4;return result;
    }
    void release(const RequestId& id) override{execution_.synchronize("release completion");
        kv_.release(id);block_tables_.erase(id);}
    void clear() override{execution_.synchronize("clear completion");
        kv_.clear();block_tables_.clear();}
    RunnerStats stats() const override{auto result=stats_;
        result.update_kv_stats(kv_.stats());
        for(const auto& item:block_tables_)result.block_table_device_bytes+=item.second->bytes;
        result.device_workspace_bytes=workspace_.bytes();
        return result;}
};
}
std::unique_ptr<ModelRunner> make_cuda_runner(std::shared_ptr<ModelWeights> weights,int context,KvCacheConfig kv){return std::make_unique<CudaRunner>(std::move(weights),context,kv);}
} // namespace miniprefix
