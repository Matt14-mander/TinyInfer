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
                                 int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens,
                                 const float* new_k=nullptr,const float* new_v=nullptr){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads),length=start+row+1;
    __shared__ float reduction[256];
    float maximum=-CUDART_INF_F,denom=0,numerator=0;
    const float query=lane<dim?q[(row*heads+head)*dim+lane]:0;
    for(int t=0;t<length;++t){
        const bool staged=new_k&&t==start;
        const std::size_t offset=staged?std::size_t(kh)*dim:
            (((std::size_t(table[t/block_tokens])*layers+layer)*block_tokens+t%block_tokens)*kvheads+kh)*dim;
        const float* keys=staged?new_k:k;
        const float* values=staged?new_v:v;
        reduction[lane]=lane<dim?query*keys[offset+lane]:0;__syncthreads();
        for(int step=128;step;step/=2){if(lane<step)reduction[lane]+=reduction[lane+step];__syncthreads();}
        float score=reduction[0]/sqrtf(float(dim));float next=fmaxf(maximum,score),scale=expf(maximum-next),weight=expf(score-next);
        denom=denom*scale+weight;if(lane<dim)numerator=numerator*scale+weight*values[offset+lane];maximum=next;
        __syncthreads();
    }
    if(lane<dim)out[(row*heads+head)*dim+lane]=numerator/denom;
}
__global__ void add_kernel(float* x,const float* y,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]+=y[i];}
__global__ void swiglu_kernel(float* gate,const float* up,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=gate[i];gate[i]=(v/(1+expf(-v)))*up[i];}}
class CudaRunner final:public ModelRunner {
    std::shared_ptr<ModelWeights> weights_;RunnerStats stats_;
    KVManager kv_;
    CostStats costs_;bool cost_timing_=false;
    struct CopyEvents {
        cudaEvent_t begin=nullptr,end=nullptr;
        CopyEvents(){check(cudaEventCreate(&begin),"COW event begin");
            try{check(cudaEventCreate(&end),"COW event end");}catch(...){cudaEventDestroy(begin);throw;}}
        ~CopyEvents(){cudaEventDestroy(begin);cudaEventDestroy(end);}
    };
    std::unique_ptr<CopyEvents> copy_events_;
    Allocation keys_,values_;
    cuda_detail::StreamBlas execution_;
    std::map<std::string,std::unique_ptr<Allocation>> parameters_;
    struct BlockTable:Allocation {bool dirty=false;};
    std::map<RequestId,std::unique_ptr<BlockTable>> block_tables_;
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
    Allocation staged_keys_,staged_values_,batch_logits_;
    void project_qkv(const std::string& p,int n){
        const auto& c=config();
        rms_kernel<<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"input_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
        linear(workspace_.norm,p+"self_attn.q_proj.weight",workspace_.q,n,c.hidden,c.hidden);
        linear(workspace_.norm,p+"self_attn.k_proj.weight",workspace_.k,n,c.hidden,c.kv_width());linear(workspace_.norm,p+"self_attn.v_proj.weight",workspace_.v,n,c.hidden,c.kv_width());
    }
    void finish_layer(const std::string& p,int n){
        const auto& c=config();
        linear(workspace_.attention,p+"self_attn.o_proj.weight",workspace_.projection,n,c.hidden,c.hidden);
        add_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,workspace_.projection.data,n*c.hidden);
        rms_kernel<<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"post_attention_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
        linear(workspace_.norm,p+"mlp.gate_proj.weight",workspace_.gate,n,c.hidden,c.intermediate);linear(workspace_.norm,p+"mlp.up_proj.weight",workspace_.up,n,c.hidden,c.intermediate);
        swiglu_kernel<<<(n*c.intermediate+255)/256,256,0,execution_.stream()>>>(workspace_.gate.data,workspace_.up.data,n*c.intermediate);
        linear(workspace_.gate,p+"mlp.down_proj.weight",workspace_.projection,n,c.intermediate,c.hidden);
        add_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,workspace_.projection.data,n*c.hidden);
    }
    void prepare_pages(KVManager::Append& append,BlockTable& table_state){
        const auto& state=append.sequence();
        const auto original_blocks=append.original_blocks();
        const bool timed=cost_timing_&&bool(append.tail_copy());
        if(timed)check(cudaEventRecord(copy_events_->begin,execution_.stream()),"COW timing begin");
        append.copy_shared_tail_region([&](const KVManager::TailCopyRegion& region){
            const auto pitch=region.pitch*sizeof(float),width=region.width*sizeof(float);
            check(cudaMemcpy2DAsync(keys_.data+region.destination,pitch,keys_.data+region.source,pitch,
                width,region.rows,cudaMemcpyDeviceToDevice,execution_.stream()),"COW keys");
            check(cudaMemcpy2DAsync(values_.data+region.destination,pitch,values_.data+region.source,pitch,
                width,region.rows,cudaMemcpyDeviceToDevice,execution_.stream()),"COW values");
            costs_.cow_bytes+=2*width*region.rows;
        });
        if(timed)check(cudaEventRecord(copy_events_->end,execution_.stream()),"COW timing end");
        const auto first=table_state.dirty?std::size_t(0):(append.tail_copy()?original_blocks-1:original_blocks);
        if(state.blocks.size()!=first){const auto bytes=(state.blocks.size()-first)*sizeof(int);
            check(cudaMemcpyAsync(reinterpret_cast<int*>(table_state.data)+first,state.blocks.data()+first,
                          bytes,cudaMemcpyHostToDevice,execution_.stream()),"upload block table entries");
            stats_.block_table_uploaded_bytes+=bytes;}
    }
    void finish_cow_timing(bool timed){
        if(timed){float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,copy_events_->begin,copy_events_->end),"COW elapsed");
            costs_.cow_copy_ns+=static_cast<std::uint64_t>(milliseconds*1e6);++costs_.cow_timed_copies;}
    }
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
    CostStats cost_stats() const override{
        auto result=kv_.cost_stats();result.cow_pages=costs_.cow_pages;result.cow_bytes=costs_.cow_bytes;
        result.cow_copy_ns=costs_.cow_copy_ns;result.cow_timed_copies=costs_.cow_timed_copies;return result;
    }
    void set_cost_timing(bool enabled) override{
        if(enabled&&!copy_events_)copy_events_=std::make_unique<CopyEvents>();
        cost_timing_=enabled;kv_.set_cost_timing(enabled);
    }
    int context_limit() const override{return kv_.context_limit();}
    KvAdmission reserve_request(const RequestId& id,std::size_t tokens) override{return kv_.reserve_request(id,tokens);}
    void release_reservation(const RequestId& id) noexcept override{kv_.release_reservation(id);}
    void create(const RequestId& id) override{
        if(kv_.contains(id))throw std::invalid_argument("sequence already exists");
        auto table=std::make_unique<BlockTable>();table->ensure(kv_.max_sequence_blocks()*sizeof(int),stats_);
        kv_.create(id);
        try{block_tables_.emplace(id,std::move(table));}catch(...){kv_.release(id);throw;}}
    void fork_sequence(const RequestId& source,const RequestId& target) override{
        execution_.synchronize("fork completion");kv_.fork_sequence(source,target);
        block_tables_.at(target)->dirty=true; // Publish on the next append, not during fork.
    }
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
        auto& table_state=*block_tables_.at(id);
        const bool timed=cost_timing_&&bool(append.tail_copy());
        if(append.tail_copy())++costs_.cow_pages;
        try{
            prepare_pages(append,table_state);
            workspace_.ensure(n,c,stats_);check(cudaMemcpyAsync(workspace_.tokens.data,tokens.data(),tokens.size()*sizeof(TokenId),cudaMemcpyHostToDevice,execution_.stream()),"upload input tokens");
            stats_.token_uploaded_bytes+=tokens.size()*sizeof(TokenId);
            embedding_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,parameter("model.embed_tokens.weight"),reinterpret_cast<int*>(workspace_.tokens.data),n,c.hidden);
            for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
                project_qkv(p,n);
                rope_kernel<<<(n*c.heads*c.head_dimension()/2+255)/256,256,0,execution_.stream()>>>(workspace_.q.data,n,c.heads,c.head_dimension(),state.used,c.rope_theta);
                rope_kernel<<<(n*c.kv_width()/2+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,n,c.kv_heads,c.head_dimension(),state.used,c.rope_theta);
                const auto* table=reinterpret_cast<const int*>(block_tables_.at(id)->data);
                append_kernel<<<(n*c.kv_width()+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,workspace_.v.data,keys_.data,values_.data,table,
                    n,c.kv_width(),state.used,l,c.layers,kv_.config().block_tokens);
                attention_kernel<<<dim3(n,c.heads),256,0,execution_.stream()>>>(workspace_.q.data,keys_.data,values_.data,workspace_.attention.data,table,
                    c.heads,c.kv_heads,c.head_dimension(),state.used,l,c.layers,kv_.config().block_tokens);
                finish_layer(p,n);
            }
            rms_kernel<<<1,256,0,execution_.stream()>>>(workspace_.x.data+std::size_t(n-1)*c.hidden,parameter("model.norm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
            linear(workspace_.norm,"lm_head.weight",workspace_.logits,1,c.hidden,c.vocabulary);
            check(cudaGetLastError(),"model kernel launch");
            check(cudaMemcpyAsync(result.data(),workspace_.logits.data,result.size()*4,cudaMemcpyDeviceToHost,execution_.stream()),"download logits");
            execution_.synchronize("model completion");
            finish_cow_timing(timed);
            for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite CUDA logits");
        }catch(...){execution_.synchronize_noexcept();table_state.dirty=true;append.rollback();throw;}
        append.commit();table_state.dirty=false;stats_.logits_downloaded_bytes+=result.size()*4;return result;
    }
    DecodeBatch prepare_decode_batch(const std::vector<DecodeRow>& rows) override{
        const auto& c=config();
        const int n=static_cast<int>(rows.size()),kv=c.kv_width(),dim=c.head_dimension();
        std::vector<TokenId> tokens;
        std::vector<int> positions;
        for(const auto& [id,token]:rows){
            if(token<0||token>=c.vocabulary)throw std::invalid_argument("invalid decode token");
            tokens.push_back(token);
            positions.push_back(kv_.sequence(id).used);
        }
        const auto stride=std::size_t(n)*kv;
        workspace_.ensure(n,c,stats_);
        staged_keys_.ensure(stride*c.layers*sizeof(float),stats_);
        staged_values_.ensure(stride*c.layers*sizeof(float),stats_);
        std::vector<float> logits(std::size_t(n)*c.vocabulary);
        batch_logits_.ensure(logits.size()*sizeof(float),stats_);
        auto stream=execution_.stream();
        try{
            for(const auto& row:rows){
                auto& table=*block_tables_.at(row.first);
                const auto& blocks=kv_.sequence(row.first).blocks;
                if(table.dirty&&!blocks.empty()){
                    const auto bytes=blocks.size()*sizeof(int);
                    check(cudaMemcpyAsync(table.data,blocks.data(),bytes,cudaMemcpyHostToDevice,stream),"restore history table");
                    stats_.block_table_uploaded_bytes+=bytes;
                }
            }
            check(cudaMemcpyAsync(workspace_.tokens.data,tokens.data(),n*sizeof(TokenId),cudaMemcpyHostToDevice,stream),"upload decode batch");
            stats_.token_uploaded_bytes+=n*sizeof(TokenId);
            embedding_kernel<<<(n*c.hidden+255)/256,256,0,stream>>>(workspace_.x.data,parameter("model.embed_tokens.weight"),
                reinterpret_cast<int*>(workspace_.tokens.data),n,c.hidden);
            for(int layer=0;layer<c.layers;++layer){
                auto p="model.layers."+std::to_string(layer)+".";
                project_qkv(p,n);
                for(int i=0;i<n;++i){
                    auto q=workspace_.q.data+std::size_t(i)*c.hidden;
                    auto k=workspace_.k.data+std::size_t(i)*kv;
                    auto v=workspace_.v.data+std::size_t(i)*kv;
                    rope_kernel<<<(c.hidden/2+255)/256,256,0,stream>>>(q,1,c.heads,dim,positions[i],c.rope_theta);
                    rope_kernel<<<(kv/2+255)/256,256,0,stream>>>(k,1,c.kv_heads,dim,positions[i],c.rope_theta);
                    const auto* table=reinterpret_cast<int*>(block_tables_.at(rows[i].first)->data);
                    attention_kernel<<<dim3(1,c.heads),256,0,stream>>>(q,keys_.data,values_.data,
                        workspace_.attention.data+std::size_t(i)*c.hidden,table,c.heads,c.kv_heads,dim,
                        positions[i],layer,c.layers,kv_.config().block_tokens,k,v);
                }
                check(cudaMemcpyAsync(staged_keys_.data+layer*stride,workspace_.k.data,stride*4,cudaMemcpyDeviceToDevice,stream),"stage keys");
                check(cudaMemcpyAsync(staged_values_.data+layer*stride,workspace_.v.data,stride*4,cudaMemcpyDeviceToDevice,stream),"stage values");
                finish_layer(p,n);
            }
            rms_kernel<<<n,256,0,stream>>>(workspace_.x.data,parameter("model.norm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
            linear(workspace_.norm,"lm_head.weight",batch_logits_,n,c.hidden,c.vocabulary);
            check(cudaGetLastError(),"batched model launch");
            check(cudaMemcpyAsync(logits.data(),batch_logits_.data,logits.size()*4,cudaMemcpyDeviceToHost,stream),"download batch logits");
            execution_.synchronize("batched model completion");
            stats_.logits_downloaded_bytes+=logits.size()*4;
        }catch(...){execution_.synchronize_noexcept();throw;}
        return [this,rows,logits=std::move(logits),stride,kv](std::size_t index){
            const auto& [id,token]=rows.at(index);
            auto append=kv_.prepare_append(id,{token});
            auto& table=*block_tables_.at(id);
            auto begin=logits.begin()+index*config().vocabulary;
            std::vector<float> result(begin,begin+config().vocabulary);
            for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite CUDA logits");
            const auto& s=append.sequence();
            if(append.tail_copy())++costs_.cow_pages;
            try{
                prepare_pages(append,table);
                const auto dst=kv_.offset(s,0,s.used);
                const auto pitch=std::size_t(kv_.config().block_tokens)*kv*sizeof(float);
                for(auto pools:{std::make_pair(keys_.data,staged_keys_.data),std::make_pair(values_.data,staged_values_.data)})
                    check(cudaMemcpy2DAsync(pools.first+dst,pitch,pools.second+index*kv,stride*sizeof(float),
                        kv*sizeof(float),config().layers,cudaMemcpyDeviceToDevice,execution_.stream()),"commit batch KV");
                execution_.synchronize("batched row commit");
                finish_cow_timing(cost_timing_&&bool(append.tail_copy()));
            }catch(...){execution_.synchronize_noexcept();table.dirty=true;throw;}
            append.commit();table.dirty=false;return result;
        };
    }
    void release(const RequestId& id) override{execution_.synchronize("release completion");
        kv_.release(id);block_tables_.erase(id);}
    void clear() override{execution_.synchronize("clear completion");
        kv_.clear();block_tables_.clear();}
    RunnerStats stats() const override{auto result=stats_;
        result.update_kv_stats(kv_.stats());
        for(const auto& item:block_tables_)result.block_table_device_bytes+=item.second->bytes;
        result.device_workspace_bytes=workspace_.bytes()+batch_logits_.bytes+staged_keys_.bytes+staged_values_.bytes;
        return result;}
};
}
std::unique_ptr<ModelRunner> make_cuda_runner(std::shared_ptr<ModelWeights> weights,int context,KvCacheConfig kv){return std::make_unique<CudaRunner>(std::move(weights),context,kv);}
} // namespace miniprefix
