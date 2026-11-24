#include "miniprefix/model.h"
#include "resources.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <limits>
#include <type_traits>

namespace miniprefix {
namespace {
using cuda_detail::check;
using cuda_detail::blas;
using cuda_detail::Allocation;
__global__ void embedding_kernel(float* x,const float* weights,const int* tokens,int rows,int width){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width)x[i]=weights[std::size_t(tokens[i/width])*width+i%width];}
template<bool AddResidual>
__global__ void rms_kernel(float* x,const float* weight,float* out,int width,float eps,const int* source_rows=nullptr,const float* residual=nullptr){
    __shared__ float sums[256];const int row=blockIdx.x,lane=threadIdx.x;float sum=0;
    const int source_row=source_rows?source_rows[row]:row;
    float* source=x+source_row*width;
    for(int d=lane;d<width;d+=256){
        float v=source[d];
        if constexpr(AddResidual){
            v+=residual[source_row*width+d];
            source[d]=v; // Keep the rounded FP32 residual for the next sublayer.
        }
        sum+=v*v;
    }sums[lane]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(lane<step)sums[lane]+=sums[lane+step];__syncthreads();}
    const float scale=1.0F/sqrtf(sums[0]/width+eps);
    for(int d=lane;d<width;d+=256)out[row*width+d]=source[d]*scale*weight[d];}
__global__ void rope_kernel(float* x,int count,int heads,int dim,int start,float theta,const int* positions){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*heads*(dim/2))return;
    int d=i%(dim/2),headrow=i/(dim/2),row=headrow/heads;
    float angle=float(positions?positions[row]:start+row)/powf(theta,float(2*d)/dim);float cs=cosf(angle),sn=sinf(angle);
    int a=headrow*dim+d,b=a+dim/2;float va=x[a],vb=x[b];x[a]=va*cs-vb*sn;x[b]=vb*cs+va*sn;}
__global__ void append_kernel(const float* k,const float* v,float* keys,float* values,const int* table,
                              int n,int width,int start,int layer,int layers,int block_tokens,const int* positions,const int* const* tables){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*width){const int row=i/width,token=positions?positions[row]:start+row;
        if(tables)table=tables[row];
        const auto offset=((std::size_t(table[token/block_tokens])*layers+layer)*block_tokens+token%block_tokens)*width+i%width;
        keys[offset]=k[i];values[offset]=v[i];}}
__global__ void attention_kernel(const float* q,const float* k,const float* v,float* out,const int* table,
                                 int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens,const int* positions,const int* const* tables){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads),length=(positions?positions[row]:start+row)+1;
    if(tables)table=tables[row];
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
__global__ void swiglu_kernel(float* gate,const float* up,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=gate[i];gate[i]=(v/(1+expf(-v)))*up[i];}}
// One block per output row. Any nonfinite logit invalidates the whole row.
__global__ void greedy_kernel(const float* logits,int vocabulary,TokenId* ids){
    const int lane=threadIdx.x;
    const float* row=logits+std::size_t(blockIdx.x)*vocabulary;
    float best=-CUDART_INF_F;
    int id=0;
    bool invalid=false;
    for(int i=lane;i<vocabulary;i+=blockDim.x){
        const float value=row[i];
        invalid|=!isfinite(value);
        if(value>best){best=value;id=i;}
    }
    if(__syncthreads_or(invalid)){
        if(lane==0)ids[blockIdx.x]=-1;
        return;
    }
    __shared__ float values[256];
    __shared__ int indices[256];
    values[lane]=best;indices[lane]=id;
    __syncthreads();
    for(int stride=128;stride;stride/=2){
        if(lane<stride){
            const int other=lane+stride;
            if(values[other]>values[lane] || (values[other]==values[lane] && indices[other]<indices[lane])){
                values[lane]=values[other];indices[lane]=indices[other];
            }
        }
        __syncthreads();
    }
    if(lane==0)ids[blockIdx.x]=indices[0];
}
class CudaRunner final:public ModelRunner {
    std::shared_ptr<ModelWeights> weights_;RunnerStats stats_;
    KVManager kv_;
    CostStats costs_;bool cost_timing_=false;
    DecodeBatchStats decode_stats_;
    PrefillBatchStats prefill_stats_;
    struct TimingEvents {
        cudaEvent_t begin=nullptr,end=nullptr;
        TimingEvents(){check(cudaEventCreate(&begin),"create timing start event");
            try{check(cudaEventCreate(&end),"create timing end event");}catch(...){cudaEventDestroy(begin);throw;}}
        ~TimingEvents(){cudaEventDestroy(begin);cudaEventDestroy(end);}
    };
    std::unique_ptr<TimingEvents> copy_events_,forward_events_;
    ExecutionStats execution_stats_;
    std::size_t sampled_bytes_=0;
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
    void forward(int n,int start,const int* table,bool batch=false,int outputs=1){
        const auto& c=config();
        const auto* positions=batch?reinterpret_cast<const int*>(workspace_.positions.data):nullptr;
        const auto* tables=batch?reinterpret_cast<const int* const*>(workspace_.tables.data):nullptr;
        embedding_kernel<<<(n*c.hidden+255)/256,256,0,execution_.stream()>>>(workspace_.x.data,parameter("model.embed_tokens.weight"),reinterpret_cast<int*>(workspace_.tokens.data),n,c.hidden);
        for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
            // MLP output from the previous layer is still in projection.
            if(l==0)rms_kernel<false><<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"input_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon);
            else rms_kernel<true><<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"input_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon,nullptr,workspace_.projection.data);
            linear(workspace_.norm,p+"self_attn.q_proj.weight",workspace_.q,n,c.hidden,c.hidden);
            linear(workspace_.norm,p+"self_attn.k_proj.weight",workspace_.k,n,c.hidden,c.kv_width());linear(workspace_.norm,p+"self_attn.v_proj.weight",workspace_.v,n,c.hidden,c.kv_width());
            rope_kernel<<<(n*c.heads*c.head_dimension()/2+255)/256,256,0,execution_.stream()>>>(workspace_.q.data,n,c.heads,c.head_dimension(),start,c.rope_theta,positions);
            rope_kernel<<<(n*c.kv_width()/2+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,n,c.kv_heads,c.head_dimension(),start,c.rope_theta,positions);
            append_kernel<<<(n*c.kv_width()+255)/256,256,0,execution_.stream()>>>(workspace_.k.data,workspace_.v.data,keys_.data,values_.data,table,
                n,c.kv_width(),start,l,c.layers,kv_.config().block_tokens,positions,tables);
            attention_kernel<<<dim3(n,c.heads),256,0,execution_.stream()>>>(workspace_.q.data,keys_.data,values_.data,workspace_.attention.data,table,
                c.heads,c.kv_heads,c.head_dimension(),start,l,c.layers,kv_.config().block_tokens,positions,tables);
            linear(workspace_.attention,p+"self_attn.o_proj.weight",workspace_.projection,n,c.hidden,c.hidden);
            rms_kernel<true><<<n,256,0,execution_.stream()>>>(workspace_.x.data,parameter(p+"post_attention_layernorm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon,nullptr,workspace_.projection.data);
            linear(workspace_.norm,p+"mlp.gate_proj.weight",workspace_.gate,n,c.hidden,c.intermediate);linear(workspace_.norm,p+"mlp.up_proj.weight",workspace_.up,n,c.hidden,c.intermediate);
            swiglu_kernel<<<(n*c.intermediate+255)/256,256,0,execution_.stream()>>>(workspace_.gate.data,workspace_.up.data,n*c.intermediate);
            linear(workspace_.gate,p+"mlp.down_proj.weight",workspace_.projection,n,c.intermediate,c.hidden);
        }
        const int* last_rows=batch && outputs!=n?reinterpret_cast<const int*>(workspace_.last_rows.data):nullptr;
        // Only chunk-final rows are consumed by the head. KV for all input rows
        // was already written before this final residual addition.
        const auto offset=batch?std::size_t(0):std::size_t(n-1)*c.hidden;
        rms_kernel<true><<<outputs,256,0,execution_.stream()>>>(workspace_.x.data+offset,parameter("model.norm.weight"),workspace_.norm.data,c.hidden,c.rms_epsilon,last_rows,workspace_.projection.data+offset);
        linear(workspace_.norm,"lm_head.weight",workspace_.logits,outputs,c.hidden,c.vocabulary);
        check(cudaGetLastError(),"model kernel launch");
    }
    // CUDA events bound the forward stream interval. Host submission overlaps
    // it. The outer wall timer includes pageable logits D2H and completion.
    template<class T> void forward_download(int rows,int start,const int* table,bool batch,int outputs,
                          std::vector<T>& host){
        using Clock=std::chrono::steady_clock;
        Clock::time_point begin,submitted;
        if(forward_events_){
            begin=Clock::now();
            check(cudaEventRecord(forward_events_->begin,execution_.stream()),"forward timer begin");
        }
        forward(rows,start,table,batch,outputs);
        if(forward_events_){
            submitted=Clock::now();
            check(cudaEventRecord(forward_events_->end,execution_.stream()),"forward timer end");
        }
        const void* source=workspace_.logits.data;
        if constexpr(std::is_same_v<T,TokenId>){
            // Embedding consumed the input IDs earlier on this stream. There are
            // at least as many input rows as output rows, even for packed prefill.
            auto* ids=reinterpret_cast<TokenId*>(workspace_.tokens.data);
            greedy_kernel<<<outputs,256,0,execution_.stream()>>>(workspace_.logits.data,config().vocabulary,ids);
            check(cudaGetLastError(),"greedy kernel launch");
            source=ids;
        }
        check(cudaMemcpyAsync(host.data(),source,host.size()*sizeof(T),cudaMemcpyDeviceToHost,execution_.stream()),"download forward result");
        execution_.synchronize("model completion");
        if constexpr(std::is_same_v<T,TokenId>)sampled_bytes_+=host.size()*sizeof(T);
        if(forward_events_){
            const auto end=Clock::now();
            float elapsed=0;
            check(cudaEventElapsedTime(&elapsed,forward_events_->begin,forward_events_->end),"forward elapsed");
            ++execution_stats_.forwards;
            execution_stats_.host_forward_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(submitted-begin).count();
            execution_stats_.forward_download_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
            execution_stats_.stream_forward_ns+=static_cast<std::uint64_t>(elapsed*1e6);
        }
    }
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
    DecodeBatchStats decode_batch_stats() const override{return decode_stats_;}
    bool supports_packed_prefill() const override{return true;}
    PrefillBatchStats prefill_batch_stats() const override{return prefill_stats_;}
    std::size_t sampled_id_downloaded_bytes() const override{return sampled_bytes_;}
    ExecutionStats execution_stats() const override{return execution_stats_;}
    void set_execution_timing(bool on) override{
        if(on && !forward_events_)forward_events_=std::make_unique<TimingEvents>();
        if(!on)forward_events_.reset();
    }
    CostStats cost_stats() const override{
        auto result=kv_.cost_stats();result.cow_pages=costs_.cow_pages;result.cow_bytes=costs_.cow_bytes;
        result.cow_copy_ns=costs_.cow_copy_ns;result.cow_timed_copies=costs_.cow_timed_copies;return result;
    }
    void set_cost_timing(bool enabled) override{
        if(enabled&&!copy_events_)copy_events_=std::make_unique<TimingEvents>();
        cost_timing_=enabled;kv_.set_cost_timing(enabled);
    }
    KvCapacityLedger capacity_ledger() const override{return kv_.capacity_ledger();}
    int context_limit() const override{return kv_.context_limit();}
    KvAdmission reserve_request(const RequestId& id,std::size_t tokens) override{return kv_.reserve_request(id,tokens);}
    KvAdmission reserve_request(const RequestId& id,std::size_t tokens,const std::vector<TokenId>& prompt) override{
        return kv_.reserve_request(id,tokens,prompt);
    }
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
        return append_one<float>(id,tokens);
    }
    GenerationResult append_generation(const RequestId& id,const std::vector<TokenId>& tokens) override{
        return append_one<TokenId>(id,tokens).front();
    }
    template<class T> bool valid_result(const std::vector<T>& host) const{
        if constexpr(std::is_same_v<T,TokenId>)return std::all_of(host.begin(),host.end(),[](TokenId id){return id>=0;});
        else return std::all_of(host.begin(),host.end(),[](float value){return std::isfinite(value);});
    }
    template<class T> std::vector<T> append_one(const RequestId& id,const std::vector<TokenId>& tokens){
        auto append=kv_.prepare_append(id,tokens);const auto& state=append.sequence();
        const auto& c=config();const int n=static_cast<int>(tokens.size());
        std::vector<T> result(std::is_same_v<T,TokenId>?1:c.vocabulary);
        const auto original_blocks=append.original_blocks();
        auto& table_state=*block_tables_.at(id);
        const bool timed=cost_timing_&&bool(append.tail_copy());
        if(append.tail_copy())++costs_.cow_pages;
        try{
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
            workspace_.ensure(n,c,stats_);check(cudaMemcpyAsync(workspace_.tokens.data,tokens.data(),tokens.size()*sizeof(TokenId),cudaMemcpyHostToDevice,execution_.stream()),"upload input tokens");
            stats_.token_uploaded_bytes+=tokens.size()*sizeof(TokenId);
            forward_download(n,state.used,reinterpret_cast<const int*>(table_state.data),false,1,result);
            if(timed){float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,copy_events_->begin,copy_events_->end),"COW elapsed");
                costs_.cow_copy_ns+=static_cast<std::uint64_t>(milliseconds*1e6);++costs_.cow_timed_copies;}
            if(!valid_result(result))throw std::runtime_error("nonfinite CUDA logits");
        }catch(...){execution_.synchronize_noexcept();table_state.dirty=true;append.rollback();throw;}
        append.commit();table_state.dirty=false;
        if constexpr(std::is_same_v<T,float>)stats_.logits_downloaded_bytes+=result.size()*sizeof(float);
        return result;
    }
    std::optional<std::vector<std::vector<float>>> append_decode_batch(const std::vector<DecodeInput>& inputs) override{
        std::vector<BatchInput> chunks;
        for(const auto& input:inputs)chunks.push_back({input.id,{input.token},true});
        return append_batch(chunks);
    }
    std::optional<std::vector<std::vector<float>>> append_batch(const std::vector<BatchInput>& inputs) override{
        return append_group<float,std::vector<float>>(inputs);
    }
    std::optional<std::vector<GenerationResult>> append_generation_batch(const std::vector<BatchInput>& inputs) override{
        return append_group<TokenId,GenerationResult>(inputs);
    }
    template<class T,class Row> std::optional<std::vector<Row>> append_group(const std::vector<BatchInput>& inputs){
        const auto& c=config();
        if(inputs.size()<2)return std::nullopt;
        const auto decodes=std::count_if(inputs.begin(),inputs.end(),[](const BatchInput& input){return input.decode;});
        const bool pure_decode=std::size_t(decodes)==inputs.size();
        auto& fallbacks=pure_decode?decode_stats_.fallbacks:prefill_stats_.fallbacks;
        const auto limit=std::size_t((std::numeric_limits<int>::max()-255)/std::max({c.hidden,c.intermediate,c.vocabulary}));
        std::size_t rows=0;
        for(const auto& input:inputs){
            if(input.tokens.empty() || (input.decode && input.tokens.size()!=1) || input.tokens.size()>limit-rows){
                ++fallbacks;return std::nullopt;
            }
            rows+=input.tokens.size();
        }
        const int n=static_cast<int>(inputs.size());
        auto transactions=kv_.prepare_batch(inputs);
        if(transactions.empty()){++fallbacks;return std::nullopt;}
        std::vector<Row> result;result.reserve(n);
        std::vector<T> flat(std::size_t(n)*(std::is_same_v<T,TokenId>?1:c.vocabulary));
        std::vector<int> tokens,positions,last_rows;
        std::vector<const int*> tables;
        tokens.reserve(rows);positions.reserve(rows);tables.reserve(rows);last_rows.reserve(n);
        std::vector<std::unique_ptr<TimingEvents>> timing(n);
        auto rollback=[&]{
            for(int row=n-1;row>=0;--row){block_tables_.at(inputs[row].id)->dirty=true;transactions[row]->rollback();}
        };
        try{
            workspace_.ensure(static_cast<int>(rows),c,stats_,true,n);
            for(int row=0;row<n;++row){
                auto& tx=*transactions[row];const auto& state=tx.sequence();
                auto& table=*block_tables_.at(inputs[row].id);
                int position=state.used;
                for(auto token:inputs[row].tokens){
                    tokens.push_back(token);
                    positions.push_back(position++);
                    tables.push_back(reinterpret_cast<const int*>(table.data));
                }
                last_rows.push_back(static_cast<int>(tokens.size()-1));
                if(tx.tail_copy()){
                    ++costs_.cow_pages;
                    if(cost_timing_){timing[row]=std::make_unique<TimingEvents>();check(cudaEventRecord(timing[row]->begin,execution_.stream()),"batch COW begin");}
                }
                tx.copy_shared_tail_region([&](const KVManager::TailCopyRegion& r){
                    const auto pitch=r.pitch*sizeof(float),width=r.width*sizeof(float);
                    check(cudaMemcpy2DAsync(keys_.data+r.destination,pitch,keys_.data+r.source,pitch,width,r.rows,cudaMemcpyDeviceToDevice,execution_.stream()),"batch COW keys");
                    check(cudaMemcpy2DAsync(values_.data+r.destination,pitch,values_.data+r.source,pitch,width,r.rows,cudaMemcpyDeviceToDevice,execution_.stream()),"batch COW values");
                    costs_.cow_bytes+=2*width*r.rows;
                });
                if(timing[row])check(cudaEventRecord(timing[row]->end,execution_.stream()),"batch COW end");
                const auto first=table.dirty?std::size_t(0):(tx.tail_copy()?tx.original_blocks()-1:tx.original_blocks());
                if(state.blocks.size()!=first){const auto bytes=(state.blocks.size()-first)*sizeof(int);
                    check(cudaMemcpyAsync(reinterpret_cast<int*>(table.data)+first,state.blocks.data()+first,bytes,cudaMemcpyHostToDevice,execution_.stream()),"batch table entries");
                    stats_.block_table_uploaded_bytes+=bytes;}
            }
            check(cudaMemcpyAsync(workspace_.tokens.data,tokens.data(),rows*sizeof(int),cudaMemcpyHostToDevice,execution_.stream()),"batch tokens");
            check(cudaMemcpyAsync(workspace_.positions.data,positions.data(),rows*sizeof(int),cudaMemcpyHostToDevice,execution_.stream()),"batch positions");
            check(cudaMemcpyAsync(workspace_.tables.data,tables.data(),rows*sizeof(const int*),cudaMemcpyHostToDevice,execution_.stream()),"batch tables");
            if(rows!=inputs.size())check(cudaMemcpyAsync(workspace_.last_rows.data,last_rows.data(),n*sizeof(int),cudaMemcpyHostToDevice,execution_.stream()),"batch output indices");
            stats_.token_uploaded_bytes+=rows*sizeof(int);
            forward_download(static_cast<int>(rows),0,nullptr,true,n,flat);
            for(const auto& t:timing)if(t){float ms=0;check(cudaEventElapsedTime(&ms,t->begin,t->end),"batch COW elapsed");
                costs_.cow_copy_ns+=static_cast<std::uint64_t>(ms*1e6);++costs_.cow_timed_copies;}
            // Replay row-specific numerical failures through ordinary append,
            // preserving successful earlier events and the original error order.
            if(!valid_result(flat)){rollback();++fallbacks;return std::nullopt;}
            for(int row=0;row<n;++row){
                if constexpr(std::is_same_v<T,TokenId>)result.emplace_back(flat[row]);
                else{
                    auto begin=flat.begin()+std::size_t(row)*c.vocabulary;
                    result.emplace_back(begin,begin+c.vocabulary);
                }
            }
        }catch(...){execution_.synchronize_noexcept();rollback();throw;}
        for(int row=0;row<n;++row){transactions[row]->commit();block_tables_.at(inputs[row].id)->dirty=false;}
        if constexpr(std::is_same_v<T,float>)stats_.logits_downloaded_bytes+=flat.size()*sizeof(float);
        if(pure_decode){
            ++decode_stats_.batches;decode_stats_.rows+=n;
        }else{
            if(decodes)++prefill_stats_.mixed_batches;
            else ++prefill_stats_.prefill_batches;
            prefill_stats_.decode_tokens+=decodes;
            prefill_stats_.prefill_tokens+=rows-decodes;
        }
        return result;
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
