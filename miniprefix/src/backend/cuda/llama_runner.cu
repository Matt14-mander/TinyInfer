#include "miniprefix/model.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace miniprefix {
namespace {
void check(cudaError_t s,const char* what){if(s!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(s));}
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("MiniPrefix cuBLAS failure "+std::to_string(int(s)));}
struct Allocation {
    float* data=nullptr;std::size_t bytes=0;
    ~Allocation(){if(data)cudaFree(data);}
    Allocation()=default;Allocation(const Allocation&)=delete;Allocation& operator=(const Allocation&)=delete;
    void ensure(std::size_t size,RunnerStats& stats){if(size<=bytes)return;float* next=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&next),size),"allocate");
        if(data){auto status=cudaFree(data);if(status!=cudaSuccess){cudaFree(next);check(status,"free old workspace");}}
        data=next;bytes=size;++stats.device_allocations;}
};
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
__global__ void append_kernel(const float* k,const float* v,float* keys,float* values,int n,int width,int start){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*width){keys[start*width+i]=k[i];values[start*width+i]=v[i];}}
__global__ void attention_kernel(const float* q,const float* k,const float* v,float* out,int heads,int kvheads,int dim,int start){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads),length=start+row+1;
    __shared__ float reduction[256];
    float maximum=-CUDART_INF_F,denom=0,numerator=0;
    const float query=lane<dim?q[(row*heads+head)*dim+lane]:0;
    for(int t=0;t<length;++t){
        const std::size_t offset=(std::size_t(t)*kvheads+kh)*dim;
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
struct Sequence{Allocation keys,values;int used=0;};
class CudaRunner final:public ModelRunner {
    std::shared_ptr<ModelWeights> weights_;int context_;RunnerStats stats_;
    cudaStream_t stream_=nullptr;cublasHandle_t blas_=nullptr;
    std::map<std::string,std::unique_ptr<Allocation>> parameters_;
    std::map<RequestId,std::unique_ptr<Sequence>> sequences_;
    Allocation x_,norm_,q_,k_,v_,attention_,projection_,gate_,up_,logits_,tokens_;
    float* parameter(const std::string& name){return parameters_.at(name)->data;}
    void linear(const Allocation& x,const std::string& name,Allocation& out,int rows,int inner,int columns){
        const float alpha=1,beta=0;
        blas(cublasSgemm(blas_,CUBLAS_OP_N,CUBLAS_OP_N,columns,rows,inner,&alpha,parameter(name),columns,x.data,inner,&beta,out.data,columns));}
    void upload_parameter(const std::string& name,const tinyinfer::Tensor& input){
        auto data=input;data.contiguous();auto memory=std::make_unique<Allocation>();memory->ensure(data.size_bytes(),stats_);
        check(cudaMemcpyAsync(memory->data,data.data<float>(),data.size_bytes(),cudaMemcpyHostToDevice,stream_),"upload weights");
        check(cudaStreamSynchronize(stream_),"weight upload completion");
        stats_.weight_uploaded_bytes+=data.size_bytes();stats_.device_weight_bytes+=data.size_bytes();parameters_.emplace(name,std::move(memory));}
    void workspace(int rows){
        auto& c=config();const auto hidden=std::size_t(rows)*c.hidden*4,kv=std::size_t(rows)*c.kv_width()*4,ff=std::size_t(rows)*c.intermediate*4;
        x_.ensure(hidden,stats_);norm_.ensure(hidden,stats_);q_.ensure(hidden,stats_);k_.ensure(kv,stats_);v_.ensure(kv,stats_);
        attention_.ensure(hidden,stats_);projection_.ensure(hidden,stats_);gate_.ensure(ff,stats_);up_.ensure(ff,stats_);
        logits_.ensure(std::size_t(c.vocabulary)*4,stats_);tokens_.ensure(std::size_t(rows)*sizeof(int),stats_);}
public:
    CudaRunner(std::shared_ptr<ModelWeights> weights,int context):weights_(std::move(weights)),context_(context){
        if(context<1||context>config().context)throw std::invalid_argument("invalid runner context");
        check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create model stream");
        try{blas(cublasCreate(&blas_));blas(cublasSetStream(blas_,stream_));blas(cublasSetMathMode(blas_,CUBLAS_PEDANTIC_MATH));blas(cublasSetAtomicsMode(blas_,CUBLAS_ATOMICS_NOT_ALLOWED));
            for(const auto& p:weights_->tensors)upload_parameter(p.first,p.second);
            if(config().tied){auto head=weights_->get("model.embed_tokens.weight");head.transpose(0,1);upload_parameter("lm_head.weight",head);}
        }catch(...){if(blas_)cublasDestroy(blas_);cudaStreamSynchronize(stream_);cudaStreamDestroy(stream_);stream_=nullptr;throw;}
    }
    ~CudaRunner() override{if(stream_)cudaStreamSynchronize(stream_);sequences_.clear();parameters_.clear();if(blas_)cublasDestroy(blas_);if(stream_)cudaStreamDestroy(stream_);}
    const ModelConfig& config() const override{return weights_->config;}
    void create(const RequestId& id) override{
        if(sequences_.count(id))throw std::invalid_argument("sequence already exists");
        auto state=std::make_unique<Sequence>();const auto bytes=std::size_t(config().layers)*context_*config().kv_width()*4;
        state->keys.ensure(bytes,stats_);state->values.ensure(bytes,stats_);sequences_.emplace(id,std::move(state));
        stats_.live_sequences=sequences_.size();stats_.live_kv_bytes+=bytes*2;stats_.peak_kv_bytes=std::max(stats_.peak_kv_bytes,stats_.live_kv_bytes);}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        auto& state=*sequences_.at(id);const auto& c=config();const int n=static_cast<int>(tokens.size());
        if(n<1||n>context_-state.used)throw std::invalid_argument("context overflow/empty input");
        for(auto token:tokens)if(token<0||token>=c.vocabulary)throw std::invalid_argument("invalid token");
        std::vector<float> result(c.vocabulary);
        try{
            workspace(n);check(cudaMemcpyAsync(tokens_.data,tokens.data(),tokens.size()*sizeof(TokenId),cudaMemcpyHostToDevice,stream_),"upload input tokens");
            stats_.token_uploaded_bytes+=tokens.size()*sizeof(TokenId);
            embedding_kernel<<<(n*c.hidden+255)/256,256,0,stream_>>>(x_.data,parameter("model.embed_tokens.weight"),reinterpret_cast<int*>(tokens_.data),n,c.hidden);
            for(int l=0;l<c.layers;++l){auto p="model.layers."+std::to_string(l)+".";
                rms_kernel<<<n,256,0,stream_>>>(x_.data,parameter(p+"input_layernorm.weight"),norm_.data,c.hidden,c.rms_epsilon);
                linear(norm_,p+"self_attn.q_proj.weight",q_,n,c.hidden,c.hidden);
                linear(norm_,p+"self_attn.k_proj.weight",k_,n,c.hidden,c.kv_width());linear(norm_,p+"self_attn.v_proj.weight",v_,n,c.hidden,c.kv_width());
                rope_kernel<<<(n*c.heads*c.head_dimension()/2+255)/256,256,0,stream_>>>(q_.data,n,c.heads,c.head_dimension(),state.used,c.rope_theta);
                rope_kernel<<<(n*c.kv_width()/2+255)/256,256,0,stream_>>>(k_.data,n,c.kv_heads,c.head_dimension(),state.used,c.rope_theta);
                auto* keys=state.keys.data+std::size_t(l)*context_*c.kv_width();auto* values=state.values.data+std::size_t(l)*context_*c.kv_width();
                append_kernel<<<(n*c.kv_width()+255)/256,256,0,stream_>>>(k_.data,v_.data,keys,values,n,c.kv_width(),state.used);
                attention_kernel<<<dim3(n,c.heads),256,0,stream_>>>(q_.data,keys,values,attention_.data,c.heads,c.kv_heads,c.head_dimension(),state.used);
                linear(attention_,p+"self_attn.o_proj.weight",projection_,n,c.hidden,c.hidden);
                add_kernel<<<(n*c.hidden+255)/256,256,0,stream_>>>(x_.data,projection_.data,n*c.hidden);
                rms_kernel<<<n,256,0,stream_>>>(x_.data,parameter(p+"post_attention_layernorm.weight"),norm_.data,c.hidden,c.rms_epsilon);
                linear(norm_,p+"mlp.gate_proj.weight",gate_,n,c.hidden,c.intermediate);linear(norm_,p+"mlp.up_proj.weight",up_,n,c.hidden,c.intermediate);
                swiglu_kernel<<<(n*c.intermediate+255)/256,256,0,stream_>>>(gate_.data,up_.data,n*c.intermediate);
                linear(gate_,p+"mlp.down_proj.weight",projection_,n,c.intermediate,c.hidden);
                add_kernel<<<(n*c.hidden+255)/256,256,0,stream_>>>(x_.data,projection_.data,n*c.hidden);
            }
            rms_kernel<<<1,256,0,stream_>>>(x_.data+std::size_t(n-1)*c.hidden,parameter("model.norm.weight"),norm_.data,c.hidden,c.rms_epsilon);
            linear(norm_,"lm_head.weight",logits_,1,c.hidden,c.vocabulary);
            check(cudaGetLastError(),"model kernel launch");
            check(cudaMemcpyAsync(result.data(),logits_.data,result.size()*4,cudaMemcpyDeviceToHost,stream_),"download logits");
            check(cudaStreamSynchronize(stream_),"model completion");
        }catch(...){cudaStreamSynchronize(stream_);throw;}
        for(float v:result)if(!std::isfinite(v))throw std::runtime_error("nonfinite CUDA logits");
        state.used+=n;stats_.forward_tokens+=n;stats_.logits_downloaded_bytes+=result.size()*4;return result;
    }
    void release(const RequestId& id) override{check(cudaStreamSynchronize(stream_),"release completion");
        if(!sequences_.erase(id))throw std::invalid_argument("unknown sequence");stats_.live_sequences=sequences_.size();stats_.live_kv_bytes=sequences_.size()*std::size_t(config().layers)*context_*config().kv_width()*8;}
    void clear() override{check(cudaStreamSynchronize(stream_),"clear completion");sequences_.clear();stats_.live_sequences=stats_.live_kv_bytes=0;}
    RunnerStats stats() const override{auto result=stats_;
        result.device_workspace_bytes=x_.bytes+norm_.bytes+q_.bytes+k_.bytes+v_.bytes+attention_.bytes+projection_.bytes+gate_.bytes+up_.bytes+logits_.bytes+tokens_.bytes;
        return result;}
};
}
std::unique_ptr<ModelRunner> make_cuda_runner(std::shared_ptr<ModelWeights> weights,int context){return std::make_unique<CudaRunner>(std::move(weights),context);}
} // namespace miniprefix
