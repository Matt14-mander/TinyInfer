#include "resources.h"
#include "miniprefix/model.h"

namespace miniprefix {
namespace cuda_detail {
void check(cudaError_t status,const char* what){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(status));
}
void blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("MiniPrefix cuBLAS failure "+std::to_string(int(status)));
}
Allocation::~Allocation(){if(data)cudaFree(data);}
void Allocation::ensure(std::size_t size,RunnerStats& stats){
    if(size<=bytes)return;
    float* next=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&next),size),"allocate");
    if(data){auto status=cudaFree(data);if(status!=cudaSuccess){cudaFree(next);check(status,"free old workspace");}}
    data=next;bytes=size;++stats.device_allocations;
}
StreamBlas::~StreamBlas(){reset();}
void StreamBlas::initialize(){
    if(stream_||handle_)throw std::logic_error("CUDA execution resources already initialized");
    check(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create model stream");
    try{
        blas(cublasCreate(&handle_));blas(cublasSetStream(handle_,stream_));
        blas(cublasSetMathMode(handle_,CUBLAS_PEDANTIC_MATH));
        blas(cublasSetAtomicsMode(handle_,CUBLAS_ATOMICS_NOT_ALLOWED));
    }catch(...){reset();throw;}
}
cudaStream_t StreamBlas::stream() const{return stream_;}
cublasHandle_t StreamBlas::handle() const{return handle_;}
void StreamBlas::synchronize(const char* what) const{check(cudaStreamSynchronize(stream_),what);}
void StreamBlas::synchronize_noexcept() const noexcept{if(stream_)cudaStreamSynchronize(stream_);}
void StreamBlas::reset() noexcept{
    if(handle_){cublasDestroy(handle_);handle_=nullptr;}
    if(stream_){cudaStreamSynchronize(stream_);cudaStreamDestroy(stream_);stream_=nullptr;}
}
void Workspace::ensure(int rows,const ModelConfig& c,RunnerStats& stats){
    const auto hidden=std::size_t(rows)*c.hidden*4,kv=std::size_t(rows)*c.kv_width()*4,ff=std::size_t(rows)*c.intermediate*4;
    x.ensure(hidden,stats);norm.ensure(hidden,stats);q.ensure(hidden,stats);k.ensure(kv,stats);v.ensure(kv,stats);
    attention.ensure(hidden,stats);projection.ensure(hidden,stats);gate.ensure(ff,stats);up.ensure(ff,stats);
    logits.ensure(std::size_t(c.vocabulary)*4,stats);tokens.ensure(std::size_t(rows)*sizeof(int),stats);
}
std::size_t Workspace::bytes() const{
    return x.bytes+norm.bytes+q.bytes+k.bytes+v.bytes+attention.bytes+projection.bytes+gate.bytes+up.bytes+logits.bytes+tokens.bytes;
}
} // namespace cuda_detail
} // namespace miniprefix
