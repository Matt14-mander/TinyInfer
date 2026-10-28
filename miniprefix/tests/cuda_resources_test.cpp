#include "resources.h"
#include "miniprefix/model.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <type_traits>
using namespace miniprefix;
using namespace miniprefix::cuda_detail;
namespace {
static_assert(!std::is_copy_constructible<Allocation>::value,"device allocation has one owner");
static_assert(!std::is_move_constructible<Allocation>::value,"device allocation addresses stay stable");
static_assert(!std::is_copy_constructible<StreamBlas>::value,"stream and handle have one owner");
ModelConfig config(){
    ModelConfig c;c.layers=1;c.hidden=8;c.heads=2;c.kv_heads=1;c.intermediate=12;
    c.vocabulary=16;c.context=32;c.eos=0;c.rms_epsilon=1e-5F;c.rope_theta=10000;c.validate();return c;
}
void allocation(){
    RunnerStats stats;Allocation memory;memory.ensure(0,stats);assert(!memory.data&&stats.device_allocations==0);
    memory.ensure(16,stats);auto* first=memory.data;
    const float value=7;check(cudaMemcpy(first,&value,sizeof(value),cudaMemcpyHostToDevice),"test upload");
    memory.ensure(8,stats);memory.ensure(16,stats);
    float actual=0;check(cudaMemcpy(&actual,memory.data,sizeof(actual),cudaMemcpyDeviceToHost),"test download");
    assert(memory.data==first&&memory.bytes==16&&actual==7&&stats.device_allocations==1);
    memory.ensure(32,stats);assert(memory.bytes==32&&stats.device_allocations==2);
    first=memory.data;bool rejected=false;
    try{memory.ensure(std::numeric_limits<std::size_t>::max(),stats);}catch(const std::runtime_error&){rejected=true;}
    cudaGetLastError();assert(rejected&&memory.data==first&&memory.bytes==32&&stats.device_allocations==2);
}
void workspace(){
    RunnerStats stats;Workspace work;const auto c=config();assert(work.bytes()==0);
    const auto expected=[&](int rows){return std::size_t(rows)*(5*c.hidden+2*c.kv_width()+2*c.intermediate+1)*4+c.vocabulary*4;};
    work.ensure(2,c,stats);assert(stats.device_allocations==11&&work.bytes()==expected(2));
    auto* first=work.x.data;auto* logits=work.logits.data;
    work.ensure(1,c,stats);work.ensure(2,c,stats);
    assert(stats.device_allocations==11&&work.x.data==first&&work.bytes()==expected(2));
    work.ensure(4,c,stats);assert(stats.device_allocations==21&&work.bytes()==expected(4)&&work.logits.data==logits);
    assert(stats.device_weight_bytes==0&&stats.live_kv_bytes==0&&stats.block_table_device_bytes==0);
}
void execution_resources(){
    StreamBlas execution;
    for(int iteration=0;iteration<3;++iteration){
        execution.initialize();
        cudaStream_t bound=nullptr;blas(cublasGetStream(execution.handle(),&bound));assert(bound==execution.stream());
        unsigned flags=0;check(cudaStreamGetFlags(bound,&flags),"stream flags");assert(flags==cudaStreamNonBlocking);
        cublasMath_t math;blas(cublasGetMathMode(execution.handle(),&math));assert(math==CUBLAS_PEDANTIC_MATH);
        cublasAtomicsMode_t atomics;blas(cublasGetAtomicsMode(execution.handle(),&atomics));assert(atomics==CUBLAS_ATOMICS_NOT_ALLOWED);
        bool rejected=false;try{execution.initialize();}catch(const std::logic_error&){rejected=true;}assert(rejected);
        RunnerStats stats;Allocation a,b,out;a.ensure(4,stats);b.ensure(4,stats);out.ensure(4,stats);
        const float left=3,right=7,alpha=1,beta=0;float result=0;
        check(cudaMemcpyAsync(a.data,&left,4,cudaMemcpyHostToDevice,bound),"left");
        check(cudaMemcpyAsync(b.data,&right,4,cudaMemcpyHostToDevice,bound),"right");
        blas(cublasSgemm(execution.handle(),CUBLAS_OP_N,CUBLAS_OP_N,1,1,1,&alpha,a.data,1,b.data,1,&beta,out.data,1));
        check(cudaMemcpyAsync(&result,out.data,4,cudaMemcpyDeviceToHost,bound),"result");
        execution.synchronize("test completion");assert(result==21);
        execution.reset();execution.reset();assert(!execution.stream()&&!execution.handle());
    }
    // Exercise cleanup during unwinding, including an incomplete runner constructor.
    for(int i=0;i<3;++i){
        bool caught=false;
        try{StreamBlas resources;resources.initialize();RunnerStats stats;Workspace work;work.ensure(3,config(),stats);throw std::runtime_error("injected");}
        catch(const std::runtime_error&){caught=true;}assert(caught);
        auto weights=std::make_shared<ModelWeights>();weights->config=config();
        caught=false;try{auto runner=make_cuda_runner(weights,16,{4,8,0});}
        catch(const std::out_of_range&){caught=true;}assert(caught);
    }
}
}
int main(int argc,char** argv){
    const bool required=argc>1&&std::string(argv[1])=="--require-device";
    int count=0;const auto status=cudaGetDeviceCount(&count);
    if(status!=cudaSuccess||count==0){std::cerr<<"CUDA device unavailable\n";return required?1:77;}
    try{allocation();workspace();execution_resources();std::cout<<"CUDA resource lifecycle and workspace tests passed\n";return 0;}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
