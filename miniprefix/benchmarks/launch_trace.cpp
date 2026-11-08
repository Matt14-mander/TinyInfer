#include "miniprefix/model.h"
#include <cupti.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>

using namespace miniprefix;
namespace {
void check(CUptiResult status){
    if(status==CUPTI_SUCCESS)return;
    const char* message=nullptr;
    cuptiGetResultString(status,&message);
    std::fprintf(stderr,"CUPTI: %s\n",message);
    std::exit(2);
}
struct Kernel {
    uint64_t start,end;
    uint32_t correlation,stream;
    std::string name;
};
struct Range {std::string phase;uint64_t start,end;};
struct Copy {uint64_t start,end,bytes;};
std::vector<Kernel> kernels;
std::vector<Copy> downloads;
std::map<uint32_t,uint64_t> calls;
std::mutex records_mutex;
void CUPTIAPI allocate(uint8_t** buffer,size_t* size,size_t* limit){
    *size=2*1024*1024;*limit=0;*buffer=static_cast<uint8_t*>(std::malloc(*size));
    if(!*buffer)std::abort();
}
void CUPTIAPI collect(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid){
    std::lock_guard<std::mutex> lock(records_mutex);
    CUpti_Activity* record=nullptr;
    CUptiResult status;
    while((status=cuptiActivityGetNextRecord(buffer,valid,&record))==CUPTI_SUCCESS){
        if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
            auto* k=reinterpret_cast<CUpti_ActivityKernel9*>(record);
            kernels.push_back({k->start,k->end,k->correlationId,k->streamId,k->name});
        }else if(record->kind==CUPTI_ACTIVITY_KIND_MEMCPY){
            auto* copy=reinterpret_cast<CUpti_ActivityMemcpy5*>(record);
            if(copy->copyKind==CUPTI_ACTIVITY_MEMCPY_KIND_DTOH)
                downloads.push_back({copy->start,copy->end,copy->bytes});
        }else if(record->kind==CUPTI_ACTIVITY_KIND_DRIVER || record->kind==CUPTI_ACTIVITY_KIND_RUNTIME){
            auto* api=reinterpret_cast<CUpti_ActivityAPI*>(record);
            calls[api->correlationId]=api->start;
        }
    }
    if(status!=CUPTI_ERROR_MAX_LIMIT_REACHED)check(status);
    size_t dropped=0;
    check(cuptiActivityGetNumDroppedRecords(context,stream,&dropped));
    if(dropped){std::fprintf(stderr,"CUPTI dropped %zu records\n",dropped);std::exit(3);}
    std::free(buffer);
}
uint64_t now(){uint64_t stamp;check(cuptiGetTimestamp(&stamp));return stamp;}
}
// A separate process per mode. Warm up exactly the same row shapes before capture.
// Report only kernels inside synchronous append calls, and start at embedding so
// that KV setup and pageable copies between calls cannot be counted as launch gaps.
int main(int argc,char** argv){try{
    if(argc!=3 && !(argc==4 && std::string(argv[3])=="--control"))
        throw std::invalid_argument("usage: launch_trace MODEL logits|greedy [--control]");
    const bool trace=argc==3;
    std::string mode=argv[2];
    if(mode!="logits" && mode!="greedy")throw std::invalid_argument("unknown output mode");
    auto runner=make_cuda_runner(ModelWeights::load(argv[1]),128,{16,128,0});
    std::vector<Range> ranges;
    if(trace)check(cuptiActivityRegisterCallbacks(allocate,collect));
    for(int repeat=-1;repeat<5;++repeat){
        if(trace && repeat==0){
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
            check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        }
        for(int row=0;row<4;++row)runner->create(std::to_string(row));
        for(int step=0;step<8;++step){
            std::vector<BatchInput> chunks;
            for(int row=0;row<4;++row)chunks.push_back({std::to_string(row),std::vector<TokenId>(step?1:8,10+row),step!=0});
            const auto start=now();
            const bool ok=mode=="greedy"?bool(runner->append_generation_batch(chunks)):bool(runner->append_batch(chunks));
            const auto end=now();
            if(!ok)throw std::runtime_error("unexpected batch fallback");
            if(repeat>=0)ranges.push_back({step?"decode4":"prefill32",start,end});
        }
        for(int row=0;row<4;++row)runner->release(std::to_string(row));
    }
    if(!trace){
        std::cout<<"mode\tphase\tcall_ns\n";
        for(const auto& r:ranges)std::cout<<mode<<'\t'<<r.phase<<'\t'<<r.end-r.start<<'\n';
        return 0;
    }
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityFlushAll(0));
    std::sort(kernels.begin(),kernels.end(),[](const Kernel& a,const Kernel& b){return a.start<b.start;});
    std::cout<<"mode\tphase\tcall\tcall_ns\td2h_bytes\td2h_ns\tstream\tapi_start_ns\tstart_ns\tend_ns\tname\n";
    for(size_t i=0;i<ranges.size();++i){
        const auto& range=ranges[i];
        uint64_t bytes=0,copy_ns=0;
        for(const auto& copy:downloads)if(copy.start>=range.start && copy.end<=range.end){
            bytes+=copy.bytes;copy_ns+=copy.end-copy.start;
        }
        const auto expected=4*(mode=="greedy"?sizeof(TokenId):sizeof(float)*runner->config().vocabulary);
        if(bytes!=expected)throw std::runtime_error("missing or unexpected D2H transfer");
        bool found=false;
        for(const auto& k:kernels){
            if(k.start<range.start || k.end>range.end)continue;
            found|=k.name.find("embedding_kernel")!=std::string::npos;
            if(found)std::cout<<mode<<'\t'<<range.phase<<'\t'<<i<<'\t'<<range.end-range.start<<'\t'
                <<bytes<<'\t'<<copy_ns<<'\t'<<k.stream<<'\t'
                <<(calls.count(k.correlation)?calls.at(k.correlation):0)<<'\t'<<k.start<<'\t'<<k.end<<'\t'<<k.name<<'\n';
        }
        if(!found)throw std::runtime_error("missing embedding in captured forward");
    }
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
