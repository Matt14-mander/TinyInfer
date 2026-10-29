// Native long-running lifecycle stress and wall-clock attribution. No worker IPC.
#include "miniprefix/engine.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <algorithm>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif
using namespace miniprefix;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Timing {double time=0;std::size_t calls=0;};
struct Scope {Timing& t;Clock::time_point start=Clock::now();~Scope(){t.time+=ms(start);++t.calls;}};
struct Profile final:ModelRunner {
    std::unique_ptr<ModelRunner> inner;
    std::map<std::string,Timing> times;
    std::map<RequestId,std::size_t> lengths,positions;
    std::map<RequestId,Clock::time_point> admitted;
    explicit Profile(std::unique_ptr<ModelRunner> r):inner(std::move(r)){}
    const ModelConfig& config() const override{return inner->config();}
    int context_limit() const override{return inner->context_limit();}
    KvAdmission reserve_request(const RequestId& id,std::size_t n) override{
        Scope t{times["admission"]};auto result=inner->reserve_request(id,n);
        if(result==KvAdmission::Ready)admitted[id]=Clock::now();return result;
    }
    void release_reservation(const RequestId& id) noexcept override{inner->release_reservation(id);}
    void create(const RequestId& id) override{Scope t{times["create"]};inner->create(id);positions[id]=0;}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& p) override{
        Scope t{times["reuse"]};return positions[id]=inner->reuse_prefix(id,p);
    }
    void cache_prefix(const RequestId& id) override{Scope t{times["publish"]};inner->cache_prefix(id);}
    void clear_prefix_cache() override{Scope t{times["clear_prefix"]};inner->clear_prefix_cache();}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        Scope t{times[positions.at(id)<lengths.at(id)?"prefill":"decode"]};
        auto result=inner->append(id,tokens);positions[id]+=tokens.size();return result;
    }
    void release(const RequestId& id) override{Scope t{times["release"]};inner->release(id);positions.erase(id);}
    void clear() override{Scope t{times["clear"]};inner->clear();positions.clear();}
    RunnerStats stats() const override{return inner->stats();}
};
std::pair<std::size_t,std::size_t> memory(){
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);
    if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m)))
        throw std::runtime_error("process memory query failed");
    return {m.PrivateUsage,m.WorkingSetSize};
#else
    return {0,0};
#endif
}
int main(int argc,char** argv){try{
    if(argc!=5)throw std::invalid_argument("usage: stress MODEL cpu|cuda SECONDS MIN_WAVES");
    const std::string backend=argv[2];const double seconds=std::stod(argv[3]);const int minimum=std::stoi(argv[4]);
    require(std::isfinite(seconds)&&seconds>0&&minimum>0,"invalid run bounds");
    auto weights=ModelWeights::load(argv[1]);
    auto make=[&](KvCacheConfig kv)->std::unique_ptr<ModelRunner>{
        if(backend=="cpu")return make_cpu_runner(weights,128,kv);
#ifdef MINIPREFIX_ENABLE_CUDA
        if(backend=="cuda")return make_cuda_runner(weights,128,kv);
#endif
        throw std::invalid_argument("backend unavailable");
    };
    std::vector<std::vector<TokenId>> prompts,oracle;
    for(int i=0;i<12;++i){
        std::vector<TokenId> p;const int length=std::vector<int>{8,17,33,65}[i%4];
        for(int j=0;j<length;++j)p.push_back(1+((i<6&&j<8)?j:(i*37+j*13))%(weights->config.vocabulary-1));
        prompts.push_back(p);
    }
    // Same-backend no-cache reference isolates cache/scheduling correctness;
    // independent Transformers correctness is a separate acceptance gate.
    {
        auto reference=make({4,128,0});
        for(const auto& p:prompts){
            reference->create("oracle");auto input=p;std::vector<TokenId> expected;
            for(int j=0;j<8;++j){auto logits=reference->append("oracle",input);
                auto token=TokenId(std::max_element(logits.begin(),logits.end())-logits.begin());
                expected.push_back(token);input={token};}
            reference->release("oracle");oracle.push_back(expected);
        }
    }
    auto profile=std::make_unique<Profile>(make({4,24,12}));auto* trace=profile.get();
    Engine engine(std::move(profile),{4,16,8,128});
    auto start=Clock::now();std::size_t total=0,total_tokens=0;int wave=0;
    std::cout<<std::setprecision(12);
    while(ms(start)<seconds*1000||wave<minimum){
        const auto wave_start=Clock::now();const auto before=trace->stats();const auto timing_before=trace->times;
        std::map<RequestId,int> index,received,terminal;
        std::map<RequestId,Clock::time_point> submitted;
        std::map<RequestId,double> first,last;
        std::map<RequestId,FinishReason> reasons;
        std::vector<RequestId> ids;
        std::size_t wave_tokens=0,forward_tokens=0,reused_tokens=0;
        for(int i=0;i<8;++i){
            auto id=std::to_string(wave*8+i);ids.push_back(id);const int k=(wave+i)%12;index[id]=k;
            Request r;r.id=id;r.prompt_token_ids=prompts[k];r.generation.max_new_tokens=8;
            if(i==6)r.generation.eos_token_id=oracle[k][0];
            if(i==7){r.prompt_token_ids.assign(97,1);r.generation.max_new_tokens=1;}
            trace->lengths[id]=r.prompt_token_ids.size();submitted[id]=Clock::now();engine.submit(std::move(r));
        }
        auto consume=[&]{for(const auto& e:engine.poll_events()){
            require(index.count(e.request_id)!=0,"unknown event");require(!terminal[e.request_id],"event after terminal");
            if(e.token_id){auto& n=received[e.request_id];require(n<8&&*e.token_id==oracle[index[e.request_id]][n],"token mismatch");
                if(n==0)first[e.request_id]=ms(submitted[e.request_id]);last[e.request_id]=ms(submitted[e.request_id]);++n;++wave_tokens;}
            if(e.finish_reason){++terminal[e.request_id];reasons[e.request_id]=*e.finish_reason;}
        }};
        // One queued cancellation and one cancellation after at least one step.
        engine.cancel(ids[5]);consume();double step_ms=0;int steps=0;
        while(!engine.idle()){
            require(++steps<=256,"no bounded progress");auto tick=Clock::now();engine.step();step_ms+=ms(tick);
            const auto step=engine.last_step_stats();forward_tokens+=step.prefill_tokens+step.decode_tokens;reused_tokens+=step.prefix_reused_tokens;consume();
            if(steps==1&&!terminal[ids[0]]){engine.cancel(ids[0]);consume();}
            require(!engine.failed(),"Engine poisoned");
        }
        for(int i=0;i<8;++i){const auto& id=ids[i];require(terminal[id]==1,"missing or duplicate terminal");
            const auto expected=i==7?FinishReason::Error:i==6?FinishReason::Eos:(i==0||i==5)?FinishReason::Cancelled:FinishReason::Length;
            require(reasons[id]==expected,"wrong finish reason");
            if(expected==FinishReason::Length)require(received[id]==8,"short output");
            if(expected==FinishReason::Eos)require(received[id]==1,"EOS not early");
        }
        const auto after=trace->stats();
        require(after.forward_tokens-before.forward_tokens==forward_tokens&&after.prefix_reused_tokens-before.prefix_reused_tokens==reused_tokens,"step/runner accounting mismatch");
        require(after.live_sequences==0&&after.block_table_device_bytes==0,"sequence/table leak");
        require(after.kv_live_blocks==after.kv_cache_only_blocks&&after.kv_cached_blocks==after.kv_cache_only_blocks,"non-cache page leak");
        require(after.kv_live_blocks<=12&&after.kv_peak_blocks<=24,"pool/cache bound");
        require(after.weight_uploaded_bytes==before.weight_uploaded_bytes&&after.kv_uploaded_bytes==before.kv_uploaded_bytes,"resident payload uploaded");
        if(wave%5==4){trace->clear_prefix_cache();require(trace->stats().kv_live_blocks==0,"clear leaked KV");}
        total+=8;total_tokens+=wave_tokens;auto mem=memory();
        std::cout<<"{\"wave\":"<<wave<<",\"elapsed_ms\":"<<ms(start)<<",\"wall_ms\":"<<ms(wave_start)
                 <<",\"step_ms\":"<<step_ms<<",\"steps\":"<<steps<<",\"tokens\":"<<wave_tokens
                 <<",\"private_bytes\":"<<mem.first<<",\"rss_bytes\":"<<mem.second
                 <<",\"cached_pages\":"<<after.kv_cached_blocks<<",\"forward_tokens\":"<<after.forward_tokens-before.forward_tokens
                 <<",\"reused_tokens\":"<<after.prefix_reused_tokens-before.prefix_reused_tokens
                 <<",\"evictions\":"<<after.prefix_evictions-before.prefix_evictions
                 <<",\"workspace_bytes\":"<<after.device_workspace_bytes<<",\"device_allocations\":"<<after.device_allocations
                 <<",\"timings\":{";
        bool comma=false;for(const auto& item:trace->times){
            auto previous=timing_before.find(item.first);Timing old;if(previous!=timing_before.end())old=previous->second;
            if(comma)std::cout<<',';comma=true;
            std::cout<<'"'<<item.first<<"\":{\"ms\":"<<item.second.time-old.time<<",\"calls\":"<<item.second.calls-old.calls<<'}';}
        std::cout<<"},\"requests\":[";
        for(int i=0;i<8;++i){const auto& id=ids[i];if(i)std::cout<<',';
            std::cout<<"{\"id\":\""<<id<<"\",\"template\":"<<index[id]<<",\"reason\":"<<int(reasons[id])<<",\"output_tokens\":"<<received[id];
            if(first.count(id))std::cout<<",\"ttft_ms\":"<<first[id]<<",\"last_ms\":"<<last[id];
            if(trace->admitted.count(id))std::cout<<",\"queue_ms\":"<<std::chrono::duration<double,std::milli>(trace->admitted[id]-submitted[id]).count();
            std::cout<<'}';trace->lengths.erase(id);trace->admitted.erase(id);}
        std::cout<<"]}"<<std::endl;++wave;
    }
    engine.shutdown();const auto final=trace->stats();require(final.live_sequences==0&&final.kv_live_blocks==0&&final.block_table_device_bytes==0,"final leak");
    std::cout<<"{\"passed\":true,\"waves\":"<<wave<<",\"requests\":"<<total<<",\"tokens\":"<<total_tokens<<",\"elapsed_ms\":"<<ms(start)<<"}"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
