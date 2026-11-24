// Native, open-loop admission comparison. Same model/backend/batching for both policies.
#include "miniprefix/engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
using namespace miniprefix;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
TokenId selected(const GenerationResult& r){
    if(const auto* token=std::get_if<TokenId>(&r))return *token;
    const auto& logits=std::get<std::vector<float>>(r);
    require(!logits.empty(),"empty logits");return TokenId(std::max_element(logits.begin(),logits.end())-logits.begin());
}
struct Probe final:ModelRunner {
    std::unique_ptr<ModelRunner> inner;bool shared;
    std::function<void(const RequestId&,KvAdmission)> admission;
    std::function<void(const RequestId&)> released;
    std::size_t resident=0,peak_resident=0;
    Probe(std::unique_ptr<ModelRunner> r,bool s):inner(std::move(r)),shared(s){}
    const ModelConfig& config() const override{return inner->config();}
    int context_limit() const override{return inner->context_limit();}
    KvAdmission reserve_request(const RequestId& id,std::size_t n,const std::vector<TokenId>& p) override{
        auto result=shared?inner->reserve_request(id,n,p):inner->reserve_request(id,n);
        if(admission)admission(id,result);return result;
    }
    void release_reservation(const RequestId& id) noexcept override{
        inner->release_reservation(id);if(released)released(id);
    }
    void create(const RequestId& id) override{inner->create(id);peak_resident=std::max(peak_resident,++resident);}
    void release(const RequestId& id) override{inner->release(id);--resident;}
    void clear() override{inner->clear();resident=0;}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& p) override{return inner->reuse_prefix(id,p);}
    void cache_prefix(const RequestId& id) override{inner->cache_prefix(id);}
    void clear_prefix_cache() override{inner->clear_prefix_cache();}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& t) override{return inner->append(id,t);}
    GenerationResult append_generation(const RequestId& id,const std::vector<TokenId>& t) override{return inner->append_generation(id,t);}
    bool supports_packed_prefill() const override{return inner->supports_packed_prefill();}
    std::optional<std::vector<GenerationResult>> append_generation_batch(const std::vector<BatchInput>& in) override{
        return inner->append_generation_batch(in);
    }
    RunnerStats stats() const override{return inner->stats();}
    KvCapacityLedger capacity_ledger() const override{return inner->capacity_ledger();}
    DecodeBatchStats decode_batch_stats() const override{return inner->decode_batch_stats();}
    PrefillBatchStats prefill_batch_stats() const override{return inner->prefill_batch_stats();}
};
std::vector<TokenId> generate(ModelRunner& r,const std::vector<TokenId>& prompt,bool publish){
    r.create("prime");std::vector<TokenId> input=prompt,result;
    for(int n=0;n<8;++n){auto token=selected(r.append_generation("prime",input));result.push_back(token);
        if(publish)r.cache_prefix("prime");input={token};}
    r.release("prime");return result;
}
struct Row {double arrival=0,submitted=0,admitted=0,first_wait=-1,wait=0;std::size_t waits=0;bool active=false,terminal=false;std::vector<double> times;};
int main(int argc,char** argv){try{
    if(argc!=9&&argc!=10)throw std::invalid_argument("usage: capacity MODEL cpu|cuda conservative|shared off|cold|hot COUNT RATE SEED POOL_PAGES [CACHE_PAGES]");
    const std::string backend=argv[2],policy=argv[3],mode=argv[4];
    const int count=std::stoi(argv[5]),seed=std::stoi(argv[7]),pages=std::stoi(argv[8]);const double rate=std::stod(argv[6]);
    const int cache_pages=argc==10?std::stoi(argv[9]):pages;
    require(cache_pages>=0&&cache_pages<=pages,"invalid cache budget");
    require((policy=="conservative"||policy=="shared")&&(mode=="off"||mode=="cold"||mode=="hot"),"invalid mode");
    require(count>=8&&std::isfinite(rate)&&rate>0&&pages>=10,"invalid workload bounds");
    auto weights=ModelWeights::load(argv[1]);
    auto make=[&](int blocks,int cache)->std::unique_ptr<ModelRunner>{
        if(backend=="cpu")return make_cpu_runner(weights,128,{8,blocks,cache});
#ifdef MINIPREFIX_ENABLE_CUDA
        if(backend=="cuda")return make_cuda_runner(weights,128,{8,blocks,cache});
#endif
        throw std::invalid_argument("backend unavailable");
    };
    std::vector<std::vector<TokenId>> prompts,oracle;
    for(int k=0;k<4;++k){std::vector<TokenId> p;
        for(int j=0;j<65+3*k;++j)p.push_back(1+(j<64?j*13:j*17+k*31)%(weights->config.vocabulary-1));prompts.push_back(p);}
    {auto reference=make(64,0);for(const auto& p:prompts)oracle.push_back(generate(*reference,p,false));}
    auto probe=std::make_unique<Probe>(make(pages,mode=="off"?0:cache_pages),policy=="shared");auto* trace=probe.get();
    Engine engine(std::move(probe),{4,32,16,128});
    for(int i=0;i<8;++i){Request r;r.id="warm"+std::to_string(i);r.prompt_token_ids=prompts[i%4];r.generation.max_new_tokens=8;engine.submit(r);}
    while(!engine.idle()){engine.step();require(!engine.failed(),"warmup failed");}
    for(const auto& e:engine.poll_events())require(e.finish_reason!=FinishReason::Error,"warmup request failed");
    trace->clear_prefix_cache();
    if(mode=="hot")for(const auto& p:prompts)generate(*trace,p,true);
    std::vector<Row> rows(count);std::mt19937 rng(seed);
    // Identical arrival slots and bounded integer jitter across both policies.
    for(int i=0;i<count;++i)rows[i].arrival=(i+(rng()%800)/1000.0)*1000/rate;
    const auto before=trace->stats();const auto decode_before=trace->decode_batch_stats();const auto packed_before=trace->prefill_batch_stats();
    std::size_t active=0,peak_active=0,peak_committed=0,peak_reserved=0,peak_saved=0,steps=0;
    double active_area=0,active_last=0,step_ms=0;trace->peak_resident=0;
    auto start=Clock::now();
    struct ClearCallbacks {Probe* probe;~ClearCallbacks(){probe->admission={};probe->released={};}} guard{trace};
    auto account=[&]{const auto now=elapsed(start);active_area+=active*(now-active_last);active_last=now;};
    trace->admission=[&](const RequestId& id,KvAdmission result){
        auto& r=rows.at(std::stoul(id));const auto now=elapsed(start);
        if(result==KvAdmission::Wait){if(r.first_wait<0)r.first_wait=now;++r.waits;}
        if(result==KvAdmission::Ready){account();r.admitted=now;r.active=true;++active;peak_active=std::max(peak_active,active);
            if(r.first_wait>=0)r.wait=now-r.first_wait;
            const auto ledger=trace->capacity_ledger();require(ledger.committed_blocks<=std::size_t(pages),"overcommitted");
            peak_committed=std::max(peak_committed,ledger.committed_blocks);peak_reserved=std::max(peak_reserved,ledger.logical_reserved_blocks);
            peak_saved=std::max(peak_saved,ledger.logical_reserved_blocks-ledger.committed_blocks);}
    };
    trace->released=[&](const RequestId& id){auto& r=rows[std::stoul(id)];if(r.active){account();--active;r.active=false;}};
    int next=0;
    while(next<count||!engine.idle()){
        require(elapsed(start)<600000,"run exceeded ten-minute progress bound");
        while(next<count&&rows[next].arrival<=elapsed(start)){
            Request r;r.id=std::to_string(next);r.prompt_token_ids=prompts[next%4];r.generation.max_new_tokens=8;
            rows[next].submitted=elapsed(start);engine.submit(std::move(r));++next;
        }
        if(engine.idle()){
            if(next<count)std::this_thread::sleep_until(start+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(rows[next].arrival)));
            continue;
        }
        auto tick=Clock::now();engine.step();step_ms+=elapsed(tick);++steps;
        require(!engine.failed(),"Engine failed");
        for(const auto& e:engine.poll_events()){
            const auto index=std::stoul(e.request_id);auto& row=rows.at(index);require(!row.terminal,"event after terminal");
            if(e.token_id){require(row.times.size()<8&&*e.token_id==oracle[index%4][row.times.size()],"token mismatch");row.times.push_back(elapsed(start));}
            if(e.finish_reason){require(*e.finish_reason==FinishReason::Length&&row.times.size()==8,"wrong terminal");row.terminal=true;}
        }
    }
    account();const double duration=elapsed(start);trace->admission={};trace->released={};
    const auto after=trace->stats();const auto db=trace->decode_batch_stats();const auto pb=trace->prefill_batch_stats();
    require(active==0&&trace->capacity_ledger().committed_blocks==0,"reservation leak");
    engine.shutdown();require(trace->stats().kv_live_blocks==0&&trace->stats().block_table_device_bytes==0,"physical leak");
    std::cout<<std::setprecision(12);
    for(int i=0;i<count;++i){const auto& r=rows[i];require(r.terminal,"missing terminal");
        std::cout<<"{\"request\":"<<i<<",\"arrival_ms\":"<<r.arrival<<",\"submit_ms\":"<<r.submitted
                 <<",\"admitted_ms\":"<<r.admitted<<",\"capacity_wait_ms\":"<<r.wait<<",\"wait_attempts\":"<<r.waits<<",\"token_times_ms\":[";
        for(std::size_t j=0;j<r.times.size();++j){if(j)std::cout<<',';std::cout<<r.times[j];}std::cout<<"]}\n";
    }
    std::cout<<"{\"passed\":true,\"backend\":\""<<backend<<"\",\"policy\":\""<<policy<<"\",\"mode\":\""<<mode
             <<"\",\"count\":"<<count<<",\"rate\":"<<rate<<",\"seed\":"<<seed<<",\"pool_pages\":"<<pages
             <<",\"cache_pages\":"<<(mode=="off"?0:cache_pages)<<",\"duration_ms\":"<<duration<<",\"step_ms\":"<<step_ms<<",\"steps\":"<<steps
             <<",\"mean_active\":"<<active_area/duration<<",\"peak_active\":"<<peak_active<<",\"peak_resident\":"<<trace->peak_resident
             <<",\"peak_committed\":"<<peak_committed<<",\"peak_logical_reserved\":"<<peak_reserved<<",\"peak_saved_pages\":"<<peak_saved
             <<",\"physical_peak\":"<<after.kv_peak_blocks<<",\"reused_tokens\":"<<after.prefix_reused_tokens-before.prefix_reused_tokens
             <<",\"forward_tokens\":"<<after.forward_tokens-before.forward_tokens<<",\"decode_batches\":"<<db.batches-decode_before.batches
             <<",\"decode_fallbacks\":"<<db.fallbacks-decode_before.fallbacks<<",\"prefill_fallbacks\":"<<pb.fallbacks-packed_before.fallbacks
             <<",\"prefill_batches\":"<<pb.prefill_batches-packed_before.prefill_batches<<",\"mixed_batches\":"<<pb.mixed_batches-packed_before.mixed_batches
             <<",\"final_blocks\":0,\"final_committed\":0}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
