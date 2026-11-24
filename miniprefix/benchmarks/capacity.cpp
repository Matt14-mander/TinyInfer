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
    std::size_t batch_prefix_size(const std::vector<BatchInput>& in) override{return inner->batch_prefix_size(in);}
    void set_batch_policy(BatchPolicy p) override{inner->set_batch_policy(p);}
    BatchSpaceStats batch_space_stats() const override{return inner->batch_space_stats();}
    void set_cost_timing(bool enabled) override{inner->set_cost_timing(enabled);}
    RunnerStats stats() const override{return inner->stats();}
    KvCapacityLedger capacity_ledger() const override{return inner->capacity_ledger();}
    DecodeBatchStats decode_batch_stats() const override{return inner->decode_batch_stats();}
    PrefillBatchStats prefill_batch_stats() const override{return inner->prefill_batch_stats();}
};
std::vector<TokenId> generate(ModelRunner& r,const std::vector<TokenId>& prompt,bool publish,int count=8){
    r.create("prime");std::vector<TokenId> input=prompt,result;
    for(int n=0;n<count;++n){auto token=selected(r.append_generation("prime",input));result.push_back(token);
        if(publish)r.cache_prefix("prime");input={token};}
    r.release("prime");return result;
}
struct Row {double arrival=0,submitted=0,admitted=0,first_wait=-1,wait=0;std::size_t waits=0;bool active=false,terminal=false;std::vector<double> times;};
int main(int argc,char** argv){try{
    if(argc<9||argc>14)throw std::invalid_argument("usage: capacity MODEL cpu|cuda conservative|shared off|cold|hot COUNT RATE SEED POOL_PAGES [CACHE_PAGES] [full|adaptive] [uniform|mixed] [SHARED_PREFIX_TOKENS] [timing-off|profile]");
    const std::string backend=argv[2],policy=argv[3],mode=argv[4];
    const int count=std::stoi(argv[5]),seed=std::stoi(argv[7]),pages=std::stoi(argv[8]);const double rate=std::stod(argv[6]);
    const int cache_pages=argc>=10?std::stoi(argv[9]):pages;
    const std::string batch_policy=argc>=11?argv[10]:"full",workload=argc>=12?argv[11]:"uniform";
    const int shared_prefix=argc>=13?std::stoi(argv[12]):(workload=="mixed"?16:64);
    const std::string timing=argc>=14?argv[13]:"timing-off";
    require(timing=="timing-off"||timing=="profile","invalid timing mode");
    require((batch_policy=="full"||batch_policy=="adaptive")&&(workload=="uniform"||workload=="mixed"),"invalid batch/workload policy");
    require(shared_prefix>=0&&shared_prefix<=(workload=="mixed"?17:65),"invalid shared prefix length");
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
    const std::vector<int> lengths=workload=="mixed"?std::vector<int>{17,33,65,89}:std::vector<int>{65,68,71,74};
    const std::vector<int> outputs=workload=="mixed"?std::vector<int>{4,8,16,24}:std::vector<int>{8,8,8,8};
    for(int k=0;k<4;++k){std::vector<TokenId> p;
        for(int j=0;j<lengths[k];++j)p.push_back(1+(j<shared_prefix?j*13:j*17+k*31)%(weights->config.vocabulary-1));prompts.push_back(p);}
    {auto reference=make(64,0);for(int k=0;k<4;++k)oracle.push_back(generate(*reference,prompts[k],false,outputs[k]));}
    auto probe=std::make_unique<Probe>(make(pages,mode=="off"?0:cache_pages),policy=="shared");auto* trace=probe.get();
    trace->set_batch_policy(batch_policy=="adaptive"?BatchPolicy::Adaptive:BatchPolicy::Full);
    Engine engine(std::move(probe),{4,32,16,128});
    for(int i=0;i<8;++i){Request r;r.id="warm"+std::to_string(i);r.prompt_token_ids=prompts[i%4];r.generation.max_new_tokens=outputs[i%4];engine.submit(r);}
    while(!engine.idle()){engine.step();require(!engine.failed(),"warmup failed");}
    for(const auto& e:engine.poll_events())require(e.finish_reason!=FinishReason::Error,"warmup request failed");
    trace->clear_prefix_cache();
    if(mode=="hot")for(int k=0;k<4;++k)generate(*trace,prompts[k],true,outputs[k]);
    trace->set_cost_timing(timing=="profile");const auto space_before=trace->batch_space_stats();
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
            Request r;r.id=std::to_string(next);r.prompt_token_ids=prompts[next%4];r.generation.max_new_tokens=outputs[next%4];
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
            if(e.token_id){require(row.times.size()<oracle[index%4].size()&&*e.token_id==oracle[index%4][row.times.size()],"token mismatch");row.times.push_back(elapsed(start));}
            if(e.finish_reason){require(*e.finish_reason==FinishReason::Length&&row.times.size()==oracle[index%4].size(),"wrong terminal");row.terminal=true;}
        }
    }
    account();const double duration=elapsed(start);trace->admission={};trace->released={};
    const auto after=trace->stats();const auto db=trace->decode_batch_stats();const auto pb=trace->prefill_batch_stats();
    require(active==0&&trace->capacity_ledger().committed_blocks==0,"reservation leak");
    engine.shutdown();require(trace->stats().kv_live_blocks==0&&trace->stats().block_table_device_bytes==0,"physical leak");
    std::cout<<std::setprecision(12);
    for(int i=0;i<count;++i){const auto& r=rows[i];require(r.terminal,"missing terminal");
        std::cout<<"{\"request\":"<<i<<",\"output_tokens\":"<<outputs[i%4]<<",\"arrival_ms\":"<<r.arrival<<",\"submit_ms\":"<<r.submitted
                 <<",\"admitted_ms\":"<<r.admitted<<",\"capacity_wait_ms\":"<<r.wait<<",\"wait_attempts\":"<<r.waits<<",\"token_times_ms\":[";
        for(std::size_t j=0;j<r.times.size();++j){if(j)std::cout<<',';std::cout<<r.times[j];}std::cout<<"]}\n";
    }
    const auto space=trace->batch_space_stats();
    std::cout<<"{\"passed\":true,\"backend\":\""<<backend<<"\",\"policy\":\""<<policy<<"\",\"mode\":\""<<mode
             <<"\",\"batch_policy\":\""<<batch_policy<<"\",\"workload\":\""<<workload
             <<"\",\"timing\":\""<<timing<<"\",\"shared_prefix_tokens\":"<<shared_prefix<<",\"count\":"<<count<<",\"rate\":"<<rate<<",\"seed\":"<<seed<<",\"pool_pages\":"<<pages
             <<",\"cache_pages\":"<<(mode=="off"?0:cache_pages)<<",\"duration_ms\":"<<duration<<",\"step_ms\":"<<step_ms<<",\"steps\":"<<steps
             <<",\"mean_active\":"<<active_area/duration<<",\"peak_active\":"<<peak_active<<",\"peak_resident\":"<<trace->peak_resident
             <<",\"peak_committed\":"<<peak_committed<<",\"peak_logical_reserved\":"<<peak_reserved<<",\"peak_saved_pages\":"<<peak_saved
             <<",\"physical_peak\":"<<after.kv_peak_blocks<<",\"reused_tokens\":"<<after.prefix_reused_tokens-before.prefix_reused_tokens
             <<",\"prefix_evictions\":"<<after.prefix_evictions-before.prefix_evictions
             <<",\"forward_tokens\":"<<after.forward_tokens-before.forward_tokens<<",\"decode_batches\":"<<db.batches-decode_before.batches
             <<",\"decode_fallbacks\":"<<db.fallbacks-decode_before.fallbacks<<",\"prefill_fallbacks\":"<<pb.fallbacks-packed_before.fallbacks
             <<",\"prefill_batches\":"<<pb.prefill_batches-packed_before.prefill_batches<<",\"mixed_batches\":"<<pb.mixed_batches-packed_before.mixed_batches
             <<",\"space_prepare_calls\":"<<space.prepare_calls-space_before.prepare_calls
             <<",\"space_prepared_batches\":"<<space.prepared_batches-space_before.prepared_batches
             <<",\"space_rejected_batches\":"<<space.rejected_batches-space_before.rejected_batches
             <<",\"space_prepare_ns\":"<<space.prepare_ns-space_before.prepare_ns
             <<",\"space_growth_pages\":"<<space.growth_pages-space_before.growth_pages
             <<",\"space_cow_pages\":"<<space.cow_pages-space_before.cow_pages
             <<",\"space_cow_work_pages\":"<<space.cow_work_pages-space_before.cow_work_pages
             <<",\"space_cancelled_cow_pages\":"<<space.cancelled_cow_pages-space_before.cancelled_cow_pages
             <<",\"space_evicted_references\":"<<space.evicted_references-space_before.evicted_references
             <<",\"space_freed_pages\":"<<space.freed_pages-space_before.freed_pages
             <<",\"space_selection_calls\":"<<space.selection_calls-space_before.selection_calls
             <<",\"space_split_calls\":"<<space.split_calls-space_before.split_calls
             <<",\"space_rows_deferred\":"<<space.rows_deferred-space_before.rows_deferred
             <<",\"space_selection_ns\":"<<space.selection_ns-space_before.selection_ns
             <<",\"space_immediate_victims_avoided\":"<<space.immediate_victims_avoided-space_before.immediate_victims_avoided
             <<",\"final_blocks\":0,\"final_committed\":0}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
