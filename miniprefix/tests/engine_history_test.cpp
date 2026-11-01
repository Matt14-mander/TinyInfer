#include "miniprefix/engine.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
using namespace miniprefix;
namespace {
class Runner final:public ModelRunner {
    ModelConfig config_;
public:
    KVManager kv;
    std::vector<std::vector<TokenId>> published;
    bool fail_publish=false,invalid_logits=false;
    Runner(int cache=16):kv({4,32,cache},1,1,128,128){config_.context=128;config_.vocabulary=128;}
    const ModelConfig& config() const override{return config_;}
    int context_limit() const override{return 128;}
    KvAdmission reserve_request(const RequestId& id,std::size_t n) override{return kv.reserve_request(id,n);}
    void release_reservation(const RequestId& id) noexcept override{kv.release_reservation(id);}
    void create(const RequestId& id) override{kv.create(id);}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& p) override{return kv.reuse_prefix(id,p);}
    void cache_prefix(const RequestId& id) override{
        if(fail_publish)throw std::runtime_error("publish failed");
        published.push_back(kv.sequence(id).tokens);kv.cache_prefix(id);
    }
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& t) override{
        auto transaction=kv.prepare_append(id,t);transaction.commit();
        std::vector<float> logits(128,0);logits[(t.back()+1)%128]=1;
        if(invalid_logits)logits[0]=std::numeric_limits<float>::quiet_NaN();return logits;
    }
    void release(const RequestId& id) override{kv.release(id);}
    void clear() override{kv.clear();}
    RunnerStats stats() const override{RunnerStats s;s.update_kv_stats(kv.stats());return s;}
};
Request request(const std::string& id,std::vector<TokenId> prompt,int count){
    Request r;r.id=id;r.prompt_token_ids=std::move(prompt);r.generation.max_new_tokens=count;return r;
}
std::vector<TokenId> finish(Engine& e){
    std::vector<TokenId> output;int terminal=0;
    const auto collect=[&]{for(auto& event:e.poll_events()){
        assert(event.finish_reason!=FinishReason::Error);if(event.token_id)output.push_back(*event.token_id);
        if(event.finish_reason)++terminal;}};
    collect(); // A caller's step may already have completed the request.
    for(int i=0;!e.idle()&&i<128;++i){e.step();collect();}
    assert(e.idle()&&terminal==1);return output;
}
void aligned_single_token_completion(){
    const std::vector<TokenId> prompt{1,2,3,4,5,6,7,8};
    for(int cache:{0,16})for(int chunk:{2,4,8}){
        auto runner=std::make_unique<Runner>(cache);auto* trace=runner.get();
        Engine e(std::move(runner),{1,8,chunk,128});
        e.submit(request("seed",prompt,1));assert((finish(e)==std::vector<TokenId>{9}));
        const auto before=trace->stats();
        e.submit(request("aligned",prompt,1));e.step();
        const auto reused=cache?4u:0u;
        assert(e.last_step_stats().prefix_reused_tokens==reused);
        assert(e.last_step_stats().prefill_tokens==std::min<std::size_t>(chunk,prompt.size()-reused));
        assert(e.last_step_stats().decode_tokens==0);
        // Once the uncached suffix fits, completion and its token are queued
        // by this first step even though the engine is already idle.
        assert(e.idle()==(std::size_t(chunk)>=prompt.size()-reused));
        assert((finish(e)==std::vector<TokenId>{9}));
        assert(e.poll_events().empty()); // The final event was delivered once.
        const auto after=trace->stats();
        assert(after.forward_tokens-before.forward_tokens==prompt.size()-reused);
        assert(after.prefix_reused_tokens-before.prefix_reused_tokens==reused);
        assert(trace->published.back()==prompt); // Sampled token 9 has no KV.
        assert(after.live_sequences==0&&after.kv_cached_blocks==(cache?2u:0u));
        e.shutdown();assert(trace->stats().kv_live_blocks==0);
    }
}
void history_and_boundaries(){
    auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine engine(std::move(runner),{2,8,2,128});
    engine.submit(request("turn1",{1,2,3},2));assert((finish(engine)==std::vector<TokenId>{4,5}));
    assert((trace->published==std::vector<std::vector<TokenId>>{{1,2,3},{1,2,3,4}}));
    assert(trace->stats().kv_cached_blocks==1); // sampled final token 5 has no KV
    engine.submit(request("turn2",{1,2,3,4,5,9},3));engine.step();
    assert(engine.last_step_stats().prefix_reused_tokens==4&&engine.last_step_stats().prefill_tokens==2);
    auto events=engine.poll_events();assert(events.size()==1&&events[0].token_id==10);
    engine.step();engine.poll_events();engine.step();events=engine.poll_events();
    assert(events.size()==1&&events[0].token_id==12&&events[0].finish_reason==FinishReason::Length);
    assert((trace->published.back()==std::vector<TokenId>{1,2,3,4,5,9,10,11}));
    // A complete aligned prompt still recomputes its last page for logits.
    engine.submit(request("aligned",{1,2,3,4,5,9,10,11},1));engine.step();
    assert(engine.last_step_stats().prefix_reused_tokens==4);finish(engine);
    engine.shutdown();assert(trace->stats().kv_live_blocks==0);
}
void cancellation_eos_and_failure(){
    for(bool eos:{false,true}){
        auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine e(std::move(runner),{1,8,8,128});
        auto r=request("a",{1,2,3},8);if(eos)r.generation.eos_token_id=5;e.submit(r);
        e.step();e.poll_events();e.step();auto events=e.poll_events();assert(events[0].token_id==5);
        if(!eos){e.cancel("a");e.poll_events();}
        assert(e.idle()&&trace->stats().kv_cached_blocks==1&&trace->published.back().back()==4);
        e.submit(request("next",{1,2,3,4,5,6},1));e.step();assert(e.last_step_stats().prefix_reused_tokens==4);e.poll_events();
    }
    for(bool nan:{false,true}){
        auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine e(std::move(runner),{1,8,8,128});
        e.submit(request("a",{1,2,3},3));e.step();e.poll_events();
        trace->fail_publish=!nan;trace->invalid_logits=nan;e.step();auto events=e.poll_events();
        assert(events.size()==1&&events[0].finish_reason==FinishReason::Error&&!events[0].token_id);
        assert(e.failed()&&trace->published.size()==1&&trace->stats().kv_live_blocks==0);
    }
}
void concurrent_fork_and_off(){
    for(int cache:{0,16}){
        auto runner=std::make_unique<Runner>(cache);auto* trace=runner.get();Engine e(std::move(runner),{2,8,8,128});
        e.submit(request("source",{1,2,3},5));e.step();e.poll_events();
        e.submit(request("fork",{1,2,3,4,50},1));e.step();auto events=e.poll_events();
        // Decoder fills/publishes page before follower restores within the same step.
        assert(e.last_step_stats().prefix_reused_tokens==(cache?4:0));
        assert(events.size()==2&&events[0].token_id==5&&events[1].token_id==51);
        assert(trace->kv.sequence("source").tokens==std::vector<TokenId>({1,2,3,4}));
        finish(e);e.shutdown();assert(trace->stats().kv_live_blocks==0);
    }
}
}
int main(){aligned_single_token_completion();history_and_boundaries();cancellation_eos_and_failure();concurrent_fork_and_off();std::cout<<"generation history publication and page boundaries passed\n";}
