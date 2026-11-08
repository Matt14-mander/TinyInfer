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
    bool batch_enabled=false;int batches=0;
    bool packed=false;int prefills=0,mixed=0;
    bool supports_packed_prefill() const override{return packed;}
    std::optional<std::vector<std::vector<float>>> append_batch(const std::vector<BatchInput>& chunks) override{
        if(!packed)return ModelRunner::append_batch(chunks);
        auto tx=kv.prepare_batch(chunks);
        if(tx.empty())return std::nullopt;
        std::vector<std::vector<float>> result(chunks.size(),std::vector<float>(128,0));
        bool has_decode=false,has_prefill=false;
        for(std::size_t i=0;i<chunks.size();++i){
            tx[i]->copy_shared_tail([](std::size_t,std::size_t,std::size_t){});
            tx[i]->commit();
            result[i][(chunks[i].tokens.back()+1)%128]=1;
            has_decode|=chunks[i].decode;
            has_prefill|=!chunks[i].decode;
        }
        if(has_prefill && has_decode)++mixed;
        else if(has_prefill)++prefills;
        return result;
    }
    Runner(int cache=16):kv({4,32,cache},1,1,128,128){config_.context=128;config_.vocabulary=128;}
    const ModelConfig& config() const override{return config_;}
    int context_limit() const override{return 128;}
    KvAdmission reserve_request(const RequestId& id,std::size_t n) override{return kv.reserve_request(id,n);}
    void release_reservation(const RequestId& id) noexcept override{kv.release_reservation(id);}
    void create(const RequestId& id) override{kv.create(id);}
    bool fail_restore=false;
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& p) override{
        if(fail_restore && id=="bad")throw KvCapacityError("restore failed");
        return kv.reuse_prefix(id,p);
    }
    void cache_prefix(const RequestId& id) override{
        if(fail_publish)throw std::runtime_error("publish failed");
        published.push_back(kv.sequence(id).tokens);kv.cache_prefix(id);
    }
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& t) override{
        auto transaction=kv.prepare_append(id,t);transaction.copy_shared_tail([](std::size_t,std::size_t,std::size_t){});transaction.commit();
        std::vector<float> logits(128,0);logits[(t.back()+1)%128]=1;
        if(invalid_logits)logits[0]=std::numeric_limits<float>::quiet_NaN();return logits;
    }
    void release(const RequestId& id) override{kv.release(id);}
    std::optional<std::vector<std::vector<float>>> append_decode_batch(const std::vector<DecodeInput>& inputs) override{
        if(!batch_enabled)return std::nullopt;
        auto group=kv.prepare_decode_batch(inputs);if(group.empty())return std::nullopt;
        std::vector<std::vector<float>> result(inputs.size(),std::vector<float>(128,0));
        for(std::size_t i=0;i<inputs.size();++i){
            group[i]->copy_shared_tail([](std::size_t,std::size_t,std::size_t){});group[i]->commit();
            result[i][(inputs[i].token+1)%128]=1;
        }
        ++batches;return result;
    }
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
        // Partial-page reuse leaves only the final prompt token for logits,
        // even when the original prompt ends exactly at a page boundary.
        const auto reused=cache?7u:0u;
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
    assert(trace->stats().kv_cached_blocks==2); // sampled final token 5 has no KV
    engine.submit(request("turn2",{1,2,3,4,5,9},3));engine.step();
    assert(engine.last_step_stats().prefix_reused_tokens==4&&engine.last_step_stats().prefill_tokens==2);
    auto events=engine.poll_events();assert(events.size()==1&&events[0].token_id==10);
    engine.step();engine.poll_events();engine.step();events=engine.poll_events();
    assert(events.size()==1&&events[0].token_id==12&&events[0].finish_reason==FinishReason::Length);
    assert((trace->published.back()==std::vector<TokenId>{1,2,3,4,5,9,10,11}));
    // A complete aligned prompt only recomputes its last token for logits.
    engine.submit(request("aligned",{1,2,3,4,5,9,10,11},1));engine.step();
    assert(engine.last_step_stats().prefix_reused_tokens==7);
    events=engine.poll_events();assert(events.size()==1&&events[0].finish_reason==FinishReason::Length);
    engine.shutdown();assert(trace->stats().kv_live_blocks==0);
}
void cancellation_eos_and_failure(){
    for(bool eos:{false,true}){
        auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine e(std::move(runner),{1,8,8,128});
        auto r=request("a",{1,2,3},8);if(eos)r.generation.eos_token_id=5;e.submit(r);
        e.step();e.poll_events();e.step();auto events=e.poll_events();assert(events[0].token_id==5);
        if(!eos){e.cancel("a");e.poll_events();}
        assert(e.idle()&&trace->stats().kv_cached_blocks==2&&trace->published.back().back()==4);
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
void batch_schedule_equivalence(){
    for(bool eos:{false,true})for(int budget:{1,2,6}){
        std::vector<std::string> reference;
        std::vector<std::vector<TokenId>> published;
        for(bool batch:{false,true}){
            auto runner=std::make_unique<Runner>();auto* trace=runner.get();trace->batch_enabled=batch;
            Engine e(std::move(runner),{3,budget,3,128});std::vector<std::string> observed;
            auto collect=[&]{for(const auto& event:e.poll_events())observed.push_back(event.request_id+":"+
                std::to_string(event.token_id.value_or(-1))+":"+std::to_string(event.finish_reason?int(*event.finish_reason):-1));};
            e.submit(request("a",{1,2,3},4));auto b=request("b",{1,2},3);
            if(eos)b.generation.eos_token_id=4;e.submit(b);
            for(int step=0;!e.idle()&&step<64;++step){
                if(step==2)e.submit(request("follower",{1,2,3,4,5,9},4));
                e.step();collect();const auto stats=e.last_step_stats();
                observed.push_back("step:"+std::to_string(stats.decode_tokens)+":"+std::to_string(stats.prefill_tokens)+":"+std::to_string(stats.prefix_reused_tokens));
                if(step==2){e.cancel("follower");collect();}
            }
            assert(e.idle()&&!e.failed());
            if(!batch){reference=observed;published=trace->published;}
            else{assert(observed==reference&&trace->published==published);assert((trace->batches>0)==(budget>1));}
            e.shutdown();assert(trace->stats().kv_live_blocks==0);
        }
    }
}
void packed_schedule_equivalence(){
    for(int cache:{0,16}){
        std::vector<std::string> reference;
        std::vector<std::vector<TokenId>> publications;
        for(bool packed:{false,true}){
            auto runner=std::make_unique<Runner>(cache);
            auto* trace=runner.get();trace->packed=packed;
            Engine engine(std::move(runner),{4,8,2,128});
            engine.submit(request("a",{1,2,3},4));
            engine.submit(request("b",{10,11,12,13,14,15,16,17,18},3));
            engine.submit(request("c",{20,21,22,23,24,25,26,27,28,29},3));
            std::vector<std::string> observed;
            for(int step=0;!engine.idle() && step<32;++step){
                if(step==2)engine.submit(request("follow",{1,2,3,4,9},2));
                engine.step();
                const auto stats=engine.last_step_stats();
                if(step==2)assert(stats.prefix_reused_tokens==(cache?4:0));
                observed.push_back(std::to_string(stats.prefill_tokens)+"/"+std::to_string(stats.decode_tokens)+"/"+std::to_string(stats.prefix_reused_tokens));
                for(const auto& event:engine.poll_events()){
                    observed.push_back(event.request_id+"/"+std::to_string(event.token_id.value_or(-1))+"/"+std::to_string(event.finish_reason?int(*event.finish_reason):-1));
                }
            }
            assert(engine.idle() && !engine.failed());
            if(packed){
                assert(trace->prefills>0 && trace->mixed>0);
                assert(observed==reference && trace->published==publications);
            }else{reference=observed;publications=trace->published;}
            engine.shutdown();assert(trace->stats().kv_live_blocks==0);
        }
    }
}
void restore_failure_preserves_order(){
    for(bool packed:{false,true}){
        auto runner=std::make_unique<Runner>(0);
        auto* trace=runner.get();trace->packed=packed;trace->fail_restore=true;
        Engine engine(std::move(runner),{4,8,2,128});
        const std::vector<RequestId> ids={"a","b","bad","d"};
        for(const auto& id:ids)engine.submit(request(id,{1},1));
        engine.step();
        const auto events=engine.poll_events();
        assert(engine.idle() && !engine.failed() && events.size()==ids.size());
        for(std::size_t i=0;i<ids.size();++i){
            assert(events[i].request_id==ids[i]);
            assert(events[i].finish_reason==(ids[i]=="bad"?FinishReason::Error:FinishReason::Length));
        }
        assert(trace->stats().live_sequences==0 && trace->stats().kv_live_blocks==0);
    }
}
int main(){aligned_single_token_completion();history_and_boundaries();cancellation_eos_and_failure();concurrent_fork_and_off();batch_schedule_equivalence();packed_schedule_equivalence();restore_failure_preserves_order();std::cout<<"generation history publication and page boundaries passed\n";}
