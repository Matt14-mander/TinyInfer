#include "miniprefix/engine.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <cstddef>
#include <type_traits>

using namespace miniprefix;
namespace {
static_assert(std::is_standard_layout<RunnerStats>::value,"RunnerStats keeps its field layout");
static_assert(sizeof(RunnerStats)==25*sizeof(std::size_t),"STATS remains 25 fields");
static_assert(offsetof(RunnerStats,device_weight_bytes)==4*sizeof(std::size_t),"preserve device field offset");
static_assert(offsetof(RunnerStats,kv_block_tokens)==11*sizeof(std::size_t),"preserve KV field offset");
void stats_mapping(){
    RunnerStats stats;stats.device_weight_bytes=123;stats.token_uploaded_bytes=456;
    stats.block_table_uploaded_bytes=789;stats.block_table_device_bytes=12;
    KvStats common;common.live_sequences=2;common.forward_tokens=20;common.kv_live_blocks=3;
    common.kv_reserved_bytes=100;common.prefix_hits=4;common.prefix_reused_tokens=16;
    stats.update_kv_stats(common);
    assert(stats.live_sequences==2&&stats.forward_tokens==20&&stats.kv_live_blocks==3);
    assert(stats.kv_reserved_bytes==100&&stats.prefix_hits==4&&stats.prefix_reused_tokens==16);
    assert(stats.device_weight_bytes==123&&stats.token_uploaded_bytes==456);
    assert(stats.block_table_uploaded_bytes==789&&stats.block_table_device_bytes==12);
}
struct Call {RequestId id;std::size_t position;std::vector<TokenId> input;};
class RecordingRunner final:public ModelRunner {
    ModelConfig config_;
public:
    std::map<RequestId,std::size_t> hits,positions,restores,publishes;
    std::vector<Call> calls;
    bool invalid_logits=false;
    RecordingRunner(){config_.context=128;config_.vocabulary=128;}
    const ModelConfig& config() const override{return config_;}
    void create(const RequestId& id) override{assert(!positions.count(id));positions[id]=0;}
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>&) override{
        ++restores[id];return positions[id]=hits[id];
    }
    void cache_prefix(const RequestId& id) override{++publishes[id];}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        assert(!tokens.empty());calls.push_back({id,positions.at(id),tokens});positions[id]+=tokens.size();
        std::vector<float> logits(128,0);logits[(tokens.back()+1)%128]=1;
        if(invalid_logits)logits[0]=std::numeric_limits<float>::quiet_NaN();
        return logits;
    }
    void release(const RequestId& id) override{assert(positions.erase(id)==1);}
    void clear() override{positions.clear();}
    RunnerStats stats() const override{RunnerStats s;s.kv_block_tokens=4;s.live_sequences=positions.size();return s;}
};
Request request(std::string id,int length,int output=2){
    Request r;r.id=std::move(id);for(int i=0;i<length;++i)r.prompt_token_ids.push_back(i+1);
    r.generation.max_new_tokens=output;return r;
}
void interleaved_suffix(){
    auto runner=std::make_unique<RecordingRunner>();auto* trace=runner.get();trace->hits["suffix"]=8;
    Engine engine(std::move(runner),{2,5,3,128});
    engine.submit(request("decoder",1,6));engine.step();engine.poll_events();
    auto suffix=request("suffix",19);engine.submit(suffix);
    std::vector<TokenId> received;
    for(int step=0;step<4;++step){
        engine.step();auto stats=engine.last_step_stats();
        assert(stats.prefix_reused_tokens==(step==0?8:0));
        assert(stats.prefill_tokens==(step==3?2:3)&&stats.decode_tokens==1);
        assert(stats.prefill_tokens+stats.decode_tokens<=5);
        for(const auto& e:engine.poll_events())if(e.request_id=="suffix"){
            assert(step==3&&e.token_id&&*e.token_id==20);received.push_back(*e.token_id);
        }
    }
    assert(trace->restores["suffix"]==1&&trace->publishes["suffix"]==1);
    std::size_t cursor=8;
    for(const auto& call:trace->calls)if(call.id=="suffix"){
        assert(call.position==cursor&&call.input.size()<=3);
        assert(std::vector<TokenId>(suffix.prompt_token_ids.begin()+cursor,
                                   suffix.prompt_token_ids.begin()+cursor+call.input.size())==call.input);
        cursor+=call.input.size();
    }
    assert(cursor==19);
    while(!engine.idle()){engine.step();for(const auto& e:engine.poll_events())if(e.request_id=="suffix"){
        assert(e.token_id&&*e.token_id==21&&e.finish_reason==FinishReason::Length);received.push_back(*e.token_id);
    }}
    assert((received==std::vector<TokenId>{20,21}));
    engine.step();assert(engine.last_step_stats().prefill_tokens==0&&engine.last_step_stats().decode_tokens==0);
}
void small_budget_and_cancel(){
    auto runner=std::make_unique<RecordingRunner>();auto* trace=runner.get();trace->hits["suffix"]=4;
    Engine engine(std::move(runner),{1,2,5,128});engine.submit(request("suffix",13));
    engine.step();assert(engine.poll_events().empty());
    assert(trace->calls.back().position==4&&trace->calls.back().input.size()==2);
    assert(engine.last_step_stats().prefix_reused_tokens==4&&engine.last_step_stats().prefill_tokens==2);
    engine.step();assert(engine.poll_events().empty()&&engine.last_step_stats().prefill_tokens==2);
    engine.cancel("suffix");auto events=engine.poll_events();
    assert(events.size()==1&&!events[0].token_id&&events[0].finish_reason==FinishReason::Cancelled);
    assert(trace->publishes.empty()&&trace->positions.empty()&&trace->restores["suffix"]==1);
}
void invalid_hit_and_logits(){
    for(std::size_t hit:{std::size_t(3),std::size_t(8),std::size_t(100)}){
        auto runner=std::make_unique<RecordingRunner>();auto* trace=runner.get();trace->hits["bad"]=hit;
        Engine engine(std::move(runner),{1,4,4,128});engine.submit(request("bad",8));engine.step();
        auto events=engine.poll_events();assert(events.size()==1&&events[0].finish_reason==FinishReason::Error);
        assert(trace->calls.empty()&&trace->positions.empty());
    }
    auto runner=std::make_unique<RecordingRunner>();auto* trace=runner.get();trace->invalid_logits=true;
    Engine engine(std::move(runner),{1,4,4,128});engine.submit(request("nan",3));engine.step();
    auto events=engine.poll_events();assert(events.size()==1&&events[0].finish_reason==FinishReason::Error);
    assert(trace->publishes.empty()&&trace->positions.empty());
}
}
int main(){stats_mapping();interleaved_suffix();small_budget_and_cancel();invalid_hit_and_logits();
    std::cout<<"incremental suffix prefill positions, budgets, interleaving, events and cleanup passed\n";}
