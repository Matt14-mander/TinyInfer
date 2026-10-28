#include "miniprefix/engine.h"
#include <cassert>
#include <algorithm>
#include <iostream>
using namespace miniprefix;
namespace {
struct Trace {
    std::vector<std::string> operations;
    std::set<RequestId> resident;
    RequestId fail_append,fail_release;
    int clears=0;
};
class Runner final:public ModelRunner {
    ModelConfig config_;
    std::shared_ptr<Trace> trace_;
public:
    explicit Runner(std::shared_ptr<Trace> trace):trace_(std::move(trace)){config_.context=128;config_.vocabulary=128;}
    const ModelConfig& config() const override{return config_;}
    int context_limit() const override{return config_.context;}
    void create(const RequestId& id) override{assert(trace_->resident.insert(id).second);trace_->operations.push_back("create:"+id);}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        assert(trace_->resident.count(id));trace_->operations.push_back("append:"+id);
        if(id==trace_->fail_append)throw std::runtime_error("forward failed");
        std::vector<float> logits(128,0);logits.at(tokens.back()+1)=1;return logits;
    }
    void release(const RequestId& id) override{
        trace_->operations.push_back("release:"+id);
        if(id==trace_->fail_release)throw std::runtime_error("release failed");
        assert(trace_->resident.erase(id)==1);
    }
    void clear() override{++trace_->clears;trace_->operations.push_back("clear");trace_->resident.clear();}
    RunnerStats stats() const override{return {};}
};
Request request(const std::string& id,int prompt=1,int output=1){
    Request r;r.id=id;r.prompt_token_ids.assign(prompt,1);r.generation.max_new_tokens=output;return r;
}
void event(const GenerationEvent& e,const std::string& id,std::optional<TokenId> token,
           std::optional<FinishReason> reason,const std::string& error=""){
    assert(e.request_id==id&&e.token_id==token&&e.finish_reason==reason&&e.error_message==error);
}
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}assert(rejected);}
void completion(){
    auto trace=std::make_shared<Trace>();Engine engine(std::make_unique<Runner>(trace),{2,8,8,128});
    engine.submit(request("length"));auto eos=request("eos");eos.generation.eos_token_id=2;engine.submit(eos);
    assert(engine.step()&&engine.idle());auto events=engine.poll_events();assert(events.size()==2);
    event(events[0],"length",2,FinishReason::Length);event(events[1],"eos",2,FinishReason::Eos);
    assert((trace->operations==std::vector<std::string>{"create:length","append:length","release:length","create:eos","append:eos","release:eos"}));
    rejects([&]{engine.cancel("length");});rejects([&]{engine.submit(request("length"));});
    assert(!engine.step()&&engine.poll_events().empty());engine.shutdown();engine.shutdown();assert(engine.poll_events().empty());
}
void cancellation(){
    auto trace=std::make_shared<Trace>();Engine engine(std::make_unique<Runner>(trace),{3,1,1,128});
    engine.submit(request("resident",4));engine.submit(request("unstarted"));engine.submit(request("survivor"));engine.submit(request("waiting"));
    engine.step();assert(engine.poll_events().empty());
    engine.cancel("waiting");engine.cancel("unstarted");engine.cancel("resident");
    auto events=engine.poll_events();assert(events.size()==3);
    event(events[0],"waiting",std::nullopt,FinishReason::Cancelled);
    event(events[1],"unstarted",std::nullopt,FinishReason::Cancelled);
    event(events[2],"resident",std::nullopt,FinishReason::Cancelled);
    assert((trace->operations==std::vector<std::string>{"create:resident","append:resident","release:resident"}));
    rejects([&]{engine.cancel("resident");});rejects([&]{engine.cancel("unknown");});
    engine.step();events=engine.poll_events();assert(events.size()==1);event(events[0],"survivor",2,FinishReason::Length);
    assert(engine.idle()&&trace->resident.empty());
}
void failure_preserves_prior_events(bool release_failure){
    auto trace=std::make_shared<Trace>();
    if(release_failure)trace->fail_release="broken";else trace->fail_append="broken";
    Engine engine(std::make_unique<Runner>(trace),{4,8,8,128});
    engine.submit(request("finished"));engine.submit(request("streaming",1,3));engine.submit(request("broken"));
    engine.submit(request("unstarted"));engine.submit(request("waiting"));
    assert(engine.step()&&engine.idle());auto events=engine.poll_events();assert(events.size()==6);
    event(events[0],"finished",2,FinishReason::Length);event(events[1],"streaming",2,std::nullopt);
    const std::string error=release_failure?"release failed":"forward failed";
    event(events[2],"streaming",std::nullopt,FinishReason::Error,error);
    event(events[3],"broken",std::nullopt,FinishReason::Error,error);
    event(events[4],"unstarted",std::nullopt,FinishReason::Error,error);
    event(events[5],"waiting",std::nullopt,FinishReason::Error,error);
    std::vector<std::string> expected={"create:finished","append:finished","release:finished","create:streaming","append:streaming","create:broken","append:broken"};
    if(release_failure)expected.push_back("release:broken");expected.push_back("clear");
    assert(trace->operations==expected&&trace->clears==1&&trace->resident.empty());
    rejects([&]{engine.step();});rejects([&]{engine.submit(request("new"));});rejects([&]{engine.cancel("broken");});
    engine.shutdown();engine.shutdown();assert(engine.poll_events().empty());
}
void shutdown_order_and_destructor(){
    auto trace=std::make_shared<Trace>();
    {
        Engine engine(std::make_unique<Runner>(trace),{2,1,1,128});
        engine.submit(request("resident",3));engine.submit(request("unstarted"));engine.submit(request("waiting"));
        engine.step();engine.shutdown();auto events=engine.poll_events();assert(events.size()==3);
        event(events[0],"resident",std::nullopt,FinishReason::Cancelled);
        event(events[1],"unstarted",std::nullopt,FinishReason::Cancelled);
        event(events[2],"waiting",std::nullopt,FinishReason::Cancelled);
        assert(engine.idle()&&trace->resident.empty());
        const auto count=trace->operations.size();engine.shutdown();
        assert(trace->operations.size()==count+1&&trace->operations.back()=="clear"&&engine.poll_events().empty());
    }
    assert(trace->resident.empty());
    auto destruction=std::make_shared<Trace>();
    {Engine engine(std::make_unique<Runner>(destruction),{1,1,1,128});engine.submit(request("unfinished",3));engine.step();}
    assert((destruction->operations==std::vector<std::string>{"create:unfinished","append:unfinished","release:unfinished","clear"}));
}
void retry_partial_shutdown(){
    auto trace=std::make_shared<Trace>();Engine engine(std::make_unique<Runner>(trace),{2,2,1,128});
    engine.submit(request("first",3));engine.submit(request("second",3));engine.submit(request("waiting"));engine.step();
    trace->fail_release="second";rejects([&]{engine.shutdown();});
    auto events=engine.poll_events();assert(events.size()==1);event(events[0],"first",std::nullopt,FinishReason::Cancelled);
    trace->fail_release.clear();engine.shutdown();events=engine.poll_events();assert(events.size()==2);
    event(events[0],"second",std::nullopt,FinishReason::Cancelled);event(events[1],"waiting",std::nullopt,FinishReason::Cancelled);
    assert(engine.idle()&&trace->resident.empty());
}
void retry_cancellation(){
    auto trace=std::make_shared<Trace>();Engine engine(std::make_unique<Runner>(trace),{1,1,1,128});
    engine.submit(request("resident",3));engine.step();trace->fail_release="resident";
    rejects([&]{engine.cancel("resident");});assert(engine.poll_events().empty()&&!engine.idle());
    trace->fail_release.clear();engine.cancel("resident");auto events=engine.poll_events();
    assert(events.size()==1);event(events[0],"resident",std::nullopt,FinishReason::Cancelled);
    assert(engine.idle()&&trace->resident.empty());
}
void scheduling_after_partial_shutdown(){
    auto trace=std::make_shared<Trace>();Engine engine(std::make_unique<Runner>(trace),{2,2,1,128});
    engine.submit(request("first",3));engine.submit(request("second",3));engine.submit(request("waiting"));engine.step();
    trace->fail_release="second";rejects([&]{engine.shutdown();});
    auto events=engine.poll_events();assert(events.size()==1);event(events[0],"first",std::nullopt,FinishReason::Cancelled);
    trace->fail_release.clear();
    engine.step();events=engine.poll_events();assert(events.size()==1);event(events[0],"waiting",2,FinishReason::Length);
    engine.step();events=engine.poll_events();assert(events.size()==1);event(events[0],"second",2,FinishReason::Length);
    assert(engine.idle()&&trace->resident.empty());
    assert(std::count(trace->operations.begin(),trace->operations.end(),"append:first")==1);
}
}
int main(int argc,char** argv){
    const bool legacy=argc==2&&std::string(argv[1])=="--legacy-compatible";
    if(argc!=1&&!legacy)return 2;
    completion();cancellation();failure_preserves_prior_events(false);failure_preserves_prior_events(true);
    shutdown_order_and_destructor();retry_cancellation();if(!legacy){retry_partial_shutdown();scheduling_after_partial_shutdown();}
    std::cout<<"terminal events, ordering, release ownership and retries passed\n";}
