#pragma once
#include "model.h"
#include <deque>
#include <set>

namespace miniprefix {
struct EngineConfig { int max_batch_size=2,token_budget=32,prefill_chunk=16,context=512; };
class Engine {
public:
    Engine(std::unique_ptr<ModelRunner>,EngineConfig);
    ~Engine();
    void submit(Request);
    bool step();
    std::vector<GenerationEvent> poll_events();
    void cancel(const RequestId&);
    void shutdown();
    bool idle() const;
    ModelRunner& runner(){return *runner_;}
private:
    struct State{Request request;std::size_t prefilled=0;std::vector<TokenId> output;bool resident=false;};
    std::unique_ptr<ModelRunner> runner_;EngineConfig config_;
    std::deque<State> waiting_;std::vector<State> active_;std::set<RequestId> seen_;
    std::vector<GenerationEvent> events_;bool failed_=false;
    void terminal(State&,FinishReason,const std::string& error="");
};
} // namespace miniprefix
