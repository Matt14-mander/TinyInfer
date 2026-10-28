#pragma once
#include "model.h"
#include "scheduler.h"
#include <deque>
#include <set>

namespace miniprefix {
struct EngineConfig { int max_batch_size=2,token_budget=32,prefill_chunk=16,context=512; };
struct EngineStepStats {
    std::size_t prefix_reused_tokens=0,prefill_tokens=0,decode_tokens=0;
};
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
    bool failed() const;
    ModelRunner& runner(){return *runner_;}
    const EngineStepStats& last_step_stats() const{return step_stats_;}
private:
    struct State{Request request;std::size_t prefilled=0;std::vector<TokenId> output;bool resident=false,finished=false;};
    enum class TerminalCleanup { ReleaseSequence, RunnerManaged };
    std::unique_ptr<ModelRunner> runner_;EngineConfig config_;Scheduler scheduler_;
    std::deque<State> waiting_;std::vector<State> active_;std::set<RequestId> seen_;
    std::vector<GenerationEvent> events_;bool failed_=false;
    EngineStepStats step_stats_;
    void restore_prefix(State&);
    void terminal(State&,FinishReason,std::optional<TokenId> token=std::nullopt,
                  const std::string& error="",TerminalCleanup cleanup=TerminalCleanup::ReleaseSequence);
    void remove_finished(bool include_waiting=false);
};
} // namespace miniprefix
