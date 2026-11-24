#pragma once
#include "model.h"
#include "scheduler.h"
#include <deque>
#include <set>
#include <chrono>
#include <functional>

namespace miniprefix {
struct EngineConfig {
    int max_batch_size=2,token_budget=32,prefill_chunk=16,context=512;
    std::size_t max_waiting_requests=1024,max_event_slots=65536,history_capacity=4096;
    std::uint32_t history_ttl_ms=300000;
};
class QueueFullError:public std::runtime_error {public:using std::runtime_error::runtime_error;};
struct LifecycleStats {std::size_t waiting=0,active=0,buffered_events=0,event_slots=0,tracked_ids=0,history_ids=0;};
struct EngineStepStats {
    std::size_t prefix_reused_tokens=0,prefill_tokens=0,decode_tokens=0;
};
class Engine {
public:
    using Clock=std::chrono::steady_clock;
    using Now=std::function<Clock::time_point()>;
    Engine(std::unique_ptr<ModelRunner>,EngineConfig,Now now=std::chrono::steady_clock::now);
    LifecycleStats lifecycle_stats() const;
    ~Engine();
    void submit(Request);
    bool step();
    std::vector<GenerationEvent> poll_events();
    void cancel(const RequestId&);
    void shutdown();
    bool idle() const;
    bool failed() const;
    void set_execution_timing(bool enabled);
    ExecutionStats execution_stats() const;
    ModelRunner& runner(){return *runner_;}
    const EngineStepStats& last_step_stats() const{return step_stats_;}
private:
    struct State{Request request;std::size_t prefilled=0,kv_block_tokens=0;std::vector<TokenId> output;bool resident=false,finished=false;std::size_t emitted_events=0;std::optional<Clock::time_point> deadline;};
    enum class TerminalCleanup { ReleaseSequence, RunnerManaged };
    std::unique_ptr<ModelRunner> runner_;EngineConfig config_;Scheduler scheduler_;
    std::deque<State> waiting_;std::vector<State> active_;std::set<RequestId> seen_;
    std::vector<GenerationEvent> events_;bool failed_=false;
    Now now_;
    std::deque<std::pair<RequestId,Clock::time_point>> history_;
    std::size_t event_slots_=0;
    void prune_history(Clock::time_point);
    EngineStepStats step_stats_;
    bool execution_timing_=false;
    std::uint64_t admission_plan_ns_=0;
    bool prefix_cache_enabled_=true;
    void restore_prefix(State&);
    void terminal(State&,FinishReason,std::optional<TokenId> token=std::nullopt,
                  const std::string& error="",TerminalCleanup cleanup=TerminalCleanup::ReleaseSequence);
    void remove_finished(bool include_waiting=false);
};
} // namespace miniprefix
