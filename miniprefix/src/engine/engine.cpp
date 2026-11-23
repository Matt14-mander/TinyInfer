#include "miniprefix/engine.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>

namespace miniprefix {
namespace {
SchedulerConfig scheduler_config(const EngineConfig& config){
    if(config.max_batch_size<1||config.token_budget<1||config.prefill_chunk<1)
        throw std::invalid_argument("invalid engine configuration");
    return {config.max_batch_size,config.token_budget,config.prefill_chunk};
}
}
Engine::Engine(std::unique_ptr<ModelRunner> runner,EngineConfig config):runner_(std::move(runner)),config_(config),
    scheduler_(scheduler_config(config)){
    if(!runner_||config.max_batch_size<1||config.token_budget<1||config.prefill_chunk<1||config.context<1||config.context>runner_->config().context||config.context>runner_->context_limit())
        throw std::invalid_argument("invalid engine configuration");
}
Engine::~Engine(){try{shutdown();}catch(...){}}
void Engine::submit(Request request){
    if(failed_)throw std::runtime_error("engine failed; create a new engine");
    if(request.id.empty()||seen_.count(request.id)||request.prompt_token_ids.empty()||request.generation.max_new_tokens<1||
       request.prompt_token_ids.size()+std::uint64_t(request.generation.max_new_tokens)>std::uint64_t(config_.context))
        throw std::invalid_argument("invalid request, duplicate ID or context overflow");
    for(auto t:request.prompt_token_ids)if(t<0||t>=runner_->config().vocabulary)throw std::invalid_argument("invalid prompt token");
    if(request.generation.eos_token_id&&(*request.generation.eos_token_id<0||*request.generation.eos_token_id>=runner_->config().vocabulary))throw std::invalid_argument("invalid EOS");
    State state;state.request=std::move(request);waiting_.push_back(std::move(state));seen_.insert(waiting_.back().request.id);
}
void Engine::terminal(State& state,FinishReason reason,std::optional<TokenId> token,
                      const std::string& error,TerminalCleanup cleanup){
    if(state.finished)return;
    // Normal completion/cancellation releases its lease first. Fatal failure
    // delegates cleanup to the runner, even if clear failed; never force per-ID release.
    if(state.resident&&cleanup==TerminalCleanup::ReleaseSequence)runner_->release(state.request.id);
    runner_->release_reservation(state.request.id);
    state.resident=false;
    events_.push_back({state.request.id,token,reason,error});
    state.finished=true;
}
void Engine::remove_finished(bool include_waiting){
    const auto finished=[](const State& state){return state.finished;};
    active_.erase(std::remove_if(active_.begin(),active_.end(),finished),active_.end());
    if(include_waiting)waiting_.erase(std::remove_if(waiting_.begin(),waiting_.end(),finished),waiting_.end());
}
void Engine::restore_prefix(State& state){
    auto& request=state.request;
    runner_->create(request.id);state.resident=true;
    const auto reused=runner_->reuse_prefix(request.id,request.prompt_token_ids);
    // No cached logits are returned by ModelRunner: at least one prompt token
    // must remain. Reject a broken runner contract before forming a suffix or
    // executing an empty append. A reusable prefix may end inside a page.
    const auto stats=runner_->stats();
    prefix_cache_enabled_=stats.prefix_cache_capacity!=0;
    const auto block_tokens=stats.kv_block_tokens;
    if(reused>=request.prompt_token_ids.size()||(reused&&!block_tokens))
        throw std::runtime_error("invalid reusable prefix range");
    state.kv_block_tokens=block_tokens;state.prefilled=reused;step_stats_.prefix_reused_tokens+=reused;
}
void Engine::set_execution_timing(bool enabled){
    runner_->set_execution_timing(enabled);
    execution_timing_=enabled;
}
ExecutionStats Engine::execution_stats() const{
    auto stats=runner_->execution_stats();
    stats.admission_plan_ns=admission_plan_ns_;
    return stats;
}
bool Engine::step(){
    step_stats_={};
    if(failed_)throw std::runtime_error("engine failed");
    bool progressed=false;
    try{
        using Clock=std::chrono::steady_clock;
        const auto begin=execution_timing_?Clock::now():Clock::time_point{};
        auto slots=scheduler_.admission_count(active_.size(),waiting_.size());
        while(slots&&!waiting_.empty()){
            auto& state=waiting_.front();
            const auto& request=state.request;
            // The final sampled token is delivered, never appended to KV.
            const auto tokens=request.prompt_token_ids.size()+std::size_t(request.generation.max_new_tokens)-1;
            const auto admission=runner_->reserve_request(request.id,tokens,request.prompt_token_ids);
            if(admission==KvAdmission::Wait&&!active_.empty())break;
            if(admission!=KvAdmission::Ready){
                terminal(state,FinishReason::Error,std::nullopt,
                    admission==KvAdmission::Impossible?"request exceeds total KV capacity":
                    "KV capacity unavailable without active requests to make progress");
                waiting_.pop_front();progressed=true;continue;
            }
            active_.push_back(std::move(state));waiting_.pop_front();--slots;progressed=true;
        }
        if(active_.empty())return progressed;
        std::vector<SchedulePhase> phases;
        for(const auto& state:active_)phases.push_back(state.output.empty()?SchedulePhase::Prefill:SchedulePhase::Decode);
        auto plan=scheduler_.plan(phases);
        if(execution_timing_)admission_plan_ns_+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
        using Results=std::vector<GenerationResult>;
        std::optional<Results> results;
        std::size_t group_begin=0,group_end=0;
        std::exception_ptr restore_failure;
        std::size_t failed_restore_at=0;
        const auto& order=plan.order();
        auto input_for=[](const State& state,SchedulePhase phase,std::size_t count){
            if(phase==SchedulePhase::Decode)return std::vector<TokenId>{state.output.back()};
            auto begin=state.request.prompt_token_ids.begin()+state.prefilled;
            return std::vector<TokenId>(begin,begin+count);
        };
        for(std::size_t i=0;i<order.size() && plan.has_budget();++i){
            const auto& work=order[i];
            auto& s=active_[work.active_index];auto& r=s.request;
            try{
                if(restore_failure && i==failed_restore_at)std::rethrow_exception(restore_failure);
                if(!s.resident)restore_prefix(s);
                if(i>=group_end){
                    results.reset();
                    group_begin=i;
                    group_end=i+1;
                    const bool packed=runner_->supports_packed_prefill();
                    if(i+1<order.size() && (packed || work.phase==SchedulePhase::Decode)){
                        auto probe=plan;
                        std::vector<BatchInput> chunks;
                        for(std::size_t j=i;j<order.size() && probe.has_budget();++j){
                            const auto phase=order[j].phase;
                            auto& candidate=active_[order[j].active_index];
                            if(!packed && phase!=SchedulePhase::Decode)break;
                            if(!candidate.resident){
                                // A prior row may publish or evict the exact prefix
                                // this request will restore. Flush before its lookup.
                                if(prefix_cache_enabled_ || !packed)break;
                                try{restore_prefix(candidate);}
                                catch(...){
                                    restore_failure=std::current_exception();
                                    failed_restore_at=j;
                                    break;
                                }
                            }
                            const auto count=probe.take_input(phase,candidate.request.prompt_token_ids.size()-candidate.prefilled);
                            chunks.push_back({candidate.request.id,input_for(candidate,phase,count),phase==SchedulePhase::Decode});
                        }
                        if(chunks.size()>1){
                            group_end=i+chunks.size();
                            results=runner_->append_generation_batch(chunks);
                            if(results && results->size()!=chunks.size())throw std::runtime_error("invalid batch arity");
                        }
                    }
                }
                const bool prefilling=work.phase==SchedulePhase::Prefill;
                const auto count=plan.take_input(work.phase,r.prompt_token_ids.size()-s.prefilled);
                auto result=results?std::move((*results)[i-group_begin]):runner_->append_generation(r.id,input_for(s,work.phase,count));
                if(prefilling){s.prefilled+=count;step_stats_.prefill_tokens+=count;}
                else step_stats_.decode_tokens+=count;
                if(s.prefilled<r.prompt_token_ids.size())continue;
                TokenId token;
                if(auto* logits=std::get_if<std::vector<float>>(&result)){
                    if(logits->size()!=std::size_t(runner_->config().vocabulary))throw std::runtime_error("invalid logits arity");
                    for(float v:*logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite model result");
                    token=static_cast<TokenId>(std::max_element(logits->begin(),logits->end())-logits->begin());
                }else{
                    token=std::get<TokenId>(result);
                    if(token<0 || token>=runner_->config().vocabulary)throw std::runtime_error("invalid sampled token");
                }
                // append committed the prompt or the PREVIOUS sampled token.
                // Publish committed history before delivering the selected token.
                runner_->cache_prefix(r.id);
                s.output.push_back(token);std::optional<FinishReason> finish;
                if(r.generation.eos_token_id&&token==*r.generation.eos_token_id)finish=FinishReason::Eos;
                else if(s.output.size()==r.generation.max_new_tokens)finish=FinishReason::Length;
                if(finish)terminal(s,*finish,token);
                else events_.push_back({r.id,token,std::nullopt,{}});
            }catch(const KvCapacityError& error){
                // Defensive isolation for bounded runners/direct diagnostic interference.
                // A failed release still reaches the fatal backend handler below.
                terminal(s,FinishReason::Error,std::nullopt,error.what());
            }
        }
        remove_finished();
    }catch(const std::exception& error){
        // Cleanup may itself fail (for example, CUDA stream synchronization).
        // Poison the engine first; backend ownership must survive failed cleanup.
        failed_=true;
        std::string message=error.what();
        try{runner_->clear();}
        catch(const std::exception& cleanup){message+="; cleanup failed: ";message+=cleanup.what();}
        catch(...){message+="; cleanup failed: unknown exception";}
        for(auto& s:active_)terminal(s,FinishReason::Error,std::nullopt,message,TerminalCleanup::RunnerManaged);
        for(auto& s:waiting_)terminal(s,FinishReason::Error,std::nullopt,message,TerminalCleanup::RunnerManaged);
        remove_finished(true);
    }
    return true;
}
std::vector<GenerationEvent> Engine::poll_events(){auto result=std::move(events_);events_.clear();return result;}
void Engine::cancel(const RequestId& id){
    for(auto& state:waiting_)if(state.request.id==id&&!state.finished){terminal(state,FinishReason::Cancelled);remove_finished(true);return;}
    for(auto& state:active_)if(state.request.id==id&&!state.finished){terminal(state,FinishReason::Cancelled);remove_finished();return;}
    throw std::invalid_argument("request is unknown or already terminal");
}
void Engine::shutdown(){
    try{
        for(auto& s:active_)terminal(s,FinishReason::Cancelled);
        for(auto& s:waiting_)terminal(s,FinishReason::Cancelled);
    }catch(...){remove_finished(true);throw;}
    remove_finished(true);runner_->clear();
}
bool Engine::idle() const{return active_.empty()&&waiting_.empty();}
bool Engine::failed() const{return failed_;}
} // namespace miniprefix
