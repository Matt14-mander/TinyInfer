#include "miniprefix/engine.h"
#include <algorithm>
#include <cmath>

namespace miniprefix {
Engine::Engine(std::unique_ptr<ModelRunner> runner,EngineConfig config):runner_(std::move(runner)),config_(config){
    if(!runner_||config.max_batch_size<1||config.token_budget<1||config.prefill_chunk<1||config.context<1||config.context>runner_->config().context)
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
void Engine::terminal(State& state,FinishReason reason,const std::string& error){
    if(state.resident){runner_->release(state.request.id);state.resident=false;}
    events_.push_back({state.request.id,std::nullopt,reason,error});
}
void Engine::restore_prefix(State& state){
    auto& request=state.request;
    runner_->create(request.id);state.resident=true;
    const auto reused=runner_->reuse_prefix(request.id,request.prompt_token_ids);
    // No cached logits are returned by ModelRunner: at least one prompt token
    // must remain. Reject a broken runner contract before forming a suffix or
    // executing an empty append. Reused positions must end at a full page.
    const auto block_tokens=runner_->stats().kv_block_tokens;
    if(reused>=request.prompt_token_ids.size()||(reused&&(!block_tokens||reused%block_tokens)))
        throw std::runtime_error("invalid reusable prefix range");
    state.prefilled=reused;step_stats_.prefix_reused_tokens+=reused;
}
bool Engine::step(){
    step_stats_={};
    if(failed_)throw std::runtime_error("engine failed");
    while(!waiting_.empty()&&active_.size()<std::size_t(config_.max_batch_size)){
        active_.push_back(std::move(waiting_.front()));waiting_.pop_front();}
    if(active_.empty())return false;
    int remaining=config_.token_budget;std::set<RequestId> done;
    try{
        // Existing decoders first, then one prefill chunk per active request.
        std::vector<std::size_t> order;
        for(std::size_t i=0;i<active_.size();++i)if(!active_[i].output.empty())order.push_back(i);
        for(std::size_t i=0;i<active_.size();++i)if(active_[i].output.empty())order.push_back(i);
        for(auto index:order){if(!remaining)break;auto& s=active_[index];auto& r=s.request;
            if(!s.resident)restore_prefix(s);
            const bool prefilling=s.output.empty();
            std::vector<TokenId> input;
            if(!prefilling)input={s.output.back()};
            else{auto count=std::min({r.prompt_token_ids.size()-s.prefilled,std::size_t(config_.prefill_chunk),std::size_t(remaining)});
                input.assign(r.prompt_token_ids.begin()+s.prefilled,r.prompt_token_ids.begin()+s.prefilled+count);}
            auto logits=runner_->append(r.id,input);remaining-=static_cast<int>(input.size());
            if(prefilling){s.prefilled+=input.size();step_stats_.prefill_tokens+=input.size();}
            else step_stats_.decode_tokens+=input.size();
            if(s.prefilled<r.prompt_token_ids.size())continue;
            if(logits.size()!=std::size_t(runner_->config().vocabulary))throw std::runtime_error("invalid logits arity");
            for(float v:logits)if(!std::isfinite(v))throw std::runtime_error("nonfinite model result");
            if(prefilling)runner_->cache_prefix(r.id);
            auto token=static_cast<TokenId>(std::max_element(logits.begin(),logits.end())-logits.begin());
            s.output.push_back(token);std::optional<FinishReason> finish;
            if(r.generation.eos_token_id&&token==*r.generation.eos_token_id)finish=FinishReason::Eos;
            else if(s.output.size()==r.generation.max_new_tokens)finish=FinishReason::Length;
            if(finish){runner_->release(r.id);s.resident=false;done.insert(r.id);}
            events_.push_back({r.id,token,finish,{}});
        }
        active_.erase(std::remove_if(active_.begin(),active_.end(),[&](const State& s){return done.count(s.request.id)!=0;}),active_.end());
    }catch(const std::exception& error){
        // Synchronous runners return only after device access is terminal.
        runner_->clear();failed_=true;
        for(auto& s:active_)if(!done.count(s.request.id))events_.push_back({s.request.id,std::nullopt,FinishReason::Error,error.what()});
        for(auto& s:waiting_)events_.push_back({s.request.id,std::nullopt,FinishReason::Error,error.what()});
        active_.clear();waiting_.clear();
    }
    return true;
}
std::vector<GenerationEvent> Engine::poll_events(){auto result=std::move(events_);events_.clear();return result;}
void Engine::cancel(const RequestId& id){
    for(auto it=waiting_.begin();it!=waiting_.end();++it)if(it->request.id==id){terminal(*it,FinishReason::Cancelled);waiting_.erase(it);return;}
    for(auto it=active_.begin();it!=active_.end();++it)if(it->request.id==id){terminal(*it,FinishReason::Cancelled);active_.erase(it);return;}
    throw std::invalid_argument("request is unknown or already terminal");
}
void Engine::shutdown(){for(auto& s:active_)terminal(s,FinishReason::Cancelled);for(auto& s:waiting_)terminal(s,FinishReason::Cancelled);active_.clear();waiting_.clear();runner_->clear();}
bool Engine::idle() const{return active_.empty()&&waiting_.empty();}
} // namespace miniprefix
