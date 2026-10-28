#include "miniprefix/scheduler.h"
#include <algorithm>
#include <stdexcept>

namespace miniprefix {
Scheduler::Scheduler(SchedulerConfig config):config_(config){
    if(config.max_batch_size<1||config.token_budget<1||config.prefill_chunk<1)
        throw std::invalid_argument("invalid scheduler configuration");
}
std::size_t Scheduler::admission_count(std::size_t active,std::size_t waiting) const{
    const auto capacity=std::size_t(config_.max_batch_size);
    return active>=capacity?0:std::min(waiting,capacity-active);
}
SchedulePlan Scheduler::plan(const std::vector<SchedulePhase>& phases) const{
    return SchedulePlan(phases,config_);
}
SchedulePlan::SchedulePlan(const std::vector<SchedulePhase>& phases,const SchedulerConfig& config):
    remaining_(config.token_budget),chunk_(config.prefill_chunk){
    for(auto phase:{SchedulePhase::Decode,SchedulePhase::Prefill})
        for(std::size_t i=0;i<phases.size();++i)
            if(phases[i]==phase)order_.push_back({i,phase});
}
const std::vector<ScheduledRequest>& SchedulePlan::order() const{return order_;}
bool SchedulePlan::has_budget() const{return remaining_!=0;}
std::size_t SchedulePlan::take_input(SchedulePhase phase,std::size_t unprefilled_tokens){
    if(!remaining_)return 0;
    if(phase==SchedulePhase::Prefill&&!unprefilled_tokens)
        throw std::invalid_argument("prefill requires an uncached suffix");
    const auto count=phase==SchedulePhase::Decode?std::size_t(1):
        std::min({unprefilled_tokens,chunk_,remaining_});
    remaining_-=count;return count;
}
} // namespace miniprefix
