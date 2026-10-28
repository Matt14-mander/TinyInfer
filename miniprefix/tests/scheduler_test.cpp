#include "miniprefix/scheduler.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace miniprefix;
namespace {
void policy_traces(){
    Scheduler scheduler({4,5,3});
    assert(scheduler.admission_count(0,8)==4);
    assert(scheduler.admission_count(3,8)==1);
    assert(scheduler.admission_count(4,8)==0);
    assert(scheduler.admission_count(5,8)==0);
    assert(scheduler.admission_count(1,0)==0);
    auto plan=scheduler.plan({SchedulePhase::Prefill,SchedulePhase::Decode,SchedulePhase::Prefill,SchedulePhase::Decode});
    const std::vector<std::size_t> indices={1,3,0,2};
    const std::vector<std::size_t> counts={1,1,3,0};
    for(std::size_t i=0;i<indices.size();++i){
        assert(plan.order()[i].active_index==indices[i]);
        assert(plan.take_input(plan.order()[i].phase,10)==counts[i]);
    }
    assert(!plan.has_budget());
    auto fresh=scheduler.plan({SchedulePhase::Prefill,SchedulePhase::Prefill});
    assert(fresh.take_input(SchedulePhase::Prefill,1)==1);
    assert(fresh.take_input(SchedulePhase::Prefill,9)==3);
    assert(fresh.has_budget()); // unused budget does not revisit a candidate
    assert(scheduler.plan({}).order().empty());
    bool rejected=false;
    try{fresh.take_input(SchedulePhase::Prefill,0);}catch(const std::invalid_argument&){rejected=true;}
    assert(rejected);
}
void legacy_policy_equivalence(){
    // Independent reference: original Engine's two passes and budget arithmetic.
    // Enumerate phase mixtures, including budgets smaller than decoder count.
    for(int size=0;size<=5;++size)for(int mask=0;mask<(1<<size);++mask)
        for(int budget:{1,2,5,12})for(int chunk:{1,3,8}){
            Scheduler scheduler({5,budget,chunk});std::vector<SchedulePhase> phases;
            for(int i=0;i<size;++i)phases.push_back(mask&(1<<i)?SchedulePhase::Decode:SchedulePhase::Prefill);
            std::vector<std::size_t> expected;
            for(int i=0;i<size;++i)if(mask&(1<<i))expected.push_back(i);
            for(int i=0;i<size;++i)if(!(mask&(1<<i)))expected.push_back(i);
            auto plan=scheduler.plan(phases);int remaining=budget;
            assert(plan.order().size()==expected.size());
            for(std::size_t j=0;j<expected.size();++j){
                const auto index=expected[j];const auto work=plan.order()[j];
                assert(work.active_index==index&&work.phase==phases[index]);
                assert(plan.has_budget()==(remaining>0));
                if(!remaining)break;
                const int suffix=int(index)*2+1;
                int count=1;
                if(phases[index]==SchedulePhase::Prefill){count=suffix;if(count>chunk)count=chunk;if(count>remaining)count=remaining;}
                assert(plan.take_input(work.phase,suffix)==std::size_t(count));remaining-=count;
            }
        }
    for(auto config:{SchedulerConfig{0,1,1},SchedulerConfig{1,0,1},SchedulerConfig{1,1,0}}){
        bool rejected=false;try{Scheduler scheduler(config);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    }
}
}
int main(){policy_traces();legacy_policy_equivalence();std::cout<<"scheduler policy traces passed\n";}
