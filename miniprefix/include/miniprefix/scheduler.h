#pragma once
#include <cstddef>
#include <vector>

namespace miniprefix {
enum class SchedulePhase { Prefill, Decode };
struct ScheduledRequest { std::size_t active_index; SchedulePhase phase; };
struct SchedulerConfig { int max_batch_size=2,token_budget=32,prefill_chunk=16; };

// One iteration's stable candidate order and input budget. Resolve each chunk
// only after Engine restores that request's prefix: an earlier request may
// publish a reusable prefix in the same iteration. No request/device ownership.
class SchedulePlan {
    friend class Scheduler;
    std::vector<ScheduledRequest> order_;
    std::size_t remaining_,chunk_;
    SchedulePlan(const std::vector<SchedulePhase>& phases,const SchedulerConfig& config);
public:
    const std::vector<ScheduledRequest>& order() const;
    bool has_budget() const;
    std::size_t take_input(SchedulePhase phase,std::size_t unprefilled_tokens);
};

class Scheduler {
    SchedulerConfig config_;
public:
    explicit Scheduler(SchedulerConfig config);
    // Engine moves this many requests from the front of its waiting queue once
    // at the iteration boundary, never refilling after a same-step completion.
    std::size_t admission_count(std::size_t active,std::size_t waiting) const;
    SchedulePlan plan(const std::vector<SchedulePhase>& active_phases) const;
};
} // namespace miniprefix
