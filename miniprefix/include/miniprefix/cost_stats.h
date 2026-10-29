#pragma once
#include <chrono>
#include <cstdint>
namespace miniprefix {
// Cumulative work counters survive release/clear. Durations are opt-in.
struct CostStats {
    std::uint64_t lookup_calls=0,lookup_ns=0,publish_calls=0,publish_ns=0;
    std::uint64_t cow_pages=0,cow_bytes=0,cow_copy_ns=0,cow_timed_copies=0;
};
class CostTimer {
    using Clock=std::chrono::steady_clock;
    std::uint64_t* total_;Clock::time_point start_;
public:
    CostTimer(bool enabled,std::uint64_t& total):total_(enabled?&total:nullptr){if(total_)start_=Clock::now();}
    ~CostTimer(){if(total_)*total_+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start_).count();}
    CostTimer(const CostTimer&)=delete;CostTimer& operator=(const CostTimer&)=delete;
};
}
