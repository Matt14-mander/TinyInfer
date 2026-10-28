#pragma once
#include "miniprefix/request.h"
namespace miniprefix::protocol {
enum class Kind { Submit, Step, StepStats, Cancel, ClearPrefix, Cache, Reuse, Forward, Release, Stats, Quit };
struct Command { Kind kind; int id=0; Request request; std::vector<TokenId> input; };
// Protocol v3 deliberately accepts trailing fields and preserves validation order.
Command parse(const std::string& line);
}
