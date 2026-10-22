#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace miniprefix {

// Initial value contracts, not a runnable Engine API. Validation, scheduling,
// tokenization and execution are introduced with their respective milestones.
using TokenId = std::int32_t;
using RequestId = std::string;

enum class FinishReason { Eos, Length, Cancelled, Error };

struct GenerationOptions {
    std::uint32_t max_new_tokens = 32;
    std::optional<TokenId> eos_token_id;
    // M1-M4 use greedy decoding. Sampling options belong here when supported.
};

struct Request {
    RequestId id;
    std::vector<TokenId> prompt_token_ids;
    GenerationOptions generation;
};

// A committed output token is emitted exactly once, including after replay.
// The finish-only event has no token; the final token may also carry a reason.
struct GenerationEvent {
    RequestId request_id;
    std::optional<TokenId> token_id;
    std::optional<FinishReason> finish_reason;
    std::string error_message;
};

} // namespace miniprefix
