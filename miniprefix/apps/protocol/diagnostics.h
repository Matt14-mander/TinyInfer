#pragma once
#include "miniprefix/model.h"
#include <set>
namespace miniprefix::protocol {
// Diagnostic IDs intentionally occupy a separate namespace from Engine IDs.
// Failed append/reuse retains a successfully created sequence, as in protocol v3.
class Diagnostics {
    ModelRunner& runner_;
    std::set<std::string> created_;
    static std::string key(int id);
    std::string ensure(int id);
public:
    explicit Diagnostics(ModelRunner& runner):runner_(runner){}
    std::vector<float> forward(int id,const std::vector<TokenId>& tokens);
    std::size_t reuse(int id,const std::vector<TokenId>& tokens);
    void cache(int id);
    void release(int id);
    void clear_prefix();
};
}
