#include "diagnostics.h"
namespace miniprefix::protocol {
std::string Diagnostics::key(int id){return "reference/"+std::to_string(id);}
std::string Diagnostics::ensure(int id){auto name=key(id);if(!created_.count(name)){runner_.create(name);created_.insert(name);}return name;}
std::vector<float> Diagnostics::forward(int id,const std::vector<TokenId>& tokens){return runner_.append(ensure(id),tokens);}
std::size_t Diagnostics::reuse(int id,const std::vector<TokenId>& tokens){return runner_.reuse_prefix(ensure(id),tokens);}
void Diagnostics::cache(int id){runner_.cache_prefix(key(id));}
void Diagnostics::release(int id){auto name=key(id);runner_.release(name);created_.erase(name);}
void Diagnostics::clear_prefix(){runner_.clear_prefix_cache();}
}
