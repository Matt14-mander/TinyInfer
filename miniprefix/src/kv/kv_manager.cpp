#include "miniprefix/kv_manager.h"

namespace miniprefix {
KVManager::KVManager(KvCacheConfig config,int layers,int width,int context,int vocabulary):
    layers_(layers),width_(width),context_(context),vocabulary_(vocabulary),
    pool_(config,layers,width),prefix_(pool_){
    if(context<1||vocabulary<1)throw std::invalid_argument("invalid KV manager context/vocabulary");
}
KVManager::~KVManager(){
    for(auto& item:sequences_)pool_.release(item.second.sequence);
    prefix_.clear();
}
const KvCacheConfig& KVManager::config() const{return pool_.config();}
int KVManager::context_limit() const{return context_;}
std::size_t KVManager::reserved_bytes() const{return pool_.reserved_bytes();}
std::size_t KVManager::max_sequence_blocks() const{
    return std::size_t(context_/config().block_tokens+(context_%config().block_tokens!=0));
}
bool KVManager::contains(const RequestId& id) const{return sequences_.count(id)!=0;}
const PagedSequence& KVManager::sequence(const RequestId& id) const{return sequences_.at(id).sequence;}
std::size_t KVManager::offset(const PagedSequence& sequence,int layer,int token) const{
    return pool_.offset(sequence,layer,token,layers_,width_);
}
KVManager::State& KVManager::idle_state(const RequestId& id){
    auto& state=sequences_.at(id);
    if(state.appending)throw std::logic_error("sequence has an active KV append");
    return state;
}
void KVManager::validate_tokens(const std::vector<TokenId>& tokens) const{
    for(auto token:tokens)if(token<0||token>=vocabulary_)throw std::invalid_argument("invalid token");
}
void KVManager::create(const RequestId& id){
    if(!sequences_.emplace(id,State{}).second)throw std::invalid_argument("sequence already exists");
}
std::size_t KVManager::reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt){
    if(prompt.empty()||prompt.size()>std::size_t(context_))throw std::invalid_argument("invalid prefix context");
    for(auto token:prompt)if(token<0||token>=vocabulary_)throw std::invalid_argument("invalid prefix token");
    return prefix_.reuse(idle_state(id).sequence,prompt);
}
void KVManager::cache_prefix(const RequestId& id){prefix_.publish(idle_state(id).sequence);}
void KVManager::clear_prefix_cache(){
    // Even cache-only pages may still be read by an in-flight append kernel.
    for(const auto& item:sequences_)if(item.second.appending)
        throw std::logic_error("cannot clear prefix cache during KV append");
    prefix_.clear();
}
KVManager::Append KVManager::prepare_append(const RequestId& id,const std::vector<TokenId>& tokens){
    return Append(*this,id,tokens);
}
KVManager::Append::Append(KVManager& owner,const RequestId& id,const std::vector<TokenId>& tokens):
    owner_(&owner),state_(&owner.idle_state(id)),tokens_(tokens),original_blocks_(state_->sequence.blocks.size()){
    auto& sequence=state_->sequence;
    if(tokens.empty()||tokens.size()>std::size_t(owner.context_-sequence.used))
        throw std::invalid_argument("context overflow/empty input");
    owner.validate_tokens(tokens);
    sequence.tokens.reserve(sequence.used+tokens.size());
    owner.prefix_.reserve(sequence,sequence.used+static_cast<int>(tokens.size()));
    state_->appending=true;
}
KVManager::Append::~Append(){rollback();}
const PagedSequence& KVManager::Append::sequence() const{
    if(!owner_)throw std::logic_error("KV append already finished");
    return state_->sequence;
}
std::size_t KVManager::Append::original_blocks() const{return original_blocks_;}
void KVManager::Append::commit(){
    if(!owner_)throw std::logic_error("KV append already finished");
    auto& sequence=state_->sequence;
    sequence.tokens.insert(sequence.tokens.end(),tokens_.begin(),tokens_.end());
    sequence.used+=static_cast<int>(tokens_.size());owner_->forward_tokens_+=tokens_.size();
    state_->appending=false;owner_=nullptr;
}
void KVManager::Append::rollback() noexcept{
    if(owner_){owner_->pool_.rollback(state_->sequence,original_blocks_);state_->appending=false;owner_=nullptr;}
}
void KVManager::reset_sequence(const RequestId& id){pool_.release(idle_state(id).sequence);}
void KVManager::release(const RequestId& id){
    auto it=sequences_.find(id);if(it==sequences_.end())throw std::invalid_argument("unknown sequence");
    if(it->second.appending)throw std::logic_error("cannot release sequence during KV append");
    pool_.release(it->second.sequence);sequences_.erase(it);
}
void KVManager::clear(){
    for(const auto& item:sequences_)if(item.second.appending)
        throw std::logic_error("cannot clear sequences during KV append");
    for(auto& item:sequences_)pool_.release(item.second.sequence);
    sequences_.clear();prefix_.clear();
}
KvStats KVManager::stats() const{
    KvStats result;result.live_sequences=sequences_.size();result.forward_tokens=forward_tokens_;
    result.kv_block_tokens=config().block_tokens;result.kv_total_blocks=config().max_blocks;
    result.kv_live_blocks=pool_.live_blocks();result.kv_peak_blocks=pool_.peak_blocks();
    result.kv_reserved_bytes=pool_.reserved_bytes();result.live_kv_bytes=pool_.live_blocks()*pool_.block_bytes();
    result.peak_kv_bytes=pool_.peak_blocks()*pool_.block_bytes();
    result.prefix_cache_capacity=config().prefix_cache_blocks;result.kv_cached_blocks=prefix_.size();
    result.kv_cache_only_blocks=prefix_.cache_only_blocks();result.prefix_hits=prefix_.hits();result.prefix_misses=prefix_.misses();
    result.prefix_reused_tokens=prefix_.reused_tokens();result.prefix_evictions=prefix_.evictions();return result;
}
} // namespace miniprefix
