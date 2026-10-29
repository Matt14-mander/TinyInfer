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
std::size_t KVManager::committed_blocks() const{
    std::size_t count=0;
    for(const auto& item:reservations_)count+=item.second;
    for(const auto& item:sequences_)if(!reservations_.count(item.first))count+=item.second.sequence.blocks.size();
    return count;
}
KvAdmission KVManager::reserve_request(const RequestId& id,std::size_t max_tokens){
    if(contains(id)||reservations_.count(id))throw std::invalid_argument("duplicate KV reservation");
    if(!max_tokens||max_tokens>std::size_t(context_))throw std::invalid_argument("invalid KV reservation context");
    const auto pages=1+(max_tokens-1)/config().block_tokens;
    const auto total=std::size_t(config().max_blocks);
    if(pages>total)return KvAdmission::Impossible;
    const auto committed=committed_blocks();
    if(committed>total||pages>total-committed)return KvAdmission::Wait;
    reservations_.emplace(id,pages);return KvAdmission::Ready;
}
void KVManager::release_reservation(const RequestId& id) noexcept{reservations_.erase(id);}
void KVManager::create(const RequestId& id){
    if(!sequences_.emplace(id,State{}).second)throw std::invalid_argument("sequence already exists");
}
std::size_t KVManager::reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt){
    if(prompt.empty()||prompt.size()>std::size_t(context_))throw std::invalid_argument("invalid prefix context");
    for(auto token:prompt)if(token<0||token>=vocabulary_)throw std::invalid_argument("invalid prefix token");
    auto& sequence=idle_state(id).sequence;
    const auto count=prefix_.reuse(sequence,prompt);
    const auto reservation=reservations_.find(id);
    if(reservation!=reservations_.end()&&sequence.blocks.size()>reservation->second){
        pool_.release(sequence);throw KvCapacityError("request exceeds KV reservation");
    }
    if(!reservations_.empty()&&committed_blocks()>std::size_t(config().max_blocks)){
        pool_.release(sequence);throw KvCapacityError("KV capacity reserved by active requests");
    }
    return count;
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
void KVManager::reserve_pages(const RequestId& id,PagedSequence& sequence,int end){
    const auto tokens=config().block_tokens;
    const auto count=std::size_t(end/tokens+(end%tokens!=0));
    const auto required=count>sequence.blocks.size()?count-sequence.blocks.size():0;
    if(!reservations_.empty()){
        const auto found=reservations_.find(id);
        if(found!=reservations_.end()){
            if(count>found->second)throw KvCapacityError("request exceeds KV reservation");
        }else{
            const auto committed=committed_blocks(),total=std::size_t(config().max_blocks);
            if(committed>total||required>total-committed)
                throw KvCapacityError("KV capacity reserved by active requests");
        }
    }
    // Eviction only drops a cache lease. Recheck physical capacity after every
    // victim, including pages still pinned by active sequences. Preserve LRU
    // order and eviction side effects even if reservation ultimately fails.
    while(pool_.free_blocks()<required&&prefix_.evict_one()){}
    pool_.reserve(sequence,end);
}
KVManager::Append::Append(KVManager& owner,const RequestId& id,const std::vector<TokenId>& tokens):
    owner_(&owner),state_(&owner.idle_state(id)),tokens_(tokens),original_blocks_(state_->sequence.blocks.size()){
    auto& sequence=state_->sequence;
    if(tokens.empty()||tokens.size()>std::size_t(owner.context_-sequence.used))
        throw std::invalid_argument("context overflow/empty input");
    owner.validate_tokens(tokens);
    sequence.tokens.reserve(sequence.used+tokens.size());
    owner.reserve_pages(id,sequence,sequence.used+static_cast<int>(tokens.size()));
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
    pool_.release(it->second.sequence);sequences_.erase(it);reservations_.erase(id);
}
void KVManager::clear(){
    for(const auto& item:sequences_)if(item.second.appending)
        throw std::logic_error("cannot clear sequences during KV append");
    for(auto& item:sequences_)pool_.release(item.second.sequence);
    sequences_.clear();prefix_.clear();reservations_.clear();
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
