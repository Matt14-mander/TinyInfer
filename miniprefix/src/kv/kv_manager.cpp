#include "miniprefix/kv_manager.h"
#include <set>
#include <algorithm>

namespace miniprefix {
KVManager::KVManager(KvCacheConfig config,int layers,int width,int context,int vocabulary):
    layers_(layers),width_(width),context_(context),vocabulary_(vocabulary),
    pool_(config,layers,width),prefix_(pool_){
    if(context<1||vocabulary<1)throw std::invalid_argument("invalid KV manager context/vocabulary");
}
KVManager::~KVManager(){
    for(auto& item:sequences_)pool_.release(item.second.sequence);
    for(auto& item:admission_prefixes_)pool_.release(item.second);
    for(auto& item:prefix_leases_)pool_.release(item.second);
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
void KVManager::require_idle() const{
    if(append_active_)throw std::logic_error("KV append transaction is active");
}
KVManager::State& KVManager::idle_state(const RequestId& id){
    require_idle();
    auto& state=sequences_.at(id);
    if(state.appending)throw std::logic_error("sequence has an active KV append");
    return state;
}
void KVManager::validate_tokens(const std::vector<TokenId>& tokens) const{
    for(auto token:tokens)if(token<0||token>=vocabulary_)throw std::invalid_argument("invalid token");
}
std::size_t KVManager::committed_blocks() const{
    std::size_t count=0;std::set<int> shared;
    for(const auto& item:reservations_){
        const auto admission=admission_prefixes_.find(item.first);
        if(admission==admission_prefixes_.end()){count+=item.second;continue;}
        const auto state=sequences_.find(item.first);
        const auto& prefix=state!=sequences_.end()&&state->second.sequence.used?
            state->second.sequence:admission->second;
        const auto full=std::size_t(prefix.used)/config().block_tokens;
        shared.insert(prefix.blocks.begin(),prefix.blocks.begin()+full);
        count+=item.second-full;
    }
    count+=shared.size();
    for(const auto& item:prefix_leases_)count+=item.second.blocks.size();
    for(const auto& item:sequences_)if(!reservations_.count(item.first))count+=item.second.sequence.blocks.size();
    return count;
}
KvAdmission KVManager::reserve_request(const RequestId& id,std::size_t max_tokens){
    require_idle();
    if(contains(id)||reservations_.count(id))throw std::invalid_argument("duplicate KV reservation");
    if(!max_tokens||max_tokens>std::size_t(context_))throw std::invalid_argument("invalid KV reservation context");
    const auto pages=1+(max_tokens-1)/config().block_tokens;
    const auto total=std::size_t(config().max_blocks);
    if(pages>total)return KvAdmission::Impossible;
    const auto committed=committed_blocks();
    if(committed>total||pages>total-committed)return KvAdmission::Wait;
    reservations_.emplace(id,pages);return KvAdmission::Ready;
}
KvAdmission KVManager::reserve_request(const RequestId& id,std::size_t max_tokens,const std::vector<TokenId>& prompt){
    require_idle();
    if(contains(id)||reservations_.count(id))throw std::invalid_argument("duplicate KV reservation");
    if(!max_tokens||max_tokens>std::size_t(context_)||prompt.empty()||prompt.size()>max_tokens)
        throw std::invalid_argument("invalid KV reservation context");
    validate_tokens(prompt);
    const auto pages=1+(max_tokens-1)/config().block_tokens;
    if(pages>std::size_t(config().max_blocks))return KvAdmission::Impossible;
    auto inserted=admission_prefixes_.emplace(id,PagedSequence{});
    try{
        auto& prefix=inserted.first->second;prefix_.acquire(prefix,prompt);
        const auto full=std::size_t(prefix.used)/config().block_tokens;
        pool_.rollback(prefix,full);prefix.used=static_cast<int>(full*config().block_tokens);
        prefix.tokens.resize(prefix.used);
        reservations_.emplace(id,pages);
        if(committed_blocks()<=std::size_t(config().max_blocks))return KvAdmission::Ready;
    }catch(...){release_reservation(id);throw;}
    release_reservation(id);return KvAdmission::Wait;
}
void KVManager::release_reservation(const RequestId& id) noexcept{
    auto found=admission_prefixes_.find(id);
    if(found!=admission_prefixes_.end()){pool_.release(found->second);admission_prefixes_.erase(found);}
    reservations_.erase(id);
}
void KVManager::create(const RequestId& id){
    require_idle();
    if(!sequences_.emplace(id,State{}).second)throw std::invalid_argument("sequence already exists");
}
void KVManager::fork_sequence(const RequestId& source,const RequestId& target){
    require_idle();
    if(admission_prefixes_.count(target))throw std::logic_error("cannot fork into a sharing-aware reservation");
    const auto& original=sequences_.at(source).sequence;auto& destination=idle_state(target).sequence;
    if(source==target||destination.used||!destination.blocks.empty()||!destination.tokens.empty())
        throw std::invalid_argument("fork requires a distinct empty target");
    const auto reservation=reservations_.find(target);
    if(reservation!=reservations_.end()){
        if(original.blocks.size()>reservation->second)throw KvCapacityError("fork exceeds KV reservation");
    }else{
        const auto used=committed_blocks(),total=std::size_t(config().max_blocks);
        if(used>total||original.blocks.size()>total-used)throw KvCapacityError("fork exceeds unreserved KV capacity");
    }
    auto snapshot=original; // Allocate host metadata before acquiring any lease.
    for(int block:snapshot.blocks)pool_.retain(block);
    destination=std::move(snapshot);
}
std::size_t KVManager::reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt){
    if(prompt.empty()||prompt.size()>std::size_t(context_))throw std::invalid_argument("invalid prefix context");
    for(auto token:prompt)if(token<0||token>=vocabulary_)throw std::invalid_argument("invalid prefix token");
    auto& sequence=idle_state(id).sequence;
    ++costs_.lookup_calls;CostTimer timer(cost_timing_,costs_.lookup_ns);
    const auto admission=admission_prefixes_.find(id);
    if(admission!=admission_prefixes_.end()){
        if(sequence.used||!sequence.blocks.empty()||!sequence.tokens.empty())
            throw std::invalid_argument("prefix reuse requires an empty sequence");
        auto& pinned=admission->second;
        if(pinned.tokens.size()>=prompt.size()||!std::equal(pinned.tokens.begin(),pinned.tokens.end(),prompt.begin()))
            throw std::invalid_argument("prompt differs from admitted prefix");
        PagedSequence candidate;
        prefix_.acquire(candidate,prompt);
        // The candidate is speculative. Keep the guaranteed lease until accounting
        // accepts replacement; no payload allocation or cache mutation occurs here.
        if(candidate.used>=pinned.used){
            sequence=std::move(candidate);candidate={};
            try{
                if(sequence.blocks.size()>reservations_.at(id)||committed_blocks()>std::size_t(config().max_blocks))
                    pool_.release(sequence);
            }catch(...){pool_.release(sequence);throw;}
        }else pool_.release(candidate);
        if(!sequence.used){sequence=std::move(pinned);pinned={};}
        else pool_.release(pinned);
        prefix_.record_reuse(sequence,prompt);return sequence.used;
    }
    const auto count=prefix_.reuse(sequence,prompt);
    const auto reservation=reservations_.find(id);
    if(reservation!=reservations_.end()&&sequence.blocks.size()>reservation->second){
        pool_.release(sequence);throw KvCapacityError("request exceeds KV reservation");
    }
    if((!reservations_.empty()||!prefix_leases_.empty())&&committed_blocks()>std::size_t(config().max_blocks)){
        pool_.release(sequence);throw KvCapacityError("KV capacity reserved by active requests");
    }
    return count;
}
void KVManager::cache_prefix(const RequestId& id){
    auto& sequence=idle_state(id).sequence;++costs_.publish_calls;
    CostTimer timer(cost_timing_,costs_.publish_ns);prefix_.publish(sequence);
}
void KVManager::clear_prefix_cache(){
    require_idle();
    // Even cache-only pages may still be read by an in-flight append kernel.
    for(const auto& item:sequences_)if(item.second.appending)
        throw std::logic_error("cannot clear prefix cache during KV append");
    prefix_.clear();
}
KVManager::Append KVManager::prepare_append(const RequestId& id,const std::vector<TokenId>& tokens){
    return Append(*this,id,tokens);
}
std::vector<std::unique_ptr<KVManager::Append>> KVManager::prepare_decode_batch(const std::vector<DecodeInput>& inputs){
    std::vector<BatchInput> chunks;
    for(const auto& input:inputs)chunks.push_back({input.id,{input.token},true});
    return prepare_batch(chunks);
}
BatchSpacePlan KVManager::plan_batch(const std::vector<BatchInput>& inputs) const{
    require_idle();
    BatchSpacePlan result;
    if(inputs.size()<2)return result;
    std::set<RequestId> seen;std::size_t growth=0,unreserved_growth=0;bool unreserved=false;
    std::vector<int> tails;
    for(const auto& input:inputs){
        const auto found=sequences_.find(input.id);
        if(found==sequences_.end()||!seen.insert(input.id).second||input.tokens.empty())return {};
        const auto& s=found->second.sequence;
        if(input.tokens.size()>std::size_t(context_-s.used))return {};
        for(auto token:input.tokens)if(token<0||token>=vocabulary_)return {};
        const auto pending=admission_prefixes_.find(input.id);
        if(pending!=admission_prefixes_.end()&&pending->second.used)return {};
        const auto end=std::size_t(s.used)+input.tokens.size();
        const auto pages=1+(end-1)/config().block_tokens;
        const auto reservation=reservations_.find(input.id);
        if(reservation!=reservations_.end()&&pages>reservation->second)return {};
        growth+=pages-s.blocks.size();
        if(reservation==reservations_.end()){unreserved=true;unreserved_growth+=pages-s.blocks.size();}
        if(s.used%config().block_tokens)tails.push_back(s.blocks.back());
    }
    if(unreserved&&(!reservations_.empty()||!prefix_leases_.empty())){
        const auto committed=committed_blocks(),total=std::size_t(config().max_blocks);
        if(committed>total||unreserved_growth>total-committed)return {};
    }
    result.valid=true;result.growth_pages=growth;
    auto required=growth;
    for(int tail:tails)if(pool_.references(tail)>1)++required;
    result.cow_pages=required-growth;
    std::size_t victims=0;
    if(required>pool_.free_blocks()){
        // Read-only planning. A cache entry is a reference, not necessarily a free
        // page. Retain all active/admission/diagnostic references in this simulation.
        std::vector<std::size_t> references(config().max_blocks);
        for(std::size_t i=0;i<references.size();++i)references[i]=pool_.references(static_cast<int>(i));
        auto free=pool_.free_blocks();
        for(int block:prefix_.eviction_candidates()){
            ++victims;if(--references[block]==0)++free;
            required=growth;
            for(int tail:tails)if(references[tail]>1)++required;
            if(free>=required)break;
        }
        // No partial eviction on fallback, including an all-pinned cache.
        if(free<required)return result;
        result.freed_pages=free-pool_.free_blocks();
    }
    result.feasible=true;result.evicted_references=victims;
    result.cow_work_pages=required-growth;return result;
}
std::size_t KVManager::batch_prefix_size(const std::vector<BatchInput>& inputs){
    require_idle();
    if(batch_policy_==BatchPolicy::Full||inputs.size()<3)return inputs.size();
    ++batch_space_stats_.selection_calls;CostTimer timer(cost_timing_,batch_space_stats_.selection_ns);
    const auto full=plan_batch(inputs);
    if(!full.valid||(full.feasible&&full.evicted_references<2))return inputs.size();
    // Prefer the largest prefix that saves >=2 references and >=25% of eviction
    // cost per row. This is a proxy, not a prediction of latency or future hits.
    for(std::size_t n=inputs.size()-1;n>=2;--n){
        const auto plan=plan_batch(std::vector<BatchInput>(inputs.begin(),inputs.begin()+n));
        if(!plan.feasible)continue;
        if(!full.feasible || (plan.evicted_references+2<=full.evicted_references &&
            double(plan.evicted_references)*inputs.size()<=.75*double(full.evicted_references)*n)){
            ++batch_space_stats_.split_calls;batch_space_stats_.rows_deferred+=inputs.size()-n;
            if(full.feasible)batch_space_stats_.immediate_victims_avoided+=full.evicted_references-plan.evicted_references;
            return n;
        }
    }
    return inputs.size();
}
std::vector<std::unique_ptr<KVManager::Append>> KVManager::prepare_batch(const std::vector<BatchInput>& inputs){
    require_idle();++batch_space_stats_.prepare_calls;CostTimer timer(cost_timing_,batch_space_stats_.prepare_ns);
    const auto plan=plan_batch(inputs);
    if(!plan.feasible){++batch_space_stats_.rejected_batches;return {};}
    // Allocate host sequence capacity before making the planned LRU changes.
    for(const auto& input:inputs){
        auto& s=sequences_.at(input.id).sequence;
        const auto end=std::size_t(s.used)+input.tokens.size();
        s.tokens.reserve(end);s.blocks.reserve(1+(end-1)/config().block_tokens);
    }
    // This is a batch-boundary pressure operation, not speculative publication.
    // Once applied, eviction effects survive rollback just like ordinary append.
    std::vector<std::unique_ptr<Append>> result;result.reserve(inputs.size());
    const auto free_before=pool_.free_blocks();
    for(std::size_t i=0;i<plan.evicted_references;++i)prefix_.evict_one();
    batch_space_stats_.evicted_references+=plan.evicted_references;
    batch_space_stats_.freed_pages+=pool_.free_blocks()-free_before;
    batch_space_stats_.growth_pages+=plan.growth_pages;batch_space_stats_.cow_pages+=plan.cow_pages;
    batch_space_stats_.cow_work_pages+=plan.cow_work_pages;
    batch_space_stats_.cancelled_cow_pages+=plan.cow_pages-plan.cow_work_pages;
    try{for(const auto& input:inputs)result.emplace_back(new Append(*this,input.id,input.tokens,true));}
    catch(const KvCapacityError&){++batch_space_stats_.rejected_batches;return {};}
    ++batch_space_stats_.prepared_batches;return result;
}
void KVManager::reserve_pages(const RequestId& id,PagedSequence& sequence,int end,bool& cow){
    const auto pending=admission_prefixes_.find(id);
    if(pending!=admission_prefixes_.end()&&pending->second.used)
        throw std::logic_error("restore admitted prefix before append");
    const auto tokens=config().block_tokens;
    const auto count=std::size_t(end/tokens+(end%tokens!=0));
    const auto required=count>sequence.blocks.size()?count-sequence.blocks.size():0;
    if(!reservations_.empty()||!prefix_leases_.empty()){
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
    while(pool_.free_blocks()<required+std::size_t(cow)&&prefix_.evict_one()){
        // A partial tail may have been shared only with the cache. Eviction
        // removes that lease, so writing in place no longer needs a COW page.
        if(cow&&pool_.references(sequence.blocks.back())==1)cow=false;
    }
    if(cow&&pool_.free_blocks()<required+1)throw KvCapacityError("paged KV block pool exhausted");
    pool_.reserve(sequence,end);
}
KVManager::Append::Append(KVManager& owner,const RequestId& id,const std::vector<TokenId>& tokens,bool batch):
    owner_(&owner),state_(batch?&owner.sequences_.at(id):&owner.idle_state(id)),tokens_(tokens),original_blocks_(state_->sequence.blocks.size()){
    auto& sequence=state_->sequence;
    if(tokens.empty()||tokens.size()>std::size_t(owner.context_-sequence.used))
        throw std::invalid_argument("context overflow/empty input");
    owner.validate_tokens(tokens);
    sequence.tokens.reserve(sequence.used+tokens.size());
    const int valid=sequence.used%owner.config().block_tokens;
    bool cow=valid&&owner.pool_.references(sequence.blocks.back())>1;
    owner.reserve_pages(id,sequence,sequence.used+static_cast<int>(tokens.size()),cow);
    if(cow){
        // reserve_pages guaranteed room for the replacement and all new pages.
        const int old=sequence.blocks[original_blocks_-1];
        const int replacement=owner.pool_.allocate();
        copy_=TailCopy{old,replacement,valid};sequence.blocks[original_blocks_-1]=replacement;
        // The original target lease stays held by this transaction until commit.
    }
    state_->appending=true;++owner.append_active_;
}
KVManager::Append::~Append(){rollback();}
const PagedSequence& KVManager::Append::sequence() const{
    if(!owner_)throw std::logic_error("KV append already finished");
    return state_->sequence;
}
std::size_t KVManager::Append::original_blocks() const{return original_blocks_;}
const std::optional<KVManager::TailCopy>& KVManager::Append::tail_copy() const{return copy_;}
void KVManager::Append::copy_shared_tail(const std::function<void(std::size_t,std::size_t,std::size_t)>& copier){
    copy_shared_tail_region([&](const TailCopyRegion& region){
        for(std::size_t row=0;row<region.rows;++row)
            copier(region.destination+row*region.pitch,region.source+row*region.pitch,region.width);
    });
}
void KVManager::Append::copy_shared_tail_region(const std::function<void(const TailCopyRegion&)>& copier){
    if(!owner_)throw std::logic_error("KV append already finished");
    if(!copy_)return;
    copied_=false;
    const auto pitch=std::size_t(owner_->config().block_tokens)*owner_->width_;
    const auto rows=std::size_t(owner_->layers_);
    copier({std::size_t(copy_->destination)*rows*pitch,std::size_t(copy_->source)*rows*pitch,
            pitch,std::size_t(copy_->valid_tokens)*owner_->width_,rows});
    copied_=true;
}
void KVManager::Append::commit(){
    if(!owner_)throw std::logic_error("KV append already finished");
    if(copy_&&!copied_)throw std::logic_error("shared tail payload was not copied");
    auto& sequence=state_->sequence;
    sequence.tokens.insert(sequence.tokens.end(),tokens_.begin(),tokens_.end());
    sequence.used+=static_cast<int>(tokens_.size());owner_->forward_tokens_+=tokens_.size();
    if(copy_)owner_->pool_.unref(copy_->source);
    state_->appending=false;--owner_->append_active_;owner_=nullptr;
}
void KVManager::Append::rollback() noexcept{
    if(owner_){
        owner_->pool_.rollback(state_->sequence,original_blocks_);
        if(copy_){state_->sequence.blocks[original_blocks_-1]=copy_->source;owner_->pool_.unref(copy_->destination);}
        state_->appending=false;--owner_->append_active_;owner_=nullptr;
    }
}
void KVManager::reset_sequence(const RequestId& id){
    auto& sequence=idle_state(id).sequence;auto pending=admission_prefixes_.find(id);
    if(pending==admission_prefixes_.end()){pool_.release(sequence);return;}
    // Preserve credited immutable pages across backend restoration failure/reset.
    if(sequence.used){
        pool_.release(pending->second);pending->second=std::move(sequence);sequence={};
        auto& prefix=pending->second;const auto full=std::size_t(prefix.used)/config().block_tokens;
        pool_.rollback(prefix,full);prefix.used=static_cast<int>(full*config().block_tokens);prefix.tokens.resize(prefix.used);
    }
}
void KVManager::release(const RequestId& id){
    require_idle();
    auto it=sequences_.find(id);if(it==sequences_.end())throw std::invalid_argument("unknown sequence");
    if(it->second.appending)throw std::logic_error("cannot release sequence during KV append");
    pool_.release(it->second.sequence);sequences_.erase(it);release_reservation(id);
}
void KVManager::clear(){
    require_idle();
    for(const auto& item:sequences_)if(item.second.appending)
        throw std::logic_error("cannot clear sequences during KV append");
    for(auto& item:sequences_)pool_.release(item.second.sequence);
    for(auto& item:admission_prefixes_)pool_.release(item.second);
    for(auto& item:prefix_leases_)pool_.release(item.second);
    admission_prefixes_.clear();prefix_leases_.clear();sequences_.clear();prefix_.clear();reservations_.clear();
}
std::size_t KVManager::probe_prefix(const std::vector<TokenId>& prompt){
    require_idle();
    if(prompt.empty()||prompt.size()>std::size_t(context_))throw std::invalid_argument("invalid prefix context");
    validate_tokens(prompt);return prefix_.probe(prompt);
}
std::size_t KVManager::acquire_prefix_lease(const RequestId& id,const std::vector<TokenId>& prompt){
    require_idle();
    if(id.empty()||prefix_leases_.count(id))throw std::invalid_argument("invalid or duplicate prefix lease");
    const auto count=probe_prefix(prompt);
    const auto pages=(count+config().block_tokens-1)/config().block_tokens;
    const auto committed=committed_blocks(),total=std::size_t(config().max_blocks);
    if(committed>total||pages>total-committed)throw KvCapacityError("prefix lease exceeds unreserved KV capacity");
    // Allocate the owner before acquiring references; synchronous lookup cannot stale.
    auto inserted=prefix_leases_.emplace(id,PagedSequence{});
    try{return prefix_.acquire(inserted.first->second,prompt);}
    catch(...){pool_.release(inserted.first->second);prefix_leases_.erase(inserted.first);throw;}
}
void KVManager::release_prefix_lease(const RequestId& id){
    require_idle();auto found=prefix_leases_.find(id);
    if(found==prefix_leases_.end())return;
    pool_.release(found->second);prefix_leases_.erase(found);
}
KvCapacityLedger KVManager::capacity_ledger() const{
    require_idle();KvCapacityLedger result;
    result.committed_blocks=committed_blocks();
    result.total_blocks=config().max_blocks;result.physical_blocks=pool_.live_blocks();
    result.free_blocks=pool_.free_blocks();result.cache_only_blocks=prefix_.cache_only_blocks();
    std::set<int> pinned,leased;std::size_t references=0;
    auto collect=[&](const PagedSequence& s){
        references+=s.blocks.size();pinned.insert(s.blocks.begin(),s.blocks.end());
        if(s.used%config().block_tokens&&!s.blocks.empty()&&pool_.references(s.blocks.back())>1)
            ++result.cow_tail_candidates;
    };
    for(const auto& item:sequences_)collect(item.second.sequence);
    for(const auto& item:prefix_leases_){collect(item.second);leased.insert(item.second.blocks.begin(),item.second.blocks.end());}
    for(const auto& item:admission_prefixes_){collect(item.second);leased.insert(item.second.blocks.begin(),item.second.blocks.end());}
    result.pinned_blocks=pinned.size();result.prefix_lease_blocks=leased.size();
    result.shared_reference_savings=references-pinned.size();
    for(const auto& item:reservations_){
        result.logical_reserved_blocks+=item.second;
        const auto found=sequences_.find(item.first);
        const auto pending=admission_prefixes_.find(item.first);
        const auto present=found!=sequences_.end()&&found->second.sequence.used?found->second.sequence.blocks.size():
            (pending==admission_prefixes_.end()?0:pending->second.blocks.size());
        if(item.second>present)result.future_private_blocks+=item.second-present;
    }
    return result;
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
