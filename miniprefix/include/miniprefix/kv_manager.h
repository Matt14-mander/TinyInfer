#pragma once
#include "request.h"
#include "cost_stats.h"
#include "prefix_cache.h"
#include <map>
#include <functional>
#include <optional>

namespace miniprefix {
struct DecodeInput {RequestId id;TokenId token;};
struct BatchInput {
    RequestId id;
    std::vector<TokenId> tokens;
    bool decode = false;
};
struct KvStats {
    std::size_t live_sequences=0,live_kv_bytes=0,peak_kv_bytes=0,forward_tokens=0;
    std::size_t kv_block_tokens=0,kv_total_blocks=0,kv_live_blocks=0,kv_peak_blocks=0;
    std::size_t kv_reserved_bytes=0;
    std::size_t prefix_cache_capacity=0,kv_cached_blocks=0,kv_cache_only_blocks=0;
    std::size_t prefix_hits=0,prefix_misses=0,prefix_reused_tokens=0,prefix_evictions=0;
};

// Idle-state accounting. committed_blocks is the admission promise, not physical
// usage; cow_tail_candidates counts holders. Other fields are page/reference counts.
struct KvCapacityLedger {
    std::size_t total_blocks=0,physical_blocks=0,free_blocks=0;
    std::size_t pinned_blocks=0,cache_only_blocks=0,shared_reference_savings=0;
    std::size_t logical_reserved_blocks=0,future_private_blocks=0,committed_blocks=0;
    std::size_t prefix_lease_blocks=0,cow_tail_candidates=0;
};

// Owns sequence metadata, page references and prefix leases. Payload, device
// tables, streams and completion stay in the backend. Single-threaded API.
class KVManager {
    struct State {PagedSequence sequence;bool appending=false;};
    int layers_,width_,context_,vocabulary_;
    KvBlockPool pool_;
    ExactPrefixCache prefix_;
    std::map<RequestId,State> sequences_;
    std::size_t forward_tokens_=0;
    std::size_t append_active_=0;
    bool cost_timing_=false;CostStats costs_;
    void require_idle() const;
    std::map<RequestId,std::size_t> reservations_;
    std::map<RequestId,PagedSequence> prefix_leases_;
    // Presence opts a reservation into sharing; pending full-page prefix owns leases.
    std::map<RequestId,PagedSequence> admission_prefixes_;
    std::size_t committed_blocks() const;
    State& idle_state(const RequestId& id);
    void validate_tokens(const std::vector<TokenId>& tokens) const;
    void reserve_pages(const RequestId& id,PagedSequence& sequence,int end,bool& cow);
public:
    // The transaction owns an immutable input snapshot. Backend code MUST finish
    // all device accesses before rollback/destruction. It must not outlive its
    // manager; release/reset/clear reject pending transactions.
    struct TailCopy {int source,destination,valid_tokens;};
    // Float-element offsets/pitch/width; rows are layers in a single K or V pool.
    struct TailCopyRegion {std::size_t destination,source,pitch,width,rows;};
    class Append {
        friend class KVManager;
        KVManager* owner_;
        State* state_;
        std::vector<TokenId> tokens_;
        std::size_t original_blocks_;
        std::optional<TailCopy> copy_;
        bool copied_=false;
        Append(KVManager& owner,const RequestId& id,const std::vector<TokenId>& tokens,bool batch=false);
    public:
        Append(const Append&)=delete;
        Append& operator=(const Append&)=delete;
        Append(Append&&)=delete;
        Append& operator=(Append&&)=delete;
        ~Append();
        const PagedSequence& sequence() const;
        std::size_t original_blocks() const;
        const std::optional<TailCopy>& tail_copy() const;
        // Element offsets in one K or V pool; copier must copy BOTH payloads.
        // GPU enqueue is allowed, but completion is mandatory before commit/rollback.
        void copy_shared_tail(const std::function<void(std::size_t,std::size_t,std::size_t)>& copier);
        // One callback for the strided region; copy BOTH K and V. Throwing
        // leaves commit disabled. Device completion rules are unchanged.
        void copy_shared_tail_region(const std::function<void(const TailCopyRegion&)>& copier);
        void commit();
        void rollback() noexcept;
    };

    KVManager(KvCacheConfig config,int layers,int width,int context,int vocabulary);
    KVManager(const KVManager&)=delete;
    KVManager& operator=(const KVManager&)=delete;
    ~KVManager();
    const KvCacheConfig& config() const;
    int context_limit() const;
    std::size_t reserved_bytes() const;
    std::size_t max_sequence_blocks() const;
    bool contains(const RequestId& id) const;
    const PagedSequence& sequence(const RequestId& id) const;
    std::size_t offset(const PagedSequence& sequence,int layer,int token) const;
    // Logical worst-case page reservation, without allocating payload or evicting cache.
    KvAdmission reserve_request(const RequestId& id,std::size_t max_tokens);
    // Prompt-aware reservation pins full pages and credits their physical sharing.
    KvAdmission reserve_request(const RequestId& id,std::size_t max_tokens,const std::vector<TokenId>& prompt);
    void release_reservation(const RequestId& id) noexcept;
    void create(const RequestId& id);
    void fork_sequence(const RequestId& source,const RequestId& target);
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt);
    void cache_prefix(const RequestId& id);
    void clear_prefix_cache();
    Append prepare_append(const RequestId& id,const std::vector<TokenId>& tokens);
    // Plan the complete group's growth/COW peak before evicting oldest cache
    // references. Insufficient plans return empty without eviction. Applied LRU
    // evictions survive append rollback; publication stays with the caller.
    // All device accesses must finish before any commit/rollback.
    std::vector<std::unique_ptr<Append>> prepare_decode_batch(const std::vector<DecodeInput>& inputs);
    std::vector<std::unique_ptr<Append>> prepare_batch(const std::vector<BatchInput>& inputs);
    // Shared reservations keep full pages as pending leases until release/restore.
    void reset_sequence(const RequestId& id);
    void release(const RequestId& id);
    void clear();
    // Manager-owned leases are independent of sequence/reservation IDs. Acquisition
    // is conservative and does not grant admission credit. clear() releases all.
    std::size_t probe_prefix(const std::vector<TokenId>& prompt);
    std::size_t acquire_prefix_lease(const RequestId& lease_id,const std::vector<TokenId>& prompt);
    void release_prefix_lease(const RequestId& lease_id);
    KvCapacityLedger capacity_ledger() const;
    KvStats stats() const;
    void set_cost_timing(bool enabled){cost_timing_=enabled;}
    CostStats cost_stats() const{return costs_;}
};
} // namespace miniprefix
