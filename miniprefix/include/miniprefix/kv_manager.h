#pragma once
#include "request.h"
#include "prefix_cache.h"
#include <map>

namespace miniprefix {
struct KvStats {
    std::size_t live_sequences=0,live_kv_bytes=0,peak_kv_bytes=0,forward_tokens=0;
    std::size_t kv_block_tokens=0,kv_total_blocks=0,kv_live_blocks=0,kv_peak_blocks=0;
    std::size_t kv_reserved_bytes=0;
    std::size_t prefix_cache_capacity=0,kv_cached_blocks=0,kv_cache_only_blocks=0;
    std::size_t prefix_hits=0,prefix_misses=0,prefix_reused_tokens=0,prefix_evictions=0;
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
    State& idle_state(const RequestId& id);
    void validate_tokens(const std::vector<TokenId>& tokens) const;
    void reserve_pages(PagedSequence& sequence,int end);
public:
    // The transaction owns an immutable input snapshot. Backend code MUST finish
    // all device accesses before rollback/destruction. It must not outlive its
    // manager; release/reset/clear reject pending transactions.
    class Append {
        friend class KVManager;
        KVManager* owner_;
        State* state_;
        std::vector<TokenId> tokens_;
        std::size_t original_blocks_;
        Append(KVManager& owner,const RequestId& id,const std::vector<TokenId>& tokens);
    public:
        Append(const Append&)=delete;
        Append& operator=(const Append&)=delete;
        Append(Append&&)=delete;
        Append& operator=(Append&&)=delete;
        ~Append();
        const PagedSequence& sequence() const;
        std::size_t original_blocks() const;
        void commit();
        void rollback() noexcept;
    };

    KVManager(KvCacheConfig config,int layers,int width,int context,int vocabulary);
    KVManager(const KVManager&)=delete;
    KVManager& operator=(const KVManager&)=delete;
    ~KVManager();
    const KvCacheConfig& config() const;
    std::size_t reserved_bytes() const;
    std::size_t max_sequence_blocks() const;
    bool contains(const RequestId& id) const;
    const PagedSequence& sequence(const RequestId& id) const;
    std::size_t offset(const PagedSequence& sequence,int layer,int token) const;
    void create(const RequestId& id);
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>& prompt);
    void cache_prefix(const RequestId& id);
    void clear_prefix_cache();
    Append prepare_append(const RequestId& id,const std::vector<TokenId>& tokens);
    void reset_sequence(const RequestId& id);
    void release(const RequestId& id);
    void clear();
    KvStats stats() const;
};
} // namespace miniprefix
