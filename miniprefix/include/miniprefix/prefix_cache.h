#pragma once
#include "paged_kv.h"
#include <memory>

namespace miniprefix {
// Exact page-token trie. Paths encode the entire causal history; no hashes or
// cross-runner keys. Structural ancestors may survive eviction without a KV lease.
class ExactPrefixCache {
    struct Index;
    KvBlockPool& pool_;
    std::unique_ptr<Index> index_;
    std::size_t hits_=0,misses_=0,reused_=0,evictions_=0;
public:
    explicit ExactPrefixCache(KvBlockPool& pool);
    ExactPrefixCache(const ExactPrefixCache&)=delete;
    ExactPrefixCache& operator=(const ExactPrefixCache&)=delete;
    ~ExactPrefixCache();
    std::size_t size() const;
    std::size_t index_nodes() const;
    std::size_t key_token_bytes() const;
    std::size_t hits() const;
    std::size_t misses() const;
    std::size_t reused_tokens() const;
    std::size_t evictions() const;
    std::size_t cache_only_blocks() const;
    void clear() noexcept;
    int reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt);
    void publish(const PagedSequence& sequence);
    // Drop one LRU cache reference, not necessarily a physical page: active
    // sequence references survive. False means no cache entry was available.
    bool evict_one();
};
} // namespace miniprefix
