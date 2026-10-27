#pragma once
#include "paged_kv.h"
#include <map>

namespace miniprefix {
// Exact keys contain the ENTIRE token history through the end of a page, not
// just that page's tokens. Model/weights/RoPE/dtype are fixed by the owning
// runner; this index never crosses runner instances or position origins.
class ExactPrefixCache {
    struct Entry {int block;std::uint64_t touched;};
    KvBlockPool& pool_;
    std::map<std::vector<std::int32_t>,Entry> entries_;
    std::uint64_t clock_=0;
    std::size_t hits_=0,misses_=0,reused_=0,evictions_=0;
    void evict_one();
public:
    explicit ExactPrefixCache(KvBlockPool& pool);
    ExactPrefixCache(const ExactPrefixCache&)=delete;
    ExactPrefixCache& operator=(const ExactPrefixCache&)=delete;
    ~ExactPrefixCache();
    std::size_t size() const;
    std::size_t hits() const;
    std::size_t misses() const;
    std::size_t reused_tokens() const;
    std::size_t evictions() const;
    std::size_t cache_only_blocks() const;
    void clear() noexcept;
    int reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt);
    void publish(const PagedSequence& sequence);
    void reserve(PagedSequence& sequence,int end);
};
} // namespace miniprefix
