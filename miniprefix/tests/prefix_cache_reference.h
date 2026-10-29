// Frozen pre-trie implementation for differential regression, not production code.
#pragma once
#include "miniprefix/paged_kv.h"
#include <map>

namespace miniprefix {
// Exact keys contain the ENTIRE token history through the end of a page, not
// just that page's tokens. Model/weights/RoPE/dtype are fixed by the owning
// runner; this index never crosses runner instances or position origins.
class LegacyPrefixCache {
    struct Entry {int block;std::uint64_t touched;};
    KvBlockPool& pool_;
    std::map<std::vector<std::int32_t>,Entry> entries_;
    std::uint64_t clock_=0;
    std::size_t hits_=0,misses_=0,reused_=0,evictions_=0;
public:
    explicit LegacyPrefixCache(KvBlockPool& pool);
    LegacyPrefixCache(const LegacyPrefixCache&)=delete;
    LegacyPrefixCache& operator=(const LegacyPrefixCache&)=delete;
    ~LegacyPrefixCache();
    std::size_t size() const;
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


#include <algorithm>
#include <limits>
#include <utility>

namespace miniprefix {
bool LegacyPrefixCache::evict_one(){
    if(entries_.empty())return false;
    auto victim=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){
        return a.second.touched<b.second.touched;});
    pool_.unref(victim->second.block);entries_.erase(victim);++evictions_;return true;
}
LegacyPrefixCache::LegacyPrefixCache(KvBlockPool& pool):pool_(pool){}
LegacyPrefixCache::~LegacyPrefixCache(){clear();}
std::size_t LegacyPrefixCache::size() const{return entries_.size();}
std::size_t LegacyPrefixCache::hits() const{return hits_;}
std::size_t LegacyPrefixCache::misses() const{return misses_;}
std::size_t LegacyPrefixCache::reused_tokens() const{return reused_;}
std::size_t LegacyPrefixCache::evictions() const{return evictions_;}
std::size_t LegacyPrefixCache::cache_only_blocks() const{
    std::size_t count=0;for(const auto& e:entries_)if(pool_.references(e.second.block)==1)++count;return count;
}
void LegacyPrefixCache::clear() noexcept{for(const auto& e:entries_)pool_.unref(e.second.block);entries_.clear();}
int LegacyPrefixCache::reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt){
    if(sequence.used||!sequence.blocks.empty()||!sequence.tokens.empty())
        throw std::invalid_argument("prefix reuse requires an empty sequence");
    if(!pool_.config().prefix_cache_blocks)return 0;
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto limit=prompt.empty()?0:(prompt.size()-1)/block_tokens;
    std::vector<int> pages;
    for(std::size_t i=0;i<limit;++i){
        auto end=prompt.begin()+(i+1)*block_tokens;
        auto found=entries_.find(std::vector<std::int32_t>(prompt.begin(),end));
        if(found==entries_.end())break;
        pages.push_back(found->second.block);found->second.touched=++clock_;
    }
    const auto count=pages.size()*block_tokens;
    std::vector<std::int32_t> tokens(prompt.begin(),prompt.begin()+count);
    for(int page:pages)pool_.retain(page);
    sequence.blocks=std::move(pages);sequence.tokens=std::move(tokens);sequence.used=static_cast<int>(count);
    if(count){++hits_;reused_+=count;}else ++misses_;
    return sequence.used;
}
void LegacyPrefixCache::publish(const PagedSequence& sequence){
    if(sequence.tokens.size()!=std::size_t(sequence.used))throw std::logic_error("invalid committed KV token history");
    const auto count=std::min(sequence.blocks.size(),std::min(std::size_t(sequence.used/pool_.config().block_tokens),
                             std::size_t(pool_.config().prefix_cache_blocks)));
    for(std::size_t i=0;i<count;++i){
        std::vector<std::int32_t> key(sequence.tokens.begin(),sequence.tokens.begin()+(i+1)*pool_.config().block_tokens);
        auto found=entries_.find(key);
        if(found!=entries_.end()){found->second.touched=++clock_;continue;}
        if(entries_.size()==std::size_t(pool_.config().prefix_cache_blocks))evict_one();
        const int page=sequence.blocks[i];pool_.retain(page);
        try{entries_.emplace(std::move(key),Entry{page,++clock_});}
        catch(...){pool_.unref(page);throw;}
    }
}
} // namespace miniprefix
