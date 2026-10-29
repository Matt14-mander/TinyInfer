// Independent flat full-history oracle for partial-page differential regression.
#pragma once
#include "miniprefix/paged_kv.h"
#include <map>

namespace miniprefix {
// Exact keys contain the ENTIRE token history through the end of a page, not
// just that page's tokens. Model/weights/RoPE/dtype are fixed by the owning
// runner; this index never crosses runner instances or position origins.
class PartialPrefixReference {
    struct Entry {int block;std::uint64_t touched;};
    KvBlockPool& pool_;
    std::map<std::vector<std::int32_t>,Entry> entries_;
    std::uint64_t clock_=0;
    std::size_t hits_=0,misses_=0,reused_=0,evictions_=0;
public:
    explicit PartialPrefixReference(KvBlockPool& pool);
    PartialPrefixReference(const PartialPrefixReference&)=delete;
    PartialPrefixReference& operator=(const PartialPrefixReference&)=delete;
    ~PartialPrefixReference();
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
bool PartialPrefixReference::evict_one(){
    if(entries_.empty())return false;
    auto victim=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){
        return a.second.touched<b.second.touched;});
    pool_.unref(victim->second.block);entries_.erase(victim);++evictions_;return true;
}
PartialPrefixReference::PartialPrefixReference(KvBlockPool& pool):pool_(pool){}
PartialPrefixReference::~PartialPrefixReference(){clear();}
std::size_t PartialPrefixReference::size() const{return entries_.size();}
std::size_t PartialPrefixReference::hits() const{return hits_;}
std::size_t PartialPrefixReference::misses() const{return misses_;}
std::size_t PartialPrefixReference::reused_tokens() const{return reused_;}
std::size_t PartialPrefixReference::evictions() const{return evictions_;}
std::size_t PartialPrefixReference::cache_only_blocks() const{
    std::map<int,std::size_t> leases;for(const auto& e:entries_)++leases[e.second.block];
    std::size_t count=0;for(const auto& e:leases)if(pool_.references(e.first)==e.second)++count;return count;
}
void PartialPrefixReference::clear() noexcept{for(const auto& e:entries_)pool_.unref(e.second.block);entries_.clear();}
int PartialPrefixReference::reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt){
    if(sequence.used||!sequence.blocks.empty()||!sequence.tokens.empty())
        throw std::invalid_argument("prefix reuse requires an empty sequence");
    if(!pool_.config().prefix_cache_blocks)return 0;
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto limit=prompt.empty()?0:prompt.size()-1;
    std::vector<int> pages;std::size_t count=0;
    while(count<limit){
        const auto length=std::min(block_tokens,limit-count);
        std::vector<std::int32_t> query(prompt.begin(),prompt.begin()+count+length);
        auto best=entries_.end();std::size_t matched=0;
        for(auto it=entries_.begin();it!=entries_.end();++it){
            const auto& key=it->first;
            if(key.size()<=count||key.size()>count+block_tokens||!std::equal(key.begin(),key.begin()+count,prompt.begin()))continue;
            std::size_t n=0;while(n<length&&count+n<key.size()&&key[count+n]==prompt[count+n])++n;
            bool tie=false;
            if(n&&n==matched){
                const bool right=!(key<query),best_right=!(best->first<query);
                tie=right!=best_right?right:(right?key<best->first:best->first<key);
            }
            if(n>matched||tie){matched=n;best=it;}
        }
        if(!matched)break;
        pages.push_back(best->second.block);best->second.touched=++clock_;count+=matched;
        if(matched!=block_tokens)break;
    }
    std::vector<std::int32_t> tokens(prompt.begin(),prompt.begin()+count);
    for(int page:pages)pool_.retain(page);
    sequence.blocks=std::move(pages);sequence.tokens=std::move(tokens);sequence.used=static_cast<int>(count);
    if(count){++hits_;reused_+=count;}else ++misses_;
    return sequence.used;
}
void PartialPrefixReference::publish(const PagedSequence& sequence){
    if(sequence.tokens.size()!=std::size_t(sequence.used))throw std::logic_error("invalid committed KV token history");
    const auto count=std::min(sequence.blocks.size(),std::min(std::size_t((sequence.used+pool_.config().block_tokens-1)/pool_.config().block_tokens),
                             std::size_t(pool_.config().prefix_cache_blocks)));
    for(std::size_t i=0;i<count;++i){
        std::vector<std::int32_t> key(sequence.tokens.begin(),sequence.tokens.begin()+std::min(std::size_t(sequence.used),(i+1)*pool_.config().block_tokens));
        auto found=entries_.find(key);
        if(found!=entries_.end()){found->second.touched=++clock_;continue;}
        if(entries_.size()==std::size_t(pool_.config().prefix_cache_blocks))evict_one();
        const int page=sequence.blocks[i];pool_.retain(page);
        try{entries_.emplace(std::move(key),Entry{page,++clock_});}
        catch(...){pool_.unref(page);throw;}
    }
}
} // namespace miniprefix
