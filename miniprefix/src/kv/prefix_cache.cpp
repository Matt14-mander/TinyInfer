#include "miniprefix/prefix_cache.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace miniprefix {
bool ExactPrefixCache::evict_one(){
    if(entries_.empty())return false;
    auto victim=std::min_element(entries_.begin(),entries_.end(),[](const auto& a,const auto& b){
        return a.second.touched<b.second.touched;});
    pool_.unref(victim->second.block);entries_.erase(victim);++evictions_;return true;
}
ExactPrefixCache::ExactPrefixCache(KvBlockPool& pool):pool_(pool){}
ExactPrefixCache::~ExactPrefixCache(){clear();}
std::size_t ExactPrefixCache::size() const{return entries_.size();}
std::size_t ExactPrefixCache::hits() const{return hits_;}
std::size_t ExactPrefixCache::misses() const{return misses_;}
std::size_t ExactPrefixCache::reused_tokens() const{return reused_;}
std::size_t ExactPrefixCache::evictions() const{return evictions_;}
std::size_t ExactPrefixCache::cache_only_blocks() const{
    std::size_t count=0;for(const auto& e:entries_)if(pool_.references(e.second.block)==1)++count;return count;
}
void ExactPrefixCache::clear() noexcept{for(const auto& e:entries_)pool_.unref(e.second.block);entries_.clear();}
int ExactPrefixCache::reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt){
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
void ExactPrefixCache::publish(const PagedSequence& sequence){
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
