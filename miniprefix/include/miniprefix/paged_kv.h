#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace miniprefix {
struct KvCacheConfig {
    int block_tokens=16;
    int max_blocks=128;
    int prefix_cache_blocks=0;
};

// Only complete immutable pages may be shared; a partial tail stays private.
struct PagedSequence {
    int used=0;
    std::vector<int> blocks;
    std::vector<std::int32_t> tokens;
};

// Backend-independent ownership. Storage is [physical block, layer, slot, KV].
// Reserve is atomic on capacity/host allocation failure. Callers synchronize
// outstanding device accesses before rolling back or returning any pages.
class KvBlockPool {
    KvCacheConfig config_;
    std::size_t block_elements_,live_=0,peak_=0;
    std::vector<int> free_;
    std::vector<std::size_t> references_;
public:
    KvBlockPool(KvCacheConfig config,int layers,int width):config_(config) {
        if(config.block_tokens<1||config.max_blocks<1||layers<1||width<1||
           config.prefix_cache_blocks<0||config.prefix_cache_blocks>config.max_blocks)
            throw std::invalid_argument("invalid paged KV configuration");
        const auto limit=std::numeric_limits<std::size_t>::max()/sizeof(float)/2;
        std::size_t size=static_cast<std::size_t>(layers);
        for(int factor:{config.block_tokens,width,config.max_blocks}) {
            if(size>limit/static_cast<std::size_t>(factor))
                throw std::invalid_argument("paged KV size overflow");
            size*=static_cast<std::size_t>(factor);
        }
        block_elements_=size/config.max_blocks;
        references_.resize(config.max_blocks);
        free_.reserve(config.max_blocks);
        for(int i=config.max_blocks-1;i>=0;--i)free_.push_back(i);
    }
    const KvCacheConfig& config() const{return config_;}
    std::size_t block_elements() const{return block_elements_;}
    std::size_t block_bytes() const{return block_elements_*sizeof(float)*2;}
    std::size_t reserved_bytes() const{return block_bytes()*config_.max_blocks;}
    std::size_t live_blocks() const{return live_;}
    std::size_t peak_blocks() const{return peak_;}
    std::size_t free_blocks() const{return free_.size();}
    std::size_t references(int block) const{return references_.at(block);}
    void retain(int block){++references_.at(block);}
    void unref(int block) noexcept {
        if(--references_[block]==0){free_.push_back(block);--live_;}
    }
    void reserve(PagedSequence& sequence,int end) {
        if(end<sequence.used)throw std::invalid_argument("paged KV cannot truncate through reserve");
        const auto count=static_cast<std::size_t>(end/config_.block_tokens+(end%config_.block_tokens!=0));
        if(count<=sequence.blocks.size())return;
        if(count-sequence.blocks.size()>free_.size())throw std::runtime_error("paged KV block pool exhausted");
        sequence.blocks.reserve(count);
        while(sequence.blocks.size()<count){const int block=free_.back();sequence.blocks.push_back(block);
            free_.pop_back();references_[block]=1;++live_;}
        peak_=std::max(peak_,live_);
    }
    void rollback(PagedSequence& sequence,std::size_t original_blocks) noexcept {
        while(sequence.blocks.size()>original_blocks){unref(sequence.blocks.back());sequence.blocks.pop_back();}
    }
    void release(PagedSequence& sequence) noexcept {rollback(sequence,0);sequence.used=0;sequence.tokens.clear();}
    std::size_t offset(const PagedSequence& sequence,int layer,int token,int layers,int width) const {
        return ((static_cast<std::size_t>(sequence.blocks.at(token/config_.block_tokens))*layers+layer)
                *config_.block_tokens+token%config_.block_tokens)*width;
    }
};
} // namespace miniprefix
