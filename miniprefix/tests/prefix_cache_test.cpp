#include "miniprefix/prefix_cache.h"
#include <cassert>
#include <iostream>
using namespace miniprefix;
namespace {
void append(KvBlockPool& pool,PagedSequence& sequence,std::vector<std::int32_t> tokens){
    pool.reserve(sequence,sequence.used+static_cast<int>(tokens.size()));
    sequence.tokens.insert(sequence.tokens.end(),tokens.begin(),tokens.end());sequence.used+=static_cast<int>(tokens.size());
}
}
int main(){
    {
        KvBlockPool pool({2,8,3},2,4);ExactPrefixCache cache(pool);PagedSequence source,a,b;
        append(pool,source,{1,2,3,4,5});cache.publish(source);
        assert(cache.size()==2&&pool.live_blocks()==3); // tail is not published
        const int shared=source.blocks[0];assert(pool.references(shared)==2);
        assert(cache.reuse(a,{1,2,3,4,5})==4&&pool.references(shared)==3);
        append(pool,a,{9,10});assert(a.blocks.back()!=source.blocks.back());
        assert(cache.reuse(b,{9,9,3,4,5})==0); // equal later block is not a prefix
        pool.release(b);
        assert(cache.reuse(b,{1,2,3,4})==2); // aligned full hit recomputes last block
        cache.clear();assert(cache.size()==0&&pool.references(shared)==3);
        pool.release(a);pool.release(b);assert(pool.references(shared)==1);
        pool.release(source);assert(pool.live_blocks()==0);
    }
    {
        KvBlockPool pool({2,6,2},1,1);ExactPrefixCache cache(pool);PagedSequence a,b,c;
        append(pool,a,{1,2,3,4});cache.publish(a);pool.release(a);
        assert(cache.reuse(b,{1,2,9})==2);pool.release(b); // refresh first page
        append(pool,c,{7,8});cache.publish(c);pool.release(c);
        assert(cache.evictions()==1&&cache.size()==2);
        assert(cache.reuse(a,{1,2,3,4,5})==2); // LRU evicted second page, not first
        pool.release(a);cache.clear();assert(pool.live_blocks()==0);
    }
    {
        KvBlockPool pool({2,3,2},1,1);ExactPrefixCache cache(pool);PagedSequence source,active,next;
        append(pool,source,{1,2,3,4});cache.publish(source);pool.release(source);
        assert(cache.cache_only_blocks()==2&&cache.reuse(active,{1,2,3,4,5})==4);
        append(pool,active,{5});assert(pool.live_blocks()==3);
        const auto first=active.blocks[0];
        assert(cache.evict_one()&&pool.references(first)==1);
        assert(pool.live_blocks()==3&&cache.evictions()==1); // active lease pins the page
        assert(cache.evict_one()&&cache.size()==0&&cache.evictions()==2);
        assert(!cache.evict_one()&&cache.evictions()==2); // empty eviction is a no-op
        assert(active.used==5&&active.tokens.size()==5);
        assert(pool.references(active.blocks[0])==1); // evicted cache, active lease remains
        pool.release(active);append(pool,next,{7});pool.release(next);assert(pool.live_blocks()==0);
    }
    {
        KvBlockPool pool({2,4,0},1,1);ExactPrefixCache cache(pool);PagedSequence a,b;
        append(pool,a,{1,2,3});cache.publish(a);assert(cache.size()==0&&cache.reuse(b,{1,2,3})==0);
        pool.release(a);assert(pool.live_blocks()==0);
    }
    std::cout<<"exact prefix keys, immutable sharing, aligned logits boundary, LRU and active leases passed\n";
}
