#include "miniprefix/prefix_cache.h"
#include <cassert>
#include <iostream>
#include <random>
#include "prefix_cache_reference.h"
using namespace miniprefix;
namespace {
void append(KvBlockPool& pool,PagedSequence& sequence,std::vector<std::int32_t> tokens){
    pool.reserve(sequence,sequence.used+static_cast<int>(tokens.size()));
    sequence.tokens.insert(sequence.tokens.end(),tokens.begin(),tokens.end());sequence.used+=static_cast<int>(tokens.size());
}
void deep_path(){
    constexpr int depth=8192;
    KvBlockPool pool({1,depth,depth},1,1);ExactPrefixCache cache(pool);PagedSequence source;
    append(pool,source,std::vector<std::int32_t>(depth,1));cache.publish(source);pool.release(source);
    assert(cache.index_nodes()==depth);
    for(int i=0;i<depth;++i)assert(cache.evict_one());
    assert(cache.index_nodes()==0&&pool.live_blocks()==0);
    append(pool,source,std::vector<std::int32_t>(depth,2));cache.publish(source);pool.release(source);
    cache.clear();assert(cache.index_nodes()==0&&pool.live_blocks()==0);
}
void differential(){
    for(int capacity:{0,1,2,5,16}){
        KvBlockPool left({2,128,capacity},1,1),right({2,128,capacity},1,1);
        ExactPrefixCache actual(left);LegacyPrefixCache expected(right);
        std::mt19937 random(7341+capacity);
        for(int step=0;step<3000;++step){
            const auto action=random()%7;
            if(action==0){assert(actual.evict_one()==expected.evict_one());}
            else if(action==1){actual.clear();expected.clear();assert(actual.index_nodes()==0&&actual.key_token_bytes()==0);}
            else{
                std::vector<std::int32_t> tokens;
                const int length=1+random()%23;
                for(int i=0;i<length;++i)tokens.push_back((i<4?int(random()%2):int(random()%5)));
                PagedSequence a,b;
                if(action<4){append(left,a,tokens);append(right,b,tokens);actual.publish(a);expected.publish(b);}
                else{assert(actual.reuse(a,tokens)==expected.reuse(b,tokens));}
                assert(a.blocks==b.blocks&&a.tokens==b.tokens&&a.used==b.used);
                // Eviction with active leases must preserve payload ownership.
                if(step%3==0)assert(actual.evict_one()==expected.evict_one());
                for(int page=0;page<128;++page)assert(left.references(page)==right.references(page));
                left.release(a);right.release(b);
            }
            assert(actual.size()==expected.size()&&actual.hits()==expected.hits()&&actual.misses()==expected.misses());
            assert(actual.evictions()==expected.evictions()&&actual.reused_tokens()==expected.reused_tokens());
            assert(actual.cache_only_blocks()==expected.cache_only_blocks()&&left.live_blocks()==right.live_blocks());
            // Every retained structural node lies on a path to a cached descendant.
            assert(actual.index_nodes()<=actual.size()*std::size_t(capacity));
        }
        actual.clear();expected.clear();assert(left.live_blocks()==0&&right.live_blocks()==0&&actual.index_nodes()==0);
    }
}
void missing_ancestor_and_pruning(){
    {
        KvBlockPool pool({2,8,2},1,1);ExactPrefixCache cache(pool);PagedSequence a,b,probe;
        append(pool,a,{1,2,3,4});cache.publish(a);cache.evict_one();
        append(pool,b,{7,8});cache.publish(b); // LRU is the sole descendant of missing [1,2]
        cache.publish(a); // eviction must not prune the ancestor being republished
        assert(cache.reuse(probe,{1,2,3,4,5})==4);
        pool.release(probe);pool.release(a);pool.release(b);cache.clear();assert(pool.live_blocks()==0);
        PagedSequence invalid;invalid.tokens={1,2};invalid.used=2;invalid.blocks={99};
        bool rejected=false;try{cache.publish(invalid);}catch(const std::out_of_range&){rejected=true;}
        assert(rejected&&cache.size()==0&&cache.index_nodes()==0);
    }

    KvBlockPool pool({2,16,3},1,1);ExactPrefixCache cache(pool);PagedSequence source,probe;
    append(pool,source,{1,2,3,4,5,6});cache.publish(source);
    assert(cache.index_nodes()==3&&cache.key_token_bytes()==24);
    assert(cache.evict_one()); // root page gone, descendants remain cached
    assert(cache.size()==2&&cache.index_nodes()==3);
    assert(cache.reuse(probe,{1,2,3,4,5,6,7})==0);
    cache.publish(source); // revive ancestor without losing cached descendants
    assert(cache.reuse(probe,{1,2,3,4,5,6,7})==6);pool.release(probe);
    cache.clear();assert(cache.index_nodes()==0&&cache.key_token_bytes()==0);
    pool.release(source);assert(pool.live_blocks()==0);
    // Repeated root churn must not retain a history of evicted token paths.
    for(int i=0;i<1000;++i){append(pool,source,{i,2,3,4,5,6});cache.publish(source);pool.release(source);}
    assert(cache.index_nodes()<=cache.size()*3);cache.clear();assert(cache.index_nodes()==0&&pool.live_blocks()==0);
}

}
int main(){
    differential();missing_ancestor_and_pruning();deep_path();
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
