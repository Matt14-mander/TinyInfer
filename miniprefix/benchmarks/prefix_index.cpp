// Metadata-only index scaling; excludes model execution, GPU payload and IPC.
#include "miniprefix/prefix_cache.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
using namespace miniprefix;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::micro>(Clock::now()-t).count();}
void check(bool ok){if(!ok)throw std::runtime_error("prefix index invariant failed");}
int main(){try{
    for(int pages:{16,64,256,1024}){
        KvBlockPool pool({4,pages,pages},1,1);ExactPrefixCache cache(pool);PagedSequence source;
        pool.reserve(source,pages*4);source.used=pages*4;
        for(int i=0;i<source.used;++i)source.tokens.push_back(1+i%100);
        auto prompt=source.tokens;prompt.push_back(1);auto miss=prompt;miss[0]=999;
        std::vector<double> hits,misses,publishes,evictions;
        std::size_t key_bytes=0;
        for(int repeat=0;repeat<11;++repeat){
            auto t=Clock::now();cache.publish(source);const double publish=elapsed(t);key_bytes=cache.key_token_bytes();
            PagedSequence restored;t=Clock::now();const auto reused=cache.reuse(restored,prompt);const double hit=elapsed(t);
            check(reused==pages*4);pool.release(restored);
            PagedSequence absent;t=Clock::now();const auto rejected=cache.reuse(absent,miss);const double nohit=elapsed(t);
            check(rejected==0);pool.release(absent);
            t=Clock::now();int victims=0;while(cache.evict_one())++victims;const double eviction=elapsed(t);
            check(victims==pages&&pool.live_blocks()==std::size_t(pages));
            // First repeat warms allocation and code; preserve ten measured repeats.
            if(repeat){publishes.push_back(publish);hits.push_back(hit);misses.push_back(nohit);evictions.push_back(eviction);}
        }
        pool.release(source);check(pool.live_blocks()==0);
        std::cout<<"{\"pages\":"<<pages<<",\"block_tokens\":4,\"stored_key_token_bytes\":"<<key_bytes;
        for(auto item:{std::make_pair("publish_us",publishes),std::make_pair("full_hit_us",hits),std::make_pair("root_miss_us",misses),std::make_pair("evict_all_us",evictions)}){
            std::cout<<",\""<<item.first<<"\":[";for(std::size_t i=0;i<item.second.size();++i){if(i)std::cout<<',';std::cout<<item.second[i];}std::cout<<']';}
        std::cout<<"}"<<std::endl;
    }
    std::cout<<"{\"passed\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}

