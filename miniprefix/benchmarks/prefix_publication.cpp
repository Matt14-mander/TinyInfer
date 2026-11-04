// Metadata-only publication cost; no model, payload copy, or IPC in this timer.
#include "miniprefix/prefix_cache.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
using namespace miniprefix;
namespace {std::size_t allocations=0;}
void* operator new(std::size_t size){
    if(auto* p=std::malloc(size?size:1)){++allocations;return p;}throw std::bad_alloc();
}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete(void* p,std::size_t) noexcept{std::free(p);}
void* operator new[](std::size_t size){return ::operator new(size);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t) noexcept{std::free(p);}
void check(bool value){if(!value)throw std::runtime_error("publication benchmark invariant failed");}
int main(){try{
    constexpr int iterations=10000;
    for(const std::string mode:{"repeat","snapshots","branches"}){
        KvBlockPool pool({16,4096,64},1,1);ExactPrefixCache cache(pool);
        std::vector<PagedSequence> sources(mode=="repeat"?1:96);
        for(std::size_t s=0;s<sources.size();++s){
            auto& seq=sources[s];seq.used=512+(mode=="repeat"?0:1+int(s%15));
            pool.reserve(seq,seq.used);seq.tokens.assign(seq.used,7);
            if(mode=="branches")seq.tokens[0]=int(s);
            if(mode!="repeat")seq.tokens.back()=100+int(s);
        }
        const auto active=pool.live_blocks();
        for(int repeat=0;repeat<11;++repeat){
            const auto before=allocations;
            const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<iterations;++i)cache.publish(sources[i%sources.size()]);
            const auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
            const auto count=allocations-before;
            check(pool.live_blocks()==active&&cache.size()<=64);
            if(repeat)std::cout<<"{\"case\":\""<<mode<<"\",\"repeat\":"<<repeat
                <<",\"publications\":"<<iterations<<",\"ns_per_publish\":"<<ns/iterations
                <<",\"allocations_per_publish\":"<<double(count)/iterations<<"}\n";
        }
        cache.clear();for(auto& seq:sources)pool.release(seq);check(pool.live_blocks()==0);
    }
    std::cout<<"{\"passed\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
