#include "miniprefix/paged_kv.h"
#include <cassert>
#include <iostream>
#include <set>

using namespace miniprefix;
int main(){
    KvBlockPool pool({3,5},2,4);PagedSequence a,b,c;
    assert(pool.reserved_bytes()==5*3*2*4*8);
    pool.reserve(a,3);a.used=3;pool.reserve(b,3);b.used=3;
    pool.reserve(a,7);a.used=7;
    assert((a.blocks==std::vector<int>{0,2,3})); // non-contiguous physical layout
    assert(pool.offset(a,1,3,2,4)==(2*2+1)*3*4);
    assert(pool.live_blocks()==4&&pool.peak_blocks()==4);
    const auto before=a.blocks;
    bool exhausted=false;try{pool.reserve(a,13);}catch(const std::runtime_error&){exhausted=true;}
    assert(exhausted&&a.blocks==before&&a.used==7&&pool.live_blocks()==4);
    pool.release(b);pool.reserve(c,6);c.used=6;
    assert((c.blocks==std::vector<int>{1,4})); // reuse a hole and remaining page
    std::set<int> unique(a.blocks.begin(),a.blocks.end());unique.insert(c.blocks.begin(),c.blocks.end());
    assert(unique.size()==5&&pool.live_blocks()==5);
    pool.release(c);const auto original=a.blocks.size();pool.reserve(a,10);
    pool.rollback(a,original);assert(a.blocks==before&&a.used==7);
    pool.release(a);assert(pool.live_blocks()==0&&pool.peak_blocks()==5);
    pool.reserve(b,15);assert(b.blocks.size()==5);pool.release(b);
    bool invalid=false;try{KvBlockPool bad({0,1},2,4);}catch(const std::invalid_argument&){invalid=true;}
    assert(invalid);invalid=false;
    try{KvBlockPool bad({2147483647,2147483647},2147483647,2147483647);}catch(const std::invalid_argument&){invalid=true;}
    assert(invalid);
    std::cout<<"paged KV ownership, addressing, rollback and reuse passed\n";
}
