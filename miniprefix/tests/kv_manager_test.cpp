#include "miniprefix/kv_manager.h"
#include <cassert>
#include <iostream>

using namespace miniprefix;
namespace {
template<class F> void rejects(F&& operation){
    bool rejected=false;try{operation();}catch(const std::exception&){rejected=true;}
    assert(rejected);
}
void commit(KVManager& manager,const RequestId& id,std::vector<TokenId> tokens){
    auto append=manager.prepare_append(id,tokens);append.commit();
}
void transactions(){
    KVManager manager({2,8,2},2,4,16,100);manager.create("a");
    std::vector<TokenId> input{1,2,3};
    {
        auto append=manager.prepare_append("a",input);
        assert(append.sequence().used==0&&append.sequence().blocks.size()==2);
        assert(manager.stats().forward_tokens==0);
        rejects([&]{manager.release("a");});rejects([&]{manager.reset_sequence("a");});
        rejects([&]{manager.clear();});rejects([&]{manager.clear_prefix_cache();});
        rejects([&]{manager.cache_prefix("a");});rejects([&]{manager.prepare_append("a",{4});});
        input[0]=99;append.commit();
        rejects([&]{append.commit();});
    }
    assert((manager.sequence("a").tokens==std::vector<TokenId>{1,2,3}));
    assert(manager.sequence("a").used==3&&manager.stats().forward_tokens==3);
    const auto blocks=manager.sequence("a").blocks;
    const auto live=manager.stats().kv_live_blocks;
    try{
        auto append=manager.prepare_append("a",{4,5});
        assert(append.original_blocks()==2&&append.sequence().blocks.size()==3);
        throw std::runtime_error("backend failure after reservation");
    }catch(const std::runtime_error&){}
    assert(manager.sequence("a").blocks==blocks&&manager.stats().kv_live_blocks==live);
    assert(manager.sequence("a").used==3&&manager.stats().forward_tokens==3);
    rejects([&]{manager.prepare_append("a",{});});rejects([&]{manager.prepare_append("a",{100});});
    rejects([&]{manager.prepare_append("a",std::vector<TokenId>(14,1));});
    commit(manager,"a",{4,5});assert(manager.stats().forward_tokens==5);
    manager.release("a");assert(manager.stats().kv_live_blocks==0);
    manager.clear();assert(manager.stats().forward_tokens==5); // cumulative counters stay intact
}
void prefix_and_reset(){
    KVManager manager({2,8,3},2,4,16,100);manager.create("source");
    commit(manager,"source",{1,2,3,4,5});manager.cache_prefix("source");manager.release("source");
    assert(manager.stats().kv_cache_only_blocks==2&&manager.stats().kv_live_blocks==2);
    manager.create("a");assert(manager.reuse_prefix("a",{1,2,3,4,6})==4);
    const auto first=manager.sequence("a").blocks[0];
    manager.create("b");assert(manager.reuse_prefix("b",{1,2,3,4,7})==4);
    assert(manager.sequence("b").blocks[0]==first&&manager.stats().kv_live_blocks==2);
    assert(manager.offset(manager.sequence("a"),1,2)==(std::size_t(manager.sequence("a").blocks[1])*2+1)*2*4);
    manager.reset_sequence("a");assert(manager.sequence("a").blocks.empty()&&manager.sequence("a").tokens.empty());
    assert(manager.reuse_prefix("a",{1,2,3,4,6})==4);
    manager.clear_prefix_cache();assert(manager.stats().kv_cached_blocks==0&&manager.stats().kv_live_blocks==2);
    commit(manager,"a",{6});assert(manager.sequence("b").used==4);
    manager.release("a");manager.release("b");assert(manager.stats().kv_live_blocks==0);
    assert(manager.stats().prefix_hits==3&&manager.stats().prefix_reused_tokens==12);
    assert(manager.stats().forward_tokens==6); // reuse is not forward
}
void exhaustion(){
    KVManager manager({2,3,2},1,1,16,100);manager.create("source");
    commit(manager,"source",{1,2,3,4});manager.cache_prefix("source");manager.release("source");
    manager.create("active");assert(manager.reuse_prefix("active",{1,2,3,4,5})==4);
    commit(manager,"active",{5});manager.create("next");
    rejects([&]{manager.prepare_append("next",{7});});
    assert(manager.sequence("next").blocks.empty()&&manager.sequence("active").used==5);
    assert(manager.stats().kv_cached_blocks==0&&manager.stats().kv_live_blocks==3);
    manager.release("active");commit(manager,"next",{7});
    assert(manager.stats().forward_tokens==6&&manager.stats().prefix_evictions==2);
    manager.clear();assert(manager.stats().live_sequences==0&&manager.stats().kv_live_blocks==0);
}
void capacity_and_lru(){
    KVManager manager({2,4,3},1,1,16,100);manager.create("source");
    commit(manager,"source",{1,2,3,4,5,6});manager.cache_prefix("source");manager.release("source");
    manager.create("touch");assert(manager.reuse_prefix("touch",{1,2,9})==2);manager.release("touch");
    manager.create("next");
    // One free page suffices; a tail append must not evict either.
    commit(manager,"next",{7});commit(manager,"next",{8});
    assert(manager.stats().prefix_evictions==0&&manager.stats().kv_cached_blocks==3);
    // Need one more page: evict only the oldest (second prefix page), then stop.
    commit(manager,"next",{9,10});
    assert(manager.stats().prefix_evictions==1&&manager.stats().kv_cached_blocks==2);
    assert(manager.stats().kv_live_blocks==4&&manager.stats().kv_peak_blocks==4);
    manager.create("probe");assert(manager.reuse_prefix("probe",{1,2,3,4,5,6,7})==2);
    assert(manager.stats().prefix_hits==2&&manager.stats().prefix_reused_tokens==4);
    manager.clear();assert(manager.stats().kv_live_blocks==0);
}
void pinned_victim_then_free_page(){
    KVManager manager({2,3,2},1,1,16,100);manager.create("source");
    commit(manager,"source",{1,2,3,4});manager.cache_prefix("source");
    manager.create("active");assert(manager.reuse_prefix("active",{1,2,9})==2);
    const auto pinned=manager.sequence("active").blocks;
    manager.cache_prefix("source"); // refresh in page order, leaving the pinned first page oldest
    manager.release("source");manager.create("occupant");commit(manager,"occupant",{7,8});
    manager.create("next");commit(manager,"next",{9});
    assert(manager.stats().prefix_evictions==2&&manager.stats().kv_cached_blocks==0);
    assert(manager.stats().kv_live_blocks==3&&manager.stats().forward_tokens==7);
    assert(manager.sequence("active").blocks==pinned&&manager.sequence("active").used==2);
    assert(manager.sequence("next").blocks[0]!=pinned[0]);
    manager.clear();assert(manager.stats().kv_live_blocks==0);
}
void rollback_keeps_eviction_effects(){
    KVManager manager({2,2,2},1,1,8,100);manager.create("source");
    commit(manager,"source",{1,2,3,4});manager.cache_prefix("source");manager.release("source");
    manager.create("next");
    // Validation happens before pressure eviction.
    rejects([&]{manager.prepare_append("next",{100});});
    rejects([&]{manager.prepare_append("next",std::vector<TokenId>(9,1));});
    assert(manager.stats().prefix_evictions==0&&manager.stats().kv_cached_blocks==2);
    {
        auto append=manager.prepare_append("next",{5});
        assert(manager.stats().prefix_evictions==1&&manager.stats().kv_live_blocks==2);
    }
    assert(manager.sequence("next").blocks.empty()&&manager.sequence("next").tokens.empty());
    assert(manager.stats().kv_cached_blocks==1&&manager.stats().kv_live_blocks==1&&manager.stats().forward_tokens==4);
    commit(manager,"next",{5});assert(manager.stats().prefix_evictions==1); // retry uses the freed page
    manager.clear();assert(manager.stats().kv_live_blocks==0);
}
void impossible_reservation_and_cache_off(){
    for(int cache:{0,2}){
        KVManager manager({2,2,cache},1,1,8,100);manager.create("source");
        commit(manager,"source",{1,2,3,4});manager.cache_prefix("source");manager.release("source");
        manager.create("next");
        bool exhausted=false;
        try{manager.prepare_append("next",{1,2,3,4,5});}
        catch(const std::runtime_error& e){exhausted=true;assert(std::string(e.what())=="paged KV block pool exhausted");}
        assert(exhausted&&manager.sequence("next").blocks.empty()&&manager.sequence("next").used==0);
        assert(manager.stats().kv_live_blocks==0&&manager.stats().kv_cached_blocks==0);
        assert(manager.stats().prefix_evictions==std::size_t(cache)&&manager.stats().forward_tokens==4);
        commit(manager,"next",{1,2,3,4});assert(manager.stats().forward_tokens==8);
        manager.clear();assert(manager.stats().kv_live_blocks==0);
    }
}
}
int main(){transactions();prefix_and_reset();exhaustion();capacity_and_lru();pinned_victim_then_free_page();
    rollback_keeps_eviction_effects();impossible_reservation_and_cache_off();
    std::cout<<"KVManager transactions, shared leases, reset, stats and exhaustion passed\n";}
