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
    auto append=manager.prepare_append(id,tokens);append.copy_shared_tail([](std::size_t,std::size_t,std::size_t){});append.commit();
}
void batch_space_preparation(){
    KVManager m({4,5,4},1,1,24,100);
    m.create("a");commit(m,"a",{1,2,3});m.create("b");m.fork_sequence("a","b");m.cache_prefix("a");
    for(int token:{10,20,30}){m.create("seed");commit(m,"seed",{token,token,token,token});m.cache_prefix("seed");m.release("seed");}
    const auto original=m.sequence("a").blocks;const auto before=m.stats();
    assert(before.kv_live_blocks==4);
    // Validation of every row precedes LRU changes.
    assert(m.prepare_decode_batch({{"a",4},{"b",100}}).empty());
    assert(m.stats().prefix_evictions==before.prefix_evictions);
    {
        auto group=m.prepare_decode_batch({{"a",4},{"b",5}});assert(group.size()==2);
        assert(group[0]->tail_copy()&&group[1]->tail_copy());
        assert(m.stats().prefix_evictions==before.prefix_evictions+2); // pinned oldest, then oldest cache-only page
        assert(m.stats().kv_live_blocks==5&&m.stats().forward_tokens==before.forward_tokens);
        rejects([&]{m.cache_prefix("a");});
        rejects([&]{group[0]->copy_shared_tail([](std::size_t,std::size_t,std::size_t){throw std::runtime_error("copy");});});
        rejects([&]{group[0]->commit();});
    }
    assert(m.sequence("a").blocks==original&&m.sequence("b").blocks==original);
    assert(m.sequence("a").used==3&&m.stats().kv_live_blocks==3);
    assert(m.probe_prefix({10,10,10,10,1})==0&&m.probe_prefix({20,20,20,20,1})==4);
    assert(m.probe_prefix({30,30,30,30,1})==4); // unchanged relative LRU
    auto group=m.prepare_decode_batch({{"a",4},{"b",5}});assert(group.size()==2);
    for(auto& tx:group){tx->copy_shared_tail([](std::size_t,std::size_t,std::size_t){});tx->commit();}
    m.cache_prefix("b");m.cache_prefix("a"); // caller controls publication order
    m.create("next");commit(m,"next",{9,9,9,9,9});
    assert(m.probe_prefix({20,20,20,20,1})==0&&m.probe_prefix({30,30,30,30,1})==4);
    m.clear();assert(m.stats().kv_live_blocks==0);
    // Cache-only sharing can disappear without freeing a page: neither row needs COW then.
    KVManager inplace({4,2,2},1,1,16,100);
    for(const auto* id:{"a","b"}){inplace.create(id);commit(inplace,id,{id[0],1,2});inplace.cache_prefix(id);}
    auto in=inplace.prepare_decode_batch({{"a",3},{"b",4}});assert(in.size()==2);
    assert(!in[0]->tail_copy()&&!in[1]->tail_copy());
    for(auto& tx:in)tx->commit();inplace.clear();assert(inplace.stats().kv_live_blocks==0);
    // An admission lease survives planned eviction; an impossible batch must not drop any cache reference.
    KVManager pinned({4,2,2},1,1,16,100);
    pinned.create("seed");commit(pinned,"seed",{1,2,3,4});pinned.cache_prefix("seed");pinned.release("seed");
    assert(pinned.acquire_prefix_lease("pending",{1,2,3,4,5})==4);
    pinned.create("a");pinned.create("b");
    const auto cache=pinned.stats().kv_cached_blocks;
    assert(pinned.prepare_batch({{"a",{6}},{"b",{7}}}).empty());
    assert(pinned.stats().kv_cached_blocks==cache&&pinned.stats().prefix_evictions==0);
    pinned.clear();assert(pinned.stats().kv_live_blocks==0);
}
void batch_reference_accounting(){
    // Two cache entries can lease one page. Only the last eviction frees it.
    KVManager aliases({4,2,2},1,1,16,100);
    aliases.create("source");commit(aliases,"source",{1,2,3,4});aliases.cache_prefix("source");
    aliases.create("short");assert(aliases.reuse_prefix("short",{1,2,3,9})==3);aliases.cache_prefix("short");
    aliases.release("source");aliases.release("short");
    assert(aliases.stats().kv_cached_blocks==2&&aliases.stats().kv_live_blocks==1);
    aliases.create("a");aliases.create("b");
    auto group=aliases.prepare_batch({{"a",{8}},{"b",{9}}});assert(group.size()==2);
    assert(aliases.stats().prefix_evictions==2&&aliases.stats().kv_cached_blocks==0);
    for(auto& tx:group)tx->commit();aliases.clear();assert(aliases.stats().kv_live_blocks==0);
    // Releasing a resident reservation converts it to conservative diagnostic
    // accounting. Reject before pressure eviction even when that row only fills a tail.
    KVManager external({4,3,3},1,1,16,100);
    assert(external.reserve_request("a",8,{1})==KvAdmission::Ready);
    external.create("a");commit(external,"a",{1,2,3,4,5,6,7});external.cache_prefix("a");
    assert(external.reserve_request("b",8,{1,2,3,4,9})==KvAdmission::Ready);
    external.create("b");assert(external.reuse_prefix("b",{1,2,3,4,9})==4);
    external.release_reservation("a");const auto before=external.stats();
    assert(external.prepare_decode_batch({{"a",8},{"b",9}}).empty());
    assert(external.stats().prefix_evictions==before.prefix_evictions&&external.stats().kv_live_blocks==before.kv_live_blocks);
    external.clear();
}
void shared_admission(){
    KVManager m({2,5,5},1,1,20,100);
    m.create("seed");commit(m,"seed",{1,2,3});m.cache_prefix("seed");m.release("seed");
    const std::vector<TokenId> prompt{1,2,3,4};
    assert(m.reserve_request("a",6,prompt)==KvAdmission::Ready);
    assert(m.reserve_request("b",6,prompt)==KvAdmission::Ready);
    assert(m.capacity_ledger().committed_blocks==5);
    const auto before=m.stats();
    assert(m.reserve_request("wait",6,prompt)==KvAdmission::Wait);
    assert(m.stats().prefix_hits==before.prefix_hits&&m.stats().prefix_evictions==before.prefix_evictions);
    assert(m.capacity_ledger().committed_blocks==5);
    m.create("a");assert(m.reuse_prefix("a",prompt)==3);
    m.create("b");assert(m.reuse_prefix("b",prompt)==3);
    m.clear_prefix_cache();
    // Two private tail budgets fund sequential COW, including source retention.
    {auto tx=m.prepare_append("a",{4,5,6});assert(tx.tail_copy());tx.rollback();}
    assert(m.capacity_ledger().committed_blocks==5);
    commit(m,"a",{4,5,6});commit(m,"b",{7,8,9});
    assert(m.stats().kv_live_blocks==5);
    m.release("a");assert(m.capacity_ledger().committed_blocks==3);
    m.release("b");assert(m.stats().kv_live_blocks==0);
    // Admission leases survive eviction and cancellation before create.
    m.create("seed");commit(m,"seed",{1,2,3,4});m.cache_prefix("seed");m.release("seed");
    assert(m.reserve_request("pending",6,prompt)==KvAdmission::Ready);
    m.clear_prefix_cache();assert(m.stats().kv_live_blocks==1);
    m.create("pending");rejects([&]{m.prepare_append("pending",{1});});
    rejects([&]{m.reuse_prefix("pending",{9,8,7});});
    assert(m.reuse_prefix("pending",prompt)==2);
    m.reset_sequence("pending");assert(m.sequence("pending").used==0);
    assert(m.capacity_ledger().committed_blocks==3);
    assert(m.reuse_prefix("pending",prompt)==2);m.release("pending");
    assert(m.stats().kv_live_blocks==0);
    // Longer prefixes published after admission remain usable.
    assert(m.reserve_request("later",6,prompt)==KvAdmission::Ready);
    m.create("seed");commit(m,"seed",{1,2,3});m.cache_prefix("seed");m.release("seed");
    m.create("later");assert(m.reuse_prefix("later",prompt)==3);m.release("later");m.clear();
    assert(m.capacity_ledger().committed_blocks==0);
    // A running request's full pages participate even when it was admitted cold.
    assert(m.reserve_request("cold",6,prompt)==KvAdmission::Ready);
    m.create("cold");commit(m,"cold",{1,2,3});m.cache_prefix("cold");
    assert(m.reserve_request("hot",6,prompt)==KvAdmission::Ready);
    m.release("cold");m.clear_prefix_cache();assert(m.stats().kv_live_blocks==1);
    m.release_reservation("hot");m.release_reservation("hot");assert(m.stats().kv_live_blocks==0);
    // No sharing: retain conservative admission and request-local Impossible.
    assert(m.reserve_request("off",6,{9})==KvAdmission::Ready);
    assert(m.reserve_request("wait",6,{8})==KvAdmission::Wait);
    assert(m.reserve_request("huge",11,{8})==KvAdmission::Impossible);m.clear();
}
void shared_restore_fallback(){
    KVManager m({2,6,6},1,1,20,100);
    m.create("seed");commit(m,"seed",{1,2});m.cache_prefix("seed");m.release("seed");
    m.create("duplicate");commit(m,"duplicate",{1,2});
    assert(m.reserve_request("a",6,{1,2,3})==KvAdmission::Ready);
    assert(m.reserve_request("b",6,{1,2,3})==KvAdmission::Ready);
    m.clear_prefix_cache();m.cache_prefix("duplicate");m.release("duplicate");
    assert(m.reserve_request("extra",2)==KvAdmission::Ready);
    assert(m.capacity_ledger().committed_blocks==6);
    m.create("a");assert(m.reuse_prefix("a",{1,2,3})==2);
    // Equivalent tokens on a different physical page would break shared credit.
    assert(m.capacity_ledger().committed_blocks==6);
    m.create("b");assert(m.reuse_prefix("b",{1,2,3})==2);
    assert(m.sequence("a").blocks==m.sequence("b").blocks);
    m.clear();assert(m.stats().kv_live_blocks==0);
}
void shared_capacity_foundation(){
    KVManager m({2,12,6},1,1,20,100);
    assert(m.reserve_request("source",8)==KvAdmission::Ready);
    auto ledger=m.capacity_ledger();assert(ledger.logical_reserved_blocks==4&&ledger.future_private_blocks==4);
    m.create("source");commit(m,"source",{1,2,3});m.cache_prefix("source");
    const auto before=m.stats();
    assert(m.probe_prefix({1,2,3,9})==3);
    assert(m.acquire_prefix_lease("first",{1,2,3,9})==3);
    assert(m.acquire_prefix_lease("second",{1,2,3,8})==3);
    const auto after=m.stats();
    assert(before.prefix_hits==after.prefix_hits&&before.prefix_misses==after.prefix_misses);
    assert(before.prefix_reused_tokens==after.prefix_reused_tokens&&before.prefix_evictions==after.prefix_evictions);
    ledger=m.capacity_ledger();
    assert(ledger.pinned_blocks==2&&ledger.prefix_lease_blocks==2&&ledger.shared_reference_savings==4);
    assert(ledger.cache_only_blocks==0&&ledger.future_private_blocks==2);
    assert(ledger.cow_tail_candidates==3);
    rejects([&]{m.acquire_prefix_lease("first",{1,2,3,9});});
    rejects([&]{m.acquire_prefix_lease("bad",{100});});
    // Lease forces COW even after cache eviction; failed copying restores pages.
    m.clear_prefix_cache();const auto original=m.sequence("source").blocks;
    {
        auto append=m.prepare_append("source",{4});assert(append.tail_copy());
        rejects([&]{m.capacity_ledger();});
        rejects([&]{m.release_prefix_lease("first");});
        rejects([&]{append.copy_shared_tail([](std::size_t,std::size_t,std::size_t){throw std::runtime_error("copy");});});
        rejects([&]{append.commit();});
    }
    assert(m.sequence("source").blocks==original&&m.stats().kv_live_blocks==2);
    commit(m,"source",{4});assert(m.stats().kv_live_blocks==3);
    m.release("source");assert(m.capacity_ledger().pinned_blocks==2);
    m.release_prefix_lease("first");m.release_prefix_lease("first");
    assert(m.stats().kv_live_blocks==2&&m.capacity_ledger().shared_reference_savings==0);
    m.release_prefix_lease("second");assert(m.stats().kv_live_blocks==0);
    assert(m.acquire_prefix_lease("miss",{7})==0);m.clear();
    assert(m.capacity_ledger().pinned_blocks==0&&m.capacity_ledger().logical_reserved_blocks==0);
    // Conservatively protect existing promises; failed acquisition leaks nothing.
    KVManager tight({2,3,2},1,1,12,100);tight.create("s");commit(tight,"s",{1,2});
    tight.cache_prefix("s");tight.release("s");
    assert(tight.reserve_request("all",6)==KvAdmission::Ready);
    rejects([&]{tight.acquire_prefix_lease("lease",{1,2,3});});
    assert(tight.stats().kv_cache_only_blocks==1);
    tight.release_reservation("all");tight.acquire_prefix_lease("lease",{1,2,3});
    assert(tight.reserve_request("all",6)==KvAdmission::Wait);
    tight.clear();assert(tight.stats().kv_live_blocks==0);
}
void reservations(){
    KVManager m({2,3,3},1,1,16,100);
    assert(m.reserve_request("too-big",7)==KvAdmission::Impossible);
    assert(m.reserve_request("a",4)==KvAdmission::Ready);
    assert(m.reserve_request("b",4)==KvAdmission::Wait);
    assert(m.stats().kv_live_blocks==0); // logical reservation only
    m.create("external");commit(m,"external",{1,2});
    rejects([&]{m.prepare_append("external",{3});});
    m.create("a");commit(m,"a",{1,2,3,4});m.cache_prefix("a");
    rejects([&]{m.prepare_append("a",{5});});
    m.release("a");assert(m.reserve_request("b",4)==KvAdmission::Ready);
    m.create("b");commit(m,"b",{5,6,7,8}); // cache eviction makes physical room
    m.clear();assert(m.reserve_request("all",6)==KvAdmission::Ready);
    m.release_reservation("all");assert(m.reserve_request("again",6)==KvAdmission::Ready);
    m.clear();assert(m.stats().kv_live_blocks==0);
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
    assert(manager.stats().kv_cache_only_blocks==3&&manager.stats().kv_live_blocks==3);
    manager.create("a");assert(manager.reuse_prefix("a",{1,2,3,4,6})==4);
    const auto first=manager.sequence("a").blocks[0];
    manager.create("b");assert(manager.reuse_prefix("b",{1,2,3,4,7})==4);
    assert(manager.sequence("b").blocks[0]==first&&manager.stats().kv_live_blocks==3);
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
int main(){batch_reference_accounting();batch_space_preparation();shared_restore_fallback();shared_admission();shared_capacity_foundation();reservations();transactions();prefix_and_reset();exhaustion();capacity_and_lru();pinned_victim_then_free_page();
    rollback_keeps_eviction_effects();impossible_reservation_and_cache_off();
    std::cout<<"KVManager transactions, shared leases, reset, stats and exhaustion passed\n";}
