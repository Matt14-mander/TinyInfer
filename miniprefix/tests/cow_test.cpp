#include "miniprefix/model.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#ifdef MINIPREFIX_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
using namespace miniprefix;
template<class F> void rejects(F operation){
    bool caught=false;try{operation();}catch(const std::exception&){caught=true;}assert(caught);
}
void metadata(){
    KVManager m({4,4,0},2,3,16,100);
    assert(m.reserve_request("a",8)==KvAdmission::Ready);
    assert(m.reserve_request("b",8)==KvAdmission::Ready);
    m.create("a");m.create("b");
    {auto tx=m.prepare_append("a",{1,2,3});tx.commit();}
    m.fork_sequence("a","b");const auto original=m.sequence("b").blocks;
    assert(m.stats().kv_live_blocks==1);
    assert(m.reserve_request("third",1)==KvAdmission::Wait);
    for(int attempt=0;attempt<2;++attempt){
        auto tx=m.prepare_append("b",{4,5});
        assert(tx.tail_copy()&&tx.tail_copy()->valid_tokens==3);
        assert(m.stats().kv_live_blocks==3);
        rejects([&]{tx.commit();});rejects([&]{m.release("a");});
        rejects([&]{m.reserve_request("third",1);});rejects([&]{m.create("c");});
        int calls=0;
        rejects([&]{tx.copy_shared_tail([&](std::size_t dst,std::size_t src,std::size_t n){
            assert(n==9&&dst!=src);if(++calls==2)throw std::runtime_error("copy failure");
        });});
        rejects([&]{tx.commit();});tx.rollback();
        assert(m.sequence("b").blocks==original&&m.sequence("b").used==3);
        assert(m.stats().kv_live_blocks==1&&m.stats().forward_tokens==3);
    }
    {auto tx=m.prepare_append("b",{4,5});int calls=0;
        tx.copy_shared_tail([&](std::size_t,std::size_t,std::size_t){++calls;});assert(calls==2);tx.commit();}
    assert(m.sequence("a").blocks==original&&m.sequence("a").used==3);
    {auto tx=m.prepare_append("a",{6,7});assert(!tx.tail_copy());tx.commit();}
    assert(m.stats().kv_live_blocks==4);m.clear();assert(m.stats().kv_live_blocks==0);
    KVManager small({4,1,0},1,1,8,100);small.create("a");small.create("b");
    {auto tx=small.prepare_append("a",{1});tx.commit();}
    rejects([&]{small.fork_sequence("a","b");});assert(small.sequence("b").blocks.empty());
    small.clear();assert(small.stats().kv_live_blocks==0);
    KVManager bounded({4,3,0},1,1,16,100);
    assert(bounded.reserve_request("a",8)==KvAdmission::Ready);
    assert(bounded.reserve_request("b",4)==KvAdmission::Ready);
    bounded.create("a");bounded.create("b");
    {auto tx=bounded.prepare_append("a",{1,2,3,4,5});tx.commit();}
    rejects([&]{bounded.fork_sequence("a","b");});
    assert(bounded.sequence("b").tokens.empty()&&bounded.stats().kv_live_blocks==2);
    bounded.clear();assert(bounded.stats().kv_live_blocks==0);
}
// Fail after one layer's real K/V copy; the backend must finish outstanding
// device work before the transaction releases either source or destination.
void payload(bool cuda,bool merged=false){
    KVManager m({4,4,0},2,3,16,100);m.create("a");m.create("b");
    {auto tx=m.prepare_append("a",{1,2,3});tx.commit();}m.fork_sequence("a","b");
    const auto original=m.sequence("a").blocks;
    const auto elements=m.reserved_bytes()/sizeof(float)/2;
    std::vector<float> keys(elements),values(elements);
    for(std::size_t i=0;i<elements;++i){keys[i]=float(i+1);values[i]=-float(i+1);}
    auto* k=keys.data();auto* v=values.data();
#ifdef MINIPREFIX_ENABLE_CUDA
    struct Device {
        float *k=nullptr,*v=nullptr;cudaStream_t stream=nullptr;
        ~Device(){if(stream)cudaStreamSynchronize(stream);cudaFree(k);cudaFree(v);if(stream)cudaStreamDestroy(stream);}
    } device;
    auto check=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
    if(cuda){check(cudaStreamCreate(&device.stream));check(cudaMalloc(&device.k,elements*4));check(cudaMalloc(&device.v,elements*4));
        k=device.k;v=device.v;check(cudaMemcpy(k,keys.data(),elements*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(v,values.data(),elements*4,cudaMemcpyHostToDevice));}
#endif
    for(bool fail:{true,false}){
        auto tx=m.prepare_append("b",{4,5});int calls=0;
        const auto copy=*tx.tail_copy();bool caught=false;
        try{if(merged){tx.copy_shared_tail_region([&](const KVManager::TailCopyRegion& r){
            assert(r.rows==2&&r.pitch==12&&r.width==9);
#ifdef MINIPREFIX_ENABLE_CUDA
            if(cuda){
                check(cudaMemcpy2DAsync(k+r.destination,r.pitch*4,k+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));
                if(fail)throw std::runtime_error("failure between merged K and V submissions");
                check(cudaMemcpy2DAsync(v+r.destination,r.pitch*4,v+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));
            }else
#endif
            {
                for(std::size_t row=0;row<r.rows;++row)std::copy_n(k+r.source+row*r.pitch,r.width,k+r.destination+row*r.pitch);
                if(fail)throw std::runtime_error("failure between merged K and V copies");
                for(std::size_t row=0;row<r.rows;++row)std::copy_n(v+r.source+row*r.pitch,r.width,v+r.destination+row*r.pitch);
            }
        });}else tx.copy_shared_tail([&](std::size_t dst,std::size_t src,std::size_t count){
#ifdef MINIPREFIX_ENABLE_CUDA
            if(cuda){check(cudaMemcpyAsync(k+dst,k+src,count*4,cudaMemcpyDeviceToDevice,device.stream));
                check(cudaMemcpyAsync(v+dst,v+src,count*4,cudaMemcpyDeviceToDevice,device.stream));}
            else
#endif
            {std::copy_n(k+src,count,k+dst);std::copy_n(v+src,count,v+dst);}
            if(++calls==1&&fail)throw std::runtime_error("injected after first layer copy");
        });}catch(const std::runtime_error&){caught=true;}
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda){check(cudaStreamSynchronize(device.stream));check(cudaMemcpy(keys.data(),k,elements*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(values.data(),v,elements*4,cudaMemcpyDeviceToHost));}
#endif
        assert(caught==fail);
        for(int layer=0;layer<(fail&&!merged?1:2);++layer)for(int i=0;i<12;++i){
            auto src=(copy.source*2+layer)*12+i,dst=(copy.destination*2+layer)*12+i;
            assert(keys[src]==float(src+1)&&values[src]==-float(src+1));
            if(i<9){assert(keys[dst]==keys[src]);
                assert(values[dst]==((merged&&fail)?-float(dst+1):values[src]));}
            else assert(keys[dst]==float(dst+1)&&values[dst]==-float(dst+1));
        }
        if(fail){rejects([&]{tx.commit();});tx.rollback();assert(m.sequence("b").blocks==original&&m.stats().kv_live_blocks==1);}
        else tx.commit();
    }
    m.release("a");m.release("b");assert(m.stats().kv_live_blocks==0);
}
void pressure_payload(bool cuda){
    const bool merged=true;
    KVManager m({4,4,1},2,3,16,100);m.create("a");m.create("b");
    {auto tx=m.prepare_append("a",{1,2,3});tx.commit();}m.fork_sequence("a","b");
    m.create("victim");{auto tx=m.prepare_append("victim",{9,9,9,9});tx.commit();}
    m.cache_prefix("victim");m.release("victim");
    const auto original=m.sequence("a").blocks;
    const auto elements=m.reserved_bytes()/sizeof(float)/2;
    std::vector<float> keys(elements),values(elements);
    for(std::size_t i=0;i<elements;++i){keys[i]=float(i+1);values[i]=-float(i+1);}
    auto* k=keys.data();auto* v=values.data();
#ifdef MINIPREFIX_ENABLE_CUDA
    struct Device {
        float *k=nullptr,*v=nullptr;cudaStream_t stream=nullptr;
        ~Device(){if(stream)cudaStreamSynchronize(stream);cudaFree(k);cudaFree(v);if(stream)cudaStreamDestroy(stream);}
    } device;
    auto check=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
    if(cuda){check(cudaStreamCreate(&device.stream));check(cudaMalloc(&device.k,elements*4));check(cudaMalloc(&device.v,elements*4));
        k=device.k;v=device.v;check(cudaMemcpy(k,keys.data(),elements*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(v,values.data(),elements*4,cudaMemcpyHostToDevice));}
#endif
    for(bool fail:{true,false}){
        auto group=m.prepare_batch({{"b",{4,5}},{"a",{6},true}});assert(group.size()==2);
        assert(m.stats().prefix_evictions==1&&m.stats().kv_live_blocks==4);
        auto& tx=*group[0];int calls=0;
        const auto copy=*tx.tail_copy();bool caught=false;
        try{if(merged){tx.copy_shared_tail_region([&](const KVManager::TailCopyRegion& r){
            assert(r.rows==2&&r.pitch==12&&r.width==9);
#ifdef MINIPREFIX_ENABLE_CUDA
            if(cuda){
                check(cudaMemcpy2DAsync(k+r.destination,r.pitch*4,k+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));
                if(fail)throw std::runtime_error("failure between merged K and V submissions");
                check(cudaMemcpy2DAsync(v+r.destination,r.pitch*4,v+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));
            }else
#endif
            {
                for(std::size_t row=0;row<r.rows;++row)std::copy_n(k+r.source+row*r.pitch,r.width,k+r.destination+row*r.pitch);
                if(fail)throw std::runtime_error("failure between merged K and V copies");
                for(std::size_t row=0;row<r.rows;++row)std::copy_n(v+r.source+row*r.pitch,r.width,v+r.destination+row*r.pitch);
            }
        });}else tx.copy_shared_tail([&](std::size_t dst,std::size_t src,std::size_t count){
#ifdef MINIPREFIX_ENABLE_CUDA
            if(cuda){check(cudaMemcpyAsync(k+dst,k+src,count*4,cudaMemcpyDeviceToDevice,device.stream));
                check(cudaMemcpyAsync(v+dst,v+src,count*4,cudaMemcpyDeviceToDevice,device.stream));}
            else
#endif
            {std::copy_n(k+src,count,k+dst);std::copy_n(v+src,count,v+dst);}
            if(++calls==1&&fail)throw std::runtime_error("injected after first layer copy");
        });}catch(const std::runtime_error&){caught=true;}
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda){check(cudaStreamSynchronize(device.stream));check(cudaMemcpy(keys.data(),k,elements*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(values.data(),v,elements*4,cudaMemcpyDeviceToHost));}
#endif
        assert(caught==fail);
        for(int layer=0;layer<(fail&&!merged?1:2);++layer)for(int i=0;i<12;++i){
            auto src=(copy.source*2+layer)*12+i,dst=(copy.destination*2+layer)*12+i;
            assert(keys[src]==float(src+1)&&values[src]==-float(src+1));
            if(i<9){assert(keys[dst]==keys[src]);
                assert(values[dst]==((merged&&fail)?-float(dst+1):values[src]));}
            else assert(keys[dst]==float(dst+1)&&values[dst]==-float(dst+1));
        }
        if(fail){rejects([&]{tx.commit();});tx.rollback();group[1]->rollback();assert(m.sequence("b").blocks==original&&m.stats().kv_live_blocks==1);}
        else{
            group[1]->copy_shared_tail_region([&](const KVManager::TailCopyRegion& r){
#ifdef MINIPREFIX_ENABLE_CUDA
                if(cuda){check(cudaMemcpy2DAsync(k+r.destination,r.pitch*4,k+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));
                    check(cudaMemcpy2DAsync(v+r.destination,r.pitch*4,v+r.source,r.pitch*4,r.width*4,r.rows,cudaMemcpyDeviceToDevice,device.stream));}
                else
#endif
                for(std::size_t row=0;row<r.rows;++row){std::copy_n(k+r.source+row*r.pitch,r.width,k+r.destination+row*r.pitch);
                    std::copy_n(v+r.source+row*r.pitch,r.width,v+r.destination+row*r.pitch);}
            });
#ifdef MINIPREFIX_ENABLE_CUDA
            if(cuda)check(cudaStreamSynchronize(device.stream));
#endif
            tx.commit();group[1]->commit();
        }
    }
    m.release("a");m.release("b");assert(m.stats().kv_live_blocks==0);
}
std::shared_ptr<ModelWeights> weights(int hidden=8,int layers=2){
    auto w=std::make_shared<ModelWeights>();auto& c=w->config;
    c.layers=layers;c.hidden=hidden;c.heads=hidden>256?8:2;c.kv_heads=1;c.intermediate=hidden+4;c.vocabulary=16;
    c.context=32;c.eos=0;c.rms_epsilon=1e-5F;c.rope_theta=10000;c.tied=false;c.validate();
    int seed=0;
    auto add=[&](std::string name,tinyinfer::Shape shape,bool norm=false){
        tinyinfer::Tensor t(shape);for(std::size_t i=0;i<t.numel();++i)t.at(i)=norm?1.0F:0.15F*std::sin(float(++seed));
        w->tensors.emplace(name,std::move(t));
    };
    add("model.embed_tokens.weight",{16,hidden});add("lm_head.weight",{hidden,16});add("model.norm.weight",{hidden},true);
    for(int l=0;l<layers;++l){auto p="model.layers."+std::to_string(l)+".";
        add(p+"input_layernorm.weight",{hidden},true);add(p+"post_attention_layernorm.weight",{hidden},true);
        add(p+"self_attn.q_proj.weight",{hidden,hidden});add(p+"self_attn.k_proj.weight",{hidden,c.kv_width()});
        add(p+"self_attn.v_proj.weight",{hidden,c.kv_width()});add(p+"self_attn.o_proj.weight",{hidden,hidden});
        add(p+"mlp.gate_proj.weight",{hidden,c.intermediate});add(p+"mlp.up_proj.weight",{hidden,c.intermediate});add(p+"mlp.down_proj.weight",{c.intermediate,hidden});
    }
    // Untied head stays finite: only appending this token injects a backend failure.
    for(int i=0;i<hidden;++i)w->tensors.at("model.embed_tokens.weight").at(15*hidden+i)=std::numeric_limits<float>::quiet_NaN();
    return w;
}
void same(const std::vector<float>& a,const std::vector<float>& b){
    assert(a.size()==b.size());for(std::size_t i=0;i<a.size();++i)assert(std::isfinite(a[i])&&std::abs(a[i]-b[i])<2e-5F);
    assert(std::max_element(a.begin(),a.end())-a.begin()==std::max_element(b.begin(),b.end())-b.begin());
}
void runner(bool cuda){
    auto w=weights();auto make=[&](){
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda)return make_cuda_runner(w,16,{4,4,0});
#endif
        return make_cpu_runner(w,16,{4,4,0});
    };
    auto r=make();auto oracle=make_cpu_runner(w,16,{4,8,0});
    for(int length:{3,4,7})for(bool fail:{false,true}){
        assert(r->reserve_request("a",8)==KvAdmission::Ready);assert(r->reserve_request("b",8)==KvAdmission::Ready);
        r->create("a");r->create("b");oracle->create("ref");
        std::vector<TokenId> history(length);for(int i=0;i<length;++i)history[i]=i+1;
        r->append("a",history);r->fork_sequence("a","b");oracle->append("ref",history);
        const auto before=r->stats();
        if(fail){
            rejects([&]{r->append("b",length==3?std::vector<TokenId>{15,9}:std::vector<TokenId>{15});});
            assert(r->stats().forward_tokens==before.forward_tokens&&r->stats().kv_live_blocks==before.kv_live_blocks);
            // Retry after source release must repair the GPU table even without COW.
            r->release("a");
        }
        const auto suffix=length==3?std::vector<TokenId>{9,8}:std::vector<TokenId>{9};
        same(r->append("b",suffix),oracle->append("ref",suffix));
        if(!fail){oracle->release("ref");oracle->create("ref");oracle->append("ref",history);
            same(r->append("a",{10}),oracle->append("ref",{10}));r->release("a");}
        r->release("b");oracle->clear();assert(r->stats().kv_live_blocks==0&&r->stats().block_table_device_bytes==0);
    }
    r->clear();assert(r->stats().live_sequences==0);
}
void partial_cache(bool cuda){
    auto w=weights();auto make=[&](KvCacheConfig config){
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda)return make_cuda_runner(w,16,config);
#endif
        return make_cpu_runner(w,16,config);
    };
    auto r=make({4,8,8});auto oracle=make_cpu_runner(w,16,{4,8,0});
    for(int length:{1,3,4,7,8}){
        std::vector<TokenId> history(length);for(int i=0;i<length;++i)history[i]=i+1;
        r->create("source");r->append("source",history);r->cache_prefix("source");
        // Match both a cached partial page and an interior prefix of a full page.
        for(int common:{1,length}){
            std::vector<TokenId> prompt(history.begin(),history.begin()+common);prompt.push_back(10);
            r->create("target");assert(r->reuse_prefix("target",prompt)==std::size_t(common));
            const auto before=r->stats();rejects([&]{r->append("target",{15});});
            assert(r->stats().forward_tokens==before.forward_tokens&&r->stats().kv_live_blocks==before.kv_live_blocks);
            oracle->create("ref");same(r->append("target",{10}),oracle->append("ref",prompt));
            r->release("target");oracle->clear();
        }
        r->create("identical");assert(r->reuse_prefix("identical",history)==std::size_t(length-1));
        oracle->create("ref");same(r->append("identical",{history.back()}),oracle->append("ref",history));
        r->clear();oracle->clear();assert(r->stats().kv_live_blocks==0&&r->stats().block_table_device_bytes==0);
    }
    // All capacity is reserved. Cache-only sharing must become exclusive when
    // its lease is evicted, without requiring an unavailable replacement page.
    r=make({4,1,1});r->create("seed");r->append("seed",{1,2,3});r->cache_prefix("seed");r->release("seed");
    assert(r->reserve_request("next",4)==KvAdmission::Ready);r->create("next");
    assert(r->reuse_prefix("next",{1,2,3,4})==3);
    oracle->create("ref");same(r->append("next",{4}),oracle->append("ref",{1,2,3,4}));
    assert(r->stats().kv_live_blocks==1&&r->stats().kv_cached_blocks==0);
    r->release("next");assert(r->stats().kv_live_blocks==0);oracle->clear();
}
void costs(bool cuda){
    for(bool timed:{false,true}){
        auto w=weights();auto r=make_cpu_runner(w,16,{4,8,8});
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda)r=make_cuda_runner(w,16,{4,8,8});
#endif
        r->set_cost_timing(timed);r->set_cost_timing(timed);
        r->create("a");r->append("a",{1,2,3});r->cache_prefix("a");r->create("b");
        assert(r->reuse_prefix("b",{1,2,3,4})==3);r->append("b",{4});
        auto cost=r->cost_stats();assert(cost.lookup_calls==1&&cost.publish_calls==1);
        assert(cost.cow_pages==1&&cost.cow_bytes==192&&cost.cow_timed_copies==std::uint64_t(timed));
        if(!timed)assert(cost.lookup_ns==0&&cost.publish_ns==0&&cost.cow_copy_ns==0);
        r->create("failed");r->fork_sequence("a","failed");rejects([&]{r->append("failed",{15});});
        cost=r->cost_stats();assert(cost.cow_pages==2&&cost.cow_bytes==384);
        // Work already copied is observable even when the model transaction fails.
        assert(cost.cow_timed_copies==(timed?2:0));
        r->clear();assert(r->cost_stats().cow_bytes==384&&r->stats().kv_live_blocks==0);
        r->set_cost_timing(false);
        const auto stopped=r->cost_stats();
        r->create("next");r->append("next",{1,2,3});r->cache_prefix("next");r->create("copy");
        assert(r->reuse_prefix("copy",{1,2,3,4})==3);r->append("copy",{4});
        const auto final=r->cost_stats();assert(final.cow_bytes==576&&final.cow_pages==3);
        assert(final.lookup_ns==stopped.lookup_ns&&final.publish_ns==stopped.publish_ns);
        assert(final.cow_copy_ns==stopped.cow_copy_ns&&final.cow_timed_copies==stopped.cow_timed_copies);
        r->clear();
    }
}
void publication_snapshots(bool cuda){
    auto w=weights();auto r=make_cpu_runner(w,16,{4,16,5});
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda)r=make_cuda_runner(w,16,{4,16,5});
#endif
    auto oracle=make_cpu_runner(w,16,{4,8,0});
    // Repeated growing snapshots exceed cache capacity. Probe after the source
    // is released, so stale/recycled index pointers cannot hide behind its lease.
    for(int round=0;round<6;++round){
        std::vector<TokenId> history={1,2,3,4,5};history[0]+=round%2;
        r->create("source");r->append("source",history);
        for(int token:{6,7,8,9,10}){
            r->cache_prefix("source");r->cache_prefix("source");
            r->append("source",{token});history.push_back(token);
        }
        r->cache_prefix("source");r->release("source");
        auto prompt=history;prompt.push_back(11);
        r->create("target");assert(r->reuse_prefix("target",prompt)==history.size());
        const auto before=r->stats();rejects([&]{r->append("target",{15});});
        assert(r->stats().kv_live_blocks==before.kv_live_blocks&&r->stats().forward_tokens==before.forward_tokens);
        oracle->create("ref");same(r->append("target",{11}),oracle->append("ref",prompt));
        r->cache_prefix("target");r->release("target");oracle->clear();
        if(round%2){r->clear_prefix_cache();assert(r->stats().kv_live_blocks==0);}
    }
    assert(r->stats().prefix_evictions>0);r->clear();
    assert(r->stats().live_sequences==0&&r->stats().kv_live_blocks==0&&r->stats().block_table_device_bytes==0);
}
void decode_transactions(){
    KVManager m({4,8,4},1,1,16,16);m.create("a");m.create("b");
    {auto tx=m.prepare_append("a",{1,2,3});tx.commit();}
    m.fork_sequence("a","b");m.cache_prefix("a");
    const auto a=m.sequence("a"),b=m.sequence("b");const auto before=m.stats();
    assert(m.prepare_decode_batch({{"a",4},{"a",5}}).empty());
    assert(m.prepare_decode_batch({{"a",4},{"b",16}}).empty());
    {
        auto group=m.prepare_decode_batch({{"a",4},{"b",5}});assert(group.size()==2);
        rejects([&]{m.clear();});rejects([&]{m.cache_prefix("a");});rejects([&]{m.prepare_append("a",{1});});
        for(auto& tx:group)tx->copy_shared_tail([](std::size_t,std::size_t,std::size_t){});
        group[0]->rollback();rejects([&]{m.clear();});group[1]->rollback();
    }
    assert(m.sequence("a").blocks==a.blocks&&m.sequence("b").blocks==b.blocks);
    assert(m.stats().forward_tokens==before.forward_tokens&&m.stats().kv_live_blocks==before.kv_live_blocks);
    auto group=m.prepare_decode_batch({{"a",4},{"b",5}});
    for(auto& tx:group){tx->copy_shared_tail([](std::size_t,std::size_t,std::size_t){});tx->commit();}
    assert(m.sequence("a").tokens.back()==4&&m.sequence("b").tokens.back()==5);m.clear();
    KVManager small({4,2,1},1,1,16,16);small.create("a");small.create("b");
    {auto tx=small.prepare_append("a",{1,2,3});tx.commit();}small.fork_sequence("a","b");small.cache_prefix("a");
    assert(small.prepare_decode_batch({{"a",4},{"b",5}}).empty());
    assert(small.stats().prefix_evictions==0&&small.stats().kv_live_blocks==1);small.clear();
    KVManager bounded({4,4,1},1,1,16,16);
    assert(bounded.reserve_request("a",8)==KvAdmission::Ready);
    assert(bounded.reserve_request("b",4)==KvAdmission::Ready);
    bounded.create("a");bounded.create("b");
    {auto tx=bounded.prepare_append("a",{1,2,3});tx.commit();}
    {auto tx=bounded.prepare_append("b",{1,2,3,4});tx.commit();}bounded.cache_prefix("a");
    const auto original=bounded.sequence("a").blocks;
    // Row A can prepare, but row B exceeds its reservation. Undo the entire
    // prepared prefix without evicting or leaving the manager locked.
    assert(bounded.prepare_decode_batch({{"a",4},{"b",5}}).empty());
    assert(bounded.sequence("a").blocks==original&&bounded.sequence("a").used==3);
    assert(bounded.stats().forward_tokens==7&&bounded.stats().kv_live_blocks==2&&bounded.stats().prefix_evictions==0);
    {auto tx=bounded.prepare_append("a",{4});tx.copy_shared_tail([](std::size_t,std::size_t,std::size_t){});tx.commit();}
    bounded.clear();assert(bounded.stats().kv_live_blocks==0);
}
void batched_decode(bool cuda){
    auto w=weights();auto r=make_cpu_runner(w,16,{4,32,20});
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda)r=make_cuda_runner(w,16,{4,32,20});
#endif
    auto oracle=make_cpu_runner(w,16,{4,32,0});
    for(int length:{3,4,7,9}){
        const auto id=std::to_string(length);std::vector<TokenId> history(length);
        for(int i=0;i<length;++i)history[i]=i%10+1;
        r->create(id);oracle->create(id);r->append(id,history);oracle->append(id,history);r->cache_prefix(id);
    }
    r->set_cost_timing(true);
    for(int step=0;step<3;++step){
        std::vector<DecodeInput> inputs;for(int length:{3,4,7,9})inputs.push_back({std::to_string(length),10+step});
        const auto before=r->stats();auto result=r->append_decode_batch(inputs);
        assert(bool(result)==cuda);
        for(std::size_t row=0;row<inputs.size();++row){const auto& in=inputs[row];
            same(result?(*result)[row]:r->append(in.id,{in.token}),oracle->append(in.id,{in.token}));r->cache_prefix(in.id);}
        if(cuda&&step)assert(r->stats().device_allocations==before.device_allocations);
    }
    if(cuda){
        const auto before=r->stats();const auto cost=r->cost_stats();
        assert(r->decode_batch_stats().batches==3&&r->decode_batch_stats().rows==12);
        assert(cost.cow_pages>0&&cost.cow_timed_copies==cost.cow_pages);
        assert(!r->append_decode_batch({{"3",11},{"4",15}})); // rollback all rows, including the good row
        assert(r->stats().forward_tokens==before.forward_tokens&&r->stats().kv_live_blocks==before.kv_live_blocks);
        auto retry=r->append_decode_batch({{"3",11},{"4",12}});assert(retry);
        same((*retry)[0],oracle->append("3",{11}));same((*retry)[1],oracle->append("4",{12}));
        assert(r->decode_batch_stats().fallbacks==1);
    }
    r->clear();oracle->clear();assert(r->stats().kv_live_blocks==0&&r->stats().block_table_device_bytes==0);
    r->create("seed");r->append("seed",{1,2,3});r->cache_prefix("seed");
    for(const auto* id:{"x","y"}){r->create(id);assert(r->reuse_prefix(id,{1,2,3,4})==3);}
    r->release("seed");
    auto shared=r->append_decode_batch({{"x",4},{"y",5}});assert(bool(shared)==cuda);
    oracle->create("x");oracle->create("y");
    same(shared?(*shared)[0]:r->append("x",{4}),oracle->append("x",{1,2,3,4}));
    same(shared?(*shared)[1]:r->append("y",{5}),oracle->append("y",{1,2,3,5}));
    r->cache_prefix("x");r->cache_prefix("y");r->release("x");r->release("y");
    r->create("probe");assert(r->reuse_prefix("probe",{1,2,3,5,6})==4);
    same(r->append("probe",{6}),oracle->append("y",{6}));r->clear();oracle->clear();

    r=make_cpu_runner(w,16,{4,2,1});
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda)r=make_cuda_runner(w,16,{4,2,1});
#endif
    r->create("a");r->create("b");r->append("a",{1,2,3});r->fork_sequence("a","b");r->cache_prefix("a");
    const auto before=r->stats();assert(!r->append_decode_batch({{"a",4},{"b",5}}));
    assert(r->stats().forward_tokens==before.forward_tokens&&r->stats().prefix_evictions==before.prefix_evictions);
    oracle->create("a");oracle->create("b");
    same(r->append("a",{4}),oracle->append("a",{1,2,3,4}));r->cache_prefix("a");r->release("a");
    same(r->append("b",{5}),oracle->append("b",{1,2,3,5}));r->release("b");r->clear();oracle->clear();
    assert(r->stats().kv_live_blocks==0&&r->stats().block_table_device_bytes==0);
}
void pressure_runner(bool cuda){
    auto w=weights();auto runner=make_cpu_runner(w,16,{4,4,3});
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda)runner=make_cuda_runner(w,16,{4,4,3});
#endif
    auto oracle=make_cpu_runner(w,16,{4,8,0});
    for(const auto* id:{"a","b"}){runner->create(id);oracle->create(id);oracle->append(id,{1,2,3});}
    runner->append("a",{1,2,3});runner->fork_sequence("a","b");runner->cache_prefix("a");
    for(int token:{6,7}){runner->create("seed");runner->append("seed",{token,token,token,token});runner->cache_prefix("seed");runner->release("seed");}
    const auto before=runner->stats();
    if(cuda){
        assert(!runner->append_decode_batch({{"a",4},{"b",15}}));
        assert(runner->stats().prefix_evictions==before.prefix_evictions+2);
        assert(runner->stats().forward_tokens==before.forward_tokens&&runner->stats().kv_live_blocks==2);
    }
    auto result=runner->append_decode_batch({{"a",4},{"b",5}});assert(bool(result)==cuda);
    same(result?result->at(0):runner->append("a",{4}),oracle->append("a",{4}));runner->cache_prefix("a");
    same(result?result->at(1):runner->append("b",{5}),oracle->append("b",{5}));runner->cache_prefix("b");
    runner->create("probe");assert(runner->reuse_prefix("probe",{1,2,3,5,8})==4);
    same(runner->append("probe",{8}),oracle->append("b",{8}));
    runner->clear();assert(runner->stats().kv_live_blocks==0&&runner->stats().block_table_device_bytes==0);
}
void packed_prefill(bool cuda){
    auto w=weights();
    auto runner=make_cpu_runner(w,32,{4,32,16});
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda)runner=make_cuda_runner(w,32,{4,32,16});
#endif
    auto oracle=make_cpu_runner(w,32,{4,32,0});
    runner->create("a");runner->create("b");runner->create("c");
    runner->append("a",{1,2,3});runner->fork_sequence("a","b");
    for(const auto* id:{"a","b","c"})oracle->create(id);
    oracle->append("a",{1,2,3});oracle->append("b",{1,2,3});
    auto compare=[&](const std::vector<BatchInput>& chunks){
        auto result=runner->append_batch(chunks);
        assert(bool(result)==cuda);
        for(std::size_t i=0;i<chunks.size();++i){
            const auto& chunk=chunks[i];
            same(result?result->at(i):runner->append(chunk.id,chunk.tokens),oracle->append(chunk.id,chunk.tokens));
            runner->cache_prefix(chunk.id);
        }
    };
    assert(runner->execution_stats().forwards==0);
    runner->set_execution_timing(true);
    compare({{"a",{4,5,6,7,8,9}},{"b",{10},true},{"c",{6,5,4}}});
    const auto timed=runner->execution_stats();
    if(cuda){
        assert(timed.forwards==1 && timed.stream_forward_ns>0);
        assert(timed.forward_download_ns>=timed.host_forward_ns && timed.host_forward_ns>0);
    }
    runner->set_execution_timing(false);
    if(cuda){
        const auto before=runner->stats();
        assert(!runner->append_batch({{"a",{4,3}},{"c",{2,15,1}}}));
        assert(runner->stats().forward_tokens==before.forward_tokens && runner->stats().kv_live_blocks==before.kv_live_blocks);
    }
    compare({{"c",{2,3,4,5}},{"a",{10,11}}});
    assert(runner->execution_stats().forwards==timed.forwards);
    if(cuda){
        const auto stats=runner->prefill_batch_stats();
        assert(stats.prefill_batches==1 && stats.mixed_batches==1 && stats.prefill_tokens==15 && stats.decode_tokens==1 && stats.fallbacks==1);
    }
    runner->clear();assert(runner->stats().kv_live_blocks==0 && runner->stats().block_table_device_bytes==0);

    KVManager small({4,5,2},1,1,32,16);
    small.create("a");small.create("b");
    {auto tx=small.prepare_append("a",{1,2,3});tx.commit();}
    small.fork_sequence("a","b");small.cache_prefix("a");
    assert(small.prepare_batch({{"a",{4,5,6,7,8,9,10,11,12}},{"b",{4,5}}}).empty());
    assert(small.stats().kv_live_blocks==1 && small.stats().prefix_evictions==0 && small.stats().forward_tokens==3);
}
void generation_results(bool cuda){
    auto w=weights();
    auto make=[&](){
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda)return make_cuda_runner(w,32,{4,32,16});
#endif
        return make_cpu_runner(w,32,{4,32,16});
    };
    auto runner=make();
    auto oracle=make_cpu_runner(w,32,{4,32,16});
    for(auto* r:{runner.get(),oracle.get()}){
        r->create("a");r->create("b");
        r->append("a",{1,2,3});r->fork_sequence("a","b");
    }
    auto before=runner->stats();
    auto compare=[&](const GenerationResult& result,const std::vector<float>& logits){
        if(cuda)assert(std::get<TokenId>(result)==std::max_element(logits.begin(),logits.end())-logits.begin());
        else same(std::get<std::vector<float>>(result),logits);
    };
    if(cuda){
        const std::vector<BatchInput> bad={{"a",{4,5}},{"b",{15},true}};
        assert(!runner->append_generation_batch(bad));
        assert(runner->stats().forward_tokens==before.forward_tokens);
        assert(runner->stats().kv_live_blocks==before.kv_live_blocks);
    }
    const std::vector<BatchInput> mixed={{"b",{7},true},{"a",{4,5}}};
    auto result=runner->append_generation_batch(mixed);
    assert(bool(result)==cuda);
    for(size_t i=0;i<mixed.size();++i){
        const auto& row=mixed[i];
        compare(result?result->at(i):runner->append_generation(row.id,row.tokens),oracle->append(row.id,row.tokens));
        runner->cache_prefix(row.id);
    }
    rejects([&]{runner->append_generation("a",{15});});
    compare(runner->append_generation("a",{8}),oracle->append("a",{8}));
    assert(runner->stats().logits_downloaded_bytes==before.logits_downloaded_bytes);
    if(cuda)assert(runner->sampled_id_downloaded_bytes()==6*sizeof(TokenId));
    same(runner->append("a",{3}),oracle->append("a",{3}));
    runner->clear();
    assert(runner->stats().kv_live_blocks==0);

    // The first maximum can be in a higher lane (2 vs 257 in lane 1).
    w->config.vocabulary=513;
    tinyinfer::Tensor embeddings({513,8});
    for(int i=0;i<513*8;++i)embeddings.at(i)=1.0F;
    w->tensors.at("model.embed_tokens.weight")=std::move(embeddings);
    tinyinfer::Tensor head({8,513});
    for(int i=0;i<8*513;++i)head.at(i)=0;
    head.at(2)=head.at(257)=1.0F;
    w->tensors.at("lm_head.weight")=std::move(head);
    auto ties=make();ties->create("tie");
    auto logits=ties->append("tie",{1});
    assert(std::max_element(logits.begin(),logits.end())-logits.begin()==2);
    ties->create("again");
    compare(ties->append_generation("again",{1}),logits);
    // Only one losing column is nonfinite. A plain maximum could otherwise
    // silently ignore it, particularly on a different lane from the winner.
    for(float invalid:{std::numeric_limits<float>::quiet_NaN(),-std::numeric_limits<float>::infinity()}){
        w->tensors.at("lm_head.weight").at(512)=invalid;
        auto damaged=make();damaged->create("bad");
        rejects([&]{damaged->append_generation("bad",{1});});
        assert(damaged->stats().forward_tokens==0 && damaged->stats().kv_live_blocks==0);
    }
}
void residual_norm_shapes(bool cuda){
    for(int width:{8,272,576})for(int layers:{1,3}){
        auto w=weights(width,layers);
        auto runner=make_cpu_runner(w,32,{4,32,16});
#ifdef MINIPREFIX_ENABLE_CUDA
        if(cuda)runner=make_cuda_runner(w,32,{4,32,16});
#endif
        auto oracle=make_cpu_runner(w,32,{4,32,0});
        auto compare=[](const std::vector<float>& actual,const std::vector<float>& expected){
            assert(actual.size()==expected.size());
            for(size_t i=0;i<actual.size();++i)
                assert(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<2e-4F);
            assert(std::max_element(actual.begin(),actual.end())-actual.begin()==
                   std::max_element(expected.begin(),expected.end())-expected.begin());
        };
        for(const auto* id:{"a","b"}){runner->create(id);oracle->create(id);}
        compare(runner->append("a",{1,2,3}),oracle->append("a",{1,2,3}));
        runner->fork_sequence("a","b");oracle->fork_sequence("a","b");
        // Last rows are gathered from indices 2 and 3, then positions change
        // independently. This also requires retaining the residual across layers.
        for(const auto& chunks:std::vector<std::vector<BatchInput>>{
            {{"a",{4,5,6}},{"b",{7},true}},{{"b",{8},true},{"a",{9},true}}}){
            auto result=runner->append_batch(chunks);
            for(size_t i=0;i<chunks.size();++i){
                const auto& c=chunks[i];
                compare(result?result->at(i):runner->append(c.id,c.tokens),oracle->append(c.id,c.tokens));
                runner->cache_prefix(c.id);
            }
        }
        runner->clear();
        assert(runner->stats().kv_live_blocks==0);
    }
}
int main(int argc,char** argv){
    std::cout<<"COW regression started"<<std::endl;
    bool cuda=false,required=false;for(int i=1;i<argc;++i){cuda|=std::string(argv[i])=="--cuda";required|=std::string(argv[i])=="--require-device";}
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda){int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return required?1:77;}
#else
    assert(!cuda);
#endif
    metadata();std::cout<<"COW metadata passed"<<std::endl;
    pressure_payload(cuda);pressure_runner(cuda);
    payload(cuda);payload(cuda,true);std::cout<<"COW payload passed"<<std::endl;
    runner(cuda);std::cout<<"COW copy, capacity, rollback, retry and release passed"<<std::endl;
    partial_cache(cuda);std::cout<<"Partial prefix cache, COW and capacity passed"<<std::endl;
    costs(cuda);std::cout<<"Cost counters and opt-in timing passed"<<std::endl;
    publication_snapshots(cuda);std::cout<<"Publication snapshot churn, rollback and release passed"<<std::endl;
    decode_transactions();batched_decode(cuda);std::cout<<"Batched decode ragged positions, COW, rollback and retry passed"<<std::endl;
    residual_norm_shapes(cuda);
    generation_results(cuda);
    packed_prefill(cuda);std::cout<<"Packed prefill and mixed chunks passed"<<std::endl;
}
