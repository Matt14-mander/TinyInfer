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
std::shared_ptr<ModelWeights> weights(){
    auto w=std::make_shared<ModelWeights>();auto& c=w->config;
    c.layers=2;c.hidden=8;c.heads=2;c.kv_heads=1;c.intermediate=12;c.vocabulary=16;
    c.context=32;c.eos=0;c.rms_epsilon=1e-5F;c.rope_theta=10000;c.tied=false;c.validate();
    int seed=0;
    auto add=[&](std::string name,tinyinfer::Shape shape,bool norm=false){
        tinyinfer::Tensor t(shape);for(std::size_t i=0;i<t.numel();++i)t.at(i)=norm?1.0F:0.15F*std::sin(float(++seed));
        w->tensors.emplace(name,std::move(t));
    };
    add("model.embed_tokens.weight",{16,8});add("lm_head.weight",{8,16});add("model.norm.weight",{8},true);
    for(int l=0;l<2;++l){auto p="model.layers."+std::to_string(l)+".";
        add(p+"input_layernorm.weight",{8},true);add(p+"post_attention_layernorm.weight",{8},true);
        add(p+"self_attn.q_proj.weight",{8,8});add(p+"self_attn.k_proj.weight",{8,4});
        add(p+"self_attn.v_proj.weight",{8,4});add(p+"self_attn.o_proj.weight",{8,8});
        add(p+"mlp.gate_proj.weight",{8,12});add(p+"mlp.up_proj.weight",{8,12});add(p+"mlp.down_proj.weight",{12,8});
    }
    // Untied head stays finite: only appending this token injects a backend failure.
    for(int i=0;i<8;++i)w->tensors.at("model.embed_tokens.weight").at(15*8+i)=std::numeric_limits<float>::quiet_NaN();
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
int main(int argc,char** argv){
    std::cout<<"COW regression started"<<std::endl;
    bool cuda=false,required=false;for(int i=1;i<argc;++i){cuda|=std::string(argv[i])=="--cuda";required|=std::string(argv[i])=="--require-device";}
#ifdef MINIPREFIX_ENABLE_CUDA
    if(cuda){int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return required?1:77;}
#else
    assert(!cuda);
#endif
    metadata();std::cout<<"COW metadata passed"<<std::endl;
    payload(cuda);payload(cuda,true);std::cout<<"COW payload passed"<<std::endl;
    runner(cuda);std::cout<<"COW copy, capacity, rollback, retry and release passed"<<std::endl;
    partial_cache(cuda);std::cout<<"Partial prefix cache, COW and capacity passed"<<std::endl;
    costs(cuda);std::cout<<"Cost counters and opt-in timing passed"<<std::endl;
}
