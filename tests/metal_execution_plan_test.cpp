#include "tinyinfer/backend/metal/execution_plan.h"
#include "tinyinfer/backend/metal/metal_runtime.h"
#include "tinyinfer/tinyinfer.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
using namespace tinyinfer;
namespace {
void require(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
template<class F> void rejects(F f) { try { f(); } catch(const std::exception&) { return; } throw std::runtime_error("expected rejection"); }
void close(const Tensor& a, const Tensor& b) {
    require(a.shape()==b.shape(), "shape mismatch");
    for(std::size_t i=0;i<a.numel();++i) require(std::isfinite(a.at(i)) && std::abs(a.at(i)-b.at(i))<=1e-5F*(1+std::abs(b.at(i))), "numerical mismatch");
}
Model chain(OpType op=OpType::ReLU) {
    Graph g; auto x=g.add_input("x", {{2,2},DataType::Float32}); auto y=x;
    for(int i=0;i<6;++i) y=g.node(g.add_node("step"+std::to_string(i),op,{y})).outputs[0];
    g.mark_output(y); return Model(std::move(g),{{"x",x}},{{"y",y}});
}
void fixtures() {
    onnx::OnnxImporter importer;
    for(const auto* name : {"phase3_mlp_gemm.onnx","rl_actor_mlp_tanh.onnx"}) {
        auto m=importer.load(std::filesystem::path(TINYINFER_TEST_SOURCE_DIR)/"fixtures"/name);
        MetalExecutionPlan p(m); CpuExecutionPlan cp(m);
        auto a=p.create_context(), b=p.create_context(); auto c=cp.create_context();
        const auto prepared=p.device_statistics();
        const auto initial=p.statistics(); require(initial.constant_uploads>0,"no constants prepared");
        for(int round=0;round<8;++round) {
            Tensor x(p.input_spec(p.inputs()[0].second).shape);
            for(std::size_t i=0;i<x.numel();++i) x.at(i)=float(int(i)+round-3)/4;
            auto& ctx=round%2?a:b;
            ctx.bind_input(p.inputs()[0].first,x); rejects([&]{ctx.output(p.outputs()[0].first);});
            c.bind_input(cp.inputs()[0].first,x); c.run(); ctx.run();
            auto expected=c.output(cp.outputs()[0].first); auto out=ctx.output(p.outputs()[0].first);
            close(out,expected); out.at(0)=999; close(ctx.output(p.outputs()[0].first),expected);
            const auto before=p.device_statistics();
            ctx.run(); close(ctx.output(p.outputs()[0].first),expected);
            require(p.device_statistics().allocations==before.allocations && p.device_statistics().uploads==before.uploads,"warm run allocated or uploaded");
        }
        require(p.statistics().constant_uploads==initial.constant_uploads && p.device_statistics().uploads==prepared.uploads+8,"constant reupload");
        require(a.statistics().gpu_runs==8 && b.statistics().gpu_runs==8,"context run counters");
        // Independent golden reference in addition to CPU comparison.
        const bool rl=std::string(name)=="rl_actor_mlp_tanh.onnx";
        auto x=Tensor::from_vector(p.input_spec(p.inputs()[0].second).shape,
            rl?std::vector<float>{0.25F,-0.5F,1,0.75F}:std::vector<float>{1,-2});
        a.bind_input(p.inputs()[0].second,x); a.run(); auto out=a.output(p.outputs()[0].second);
        close(out,Tensor::from_vector(out.shape(),rl?std::vector<float>{0.58088166F,-0.47540686F}:std::vector<float>{0.26894143F,0.73105860F}));
    }
}
void contracts() {
    auto m=chain(); MetalExecutionPlan p(m); auto c=p.create_context();
    rejects([&]{c.run();}); rejects([&]{c.output("y");}); rejects([&]{c.bind_input("bad",Tensor({2,2}));});
    rejects([&]{c.bind_input("x",Tensor({4}));});
    auto x=Tensor::from_vector({2,2},{-1,2,3,-4});
    c.bind_input("x",x); c.run(); close(c.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    require(c.statistics().reuse_waits>0,"arena reuse did not wait");
    require(p.statistics().activation_bytes<6*x.size_bytes(),"arena not reused");
    auto strided=x; strided.transpose(0,1); rejects([&]{c.bind_input("x",std::move(strided));});
    close(c.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    auto moved=std::move(c); rejects([&]{c.run();}); moved.run();
    auto survivor=[] { MetalExecutionPlan local(chain()); auto ctx=local.create_context(); ctx.bind_input("x",Tensor::from_vector({2,2},{1,2,3,4})); return ctx; }();
    survivor.run(); close(survivor.output("y"),Tensor::from_vector({2,2},{1,2,3,4}));
    MetalExecutionPlan fallback(m,{0,FallbackPolicy::WholeModelCpu}); auto f=fallback.create_context();
    auto t=x; t.transpose(0,1); Tensor expected(t); expected.at(0)=0; expected.at(3)=0;
    f.bind_input("x",std::move(t)); f.run(); close(f.output("y"),expected);
    require(f.placement().used_fallback && !f.placement().reason.empty(),"runtime fallback reason missing");
    f.bind_input("x",x); f.run(); require(f.placement().effective.kind==DeviceKind::Metal,"GPU reentry failed");
    auto unsupported=chain(OpType::GELU);
    rejects([&]{MetalExecutionPlan strict(unsupported);});
    MetalExecutionPlan cpu(unsupported,{0,FallbackPolicy::WholeModelCpu}); auto fc=cpu.create_context();
    fc.bind_input("x",x); fc.run(); require(fc.statistics().cpu_runs==1 && cpu.statistics().constant_uploads==0,"static fallback");
    rejects([&]{MetalExecutionPlan strict(m,{std::numeric_limits<std::uint32_t>::max(),FallbackPolicy::Strict});});
    MetalExecutionPlan unavailable(m,{std::numeric_limits<std::uint32_t>::max(),FallbackPolicy::WholeModelCpu});
    auto u=unavailable.create_context(); u.bind_input("x",x); u.run(); close(u.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
}
void endpoints() {
    {
        Graph g; auto x=g.add_input("x",{{0,2},DataType::Float32});
        auto y=g.node(g.add_node("empty",OpType::ReLU,{x})).outputs[0]; g.mark_output(y);
        Model m(std::move(g),{{"x",x}},{{"y",y}}); MetalExecutionPlan p(m); auto c=p.create_context();
        c.bind_input("x",Tensor({0,2})); c.run(); require(c.output("y").numel()==0,"empty output");
    }
    {
        Graph g; auto x=g.add_input("x",{{1,1,1,1,1},DataType::Float32}); g.mark_output(x);
        Model m(std::move(g),{{"x",x}},{{"y",x}});
        rejects([&]{MetalExecutionPlan p(m);});
        MetalExecutionPlan p(m,{0,FallbackPolicy::WholeModelCpu}); auto c=p.create_context();
        c.bind_input("x",Tensor::from_vector({1,1,1,1,1},{2})); c.run(); require(c.output("y").at(0)==2,"endpoint fallback");
    }

    {
        Graph g; auto x=g.add_input("x",{{2},DataType::Float32});
        auto w=g.add_constant("w",Tensor::from_vector({2},{3,4}));
        auto y=g.node(g.add_node("add",OpType::Add,{x,w})).outputs[0]; g.mark_output(y);
        Model m(std::move(g),{{"x",x}},{{"y",y}}); MetalExecutionPlan p(m);
        static_cast<float*>(m.graph().constant(w).storage().buffer()->data())[0]=100;
        auto c=p.create_context(); c.bind_input("x",Tensor::from_vector({2},{1,2})); c.run();
        close(c.output("y"),Tensor::from_vector({2},{4,6}));
    }
    for(bool constant : {false,true}) {
        Graph g; auto x=constant?g.add_constant("x",Tensor::from_vector({2},{1,2})):g.add_input("x",{{2},DataType::Float32});
        g.mark_output(x); Model m(std::move(g),constant?std::vector<Model::NamedValue>{}:std::vector<Model::NamedValue>{{"x",x}},{{"y",x}});
        MetalExecutionPlan p(m); auto c=p.create_context();
        if(!constant) { auto backing=Tensor::from_vector({4},{99,1,2,99}); c.bind_input("x",backing.narrow(0,1,2)); }
        c.run(); close(c.output("y"),Tensor::from_vector({2},{1,2}));
    }
    Graph g; auto a=g.add_input("a",{{2,3},DataType::Float32}); auto b=g.add_input("b",{{3,2},DataType::Float32});
    auto out=g.node(g.add_node("mm",OpType::MatMul,{a,b})).outputs[0]; g.mark_output(out);
    Model m(std::move(g),{{"a",a},{"b",b}},{{"out",out}}); MetalExecutionPlan p(m); CpuExecutionPlan cpu(m);
    auto c=p.create_context(); auto ref=cpu.create_context();
    for(int r=0;r<3;++r) {
        auto ta=Tensor::from_vector({2,3},{1,2,3,4,5,6}); auto tb=Tensor::from_vector({3,2},{float(r),2,3,4,5,6});
        c.bind_input("a",ta); c.bind_input("b",tb); ref.bind_input("a",ta); ref.bind_input("b",tb);
        c.run(); ref.run(); close(c.output("out"),ref.output("out"));
    }
}
}
int main(int argc,char**) {
    if(MetalDeviceRuntime::enumerate_devices().empty()) { std::cerr<<"No Metal device\n"; return argc>1?1:77; }
    try { fixtures(); contracts(); endpoints(); std::cout<<"Prepared Metal fixtures, reuse, lifetime, fallback and endpoint tests passed\n"; }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
