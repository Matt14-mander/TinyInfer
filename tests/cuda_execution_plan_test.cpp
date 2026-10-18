#include "tinyinfer/backend/cuda/execution_plan.h"
#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include "tinyinfer/tinyinfer.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
using namespace tinyinfer;
namespace {
void require(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
template<class F> void rejects(F f) { try { f(); } catch(const std::exception&) { return; } throw std::runtime_error("expected rejection"); }
void close(const Tensor& a, const Tensor& b) {
    require(a.shape()==b.shape(), "shape mismatch");
    for(std::size_t i=0;i<a.numel();++i) {
        const auto x=a.at(i), y=b.at(i);
        if(std::isnan(y)) require(std::isnan(x), "NaN mismatch");
        else if(std::isinf(y)) require(x==y, "infinity mismatch");
        else require(std::isfinite(x) && std::abs(x-y)<=1e-5F*(1+std::abs(y)), "numerical mismatch");
        if(x==0 && y==0) require(std::signbit(x)==std::signbit(y), "signed-zero mismatch");
    }
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
        CudaExecutionPlan p(m); CpuExecutionPlan cp(m);
        auto a=p.create_context(), b=p.create_context(); auto c=cp.create_context();
        const auto prepared=p.device_statistics();
        const auto initial=p.statistics(); require(initial.constant_uploads>0,"no constants prepared");
        std::optional<Tensor> saved_a, saved_b;
        for(int round=0;round<8;++round) {
            Tensor x(p.input_spec(p.inputs()[0].second).shape);
            for(std::size_t i=0;i<x.numel();++i) x.at(i)=float(int(i)+round-3)/4;
            auto& ctx=round%2?a:b;
            ctx.bind_input(p.inputs()[0].first,x); rejects([&]{ctx.output(p.outputs()[0].first);});
            c.bind_input(cp.inputs()[0].first,x); c.run(); ctx.run();
            auto expected=c.output(cp.outputs()[0].first); auto out=ctx.output(p.outputs()[0].first);
            if (round%2) {
                saved_a = expected;
                if (saved_b) close(b.output(p.outputs()[0].first), *saved_b);
            } else {
                saved_b = expected;
                if (saved_a) close(a.output(p.outputs()[0].first), *saved_a);
            }
            close(out,expected); out.at(0)=999; close(ctx.output(p.outputs()[0].first),expected);
            const auto before=p.device_statistics();
            ctx.run();
            const auto warm=p.device_statistics();
            require(warm.allocations==before.allocations && warm.uploads==before.uploads &&
                    warm.downloads==before.downloads && warm.copies==before.copies, "warm run allocated or transferred data");
            close(ctx.output(p.outputs()[0].first),expected);
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
class CountingAllocator final : public Allocator {
public:
    std::size_t allocations{0};
    std::string_view name() const noexcept override { return "counting"; }
    void* allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocations; return backing.allocate(bytes, alignment);
    }
    void deallocate(void* p, std::size_t bytes, std::size_t alignment) noexcept override {
        backing.deallocate(p, bytes, alignment);
    }
private: CpuAllocator backing;
};
void fallback_input_ownership() {
    auto model = chain(OpType::GELU);
    CudaExecutionPlan plan(model, {0, FallbackPolicy::WholeModelCpu});
    auto context = plan.create_context();
    CpuExecutionPlan reference(model); auto cpu = reference.create_context();
    auto allocator = std::make_shared<CountingAllocator>();
    Tensor input({2,2}, DataType::Float32, allocator);
    for (std::size_t i=0;i<input.numel();++i) input.at(i)=float(i)-1;
    cpu.bind_input("x", input); cpu.run(); auto expected=cpu.output("y");
    auto before=allocator->allocations;
    context.bind_input("x", input);
    require(allocator->allocations==before+1, "fallback duplicated owned input snapshot");
    input.at(0)=100;
    context.run(); close(context.output("y"),expected);
    before=allocator->allocations;
    input.transpose(0,1);
    cpu.bind_input("x", input); cpu.run(); expected=cpu.output("y");
    before=allocator->allocations;
    context.bind_input("x", std::move(input));
    require(allocator->allocations==before, "moved fallback input copied");
    context.run(); close(context.output("y"),expected);
}
void contracts() {
    auto m=chain(); CudaExecutionPlan p(m); auto c=p.create_context();
    rejects([&]{c.run();}); rejects([&]{c.output("y");}); rejects([&]{c.bind_input("bad",Tensor({2,2}));});
    rejects([&]{c.bind_input("x",Tensor({4}));});
    auto x=Tensor::from_vector({2,2},{-1,2,3,-4});
    c.bind_input("x",x); c.run(); close(c.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    require(c.statistics().reuse_waits>0,"arena reuse did not wait");
    require(p.statistics().activation_bytes<6*x.size_bytes(),"arena not reused");
    auto strided=x; strided.transpose(0,1); rejects([&]{c.bind_input("x",std::move(strided));});
    close(c.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    auto moved=std::move(c); rejects([&]{c.run();}); moved.run();
    auto destination=p.create_context(); destination=std::move(moved);
    rejects([&]{moved.output("y");}); destination.run();
    close(destination.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    rejects([&]{CudaExecutionPlan invalid(m,{0,static_cast<FallbackPolicy>(99)});});
    auto survivor=[] { CudaExecutionPlan local(chain()); auto ctx=local.create_context(); ctx.bind_input("x",Tensor::from_vector({2,2},{1,2,3,4})); return ctx; }();
    survivor.run(); close(survivor.output("y"),Tensor::from_vector({2,2},{1,2,3,4}));
    CudaExecutionPlan fallback(m,{0,FallbackPolicy::WholeModelCpu}); auto f=fallback.create_context();
    auto t=x; t.transpose(0,1); Tensor expected(t); expected.at(0)=0; expected.at(3)=0;
    f.bind_input("x",std::move(t)); f.run(); close(f.output("y"),expected);
    require(f.placement().used_fallback && !f.placement().reason.empty(),"runtime fallback reason missing");
    f.bind_input("x",x); f.run(); require(f.placement().effective.kind==DeviceKind::Cuda,"GPU reentry failed");
    auto unsupported=chain(OpType::GELU);
    rejects([&]{CudaExecutionPlan strict(unsupported);});
    CudaExecutionPlan cpu(unsupported,{0,FallbackPolicy::WholeModelCpu}); auto fc=cpu.create_context();
    fc.bind_input("x",x); fc.run(); require(fc.statistics().cpu_runs==1 && cpu.statistics().constant_uploads==0,"static fallback");
    rejects([&]{CudaExecutionPlan strict(m,{std::numeric_limits<std::uint32_t>::max(),FallbackPolicy::Strict});});
    CudaExecutionPlan unavailable(m,{std::numeric_limits<std::uint32_t>::max(),FallbackPolicy::WholeModelCpu});
    auto u=unavailable.create_context(); u.bind_input("x",x); u.run(); close(u.output("y"),Tensor::from_vector({2,2},{0,2,3,0}));
    require(u.placement().used_fallback && !u.placement().reason.empty(), "unavailable fallback diagnostic missing");
    require(unavailable.device_statistics().allocations==0 && unavailable.kernel_statistics().dispatches==0,
            "CPU placement allocated CUDA model buffers or submitted kernels");
}
void changing_gemm() {
    Graph g;
    auto a=g.add_input("a",{{2,3},DataType::Float32});
    auto b=g.add_input("b",{{3,2},DataType::Float32});
    auto bias=g.add_input("bias",{{2},DataType::Float32});
    auto out=g.node(g.add_node("gemm",OpType::Gemm,{a,b,bias},{{"alpha",-.5F},{"beta",1.5F}})).outputs[0];
    g.mark_output(out);
    Model m(std::move(g),{{"a",a},{"b",b},{"bias",bias}},{{"out",out}});
    CudaExecutionPlan p(m,{0,FallbackPolicy::WholeModelCpu}); CpuExecutionPlan reference(m);
    auto c=p.create_context(), isolated=p.create_context(); auto cpu=reference.create_context();
    const auto allocations=p.device_statistics().allocations;
    std::optional<Tensor> retained;
    for(int round=0;round<5;++round) {
        auto ta=Tensor::from_vector({2,3},{float(round),2,3,4,5,6});
        auto tb=Tensor::from_vector({3,2},{1,float(round),3,4,5,6});
        auto tc=Tensor::from_vector({2},{float(round)-2,3});
        c.bind_input("a",ta); c.bind_input("b",tb); c.bind_input("bias",tc);
        cpu.bind_input("a",ta); cpu.bind_input("b",tb); cpu.bind_input("bias",tc);
        if(round==0) {
            isolated.bind_input("a",ta); isolated.bind_input("b",tb); isolated.bind_input("bias",tc);
            isolated.run(); retained=isolated.output("out");
        }
        // Runtime binding must be an owned snapshot, not a host alias.
        ta.at(0)=999; tb.at(0)=999; tc.at(0)=999;
        c.run(); cpu.run(); close(c.output("out"),cpu.output("out"));
        close(isolated.output("out"),*retained);
        require(p.device_statistics().allocations==allocations, "changed binding allocated device memory");
    }
    // One moved strided RHS selects the whole CPU model, with no GPU dispatch.
    auto ta=Tensor::from_vector({2,3},{1,2,3,4,5,6});
    auto backing=Tensor::from_vector({2,3},{1,2,3,4,5,6}); backing.transpose(0,1);
    auto tc=Tensor::from_vector({2},{2,3});
    cpu.bind_input("a",ta); cpu.bind_input("b",backing); cpu.bind_input("bias",tc); cpu.run();
    c.bind_input("a",ta); c.bind_input("b",std::move(backing)); c.bind_input("bias",tc);
    const auto before=p.kernel_statistics().dispatches;
    c.run(); close(c.output("out"),cpu.output("out"));
    require(c.placement().effective.kind==DeviceKind::Cpu && p.kernel_statistics().dispatches==before,
            "runtime fallback mixed GPU nodes into CPU run");
    c.bind_input("b",Tensor::from_vector({3,2},{1,4,2,5,3,6})); c.run();
    require(c.placement().effective.kind==DeviceKind::Cuda, "dynamic RHS failed GPU reentry");
}
void branches_and_snapshot() {
    Graph g; auto x=g.add_input("x",{{2,2},DataType::Float32});
    auto w=g.add_constant("w",Tensor::from_vector({2,2},{1,2,3,4}));
    auto unused=g.add_constant("unused",Tensor::from_vector({1},{99})); (void)unused;
    auto a=g.node(g.add_node("a",OpType::Add,{x,w})).outputs[0];
    auto b=g.node(g.add_node("b",OpType::Tanh,{a})).outputs[0];
    auto c=g.node(g.add_node("c",OpType::ReLU,{a})).outputs[0];
    auto d=g.node(g.add_node("d",OpType::Add,{b,c})).outputs[0];
    for(auto id:{x,w,a,d}) g.mark_output(id);
    Model m(std::move(g),{{"x",x}},{{"input",x},{"constant",w},{"early",a},{"final",d}});
    CudaExecutionPlan p(m); CpuExecutionPlan reference(m);
    auto copied=p; auto gpu=copied.create_context(); auto cpu=reference.create_context();
    require(p.statistics().constant_uploads==1, "unused constant uploaded");
    static_cast<float*>(m.graph().constant(w).storage().buffer()->data())[0]=999;
    auto input=Tensor::from_vector({2,2},{-2,-1,1,2});
    gpu.bind_input("x",input); cpu.bind_input("x",input); gpu.run(); cpu.run();
    for(const auto& output:p.outputs()) close(gpu.output(output.first),cpu.output(output.first));
    auto preserved=gpu.output("final"); auto expected=cpu.output("final");
    gpu.bind_input("x",Tensor::from_vector({2,2},{10,20,30,40})); gpu.run();
    close(preserved,expected);
    require(p.statistics().constant_uploads==1, "copied plan reuploaded constants");
    rejects([&]{gpu.output(p.input_id("x")+1000);});
    rejects([&]{gpu.bind_input(w,Tensor({2,2}));});
    rejects([&]{p.input_spec(w);});
    require(gpu.statistics().gpu_runs==2 && gpu.statistics().cpu_runs==0, "unexpected branch placement");
}
void endpoint_limits() {
    const auto huge=std::int64_t(std::numeric_limits<std::int32_t>::max())+1;
    for(auto shape:std::vector<Shape>{{0,huge},{1,1,1,1,1}}) {
        Graph g; auto x=g.add_input("x",{shape,DataType::Float32}); g.mark_output(x);
        Model m(std::move(g),{{"x",x}},{{"y",x}});
        rejects([&]{CudaExecutionPlan strict(m);});
        CudaExecutionPlan p(m,{0,FallbackPolicy::WholeModelCpu}); auto c=p.create_context();
        c.bind_input("x",Tensor(shape)); c.run(); require(c.output("y").shape()==shape, "endpoint fallback shape");
        require(c.placement().used_fallback && !c.placement().reason.empty(), "endpoint fallback missing");
    }
}
void zero_k_and_specials() {
    Graph g; auto a=g.add_input("a",{{3,0},DataType::Float32});
    auto b=g.add_constant("b",Tensor({0,2}));
    auto bias=g.add_input("bias",{{2},DataType::Float32});
    auto y=g.node(g.add_node("zero_k",OpType::Gemm,{a,b,bias},{{"alpha",-0.F},{"beta",0.F}})).outputs[0];
    auto z=g.node(g.add_node("fused",OpType::FusedGemmActivation,{a,b,bias},
        {{"activation",std::string("relu")},{"beta",0.F}})).outputs[0];
    g.mark_output(y); g.mark_output(z);
    Model m(std::move(g),{{"a",a},{"bias",bias}},{{"y",y},{"z",z}});
    CudaExecutionPlan p(m); CpuExecutionPlan reference(m); auto c=p.create_context(); auto cpu=reference.create_context();
    const float nan=std::numeric_limits<float>::quiet_NaN();
    for(const auto& input:std::vector<Tensor>{Tensor::from_vector({2},{nan,1}), Tensor::from_vector({2},{-1,1})}) {
        c.bind_input("a",Tensor({3,0})); c.bind_input("bias",input);
        cpu.bind_input("a",Tensor({3,0})); cpu.bind_input("bias",input);
        c.run(); cpu.run(); close(c.output("y"),cpu.output("y")); close(c.output("z"),cpu.output("z"));
    }
    require(p.kernel_statistics().cublas_calls==0, "zero K called cuBLAS");
}
void endpoints() {
    {
        Graph g; auto x=g.add_input("x",{{0,2},DataType::Float32});
        auto y=g.node(g.add_node("empty",OpType::ReLU,{x})).outputs[0]; g.mark_output(y);
        Model m(std::move(g),{{"x",x}},{{"y",y}}); CudaExecutionPlan p(m); auto c=p.create_context();
        c.bind_input("x",Tensor({0,2})); c.run(); require(c.output("y").numel()==0,"empty output");
    }
    {
        Graph g; auto x=g.add_input("x",{{1,1,1,1,1},DataType::Float32}); g.mark_output(x);
        Model m(std::move(g),{{"x",x}},{{"y",x}});
        rejects([&]{CudaExecutionPlan p(m);});
        CudaExecutionPlan p(m,{0,FallbackPolicy::WholeModelCpu}); auto c=p.create_context();
        c.bind_input("x",Tensor::from_vector({1,1,1,1,1},{2})); c.run(); require(c.output("y").at(0)==2,"endpoint fallback");
    }

    {
        Graph g; auto x=g.add_input("x",{{2},DataType::Float32});
        auto w=g.add_constant("w",Tensor::from_vector({2},{3,4}));
        auto y=g.node(g.add_node("add",OpType::Add,{x,w})).outputs[0]; g.mark_output(y);
        Model m(std::move(g),{{"x",x}},{{"y",y}}); CudaExecutionPlan p(m);
        static_cast<float*>(m.graph().constant(w).storage().buffer()->data())[0]=100;
        auto c=p.create_context(); c.bind_input("x",Tensor::from_vector({2},{1,2})); c.run();
        close(c.output("y"),Tensor::from_vector({2},{4,6}));
    }
    for(bool constant : {false,true}) {
        Graph g; auto x=constant?g.add_constant("x",Tensor::from_vector({2},{1,2})):g.add_input("x",{{2},DataType::Float32});
        g.mark_output(x); Model m(std::move(g),constant?std::vector<Model::NamedValue>{}:std::vector<Model::NamedValue>{{"x",x}},{{"y",x}});
        CudaExecutionPlan p(m); auto c=p.create_context();
        if(!constant) { auto backing=Tensor::from_vector({4},{99,1,2,99}); c.bind_input("x",backing.narrow(0,1,2)); }
        c.run(); close(c.output("y"),Tensor::from_vector({2},{1,2}));
    }
    Graph g; auto a=g.add_input("a",{{2,3},DataType::Float32}); auto b=g.add_input("b",{{3,2},DataType::Float32});
    auto out=g.node(g.add_node("mm",OpType::MatMul,{a,b})).outputs[0]; g.mark_output(out);
    Model m(std::move(g),{{"a",a},{"b",b}},{{"out",out}}); CudaExecutionPlan p(m); CpuExecutionPlan cpu(m);
    auto c=p.create_context(); auto ref=cpu.create_context();
    for(int r=0;r<3;++r) {
        auto ta=Tensor::from_vector({2,3},{1,2,3,4,5,6}); auto tb=Tensor::from_vector({3,2},{float(r),2,3,4,5,6});
        c.bind_input("a",ta); c.bind_input("b",tb); ref.bind_input("a",ta); ref.bind_input("b",tb);
        c.run(); ref.run(); close(c.output("out"),ref.output("out"));
    }
}
}
int main(int argc,char**) {
    if(CudaDeviceRuntime::enumerate_devices().empty()) { std::cerr<<"No Cuda device\n"; return argc>1?1:77; }
    try {
        fixtures(); std::cout<<"PASS fixtures: CPU/golden, constant sharing, warm runs, context isolation\n";
        contracts(); std::cout<<"PASS contracts: arena completion waits, moves, lifetime, Strict/whole-model fallback\n";
        endpoints(); endpoint_limits(); std::cout<<"PASS endpoints: empty, identity, constants, offsets, size/rank limits\n";
        changing_gemm(); std::cout<<"PASS changing Gemm: inputs/RHS/bias snapshots, CPU fallback, GPU reentry\n";
        branches_and_snapshot(); std::cout<<"PASS branches: multiple outputs, model/plan/output ownership\n";
        zero_k_and_specials(); std::cout<<"PASS special values: zero K, fused ReLU, NaN bias, signed zero\n";
        fallback_input_ownership(); std::cout<<"PASS fallback ownership: one lvalue snapshot, no moved-input copy\n";
    }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
