#include "tinyinfer/backend/cuda/execution_plan.h"
#include "tinyinfer/backend/cpu/execution_plan.h"
#include "tinyinfer/tinyinfer.h"
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace tinyinfer;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
Model model(OpType op) {
    Graph g; auto x=g.add_input("x",{{2,2},DataType::Float32});
    auto w=g.add_constant("w",Tensor::from_vector({2,2},{1,2,3,4}));
    auto a=g.node(g.add_node("add",OpType::Add,{x,w})).outputs[0];
    auto y=g.node(g.add_node("last",op,{a})).outputs[0]; g.mark_output(y);
    return Model(std::move(g),{{"x",x}},{{"y",y}});
}
void check_static_fallback() {
    const auto m=model(OpType::GELU);
    const auto ordinal=std::numeric_limits<std::uint32_t>::max();
    // Both default and invalid ordinal must reject from the static contract,
    // before device discovery. Error priority is intentional and stable.
    for (auto id:{std::uint32_t{0},ordinal}) {
        bool rejected=false;
        try { CudaExecutionPlan strict(m,{id,FallbackPolicy::Strict}); }
        catch(const std::runtime_error& error) {
            rejected=std::string(error.what()).find("unsupported node 'last'")!=std::string::npos;
        }
        require(rejected,"Strict did not reject from static contract");
    }
    CudaExecutionPlan p(m,{ordinal,FallbackPolicy::WholeModelCpu});
    require(p.placement().requested==DeviceId{DeviceKind::Cuda,ordinal} && p.placement().used_fallback,
            "requested device/fallback diagnostic lost");
    require(p.placement().reason.find("unsupported node 'last'")!=std::string::npos,
            "static rejection misreported as unavailable device");
    require(p.placement().nodes.size()==2 && p.placement().nodes[0].capability.supported &&
            !p.placement().nodes[1].capability.supported,"node diagnostics lost");
    auto a=p.create_context(), b=p.create_context();
    CpuExecutionPlan cpu(m); auto ref=cpu.create_context();
    for(int round=0;round<3;++round) {
        auto x=Tensor::from_vector({2,2},{float(round),-1,1,-2});
        auto& context=round%2?a:b;
        context.bind_input("x",x); ref.bind_input("x",x); context.run(); ref.run();
        const auto actual=context.output("y"), expected=ref.output("y");
        for(std::size_t i=0;i<actual.numel();++i)
            require(std::abs(actual.at(i)-expected.at(i))<=1e-5F*(1+std::abs(expected.at(i))),"CPU result differs");
        require(context.statistics().gpu_runs==0 && context.placement().effective==DeviceId{},"mixed execution");
    }
    require(p.device_statistics().allocations==0 && p.statistics().constant_uploads==0 &&
            p.kernel_statistics().dispatches==0,"static fallback used CUDA resources");
    const auto huge=std::int64_t(std::numeric_limits<std::int32_t>::max())+1;
    for(const auto& shape:std::vector<Shape>{{1,1,1,1,1},{0,huge}}) {
        Graph g; auto x=g.add_input("x",{shape,DataType::Float32}); g.mark_output(x);
        Model identity(std::move(g),{{"x",x}},{{"y",x}});
        CudaExecutionPlan plan(identity,{0,FallbackPolicy::WholeModelCpu});
        require(plan.placement().reason.find("value 'x'")!=std::string::npos,"endpoint preflight missed");
        auto c=plan.create_context(); c.bind_input("x",Tensor(shape)); c.run();
        require(c.output("y").shape()==shape,"identity fallback differs");
    }
}
void checked(CUresult result) { if(result!=CUDA_SUCCESS)throw std::runtime_error("driver context probe failed"); }
bool active(CUdevice device) {
    unsigned flags{}; int value{}; checked(cuDevicePrimaryCtxGetState(device,&flags,&value)); return value!=0;
}
}
int main(int argc,char** argv) {
    try {
        bool probe=false, required=false;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(key=="--probe-context")probe=true;
            else if(key=="--require-device")required=true;
            else throw std::invalid_argument("unknown test argument");
        }
        CUdevice device{};
        if(probe) {
            int count{};
            if(cuInit(0)!=CUDA_SUCCESS || cuDeviceGetCount(&count)!=CUDA_SUCCESS || count==0) return required?1:77;
            checked(cuDeviceGet(&device,0)); require(!active(device),"probe process started with active primary context");
        }
        check_static_fallback();
        if(probe) {
            require(!active(device),"static rejection/fallback activated CUDA primary context");
            // Positive control: passing static checks must still take the native
            // path, activate a context and perform real CUDA work.
            CudaExecutionPlan native(model(OpType::ReLU)); auto c=native.create_context();
            c.bind_input("x",Tensor::from_vector({2,2},{-2,-2,0,0})); c.run();
            require(active(device) && c.statistics().gpu_runs==1,"native validation/initialization was bypassed");
            require(c.output("y").at(3)==4,"native positive control differs");
            std::cout<<"PASS: primary context inactive across static fallback; native positive control active\n";
        }
        std::cout<<"PASS: static Strict/CPU fallback, diagnostics, contexts and endpoint limits\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
