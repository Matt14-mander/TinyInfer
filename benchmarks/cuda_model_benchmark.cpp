#include "tinyinfer/backend/cuda/execution_plan.h"
#include "tinyinfer/backend/cuda/cuda_runtime.h"
#include "tinyinfer/tinyinfer.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <cuda_runtime_api.h>
using namespace tinyinfer;
namespace {
using Clock=std::chrono::steady_clock;
using Times=std::map<std::string,std::vector<double>>;
volatile float sink=0;

PROCESS_MEMORY_COUNTERS_EX process_memory() {
    PROCESS_MEMORY_COUNTERS_EX p{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&p), sizeof(p)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    return p;
}
void memory_point(const char* stage) {
    std::size_t free_bytes{},total{};
    if(cudaMemGetInfo(&free_bytes,&total)!=cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
    const auto p=process_memory();
    std::cout<<"{\"stage\":\""<<stage<<"\",\"device_free_bytes\":"<<free_bytes<<",\"device_total_bytes\":"<<total
             <<",\"working_set_bytes\":"<<p.WorkingSetSize<<",\"peak_working_set_bytes\":"<<p.PeakWorkingSetSize
             <<",\"private_usage_bytes\":"<<p.PrivateUsage<<"}";
}

template<class F> double timed(F f) { auto start=Clock::now(); f(); return std::chrono::duration<double,std::micro>(Clock::now()-start).count(); }
void check(const Tensor& actual,const Tensor& expected) {
    if(actual.shape()!=expected.shape()) throw std::runtime_error("shape differs");
    for(std::size_t i=0;i<actual.numel();++i)
        if(!std::isfinite(actual.at(i)) || std::abs(actual.at(i)-expected.at(i))>1e-5F*(1+std::abs(expected.at(i))))
            throw std::runtime_error("FP32 correctness gate failed at "+std::to_string(i));
}
Tensor values(Shape shape, int seed) {
    Tensor t(shape);
    for(std::size_t i=0;i<t.numel();++i) t.at(i)=float(int((i*17+seed)%31)-15)/128;
    return t;
}
Model mlp(std::int64_t m,std::int64_t k,std::int64_t n) {
    Graph g; auto a=g.add_input("input",{{m,k},DataType::Float32});
    auto b=g.add_constant("weight",values({k,n},3)); auto c=g.add_constant("bias",values({n},5));
    auto out=g.node(g.add_node("gemm_relu",OpType::FusedGemmActivation,{a,b,c},{{"activation",std::string("relu")}})).outputs[0];
    g.mark_output(out); return Model(std::move(g),{{"input",a}},{{"output",out}});
}
Model fallback() {
    Graph g; auto a=g.add_input("input",{{64,512},DataType::Float32});
    auto out=g.node(g.add_node("gelu",OpType::GELU,{a})).outputs[0];g.mark_output(out);
    return Model(std::move(g),{{"input",a}},{{"output",out}});
}
template<class Context> void inference(Context& ctx, ValueId in, ValueId out, const Tensor& x) {
    ctx.bind_input(in,x); ctx.run(); auto y=ctx.output(out); if(y.numel()) sink=y.at(0);
}
template<class Context> void diagnostic(Context& ctx,ValueId in,ValueId out,const Tensor& x,Times& times) {
    times["bind_host_input_us"].push_back(timed([&]{ctx.bind_input(in,x);}));
    times["resident_run_completed_us"].push_back(timed([&]{ctx.run();}));
    Tensor y;
    times["copy_host_output_us"].push_back(timed([&]{y=ctx.output(out);}));
    if(y.numel()) sink=y.at(0);
}
void array(const std::vector<double>& v) { std::cout<<'[';for(std::size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']'; }
void emit(const char* backend,const Times& times,std::size_t weights,std::size_t arena,std::size_t allocated,std::size_t uploads,std::size_t downloads,std::size_t rss,bool gpu) {
    std::cout<<"{\"backend\":\""<<backend<<"\",\"effective\":\""<<(gpu?"cuda":"cpu")<<"\",\"weight_bytes\":"<<weights
             <<",\"activation_bytes\":"<<arena<<",\"logical_device_allocated_bytes\":"<<allocated
             <<",\"upload_requests\":"<<uploads<<",\"download_requests\":"<<downloads<<",\"process_peak_working_set_bytes\":"<<rss<<",\"timings\":{";
    bool comma=false;for(const auto& entry:times){if(comma)std::cout<<',';comma=true;std::cout<<'"'<<entry.first<<"\":";array(entry.second);}std::cout<<"}}";
}
void benchmark(const std::string& name,int samples,int warmup,bool gpu_first) {
    std::unique_ptr<Model> model;
    auto load=timed([&]{
        if(name=="target") model=std::make_unique<Model>(mlp(128,1024,1024));
        else if(name=="control") model=std::make_unique<Model>(mlp(64,512,512));
        else if(name=="fallback") model=std::make_unique<Model>(fallback());
        else { onnx::OnnxImporter importer;model=std::make_unique<Model>(importer.load(std::string(TINYINFER_PROJECT_SOURCE_DIR)+"/tests/fixtures/"+name+".onnx")); }
    });
    Times ct,gt;std::unique_ptr<CpuExecutionPlan> cpu;std::unique_ptr<CudaExecutionPlan> gpu;
    auto prepare_cpu=[&]{ct["prepare_us"].push_back(timed([&]{cpu=std::make_unique<CpuExecutionPlan>(*model);}));};
    auto prepare_gpu=[&]{gt["prepare_us"].push_back(timed([&]{gpu=std::make_unique<CudaExecutionPlan>(*model,CudaExecutionPlanOptions{0,name=="fallback"?FallbackPolicy::WholeModelCpu:FallbackPolicy::Strict});}));};
    if(gpu_first){prepare_gpu();prepare_cpu();}else{prepare_cpu();prepare_gpu();}
    std::unique_ptr<CpuExecutionContext> cc;std::unique_ptr<CudaExecutionContext> gc;
    ct["context_create_us"].push_back(timed([&]{cc=std::make_unique<CpuExecutionContext>(cpu->create_context());}));
    gt["context_create_us"].push_back(timed([&]{gc=std::make_unique<CudaExecutionContext>(gpu->create_context());}));
    auto in=model->inputs()[0].second,out=model->outputs()[0].second;
    auto x=values(model->graph().value(in).spec.shape,7), alternate=values(x.shape(),11);
    auto first_cpu=[&]{ct["first_inference_us"].push_back(timed([&]{inference(*cc,in,out,x);}));};
    auto first_gpu=[&]{gt["first_inference_us"].push_back(timed([&]{inference(*gc,in,out,x);}));};
    if(gpu_first){first_gpu();first_cpu();}else{first_cpu();first_gpu();}
    check(gc->output(out),cc->output(out));
    for(int i=0;i<warmup;++i) { if(((i%2)!=0)==gpu_first){inference(*gc,in,out,x);inference(*cc,in,out,x);}else{inference(*cc,in,out,x);inference(*gc,in,out,x);} }
    const auto before=gpu->device_statistics();
    const auto stage_before=gpu->transfer_statistics();
    for(int i=0;i<samples;++i) {
        const auto& input=i%2?alternate:x;
        auto c=[&]{ct["warm_e2e_us"].push_back(timed([&]{inference(*cc,in,out,input);}));};
        auto g=[&]{gt["warm_e2e_us"].push_back(timed([&]{inference(*gc,in,out,input);}));};
        if(((i%2)!=0)==gpu_first){g();c();}else{c();g();}
        check(gc->output(out),cc->output(out));
    }
    // Independent diagnostic samples: never sum these to claim E2E latency.
    for(int i=0;i<samples;++i) {
        if(((i%2)!=0)==gpu_first){diagnostic(*gc,in,out,x,gt);diagnostic(*cc,in,out,x,ct);}
        else{diagnostic(*cc,in,out,x,ct);diagnostic(*gc,in,out,x,gt);}
    }
    cc->bind_input(in,x);gc->bind_input(in,x);
    auto resident_before=gpu->device_statistics();
    for(int i=0;i<samples;++i) {
        auto c=[&]{ct["resident_only_us"].push_back(timed([&]{cc->run();}));};
        auto g=[&]{gt["resident_only_us"].push_back(timed([&]{gc->run();}));};
        if(((i%2)!=0)==gpu_first){g();c();}else{c();g();}
    }
    const auto after=gpu->device_statistics();
    if(after.allocations!=before.allocations || after.uploads!=resident_before.uploads || after.downloads!=resident_before.downloads || after.copies!=resident_before.copies)
        throw std::runtime_error("resident run allocated or transferred");
    check(gc->output(out),cc->output(out));
    const bool actual_gpu=gc->placement().effective.kind==DeviceKind::Cuda;
    if(actual_gpu!=(name!="fallback")) throw std::runtime_error("unexpected execution placement");
    if(gpu->transfer_statistics().staging_allocations!=stage_before.staging_allocations)
        throw std::runtime_error("warm transfer allocated another native staging buffer");
    auto stats=gpu->statistics();
    const auto usage=process_memory();
    std::cout<<"{\"workload\":\""<<name<<"\",\"model_load_or_construct_us\":"<<load<<",\"correctness\":true,\"results\":[";
    emit("cpu_auto",ct,cpu->packed_weight_bytes(),cpu->activation_bytes(),0,0,0,usage.PeakWorkingSetSize,false);std::cout<<',';
    emit("cuda",gt,stats.constant_bytes,stats.activation_bytes,after.allocated_bytes,after.uploads,after.downloads,usage.PeakWorkingSetSize,actual_gpu);
    auto transfer=gpu->transfer_statistics();
    const auto kernels=gpu->kernel_statistics();
    std::cout<<"],\"staging_allocations\":"<<transfer.staging_allocations<<",\"staging_reuses\":"<<transfer.staging_reuses<<",\"staging_cached_bytes\":"<<transfer.cached_bytes
             <<",\"logical_allocations\":"<<after.allocations<<",\"uploaded_bytes\":"<<after.uploaded_bytes
             <<",\"downloaded_bytes\":"<<after.downloaded_bytes<<",\"dispatches\":"<<kernels.dispatches
             <<",\"cublas_calls\":"<<kernels.cublas_calls<<",\"kernel_launches\":"<<kernels.kernel_launches<<"}";
}
void memory_benchmark() {
    if(cudaSetDevice(0)!=cudaSuccess || cudaFree(nullptr)!=cudaSuccess)
        throw std::runtime_error("CUDA context initialization failed");
    std::cout<<"{\"schema\":1,\"memory_only\":true,\"cases\":[";
    bool comma=false;
    for(const auto& name:std::vector<std::string>{"phase3_mlp_gemm","rl_actor_mlp_tanh","control","target","fallback"}) {
        if(comma) std::cout<<','; comma=true;
        std::cout<<"{\"workload\":\""<<name<<"\",\"snapshots\":[";
        memory_point("initialized_baseline");
        {
            std::unique_ptr<Model> model;
            if(name=="target") model=std::make_unique<Model>(mlp(128,1024,1024));
            else if(name=="control") model=std::make_unique<Model>(mlp(64,512,512));
            else if(name=="fallback") model=std::make_unique<Model>(fallback());
            else { onnx::OnnxImporter importer; model=std::make_unique<Model>(importer.load(std::string(TINYINFER_PROJECT_SOURCE_DIR)+"/tests/fixtures/"+name+".onnx")); }
            CudaExecutionPlan p(*model,{0,name=="fallback"?FallbackPolicy::WholeModelCpu:FallbackPolicy::Strict});
            std::cout<<',';memory_point("prepared");
            {
                auto first=p.create_context();
                std::cout<<',';memory_point("one_context");
                auto input=values(p.input_spec(p.inputs()[0].second).shape,7);
                inference(first,p.inputs()[0].second,p.outputs()[0].second,input);
                std::cout<<',';memory_point("first_completed");
                {
                    auto second=p.create_context();
                    inference(second,p.inputs()[0].second,p.outputs()[0].second,input);
                    std::cout<<',';memory_point("two_contexts_completed");
                }
            }
            std::cout<<',';memory_point("contexts_destroyed");
        }
        std::cout<<',';memory_point("plan_destroyed");
        std::cout<<"]}";
    }
    std::cout<<"]}\n";
}

}
int main(int argc,char** argv) {
    try {
        int samples=31,warmup=5;bool gpu_first=false,memory_only=false;std::string workload;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i]; if(key=="--gpu-first")gpu_first=true;
            else if(key=="--memory-only")memory_only=true;
            else if(key=="--workload" && i+1<argc)workload=argv[++i];
            else if((key=="--samples" || key=="--warmup") && i+1<argc) {
                std::string text=argv[++i];std::size_t end=0;int value=std::stoi(text,&end);
                if(end!=text.size() || value<1 || value>10000)throw std::invalid_argument("invalid count");
                if(key=="--samples")samples=value;else warmup=value;
            }else throw std::invalid_argument("unknown argument");
        }
        if(memory_only) { memory_benchmark(); return 0; }
        auto devices=CudaDeviceRuntime::enumerate_devices();if(devices.empty())throw std::runtime_error("actual Cuda GPU required");
        if(std::string(TINYINFER_BUILD_TYPE)!="Release" || !TINYINFER_NATIVE_ARCH)throw std::runtime_error("Release native CPU baseline required");
        std::cout<<std::fixed<<std::setprecision(6)<<"{\"schema\":1,\"samples\":"<<samples<<",\"warmup\":"<<warmup
                 <<",\"profiling\":"<<"false"<<",\"gpu_first\":"<<(gpu_first?"true":"false")<<",\"cases\":[";
        std::vector<std::string> cases{"phase3_mlp_gemm","rl_actor_mlp_tanh","control","target","fallback"};
        if(!workload.empty()) {
            if(std::find(cases.begin(),cases.end(),workload)==cases.end())throw std::invalid_argument("unknown workload");
            cases={workload};
        }
        if(gpu_first)std::reverse(cases.begin(),cases.end());
        for(std::size_t i=0;i<cases.size();++i){if(i)std::cout<<',';benchmark(cases[i],samples,warmup,gpu_first);}
        std::cout<<"]}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
