#include "tinyinfer/backend/cuda/operator_contract.h"
#include "tinyinfer/tinyinfer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
using namespace tinyinfer;
namespace {
std::size_t cases=0;
void require(bool b,const char* text) { if(!b)throw std::runtime_error(text); }
template<class F> void rejects(F f) { try{f();}catch(const std::exception&){return;}throw std::runtime_error("invalid descriptor accepted"); }
Tensor values(Shape shape,int seed) { Tensor t(shape);for(std::size_t i=0;i<t.numel();++i)t.at(i)=float(int((i*7+seed)%19)-9)/16;return t; }
CapabilityQuery query(OpType op,const std::vector<Tensor>& inputs,NodeAttributes attrs={}) {
    Node n;n.op=op;n.attributes=std::move(attrs);n.inputs.resize(inputs.size());n.outputs={inputs.size()};
    CapabilityQuery q{{DeviceKind::Cuda,0},n,{},{}};std::vector<TensorSpec> specs;
    for(const auto& t:inputs){specs.push_back({t.shape(),t.dtype()});q.inputs.push_back({specs.back(),t.strides()});}
    for(auto spec:infer_output_specs(op,specs,n.attributes))q.outputs.push_back({spec,TensorLayout(spec.shape).strides()});
    return q;
}
// Independent CPU interpreter of the *column-major vendor call*, not CUDA
// execution. Native element access uses cuBLAS leading dimensions and N/T flags.
Tensor interpret(const cuda::GemmDescriptor& d,const std::vector<Tensor>& inputs) {
    Tensor out({d.rows,d.columns});
    auto column_value=[](const Tensor& t,int row,int column,int leading,bool transpose) {
        return t.at(std::size_t(transpose?column:row)+std::size_t(transpose?row:column)*leading);
    };
    if(d.work==cuda::GemmWork::MatrixProduct) for(int j=0;j<d.native_n;++j)for(int i=0;i<d.native_m;++i) {
        float sum=0;
        for(int k=0;k<d.native_k;++k)sum+=column_value(inputs[d.left_input],i,k,d.lda,d.transpose_left)*column_value(inputs[d.right_input],k,j,d.ldb,d.transpose_right);
        out.at(std::size_t(i)+std::size_t(j)*d.ldc)=d.product_alpha*sum;
    }
    for(int r=0;r<d.rows;++r)for(int c=0;c<d.columns;++c) {
        const auto index=std::size_t(r)*d.columns+c;
        float v=d.epilogue_alpha*out.at(index);
        if(d.has_bias)v+=d.epilogue_beta*inputs[2].at(std::size_t(d.bias_rows==1?0:r)*d.bias_columns+(d.bias_columns==1?0:c));
        if(d.relu)v=std::max(0.0F,v);
        out.at(index)=v;
    }
    return out;
}
void check(OpType op,const std::vector<Tensor>& inputs,NodeAttributes attrs={}) {
    auto q=query(op,inputs,attrs);auto d=cuda::make_gemm_descriptor(q);
    require(d.product_alpha==1 && d.product_beta==0,"vendor alpha/beta changed");
    auto actual=interpret(d,inputs);
    Graph g;std::vector<ValueId> ids;for(std::size_t i=0;i<inputs.size();++i)ids.push_back(g.add_constant("input"+std::to_string(i),inputs[i]));
    auto output=g.node(g.add_node("op",op,ids,attrs)).outputs[0];g.mark_output(output);
    ExecutionContext ctx(g);CpuBackend cpu;Executor executor(cpu);executor.run(g,ctx);const auto& expected=ctx.output(output);
    require(actual.shape()==expected.shape(),"lowering shape differs");
    for(std::size_t i=0;i<actual.numel();++i) {
        const auto a=actual.at(i),b=expected.at(i);
        if(std::isnan(b))require(std::isnan(a),"NaN epilogue lost");
        else if(std::isinf(b))require(a==b,"infinity epilogue differs");
        else require(std::isfinite(a)&&std::abs(a-b)<=1e-5F*(1+std::abs(b)),"column-major result differs");
        if(a==0 && b==0)require(std::signbit(a)==std::signbit(b),"signed-zero differs");
    }
    ++cases;
}
void mapping() {
    for(bool ta:{false,true})for(bool tb:{false,true})for(int k:{0,1,7})for(float alpha:{1.F,-.5F,0.F,-0.F}) {
        auto a=values(ta?Shape{k,3}:Shape{3,k},1),b=values(tb?Shape{5,k}:Shape{k,5},3);
        for(auto op:{OpType::Gemm,OpType::FusedGemmActivation}) {
            NodeAttributes attrs{{"transA",std::int64_t{ta}},{"transB",std::int64_t{tb}},{"alpha",alpha},{"beta",-.75F}};
            if(op==OpType::FusedGemmActivation)attrs["activation"]=std::string("relu");
            check(op,{a,b},attrs);
            for(auto shape:std::vector<Shape>{{},{1},{1,1},{5},{1,5},{3,1},{3,5}})check(op,{a,b,values(shape,5)},attrs);
        }
    }
    check(OpType::MatMul,{values({3,7},1),values({7,5},2)});
    check(OpType::MatMul,{values({0,7},1),values({7,5},2)});
    check(OpType::Gemm,{values({3,7},1),values({7,0},2)});
    const float nan=std::numeric_limits<float>::quiet_NaN();
    check(OpType::Gemm,{Tensor::from_vector({1,1},{nan}),Tensor::from_vector({1,1},{1})},{{"alpha",0.F}});
    check(OpType::Gemm,{values({1,1},1),values({1,1},2),Tensor::from_vector({}, {nan})},{{"beta",0.F}});
}
void rejection() {
    require(cuda::make_gemm_descriptor(query(OpType::MatMul,{values({3,0},1),values({0,5},2)})).work==cuda::GemmWork::ZeroK,"zero K must bypass vendor product");
    require(cuda::make_gemm_descriptor(query(OpType::MatMul,{values({0,7},1),values({7,5},2)})).work==cuda::GemmWork::EmptyOutput,"empty output dispatch was not skipped");
    require(cuda::make_gemm_descriptor(query(OpType::MatMul,{values({3,7},1),values({7,5},2)})).work==cuda::GemmWork::MatrixProduct,"ordinary matrix product skipped");
    auto add=query(OpType::Add,{values({2,1,3,1},1),values({1,4,1,5},2)});
    require(cuda::validate_operator_contract(add).supported,"rank four broadcast rejected");
    auto softmax=query(OpType::Softmax,{values({2,3},1)},{{"axis",std::int64_t{-1}}});
    require(cuda::validate_operator_contract(softmax).supported,"softmax axis rejected");
    softmax.node.attributes["axis"]=std::int64_t{2};
    require(!cuda::validate_operator_contract(softmax).supported,"out-of-range softmax axis accepted");

    const auto base=query(OpType::Gemm,{values({3,7},1),values({7,5},2)});
    auto q=base;require(cuda::validate_operator_contract(q).supported,"valid matrix rejected");
    q.inputs[0].strides.reset();require(cuda::validate_operator_contract(q).requires_runtime_layout_check,"runtime check missing");
    q=base;q.device={};require(!cuda::validate_operator_contract(q).supported,"foreign device accepted");
    q=base;q.inputs[0].strides=Strides{14,2};rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.inputs[0].spec.dtype=DataType::Float16;rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.node.attributes["alpha"]=std::int64_t{1};rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.node.attributes["transB"]=std::int64_t{2};rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.outputs[0].spec.shape={3,6};rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.node.inputs.clear();rejects([&]{cuda::make_gemm_descriptor(q);});
    q=base;q.node.op=OpType::GELU;require(!cuda::validate_operator_contract(q).supported,"unplanned operator accepted");
    auto unary=query(OpType::ReLU,{values({2,3},1)});require(cuda::validate_operator_contract(unary).supported,"relu contract rejected");
    rejects([&]{cuda::make_gemm_descriptor(unary);});
    unary=query(OpType::ReLU,{values({1,1,1,1,1},1)});require(!cuda::validate_operator_contract(unary).supported,"rank limit missing");
    const auto large=std::int64_t(std::numeric_limits<std::int32_t>::max())+1;
    unary.node.inputs={0};unary.inputs={{{{0,large},DataType::Float32},std::nullopt}};unary.outputs=unary.inputs;
    require(!cuda::validate_operator_contract(unary).supported,"zero-sized oversized dimension narrowed");
    unary.inputs={{{{46341,46341},DataType::Float32},std::nullopt}};unary.outputs=unary.inputs;
    require(!cuda::validate_operator_contract(unary).supported,"oversized element count accepted");
}
void fixtures() {
    onnx::OnnxImporter importer;
    for(auto name:{"phase3_mlp_gemm.onnx","rl_actor_mlp_tanh.onnx"}) {
        auto m=importer.load(std::filesystem::path(TINYINFER_TEST_SOURCE_DIR)/"fixtures"/name);
        for(const auto& node:m.graph().nodes()) {
            CapabilityQuery q{{DeviceKind::Cuda,0},node,{},{}};
            for(auto id:node.inputs)q.inputs.push_back({m.graph().value(id).spec,std::nullopt});
            for(auto id:node.outputs){auto spec=m.graph().value(id).spec;q.outputs.push_back({spec,TensorLayout(spec.shape).strides()});}
            require(cuda::validate_operator_contract(q).supported,"fixture exceeds contract");
            if(node.op==OpType::Gemm || node.op==OpType::MatMul || node.op==OpType::FusedGemmActivation)(void)cuda::make_gemm_descriptor(q);
        }
    }
}
}
int main() {try{mapping();rejection();fixtures();std::cout<<cases<<" column-major CPU interpretation cases passed; CUDA GPU execution NOT tested\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
