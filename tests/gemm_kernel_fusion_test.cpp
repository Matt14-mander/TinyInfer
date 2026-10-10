#include "tinyinfer/tinyinfer.h"
#include "ops/gemm_internal.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>

using namespace tinyinfer;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& fn) { bool ok=false; try { fn(); } catch(const std::exception&) {ok=true;} require(ok,"invalid descriptor accepted"); }
void close(const Tensor& actual,const Tensor& expected) {
    require(actual.shape()==expected.shape(),"shape mismatch");
    for(std::size_t i=0;i<actual.numel();++i) {
        const auto a=actual.at(i), b=expected.at(i);
        if(std::isnan(b)) require(std::isnan(a),"NaN differs");
        else if(std::isinf(b)) require(a==b,"infinity differs");
        else require(std::isfinite(a)&&std::abs(a-b)<=1e-5F*(1+std::abs(b)),"finite output differs");
        if(a==0&&b==0) require(std::signbit(a)==std::signbit(b),"signed zero differs");
    }
}
Tensor values(Shape shape,int seed) { Tensor t(shape); for(std::size_t i=0;i<t.numel();++i) t.at(i)=float(int((i*17+seed)%31)-15)/19; return t; }
void check(const Tensor& a,const Tensor& b,const Tensor* c,float alpha,float beta,bool relu,cpu::MatMulBlockSize block={}) {
    cpu::PackedMatMulRhs packed(b);
    Tensor expected({a.shape()[0],b.shape()[1]});
    cpu::matmul_packed_simd(a,packed,expected,block);
    ops::detail::apply_gemm_epilogue(expected,c,alpha,beta,relu);
    const cpu::GemmEpilogue prototype(expected.shape(),c?&c->shape():nullptr,alpha,beta,relu);
    require(prototype.bias_data()==nullptr,"prototype leaks bias storage");
    const auto bound=prototype.bind(c);
    Tensor fused(expected.shape()), specialized(expected.shape());
    cpu::matmul_packed_gemm(a,packed,fused,bound,block);
    cpu::matmul_packed_simd(a,packed,specialized,block);
    cpu::apply_gemm_epilogue(specialized,bound);
    close(fused,expected); close(specialized,expected);
}
void shapes() {
    const auto width=static_cast<std::int64_t>(cpu::matmul_simd_width());
    for(auto k: {0,1,31,32,33,65,127}) for(auto n: std::vector<std::int64_t>{1,width,2*width+1}) {
        const auto a=values({3,k},1), b=values({k,n},2);
        for(bool relu:{false,true}) {
            check(a,b,nullptr,1,1,relu);
            for(const auto& shape:std::vector<Shape>{{},{n},{1,n},{3,1},{3,n}}) {
                const auto c=values(shape,3); check(a,b,&c,1,1,relu);
                check(a,b,&c,-0.5F,1.5F,relu);
            }
        }
    }
    check(values({0,33},1),values({33,7},2),nullptr,1,1,true);
    check(values({3,33},1),values({33,0},2),nullptr,1,1,true);
    const auto c=values({5},3);
    // Non-standard block column widths force scalar columns inside every tile.
    check(values({3,67},1),values({67,5},2),&c,1,1,true,{2,3,7});
    Tensor a({1,33}),b({33,width+1}),bias({width+1});
    std::fill(a.data<float>(),a.data<float>()+a.numel(),1);
    std::fill(b.data<float>(),b.data<float>()+b.numel(),-1);
    for(std::int64_t col=0;col<width+1;++col) { b.at({32,col})=40; bias.at(col)=3; }
    cpu::PackedMatMulRhs packed(b); Tensor out({1,width+1});
    auto epi=cpu::GemmEpilogue(out.shape(),&bias.shape(),1,1,true).bind(&bias);
    cpu::matmul_packed_gemm(a,packed,out,epi);
    for(std::size_t i=0;i<out.numel();++i) require(out.at(i)==11,"epilogue applied before final K block");
}
void specials() {
    const std::vector<float> special{0,-0.0F,1,-1,std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
    const auto n=static_cast<std::int64_t>(2*cpu::matmul_simd_width()+1);
    for(auto x:special) for(bool relu:{false,true}) {
        Tensor a=Tensor::from_vector({1,1},{x}),b({1,n}),c({n});
        for(std::size_t i=0;i<b.numel();++i) {b.at(i)=special[i%special.size()];c.at(i)=special[(i+2)%special.size()];}
        check(a,b,nullptr,1,1,relu); check(a,b,nullptr,-1,1,relu);
        check(a,b,&c,1,1,relu); check(a,b,&c,0,0,relu);
        check(a,b,&c,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),relu);
    }
}
void strides_and_rebinding() {
    auto backing=values({3,134},1); auto a=backing.slice(1,0,134,2);
    auto b=values({67,17},2);
    auto bias_backing=values({3,34},3); auto c=bias_backing.slice(1,0,34,2);
    check(a,b,&c,1,1,true);
    const auto proto=cpu::GemmEpilogue({3,17},&c.shape(),1,1,true);
    require(!proto.bind(&c).supports_fusion(),"strided columns entered SIMD fusion");
    Graph graph;
    const auto x=graph.add_input("x",{{3,67},DataType::Float32});
    const auto w=graph.add_constant("b",b);
    const auto ci=graph.add_input("c",{{3,17},DataType::Float32});
    const auto node=graph.add_node("g",OpType::FusedGemmActivation,{x,w,ci},{{"activation",std::string("relu")}});
    const auto out=graph.node(node).outputs[0];graph.mark_output(out);
    Model model(std::move(graph),{{"x",x},{"c",ci}},{{"y",out}});
    CpuExecutionPlan plan(model, {CpuGemmEpilogueMode::Fused}), legacy(model,{CpuGemmEpilogueMode::Legacy});
    auto session=plan.create_context(), reference=legacy.create_context();
    for(int i=0;i<4;++i) {
        auto storage=values({3,34},i+3); auto view=storage.slice(1,0,34,2);
        auto input=values({3,67},i+1);
        reference.bind_input("x",input); reference.bind_input("c",view);
        session.bind_input("x",input);
        if(i%2==0) session.bind_input("c",std::move(view));
        else session.bind_input("c",view); // Copy makes the same logical bias contiguous.
        reference.run(); session.run();close(session.output("y"),reference.output("y"));
    }
    require(session.fused_gemm_count()==2&&session.specialized_gemm_count()==2,"runtime bias layout was cached or ignored");
    require(session.runtime_pack_count()==0,"constant RHS repacked");
}
void invalid() {
    Shape wrong{2,7}; rejects([&]{cpu::GemmEpilogue({3,5},&wrong,1,1,false);});
    const Shape shape{5}; auto proto=cpu::GemmEpilogue({3,5},&shape,1,1,false);
    Tensor output({3,5}); rejects([&]{proto.validate(output);});
    auto c=values({7},1); rejects([&]{proto.bind(&c);});
    auto a=values({3,2},1), b=values({2,5},2), valid=values({5},3);
    cpu::PackedMatMulRhs packed(b); auto epi=proto.bind(&valid);
    rejects([&]{cpu::matmul_packed_gemm(a,packed,output,epi,{0,3,4});});
}
void no_bias_fast_paths() {
    const auto n=static_cast<std::int64_t>(2*cpu::matmul_simd_width()+1);
    const Shape shape{3,n};
    const std::vector<float> special{0,-0.0F,1,-1,
        std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    for(float alpha : {1.0F,0.0F,-0.0F,-1.0F,
                       std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()}) {
        for(bool relu : {false,true}) {
            Tensor original(shape);
            for(std::size_t i=0;i<original.numel();++i) original.at(i)=special[i%special.size()];
            Tensor expected(original),actual(original);
            const auto epi=cpu::GemmEpilogue(shape,nullptr,alpha,
                std::numeric_limits<float>::quiet_NaN(),relu).bind(nullptr);
            require(epi.is_identity()==(alpha==1.0F&&!relu),"identity classification differs");
            require(epi.supports_fusion()!=epi.is_identity(),"no-bias fusion eligibility differs");
            ops::detail::apply_gemm_epilogue(expected,nullptr,alpha,epi.beta(),relu);
            cpu::apply_gemm_epilogue(actual,epi);
            close(actual,expected);
            // Exercise zero-K, multiple K blocks, and scalar columns per tile.
            for(auto k : {0,33,67}) check(values({3,k},1),values({k,n},2),nullptr,
                                        alpha,epi.beta(),relu,{2,3,7});
        }
    }
    const auto identity=cpu::GemmEpilogue(shape,nullptr,1,1,false).bind(nullptr);
    Tensor wrong({1,n}); rejects([&]{cpu::apply_gemm_epilogue(wrong,identity);});
    const auto a=values({3,33},1),b=values({33,n},2);
    cpu::PackedMatMulRhs packed(b); Tensor output(shape);
    rejects([&]{cpu::matmul_packed_gemm(a,packed,output,identity,{0,3,7});});
    rejects([&]{cpu::matmul_packed_gemm(a,packed,wrong,identity);});

    Graph graph;
    const auto x=graph.add_input("x",{{3,33},DataType::Float32});
    const auto w=graph.add_constant("w",b);
    const auto node=graph.add_node("identity_epilogue",OpType::Gemm,{x,w});
    const auto out=graph.node(node).outputs[0]; graph.mark_output(out);
    Model model(std::move(graph),{{"x",x}},{{"y",out}});
    CpuExecutionPlan fused(model,{CpuGemmEpilogueMode::Fused}),
                     legacy(model,{CpuGemmEpilogueMode::Legacy});
    auto session=fused.create_context(),reference=legacy.create_context();
    session.bind_input("x",a); reference.bind_input("x",a);
    session.run(); reference.run(); close(session.output("y"),reference.output("y"));
    require(session.fused_gemm_count()==0,"identity epilogue counted as register fusion");
    require(session.specialized_gemm_count()==1,"identity fallback dispatch not recorded");
    require(session.runtime_pack_count()==0,"identity fast path repacked constant RHS");
}
int main(){
    try { shapes(); specials(); strides_and_rebinding(); invalid(); no_bias_fast_paths(); }
    catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
