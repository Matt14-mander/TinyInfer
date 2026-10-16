#include "tinyinfer/backend/cuda/operator_contract.h"
#include "tinyinfer/backend/cpu_backend.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tinyinfer::cuda {
namespace {
bool matrix(OpType op) { return op==OpType::MatMul || op==OpType::Gemm || op==OpType::FusedGemmActivation; }
bool initial_operator(OpType op) {
    return matrix(op) || op==OpType::Add || op==OpType::ReLU || op==OpType::Tanh || op==OpType::Softmax;
}
template<class T> T attribute(const NodeAttributes& attrs, const char* name, T fallback) {
    auto found=attrs.find(name); return found==attrs.end()?fallback:std::get<T>(found->second);
}
}
NodeCapability validate_operator_contract(const CapabilityQuery& query) {
    if (query.device.kind!=DeviceKind::Cuda) return {false,false,"CUDA contract requires a CUDA device identity"};
    if (!initial_operator(query.node.op)) return {false,false,"operator outside initial CUDA contract"};
    try {
        auto cpu_query=query; cpu_query.device={};
        auto schema=CpuBackend().query_capability(cpu_query);
        if (!schema.supported) return schema;
        bool runtime_check=false;
        auto validate=[&](const TensorDescriptor& value) -> std::string {
            const TensorLayout layout(value.spec.shape);
            if (value.spec.dtype!=DataType::Float32) return "initial CUDA contract requires FP32";
            if (layout.rank()>4) return "initial CUDA contract supports rank <= 4";
            // A conservative common bound for kernel indices and cuBLAS int APIs.
            if (layout.numel()>std::size_t(std::numeric_limits<std::int32_t>::max()))
                return "CUDA element count exceeds int32 contract";
            for (auto dim:value.spec.shape) if (dim>std::numeric_limits<std::int32_t>::max())
                return "CUDA dimension exceeds int32 contract";
            (void)layout.size_bytes(value.spec.dtype);
            if (!value.strides) runtime_check=true;
            else if (!TensorLayout(value.spec.shape,*value.strides).is_contiguous())
                return "initial CUDA contract requires contiguous physical layout";
            return {};
        };
        for (const auto& value:query.inputs) { auto reason=validate(value);if(!reason.empty())return {false,false,reason}; }
        for (const auto& value:query.outputs) { auto reason=validate(value);if(!reason.empty())return {false,false,reason}; }
        return {true,runtime_check,"portable CUDA contract valid; device, spans and native execution remain unverified"};
    } catch (const std::invalid_argument& error) { return {false,false,error.what()}; }
      catch (const std::out_of_range& error) { return {false,false,error.what()}; }
      catch (const std::overflow_error& error) { return {false,false,error.what()}; }
}
GemmDescriptor make_gemm_descriptor(const CapabilityQuery& query) {
    auto capability=validate_operator_contract(query);
    if (!capability.supported) throw std::invalid_argument(capability.reason);
    if (!matrix(query.node.op)) throw std::invalid_argument("Gemm descriptor requires a matrix operator");
    const bool ta=attribute<std::int64_t>(query.node.attributes,"transA",0)!=0;
    const bool tb=attribute<std::int64_t>(query.node.attributes,"transB",0)!=0;
    const auto& a=query.inputs.at(0).spec.shape;const auto& b=query.inputs.at(1).spec.shape;
    GemmDescriptor d;
    d.rows=static_cast<std::int32_t>(a[ta?1:0]);
    d.reduction=static_cast<std::int32_t>(a[ta?0:1]);
    d.columns=static_cast<std::int32_t>(b[tb?0:1]);
    d.native_m=d.columns; d.native_n=d.rows;d.native_k=d.reduction;
    d.transpose_left=tb;d.transpose_right=ta;
    d.lda=static_cast<std::int32_t>(std::max<std::int64_t>(1,b[1]));
    d.ldb=static_cast<std::int32_t>(std::max<std::int64_t>(1,a[1]));d.ldc=std::max(1,d.columns);
    d.epilogue_alpha=attribute<float>(query.node.attributes,"alpha",1);
    d.epilogue_beta=attribute<float>(query.node.attributes,"beta",1);
    d.has_bias=query.inputs.size()==3;
    d.relu=query.node.op==OpType::FusedGemmActivation;
    if(d.has_bias) {
        const auto& bias=query.inputs[2].spec.shape;
        if(bias.size()==1) d.bias_columns=static_cast<std::int32_t>(bias[0]);
        else if(bias.size()==2) { d.bias_rows=static_cast<std::int32_t>(bias[0]);d.bias_columns=static_cast<std::int32_t>(bias[1]); }
    }
    d.work=!d.rows || !d.columns?GemmWork::EmptyOutput:!d.reduction?GemmWork::ZeroK:GemmWork::MatrixProduct;
    return d;
}
} // namespace tinyinfer::cuda
