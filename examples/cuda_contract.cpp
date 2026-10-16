#include "tinyinfer/backend/cuda/operator_contract.h"
#include "tinyinfer/model/onnx/onnx_importer.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=2){std::cerr<<"Usage: tinyinfer_cuda_contract_inspect model.onnx\n";return 1;}
        tinyinfer::onnx::OnnxImporter importer;auto model=importer.load(argv[1]);
        std::cout<<"Portable CUDA contract inspection only; no GPU execution or placement\n";
        for(auto id:model.graph().topological_order()) {
            const auto& node=model.graph().node(id);
            tinyinfer::CapabilityQuery query{{tinyinfer::DeviceKind::Cuda,0},node,{},{}};
            for(auto value:node.inputs)query.inputs.push_back({model.graph().value(value).spec,std::nullopt});
            for(auto value:node.outputs){auto spec=model.graph().value(value).spec;query.outputs.push_back({spec,tinyinfer::TensorLayout(spec.shape).strides()});}
            auto result=tinyinfer::cuda::validate_operator_contract(query);
            std::cout<<node.name<<": "<<result.reason<<'\n';
            if(!result.supported)return 1;
            if(node.op==tinyinfer::OpType::Gemm || node.op==tinyinfer::OpType::MatMul || node.op==tinyinfer::OpType::FusedGemmActivation) {
                auto d=tinyinfer::cuda::make_gemm_descriptor(query);
                std::cout<<"  prospective column-major call m="<<d.native_m<<", n="<<d.native_n<<", k="<<d.native_k
                         <<", left=B, right=A, lda="<<d.lda<<", ldb="<<d.ldb<<", ldc="<<d.ldc<<'\n';
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
