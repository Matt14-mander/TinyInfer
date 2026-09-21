#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

class InMemoryParser final : public tinyinfer::onnx::ModelParser {
public:
    explicit InMemoryParser(tinyinfer::onnx::ModelProto model)
        : model_(std::move(model)) {}

    tinyinfer::onnx::ModelProto parse(
        const std::filesystem::path& path) const override {
        assert(path == "phase0_mlp.onnx");
        return model_;
    }

private:
    tinyinfer::onnx::ModelProto model_;
};

tinyinfer::onnx::ModelProto make_mlp_model() {
    using tinyinfer::DataType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;
    using tinyinfer::onnx::Initializer;
    using tinyinfer::onnx::NodeProto;
    using tinyinfer::onnx::ValueInfo;

    tinyinfer::onnx::ModelProto model;
    model.opset_version = 17;
    model.graph.inputs = {
        ValueInfo{"x", TensorSpec{{1, 2}, DataType::Float32}},
        ValueInfo{"w1", TensorSpec{{2, 3}, DataType::Float32}}};
    model.graph.initializers = {
        Initializer{"w1", Tensor::from_vector(
                              {2, 3}, {1.0F, 0.0F, -1.0F,
                                       1.0F, -1.0F, 1.0F})},
        Initializer{"b1", Tensor::from_vector(
                              {3}, {0.0F, 0.0F, 1.0F})},
        Initializer{"w2", Tensor::from_vector(
                              {3, 2}, {1.0F, 0.0F, 1.0F,
                                       2.0F, 0.0F, 1.0F})},
        Initializer{"b2", Tensor::from_vector({2}, {0.0F, -1.0F})}};

    // Deliberately not topologically sorted: the importer resolves dependencies.
    model.graph.nodes = {
        NodeProto{"softmax", "Softmax", {"logits"}, {"probabilities"},
                  {{"axis", std::int64_t{-1}}}},
        NodeProto{"matmul1", "MatMul", {"x", "w1"}, {"mm1"}, {}},
        NodeProto{"add1", "Add", {"mm1", "b1"}, {"hidden_pre"}, {}},
        NodeProto{"relu", "Relu", {"hidden_pre"}, {"hidden"}, {}},
        NodeProto{"matmul2", "MatMul", {"hidden", "w2"}, {"mm2"}, {}},
        NodeProto{"add2", "Add", {"mm2", "b2"}, {"logits"}, {}}};
    model.graph.outputs = {
        ValueInfo{"probabilities", TensorSpec{{1, 2}, DataType::Float32}}};
    return model;
}

}  // namespace

int main() {
    using tinyinfer::CpuBackend;
    using tinyinfer::ExecutionContext;
    using tinyinfer::Executor;
    using tinyinfer::ModelLoader;
    using tinyinfer::Tensor;
    using tinyinfer::onnx::OnnxImportError;
    using tinyinfer::onnx::OnnxImporter;

    const auto source_model = make_mlp_model();
    const auto parser = std::make_shared<InMemoryParser>(source_model);
    OnnxImporter importer(parser);
    const ModelLoader& loader = importer;
    const auto model = loader.load("phase0_mlp.onnx");

    assert(model.inputs().size() == 1);
    assert(model.outputs().size() == 1);
    assert(model.graph().size() == 6);
    assert(model.graph().topological_order().size() == 6);
    assert(model.input_id("x") == model.inputs().front().second);
    assert(model.output_id("probabilities") == model.outputs().front().second);
    expect_throw<std::out_of_range>([&] { model.input_id("missing"); });

    ExecutionContext context(model.graph());
    context.bind_input(
        model.input_id("x"),
        Tensor::from_vector({1, 2}, {1.0F, -2.0F}));
    CpuBackend backend;
    Executor executor(backend);
    executor.run(model.graph(), context);

    const auto& probabilities =
        context.output(model.output_id("probabilities"));
    assert(std::fabs(probabilities.at(0) - 0.26894143F) < 1e-5F);
    assert(std::fabs(probabilities.at(1) - 0.73105860F) < 1e-5F);

    OnnxImporter parserless;
    expect_throw<OnnxImportError>([&] { parserless.load("model.onnx"); });

    auto unsupported = make_mlp_model();
    unsupported.graph.nodes[1].op_type = "Conv";
    expect_throw<OnnxImportError>([&] {
        parserless.import_model(unsupported);
    });

    auto old_opset = make_mlp_model();
    old_opset.opset_version = 11;
    expect_throw<OnnxImportError>([&] {
        parserless.import_model(old_opset);
    });

    auto unresolved = make_mlp_model();
    unresolved.graph.nodes[1].inputs[0] = "missing";
    expect_throw<OnnxImportError>([&] {
        parserless.import_model(unresolved);
    });
}
