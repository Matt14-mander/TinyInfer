#include "tinyinfer/tinyinfer.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

namespace {

tinyinfer::onnx::ModelProto make_add_model() {
    using tinyinfer::DataType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;
    using tinyinfer::onnx::Initializer;
    using tinyinfer::onnx::NodeProto;
    using tinyinfer::onnx::ValueInfo;

    tinyinfer::onnx::ModelProto model;
    model.opset_version = 17;
    model.graph.name = "compatibility_graph";
    model.graph.inputs = {
        ValueInfo{"x", TensorSpec{{1, 2}, DataType::Float32}}};
    model.graph.initializers = {
        Initializer{"bias", Tensor::from_vector({2}, {0.5F, -0.5F})}};
    model.graph.nodes = {
        NodeProto{"add_bias", "Add", {"x", "bias"}, {"y"}, {}}};
    model.graph.outputs = {
        ValueInfo{"y", TensorSpec{{1, 2}, DataType::Float32}}};
    return model;
}

template <typename Function>
tinyinfer::onnx::OnnxImportDiagnostic import_error(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const tinyinfer::onnx::OnnxImportError& error) {
        return error.diagnostic();
    }
    assert(false && "expected OnnxImportError");
    return {};
}

}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::TensorSpec;
    using tinyinfer::onnx::OnnxImportStage;
    using tinyinfer::onnx::OnnxImporter;

    OnnxImporter importer;

    // Supported baseline: default domain, opset 17, static FP32 Add.
    const auto valid = importer.import_model(make_add_model());
    assert(valid.graph().size() == 1);

    auto explicit_default_domain = make_add_model();
    explicit_default_domain.graph.nodes[0].domain = "ai.onnx";
    assert(importer.import_model(explicit_default_domain).graph().size() == 1);

    auto old_opset = make_add_model();
    old_opset.opset_version = 11;
    auto diagnostic = import_error(
        [&] { importer.import_model(old_opset); });
    assert(diagnostic.stage == OnnxImportStage::ModelValidation);
    assert(diagnostic.graph_name == "compatibility_graph");

    auto layer_norm = make_add_model();
    layer_norm.graph.initializers.push_back(
        {"scale", tinyinfer::Tensor::from_vector({2}, {1.0F, 1.0F})});
    layer_norm.graph.nodes[0].op_type = "LayerNormalization";
    layer_norm.graph.nodes[0].inputs = {"x", "scale", "bias"};
    layer_norm.opset_version = 16;
    diagnostic = import_error([&] { importer.import_model(layer_norm); });
    assert(diagnostic.stage == OnnxImportStage::OperatorTranslation);
    assert(diagnostic.op_type == "LayerNormalization");
    assert(diagnostic.message.find("opset=16") != std::string::npos);
    layer_norm.opset_version = 17;
    assert(importer.import_model(layer_norm).graph().size() == 1);

    auto gelu = make_add_model();
    gelu.graph.nodes[0].op_type = "Gelu";
    gelu.graph.nodes[0].inputs = {"x"};
    gelu.opset_version = 19;
    diagnostic = import_error([&] { importer.import_model(gelu); });
    assert(diagnostic.stage == OnnxImportStage::OperatorTranslation);
    assert(diagnostic.op_type == "Gelu");
    assert(diagnostic.message.find("opset=19") != std::string::npos);
    gelu.opset_version = 20;
    assert(importer.import_model(gelu).graph().size() == 1);

    auto unsupported_operator = make_add_model();
    unsupported_operator.graph.nodes[0].op_type = "Conv";
    diagnostic = import_error(
        [&] { importer.import_model(unsupported_operator); });
    assert(diagnostic.stage == OnnxImportStage::OperatorTranslation);
    assert(diagnostic.node_index == 0);
    assert(diagnostic.node_name == "add_bias");
    assert(diagnostic.op_type == "Conv");

    auto unsupported_domain = make_add_model();
    unsupported_domain.graph.nodes[0].domain = "com.example";
    diagnostic = import_error(
        [&] { importer.import_model(unsupported_domain); });
    assert(diagnostic.stage == OnnxImportStage::OperatorTranslation);
    assert(diagnostic.domain == "com.example");
    assert(diagnostic.node_name == "add_bias");

    auto invalid_attribute = make_add_model();
    invalid_attribute.graph.nodes[0].attributes.emplace(
        "axis", std::int64_t{0});
    diagnostic = import_error(
        [&] { importer.import_model(invalid_attribute); });
    assert(diagnostic.stage == OnnxImportStage::ShapeInference);
    assert(diagnostic.op_type == "Add");
    assert(diagnostic.message.find("attribute") != std::string::npos);

    auto missing_input = make_add_model();
    missing_input.graph.nodes[0].inputs[1] = "missing_bias";
    diagnostic = import_error(
        [&] { importer.import_model(missing_input); });
    assert(diagnostic.stage == OnnxImportStage::GraphImport);
    assert(diagnostic.node_name == "add_bias");
    assert(diagnostic.value_name == "missing_bias");

    auto optional_input = make_add_model();
    optional_input.graph.nodes[0].inputs[1].clear();
    diagnostic = import_error(
        [&] { importer.import_model(optional_input); });
    assert(diagnostic.stage == OnnxImportStage::GraphImport);
    assert(diagnostic.message.find("optional") != std::string::npos);

    auto multiple_outputs = make_add_model();
    multiple_outputs.graph.nodes[0].outputs.push_back("extra");
    diagnostic = import_error(
        [&] { importer.import_model(multiple_outputs); });
    assert(diagnostic.stage == OnnxImportStage::GraphImport);
    assert(diagnostic.node_name == "add_bias");
    assert(diagnostic.message.find("one non-empty output") !=
           std::string::npos);

    auto output_mismatch = make_add_model();
    output_mismatch.graph.outputs[0].spec =
        TensorSpec{{1, 3}, DataType::Float32};
    diagnostic = import_error(
        [&] { importer.import_model(output_mismatch); });
    assert(diagnostic.stage == OnnxImportStage::GraphValidation);
    assert(diagnostic.value_name == "y");

    auto duplicate_initializer = make_add_model();
    duplicate_initializer.graph.initializers.push_back(
        duplicate_initializer.graph.initializers.front());
    diagnostic = import_error(
        [&] { importer.import_model(duplicate_initializer); });
    assert(diagnostic.stage == OnnxImportStage::GraphImport);
    assert(diagnostic.value_name == "bias");
}
