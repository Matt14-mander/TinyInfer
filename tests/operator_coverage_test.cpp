#include "tinyinfer/tinyinfer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tinyinfer::NodeAttributes;
using tinyinfer::OpType;
using tinyinfer::Shape;
using tinyinfer::Tensor;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Case {
    OpType op;
    const char* schema_name;
    std::vector<Tensor> inputs;
    NodeAttributes attributes;
    Shape expected_shape;
    std::vector<float> expected_values;
    const char* onnx_name;
    std::int64_t first_opset;
};

constexpr auto kOperatorCount = static_cast<std::size_t>(OpType::Count);

std::array<Case, kOperatorCount> cases() {
    return {{
        {OpType::Add, "Add",
         {Tensor::from_vector({2}, {1, 2}),
          Tensor::from_vector({2}, {3, 4})},
         {}, {2}, {4, 6}, "Add", 13},
        {OpType::Multiply, "Multiply",
         {Tensor::from_vector({2}, {1, 2}),
          Tensor::from_vector({2}, {3, 4})},
         {}, {2}, {3, 8}, "Mul", 13},
        {OpType::Subtract, "Subtract",
         {Tensor::from_vector({2}, {1, 2}),
          Tensor::from_vector({2}, {3, 4})},
         {}, {2}, {-2, -2}, "Sub", 13},
        {OpType::MatMul, "MatMul",
         {Tensor::from_vector({2, 2}, {1, 2, 3, 4}),
          Tensor::from_vector({2, 2}, {5, 6, 7, 8})},
         {}, {2, 2}, {19, 22, 43, 50}, "MatMul", 13},
        {OpType::Gemm, "Gemm",
         {Tensor::from_vector({2, 2}, {1, 2, 3, 4}),
          Tensor::from_vector({2, 2}, {5, 6, 7, 8}),
          Tensor::from_vector({2}, {1, 2})},
         {{"alpha", 0.5F}, {"beta", 2.0F}},
         {2, 2}, {11.5F, 15, 23.5F, 29}, "Gemm", 13},
        {OpType::ReLU, "ReLU",
         {Tensor::from_vector({2}, {-1, 2})},
         {}, {2}, {0, 2}, "Relu", 13},
        {OpType::Tanh, "Tanh",
         {Tensor::from_vector({2}, {-1, 1})},
         {}, {2}, {-0.76159416F, 0.76159416F}, "Tanh", 13},
        {OpType::GELU, "GELU",
         {Tensor::from_vector({2}, {-1, 1})},
         {{"approximate", std::string{"none"}}},
         {2}, {-0.15865526F, 0.84134474F}, "Gelu", 20},
        {OpType::Softmax, "Softmax",
         {Tensor::from_vector({2}, {0, std::log(3.0F)})},
         {}, {2}, {0.25F, 0.75F}, "Softmax", 13},
        {OpType::LayerNorm, "LayerNorm",
         {Tensor::from_vector({1, 2}, {1, 3}),
          Tensor::from_vector({2}, {2, 4}),
          Tensor::from_vector({2}, {0.5F, -0.5F})},
         {{"epsilon", 0.0F}},
         {1, 2}, {-1.5F, 3.5F}, "LayerNormalization", 17},
    }};
}

void check_values(const Tensor& actual, const Case& test) {
    require(actual.shape() == test.expected_shape,
            std::string(test.schema_name) + ": numerical output shape mismatch");
    require(actual.numel() == test.expected_values.size(),
            std::string(test.schema_name) + ": numerical output size mismatch");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        require(std::fabs(actual.at(index) - test.expected_values[index]) < 1e-5F,
                std::string(test.schema_name) + ": numerical mismatch at " +
                    std::to_string(index));
    }
}

void check_native(const Case& test) {
    using tinyinfer::DataType;
    using tinyinfer::TensorSpec;

    const auto& schema = tinyinfer::operator_schema(test.op);
    require(schema.op == test.op && schema.name == test.schema_name,
            std::string(test.schema_name) + ": schema missing or mismatched");
    require(test.inputs.size() >= schema.minimum_inputs &&
                test.inputs.size() <= schema.maximum_inputs &&
                schema.output_count == 1,
            std::string(test.schema_name) + ": schema arity mismatch");

    std::vector<TensorSpec> specs;
    tinyinfer::Graph graph;
    std::vector<tinyinfer::ValueId> input_ids;
    for (std::size_t index = 0; index < test.inputs.size(); ++index) {
        const auto& tensor = test.inputs[index];
        specs.push_back({tensor.shape(), tensor.dtype()});
        input_ids.push_back(graph.add_input("input_" + std::to_string(index),
                                      specs.back()));
    }
    const auto outputs = tinyinfer::infer_output_specs(
        test.op, specs, test.attributes);
    require(outputs.size() == schema.output_count &&
                outputs.front() == (TensorSpec{test.expected_shape, DataType::Float32}),
            std::string(test.schema_name) + ": shape inference mismatch");

    tinyinfer::CpuBackend backend;
    require(backend.supports(test.op),
            std::string(test.schema_name) + ": CPU backend missing");
    const auto node_id = graph.add_node("tested_operator", test.op, input_ids,
                                        test.attributes);
    const auto output_id = graph.node(node_id).outputs.front();
    graph.mark_output(output_id);
    tinyinfer::ExecutionContext context(graph);
    for (std::size_t index = 0; index < input_ids.size(); ++index) {
        context.bind_input(input_ids[index], test.inputs[index]);
    }
    tinyinfer::Executor executor(backend);
    executor.run(graph, context);
    check_values(context.output(output_id), test);
}

void check_onnx(const Case& test,
                const tinyinfer::onnx::OperatorRegistry& registry) {
    require(test.onnx_name != nullptr && test.first_opset > 0,
            std::string(test.schema_name) + ": ONNX test metadata missing");
    require(!registry.supports("", test.onnx_name, test.first_opset - 1) &&
                registry.supports("", test.onnx_name, test.first_opset),
            std::string(test.schema_name) + ": ONNX opset boundary mismatch");

    tinyinfer::onnx::ModelProto proto;
    proto.opset_version = test.first_opset;
    proto.graph.name = "coverage";
    tinyinfer::onnx::NodeProto node;
    node.name = "tested_operator";
    node.op_type = test.onnx_name;
    node.outputs = {"output"};
    node.attributes = test.attributes;
    for (std::size_t index = 0; index < test.inputs.size(); ++index) {
        const auto name = "input_" + std::to_string(index);
        node.inputs.push_back(name);
        proto.graph.inputs.push_back(
            {name, {test.inputs[index].shape(), test.inputs[index].dtype()}});
    }
    proto.graph.nodes.push_back(node);
    proto.graph.outputs.push_back(
        {"output", {test.expected_shape, tinyinfer::DataType::Float32}});

    require(registry.translate(node, test.first_opset).op == test.op,
            std::string(test.schema_name) + ": ONNX translation mismatch");
    bool lower_version_rejected = false;
    try {
        static_cast<void>(registry.translate(node, test.first_opset - 1));
    } catch (const std::invalid_argument&) {
        lower_version_rejected = true;
    }
    require(lower_version_rejected,
            std::string(test.schema_name) + ": older ONNX opset was accepted");

    tinyinfer::onnx::OnnxImporter importer;
    const auto model = importer.import_model(proto);
    require(model.graph().node(0).op == test.op,
            std::string(test.schema_name) + ": ONNX import mismatch");
    tinyinfer::ExecutionContext context(model.graph());
    for (std::size_t index = 0; index < test.inputs.size(); ++index) {
        context.bind_input(model.input_id(node.inputs[index]), test.inputs[index]);
    }
    tinyinfer::CpuBackend backend;
    tinyinfer::Executor executor(backend);
    executor.run(model.graph(), context);
    check_values(context.output(model.output_id("output")), test);
}

}  // namespace

int main() {
    const auto all_cases = cases();
    tinyinfer::onnx::OperatorRegistry onnx_registry;
    const auto registrations = onnx_registry.registrations();

    // Every declared ONNX translation must have a matching opset case.
    std::size_t declared_onnx_cases = 0;
    for (const auto& registration : registrations) {
        require(registration.domain.empty(),
                "ONNX coverage needs a case for domain " + registration.domain);
        std::size_t matches = 0;
        for (const auto& test : all_cases) {
            if (test.onnx_name != nullptr &&
                registration.op_type == test.onnx_name &&
                registration.first_opset == test.first_opset) {
                ++matches;
            }
        }
        require(matches == 1,
                "ONNX registration lacks exactly one opset case: " +
                    registration.op_type + " since " +
                    std::to_string(registration.first_opset));
        require(onnx_registry.supports(registration.domain,
                                       registration.op_type,
                                       registration.last_opset),
                "ONNX registration's last opset is not supported: " +
                    registration.op_type);
        if (registration.last_opset <
            std::numeric_limits<std::int64_t>::max()) {
            require(!onnx_registry.supports(registration.domain,
                                            registration.op_type,
                                            registration.last_opset + 1),
                    "ONNX registration accepts an opset past its upper bound: " +
                        registration.op_type);
        }
        ++declared_onnx_cases;
    }

    for (std::size_t index = 0; index < all_cases.size(); ++index) {
        const auto& test = all_cases[index];
        require(static_cast<std::size_t>(test.op) == index,
                "OpType " + std::to_string(index) +
                    " lacks a coverage case in enum order");
        std::cout << "Checking " << test.schema_name << '\n';
        check_native(test);
        if (test.onnx_name != nullptr) {
            check_onnx(test, onnx_registry);
        }
    }
    std::size_t covered_onnx_cases = 0;
    for (const auto& test : all_cases) {
        if (test.onnx_name != nullptr) ++covered_onnx_cases;
    }
    require(declared_onnx_cases == covered_onnx_cases,
            "ONNX coverage case and registry counts differ");
}
