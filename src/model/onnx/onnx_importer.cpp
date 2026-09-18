#include "tinyinfer/model/onnx/onnx_importer.h"

#include "tinyinfer/model/onnx/protobuf_model_parser.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinyinfer::onnx {
namespace {

std::string node_name(std::size_t index,
                      const std::unordered_set<std::string>& reserved) {
    auto name = std::string("__onnx_node_") + std::to_string(index);
    while (reserved.find(name) != reserved.end()) name.push_back('_');
    return name;
}

}  // namespace

OnnxImporter::OnnxImporter(std::shared_ptr<const ModelParser> parser)
    : parser_(parser ? std::move(parser)
                     : std::make_shared<ProtobufModelParser>()) {}

Model OnnxImporter::load(const std::filesystem::path& path) const {
    return import_model(parser_->parse(path));
}

Model OnnxImporter::import_model(const ModelProto& model) const {
    if (model.opset_version < 13) {
        throw std::invalid_argument("TinyInfer ONNX importer requires opset >= 13");
    }

    Graph graph;
    std::unordered_map<std::string, ValueId> symbols;
    std::unordered_set<std::string> initializer_names;
    std::unordered_set<std::string> reserved_names;
    std::vector<Model::NamedValue> model_inputs;

    for (const auto& initializer : model.graph.initializers) {
        if (initializer.name.empty() ||
            !initializer_names.insert(initializer.name).second) {
            throw std::invalid_argument(
                "ONNX initializer names must be non-empty and unique");
        }
        reserved_names.insert(initializer.name);
        symbols.emplace(initializer.name,
                        graph.add_constant(initializer.name, initializer.value));
    }

    for (const auto& input : model.graph.inputs) {
        if (initializer_names.find(input.name) != initializer_names.end()) {
            const auto id = symbols.at(input.name);
            if (graph.value(id).spec != input.spec) {
                throw std::invalid_argument(
                    "ONNX initializer metadata does not match graph input");
            }
            continue;
        }
        if (input.name.empty() || symbols.find(input.name) != symbols.end()) {
            throw std::invalid_argument(
                "ONNX graph input names must be non-empty and unique");
        }
        const auto id = graph.add_input(input.name, input.spec);
        symbols.emplace(input.name, id);
        reserved_names.insert(input.name);
        model_inputs.emplace_back(input.name, id);
    }

    std::vector<bool> imported(model.graph.nodes.size(), false);
    std::size_t remaining = model.graph.nodes.size();
    while (remaining != 0) {
        bool progress = false;
        for (std::size_t index = 0; index < model.graph.nodes.size(); ++index) {
            if (imported[index]) continue;
            const auto& source = model.graph.nodes[index];
            if (!source.domain.empty() && source.domain != "ai.onnx") {
                throw std::invalid_argument("unsupported ONNX operator domain: " +
                                            source.domain);
            }
            bool inputs_ready = true;
            std::vector<ValueId> inputs;
            inputs.reserve(source.inputs.size());
            for (const auto& name : source.inputs) {
                const auto input = symbols.find(name);
                if (name.empty() || input == symbols.end()) {
                    inputs_ready = false;
                    break;
                }
                inputs.push_back(input->second);
            }
            if (!inputs_ready) continue;
            if (source.outputs.size() != 1 || source.outputs.front().empty()) {
                throw std::invalid_argument(
                    "TinyInfer currently imports one non-empty output per ONNX node");
            }
            if (symbols.find(source.outputs.front()) != symbols.end()) {
                throw std::invalid_argument("ONNX value name is defined more than once");
            }

            const auto translated = operators_.translate(source);
            const auto imported_node = graph.add_node(
                node_name(index, reserved_names), translated.op,
                std::move(inputs), translated.attributes);
            const auto output = graph.node(imported_node).outputs.front();
            symbols.emplace(source.outputs.front(), output);
            imported[index] = true;
            --remaining;
            progress = true;
        }
        if (!progress) {
            throw std::invalid_argument(
                "ONNX graph has unresolved inputs, unsupported ordering, or a cycle");
        }
    }

    std::vector<Model::NamedValue> model_outputs;
    for (const auto& output : model.graph.outputs) {
        const auto value = symbols.find(output.name);
        if (value == symbols.end()) {
            throw std::invalid_argument("ONNX graph output is not defined: " +
                                        output.name);
        }
        if (graph.value(value->second).spec != output.spec) {
            throw std::invalid_argument(
                "ONNX graph output metadata does not match inferred TensorSpec");
        }
        graph.mark_output(value->second);
        model_outputs.emplace_back(output.name, value->second);
    }

    return Model(std::move(graph), std::move(model_inputs),
                 std::move(model_outputs));
}

}  // namespace tinyinfer::onnx
