#include "tinyinfer/model/onnx/onnx_importer.h"

#include "tinyinfer/model/onnx/import_diagnostic.h"
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

OnnxImportDiagnostic model_diagnostic(const ModelProto& model,
                                      OnnxImportStage stage,
                                      std::string message) {
    OnnxImportDiagnostic diagnostic;
    diagnostic.stage = stage;
    diagnostic.graph_name = model.graph.name;
    diagnostic.message = std::move(message);
    return diagnostic;
}

OnnxImportDiagnostic node_diagnostic(const ModelProto& model,
                                     const NodeProto& node,
                                     std::size_t index,
                                     OnnxImportStage stage,
                                     std::string message) {
    auto diagnostic = model_diagnostic(model, stage, std::move(message));
    diagnostic.node_index = index;
    diagnostic.node_name = node.name;
    diagnostic.op_type = node.op_type;
    diagnostic.domain = node.domain;
    return diagnostic;
}

[[noreturn]] void throw_model_error(const ModelProto& model,
                                    OnnxImportStage stage,
                                    std::string message) {
    throw OnnxImportError(
        model_diagnostic(model, stage, std::move(message)));
}

[[noreturn]] void throw_node_error(const ModelProto& model,
                                   const NodeProto& node,
                                   std::size_t index,
                                   OnnxImportStage stage,
                                   std::string message,
                                   std::string value_name = {}) {
    auto diagnostic =
        node_diagnostic(model, node, index, stage, std::move(message));
    diagnostic.value_name = std::move(value_name);
    throw OnnxImportError(std::move(diagnostic));
}

}  // namespace

OnnxImporter::OnnxImporter(std::shared_ptr<const ModelParser> parser)
    : parser_(parser ? std::move(parser)
                     : std::make_shared<ProtobufModelParser>()) {}

Model OnnxImporter::load(const std::filesystem::path& path) const {
    try {
        return import_model(parser_->parse(path));
    } catch (const OnnxImportError& error) {
        if (!error.diagnostic().model_path.empty()) throw;
        auto diagnostic = error.diagnostic();
        diagnostic.model_path = path;
        throw OnnxImportError(std::move(diagnostic));
    } catch (const std::exception& error) {
        OnnxImportDiagnostic diagnostic;
        diagnostic.stage = OnnxImportStage::ProtobufParse;
        diagnostic.model_path = path;
        diagnostic.message = error.what();
        throw OnnxImportError(std::move(diagnostic));
    }
}

Model OnnxImporter::import_model(const ModelProto& model) const {
    if (model.opset_version < 13) {
        throw_model_error(model, OnnxImportStage::ModelValidation,
                          "TinyInfer ONNX importer requires opset >= 13");
    }

    Graph graph;
    std::unordered_map<std::string, ValueId> symbols;
    std::unordered_set<std::string> initializer_names;
    std::unordered_set<std::string> reserved_names;
    std::vector<Model::NamedValue> model_inputs;

    for (const auto& initializer : model.graph.initializers) {
        if (initializer.name.empty() ||
            !initializer_names.insert(initializer.name).second) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphImport,
                "ONNX initializer names must be non-empty and unique");
            diagnostic.value_name = initializer.name;
            throw OnnxImportError(std::move(diagnostic));
        }
        reserved_names.insert(initializer.name);
        try {
            symbols.emplace(
                initializer.name,
                graph.add_constant(initializer.name, initializer.value));
        } catch (const std::exception& error) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphImport, error.what());
            diagnostic.value_name = initializer.name;
            throw OnnxImportError(std::move(diagnostic));
        }
    }

    for (const auto& input : model.graph.inputs) {
        if (initializer_names.find(input.name) != initializer_names.end()) {
            const auto id = symbols.at(input.name);
            if (graph.value(id).spec != input.spec) {
                auto diagnostic = model_diagnostic(
                    model, OnnxImportStage::GraphImport,
                    "ONNX initializer metadata does not match graph input");
                diagnostic.value_name = input.name;
                throw OnnxImportError(std::move(diagnostic));
            }
            continue;
        }
        if (input.name.empty() || symbols.find(input.name) != symbols.end()) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphImport,
                "ONNX graph input names must be non-empty and unique");
            diagnostic.value_name = input.name;
            throw OnnxImportError(std::move(diagnostic));
        }
        ValueId id;
        try {
            id = graph.add_input(input.name, input.spec);
        } catch (const std::exception& error) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphImport, error.what());
            diagnostic.value_name = input.name;
            throw OnnxImportError(std::move(diagnostic));
        }
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
                throw_node_error(model, source, index,
                                 OnnxImportStage::OperatorTranslation,
                                 "unsupported ONNX operator domain: " +
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
                throw_node_error(
                    model, source, index, OnnxImportStage::GraphImport,
                    "TinyInfer currently imports one non-empty output per ONNX node");
            }
            if (symbols.find(source.outputs.front()) != symbols.end()) {
                throw_node_error(model, source, index,
                                 OnnxImportStage::GraphImport,
                                 "ONNX value name is defined more than once",
                                 source.outputs.front());
            }

            TranslatedOperator translated;
            try {
                translated = operators_.translate(source);
            } catch (const std::exception& error) {
                throw_node_error(model, source, index,
                                 OnnxImportStage::OperatorTranslation,
                                 error.what());
            }
            NodeId imported_node;
            try {
                imported_node = graph.add_node(
                    node_name(index, reserved_names), translated.op,
                    std::move(inputs), translated.attributes);
            } catch (const std::exception& error) {
                throw_node_error(model, source, index,
                                 OnnxImportStage::ShapeInference,
                                 error.what());
            }
            const auto output = graph.node(imported_node).outputs.front();
            symbols.emplace(source.outputs.front(), output);
            imported[index] = true;
            --remaining;
            progress = true;
        }
        if (!progress) {
            for (std::size_t index = 0; index < model.graph.nodes.size(); ++index) {
                if (imported[index]) continue;
                const auto& source = model.graph.nodes[index];
                for (const auto& name : source.inputs) {
                    if (!name.empty() && symbols.find(name) != symbols.end()) continue;
                    throw_node_error(
                        model, source, index, OnnxImportStage::GraphImport,
                        name.empty() ? "optional or empty ONNX inputs are not supported"
                                     : "ONNX node input is not defined",
                        name);
                }
                throw_node_error(
                    model, source, index, OnnxImportStage::GraphImport,
                    "ONNX graph dependencies contain a cycle");
            }
            throw_model_error(model, OnnxImportStage::GraphImport,
                              "ONNX graph import made no progress");
        }
    }

    std::vector<Model::NamedValue> model_outputs;
    for (const auto& output : model.graph.outputs) {
        const auto value = symbols.find(output.name);
        if (value == symbols.end()) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphValidation,
                "ONNX graph output is not defined");
            diagnostic.value_name = output.name;
            throw OnnxImportError(std::move(diagnostic));
        }
        if (graph.value(value->second).spec != output.spec) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphValidation,
                "ONNX graph output metadata does not match inferred TensorSpec");
            diagnostic.value_name = output.name;
            throw OnnxImportError(std::move(diagnostic));
        }
        try {
            graph.mark_output(value->second);
        } catch (const std::exception& error) {
            auto diagnostic = model_diagnostic(
                model, OnnxImportStage::GraphValidation, error.what());
            diagnostic.value_name = output.name;
            throw OnnxImportError(std::move(diagnostic));
        }
        model_outputs.emplace_back(output.name, value->second);
    }

    try {
        return Model(std::move(graph), std::move(model_inputs),
                     std::move(model_outputs));
    } catch (const std::exception& error) {
        throw_model_error(model, OnnxImportStage::GraphValidation,
                          error.what());
    }
}

}  // namespace tinyinfer::onnx
