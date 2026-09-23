#include "tinyinfer/graph/optimizer/graph_rewriter.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer::optimizer {

GraphRewriter::GraphRewriter(const Model& source)
    : source_(source),
      value_mapping_(source.graph().value_count(), kInvalidValueId) {
    const auto& source_graph = source_.graph();
    for (const auto source_input : source_graph.inputs()) {
        const auto& value = source_graph.value(source_input);
        value_mapping_[source_input] = graph_.add_input(value.name, value.spec);
    }
}

ValueId GraphRewriter::copy_value(ValueId source_value) {
    if (finished_) {
        throw std::logic_error("cannot modify a finished graph rewrite");
    }
    const auto& source_graph = source_.graph();
    const auto& value = source_graph.value(source_value);
    if (value_mapping_[source_value] != kInvalidValueId) {
        return value_mapping_[source_value];
    }

    if (value.kind == ValueKind::Constant) {
        const auto mapped = graph_.add_constant(
            value.name, source_graph.constant(source_value));
        value_mapping_[source_value] = mapped;
        return mapped;
    }
    if (value.kind == ValueKind::Input) {
        throw std::logic_error("graph input was not initialized by GraphRewriter");
    }
    throw std::logic_error(
        "cannot copy an intermediate before copying its producer");
}

NodeId GraphRewriter::copy_node(NodeId source_node) {
    if (finished_) {
        throw std::logic_error("cannot modify a finished graph rewrite");
    }
    const auto& node = source_.graph().node(source_node);
    for (const auto output : node.outputs) {
        if (value_mapping_[output] != kInvalidValueId) {
            throw std::logic_error("source value was rewritten more than once");
        }
    }
    std::vector<ValueId> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto input : node.inputs) {
        inputs.push_back(copy_value(input));
    }

    const auto mapped_node = graph_.add_node(
        node.name, node.op, std::move(inputs), node.attributes);
    const auto& mapped_outputs = graph_.node(mapped_node).outputs;
    if (mapped_outputs.size() != node.outputs.size()) {
        throw std::logic_error("rewritten node output count changed");
    }
    for (std::size_t index = 0; index < node.outputs.size(); ++index) {
        const auto source_output = node.outputs[index];
        value_mapping_[source_output] = mapped_outputs[index];
    }
    return mapped_node;
}

ValueId GraphRewriter::replace_with_constant(ValueId source_value,
                                             Tensor value) {
    if (finished_) {
        throw std::logic_error("cannot modify a finished graph rewrite");
    }
    const auto& source = source_.graph().value(source_value);
    if (value_mapping_[source_value] != kInvalidValueId) {
        throw std::logic_error("source value was rewritten more than once");
    }
    if (value.shape() != source.spec.shape || value.dtype() != source.spec.dtype) {
        throw std::invalid_argument(
            "replacement constant does not match the source TensorSpec");
    }
    const auto mapped = graph_.add_constant(source.name, std::move(value));
    value_mapping_[source_value] = mapped;
    return mapped;
}

Model GraphRewriter::finish() {
    if (finished_) {
        throw std::logic_error("graph rewrite was already finished");
    }

    std::vector<Model::NamedValue> inputs;
    inputs.reserve(source_.inputs().size());
    for (const auto& input : source_.inputs()) {
        const auto mapped = value_mapping_.at(input.second);
        if (mapped == kInvalidValueId) {
            throw std::logic_error("rewritten model input is missing");
        }
        inputs.emplace_back(input.first, mapped);
    }

    std::vector<Model::NamedValue> outputs;
    outputs.reserve(source_.outputs().size());
    for (const auto& output : source_.outputs()) {
        const auto mapped = value_mapping_.at(output.second);
        if (mapped == kInvalidValueId) {
            throw std::logic_error("rewritten model output is missing");
        }
        graph_.mark_output(mapped);
        outputs.emplace_back(output.first, mapped);
    }

    finished_ = true;
    return Model(std::move(graph_), std::move(inputs), std::move(outputs));
}

}  // namespace tinyinfer::optimizer
