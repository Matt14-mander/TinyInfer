#include "tinyinfer/graph/graph_analysis.h"

#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_out_of_range(Function&& function) {
    try {
        function();
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error("GraphAnalysis should reject an invalid ID");
}

}  // namespace

int main() {
    using tinyinfer::DataType;
    using tinyinfer::Graph;
    using tinyinfer::GraphAnalysis;
    using tinyinfer::NodeId;
    using tinyinfer::OpType;
    using tinyinfer::Tensor;
    using tinyinfer::TensorSpec;
    using tinyinfer::ValueId;

    Graph graph;
    const auto x = graph.add_input("x", TensorSpec{{2}, DataType::Float32});
    const auto constant = graph.add_constant(
        "constant", Tensor::from_vector({2}, {2.0F, 3.0F}));
    const auto repeated = graph.add_node("repeated", OpType::Add, {x, x});
    const auto shared = graph.node(repeated).outputs.front();
    const auto multiply = graph.add_node(
        "multiply", OpType::Multiply, {shared, constant});
    const auto left = graph.node(multiply).outputs.front();
    const auto relu = graph.add_node("relu", OpType::ReLU, {shared});
    const auto right = graph.node(relu).outputs.front();
    const auto join = graph.add_node("join", OpType::Add, {left, right});
    const auto final = graph.node(join).outputs.front();
    const auto unused = graph.add_node("unused", OpType::ReLU, {constant});
    const auto dead = graph.node(unused).outputs.front();
    graph.mark_output(shared);
    graph.mark_output(final);
    graph.mark_output(x);
    graph.mark_output(constant);

    const GraphAnalysis analysis(graph);
    require(analysis.consumers(x) == std::vector<NodeId>{repeated},
            "duplicate inputs should name one consumer node");
    require(analysis.use_count(x) == 2,
            "duplicate inputs should count as two uses");
    require(analysis.consumers(shared) ==
                (std::vector<NodeId>{multiply, relu}),
            "branch consumers should follow NodeId order");
    require(analysis.use_count(shared) == 2,
            "branching value should have two uses");
    require(analysis.consumers(constant) ==
                (std::vector<NodeId>{multiply, unused}),
            "constant consumers should include all nodes");
    require(analysis.use_count(constant) == 2,
            "constant use count mismatch");
    require(analysis.use_count(left) == 1 && analysis.use_count(right) == 1,
            "intermediate use counts mismatch");
    require(analysis.consumers(final).empty() && analysis.use_count(final) == 0,
            "terminal graph output has no node consumers");
    require(analysis.consumers(dead).empty() && analysis.use_count(dead) == 0,
            "dead value has no consumers");

    require(analysis.is_graph_output(shared) &&
                analysis.is_graph_output(final) &&
                analysis.is_graph_output(x) &&
                analysis.is_graph_output(constant),
            "graph output flags should include intermediate, input, constant");
    require(!analysis.is_graph_output(left) &&
                !analysis.is_graph_output(dead),
            "non-outputs should not have graph output flag");
    require(analysis.graph_outputs() ==
                (std::vector<ValueId>{shared, final, x, constant}),
            "graph outputs should preserve registration order");
    require(analysis.outputs(repeated) == std::vector<ValueId>{shared} &&
                analysis.outputs(join) == std::vector<ValueId>{final},
            "node output lookup mismatch");

    require_out_of_range([&] { static_cast<void>(analysis.consumers(999)); });
    require_out_of_range([&] { static_cast<void>(analysis.use_count(999)); });
    require_out_of_range([&] {
        static_cast<void>(analysis.is_graph_output(999));
    });
    require_out_of_range([&] { static_cast<void>(analysis.outputs(999)); });

    // Analysis is a snapshot; re-run it after modifying the graph.
    graph.mark_output(dead);
    require(!analysis.is_graph_output(dead),
            "existing analysis should remain a snapshot");
    const GraphAnalysis updated(graph);
    require(updated.is_graph_output(dead),
            "new analysis should see the newly registered output");

    Graph incomplete;
    incomplete.add_input("only_input", TensorSpec{{1}, DataType::Float32});
    bool rejected = false;
    try {
        static_cast<void>(GraphAnalysis(incomplete));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "analysis should require a valid graph");
}
