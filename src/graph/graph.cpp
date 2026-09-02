#include "tinyinfer/graph/graph.h"
#include <stdexcept>
#include <utility>

namespace tinyinfer {
NodeId Graph::add_node(std::string name, OpType op, std::vector<NodeId> inputs) {
    for (const NodeId input : inputs) {
        if (input >= nodes_.size()) throw std::invalid_argument("graph input node does not exist");
    }
    const NodeId id = nodes_.size();
    nodes_.push_back(Node{id, std::move(name), op, std::move(inputs)});
    return id;
}

const Node& Graph::node(NodeId id) const {
    if (id >= nodes_.size()) throw std::out_of_range("graph node id is out of range");
    return nodes_[id];
}
}  // namespace tinyinfer
