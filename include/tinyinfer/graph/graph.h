#pragma once
#include <string>
#include <vector>
#include "tinyinfer/graph/node.h"

namespace tinyinfer {
class Graph {
public:
    NodeId add_node(std::string name, OpType op, std::vector<NodeId> inputs = {});
    const Node& node(NodeId id) const;
    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    bool empty() const noexcept { return nodes_.empty(); }
    std::size_t size() const noexcept { return nodes_.size(); }
private:
    std::vector<Node> nodes_;
};
}  // namespace tinyinfer
