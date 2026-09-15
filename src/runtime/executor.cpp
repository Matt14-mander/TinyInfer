#include "tinyinfer/runtime/executor.h"
#include <stdexcept>

namespace tinyinfer {
void Executor::run(const Graph& graph) {
    for (const auto& node : graph.nodes()) {
        if (!backend_.supports(node.op)) throw std::runtime_error("operator is not supported by the selected backend");
        backend_.execute(node);
    }
}
}  // namespace tinyinfer
