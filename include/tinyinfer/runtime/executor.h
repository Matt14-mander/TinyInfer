#pragma once
#include "tinyinfer/backend/backend.h"
#include "tinyinfer/graph/graph.h"
#include "tinyinfer/runtime/execution_context.h"

namespace tinyinfer {
class Executor {
public:
    explicit Executor(Backend& backend) : backend_(backend) {}
    void run(const Graph& graph, ExecutionContext& context);
private:
    Backend& backend_;
};
}  // namespace tinyinfer
