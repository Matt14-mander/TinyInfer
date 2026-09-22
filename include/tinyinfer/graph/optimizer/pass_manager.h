#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "tinyinfer/graph/optimizer/graph_pass.h"

namespace tinyinfer::optimizer {

class PassManager {
public:
    void add_pass(std::unique_ptr<GraphPass> pass);

    std::size_t size() const noexcept { return passes_.size(); }
    bool empty() const noexcept { return passes_.empty(); }

    OptimizationResult run(const Model& model) const;

private:
    std::vector<std::unique_ptr<GraphPass>> passes_;
};

}  // namespace tinyinfer::optimizer
