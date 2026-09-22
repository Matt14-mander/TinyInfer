#include "tinyinfer/graph/optimizer/pass_manager.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinyinfer::optimizer {
namespace {

std::vector<ValueId> identity_mapping(std::size_t value_count) {
    std::vector<ValueId> mapping(value_count);
    std::iota(mapping.begin(), mapping.end(), ValueId{0});
    return mapping;
}

void validate_interface(const Model& before, const Model& after,
                        const std::vector<ValueId>& mapping,
                        std::string_view pass_name) {
    if (mapping.size() != before.graph().value_count()) {
        throw std::logic_error(std::string(pass_name) +
                               " returned a ValueId mapping with the wrong size");
    }
    for (const auto mapped : mapping) {
        if (mapped != kInvalidValueId && mapped >= after.graph().value_count()) {
            throw std::logic_error(std::string(pass_name) +
                                   " returned an out-of-range mapped ValueId");
        }
    }

    if (before.inputs().size() != after.inputs().size() ||
        before.outputs().size() != after.outputs().size()) {
        throw std::logic_error(std::string(pass_name) +
                               " changed the model interface size");
    }
    for (std::size_t index = 0; index < before.inputs().size(); ++index) {
        const auto& source = before.inputs()[index];
        const auto& target = after.inputs()[index];
        if (source.first != target.first || mapping[source.second] != target.second) {
            throw std::logic_error(std::string(pass_name) +
                                   " did not preserve a model input binding");
        }
    }
    for (std::size_t index = 0; index < before.outputs().size(); ++index) {
        const auto& source = before.outputs()[index];
        const auto& target = after.outputs()[index];
        if (source.first != target.first || mapping[source.second] != target.second) {
            throw std::logic_error(std::string(pass_name) +
                                   " did not preserve a model output binding");
        }
    }
}

void validate_statistics(const Model& before, const Model& after,
                         const OptimizationResult& result,
                         std::string_view pass_name) {
    if (result.pass_statistics.size() != 1) {
        throw std::logic_error(std::string(pass_name) +
                               " must return exactly one statistics record");
    }
    const auto& statistics = result.pass_statistics.front();
    if (statistics.pass_name != std::string(pass_name) ||
        statistics.nodes_before != before.graph().size() ||
        statistics.nodes_after != after.graph().size() ||
        statistics.values_before != before.graph().value_count() ||
        statistics.values_after != after.graph().value_count()) {
        throw std::logic_error(std::string(pass_name) +
                               " returned inconsistent pass statistics");
    }
}

}  // namespace

void PassManager::add_pass(std::unique_ptr<GraphPass> pass) {
    if (!pass) throw std::invalid_argument("graph pass must not be null");
    passes_.push_back(std::move(pass));
}

OptimizationResult PassManager::run(const Model& model) const {
    auto pipeline_mapping = identity_mapping(model.graph().value_count());
    std::vector<PassStatistics> pipeline_statistics;
    std::optional<Model> current_model;
    const Model* pass_input = &model;

    for (const auto& pass : passes_) {
        auto pass_result = pass->run(*pass_input);
        validate_interface(*pass_input, pass_result.model,
                           pass_result.value_mapping, pass->name());
        validate_statistics(*pass_input, pass_result.model,
                            pass_result, pass->name());

        for (auto& mapped : pipeline_mapping) {
            if (mapped == kInvalidValueId) continue;
            mapped = pass_result.value_mapping[mapped];
        }
        pipeline_statistics.push_back(
            std::move(pass_result.pass_statistics.front()));
        current_model.emplace(std::move(pass_result.model));
        pass_input = &*current_model;
    }

    if (!current_model) {
        return OptimizationResult{model, std::move(pipeline_mapping), {}};
    }
    return OptimizationResult{std::move(*current_model),
                              std::move(pipeline_mapping),
                              std::move(pipeline_statistics)};
}

}  // namespace tinyinfer::optimizer
