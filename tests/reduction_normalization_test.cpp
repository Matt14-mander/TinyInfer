#include "tinyinfer/core/reduction_iterator.h"
#include "tinyinfer/ops/basic_ops.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool close(float actual, float expected, float tolerance = 1e-5F) {
    return std::fabs(actual - expected) <= tolerance;
}

template <typename Exception, typename Function>
void expect_throw(Function&& function) {
    bool thrown = false;
    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

}  // namespace

int main() {
    using tinyinfer::ReductionIterator;
    using tinyinfer::Shape;
    using tinyinfer::Tensor;
    using tinyinfer::TensorLayout;
    using namespace tinyinfer::ops;

    const ReductionIterator trailing(TensorLayout({2, 3, 4}), {-1});
    assert(trailing.output_shape() == Shape({2, 3}));
    assert(trailing.output_numel() == 6);
    assert(trailing.reduction_numel() == 4);
    assert(trailing.has_contiguous_reduction());
    assert(trailing.input_offset(5, 3) == 23);

    const ReductionIterator multi_axis(TensorLayout({2, 3, 4}), {0, 2}, true);
    assert(multi_axis.output_shape() == Shape({1, 3, 1}));
    assert(multi_axis.reduction_numel() == 8);
    assert(!multi_axis.has_contiguous_reduction());
    assert(multi_axis.input_offset(1, 7) == 19);

    const auto transposed_layout = TensorLayout({2, 3}).transposed(0, 1);
    const ReductionIterator strided(transposed_layout, {0});
    assert(strided.output_shape() == Shape({2}));
    assert(strided.input_offset(1, 2) == 5);

    const auto matrix = Tensor::from_vector(
        {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const auto row_sums = reduce_sum(matrix, {1});
    assert(row_sums.shape() == Shape({2}));
    assert(row_sums.at(0) == 6.0F);
    assert(row_sums.at(1) == 15.0F);

    const auto column_max = reduce_max(matrix, {0}, true);
    assert(column_max.shape() == Shape({1, 3}));
    assert(column_max.at({0, 0}) == 4.0F);
    assert(column_max.at({0, 2}) == 6.0F);

    const auto total = reduce_sum(matrix, {0, 1});
    assert(total.shape().empty());
    assert(total.at(0) == 21.0F);

    const auto column_probabilities = softmax(matrix, 0);
    assert(close(column_probabilities.at({0, 0}), 0.04742587F));
    assert(close(column_probabilities.at({1, 0}), 0.95257413F));
    assert(close(column_probabilities.at({0, 2}) +
                 column_probabilities.at({1, 2}),
                 1.0F));

    const auto normalized = layer_norm(matrix);
    assert(normalized.shape() == matrix.shape());
    assert(close(normalized.at({0, 0}), -1.2247356F));
    assert(close(normalized.at({0, 1}), 0.0F));
    assert(close(normalized.at({0, 2}), 1.2247356F));

    const auto weight = Tensor::from_vector({3}, {1.0F, 2.0F, 3.0F});
    const auto bias = Tensor::from_vector({3}, {0.5F, 0.5F, 0.5F});
    const auto affine = layer_norm(matrix, weight, bias);
    assert(close(affine.at({0, 0}), -0.7247356F));
    assert(close(affine.at({0, 1}), 0.5F));
    assert(close(affine.at({0, 2}), 4.1742068F));

    auto transposed = matrix;
    transposed.transpose(0, 1);
    const auto strided_norm = layer_norm(transposed);
    assert(strided_norm.shape() == Shape({3, 2}));
    assert(close(strided_norm.at({0, 0}), -0.9999978F));
    assert(close(strided_norm.at({0, 1}), 0.9999978F));

    expect_throw<std::invalid_argument>([&] {
        ReductionIterator(matrix.layout(), {});
    });
    expect_throw<std::invalid_argument>([&] {
        ReductionIterator(matrix.layout(), {0, 0});
    });
    expect_throw<std::out_of_range>([&] {
        ReductionIterator(matrix.layout(), {2});
    });
    expect_throw<std::invalid_argument>([&] {
        const Tensor empty({2, 0});
        reduce_max(empty, {1});
    });
    expect_throw<std::invalid_argument>([&] {
        layer_norm(matrix, -1.0F);
    });
    expect_throw<std::invalid_argument>([&] {
        const Tensor wrong_weight({2});
        layer_norm(matrix, wrong_weight, bias);
    });
}
