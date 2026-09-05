#include <cassert>
#include <cmath>
#include <stdexcept>

#include "tinyinfer/tinyinfer.h"

namespace {
bool close(float actual, float expected, float tolerance = 1e-5F) {
    return std::fabs(actual - expected) <= tolerance;
}
}  // namespace

int main() {
    const auto lhs = tinyinfer::Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
    const auto rhs = tinyinfer::Tensor::from_vector({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F});
    const auto product = tinyinfer::ops::matmul(lhs, rhs);
    assert(close(product.at(0), 19.0F));
    assert(close(product.at(1), 22.0F));
    assert(close(product.at(2), 43.0F));
    assert(close(product.at(3), 50.0F));

    const auto biased = tinyinfer::ops::add(product, tinyinfer::Tensor::from_vector({2}, {1.0F, -2.0F}));
    assert(close(biased.at(0), 20.0F));
    assert(close(biased.at(1), 20.0F));
    assert(close(biased.at(2), 44.0F));
    assert(close(biased.at(3), 48.0F));

    const auto activated = tinyinfer::ops::relu(tinyinfer::Tensor::from_vector({3}, {-1.0F, 0.0F, 2.0F}));
    assert(close(activated.at(0), 0.0F));
    assert(close(activated.at(2), 2.0F));

    const auto probabilities = tinyinfer::ops::softmax(tinyinfer::Tensor::from_vector({1, 2}, {2.0F, 3.0F}));
    assert(close(probabilities.at(0), 0.26894143F));
    assert(close(probabilities.at(1), 0.73105860F));
    assert(close(probabilities.at(0) + probabilities.at(1), 1.0F));

    const auto x = tinyinfer::Tensor::from_vector({1, 2}, {1.0F, -2.0F});
    const auto w1 = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 0.0F, -1.0F,
                                                             1.0F, -1.0F, 1.0F});
    const auto b1 = tinyinfer::Tensor::from_vector({3}, {0.0F, 0.0F, 1.0F});
    const auto w2 = tinyinfer::Tensor::from_vector({3, 2}, {1.0F, 0.0F,
                                                             1.0F, 2.0F,
                                                             0.0F, 1.0F});
    const auto b2 = tinyinfer::Tensor::from_vector({2}, {0.0F, -1.0F});
    const auto hidden = tinyinfer::ops::relu(tinyinfer::ops::linear(x, w1, b1));
    const auto logits = tinyinfer::ops::linear(hidden, w2, b2);
    const auto mlp_output = tinyinfer::ops::softmax(logits);
    assert(close(logits.at(0), 2.0F));
    assert(close(logits.at(1), 3.0F));
    assert(close(mlp_output.at(0), 0.26894143F));
    assert(close(mlp_output.at(1), 0.73105860F));

    bool rejected_bad_shape = false;
    try {
        tinyinfer::ops::matmul(lhs, tinyinfer::Tensor({3, 1}));
    } catch (const std::invalid_argument&) {
        rejected_bad_shape = true;
    }
    assert(rejected_bad_shape);
    return 0;
}
