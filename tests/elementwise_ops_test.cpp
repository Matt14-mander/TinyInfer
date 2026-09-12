#include "tinyinfer/ops/basic_ops.h"

#include <cassert>
#include <cmath>
#include <cstdint>
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
    using tinyinfer::Shape;
    using tinyinfer::Tensor;
    using namespace tinyinfer::ops;

    const auto lhs = Tensor::from_vector(
        {2, 3}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
    const auto rhs = Tensor::from_vector(
        {2, 3}, {6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F});

    const auto difference = sub(lhs, rhs);
    assert(difference.shape() == Shape({2, 3}));
    assert(difference.at({0, 0}) == -5.0F);
    assert(difference.at({1, 2}) == 5.0F);

    const auto row = Tensor::from_vector({1, 3}, {10.0F, 20.0F, 30.0F});
    const auto column = Tensor::from_vector({2, 1}, {2.0F, 3.0F});
    const auto product = mul(column, row);
    assert(product.shape() == Shape({2, 3}));
    assert(product.at({0, 2}) == 60.0F);
    assert(product.at({1, 1}) == 60.0F);

    auto transposed_lhs = lhs;
    auto transposed_rhs = rhs;
    transposed_lhs.transpose(0, 1);
    transposed_rhs.transpose(0, 1);
    const auto strided_difference = sub(transposed_lhs, transposed_rhs);
    assert(strided_difference.shape() == Shape({3, 2}));
    assert(strided_difference.at({0, 1}) == 1.0F);
    assert(strided_difference.at({2, 0}) == -1.0F);

    const auto values = Tensor::from_vector(
        {5}, {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    const auto activated = gelu(values);
    assert(close(activated.at(0), -0.0454023F));
    assert(close(activated.at(1), -0.158808F));
    assert(close(activated.at(2), 0.0F));
    assert(close(activated.at(3), 0.841192F));
    assert(close(activated.at(4), 1.954598F));

    const auto sliced = values.slice(0, 0, 5, 2);
    const auto sliced_gelu = gelu(sliced);
    assert(sliced_gelu.shape() == Shape({3}));
    assert(close(sliced_gelu.at(0), -0.0454023F));
    assert(close(sliced_gelu.at(1), 0.0F));
    assert(close(sliced_gelu.at(2), 1.954598F));

    const auto integers = Tensor::from_vector<std::int32_t>({2}, {1, 2});
    expect_throw<std::invalid_argument>([&] { sub(integers, integers); });
    expect_throw<std::invalid_argument>([&] { mul(lhs, integers); });
    expect_throw<std::invalid_argument>([&] { gelu(integers); });
    expect_throw<std::invalid_argument>([&] {
        const Tensor incompatible({4, 2});
        sub(lhs, incompatible);
    });
}
