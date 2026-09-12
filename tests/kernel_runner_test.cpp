#include "tinyinfer/ops/kernel_runner.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

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
    using tinyinfer::DataType;
    using tinyinfer::Shape;
    using tinyinfer::Tensor;
    using tinyinfer::ops::run_binary_kernel;
    using tinyinfer::ops::run_unary_kernel;

    const auto contiguous = Tensor::from_vector(
        {2, 3}, {-3.0F, -2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    const auto squared = run_unary_kernel<float>(
        contiguous, [](float value) { return value * value; });
    assert(squared.shape() == Shape({2, 3}));
    assert(squared.at({0, 0}) == 9.0F);
    assert(squared.at({1, 2}) == 4.0F);

    auto transposed = contiguous;
    transposed.transpose(0, 1);
    const auto negated = run_unary_kernel<float>(
        transposed, [](float value) { return -value; });
    assert(negated.shape() == Shape({3, 2}));
    assert(negated.at({0, 1}) == 0.0F);
    assert(negated.at({2, 1}) == -2.0F);

    const auto lhs = Tensor::from_vector({2, 1}, {1.0F, 2.0F});
    const auto rhs = Tensor::from_vector({1, 3}, {10.0F, 20.0F, 30.0F});
    const auto product = run_binary_kernel<float>(
        lhs, rhs, [](float left, float right) { return left * right; });
    assert(product.shape() == Shape({2, 3}));
    assert(product.at({0, 2}) == 30.0F);
    assert(product.at({1, 1}) == 40.0F);

    const auto integers = Tensor::from_vector<std::int32_t>(
        {3}, {1, 2, 3});
    const auto incremented = run_unary_kernel<std::int32_t>(
        integers, [](std::int32_t value) { return value + 1; });
    assert(incremented.dtype() == DataType::Int32);
    assert(incremented.at<std::int32_t>(2) == 4);

    expect_throw<std::invalid_argument>([&] {
        run_unary_kernel<float>(integers, [](float value) { return value; });
    });
    expect_throw<std::invalid_argument>([&] {
        run_binary_kernel<float>(
            lhs, integers, [](float left, float right) { return left + right; });
    });
    expect_throw<std::invalid_argument>([&] {
        const Tensor incompatible({3, 2});
        run_binary_kernel<float>(
            lhs, incompatible,
            [](float left, float right) { return left + right; });
    });
}
