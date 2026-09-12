#include "tinyinfer/core/tensor_iterator.h"
#include "tinyinfer/ops/basic_ops.h"

#include <cassert>
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
    using tinyinfer::Shape;
    using tinyinfer::Strides;
    using tinyinfer::Tensor;
    using tinyinfer::TensorIterator;
    using tinyinfer::TensorLayout;

    const TensorIterator bias_iterator({TensorLayout({2, 3}),
                                        TensorLayout({3})});
    assert(bias_iterator.shape() == Shape({2, 3}));
    assert(bias_iterator.numel() == 6);
    assert(bias_iterator.operand_strides(0) == Strides({3, 1}));
    assert(bias_iterator.operand_strides(1) == Strides({0, 1}));
    assert(bias_iterator.operand_offset(0, 5) == 5);
    assert(bias_iterator.operand_offset(1, 5) == 2);

    const TensorIterator multidirectional({TensorLayout({2, 1, 3}),
                                           TensorLayout({1, 4, 1})});
    assert(multidirectional.shape() == Shape({2, 4, 3}));
    assert(multidirectional.numel() == 24);
    assert(multidirectional.operand_strides(0) == Strides({3, 0, 1}));
    assert(multidirectional.operand_strides(1) == Strides({0, 1, 0}));
    assert(multidirectional.operand_offset(0, 23) == 5);
    assert(multidirectional.operand_offset(1, 23) == 3);

    const auto transposed = TensorLayout({2, 3}).transposed(0, 1);
    const TensorIterator strided({transposed});
    assert(strided.shape() == Shape({3, 2}));
    assert(strided.operand_offset(0, 0) == 0);
    assert(strided.operand_offset(0, 1) == 3);
    assert(strided.operand_offset(0, 5) == 5);

    const TensorIterator scalar({TensorLayout({2, 2}), TensorLayout()});
    assert(scalar.shape() == Shape({2, 2}));
    assert(scalar.operand_strides(1) == Strides({0, 0}));
    assert(scalar.operand_offset(1, 3) == 0);

    auto lhs = Tensor::from_vector({2, 1}, {1.0F, 2.0F});
    auto rhs = Tensor::from_vector({1, 3}, {10.0F, 20.0F, 30.0F});
    const auto sum = tinyinfer::ops::add(lhs, rhs);
    assert(sum.shape() == Shape({2, 3}));
    assert(sum.at({0, 0}) == 11.0F);
    assert(sum.at({0, 2}) == 31.0F);
    assert(sum.at({1, 1}) == 22.0F);

    auto matrix = Tensor::from_vector(
        {2, 3}, {-1.0F, 2.0F, -3.0F, 4.0F, -5.0F, 6.0F});
    matrix.transpose(0, 1);
    const auto activated = tinyinfer::ops::relu(matrix);
    assert(activated.shape() == Shape({3, 2}));
    assert(activated.at({0, 0}) == 0.0F);
    assert(activated.at({0, 1}) == 4.0F);
    assert(activated.at({2, 1}) == 6.0F);

    expect_throw<std::invalid_argument>([] {
        TensorIterator(std::vector<TensorLayout>{});
    });
    expect_throw<std::invalid_argument>([] {
        TensorIterator({TensorLayout({2, 3}), TensorLayout({2, 2})});
    });
    expect_throw<std::out_of_range>([&] {
        bias_iterator.operand_offset(2, 0);
    });
    expect_throw<std::out_of_range>([&] {
        bias_iterator.operand_offset(0, 6);
    });
}
