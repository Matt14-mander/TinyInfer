#include <cassert>
#include <cmath>
#include <stdexcept>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/ops/basic_ops.h"

int main() {
    auto matrix = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                            4.0F, 5.0F, 6.0F});
    const void* storage = matrix.data();
    tinyinfer::Tensor& result = matrix.transpose(0, 1);

    assert(&result == &matrix);
    assert(matrix.data() == storage);
    assert((matrix.shape() == tinyinfer::Shape{3, 2}));
    assert((matrix.strides() == tinyinfer::Strides{1, 3}));
    assert(!matrix.is_contiguous());
    assert(matrix.at({0, 0}) == 1.0F);
    assert(matrix.at({0, 1}) == 4.0F);
    assert(matrix.at({1, 0}) == 2.0F);
    assert(matrix.at({2, 1}) == 6.0F);
    assert(matrix.to_string() ==
           "Tensor(shape=[3, 2], dtype=float32, data=[[1, 4], [2, 5], [3, 6]])");

    const auto column = tinyinfer::Tensor::from_vector({2, 1}, {10.0F, 1.0F});
    const auto product = tinyinfer::ops::matmul(matrix, column);
    assert((product.shape() == tinyinfer::Shape{3, 1}));
    assert(product.at({0, 0}) == 14.0F);
    assert(product.at({1, 0}) == 25.0F);
    assert(product.at({2, 0}) == 36.0F);

    const auto activated = tinyinfer::ops::relu(matrix);
    assert(activated.is_contiguous());
    assert(activated.to_string() ==
           "Tensor(shape=[3, 2], dtype=float32, data=[[1, 4], [2, 5], [3, 6]])");

    const auto biased = tinyinfer::ops::add(
        matrix, tinyinfer::Tensor::from_vector({2}, {100.0F, 200.0F}));
    assert(biased.at({0, 0}) == 101.0F);
    assert(biased.at({0, 1}) == 204.0F);
    assert(biased.at({2, 0}) == 103.0F);
    assert(biased.at({2, 1}) == 206.0F);

    const auto probabilities = tinyinfer::ops::softmax(matrix);
    assert(std::fabs(probabilities.at({0, 0}) - 0.04742587F) < 1e-6F);
    assert(std::fabs(probabilities.at({0, 1}) - 0.95257413F) < 1e-6F);
    assert(std::fabs(probabilities.at({2, 0}) - 0.04742587F) < 1e-6F);
    assert(std::fabs(probabilities.at({2, 1}) - 0.95257413F) < 1e-6F);

    bool rejected_non_contiguous_reshape = false;
    try {
        matrix.reshape({6});
    } catch (const std::logic_error&) {
        rejected_non_contiguous_reshape = true;
    }
    assert(rejected_non_contiguous_reshape);

    matrix.transpose(0, 1);
    assert((matrix.shape() == tinyinfer::Shape{2, 3}));
    assert((matrix.strides() == tinyinfer::Strides{3, 1}));
    assert(matrix.is_contiguous());
    assert(matrix.data() == storage);

    const auto shape_before_error = matrix.shape();
    const auto strides_before_error = matrix.strides();
    bool rejected_dimension = false;
    try {
        matrix.transpose(0, 2);
    } catch (const std::out_of_range&) {
        rejected_dimension = true;
    }
    assert(rejected_dimension);
    assert(matrix.shape() == shape_before_error);
    assert(matrix.strides() == strides_before_error);

    matrix.transpose(1, 1);
    assert(matrix.shape() == shape_before_error);
    assert(matrix.strides() == strides_before_error);

    auto tensor3d = tinyinfer::Tensor::from_vector(
        {2, 2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                    6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F});
    tensor3d.transpose(0, 2);
    assert((tensor3d.shape() == tinyinfer::Shape{3, 2, 2}));
    assert((tensor3d.strides() == tinyinfer::Strides{1, 3, 6}));
    assert(tensor3d.at({2, 1, 1}) == 11.0F);

    bool rejected_scalar_dimension = false;
    try {
        tinyinfer::Tensor scalar;
        scalar.transpose(0, 0);
    } catch (const std::out_of_range&) {
        rejected_scalar_dimension = true;
    }
    assert(rejected_scalar_dimension);
    return 0;
}
