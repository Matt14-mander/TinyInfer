#include <cassert>
#include <stdexcept>

#include "tinyinfer/core/tensor.h"

int main() {
    auto tensor = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                            4.0F, 5.0F, 6.0F});
    const void* storage = tensor.data();

    tinyinfer::Tensor& result = tensor.reshape({3, 2});
    assert(&result == &tensor);
    assert(tensor.data() == storage);
    assert((tensor.shape() == tinyinfer::Shape{3, 2}));
    assert((tensor.strides() == tinyinfer::Strides{2, 1}));
    assert(tensor.at({0, 0}) == 1.0F);
    assert(tensor.at({1, 0}) == 3.0F);
    assert(tensor.at({2, 1}) == 6.0F);

    tensor.reshape({-1});
    assert((tensor.shape() == tinyinfer::Shape{6}));
    assert((tensor.strides() == tinyinfer::Strides{1}));
    assert(tensor.data() == storage);
    assert(tensor.at({5}) == 6.0F);

    tensor.reshape({2, -1, 1});
    assert((tensor.shape() == tinyinfer::Shape{2, 3, 1}));
    assert((tensor.strides() == tinyinfer::Strides{3, 1, 1}));
    assert(tensor.at({1, 2, 0}) == 6.0F);

    auto scalar = tinyinfer::Tensor::from_vector({}, {7.0F});
    scalar.reshape({1, 1});
    assert((scalar.shape() == tinyinfer::Shape{1, 1}));
    assert(scalar.at({0, 0}) == 7.0F);
    scalar.reshape({});
    assert(scalar.rank() == 0);
    assert(scalar.at({}) == 7.0F);

    auto empty = tinyinfer::Tensor({0, 3});
    empty.reshape({0, 1, 3});
    assert((empty.shape() == tinyinfer::Shape{0, 1, 3}));
    assert(empty.numel() == 0);

    const auto shape_before_error = tensor.shape();
    const auto strides_before_error = tensor.strides();
    const void* storage_before_error = tensor.data();

    bool rejected_element_mismatch = false;
    try {
        tensor.reshape({4, 2});
    } catch (const std::invalid_argument&) {
        rejected_element_mismatch = true;
    }
    assert(rejected_element_mismatch);

    bool rejected_multiple_inferred = false;
    try {
        tensor.reshape({-1, -1});
    } catch (const std::invalid_argument&) {
        rejected_multiple_inferred = true;
    }
    assert(rejected_multiple_inferred);

    bool rejected_negative_dimension = false;
    try {
        tensor.reshape({2, -2});
    } catch (const std::invalid_argument&) {
        rejected_negative_dimension = true;
    }
    assert(rejected_negative_dimension);

    bool rejected_non_integral_inference = false;
    try {
        tensor.reshape({4, -1});
    } catch (const std::invalid_argument&) {
        rejected_non_integral_inference = true;
    }
    assert(rejected_non_integral_inference);

    bool rejected_ambiguous_zero = false;
    try {
        empty.reshape({0, -1});
    } catch (const std::invalid_argument&) {
        rejected_ambiguous_zero = true;
    }
    assert(rejected_ambiguous_zero);

    assert(tensor.shape() == shape_before_error);
    assert(tensor.strides() == strides_before_error);
    assert(tensor.data() == storage_before_error);
    return 0;
}
