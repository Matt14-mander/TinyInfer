#include <cassert>
#include <stdexcept>

#include "tinyinfer/core/tensor.h"

int main() {
    auto tensor = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                            4.0F, 5.0F, 6.0F});
    auto reshaped_view = tensor.view({3, -1});
    assert((reshaped_view.shape() == tinyinfer::Shape{3, 2}));
    assert((reshaped_view.strides() == tinyinfer::Strides{2, 1}));
    assert(reshaped_view.storage().buffer() == tensor.storage().buffer());
    assert(reshaped_view.data() == tensor.data());

    reshaped_view.at({1, 0}) = 30.0F;
    assert(tensor.at({0, 2}) == 30.0F);
    tensor.at({1, 2}) = 60.0F;
    assert(reshaped_view.at({2, 1}) == 60.0F);

    auto deep_copy = reshaped_view;
    assert(deep_copy.storage().buffer() != tensor.storage().buffer());
    assert(deep_copy.is_contiguous());
    deep_copy.at({0, 0}) = -1.0F;
    assert(tensor.at({0, 0}) == 1.0F);

    auto row = tensor.narrow(0, 1, 1);
    assert((row.shape() == tinyinfer::Shape{1, 3}));
    assert((row.strides() == tinyinfer::Strides{3, 1}));
    assert(row.storage().buffer() == tensor.storage().buffer());
    assert(row.storage().byte_offset() == 3 * sizeof(float));
    assert(row.data() == tensor.data<float>() + 3);
    assert(row.to_string() ==
           "Tensor(shape=[1, 3], dtype=float32, data=[[4, 5, 60]])");
    row.at({0, 0}) = 40.0F;
    assert(tensor.at({1, 0}) == 40.0F);

    auto transposed = tensor.view({2, 3});
    transposed.transpose(0, 1);
    auto columns = transposed.narrow(0, 1, 2);
    assert(!columns.is_contiguous());
    assert(columns.storage().buffer() == tensor.storage().buffer());
    assert(columns.to_string() ==
           "Tensor(shape=[2, 2], dtype=float32, data=[[2, 5], [30, 60]])");

    auto detached_copy = columns;
    assert(detached_copy.is_contiguous());
    assert(detached_copy.storage().buffer() != columns.storage().buffer());
    assert(detached_copy.to_string() == columns.to_string());
    detached_copy.at({0, 0}) = -2.0F;
    assert(columns.at({0, 0}) == 2.0F);

    const auto shared_buffer = columns.storage().buffer();
    columns.contiguous();
    assert(columns.is_contiguous());
    assert(columns.storage().buffer() != shared_buffer);
    assert(columns.to_string() ==
           "Tensor(shape=[2, 2], dtype=float32, data=[[2, 5], [30, 60]])");

    tinyinfer::Tensor surviving_view;
    {
        auto temporary = tinyinfer::Tensor::from_vector({2, 2}, {7.0F, 8.0F, 9.0F, 10.0F});
        surviving_view = temporary.view({4});
    }
    assert(surviving_view.to_string() ==
           "Tensor(shape=[4], dtype=float32, data=[7, 8, 9, 10])");

    auto empty_view = tensor.narrow(0, 2, 0);
    assert((empty_view.shape() == tinyinfer::Shape{0, 3}));
    assert(empty_view.numel() == 0);
    assert(empty_view.storage().buffer() == tensor.storage().buffer());

    bool rejected_non_contiguous_view = false;
    try {
        auto non_contiguous = tensor.view({2, 3});
        non_contiguous.transpose(0, 1);
        (void)non_contiguous.view({6});
    } catch (const std::logic_error&) {
        rejected_non_contiguous_view = true;
    }
    assert(rejected_non_contiguous_view);

    bool rejected_bad_view_shape = false;
    try {
        (void)tensor.view({4, 2});
    } catch (const std::invalid_argument&) {
        rejected_bad_view_shape = true;
    }
    assert(rejected_bad_view_shape);

    bool rejected_bad_narrow_range = false;
    try {
        (void)tensor.narrow(1, 2, 2);
    } catch (const std::out_of_range&) {
        rejected_bad_narrow_range = true;
    }
    assert(rejected_bad_narrow_range);
    return 0;
}
