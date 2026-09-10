#include <cassert>
#include <stdexcept>

#include "tinyinfer/core/tensor.h"
#include "tinyinfer/ops/basic_ops.h"

int main() {
    auto tensor = tinyinfer::Tensor::from_vector(
        {2, 6}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F});

    auto odd_columns = tensor.slice(1, 1, 6, 2);
    assert((odd_columns.shape() == tinyinfer::Shape{2, 3}));
    assert((odd_columns.strides() == tinyinfer::Strides{6, 2}));
    assert(!odd_columns.is_contiguous());
    assert(odd_columns.storage().buffer() == tensor.storage().buffer());
    assert(odd_columns.storage().byte_offset() == sizeof(float));
    assert(odd_columns.to_string() ==
           "Tensor(shape=[2, 3], dtype=float32, data=[[1, 3, 5], [7, 9, 11]])");

    odd_columns.at({1, 1}) = 90.0F;
    assert(tensor.at({1, 3}) == 90.0F);

    auto chained = odd_columns.slice(1, 1, 3);
    assert((chained.shape() == tinyinfer::Shape{2, 2}));
    assert((chained.strides() == tinyinfer::Strides{6, 2}));
    assert(chained.storage().buffer() == tensor.storage().buffer());
    assert(chained.storage().byte_offset() == 3 * sizeof(float));
    assert(chained.to_string() ==
           "Tensor(shape=[2, 2], dtype=float32, data=[[3, 5], [90, 11]])");

    auto materialized = chained;
    assert(materialized.is_contiguous());
    assert(materialized.storage().buffer() != tensor.storage().buffer());
    assert(materialized.to_string() == chained.to_string());

    const auto activated = tinyinfer::ops::relu(chained);
    assert(activated.is_contiguous());
    assert(activated.to_string() == chained.to_string());

    auto full_slice = tensor.slice(1, 0, 6);
    assert(full_slice.is_contiguous());
    assert(full_slice.data() == tensor.data());
    assert(full_slice.storage().buffer() == tensor.storage().buffer());

    auto single = tensor.slice(1, 2, 3, 10);
    assert((single.shape() == tinyinfer::Shape{2, 1}));
    assert((single.strides() == tinyinfer::Strides{6, 10}));
    assert(single.to_string() ==
           "Tensor(shape=[2, 1], dtype=float32, data=[[2], [8]])");

    auto transposed = tensor.view({2, 6});
    transposed.transpose(0, 1);
    auto every_other_row = transposed.slice(0, 0, 6, 2);
    assert((every_other_row.shape() == tinyinfer::Shape{3, 2}));
    assert((every_other_row.strides() == tinyinfer::Strides{2, 6}));
    assert(every_other_row.to_string() ==
           "Tensor(shape=[3, 2], dtype=float32, data=[[0, 6], [2, 8], [4, 10]])");
    every_other_row.contiguous();
    assert(every_other_row.is_contiguous());
    assert(every_other_row.to_string() ==
           "Tensor(shape=[3, 2], dtype=float32, data=[[0, 6], [2, 8], [4, 10]])");

    auto empty = tensor.slice(1, 5, 5, 3);
    assert((empty.shape() == tinyinfer::Shape{2, 0}));
    assert(empty.numel() == 0);
    assert(empty.storage().buffer() == tensor.storage().buffer());
    assert(empty.to_string() ==
           "Tensor(shape=[2, 0], dtype=float32, data=[[], []])");

    bool rejected_dimension = false;
    try {
        (void)tensor.slice(2, 0, 1);
    } catch (const std::out_of_range&) {
        rejected_dimension = true;
    }
    assert(rejected_dimension);

    bool rejected_negative_bound = false;
    try {
        (void)tensor.slice(1, -1, 3);
    } catch (const std::invalid_argument&) {
        rejected_negative_bound = true;
    }
    assert(rejected_negative_bound);

    bool rejected_reversed_range = false;
    try {
        (void)tensor.slice(1, 4, 2);
    } catch (const std::invalid_argument&) {
        rejected_reversed_range = true;
    }
    assert(rejected_reversed_range);

    bool rejected_end = false;
    try {
        (void)tensor.slice(1, 0, 7);
    } catch (const std::out_of_range&) {
        rejected_end = true;
    }
    assert(rejected_end);

    bool rejected_zero_step = false;
    try {
        (void)tensor.slice(1, 0, 6, 0);
    } catch (const std::invalid_argument&) {
        rejected_zero_step = true;
    }
    assert(rejected_zero_step);

    bool rejected_negative_step = false;
    try {
        (void)tensor.slice(1, 0, 6, -1);
    } catch (const std::invalid_argument&) {
        rejected_negative_step = true;
    }
    assert(rejected_negative_step);
    return 0;
}
