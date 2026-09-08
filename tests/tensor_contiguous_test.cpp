#include <cassert>
#include <cstdint>

#include "tinyinfer/core/tensor.h"

int main() {
    auto matrix = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 2.0F, 3.0F,
                                                            4.0F, 5.0F, 6.0F});
    matrix.transpose(0, 1);
    assert(!matrix.is_contiguous());
    const void* transposed_storage = matrix.data();
    const auto logical_before = matrix.to_string();

    tinyinfer::Tensor& result = matrix.contiguous();
    assert(&result == &matrix);
    assert(matrix.data() != transposed_storage);
    assert(matrix.is_contiguous());
    assert((matrix.shape() == tinyinfer::Shape{3, 2}));
    assert((matrix.strides() == tinyinfer::Strides{2, 1}));
    assert(matrix.to_string() == logical_before);
    assert(matrix.at(0) == 1.0F);
    assert(matrix.at(1) == 4.0F);
    assert(matrix.at(2) == 2.0F);
    assert(matrix.at(3) == 5.0F);
    assert(matrix.at(4) == 3.0F);
    assert(matrix.at(5) == 6.0F);

    const void* contiguous_storage = matrix.data();
    matrix.contiguous();
    assert(matrix.data() == contiguous_storage);

    matrix.reshape({6});
    assert((matrix.shape() == tinyinfer::Shape{6}));
    assert(matrix.to_string() ==
           "Tensor(shape=[6], dtype=float32, data=[1, 4, 2, 5, 3, 6])");

    auto tensor3d = tinyinfer::Tensor::from_vector(
        {2, 2, 3}, {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                    6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F});
    tensor3d.transpose(0, 2);
    const auto tensor3d_logical = tensor3d.to_string();
    tensor3d.contiguous();
    assert(tensor3d.is_contiguous());
    assert((tensor3d.strides() == tinyinfer::Strides{4, 2, 1}));
    assert(tensor3d.to_string() == tensor3d_logical);
    assert(tensor3d.at({2, 1, 1}) == 11.0F);

    tinyinfer::Tensor int32_tensor({2, 3}, tinyinfer::DataType::Int32);
    auto* int32_data = static_cast<std::int32_t*>(int32_tensor.data());
    for (std::int32_t i = 0; i < 6; ++i) int32_data[i] = i + 1;
    int32_tensor.transpose(0, 1);
    const auto int32_logical = int32_tensor.to_string();
    int32_tensor.contiguous();
    assert(int32_tensor.to_string() == int32_logical);
    const auto* reordered = static_cast<const std::int32_t*>(int32_tensor.data());
    assert(reordered[0] == 1);
    assert(reordered[1] == 4);
    assert(reordered[2] == 2);
    assert(reordered[3] == 5);
    assert(reordered[4] == 3);
    assert(reordered[5] == 6);

    tinyinfer::Tensor empty({0, 3});
    empty.transpose(0, 1);
    assert(!empty.is_contiguous());
    empty.contiguous();
    assert(empty.is_contiguous());
    assert((empty.shape() == tinyinfer::Shape{3, 0}));
    assert((empty.strides() == tinyinfer::Strides{0, 1}));
    assert(empty.data() == nullptr);
    return 0;
}
