#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tinyinfer/core/tensor.h"

int main() {
    std::vector<float> values(24);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);

    tinyinfer::Tensor tensor = tinyinfer::Tensor::from_vector({2, 3, 4}, values);
    assert((tensor.strides() == tinyinfer::Strides{12, 4, 1}));
    assert(tensor.offset({0, 0, 0}) == 0);
    assert(tensor.offset({1, 0, 0}) == 12);
    assert(tensor.offset({1, 2, 3}) == 23);
    assert(tensor.at({1, 2, 3}) == 23.0F);

    const tinyinfer::Shape coordinates{0, 1, 2};
    tensor.at(coordinates) = 42.0F;
    assert(tensor.at({0, 1, 2}) == 42.0F);
    assert(tensor.at(6) == 42.0F);

    const tinyinfer::Tensor& const_tensor = tensor;
    assert(const_tensor.at({1, 1, 1}) == 17.0F);

    const auto vector = tinyinfer::Tensor::from_vector({3}, {10.0F, 20.0F, 30.0F});
    assert(vector.at({1}) == 20.0F);

    const auto scalar = tinyinfer::Tensor::from_vector({}, {7.0F});
    assert(scalar.at({}) == 7.0F);
    assert(scalar.offset({}) == 0);

    bool rejected_rank_mismatch = false;
    try {
        (void)tensor.at({1, 2});
    } catch (const std::invalid_argument&) {
        rejected_rank_mismatch = true;
    }
    assert(rejected_rank_mismatch);

    bool rejected_upper_bound = false;
    try {
        (void)tensor.at({2, 0, 0});
    } catch (const std::out_of_range&) {
        rejected_upper_bound = true;
    }
    assert(rejected_upper_bound);

    bool rejected_negative_index = false;
    try {
        (void)tensor.at({0, -1, 0});
    } catch (const std::out_of_range&) {
        rejected_negative_index = true;
    }
    assert(rejected_negative_index);
    return 0;
}
