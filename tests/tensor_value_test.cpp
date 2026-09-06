#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "tinyinfer/core/tensor.h"

int main() {
    const auto original = tinyinfer::Tensor::from_vector({2, 2}, {1.0F, 2.0F, 3.5F, -4.0F});

    auto copy = original;
    assert(copy.data() != original.data());
    copy.at({0, 0}) = 99.0F;
    assert(original.at({0, 0}) == 1.0F);
    assert(copy.at({0, 0}) == 99.0F);

    auto assigned = tinyinfer::Tensor::from_vector({1}, {-1.0F});
    assigned = original;
    assert(assigned.data() != original.data());
    assigned.at({1, 1}) = 8.0F;
    assert(original.at({1, 1}) == -4.0F);
    const tinyinfer::Tensor& assigned_alias = assigned;
    assigned = assigned_alias;
    assert(assigned.at({1, 1}) == 8.0F);

    auto move_source = tinyinfer::Tensor::from_vector({3}, {10.0F, 20.0F, 30.0F});
    const void* source_storage = move_source.data();
    tinyinfer::Tensor moved(std::move(move_source));
    assert(moved.data() == source_storage);
    assert(moved.at({2}) == 30.0F);

    auto move_assign_source = tinyinfer::Tensor::from_vector({2}, {5.0F, 6.0F});
    const void* assignment_storage = move_assign_source.data();
    tinyinfer::Tensor move_assigned;
    move_assigned = std::move(move_assign_source);
    assert(move_assigned.data() == assignment_storage);
    assert(move_assigned.at({1}) == 6.0F);

    assert(original.to_string() ==
           "Tensor(shape=[2, 2], dtype=float32, data=[[1, 2], [3.5, -4]])");
    std::ostringstream output;
    output << original;
    assert(output.str() == original.to_string());

    const auto scalar = tinyinfer::Tensor::from_vector({}, {7.25F});
    assert(scalar.to_string() == "Tensor(shape=[], dtype=float32, data=7.25)");

    tinyinfer::Tensor int8_tensor({3}, tinyinfer::DataType::Int8);
    auto* int8_data = static_cast<std::int8_t*>(int8_tensor.data());
    int8_data[0] = -2;
    int8_data[1] = 0;
    int8_data[2] = 127;
    assert(int8_tensor.to_string() == "Tensor(shape=[3], dtype=int8, data=[-2, 0, 127])");

    tinyinfer::Tensor int32_tensor({2}, tinyinfer::DataType::Int32);
    auto* int32_data = static_cast<std::int32_t*>(int32_tensor.data());
    int32_data[0] = -1000;
    int32_data[1] = 2000;
    assert(int32_tensor.to_string() == "Tensor(shape=[2], dtype=int32, data=[-1000, 2000])");

    tinyinfer::Tensor float16_tensor({2}, tinyinfer::DataType::Float16);
    auto* float16_data = static_cast<std::uint16_t*>(float16_tensor.data());
    float16_data[0] = 0x3C00U;  // 1.0
    float16_data[1] = 0xC000U;  // -2.0
    assert(float16_tensor.to_string() == "Tensor(shape=[2], dtype=float16, data=[1, -2])");
    return 0;
}
