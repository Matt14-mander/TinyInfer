#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tinyinfer/core/tensor.h"

static_assert(tinyinfer::data_type_of<float>() == tinyinfer::DataType::Float32);
static_assert(tinyinfer::data_type_of<std::uint16_t>() == tinyinfer::DataType::Float16);
static_assert(tinyinfer::data_type_of<std::int8_t>() == tinyinfer::DataType::Int8);
static_assert(tinyinfer::data_type_of<std::int32_t>() == tinyinfer::DataType::Int32);
static_assert(!tinyinfer::is_supported_storage_type_v<double>);

int main() {
    auto int32_tensor = tinyinfer::Tensor::from_vector<std::int32_t>(
        {2, 2}, std::vector<std::int32_t>{10, 20, 30, 40});
    assert(int32_tensor.dtype() == tinyinfer::DataType::Int32);
    assert(int32_tensor.data<std::int32_t>()[2] == 30);
    assert(int32_tensor.at<std::int32_t>(3) == 40);
    assert(int32_tensor.at<std::int32_t>({1, 0}) == 30);
    int32_tensor.at<std::int32_t>({0, 1}) = -5;
    assert(int32_tensor.at<std::int32_t>({0, 1}) == -5);

    const tinyinfer::Tensor& const_int32 = int32_tensor;
    assert(const_int32.data<std::int32_t>()[0] == 10);
    assert(const_int32.at<std::int32_t>({1, 1}) == 40);

    int32_tensor.transpose(0, 1);
    assert(int32_tensor.at<std::int32_t>({0, 1}) == 30);
    assert(int32_tensor.at<std::int32_t>({1, 0}) == -5);
    int32_tensor.contiguous();
    assert(int32_tensor.data<std::int32_t>()[0] == 10);
    assert(int32_tensor.data<std::int32_t>()[1] == 30);
    assert(int32_tensor.data<std::int32_t>()[2] == -5);
    assert(int32_tensor.data<std::int32_t>()[3] == 40);

    auto int8_tensor = tinyinfer::Tensor::from_vector<std::int8_t>(
        {3}, std::vector<std::int8_t>{-3, 0, 127});
    assert(int8_tensor.dtype() == tinyinfer::DataType::Int8);
    assert(int8_tensor.at<std::int8_t>({0}) == -3);
    assert(int8_tensor.at<std::int8_t>({2}) == 127);

    auto float16_tensor = tinyinfer::Tensor::from_vector<std::uint16_t>(
        {2}, std::vector<std::uint16_t>{0x3C00U, 0xC000U});
    assert(float16_tensor.dtype() == tinyinfer::DataType::Float16);
    assert(float16_tensor.at<std::uint16_t>({0}) == 0x3C00U);
    assert(float16_tensor.at<std::uint16_t>({1}) == 0xC000U);
    assert(float16_tensor.to_string() ==
           "Tensor(shape=[2], dtype=float16, data=[1, -2])");

    bool rejected_wrong_data_type = false;
    try {
        (void)int32_tensor.data<float>();
    } catch (const std::logic_error&) {
        rejected_wrong_data_type = true;
    }
    assert(rejected_wrong_data_type);

    bool rejected_wrong_at_type = false;
    try {
        (void)int8_tensor.at<std::int32_t>({0});
    } catch (const std::logic_error&) {
        rejected_wrong_at_type = true;
    }
    assert(rejected_wrong_at_type);

    bool rejected_bad_value_count = false;
    try {
        (void)tinyinfer::Tensor::from_vector<std::int32_t>(
            {2, 2}, std::vector<std::int32_t>{1, 2, 3});
    } catch (const std::invalid_argument&) {
        rejected_bad_value_count = true;
    }
    assert(rejected_bad_value_count);
    return 0;
}
