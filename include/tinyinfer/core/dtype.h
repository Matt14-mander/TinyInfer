#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace tinyinfer {
enum class DataType : std::uint8_t { Float32, Float16, Int8, Int32 };

constexpr std::string_view to_string(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int8: return "int8";
        case DataType::Int32: return "int32";
    }
    return "unknown";
}

constexpr std::size_t size_of(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int8: return 1;
        case DataType::Int32: return 4;
    }
    throw std::invalid_argument("unsupported data type");
}
}  // namespace tinyinfer
