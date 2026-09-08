#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace tinyinfer {
enum class DataType : std::uint8_t { Float32, Float16, Int8, Int32 };

template <typename T>
struct DataTypeTraits {
    static constexpr bool supported = false;
    static constexpr DataType value = DataType::Float32;
};

template <>
struct DataTypeTraits<float> {
    static constexpr bool supported = true;
    static constexpr DataType value = DataType::Float32;
};

// Float16 is currently exposed as its raw IEEE 754 binary16 storage bits.
template <>
struct DataTypeTraits<std::uint16_t> {
    static constexpr bool supported = true;
    static constexpr DataType value = DataType::Float16;
};

template <>
struct DataTypeTraits<std::int8_t> {
    static constexpr bool supported = true;
    static constexpr DataType value = DataType::Int8;
};

template <>
struct DataTypeTraits<std::int32_t> {
    static constexpr bool supported = true;
    static constexpr DataType value = DataType::Int32;
};

template <typename T>
inline constexpr bool is_supported_storage_type_v =
    DataTypeTraits<std::remove_cv_t<T>>::supported;

template <typename T>
constexpr DataType data_type_of() {
    using StorageType = std::remove_cv_t<T>;
    static_assert(DataTypeTraits<StorageType>::supported,
                  "unsupported TinyInfer C++ storage type");
    return DataTypeTraits<StorageType>::value;
}

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
