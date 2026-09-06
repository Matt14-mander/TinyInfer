#include "tinyinfer/core/tensor.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <new>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
namespace {

float float16_to_float32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000U) != 0;
    const auto exponent = static_cast<unsigned>((bits >> 10U) & 0x1FU);
    const auto fraction = static_cast<unsigned>(bits & 0x03FFU);

    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent == 31) {
        value = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(fraction) / 1024.0F,
                           static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

void write_value(std::ostream& stream, const Tensor& tensor, std::size_t index) {
    switch (tensor.dtype()) {
        case DataType::Float32:
            stream << static_cast<const float*>(tensor.data())[index];
            return;
        case DataType::Float16:
            stream << float16_to_float32(static_cast<const std::uint16_t*>(tensor.data())[index]);
            return;
        case DataType::Int8:
            stream << static_cast<int>(static_cast<const std::int8_t*>(tensor.data())[index]);
            return;
        case DataType::Int32:
            stream << static_cast<const std::int32_t*>(tensor.data())[index];
            return;
    }
}

void write_data(std::ostream& stream, const Tensor& tensor, std::size_t dimension,
                std::size_t base_offset) {
    if (tensor.rank() == 0) {
        write_value(stream, tensor, 0);
        return;
    }

    stream << '[';
    const auto count = static_cast<std::size_t>(tensor.shape()[dimension]);
    const auto stride = static_cast<std::size_t>(tensor.strides()[dimension]);
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) stream << ", ";
        const auto element_offset = base_offset + index * stride;
        if (dimension + 1 == tensor.rank()) {
            write_value(stream, tensor, element_offset);
        } else {
            write_data(stream, tensor, dimension + 1, element_offset);
        }
    }
    stream << ']';
}

}  // namespace

Tensor::Tensor() : Tensor(Shape{}) {}

Tensor::Tensor(Shape shape, DataType dtype)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), dtype_(dtype) {
    if (std::any_of(shape_.begin(), shape_.end(), [](std::int64_t dim) { return dim < 0; })) {
        throw std::invalid_argument("tensor dimensions must be non-negative");
    }
    const auto bytes = size_bytes();
    if (bytes > 0) {
        storage_.reset(::operator new(bytes), [](void* ptr) { ::operator delete(ptr); });
        std::memset(storage_.get(), 0, bytes);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_) {
    const auto bytes = size_bytes();
    if (bytes > 0) {
        storage_.reset(::operator new(bytes), [](void* ptr) { ::operator delete(ptr); });
        if (other.storage_) {
            std::memcpy(storage_.get(), other.storage_.get(), bytes);
        } else {
            std::memset(storage_.get(), 0, bytes);
        }
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    Tensor copy(other);
    swap(copy);
    return *this;
}

Tensor Tensor::from_vector(Shape shape, const std::vector<float>& values) {
    Tensor tensor(std::move(shape), DataType::Float32);
    if (tensor.numel() != values.size()) {
        throw std::invalid_argument("tensor shape does not match the number of values");
    }
    std::copy(values.begin(), values.end(), tensor.data_f32());
    return tensor;
}

std::size_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 1;
    std::size_t result = 1;
    for (const auto dim : shape_) result *= static_cast<std::size_t>(dim);
    return result;
}

std::size_t Tensor::size_bytes() const noexcept { return numel() * size_of(dtype_); }
bool Tensor::is_contiguous() const noexcept { return strides_ == contiguous_strides(shape_); }

float* Tensor::data_f32() {
    if (dtype_ != DataType::Float32) throw std::logic_error("tensor data type is not float32");
    return static_cast<float*>(storage_.get());
}

const float* Tensor::data_f32() const {
    if (dtype_ != DataType::Float32) throw std::logic_error("tensor data type is not float32");
    return static_cast<const float*>(storage_.get());
}

std::size_t Tensor::offset(const Shape& indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument("number of indices must match tensor rank");
    }

    std::size_t result = 0;
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        const auto index = indices[dimension];
        if (index < 0 || index >= shape_[dimension]) {
            throw std::out_of_range("tensor index is out of range");
        }
        result += static_cast<std::size_t>(index * strides_[dimension]);
    }
    return result;
}

std::size_t Tensor::offset(std::initializer_list<std::int64_t> indices) const {
    return offset(Shape(indices));
}

float& Tensor::at(std::size_t index) {
    if (index >= numel()) throw std::out_of_range("tensor index is out of range");
    return data_f32()[index];
}

const float& Tensor::at(std::size_t index) const {
    if (index >= numel()) throw std::out_of_range("tensor index is out of range");
    return data_f32()[index];
}

float& Tensor::at(const Shape& indices) { return data_f32()[offset(indices)]; }

const float& Tensor::at(const Shape& indices) const { return data_f32()[offset(indices)]; }

float& Tensor::at(std::initializer_list<std::int64_t> indices) {
    return at(Shape(indices));
}

const float& Tensor::at(std::initializer_list<std::int64_t> indices) const {
    return at(Shape(indices));
}

std::string Tensor::to_string() const {
    std::ostringstream stream;
    stream << "Tensor(shape=[";
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        if (dimension != 0) stream << ", ";
        stream << shape_[dimension];
    }
    stream << "], dtype=" << tinyinfer::to_string(dtype_) << ", data=";
    write_data(stream, *this, 0, 0);
    stream << ')';
    return stream.str();
}

void Tensor::swap(Tensor& other) noexcept {
    using std::swap;
    swap(shape_, other.shape_);
    swap(strides_, other.strides_);
    swap(dtype_, other.dtype_);
    swap(storage_, other.storage_);
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.to_string();
}

Strides Tensor::contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
    return strides;
}
}  // namespace tinyinfer
