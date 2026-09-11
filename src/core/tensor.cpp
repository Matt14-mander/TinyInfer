#include "tinyinfer/core/tensor.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
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
            stream << tensor.data<float>()[index];
            return;
        case DataType::Float16:
            stream << float16_to_float32(tensor.data<std::uint16_t>()[index]);
            return;
        case DataType::Int8:
            stream << static_cast<int>(tensor.data<std::int8_t>()[index]);
            return;
        case DataType::Int32:
            stream << tensor.data<std::int32_t>()[index];
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

Tensor::Tensor(Shape shape, DataType dtype, std::shared_ptr<Allocator> allocator)
    : layout_(std::move(shape)), dtype_(dtype) {
    const auto bytes = size_bytes();
    storage_ = Storage(bytes, std::move(allocator));
    if (bytes > 0) std::memset(storage_.data(), 0, bytes);
}

Tensor::Tensor(ViewTag, TensorLayout layout, DataType dtype, Storage storage)
    : layout_(std::move(layout)), dtype_(dtype),
      storage_(std::move(storage)) {}

Tensor::Tensor(const Tensor& other)
    : layout_(other.shape()), dtype_(other.dtype_) {
    const auto bytes = size_bytes();
    const auto allocator = other.storage_.valid() ? other.storage_.allocator() : default_allocator();
    storage_ = Storage(bytes, allocator);
    if (bytes > 0) {
        if (other.storage_.valid() && other.storage_.data()) {
            const auto element_size = size_of(dtype_);
            const auto* source = static_cast<const unsigned char*>(other.storage_.data());
            auto* destination = static_cast<unsigned char*>(storage_.data());
            for (std::size_t logical_index = 0; logical_index < numel(); ++logical_index) {
                const auto source_offset = other.layout_.storage_offset(logical_index);
                std::memcpy(destination + logical_index * element_size,
                            source + source_offset * element_size, element_size);
            }
        } else {
            std::memset(storage_.data(), 0, bytes);
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

float* Tensor::data_f32() {
    return data<float>();
}

const float* Tensor::data_f32() const {
    return data<float>();
}

std::size_t Tensor::offset(const Shape& indices) const {
    return layout_.offset(indices);
}

std::size_t Tensor::offset(std::initializer_list<std::int64_t> indices) const {
    return offset(Shape(indices));
}

float& Tensor::at(std::size_t index) {
    return at<float>(index);
}

const float& Tensor::at(std::size_t index) const {
    return at<float>(index);
}

float& Tensor::at(const Shape& indices) { return at<float>(indices); }

const float& Tensor::at(const Shape& indices) const { return at<float>(indices); }

float& Tensor::at(std::initializer_list<std::int64_t> indices) {
    return at(Shape(indices));
}

const float& Tensor::at(std::initializer_list<std::int64_t> indices) const {
    return at(Shape(indices));
}

Tensor& Tensor::reshape(Shape new_shape) {
    layout_ = layout_.reshaped(std::move(new_shape));
    return *this;
}

Tensor& Tensor::transpose(std::size_t dimension0, std::size_t dimension1) {
    layout_ = layout_.transposed(dimension0, dimension1);
    return *this;
}

Tensor& Tensor::contiguous() {
    if (is_contiguous()) return *this;

    TensorLayout new_layout(shape());
    Storage new_storage;
    const auto bytes = size_bytes();
    const auto element_size = size_of(dtype_);

    if (bytes > 0) {
        new_storage = Storage(bytes, allocator());
        const auto* source = static_cast<const unsigned char*>(storage_.data());
        auto* destination = static_cast<unsigned char*>(new_storage.data());

        for (std::size_t logical_index = 0; logical_index < numel(); ++logical_index) {
            const auto source_offset = layout_.storage_offset(logical_index);
            std::memcpy(destination + logical_index * element_size,
                        source + source_offset * element_size, element_size);
        }
    }

    if (bytes == 0) new_storage = Storage(0, allocator());
    storage_ = std::move(new_storage);
    layout_ = std::move(new_layout);
    return *this;
}

Tensor Tensor::view(Shape new_shape) const {
    return Tensor(ViewTag{}, layout_.reshaped(std::move(new_shape)), dtype_, storage_);
}

Tensor Tensor::narrow(std::size_t dimension, std::int64_t start,
                      std::int64_t length) const {
    if (dimension >= rank()) throw std::out_of_range("narrow dimension is out of range");
    if (start < 0 || length < 0) {
        throw std::invalid_argument("narrow start and length must be non-negative");
    }
    if (start > shape()[dimension] || length > shape()[dimension] - start) {
        throw std::out_of_range("narrow range exceeds tensor dimension");
    }

    auto new_shape = shape();
    new_shape[dimension] = length;
    const auto storage_offset = static_cast<std::size_t>(start * strides()[dimension]);
    return as_strided_view(TensorLayout(std::move(new_shape), strides()), storage_offset);
}

Tensor Tensor::slice(std::size_t dimension, std::int64_t start,
                     std::int64_t end, std::int64_t step) const {
    if (dimension >= rank()) throw std::out_of_range("slice dimension is out of range");
    if (step <= 0) throw std::invalid_argument("slice step must be positive");
    if (start < 0 || end < 0) {
        throw std::invalid_argument("slice start and end must be non-negative");
    }
    if (start > end) throw std::invalid_argument("slice start must not exceed end");
    if (end > shape()[dimension]) throw std::out_of_range("slice end exceeds tensor dimension");

    const auto distance = end - start;
    const auto length = distance / step + (distance % step != 0 ? 1 : 0);
    if (strides()[dimension] != 0 &&
        step > std::numeric_limits<std::int64_t>::max() / strides()[dimension]) {
        throw std::overflow_error("slice stride overflows int64_t");
    }
    if (strides()[dimension] != 0 &&
        start > std::numeric_limits<std::int64_t>::max() / strides()[dimension]) {
        throw std::overflow_error("slice storage offset overflows int64_t");
    }

    auto new_shape = shape();
    auto new_strides = strides();
    new_shape[dimension] = length;
    new_strides[dimension] *= step;
    const auto storage_offset = static_cast<std::size_t>(start * strides()[dimension]);
    return as_strided_view(TensorLayout(std::move(new_shape), std::move(new_strides)),
                           storage_offset);
}

Tensor Tensor::as_strided_view(TensorLayout layout,
                               std::size_t storage_offset_elements) const {
    const auto element_size = size_of(dtype_);
    const auto required_elements = layout.storage_span();
    if (storage_offset_elements > std::numeric_limits<std::size_t>::max() / element_size ||
        required_elements > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("view byte range overflows size_t");
    }

    const auto relative_byte_offset = required_elements == 0
                                          ? 0
                                          : storage_offset_elements * element_size;
    const auto required_bytes = required_elements * element_size;
    if (relative_byte_offset > storage_.size_bytes() ||
        required_bytes > storage_.size_bytes() - relative_byte_offset) {
        throw std::out_of_range("view range exceeds tensor storage");
    }

    Storage view_storage(storage_.buffer(), storage_.byte_offset() + relative_byte_offset,
                         required_bytes);
    return Tensor(ViewTag{}, std::move(layout), dtype_,
                  std::move(view_storage));
}

std::string Tensor::to_string() const {
    std::ostringstream stream;
    stream << "Tensor(shape=[";
    for (std::size_t dimension = 0; dimension < rank(); ++dimension) {
        if (dimension != 0) stream << ", ";
        stream << shape()[dimension];
    }
    stream << "], dtype=" << tinyinfer::to_string(dtype_) << ", data=";
    write_data(stream, *this, 0, 0);
    stream << ')';
    return stream.str();
}

void Tensor::swap(Tensor& other) noexcept {
    using std::swap;
    swap(layout_, other.layout_);
    swap(dtype_, other.dtype_);
    swap(storage_, other.storage_);
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.to_string();
}

}  // namespace tinyinfer
