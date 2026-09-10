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

Shape resolve_reshape_shape(Shape new_shape, std::size_t current_elements) {
    std::size_t known_elements = 1;
    std::size_t inferred_dimension = new_shape.size();

    for (std::size_t dimension = 0; dimension < new_shape.size(); ++dimension) {
        const auto size = new_shape[dimension];
        if (size == -1) {
            if (inferred_dimension != new_shape.size()) {
                throw std::invalid_argument("reshape allows at most one inferred dimension");
            }
            inferred_dimension = dimension;
            continue;
        }
        if (size < 0) {
            throw std::invalid_argument("reshape dimensions must be non-negative or -1");
        }

        const auto unsigned_size = static_cast<std::size_t>(size);
        if (unsigned_size != 0 &&
            known_elements > std::numeric_limits<std::size_t>::max() / unsigned_size) {
            throw std::overflow_error("reshape element count overflows size_t");
        }
        known_elements *= unsigned_size;
    }

    if (inferred_dimension != new_shape.size()) {
        if (known_elements == 0) {
            throw std::invalid_argument("cannot infer a reshape dimension when known dimensions multiply to zero");
        }
        if (current_elements % known_elements != 0) {
            throw std::invalid_argument("reshape cannot infer an integral dimension");
        }
        const auto inferred_size = current_elements / known_elements;
        if (inferred_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("inferred reshape dimension overflows int64_t");
        }
        new_shape[inferred_dimension] = static_cast<std::int64_t>(inferred_size);
        known_elements *= inferred_size;
    }

    if (known_elements != current_elements) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    return new_shape;
}

std::size_t storage_offset_from_logical_index(const Shape& shape,
                                              const Strides& strides,
                                              std::size_t logical_index) {
    std::size_t storage_offset = 0;
    for (std::size_t dimension = shape.size(); dimension > 0; --dimension) {
        const auto size = static_cast<std::size_t>(shape[dimension - 1]);
        const auto coordinate = logical_index % size;
        logical_index /= size;
        storage_offset += coordinate * static_cast<std::size_t>(strides[dimension - 1]);
    }
    return storage_offset;
}

std::size_t required_storage_elements(const Shape& shape, const Strides& strides) {
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("view shape and strides must have the same rank");
    }
    if (shape.empty()) return 1;

    std::size_t maximum_offset = 0;
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (shape[dimension] < 0 || strides[dimension] < 0) {
            throw std::invalid_argument("view shape and strides must be non-negative");
        }
        if (shape[dimension] == 0) return 0;

        const auto extent = static_cast<std::size_t>(shape[dimension] - 1);
        const auto stride = static_cast<std::size_t>(strides[dimension]);
        if (extent != 0 && stride > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("view storage span overflows size_t");
        }
        const auto contribution = extent * stride;
        if (maximum_offset > std::numeric_limits<std::size_t>::max() - contribution) {
            throw std::overflow_error("view storage span overflows size_t");
        }
        maximum_offset += contribution;
    }
    if (maximum_offset == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("view storage span overflows size_t");
    }
    return maximum_offset + 1;
}

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
    : shape_(std::move(shape)),
      strides_(contiguous_strides(shape_)),
      dtype_(dtype) {
    if (std::any_of(shape_.begin(), shape_.end(), [](std::int64_t dim) { return dim < 0; })) {
        throw std::invalid_argument("tensor dimensions must be non-negative");
    }
    const auto bytes = size_bytes();
    storage_ = Storage(bytes, std::move(allocator));
    if (bytes > 0) std::memset(storage_.data(), 0, bytes);
}

Tensor::Tensor(ViewTag, Shape shape, Strides strides, DataType dtype, Storage storage)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype),
      storage_(std::move(storage)) {}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      strides_(contiguous_strides(other.shape_)),
      dtype_(other.dtype_) {
    const auto bytes = size_bytes();
    const auto allocator = other.storage_.valid() ? other.storage_.allocator() : default_allocator();
    storage_ = Storage(bytes, allocator);
    if (bytes > 0) {
        if (other.storage_.valid() && other.storage_.data()) {
            const auto element_size = size_of(dtype_);
            const auto* source = static_cast<const unsigned char*>(other.storage_.data());
            auto* destination = static_cast<unsigned char*>(storage_.data());
            for (std::size_t logical_index = 0; logical_index < numel(); ++logical_index) {
                const auto source_offset = storage_offset_from_logical_index(
                    other.shape_, other.strides_, logical_index);
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

std::size_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 1;
    std::size_t result = 1;
    for (const auto dim : shape_) result *= static_cast<std::size_t>(dim);
    return result;
}

std::size_t Tensor::size_bytes() const noexcept { return numel() * size_of(dtype_); }
bool Tensor::is_contiguous() const noexcept { return strides_ == contiguous_strides(shape_); }

float* Tensor::data_f32() {
    return data<float>();
}

const float* Tensor::data_f32() const {
    return data<float>();
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
    if (!is_contiguous()) {
        throw std::logic_error("reshape requires a contiguous tensor");
    }

    new_shape = resolve_reshape_shape(std::move(new_shape), numel());
    auto new_strides = contiguous_strides(new_shape);
    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
    return *this;
}

Tensor& Tensor::transpose(std::size_t dimension0, std::size_t dimension1) {
    if (dimension0 >= rank() || dimension1 >= rank()) {
        throw std::out_of_range("transpose dimension is out of range");
    }
    if (dimension0 == dimension1) return *this;

    std::swap(shape_[dimension0], shape_[dimension1]);
    std::swap(strides_[dimension0], strides_[dimension1]);
    return *this;
}

Tensor& Tensor::contiguous() {
    if (is_contiguous()) return *this;

    auto new_strides = contiguous_strides(shape_);
    Storage new_storage;
    const auto bytes = size_bytes();
    const auto element_size = size_of(dtype_);

    if (bytes > 0) {
        new_storage = Storage(bytes, allocator());
        const auto* source = static_cast<const unsigned char*>(storage_.data());
        auto* destination = static_cast<unsigned char*>(new_storage.data());

        for (std::size_t logical_index = 0; logical_index < numel(); ++logical_index) {
            const auto source_offset = storage_offset_from_logical_index(
                shape_, strides_, logical_index);
            std::memcpy(destination + logical_index * element_size,
                        source + source_offset * element_size, element_size);
        }
    }

    if (bytes == 0) new_storage = Storage(0, allocator());
    storage_ = std::move(new_storage);
    strides_ = std::move(new_strides);
    return *this;
}

Tensor Tensor::view(Shape new_shape) const {
    if (!is_contiguous()) {
        throw std::logic_error("view reshape requires a contiguous tensor");
    }
    new_shape = resolve_reshape_shape(std::move(new_shape), numel());
    return Tensor(ViewTag{}, new_shape, contiguous_strides(new_shape), dtype_, storage_);
}

Tensor Tensor::narrow(std::size_t dimension, std::int64_t start,
                      std::int64_t length) const {
    if (dimension >= rank()) throw std::out_of_range("narrow dimension is out of range");
    if (start < 0 || length < 0) {
        throw std::invalid_argument("narrow start and length must be non-negative");
    }
    if (start > shape_[dimension] || length > shape_[dimension] - start) {
        throw std::out_of_range("narrow range exceeds tensor dimension");
    }

    auto new_shape = shape_;
    new_shape[dimension] = length;
    const auto storage_offset = static_cast<std::size_t>(start * strides_[dimension]);
    return as_strided_view(std::move(new_shape), strides_, storage_offset);
}

Tensor Tensor::as_strided_view(Shape shape, Strides strides,
                               std::size_t storage_offset_elements) const {
    const auto element_size = size_of(dtype_);
    const auto required_elements = required_storage_elements(shape, strides);
    if (storage_offset_elements > std::numeric_limits<std::size_t>::max() / element_size ||
        required_elements > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("view byte range overflows size_t");
    }

    const auto relative_byte_offset = storage_offset_elements * element_size;
    const auto required_bytes = required_elements * element_size;
    if (relative_byte_offset > storage_.size_bytes() ||
        required_bytes > storage_.size_bytes() - relative_byte_offset) {
        throw std::out_of_range("view range exceeds tensor storage");
    }

    Storage view_storage(storage_.buffer(), storage_.byte_offset() + relative_byte_offset,
                         required_bytes);
    return Tensor(ViewTag{}, std::move(shape), std::move(strides), dtype_,
                  std::move(view_storage));
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
