#include "tinyinfer/core/tensor.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
Tensor::Tensor(Shape shape, DataType dtype)
    : shape_(std::move(shape)), strides_(contiguous_strides(shape_)), dtype_(dtype) {
    if (std::any_of(shape_.begin(), shape_.end(), [](std::int64_t dim) { return dim < 0; })) {
        throw std::invalid_argument("tensor dimensions must be non-negative");
    }
    const auto bytes = size_bytes();
    if (bytes > 0) storage_.reset(::operator new(bytes), [](void* ptr) { ::operator delete(ptr); });
}

std::size_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 1;
    std::size_t result = 1;
    for (const auto dim : shape_) result *= static_cast<std::size_t>(dim);
    return result;
}

std::size_t Tensor::size_bytes() const noexcept { return numel() * size_of(dtype_); }
bool Tensor::is_contiguous() const noexcept { return strides_ == contiguous_strides(shape_); }

Strides Tensor::contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i) strides[i - 2] = strides[i - 1] * shape[i - 1];
    return strides;
}
}  // namespace tinyinfer
