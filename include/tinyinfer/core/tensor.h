#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "tinyinfer/core/dtype.h"
#include "tinyinfer/core/memory/allocator.h"

namespace tinyinfer {
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

class Tensor {
public:
    Tensor();
    explicit Tensor(Shape shape, DataType dtype = DataType::Float32,
                    std::shared_ptr<Allocator> allocator = default_allocator());
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    static Tensor from_vector(Shape shape, const std::vector<float>& values);
    template <typename T>
    static Tensor from_vector(Shape shape, const std::vector<T>& values);
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    DataType dtype() const noexcept { return dtype_; }
    const std::shared_ptr<Allocator>& allocator() const noexcept { return allocator_; }
    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t numel() const noexcept;
    std::size_t size_bytes() const noexcept;
    bool is_contiguous() const noexcept;
    void* data() noexcept { return storage_.get(); }
    const void* data() const noexcept { return storage_.get(); }
    template <typename T>
    T* data();
    template <typename T>
    const T* data() const;
    float* data_f32();
    const float* data_f32() const;
    std::size_t offset(const Shape& indices) const;
    std::size_t offset(std::initializer_list<std::int64_t> indices) const;
    float& at(std::size_t index);
    const float& at(std::size_t index) const;
    float& at(const Shape& indices);
    const float& at(const Shape& indices) const;
    float& at(std::initializer_list<std::int64_t> indices);
    const float& at(std::initializer_list<std::int64_t> indices) const;
    template <typename T>
    T& at(std::size_t index);
    template <typename T>
    const T& at(std::size_t index) const;
    template <typename T>
    T& at(const Shape& indices);
    template <typename T>
    const T& at(const Shape& indices) const;
    template <typename T>
    T& at(std::initializer_list<std::int64_t> indices);
    template <typename T>
    const T& at(std::initializer_list<std::int64_t> indices) const;
    Tensor& reshape(Shape new_shape);
    Tensor& transpose(std::size_t dimension0, std::size_t dimension1);
    Tensor& contiguous();
    std::string to_string() const;
    void swap(Tensor& other) noexcept;

private:
    static Strides contiguous_strides(const Shape& shape);
    Shape shape_;
    Strides strides_;
    DataType dtype_{DataType::Float32};
    std::shared_ptr<Allocator> allocator_;
    std::shared_ptr<void> storage_;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

template <typename T>
Tensor Tensor::from_vector(Shape shape, const std::vector<T>& values) {
    using StorageType = std::remove_cv_t<T>;
    static_assert(std::is_same_v<T, StorageType>, "Tensor storage type must be unqualified");
    Tensor tensor(std::move(shape), data_type_of<StorageType>());
    if (tensor.numel() != values.size()) {
        throw std::invalid_argument("tensor shape does not match the number of values");
    }
    std::copy(values.begin(), values.end(), tensor.data<StorageType>());
    return tensor;
}

template <typename T>
T* Tensor::data() {
    using StorageType = std::remove_cv_t<T>;
    static_assert(std::is_same_v<T, StorageType>, "Tensor storage type must be unqualified");
    static_assert(is_supported_storage_type_v<StorageType>,
                  "unsupported TinyInfer C++ storage type");
    if (dtype_ != data_type_of<StorageType>()) {
        throw std::logic_error("tensor dtype does not match requested C++ storage type");
    }
    return static_cast<StorageType*>(storage_.get());
}

template <typename T>
const T* Tensor::data() const {
    using StorageType = std::remove_cv_t<T>;
    static_assert(std::is_same_v<T, StorageType>, "Tensor storage type must be unqualified");
    static_assert(is_supported_storage_type_v<StorageType>,
                  "unsupported TinyInfer C++ storage type");
    if (dtype_ != data_type_of<StorageType>()) {
        throw std::logic_error("tensor dtype does not match requested C++ storage type");
    }
    return static_cast<const StorageType*>(storage_.get());
}

template <typename T>
T& Tensor::at(std::size_t index) {
    if (index >= numel()) throw std::out_of_range("tensor index is out of range");
    return data<T>()[index];
}

template <typename T>
const T& Tensor::at(std::size_t index) const {
    if (index >= numel()) throw std::out_of_range("tensor index is out of range");
    return data<T>()[index];
}

template <typename T>
T& Tensor::at(const Shape& indices) {
    return data<T>()[offset(indices)];
}

template <typename T>
const T& Tensor::at(const Shape& indices) const {
    return data<T>()[offset(indices)];
}

template <typename T>
T& Tensor::at(std::initializer_list<std::int64_t> indices) {
    return at<T>(Shape(indices));
}

template <typename T>
const T& Tensor::at(std::initializer_list<std::int64_t> indices) const {
    return at<T>(Shape(indices));
}
}  // namespace tinyinfer
