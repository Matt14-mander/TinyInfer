#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include "tinyinfer/core/dtype.h"

namespace tinyinfer {
using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

class Tensor {
public:
    Tensor();
    explicit Tensor(Shape shape, DataType dtype = DataType::Float32);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    static Tensor from_vector(Shape shape, const std::vector<float>& values);
    const Shape& shape() const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    DataType dtype() const noexcept { return dtype_; }
    std::size_t rank() const noexcept { return shape_.size(); }
    std::size_t numel() const noexcept;
    std::size_t size_bytes() const noexcept;
    bool is_contiguous() const noexcept;
    void* data() noexcept { return storage_.get(); }
    const void* data() const noexcept { return storage_.get(); }
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
    std::string to_string() const;
    void swap(Tensor& other) noexcept;

private:
    static Strides contiguous_strides(const Shape& shape);
    Shape shape_;
    Strides strides_;
    DataType dtype_{DataType::Float32};
    std::shared_ptr<void> storage_;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
}  // namespace tinyinfer
