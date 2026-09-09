#pragma once

#include <cstddef>
#include <memory>

#include "tinyinfer/core/memory/allocator.h"

namespace tinyinfer {

inline constexpr std::size_t kDefaultBufferAlignment = alignof(std::max_align_t);

class Buffer final {
public:
    explicit Buffer(std::size_t size_bytes,
                    std::shared_ptr<Allocator> allocator = default_allocator(),
                    std::size_t alignment = kDefaultBufferAlignment);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size_bytes() const noexcept { return size_bytes_; }
    std::size_t alignment() const noexcept { return alignment_; }
    const std::shared_ptr<Allocator>& allocator() const noexcept { return allocator_; }

private:
    std::size_t size_bytes_{};
    std::size_t alignment_{};
    std::shared_ptr<Allocator> allocator_;
    void* data_{};
};

}  // namespace tinyinfer
