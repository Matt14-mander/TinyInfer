#pragma once

#include <cstddef>
#include <memory>

#include "tinyinfer/core/memory/buffer.h"

namespace tinyinfer {

class Storage {
public:
    Storage() = default;
    explicit Storage(std::size_t size_bytes,
                     std::shared_ptr<Allocator> allocator = default_allocator(),
                     std::size_t alignment = kDefaultBufferAlignment);
    Storage(std::shared_ptr<Buffer> buffer, std::size_t byte_offset,
            std::size_t size_bytes);

    void* data() noexcept;
    const void* data() const noexcept;
    std::size_t size_bytes() const noexcept { return size_bytes_; }
    std::size_t byte_offset() const noexcept { return byte_offset_; }
    bool valid() const noexcept { return static_cast<bool>(buffer_); }
    const std::shared_ptr<Buffer>& buffer() const noexcept { return buffer_; }
    const std::shared_ptr<Allocator>& allocator() const;

private:
    std::shared_ptr<Buffer> buffer_;
    std::size_t byte_offset_{};
    std::size_t size_bytes_{};
};

}  // namespace tinyinfer
