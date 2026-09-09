#include "tinyinfer/core/memory/storage.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer {

Storage::Storage(std::size_t size_bytes, std::shared_ptr<Allocator> allocator,
                 std::size_t alignment)
    : buffer_(std::make_shared<Buffer>(size_bytes, std::move(allocator), alignment)),
      size_bytes_(size_bytes) {}

Storage::Storage(std::shared_ptr<Buffer> buffer, std::size_t byte_offset,
                 std::size_t size_bytes)
    : buffer_(std::move(buffer)), byte_offset_(byte_offset), size_bytes_(size_bytes) {
    if (!buffer_) throw std::invalid_argument("storage buffer must not be null");
    if (byte_offset_ > buffer_->size_bytes() ||
        size_bytes_ > buffer_->size_bytes() - byte_offset_) {
        throw std::out_of_range("storage range exceeds buffer capacity");
    }
}

void* Storage::data() noexcept {
    if (!buffer_ || !buffer_->data()) return nullptr;
    return static_cast<unsigned char*>(buffer_->data()) + byte_offset_;
}

const void* Storage::data() const noexcept {
    if (!buffer_ || !buffer_->data()) return nullptr;
    return static_cast<const unsigned char*>(buffer_->data()) + byte_offset_;
}

const std::shared_ptr<Allocator>& Storage::allocator() const {
    if (!buffer_) throw std::logic_error("empty storage has no allocator");
    return buffer_->allocator();
}

}  // namespace tinyinfer
