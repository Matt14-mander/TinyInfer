#include "tinyinfer/core/memory/buffer.h"

#include <new>
#include <stdexcept>
#include <utility>

namespace tinyinfer {

Buffer::Buffer(std::size_t size_bytes, std::shared_ptr<Allocator> allocator,
               std::size_t alignment)
    : size_bytes_(size_bytes),
      alignment_(alignment),
      allocator_(std::move(allocator)) {
    if (!allocator_) throw std::invalid_argument("buffer allocator must not be null");
    if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) {
        throw std::invalid_argument("buffer alignment must be a non-zero power of two");
    }
    if (size_bytes_ > 0) {
        data_ = allocator_->allocate(size_bytes_, alignment_);
        if (data_ == nullptr) throw std::bad_alloc();
    }
}

Buffer::~Buffer() {
    allocator_->deallocate(data_, size_bytes_, alignment_);
}

}  // namespace tinyinfer
