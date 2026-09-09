#include "tinyinfer/core/memory/arena_allocator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace tinyinfer {
namespace {

bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

ArenaAllocator::ArenaAllocator(std::size_t capacity,
                               std::shared_ptr<Allocator> backing_allocator,
                               std::size_t alignment)
    : capacity_(capacity),
      alignment_(alignment),
      backing_allocator_(std::move(backing_allocator)) {
    if (!backing_allocator_) throw std::invalid_argument("arena backing allocator must not be null");
    if (!is_power_of_two(alignment_)) {
        throw std::invalid_argument("arena alignment must be a non-zero power of two");
    }
    if (capacity_ > 0) {
        buffer_ = backing_allocator_->allocate(capacity_, alignment_);
        if (buffer_ == nullptr) throw std::bad_alloc();
    }
}

ArenaAllocator::~ArenaAllocator() {
    backing_allocator_->deallocate(buffer_, capacity_, alignment_);
}

void* ArenaAllocator::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return nullptr;
    if (!is_power_of_two(alignment)) {
        throw std::invalid_argument("allocation alignment must be a non-zero power of two");
    }
    if (alignment > alignment_) {
        throw std::invalid_argument("requested alignment exceeds arena alignment");
    }
    if (offset_ > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::bad_alloc();
    }

    const auto aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned_offset > capacity_ || bytes > capacity_ - aligned_offset) {
        throw std::bad_alloc();
    }

    auto* pointer = static_cast<unsigned char*>(buffer_) + aligned_offset;
    offset_ = aligned_offset + bytes;
    peak_used_ = std::max(peak_used_, offset_);
    ++allocation_count_;
    ++active_allocations_;
    return pointer;
}

void ArenaAllocator::deallocate(void* pointer, std::size_t, std::size_t) noexcept {
    if (pointer == nullptr || buffer_ == nullptr || active_allocations_ == 0) return;

    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    if (address >= begin && address < begin + capacity_) --active_allocations_;
}

void ArenaAllocator::reset() {
    if (active_allocations_ != 0) {
        throw std::logic_error("cannot reset arena while allocations are still active");
    }
    offset_ = 0;
}

}  // namespace tinyinfer
