#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

#include "tinyinfer/core/memory/allocator.h"

namespace tinyinfer {

class ArenaAllocator final : public Allocator {
public:
    explicit ArenaAllocator(
        std::size_t capacity,
        std::shared_ptr<Allocator> backing_allocator = default_allocator(),
        std::size_t alignment = alignof(std::max_align_t));
    ~ArenaAllocator() override;

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;
    ArenaAllocator(ArenaAllocator&&) = delete;
    ArenaAllocator& operator=(ArenaAllocator&&) = delete;

    std::string_view name() const noexcept override { return "arena"; }
    void* allocate(std::size_t bytes, std::size_t alignment) override;
    void deallocate(void* pointer, std::size_t bytes,
                    std::size_t alignment) noexcept override;

    void reset();

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t used() const noexcept { return offset_; }
    std::size_t remaining() const noexcept { return capacity_ - offset_; }
    std::size_t peak_used() const noexcept { return peak_used_; }
    std::size_t allocation_count() const noexcept { return allocation_count_; }
    std::size_t active_allocations() const noexcept { return active_allocations_; }
    const std::shared_ptr<Allocator>& backing_allocator() const noexcept {
        return backing_allocator_;
    }

private:
    std::size_t capacity_{};
    std::size_t alignment_{};
    std::size_t offset_{};
    std::size_t peak_used_{};
    std::size_t allocation_count_{};
    std::size_t active_allocations_{};
    std::shared_ptr<Allocator> backing_allocator_;
    void* buffer_{};
};

}  // namespace tinyinfer
