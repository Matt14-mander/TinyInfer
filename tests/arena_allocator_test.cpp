#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include "tinyinfer/core/memory/arena_allocator.h"
#include "tinyinfer/core/tensor.h"

namespace {

class CountingBackingAllocator final : public tinyinfer::Allocator {
public:
    std::string_view name() const noexcept override { return "counting-backing"; }

    void* allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocations;
        ++live_allocations;
        return ::operator new(bytes, std::align_val_t(alignment));
    }

    void deallocate(void* pointer, std::size_t,
                    std::size_t alignment) noexcept override {
        if (pointer == nullptr) return;
        ++deallocations;
        --live_allocations;
        ::operator delete(pointer, std::align_val_t(alignment));
    }

    std::size_t allocations{};
    std::size_t deallocations{};
    std::size_t live_allocations{};
};

}  // namespace

int main() {
    {
        tinyinfer::ArenaAllocator arena(64, tinyinfer::default_allocator(), 16);
        assert(arena.name() == "arena");
        assert(arena.capacity() == 64);
        assert(arena.used() == 0);
        assert(arena.remaining() == 64);

        void* first = arena.allocate(3, 1);
        void* second = arena.allocate(8, 8);
        assert(first != nullptr);
        assert(second != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(second) % 8 == 0);
        assert(arena.used() == 16);
        assert(arena.remaining() == 48);
        assert(arena.peak_used() == 16);
        assert(arena.allocation_count() == 2);
        assert(arena.active_allocations() == 2);

        bool rejected_live_reset = false;
        try {
            arena.reset();
        } catch (const std::logic_error&) {
            rejected_live_reset = true;
        }
        assert(rejected_live_reset);

        arena.deallocate(second, 8, 8);
        arena.deallocate(first, 3, 1);
        assert(arena.active_allocations() == 0);
        assert(arena.used() == 16);
        arena.reset();
        assert(arena.used() == 0);
        assert(arena.remaining() == 64);
        assert(arena.peak_used() == 16);

        bool exhausted = false;
        try {
            (void)arena.allocate(65, 1);
        } catch (const std::bad_alloc&) {
            exhausted = true;
        }
        assert(exhausted);
    }

    auto backing = std::make_shared<CountingBackingAllocator>();
    {
        auto arena = std::make_shared<tinyinfer::ArenaAllocator>(256, backing);
        assert(backing->allocations == 1);
        assert(backing->live_allocations == 1);
        assert(arena->backing_allocator() == backing);

        const void* first_tensor_storage = nullptr;
        {
            tinyinfer::Tensor tensor({2, 3}, tinyinfer::DataType::Float32, arena);
            first_tensor_storage = tensor.data();
            for (std::size_t i = 0; i < tensor.numel(); ++i) {
                tensor.at<float>(i) = static_cast<float>(i + 1);
            }
            auto copy = tensor;
            assert(arena->active_allocations() == 2);
            assert(backing->allocations == 1);

            tensor.transpose(0, 1);
            tensor.contiguous();
            assert(arena->active_allocations() == 2);
            assert(backing->allocations == 1);
            assert(tensor.to_string() ==
                   "Tensor(shape=[3, 2], dtype=float32, data=[[1, 4], [2, 5], [3, 6]])");

            bool rejected_tensor_reset = false;
            try {
                arena->reset();
            } catch (const std::logic_error&) {
                rejected_tensor_reset = true;
            }
            assert(rejected_tensor_reset);
        }

        assert(arena->active_allocations() == 0);
        assert(arena->used() > 0);
        arena->reset();
        assert(arena->used() == 0);

        tinyinfer::Tensor reused({2, 3}, tinyinfer::DataType::Float32, arena);
        assert(reused.data() == first_tensor_storage);
        assert(backing->allocations == 1);
    }
    assert(backing->live_allocations == 0);
    assert(backing->allocations == 1);
    assert(backing->deallocations == 1);

    bool rejected_null_backing = false;
    try {
        tinyinfer::ArenaAllocator invalid(64, nullptr);
    } catch (const std::invalid_argument&) {
        rejected_null_backing = true;
    }
    assert(rejected_null_backing);

    bool rejected_bad_alignment = false;
    try {
        tinyinfer::ArenaAllocator invalid(64, tinyinfer::default_allocator(), 3);
    } catch (const std::invalid_argument&) {
        rejected_bad_alignment = true;
    }
    assert(rejected_bad_alignment);
    return 0;
}
