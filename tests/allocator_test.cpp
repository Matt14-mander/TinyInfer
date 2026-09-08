#include <cassert>
#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include "tinyinfer/core/memory/allocator.h"
#include "tinyinfer/core/tensor.h"

namespace {

class CountingAllocator final : public tinyinfer::Allocator {
public:
    std::string_view name() const noexcept override { return "counting"; }

    void* allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocation_count;
        allocated_bytes += bytes;
        ++live_allocations;
        return ::operator new(bytes, std::align_val_t(alignment));
    }

    void deallocate(void* pointer, std::size_t bytes,
                    std::size_t alignment) noexcept override {
        if (pointer == nullptr) return;
        ++deallocation_count;
        deallocated_bytes += bytes;
        --live_allocations;
        ::operator delete(pointer, std::align_val_t(alignment));
    }

    std::size_t allocation_count{};
    std::size_t deallocation_count{};
    std::size_t allocated_bytes{};
    std::size_t deallocated_bytes{};
    std::size_t live_allocations{};
};

}  // namespace

int main() {
    const auto default_cpu = tinyinfer::default_allocator();
    assert(default_cpu);
    assert(default_cpu->name() == "cpu");
    assert(default_cpu == tinyinfer::default_allocator());

    auto allocator = std::make_shared<CountingAllocator>();
    {
        tinyinfer::Tensor tensor({2, 3}, tinyinfer::DataType::Float32, allocator);
        assert(tensor.allocator() == allocator);
        assert(allocator->allocation_count == 1);
        assert(allocator->live_allocations == 1);
        assert(allocator->allocated_bytes == tensor.size_bytes());
        for (std::size_t i = 0; i < tensor.numel(); ++i) tensor.at<float>(i) = static_cast<float>(i + 1);

        auto copy = tensor;
        assert(copy.allocator() == allocator);
        assert(copy.data() != tensor.data());
        assert(allocator->allocation_count == 2);
        assert(allocator->live_allocations == 2);

        tensor.transpose(0, 1);
        const void* old_storage = tensor.data();
        tensor.contiguous();
        assert(tensor.data() != old_storage);
        assert(allocator->allocation_count == 3);
        assert(allocator->deallocation_count == 1);
        assert(allocator->live_allocations == 2);
        assert(tensor.to_string() ==
               "Tensor(shape=[3, 2], dtype=float32, data=[[1, 4], [2, 5], [3, 6]])");
    }
    assert(allocator->live_allocations == 0);
    assert(allocator->allocation_count == allocator->deallocation_count);
    assert(allocator->allocated_bytes == allocator->deallocated_bytes);

    tinyinfer::Tensor empty({0, 4}, tinyinfer::DataType::Float32, allocator);
    assert(empty.data() == nullptr);
    assert(allocator->live_allocations == 0);

    bool rejected_null_allocator = false;
    try {
        tinyinfer::Tensor invalid({1}, tinyinfer::DataType::Float32, nullptr);
    } catch (const std::invalid_argument&) {
        rejected_null_allocator = true;
    }
    assert(rejected_null_allocator);

    tinyinfer::CpuAllocator cpu;
    bool rejected_alignment = false;
    try {
        (void)cpu.allocate(16, 3);
    } catch (const std::invalid_argument&) {
        rejected_alignment = true;
    }
    assert(rejected_alignment);
    assert(cpu.allocate(0, alignof(std::max_align_t)) == nullptr);
    return 0;
}
