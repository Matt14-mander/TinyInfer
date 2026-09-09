#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include "tinyinfer/core/memory/buffer.h"
#include "tinyinfer/core/memory/storage.h"
#include "tinyinfer/core/tensor.h"

namespace {

class CountingAllocator final : public tinyinfer::Allocator {
public:
    std::string_view name() const noexcept override { return "buffer-counting"; }

    void* allocate(std::size_t bytes, std::size_t alignment) override {
        ++allocations;
        allocated_bytes += bytes;
        return ::operator new(bytes, std::align_val_t(alignment));
    }

    void deallocate(void* pointer, std::size_t bytes,
                    std::size_t alignment) noexcept override {
        if (!pointer) return;
        ++deallocations;
        deallocated_bytes += bytes;
        ::operator delete(pointer, std::align_val_t(alignment));
    }

    std::size_t allocations{};
    std::size_t deallocations{};
    std::size_t allocated_bytes{};
    std::size_t deallocated_bytes{};
};

}  // namespace

int main() {
    auto allocator = std::make_shared<CountingAllocator>();
    {
        auto buffer = std::make_shared<tinyinfer::Buffer>(64, allocator, 16);
        assert(buffer->data() != nullptr);
        assert(buffer->size_bytes() == 64);
        assert(buffer->alignment() == 16);
        assert(buffer->allocator() == allocator);
        assert(reinterpret_cast<std::uintptr_t>(buffer->data()) % 16 == 0);
        assert(allocator->allocations == 1);

        tinyinfer::Storage full(buffer, 0, 64);
        tinyinfer::Storage slice(buffer, 16, 24);
        assert(full.buffer() == slice.buffer());
        assert(full.data() == buffer->data());
        assert(slice.data() == static_cast<unsigned char*>(buffer->data()) + 16);
        assert(slice.byte_offset() == 16);
        assert(slice.size_bytes() == 24);
        assert(slice.allocator() == allocator);

        bool rejected_range = false;
        try {
            tinyinfer::Storage invalid(buffer, 48, 17);
        } catch (const std::out_of_range&) {
            rejected_range = true;
        }
        assert(rejected_range);
    }
    assert(allocator->allocations == 1);
    assert(allocator->deallocations == 1);
    assert(allocator->allocated_bytes == allocator->deallocated_bytes);

    tinyinfer::Storage empty(0, allocator);
    assert(empty.valid());
    assert(empty.data() == nullptr);
    assert(empty.size_bytes() == 0);
    assert(empty.allocator() == allocator);

    tinyinfer::Tensor tensor({2, 3}, tinyinfer::DataType::Float32, allocator);
    assert(tensor.storage().valid());
    assert(tensor.storage().data() == tensor.data());
    assert(tensor.storage().size_bytes() == tensor.size_bytes());
    assert(tensor.storage().byte_offset() == 0);
    assert(tensor.storage().allocator() == allocator);

    const auto original_buffer = tensor.storage().buffer();
    auto copy = tensor;
    assert(copy.storage().buffer() != original_buffer);
    assert(copy.allocator() == allocator);

    tensor.transpose(0, 1);
    assert(tensor.storage().buffer() == original_buffer);
    tensor.contiguous();
    assert(tensor.storage().buffer() != original_buffer);
    assert(tensor.storage().size_bytes() == tensor.size_bytes());

    bool rejected_null_buffer = false;
    try {
        tinyinfer::Storage invalid(nullptr, 0, 0);
    } catch (const std::invalid_argument&) {
        rejected_null_buffer = true;
    }
    assert(rejected_null_buffer);
    return 0;
}
