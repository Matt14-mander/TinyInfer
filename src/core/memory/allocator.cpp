#include "tinyinfer/core/memory/allocator.h"

#include <new>
#include <stdexcept>

namespace tinyinfer {

void* CpuAllocator::allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) return nullptr;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("allocation alignment must be a non-zero power of two");
    }
    return ::operator new(bytes, std::align_val_t(alignment));
}

void CpuAllocator::deallocate(void* pointer, std::size_t, std::size_t alignment) noexcept {
    if (pointer == nullptr) return;
    ::operator delete(pointer, std::align_val_t(alignment));
}

std::shared_ptr<Allocator> default_allocator() {
    static const auto allocator = std::make_shared<CpuAllocator>();
    return allocator;
}

}  // namespace tinyinfer
