#pragma once

#include <cstddef>
#include <memory>
#include <string_view>

namespace tinyinfer {

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual void* allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void deallocate(void* pointer, std::size_t bytes,
                            std::size_t alignment) noexcept = 0;
};

class CpuAllocator final : public Allocator {
public:
    std::string_view name() const noexcept override { return "cpu"; }
    void* allocate(std::size_t bytes, std::size_t alignment) override;
    void deallocate(void* pointer, std::size_t bytes,
                    std::size_t alignment) noexcept override;
};

std::shared_ptr<Allocator> default_allocator();

}  // namespace tinyinfer
