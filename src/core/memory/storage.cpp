#include "tinyinfer/core/memory/storage.h"

#include <stdexcept>
#include <utility>

namespace tinyinfer {

// 基于偏移量的共享内存视图，通过智能指针管理内存，确保在多个Storage实例之间共享Buffer时不会发生内存泄漏或悬空指针问题。
// 分配全新的内存缓冲区时，Storage会创建一个新的Buffer实例，并使用指定的分配器-allocator和对齐方式-alignment进行内存分配。
// 全量构造函数
Storage::Storage(std::size_t size_bytes, std::shared_ptr<Allocator> allocator,
                 std::size_t alignment)
    : buffer_(std::make_shared<Buffer>(size_bytes, std::move(allocator), alignment)),
      size_bytes_(size_bytes) {}

// 当需要基于已有的Buffer对象创建Storage实例时，可以使用视图构造函数。该构造函数允许指定一个偏移量和大小，从而在不复制数据的情况下创建对原始缓冲区的子视图。这对于处理大型数据集或需要在不同组件之间共享内存的场景非常有用。
// 视图构造函数
Storage::Storage(std::shared_ptr<Buffer> buffer, std::size_t byte_offset,
                 std::size_t size_bytes)
    : buffer_(std::move(buffer)), byte_offset_(byte_offset), size_bytes_(size_bytes) {
    if (!buffer_) throw std::invalid_argument("storage buffer must not be null"); // 真正持有内存的底层缓冲区对象
    if (byte_offset_ > buffer_->size_bytes() || 
        size_bytes_ > buffer_->size_bytes() - byte_offset_) { // 检查偏移量和大小是否超出缓冲区容量
        throw std::out_of_range("storage range exceeds buffer capacity");
    }
}

// 数据访问接口
void* Storage::data() noexcept {
    if (!buffer_ || !buffer_->data()) return nullptr;
    return static_cast<unsigned char*>(buffer_->data()) + byte_offset_;
}

const void* Storage::data() const noexcept {
    if (!buffer_ || !buffer_->data()) return nullptr;
    return static_cast<const unsigned char*>(buffer_->data()) + byte_offset_;
}

// 获取分配器
const std::shared_ptr<Allocator>& Storage::allocator() const {
    if (!buffer_) throw std::logic_error("empty storage has no allocator");
    return buffer_->allocator();
}

}  // namespace tinyinfer
