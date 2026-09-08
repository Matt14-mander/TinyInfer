# Memory

The memory module now defines:

- `Allocator`: an interface for aligned byte allocation and deallocation.
- `CpuAllocator`: the default implementation backed by aligned C++ allocation.
- `default_allocator()`: a process-wide shared CPU allocator.

Tensor retains a shared reference to its allocator so the allocator remains alive until every buffer it created has been released. Tensor construction, deep copying, and contiguous materialization all use this interface. Zero-byte tensors do not call the allocator.

Arena allocation, pooling, device allocators, and memory planning remain future work.
