# Memory

The memory module now defines:

- `Allocator`: an interface for aligned byte allocation and deallocation.
- `CpuAllocator`: the default implementation backed by aligned C++ allocation.
- `ArenaAllocator`: a fixed-capacity bump allocator for groups of temporary tensors.
- `default_allocator()`: a process-wide shared CPU allocator.

Tensor retains a shared reference to its allocator so the allocator remains alive until every buffer it created has been released. Tensor construction, deep copying, and contiguous materialization all use this interface. Zero-byte tensors do not call the allocator.

## Arena allocation

`ArenaAllocator` obtains one buffer from a backing allocator and serves aligned suballocations by advancing an offset. Individual deallocation only updates the active-allocation count; memory becomes reusable after every allocation is released and `reset()` is called. Reset is rejected while live Tensor buffers remain, preventing accidental invalidation.

The arena reports capacity, used and remaining bytes, peak usage, total allocation count, and active allocation count. It is intentionally single-threaded at this stage.

Pooling, device allocators, and graph-driven memory planning remain future work.
