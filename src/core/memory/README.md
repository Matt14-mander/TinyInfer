# Memory

The memory module now defines:

- `Allocator`: an interface for aligned byte allocation and deallocation.
- `CpuAllocator`: the default implementation backed by aligned C++ allocation.
- `ArenaAllocator`: a fixed-capacity bump allocator for groups of temporary tensors.
- `Buffer`: RAII ownership of one allocator-backed byte buffer.
- `Storage`: a shared reference to a byte range within a Buffer.
- `default_allocator()`: a process-wide shared CPU allocator.

Tensor retains a shared reference to its allocator so the allocator remains alive until every buffer it created has been released. Tensor construction, deep copying, and contiguous materialization all use this interface. Zero-byte tensors do not call the allocator.

## Buffer and Storage

`Buffer` is the only layer that directly calls `Allocator::allocate` and `Allocator::deallocate`. It is non-copyable and owns the allocation for its full lifetime. `Storage` holds a shared Buffer plus a byte offset and byte length, allowing multiple future tensor views to reference different ranges of one allocation safely.

Tensor now owns a `Storage` rather than an allocator and raw pointer separately. Tensor deep copy creates a new Buffer, transpose keeps the same Buffer, and contiguous materialization replaces it with a new Buffer. Zero-byte Storage still retains its allocator through a zero-capacity Buffer.

Tensor `view()` and `narrow()` now copy Storage metadata instead of Buffer contents. Reshaped views share the full range, while narrowed views use a validated byte offset and span. Ordinary Tensor copy construction deliberately materializes an independent contiguous Buffer.

## Arena allocation

`ArenaAllocator` obtains one buffer from a backing allocator and serves aligned suballocations by advancing an offset. Individual deallocation only updates the active-allocation count; memory becomes reusable after every allocation is released and `reset()` is called. Reset is rejected while live Tensor buffers remain, preventing accidental invalidation.

The arena reports capacity, used and remaining bytes, peak usage, total allocation count, and active allocation count. It is intentionally single-threaded at this stage.

Pooling, device allocators, and graph-driven memory planning remain future work.
