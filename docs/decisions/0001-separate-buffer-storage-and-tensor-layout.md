# ADR-0001: Separate Buffer, Storage, and TensorLayout

- Status: Accepted
- Date: 2026-09-20

## Context

The first Tensor prototype could keep a data pointer, shape, strides, dtype,
and allocation logic in one class. That representation is sufficient while
every Tensor owns one contiguous allocation. It stops being a good model once
TinyInfer needs transpose, reshape, narrow, slice, and other views.

Those operations change different parts of a Tensor independently:

- `transpose` changes shape and strides but not the allocation;
- `view` changes the logical shape but shares the same bytes;
- `narrow` and `slice` share an allocation but start at a byte offset and may
  expose only a subrange;
- `contiguous` preserves logical values while replacing both the allocation
  and layout;
- a view must remain alive even after the Tensor that created it is destroyed;
- different allocators must remain responsible for releasing the memory they
  allocated.

Keeping all of these concerns directly inside `Tensor` would mix logical
indexing, byte-range validation, allocation ownership, and allocator lifetime.
It would also make metadata-only operations difficult to distinguish from
operations that allocate or copy memory.

## Decision

TinyInfer separates the Tensor representation into three independently
testable abstractions and composes them in `Tensor`:

```text
Tensor
├── TensorLayout    logical shape, strides, and index mapping
├── DataType        interpretation and size of one element
└── Storage         visible byte range
    └── shared Buffer
        └── Allocator-owned physical allocation
```
```text
Tensor
├── TensorLayout：逻辑上怎么看数据
├── DataType：每个元素如何解释
└── Storage：当前 Tensor 能访问哪段内存
    └── Buffer：实际内存由谁拥有
        └── Allocator：内存如何申请和释放
```

### Buffer owns one physical allocation

`Buffer` calls an `Allocator` once during construction and returns the same
pointer to that allocator during destruction. It stores the allocation size
and alignment and is deliberately non-copyable and non-movable.

Buffer is shared through `std::shared_ptr`. The final Storage that references
the Buffer determines when the physical allocation is released. A Tensor view
therefore remains valid after its source Tensor object is destroyed.

### Storage describes a byte range

`Storage` holds a shared Buffer together with a byte offset and a byte size. It
does not own a second allocation and it does not know about Tensor dimensions,
strides, or dtype.

Constructing a Storage range validates that:

```text
byte_offset + size_bytes <= buffer.size_bytes
```

This lets a full Tensor and multiple views share one Buffer while exposing
different starting addresses and bounded ranges. `Storage::data()` returns the
Buffer address plus the Storage byte offset.

### TensorLayout describes logical indexing

`TensorLayout` is a memory-independent value object containing shape and
element strides. It calculates rank, logical element count, required storage
span, contiguity, and the relative element offset for a logical coordinate.

It does not allocate memory, retain a pointer, or determine dtype. This allows
the same layout rules to be tested independently and reused by
`TensorIterator`, `ReductionIterator`, graph validation, and shape inference.

The layout offset is relative to `Storage::data()`. The final byte address of
an element is therefore:

```text
buffer address
+ storage byte offset
+ layout element offset * dtype size
```
```text
最终地址
= Buffer 起始地址
+ Storage 字节偏移
+ TensorLayout 元素偏移 × dtype 字节数
```

### Tensor composes the parts

`Tensor` is the user-facing object. It combines a TensorLayout, a DataType, and
a Storage. Metadata-only operations replace the layout or create another
Tensor with shared Storage. Materializing operations allocate a new Buffer and
replace the Storage and layout together.

Normal C++ Tensor copying remains a deep copy by project policy. Explicit view
operations are the only public path that shares Buffer ownership between
Tensor objects.

## Alternatives considered

### One monolithic Tensor class

Tensor could directly own its pointer, allocator, shape, strides, offset, and
allocation size. This is simpler for the first contiguous implementation, but
every view operation would have to reproduce ownership and bounds rules.
Logical layout code would also remain coupled to allocation code and would be
harder to reuse outside Tensor.

### Every Tensor owns an independent allocation

Transpose, slice, and reshape could always copy data into new buffers. This
avoids shared lifetime management, but turns metadata operations into hidden
allocations, uses more memory, and prevents TinyInfer from learning and
implementing the view semantics used by practical runtimes.

### Share a raw pointer plus an ownership flag

A Tensor could store a raw pointer and a Boolean such as `owns_data`. This does
not express multiple surviving views safely: the original owner may die first,
ownership transfer becomes fragile, and custom allocator state may be lost.
Reference-counted Buffer ownership represents the actual lifetime directly.

### Combine Buffer and Storage

Each view could hold a shared allocation object plus offset fields directly.
That removes one type but conflates the lifetime of the full allocation with a
particular visible range. A separate Storage gives range validation and view
construction one boundary while keeping Buffer responsible only for RAII.

### Put byte offset inside TensorLayout

Layout could contain both strides and the view's starting byte offset. That
would couple an otherwise dtype-independent element mapping to physical byte
addressing. Keeping the starting byte in Storage allows TensorLayout to remain
a reusable description of relative logical indexing.

## Consequences

### Benefits

- Transpose and reshape can remain metadata-only when their layout rules allow
  it.
- Narrow and slice can create bounded, zero-copy views.
- Views safely outlive the Tensor that created them.
- Allocation and deallocation remain paired with the same Allocator.
- Layout arithmetic and overflow checks can be tested without allocating
  memory.
- Iterators and graph code can reuse TensorLayout without depending on Tensor
  ownership.
- Future memory planners and backend-specific allocators have a clear boundary
  below Tensor semantics.

### Costs

- Tensor construction and view creation involve more types and invariants.
- Buffer lifetime uses atomic-capable shared ownership rather than unique
  ownership with no reference count.
- Correctness depends on keeping the Layout storage span inside the Storage
  range; view construction must validate this composition.
- Aliasing is now possible and must be explicit in operator and mutation
  reasoning.
- Buffer and Storage are currently CPU-addressable abstractions. Device memory
  may later require extending their interfaces without collapsing these
  responsibilities back into Tensor.

## Current constraints

- TensorLayout currently accepts non-negative strides only.
- `view` requires a contiguous source layout, while `narrow` and positive-step
  `slice` can preserve non-contiguous strides.
- `contiguous()` materializes logical order into a new Buffer when necessary.
- Storage ranges and Layout spans are validated when TinyInfer creates a view;
  arbitrary public `as_strided` construction is intentionally unavailable.

## Validation in the codebase

- `tests/buffer_storage_test.cpp` checks allocator pairing, shared Buffer
  ranges, byte offsets, bounds rejection, and Buffer replacement during
  materialization.
- `tests/tensor_layout_test.cpp` checks indexing, storage spans, contiguity,
  reshape and transpose metadata, and arithmetic overflow.
- `tests/tensor_view_test.cpp` checks shared mutation, source-independent view
  lifetime, deep-copy detachment, narrow, and non-contiguous materialization.

The corresponding current-state description lives in
[`../architecture/tensor.md`](../architecture/tensor.md).
