# ADR-0002: Deep-copy Tensors and share Storage explicitly through views

- Status: Accepted
- Date: 2026-09-20

## Context

After Buffer ownership became shareable, the default meaning of copying a
`Tensor` had to be chosen. A copied C++ object could either share the same
Buffer or allocate an independent Buffer containing the same logical values.

This choice is observable because TinyInfer currently exposes mutable element
access and in-place metadata or materialization operations. If ordinary copies
shared Storage, seemingly local code could mutate another Tensor:

```cpp
auto copy = original;
copy.at({0, 0}) = 42.0F;
```

For a learning-oriented runtime, an assignment that silently introduces
aliasing makes ownership, mutation, and debugging harder to reason about. At
the same time, always copying would make reshape, narrow, and slice needlessly
expensive and would prevent TinyInfer from representing non-owning tensor
views.

TinyInfer therefore needs both independent values and shared views, with a
clear syntax boundary between them.

## Decision

Normal C++ copy construction and copy assignment give `Tensor` value
semantics: the destination receives an independent, contiguous Buffer with the
same dtype, shape, and logical values.

```cpp
auto copy = tensor;  // Independent data.
copy.at({0, 0}) = 42.0F;
// tensor is unchanged.
```

Storage sharing is introduced only by an operation whose name expresses view
semantics:

```cpp
auto reshaped = tensor.view({3, 2});
auto row = tensor.narrow(0, 1, 1);
auto columns = tensor.slice(1, 0, 6, 2);
```

These results own independent Tensor metadata but intentionally share the
source Buffer. Mutation through either alias is visible through the other.

The semantic boundary is:

```text
Tensor copy       = same logical values, independent Buffer
Tensor move       = transfer the existing Tensor representation
Tensor view       = independent metadata, explicitly shared Buffer
contiguous()      = materialize and detach only when layout requires it
```

### Copy construction materializes logical order

The copy constructor allocates `numel * dtype_size` bytes using the source
Tensor's allocator. It iterates over the source's logical indices through
`TensorLayout::storage_offset` and writes the values into contiguous row-major
order.

Consequently, copying a non-contiguous transpose or slice does more than copy
its visible byte range. It preserves the Tensor's logical values while
removing gaps, offsets, and non-contiguous strides:

```text
source view: shape=[2, 3], strides=[6, 2], shared Buffer
copied value: shape=[2, 3], strides=[3, 1], independent Buffer
```

This makes the copy a self-contained value rather than a duplicate handle to
the source view description.

### Copy assignment uses copy-and-swap

Copy assignment first constructs a complete deep copy and then swaps it into
the destination. This provides self-assignment safety and the strong exception
guarantee: if allocation or copying fails, the destination remains unchanged.

### Move operations remain cheap

Move construction and move assignment transfer TensorLayout, dtype, and
Storage without copying element data. APIs that take a Tensor by value can
therefore accept either an intentional copy from an lvalue or a cheap transfer
from an rvalue:

```cpp
context.set_value(id, std::move(result));
```

This is the normal path used to pass newly produced operator outputs into the
execution context.

### View creation bypasses normal copy semantics deliberately

The internal `ViewTag` constructor accepts a copied Storage handle. Because
Storage contains a `shared_ptr<Buffer>`, the new Tensor safely shares Buffer
ownership. The public `view`, `narrow`, and `slice` functions are the explicit
entry points that use this path.

This separation prevents an implementation detail such as Storage being
copyable from changing the public meaning of copying a Tensor.

## Alternatives considered

### Shallow copy by default

Every Tensor copy could share Storage, similar to copying a reference-counted
tensor handle in some production frameworks. It is efficient, but ordinary
C++ assignment would create mutable aliasing without revealing that fact at
the call site. TinyInfer chooses simpler value reasoning over this default.

### Explicit `clone()` with shallow C++ copies

Tensor copies could be shallow while callers request independent memory with
`clone()`. This can be performant when users understand framework-specific
semantics, but it makes C++ copy syntax behave unlike a conventional value and
makes accidental input mutation easier in this educational codebase.

### Copy-on-write

Copies could initially share a Buffer and detach on the first mutation. That
can avoid unused copies, but mutable pointers returned by `data()` make writes
hard to intercept. Copy-on-write also hides allocation latency at mutation
sites and adds state and synchronization complexity.

### Make Tensor move-only

Deleting copy operations would force all duplication to be explicit. This
provides a strong ownership signal but makes graph constants, test fixtures,
containers, and general API composition more cumbersome. TinyInfer retains a
regular copyable value type while keeping the expensive semantics documented.

### Always deep-copy, including view operations

This removes aliasing entirely, but makes transpose-like metadata operations
and slicing allocate memory. It also prevents the runtime from representing
the zero-copy views required for stride-aware kernels and later memory
planning.

## Consequences

### Benefits

- Ordinary copies can be mutated independently and are easy to reason about.
- Aliasing appears only at explicitly named view operations.
- Copying a non-contiguous view produces a compact, contiguous value.
- A copied Tensor does not retain a large source Buffer just to expose a small
  slice.
- Copy assignment remains exception-safe through copy-and-swap.
- Move semantics provide a zero-copy path for ownership transfer between
  operators, graphs, and execution contexts.

### Costs

- Copying a Tensor is proportional to its logical element count and allocates
  a new Buffer.
- Innocent-looking lvalue copies such as `auto temporary = input;` may be
  expensive for large tensors.
- Graph construction and ExecutionContext initialization can duplicate
  constants when callers do not move them.
- Copying a non-contiguous Tensor performs stride-aware element traversal and
  may be significantly slower than a contiguous memory copy.
- View mutation introduces real aliasing, so callers must treat view creation
  as an intentional sharing decision.
- The current mutable view returned from a `const Tensor` does not provide
  read-only aliasing. A future const-view abstraction may be needed as the API
  matures.

## Usage rules

- Use ordinary copy syntax when an independent value is required.
- Use `std::move` when transferring a Tensor that will no longer be used.
- Use `view`, `narrow`, or `slice` only when shared mutations and lifetime are
  understood.
- Use `contiguous()` to materialize the current logical order when a kernel
  requires contiguous storage.
- Do not introduce an implicit shallow-copy path merely to optimize a hot
  call site. Prefer an explicit view, move, output buffer, or API redesign.

## Validation in the codebase

- `tests/tensor_value_test.cpp` verifies independent copy construction and
  assignment, self-assignment, and pointer-preserving moves.
- `tests/tensor_view_test.cpp` verifies shared mutation through views, deep-copy
  detachment, contiguous copies of non-contiguous views, and view lifetime
  after the source Tensor is destroyed.
- `tests/tensor_slice_test.cpp` verifies that slices share their source Buffer
  while copying a slice materializes an independent contiguous Tensor.
- `tests/tensor_contiguous_test.cpp` verifies that materialization preserves
  logical values and replaces storage only when needed.

This decision builds on
[ADR-0001](0001-separate-buffer-storage-and-tensor-layout.md), which separates
physical allocation ownership, visible byte ranges, and logical layout.
