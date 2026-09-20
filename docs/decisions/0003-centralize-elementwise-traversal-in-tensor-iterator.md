# ADR-0003: Centralize elementwise traversal in TensorIterator

- Status: Accepted
- Date: 2026-09-20

## Context

An elementwise operator looks simple while all inputs are contiguous and have
the same shape:

```cpp
for (std::size_t i = 0; i < input.numel(); ++i) {
    output[i] = operation(input[i]);
}
```

That loop is not generally correct once TinyInfer supports Tensor views.
Elementwise operators must also handle:

- transposed inputs with non-contiguous strides;
- sliced inputs with gaps between visible elements;
- operands with different ranks;
- singleton-dimension and scalar broadcasting;
- invalid broadcasting combinations;
- size-one dimensions that do not affect addressing;
- a fast path when every operand is already contiguous.

Without a shared mechanism, Add, Subtract, Multiply, ReLU, GELU, and every
future elementwise operator would each need to implement the same shape
alignment, broadcasting, coordinate decomposition, offset calculation, and
fast-path selection. Small differences between those implementations would
make operator behavior inconsistent and multiply the test surface.

Graph shape inference has the same broadcasting question even though it does
not execute a kernel. The runtime therefore needs one layout-level definition
of elementwise iteration that is independent of the scalar operation and the
underlying data pointers.

## Decision

TinyInfer centralizes elementwise traversal in `TensorIterator` and centralizes
common unary and binary execution policy in `run_unary_kernel` and
`run_binary_kernel`.

The responsibilities are separated as follows:

```text
TensorLayout
    describes each operand's shape and physical strides
          |
          v
TensorIterator
    computes broadcast shape and per-operand iteration strides
    removes size-one dimensions
    merges dimensions when all operands advance linearly
    maps a logical output index to each operand's storage offset
    detects the all-contiguous fast path
          |
          v
Unary/Binary Kernel Runner
    validates dtype
    creates a contiguous output
    selects direct traversal or stride-aware traversal
    invokes the scalar operation
          |
          v
Operator
    supplies only f(x) or f(a, b)
```

`TensorIterator` owns layout metadata only. It does not own Tensor memory,
read data pointers, interpret dtype, allocate output, or implement an
operator's numerical formula.

### One broadcasted logical iteration space

The iterator right-aligns operand dimensions and computes a shared output
shape using NumPy-style broadcasting rules. An operand dimension is compatible
when it equals the output extent or is one.

For example:

```text
lhs shape:             [2, 1, 3]
rhs shape:             [1, 4, 1]
broadcast output:      [2, 4, 3]

lhs broadcast strides: [3, 0, 1]
rhs broadcast strides: [0, 1, 0]
```

A zero stride means that advancing along that output dimension reuses the same
input element. Broadcasting is therefore represented as address metadata; it
does not allocate an expanded Tensor.

Every output element has one logical linear index. `operand_offset(operand,
logical_index)` decomposes that index over the optimized iteration shape and
combines the coordinates with that operand's strides.

### Dimension coalescing is traversal optimization, not a semantic change

The public broadcast shape remains unchanged, but TensorIterator maintains a
second `iteration_shape`. Dimensions of size one are removed because they
never change an operand offset. Adjacent dimensions are merged only when every
operand advances linearly across their boundary.

For two contiguous operands:

```text
broadcast shape: [2, 3, 4]
iteration shape: [24]
strides:         [1]
```

For a broadcast or transposed layout, dimensions remain separate wherever a
merge would change addressing. Coalescing reduces coordinate calculations in
the general path without changing output shape or element order.

### The contiguous fast path remains centralized

When every operand maps logical index `i` directly to storage offset `i`, the
runner uses direct pointer traversal:

```cpp
output[i] = operation(lhs[i], rhs[i]);
```

Otherwise it uses TensorIterator offsets:

```cpp
output[i] = operation(
    lhs[iterator.operand_offset(0, i)],
    rhs[iterator.operand_offset(1, i)]);
```

Operators do not repeat this branch. Improvements to fast-path detection or
general traversal apply to the complete elementwise operator set.

### Operators provide scalar behavior only

The Add implementation is intentionally reduced to:

```cpp
return run_binary_kernel<float>(lhs, rhs,
                                [](float a, float b) { return a + b; });
```

Subtract and Multiply replace only the scalar lambda. ReLU and GELU use the
unary runner in the same way. This makes the numerical formula visible without
mixing it with layout mechanics.

### Shape inference reuses the same broadcast definition

Operator Schema constructs a TensorIterator from input TensorLayouts to infer
the output shape of broadcast elementwise operations and Gemm bias addition.
This keeps graph construction and runtime execution aligned: a shape accepted
by inference follows the same broadcast rules used by the CPU kernel.

## Alternatives considered

### Duplicate loops in every operator

Each operator could implement its own contiguous path, broadcast validation,
and strided loop. This is initially direct but creates repeated code and makes
bug fixes or optimizations operator-specific. New elementwise operators would
carry a large amount of non-numerical boilerplate.

### Require contiguous, equal-shaped inputs

Every operator could reject non-contiguous inputs and broadcasting, or call
`contiguous()` and explicitly expand operands first. Kernel loops would be
simple, but metadata-only views would trigger hidden allocations and
broadcasting would waste memory by materializing repeated values.

### Put iteration logic directly in Tensor

Tensor could expose a general multidimensional iterator. A single-Tensor
iterator does not naturally express a common broadcasted space across several
different layouts. It would also mix multi-operand execution policy into the
Tensor value abstraction.

### Let TensorIterator own pointers, dtype, output, and the kernel

A larger iterator object could perform complete operator execution. That would
couple reusable layout reasoning to Tensor memory and numerical dispatch. The
current split keeps TensorIterator usable by graph shape inference, where no
runtime Tensor data exists.

### Use recursive nested loops

Recursion can mirror tensor rank and handle arbitrary dimensions, but adds
control overhead and complicates a direct one-dimensional fast path. A logical
linear index provides one outer kernel loop while TensorIterator owns the
coordinate-to-offset mapping.

### Generate specialized loops for every layout

Code generation or rank-specialized templates could reduce general indexing
overhead further. That complexity is premature for the current learning
runtime. TensorIterator provides a correctness baseline and a stable place to
add specialized paths later.

## Consequences

### Benefits

- All current elementwise operators share broadcasting and strided-view
  semantics.
- Adding a unary or binary operator requires only its scalar computation.
- Broadcasting uses zero strides instead of materialized expanded tensors.
- Transposed and sliced inputs work without first becoming contiguous.
- Fast-path selection and dimension coalescing are implemented once.
- Graph shape inference and runtime execution reuse one broadcast definition.
- TensorIterator can be tested without allocating or reading Tensor data.
- Future vectorized or parallel elementwise paths have one integration point.

### Costs

- The general path performs logical-index decomposition and offset calculation
  for each operand and output element.
- `operand_offset` currently performs division and modulo operations, which can
  dominate very cheap scalar operations on non-contiguous inputs.
- The runner always materializes a new contiguous output; in-place and
  arbitrary-strided outputs are not represented.
- Separate unary and binary runners do not yet cover ternary, variadic, mixed
  dtype, or type-promoting operators.
- A fully contiguous input set gets a direct loop, but there are not yet
  specialized vectorized paths for partially contiguous or scalar-broadcast
  cases.
- Reusing TensorIterator for shape inference is convenient but constructs more
  iteration metadata than shape-only inference strictly requires.

## Current constraints

- All strides are non-negative because TensorLayout does not yet support
  negative-stride views.
- Outputs produced by the runners are newly allocated and contiguous.
- Operands must have the storage type selected by the runner template; general
  dtype promotion is not implemented.
- TensorIterator represents input layouts only. Output indexing is currently a
  direct contiguous index.
- Dimension coalescing requires every operand to cross a boundary linearly, so
  one irregular operand prevents that merge for the shared iteration space.

## Validation in the codebase

- `tests/tensor_iterator_test.cpp` covers right-aligned broadcasting, zero
  strides, scalar operands, transposed layouts, dimension coalescing,
  contiguous detection, invalid shapes, and bounds errors.
- `tests/kernel_runner_test.cpp` covers reusable unary and binary formulas,
  contiguous and transposed inputs, broadcasting, dtype validation, and
  incompatible shapes.
- Elementwise tests exercise Add, Subtract, Multiply, ReLU, and GELU through
  the shared runner rather than operator-specific traversal code.

The current implementation is described in
[`../architecture/tensor.md`](../architecture/tensor.md). This decision builds
on [ADR-0001](0001-separate-buffer-storage-and-tensor-layout.md), which makes
layout independently reusable, and [ADR-0002](0002-deep-copy-tensors-and-share-storage-explicitly.md),
which defines when Tensor memory is independent or shared.
