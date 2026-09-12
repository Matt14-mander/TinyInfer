# Operators

`kernel_runner.h` centralizes elementwise execution. `run_unary_kernel<T>` and
`run_binary_kernel<T>` create contiguous outputs, validate dtype, select the
contiguous fast path, and fall back to TensorIterator offsets for broadcasting
or non-contiguous inputs.

MatMul is split into operator validation and CPU kernels. The CPU layer keeps a
Tensor-indexed reference kernel, a direct stride-aware kernel, and a
cache-blocked kernel, and a reusable packed-RHS SIMD micro-kernel. The public
`matmul` operator packs RHS and selects the SIMD path. Callers executing the
same weights repeatedly can construct `PackedMatMulRhs` once and invoke the
kernel directly to amortize packing.

`basic_ops.cpp` contains the remaining readable FP32 implementations:

- Add, including NumPy-style broadcasting
- Sub and Mul, including NumPy-style broadcasting
- Rank-2 MatMul through the CPU kernel layer
- ReLU
- GELU using the tanh approximation
- ReduceSum and ReduceMax over one or more axes
- Softmax over the final dimension
- LayerNorm over the final dimension, with optional affine weight and bias
- Linear, composed from MatMul and Add

Phase 1 will split operator contracts, shape inference, and backend-specific kernels as those responsibilities become necessary.
