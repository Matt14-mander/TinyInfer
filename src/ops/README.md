# Operators

`kernel_runner.h` centralizes elementwise execution. `run_unary_kernel<T>` and
`run_binary_kernel<T>` create contiguous outputs, validate dtype, select the
contiguous fast path, and fall back to TensorIterator offsets for broadcasting
or non-contiguous inputs.

`operator_schema.cpp` defines graph-time contracts for the current operators:
input arity, output count, accepted attributes, FP32 requirements, and output
TensorSpec inference. Graph construction calls this layer before output Values
are created.

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
- GELU using the tanh approximation by default; exact-erf mode for ONNX `Gelu`
- ReduceSum and ReduceMax over one or more axes
- Softmax over the final dimension
- LayerNorm over the final dimension, with optional affine weight and bias
- Linear, composed from MatMul and Add

Phase 2 now uses these contracts to validate nodes and infer output metadata
during graph construction. Numerical dispatch remains a later Phase 2 step.
