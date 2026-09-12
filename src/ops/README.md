# Operators

`kernel_runner.h` centralizes elementwise execution. `run_unary_kernel<T>` and
`run_binary_kernel<T>` create contiguous outputs, validate dtype, select the
contiguous fast path, and fall back to TensorIterator offsets for broadcasting
or non-contiguous inputs.

`basic_ops.cpp` contains the readable FP32 reference implementations used by Phase 0:

- Add, including NumPy-style broadcasting
- Rank-2 MatMul
- ReLU
- Softmax over the final dimension
- Linear, composed from MatMul and Add

Phase 1 will split operator contracts, shape inference, and backend-specific kernels as those responsibilities become necessary.
