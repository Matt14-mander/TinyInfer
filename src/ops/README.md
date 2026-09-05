# Operators

`basic_ops.cpp` contains the readable FP32 reference implementations used by Phase 0:

- Add, including one-dimensional bias broadcasting
- Rank-2 MatMul
- ReLU
- Softmax over the final dimension
- Linear, composed from MatMul and Add

Phase 1 will split operator contracts, shape inference, and backend-specific kernels as those responsibilities become necessary.
