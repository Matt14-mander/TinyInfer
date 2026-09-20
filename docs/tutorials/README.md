# Tutorials

Tutorials explain TinyInfer as a sequence of small, reproducible engineering
steps. They should teach the underlying runtime concept rather than only list
public APIs.

## Tutorial structure

Each tutorial should contain:

1. The problem and the runtime concept it exposes.
2. A minimal correctness-first implementation.
3. Tests or a known numerical example.
4. Limitations of the minimal implementation.
5. The evolution toward TinyInfer's current design.
6. A benchmark when the topic concerns performance.
7. Links to the relevant architecture document and source code.

## Planned series

1. Tensor memory, shape, and strides.
2. Storage, Buffer, views, and ownership.
3. TensorIterator and reusable elementwise kernels.
4. Reduction, Softmax, and LayerNorm.
5. MatMul from reference loops to packing and SIMD.
6. From eager operators to a computation graph.
7. Operator Schema, shape inference, and graph validation.
8. ExecutionContext, Backend, and operator dispatch.
9. From ONNX protobuf bytes to an executable TinyInfer graph.
10. Gemm and a real ONNX MLP end to end.

Tutorial implementation begins after the corresponding subsystem is stable
enough to explain without immediately invalidating the walkthrough.
