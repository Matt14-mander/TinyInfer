# ONNX importer architecture

Phase 3 separates ONNX file parsing from TinyInfer graph construction:

```text
.onnx bytes
    |
    v
ProtobufModelParser         built-in minimal protobuf wire reader
    |
    v
onnx::ModelProto            lightweight, dependency-free description
    |
    +--> ONNX OperatorRegistry --> OpType + normalized attributes
    |
    v
OnnxImporter                names, initializers, dependencies, outputs
    |
    v
Model                       immutable Graph + named input/output bindings
    |
    v
ExecutionContext + Executor + Backend
```

`ModelLoader` is the format-independent file-loading interface. `OnnxImporter`
uses the built-in `ProtobufModelParser` by default while retaining the injected
`onnx::ModelParser` boundary for tests or alternative parsers. The wire reader
parses only the required ONNX protobuf fields, so TinyInfer does not link the
protobuf runtime or the full ONNX source tree.

The first importer slice supports one-output nodes and maps Add, Sub, Mul,
MatMul, Gemm, Relu, Gelu, Softmax, and LayerNormalization. It imports
initializers as Graph constants, excludes initializer-backed values from runtime inputs,
resolves nodes by their data dependencies, validates inferred output metadata,
and preserves external model input/output names through `Model` bindings.
TensorProto decoding supports static Float32, Float16, Int8, and Int32 tensors
from `raw_data`, plus ONNX `float_data` and `int32_data` typed storage.

External tensor files, non-default ONNX domains, optional inputs, multiple
outputs, dynamic dimensions, sparse tensors, and broader dtype/operator
coverage are deliberately deferred to later Phase 3 slices.

## Gemm analysis

ONNX Gemm computes `Y = alpha * A' * B' + beta * C`, where `transA` and
`transB` select the transposed matrix views. The checked MLP fixture uses the
common exporter layout: weights are stored as `[out_features, in_features]`,
`transB=1`, and bias is broadcast across the batch dimension.

TinyInfer represents Gemm as a Graph operator with two or three inputs. Shape
inference validates matrix ranks, transpose flags, inner dimensions, dtypes,
and bias broadcasting. The CPU kernel applies optional transposes, MatMul,
alpha/beta scaling, and bias addition using existing Tensor kernels. This is a
correctness-first implementation; avoiding transpose copies and fusing bias
remain later optimizations.

`tests/fixtures/phase3_mlp_gemm.onnx` is generated from the official ONNX
protobuf definition by `tools/generate_onnx_mlp_fixture.py`. It contains two
Gemm nodes, Relu, and Softmax with fixed weights and a checked reference output.
