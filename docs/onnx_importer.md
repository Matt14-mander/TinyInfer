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
MatMul, Relu, Gelu, Softmax, and LayerNormalization. It imports initializers as
Graph constants, excludes initializer-backed values from runtime inputs,
resolves nodes by their data dependencies, validates inferred output metadata,
and preserves external model input/output names through `Model` bindings.
TensorProto decoding supports static Float32, Float16, Int8, and Int32 tensors
from `raw_data`, plus ONNX `float_data` and `int32_data` typed storage.

External tensor files, non-default ONNX domains, optional inputs, multiple
outputs, dynamic dimensions, sparse tensors, and broader dtype/operator
coverage are deliberately deferred to later Phase 3 slices.
