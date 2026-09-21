# ONNX compatibility profile

This document defines the ONNX subset supported by TinyInfer's v0.3 Model
Runtime. Compatibility is split into parsing, graph import, and CPU execution;
support at one layer does not imply support at the next layer.

## Model envelope

| Feature | v0.3 behavior |
| --- | --- |
| File format | Binary protobuf `.onnx` read by the built-in parser |
| Opset | Default-domain opset 13 or newer; end-to-end fixture uses opset 17 |
| Operator domain | Empty domain or `ai.onnx` |
| Shapes | Static dimensions only |
| Node outputs | Exactly one non-empty output per node |
| Optional inputs | Empty input names are rejected |
| Initializer storage | Embedded `raw_data`, `float_data`, or `int32_data` |
| External data | Rejected with a `TensorDecode` diagnostic |
| Sparse tensors | Not supported |
| Graph ordering | Nodes may be out of topological order |
| Cycles and missing values | Rejected during `GraphImport` |

Opset 13 or newer is an importer gate, not a claim of complete conformance for
every later ONNX opset. Each operator below supports only the attributes and
semantics explicitly implemented by TinyInfer.

## Dtype layers

| Dtype | TensorProto decode | Graph operator execution |
| --- | --- | --- |
| Float32 | Supported | Supported |
| Float16 | Supported as raw binary16 storage | Not yet supported by imported operators |
| Int8 | Supported | Not yet supported by imported operators |
| Int32 | Supported | Not yet supported by imported operators |

Operator Schema currently requires Float32 inputs. Parser support for a dtype
therefore means an initializer can be decoded, not that a CPU kernel can use it
in an imported graph.

## Operator profile

| ONNX operator | Inputs | Supported attributes | Current constraints |
| --- | ---: | --- | --- |
| `Add` | 2 | None | Float32, NumPy-style broadcasting |
| `Sub` | 2 | None | Float32, NumPy-style broadcasting |
| `Mul` | 2 | None | Float32, NumPy-style broadcasting |
| `MatMul` | 2 | None | Float32 rank-2 matrices only |
| `Gemm` | 2–3 | `alpha`, `beta`, `transA`, `transB` | Float32 rank-2 A/B; optional broadcastable C |
| `Relu` | 1 | None | Float32 |
| `Gelu` | 1 | None | Float32 tanh approximation |
| `Softmax` | 1 | `axis` | Float32, non-empty selected axis |
| `LayerNormalization` | 1 or 3 | `epsilon` | Float32; final-axis normalization; affine form requires weight and bias |

Unknown attributes are rejected during Shape Inference. This is intentional:
silently ignoring an ONNX attribute could execute a model with different
semantics.

## Structured diagnostics

All public ONNX loading and importing failures use `OnnxImportError`. Its
`OnnxImportDiagnostic` records the stage and available context:

```text
stage
model_path
graph_name
node_index
node_name
op_type
domain
value_name
message
```

The stages identify the boundary that rejected the model:

| Stage | Meaning |
| --- | --- |
| `FileRead` | Model file could not be opened or read |
| `ProtobufParse` | Wire data or ONNX model metadata could not be parsed |
| `TensorDecode` | TensorProto storage or dtype could not be decoded |
| `ModelValidation` | Model-level constraints such as opset were rejected |
| `GraphImport` | Names, dependencies, input/output arity, or graph construction failed |
| `OperatorTranslation` | Domain or OpType has no TinyInfer translation |
| `ShapeInference` | Operator Schema rejected inputs or attributes |
| `GraphValidation` | Imported outputs or final Graph invariants failed |

Example:

```text
ONNX import failed [stage=OperatorTranslation, graph=compatibility_graph,
node_index=0, node=add_bias, op=Conv]: unsupported ONNX operator: Conv
```

## Compatibility test matrix

| Case | Expected result | Test |
| --- | --- | --- |
| Real opset-17 Gemm MLP | Import and execute with checked probabilities | `onnx_gemm_fixture_test.cpp` |
| Supported static Add model | Import successfully | `onnx_compatibility_test.cpp` |
| Opset below 13 | `ModelValidation` | `onnx_compatibility_test.cpp` |
| Unsupported operator | `OperatorTranslation` with node and OpType | `onnx_compatibility_test.cpp` |
| Non-default domain | `OperatorTranslation` with domain | `onnx_compatibility_test.cpp` |
| Unknown operator attribute | `ShapeInference` with node context | `onnx_compatibility_test.cpp` |
| Missing or empty input | `GraphImport` with value and node context | `onnx_compatibility_test.cpp` |
| Multiple node outputs | `GraphImport` | `onnx_compatibility_test.cpp` |
| Incorrect declared graph output | `GraphValidation` | `onnx_compatibility_test.cpp` |
| Duplicate initializer | `GraphImport` with initializer name | `onnx_compatibility_test.cpp` |
| Missing file | `FileRead` with path | `protobuf_model_parser_test.cpp` |
| Malformed protobuf | `ProtobufParse` with path | `protobuf_model_parser_test.cpp` |
| Dynamic dimension | `ProtobufParse` with explicit reason | `protobuf_model_parser_test.cpp` |
| External TensorProto data | `TensorDecode` with tensor name | `protobuf_model_parser_test.cpp` |
| Inconsistent TensorProto payload | `TensorDecode` with tensor name | `protobuf_model_parser_test.cpp` |

This matrix is the v0.3 compatibility contract. A new ONNX feature is complete
only when this profile and its positive and negative tests are updated.
