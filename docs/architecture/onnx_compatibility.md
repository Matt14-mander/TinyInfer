# ONNX compatibility profile

This document records two different contracts: what TinyInfer can execute as
its **own Graph operators**, and which **ONNX subsets it can import with
explicitly tested semantics**. They are not interchangeable. An `OpType`,
CPU kernel, or ONNX registry entry does not by itself establish ONNX semantic
compatibility. No operator is claimed to pass the complete ONNX conformance
suite; the supported cases and the evidence for them are listed separately
below.

## Model envelope

| Feature | v0.3 behavior |
| --- | --- |
| File format | Binary protobuf `.onnx` read by the built-in parser |
| Opset | Default-domain opset 13 or newer, with per-operator minimum versions; end-to-end fixture uses opset 17 |
| Operator domain | Empty domain or `ai.onnx` |
| Shapes | Static dimensions only |
| Node outputs | Exactly one non-empty output per node |
| Optional inputs | Empty input names are rejected |
| Initializer storage | Embedded `raw_data`, `float_data`, or `int32_data` |
| External data | Rejected with a `TensorDecode` diagnostic |
| Sparse tensors | Not supported |
| Graph ordering | Nodes may be out of topological order |
| Cycles and missing values | Rejected during `GraphImport` |

The importer passes the default-domain opset to a version-aware operator
registry. Registrations use inclusive version ranges and reject overlaps. A
registry hit means only that a translator is available for that opset; shape
checks, dtype checks, import restrictions, execution, and ONNX semantics are
separate gates. In particular, the open-ended registry ranges do **not** mean
that every future ONNX operator revision has been reviewed.

## Data type boundaries

| Dtype | ONNX TensorProto decode | Internal Graph/CPU operators | Imported ONNX graph execution |
| --- | --- | --- | --- |
| Float32 | Supported | Supported within the operator shapes below | Supported within the ONNX subset below |
| Float16 | Supported as raw binary16 storage | Not supported by current Graph operator schemas | Not supported |
| Int8 | Supported | Not supported by current Graph operator schemas | Not supported |
| Int32 | Supported | Not supported by current Graph operator schemas | Not supported |

Tensor storage and TensorProto decoding are not evidence of operator execution.
Every current Graph operator schema requires Float32 inputs, even when an
initializer of another dtype can be decoded.

## Internal operator support (no ONNX claim)

This table describes `OpType` + Schema/shape inference + CPU Backend, as
tested with TinyInfer Graphs. All entries are Float32 and have one output.

| Internal `OpType` | Implemented Graph/CPU subset | Internal test gate |
| --- | --- | --- |
| `Add`, `Subtract`, `Multiply` | Two inputs; multidirectional elementwise broadcasting | `operator_coverage_test.cpp` |
| `MatMul` | Two rank-2 matrices | `operator_coverage_test.cpp`, `matmul_kernel_test.cpp` |
| `Gemm` | Rank-2 A/B; optional broadcastable C; `alpha`, `beta`, `transA`, `transB` (transpose flags 0 or 1) | `operator_coverage_test.cpp`, `gemm_test.cpp` |
| `ReLU`, `Tanh` | Unary, output shape equals input shape | `operator_coverage_test.cpp` |
| `GELU` | Unary; internal default is the `tanh` approximation, with `none` also available | `operator_coverage_test.cpp`, `elementwise_ops_test.cpp` |
| `Softmax` | Unary; chosen non-empty axis; default axis is -1 | `operator_coverage_test.cpp`, `reduction_normalization_test.cpp` |
| `LayerNorm` | Unary input with optional final-dimension weight and bias; `epsilon` | `operator_coverage_test.cpp`, `reduction_normalization_test.cpp` |

The `LayerNorm` one-input and `GELU` default behaviors in this table are
**internal APIs**, not promises about ONNX `LayerNormalization` or `Gelu`.

## ONNX import subset and semantic evidence

All rows below are conditional on the model envelope above: a static-shape,
default-domain model with one non-empty output per node and Float32 execution.
"Registered from" reports the translator's lower opset bound, not full
support for every model or later opset. "Evidence" distinguishes synthetic
TinyInfer-built ONNX models from a file actually exported by PyTorch.

| ONNX operator | Registered from | Accepted subset / semantic boundary | Evidence available; not a full conformance claim |
| --- | ---: | --- | --- |
| `Add`, `Sub`, `Mul` | 13 | Two Float32 inputs; elementwise broadcasting; no attributes | Synthesized one-node import/execution in `operator_coverage_test.cpp`; a static Add model in `onnx_compatibility_test.cpp`. No broad ONNX backend suite. |
| `MatMul` | 13 | Float32 rank-2 matrices only. [ONNX also defines N-dimensional MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html), which is outside this subset. | Synthesized one-node import/execution; no batched/vector ONNX cases. |
| `Gemm` | 13 | Float32 rank-2 A/B; two inputs or a non-empty optional C; broadcastable C; `alpha`, `beta`, `transA`, `transB` with transpose flags 0 or 1. Explicit empty-string optional inputs are not accepted. | Synthesized one-node case, checked Gemm MLP fixture, and PyTorch-exported actor MLP. These exercise selected values/attributes, not every [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html) combination. |
| `Relu` | 13 | Unary Float32; no attributes | Synthesized one-node case and checked Gemm MLP fixture. |
| `Tanh` | 13 | Unary Float32; no attributes | Synthesized one-node case and PyTorch-exported actor MLP compared with two PyTorch outputs. |
| `Gelu` | 20 | Unary Float32; absent `approximate` becomes exact `none` (erf); explicit `none` or `tanh` accepted. This differs from internal GELU's default. | Synthesized one-node case plus default/explicit/invalid-attribute tests; no exported-model or broad differential suite. |
| `Softmax` | 13 | Unary Float32; valid, non-empty axis; default -1 | Synthesized one-node case and checked Gemm MLP fixture; no broad rank/axis differential suite. |
| `LayerNormalization` | 17 | Float32 X and 1-D Scale, optional 1-D Bias; final axis only (`axis=-1` or omitted), `stash_type=1` or omitted, one Y output. Other [ONNX axes, broadcast forms, and optional Mean/InvStdDev outputs](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) are outside this subset. | Synthesized one-node case plus accepted/rejected attribute and input tests; no exported-model or broad differential suite. |

Unknown attributes are rejected during shape inference; the ONNX-specific
`Gelu` and `LayerNormalization` translators reject unsupported semantics
earlier. An accepted model has passed the implemented restrictions, **not**
an ONNX reference-backend or full conformance comparison. That distinction
matters especially for numerical tolerances, unusual shapes, and future
opset revisions.

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
| `OperatorTranslation` | Domain, OpType, or opset has no TinyInfer translation |
| `ShapeInference` | Operator Schema rejected inputs or attributes |
| `GraphValidation` | Imported outputs or final Graph invariants failed |

Example:

```text
ONNX import failed [stage=OperatorTranslation, graph=compatibility_graph,
node_index=0, node=add_bias, op=Conv]: unsupported ONNX operator or opset:
domain='', op='Conv', opset=17
```

## Test matrix: what each test actually establishes

The gates are deliberately separate. `operator_coverage_test.cpp` checks each
internal `OpType` in enum order: Schema, shape inference, CPU registration,
and a forward result. For each declared ONNX translator it additionally
checks the lower opset boundary, translation, and execution of a *synthetic*
one-node imported model. That is cross-layer plumbing, not a full ONNX
semantic test. Its checks remain active in Release builds.

| Layer / evidence level | What is verified | What is **not** established |
| --- | --- | --- |
| Internal operator coverage | TinyInfer Graph can infer and execute one positive Float32 case per `OpType` | ONNX import or ONNX semantics |
| Translator/opset coverage | Registration boundary and synthetic one-node import/execution | All legal ONNX shapes, dtypes, attributes, and future opsets |
| Targeted negative tests | Unsupported inputs/attributes fail with expected diagnostics | Rejection of every invalid ONNX model |
| File fixtures | Built-in protobuf parser plus end-to-end inference; PyTorch fixture compares two outputs | Full operator conformance or numerical parity over diverse models |

| Case | Expected result | Test |
| --- | --- | --- |
| Hand-built opset-17 Gemm MLP | Import and execute with checked probabilities | `onnx_gemm_fixture_test.cpp` |
| PyTorch-exported opset-17 RL actor MLP (`Gemm → Tanh → Gemm → Tanh`) | Import and compare actions for two observations with PyTorch references | `onnx_rl_mlp_fixture_test.cpp` |
| Supported static Add model | Import successfully | `onnx_compatibility_test.cpp` |
| Opset below 13 | `ModelValidation` | `onnx_compatibility_test.cpp` |
| LayerNormalization at opset 16 / 17 | Reject 16; execute 17 with Scale and optional Bias | `onnx_compatibility_test.cpp` |
| LayerNormalization unsupported axis/stash/shape/missing Scale | Reject with diagnostic | `onnx_compatibility_test.cpp` |
| Gelu at opset 19 / 20 | Reject 19; execute 20 exact default and explicit tanh | `onnx_compatibility_test.cpp` |
| Gelu unsupported approximation | Reject with diagnostic | `onnx_compatibility_test.cpp` |
| Versioned registry and overlapping ranges | Select matching translator; reject overlap | `onnx_operator_registry_test.cpp` |
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

This is the current *tested ONNX subset*, not a general ONNX compatibility
claim. A new operator first needs internal Schema/CPU coverage; ONNX support
then requires a versioned translator, an explicitly bounded semantic subset,
positive and negative import tests, and—before any broader compatibility
claim—reference-backend or exporter differential tests for the claimed cases.
