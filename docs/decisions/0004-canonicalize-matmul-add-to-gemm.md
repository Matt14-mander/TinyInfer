# ADR-0004: Canonicalize eligible MatMul + Add to Gemm

- Status: Accepted (implemented)
- Date: 2026-09-25

## Context

TinyInfer currently has separate rank-2 Float32 `MatMul`, broadcasted `Add`,
and `Gemm` graph operators. The CPU `Gemm` kernel computes
`alpha * (A' @ B') + beta * C`, where transpose flags select `A'` and `B'`.
`C` is optional and must broadcast **to the rank-2 product**. The ONNX
specification gives `Gemm` this unidirectional-broadcasting contract, while
ONNX `Add` permits multidirectional broadcasting and ONNX `MatMul` also covers
more ranks than TinyInfer currently implements.

Sources: [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html),
[ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html), and
[ONNX Add](https://onnx.ai/onnx/operators/onnx__Add.html).

A future Linear/Bias/Activation fusion pass needs one predictable affine
pattern. But blindly merging every `MatMul + Add` would change the meaning of
some graphs or remove a value that is still observable.

## Decision

Use `Gemm` as the **operator-level canonical affine form** for the supported
rank-2 case. Rewrite either operand order of:

```text
t = MatMul(A, B)
y = Add(t, C)          // or Add(C, t)
```

to:

```text
y = Gemm(A, B, C; alpha=1, beta=1, transA=0, transB=0)
```

The pass must satisfy all of these conditions before rewriting:

1. `Add` has exactly one input produced by a `MatMul` node and uses that
   product once. `GraphAnalysis::use_count(t) == 1`, and `t` is not a
   registered graph output. A graph output remains observable even when it
   has no node consumers.
2. `A` and `B` are rank-2 Float32 tensors accepted by the current `MatMul`
   schema. The result `t` and `C` are Float32.
3. The existing `Add` output shape is exactly the `MatMul` product shape.
   In particular, a multidirectional broadcast that expands the result to a
   higher rank is **not** a legal `Gemm` bias.
4. `infer_output_specs(Gemm, {A, B, C}, canonical_attributes)` succeeds and
   its output TensorSpec equals the original `Add` output TensorSpec. This is
   the final contract check; do not infer eligibility merely from a familiar
   bias shape such as `[N]`.

`C` may be a constant or a runtime value, and any shape accepted by the
current `Gemm` schema is eligible (including scalar, `[N]`, `[1,N]`,
`[M,1]`, or `[M,N]` where broadcast rules permit). The rule is an affine
canonicalization, not a special-case bias-vector matcher.

Keep a standalone `MatMul` as `MatMul`, and keep an imported or existing
`Gemm` as `Gemm`. Do **not** expand `Gemm` into `MatMul + Add`, even when some
attributes have their defaults. For matching purposes, absent `Gemm`
attributes have the schema/kernel defaults `alpha=1`, `beta=1`, and
`transA=transB=0`; an existing node need not be rewritten just to spell out
those defaults. The newly fused node should emit explicit defaults for
deterministic inspection. More general `Gemm` attributes and transpositions
remain on the original node; they are not silently discarded or folded into
weights in this pass.

## Implementation boundary

`MatMulAddCanonicalizationPass` uses `GraphRewriter::replace_node` to replace
the source `Add` with `Gemm` and map the surviving output. It uses a validated
`GraphAnalysis` snapshot for consumers,
use-count, and graph-output checks. Rebuild the graph in topological order;
do not mutate the source graph in place. Map the old `Add` output to the new
`Gemm` output, map the removed `MatMul` output to `kInvalidValueId`, and
preserve external input/output names and ordered bindings. The pass must be
idempotent: a second run finds no `MatMul + Add` pair and reports no change.
Place it after the existing Constant Folding and Dead Code Elimination passes,
then run Dead Code Elimination again if the rewrite leaves unused constants.
When no pattern matches, the pass returns an independent model copy and an
identity ValueId mapping. When it does match, it preserves unrelated constants;
dead-value removal remains a separate DCE responsibility.

This decision does **not** claim a performance win today. The current CPU
`Gemm` computes the matrix product and then applies scaling/bias in a separate
loop; it is not yet a fused micro-kernel. Graph-level canonicalization may
reduce an intermediate value and prepare later kernel fusion, but benchmark
and memory-plan measurements must establish any actual gain.

## Verification

- Positive: both `Add` input orders; constant and dynamic `C`; scalar,
  vector, row, column, and full-matrix broadcast cases that preserve `[M,N]`.
- Negative: `MatMul` result with multiple uses; `MatMul` result registered as
  graph output; `Add` that expands shape; any candidate failing Gemm schema;
  standalone `MatMul`; existing `Gemm` with transpose/scaling attributes.
- Verify original and rewritten model outputs numerically on representative
  Float32 inputs using a documented tolerance, plus shape/dtype equality,
  external bindings, ValueId mapping, memory-plan behavior, and a no-change
  second pass. Do not promise bitwise identity from a graph rewrite.

`matmul_add_canonicalization_test.cpp` covers these cases using `1e-5`
absolute tolerance for finite FP32 outputs, including both Add orders,
different C broadcast shapes, a downstream consumer of the Add result,
PassManager composition, planned execution, unchanged source graphs, and
preservation of unrelated constants.

## Alternatives considered

- **Canonicalize to `MatMul + Add`:** exposes primitive operations but loses
  `Gemm`'s alpha/beta/transposition semantics or needs more graph operators,
  creates an additional intermediate, and works against a future affine
  kernel. Rejected for this phase.
- **Fuse every `MatMul + Add`:** incorrect when Add expands shape or the
  product is observable elsewhere. Rejected.
- **Introduce `Linear` immediately:** duplicates `Gemm`'s current rank-2
  affine role before a distinct Linear contract or backend benefit exists.
  Deferred.

## Consequences

The optimizer has one direction of normalization and one conservative
eligibility test. Later activation fusion can match `Gemm -> Activation`
without also matching `MatMul -> Add -> Activation`. The ONNX importer keeps
its current direct operator translations; canonicalization is an optional
post-import optimization, not an import-time semantic change.
