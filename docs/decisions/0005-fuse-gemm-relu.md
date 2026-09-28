# ADR-0005: Fuse eligible Gemm + ReLU

- Status: Accepted (implemented)
- Date: 2026-09-28

## Context

ADR-0004 established `Gemm` as TinyInfer's canonical rank-2 affine operator.
An immediately following ReLU otherwise requires a second graph node, an
intermediate Tensor, and a separate output traversal. Phase 4.3 needs a small,
measurable fusion slice without introducing a general compiler IR or changing
ONNX import semantics.

## Decision

Rewrite:

```text
t = Gemm(A, B [, C]; alpha, beta, transA, transB)
y = ReLU(t)
```

to the internal operator:

```text
y = FusedGemmActivation(A, B [, C];
                        alpha, beta, transA, transB,
                        activation="relu")
```

only when the Gemm result has one use, its sole consumer is that ReLU, and it
is not a graph output. Schema inference for the proposed fused node must
produce the same output TensorSpec as the original ReLU. Existing Gemm
attributes and inputs are preserved. The removed Gemm output maps to
`kInvalidValueId`; the ReLU output maps to the fused output.

`FusedGemmActivation` is an internal operator, not an ONNX import target. Its
schema deliberately requires an `activation` string and currently accepts
only `relu`, leaving room for explicit, reviewed extensions rather than
silently changing semantics.

The CPU implementation shares one Gemm implementation between fused and
unfused execution. MatMul still writes the output first, but alpha scaling,
optional bias, and ReLU are applied in one epilogue traversal. Therefore this
slice removes a graph intermediate and one full ReLU traversal; it does not
claim in-register fusion inside the SIMD MatMul micro-kernel.

The recommended opt-in pipeline is:

```text
ConstantFoldingPass -> DeadCodeEliminationPass
    -> MatMulAddCanonicalizationPass
    -> GemmActivationFusionPass
    -> DeadCodeEliminationPass
```

## Alternatives considered

- **Fuse during ONNX import:** rejected because import should preserve the
  source graph contract and optimization remains explicitly opt-in.
- **Mutate Gemm into a generic activation-bearing node:** rejected because it
  weakens operator semantics and makes backend support ambiguous.
- **Implement a fully fused SIMD micro-kernel first:** deferred. It couples
  graph matching, packing, ISA dispatch, and numerical validation into a much
  larger change than this vertical slice.
- **Fuse when the Gemm result has other users or is an output:** rejected
  because the pre-activation value remains observable.

## Consequences

Eligible graphs contain one fewer node and intermediate Tensor, and planned
execution needs less logical intermediate storage. The CPU backend has a
clear fused-kernel dispatch point that later ISA-specific epilogues can
replace. Tests must cover transpose/scaling/bias combinations, planned and
unplanned execution, floating-point edge cases, negative matches, mapping,
idempotence, and composition with MatMul + Add canonicalization. Performance
claims must come from the dedicated paired benchmark.
