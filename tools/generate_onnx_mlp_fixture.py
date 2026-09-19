#!/usr/bin/env python3
"""Generate the fixed Phase 3 Gemm MLP fixture.

Prefer the installed `onnx` package so the model is checked by ONNX. For
dependency-light development, TINYINFER_ONNX_PROTO_PYTHON may point at Python
bindings generated from the official onnx.proto with protoc.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
import struct
import sys


def load_onnx_api():
    try:
        import onnx  # type: ignore

        return onnx, onnx.checker.check_model
    except ImportError:
        generated = os.environ.get("TINYINFER_ONNX_PROTO_PYTHON")
        if not generated:
            raise RuntimeError(
                "install onnx or set TINYINFER_ONNX_PROTO_PYTHON to generated bindings"
            )
        sys.path.insert(0, generated)
        module = importlib.import_module("tinyinfer_onnx_pb2")
        return module, lambda model: None


def add_value_info(proto, graph, name: str, shape: list[int], output: bool = False):
    value = graph.output.add() if output else graph.input.add()
    value.name = name
    value.type.tensor_type.elem_type = proto.TensorProto.FLOAT
    for size in shape:
        value.type.tensor_type.shape.dim.add().dim_value = size


def add_initializer(proto, graph, name: str, shape: list[int], values: list[float]):
    tensor = graph.initializer.add()
    tensor.name = name
    tensor.data_type = proto.TensorProto.FLOAT
    tensor.dims.extend(shape)
    tensor.raw_data = struct.pack(f"<{len(values)}f", *values)


def add_attribute(proto, node, name: str, value: int):
    attribute = node.attribute.add()
    attribute.name = name
    attribute.type = proto.AttributeProto.INT
    attribute.i = value


def build_model(proto):
    model = proto.ModelProto()
    model.ir_version = 8
    model.producer_name = "TinyInfer fixture generator"
    opset = model.opset_import.add()
    opset.domain = ""
    opset.version = 17
    graph = model.graph
    graph.name = "phase3_gemm_mlp"

    add_value_info(proto, graph, "x", [1, 2])
    add_value_info(proto, graph, "probabilities", [1, 2], output=True)
    add_initializer(proto, graph, "w1", [3, 2], [1, 1, 0, -1, -1, 1])
    add_initializer(proto, graph, "b1", [3], [0, 0, 1])
    add_initializer(proto, graph, "w2", [2, 3], [1, 1, 0, 0, 2, 1])
    add_initializer(proto, graph, "b2", [2], [0, -1])

    gemm1 = graph.node.add()
    gemm1.name = "linear1"
    gemm1.op_type = "Gemm"
    gemm1.input.extend(["x", "w1", "b1"])
    gemm1.output.append("hidden_pre")
    add_attribute(proto, gemm1, "transB", 1)

    relu = graph.node.add()
    relu.name = "relu"
    relu.op_type = "Relu"
    relu.input.append("hidden_pre")
    relu.output.append("hidden")

    gemm2 = graph.node.add()
    gemm2.name = "linear2"
    gemm2.op_type = "Gemm"
    gemm2.input.extend(["hidden", "w2", "b2"])
    gemm2.output.append("logits")
    add_attribute(proto, gemm2, "transB", 1)

    softmax = graph.node.add()
    softmax.name = "softmax"
    softmax.op_type = "Softmax"
    softmax.input.append("logits")
    softmax.output.append("probabilities")
    add_attribute(proto, softmax, "axis", -1)
    return model


def main():
    proto, check_model = load_onnx_api()
    model = build_model(proto)
    check_model(model)
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/phase3_mlp_gemm.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(model.SerializeToString())

    denominator = math.exp(2.0) + math.exp(3.0)
    print(f"wrote {output}")
    print(f"reference probabilities: {math.exp(2.0) / denominator:.8f}, "
          f"{math.exp(3.0) / denominator:.8f}")


if __name__ == "__main__":
    main()
