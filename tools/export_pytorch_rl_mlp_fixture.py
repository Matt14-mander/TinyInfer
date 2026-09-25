#!/usr/bin/env python3
"""Export a deterministic PyTorch actor MLP to an ONNX test fixture.

Requires torch==2.2.2, onnx==1.16.0, and numpy<2. The model is intentionally
small, but uses the ordinary Linear/Tanh stack of a continuous-control actor.
No ONNX graph or tensor payload is constructed by hand.
"""

from pathlib import Path

import onnx
import torch


class Actor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
            torch.nn.Tanh(),
        )
        with torch.no_grad():
            self.net[0].weight.copy_(torch.tensor([
                [0.5, -0.25, 0.75, 0.0],
                [-0.5, 0.5, 0.0, 0.25],
                [0.25, 0.25, -0.5, 0.5],
                [1.0, 0.0, 0.25, -0.25],
                [-0.25, 0.75, 0.5, 0.25],
            ]))
            self.net[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0, -0.1]))
            self.net[2].weight.copy_(torch.tensor([
                [0.5, -0.25, 0.75, 0.0, 0.25],
                [-0.5, 0.5, 0.25, 0.75, -0.25],
            ]))
            self.net[2].bias.copy_(torch.tensor([0.05, -0.15]))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def main() -> None:
    if torch.__version__.split("+")[0] != "2.2.2" or onnx.__version__ != "1.16.0":
        raise RuntimeError("fixture export requires torch==2.2.2 and onnx==1.16.0")
    model = Actor().eval()
    output = Path(__file__).resolve().parents[1] / "tests/fixtures/rl_actor_mlp_tanh.onnx"
    observations = torch.tensor([[0.25, -0.5, 1.0, 0.75]], dtype=torch.float32)
    torch.onnx.export(
        model, observations, str(output), export_params=True,
        opset_version=17, do_constant_folding=True,
        input_names=["observations"], output_names=["actions"],
    )
    exported = onnx.load(str(output))
    onnx.checker.check_model(exported)
    if exported.producer_name != "pytorch":
        raise RuntimeError("fixture was not produced by PyTorch")
    if [(item.domain, item.version) for item in exported.opset_import] != [("", 17)]:
        raise RuntimeError("unexpected ONNX opset imports")
    kinds = [node.op_type for node in exported.graph.node]
    if kinds != ["Gemm", "Tanh", "Gemm", "Tanh"]:
        raise RuntimeError(f"unexpected exported operators: {kinds}")
    print(f"wrote {output}")
    print(f"torch {torch.__version__}; onnx {onnx.__version__}; opset 17; nodes {kinds}")
    for sample in (
        observations,
        torch.tensor([[-1.0, 0.0, 0.5, -0.25]], dtype=torch.float32),
    ):
        with torch.no_grad():
            print(f"input {sample.tolist()} -> actions {model(sample).tolist()}")


if __name__ == "__main__":
    main()
