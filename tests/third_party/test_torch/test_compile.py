r"""Test several compile/forward features."""

import torch
from torch import Tensor, nn


class Trace(nn.Module):
    def get_powers(self, A: Tensor, k: int, /):
        r"""Yield $tr(A), tr(A²), …, tr(Aᵏ)$."""
        power = A
        for _ in range(k):
            yield torch.trace(power)
            power = power @ A


class SumTraces(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trace = Trace()

    def forward(self, A: Tensor, k: int) -> Tensor:
        total = A.new_zeros(())
        for value in self.trace.get_powers(A, k):
            total = total + value
        return total


class TestGenerator:
    def test_compile_fullgraph(self) -> None:
        model = SumTraces()
        compiled = torch.compile(model, fullgraph=True)

        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        k = 4

        expected = model(A, k)
        actual = compiled(A, k)

        assert torch.allclose(actual, expected)

    def test_export(self) -> None:
        model = SumTraces()

        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        k = 4

        exported = torch.export.export(model, args=(A, k)).module()
        expected = model(A, k)
        actual = exported(A, k)

        assert torch.allclose(actual, expected)
