import torch
from torch import nn

from linodenet.nn.rezero import ReZero, resolve_gate
from tests.testing import TestSuite


class ShiftScalar(nn.Module):
    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        return x + 1.0


class TestReZero(TestSuite):
    def test_resolve_gate_uses_identity_scalar_map(self) -> None:
        gate = resolve_gate("rezero")

        assert isinstance(gate, ReZero)
        assert isinstance(gate.scalar_map, nn.Identity)

    def test_default_module_is_identity(self) -> None:
        module = ReZero()

        assert isinstance(module.module, nn.Identity)

    def test_default_scalar_map_is_identity(self) -> None:
        module = ReZero(nn.Identity())

        assert isinstance(module.scalar_map, nn.Identity)

    def test_scalar_map_is_applied_to_scalar(self) -> None:
        module = ReZero(nn.Identity(), scalar_map=ShiftScalar())
        with torch.no_grad():
            module.scalar.copy_(torch.tensor(2.0))
        x = torch.tensor([1.0, -2.0, 3.0])

        y = module(x)

        self.assert_close(y, 3.0 * x, atol=1e-6, rtol=1e-6)
