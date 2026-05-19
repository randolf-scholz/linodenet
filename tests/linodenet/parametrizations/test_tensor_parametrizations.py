r"""Tests for tensor parametrizations."""

import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim import SGD

from linodenet.parametrizations import (
    ParametrizationList,
    ReZero,
    get_parametrizations,
    register_optimizer_hook,
    register_parametrization,
)
from tests.testing import DEVICES, TestSuite


class ResidualModel(nn.Module):
    def __init__(self, in_feat: int, out_feat: int) -> None:
        super().__init__()
        self.residual = nn.Linear(in_feat, in_feat, bias=False)
        self.output = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        hidden = x + self.residual(torch.relu(x))
        return self.output(hidden)


class TestReZero(TestSuite):
    NUM_ITERATIONS = 3
    VECTOR_SIZE = 4
    TARGET_SIZE = 2
    BATCH_SIZE = 8

    @pytest.mark.parametrize("device", DEVICES)
    def test_trainable(self, device: str) -> None:
        torch.manual_seed(0)
        model = ResidualModel(self.VECTOR_SIZE, self.TARGET_SIZE).to(device=device)
        x = torch.randn(self.BATCH_SIZE, self.VECTOR_SIZE, device=device)
        y = torch.randn(self.BATCH_SIZE, self.TARGET_SIZE, device=device)

        register_parametrization(model.residual, "weight", ReZero)
        optimizer = SGD(model.parameters(), lr=0.1)
        register_optimizer_hook(optimizer, model)

        parametrizations = get_parametrizations(model.residual)
        assert parametrizations is not None
        parametrization = parametrizations["weight"]
        assert isinstance(parametrization, ParametrizationList)
        assert len(parametrization) == 1
        rezero = parametrization[0]
        assert isinstance(rezero, ReZero)
        self.assert_close(
            rezero.scalar.detach(),
            torch.tensor(0.0, device=device),
            atol=1e-6,
            rtol=1e-6,
        )

        original_scalar = rezero.scalar.detach().clone()
        original_output = model(x).detach().clone()
        original_loss = mse_loss(original_output, y)

        for _ in range(self.NUM_ITERATIONS):
            model.zero_grad(set_to_none=True)
            loss = mse_loss(model(x), y)
            loss.backward()
            optimizer.step()

        assert not torch.allclose(rezero.scalar, original_scalar)
        assert not torch.allclose(model(x), original_output)
        assert mse_loss(model(x), y) < original_loss
