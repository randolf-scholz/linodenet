import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from linodenet.mappings import (
    ContractiveFP,
    ContractiveNew,
    ContractiveTransform,
    TransformBase,
)
from linodenet.mappings.linear import LinearContraction
from tests.testing import DEVICES, SEEDS_5, TestCase


class ShiftedHalfContraction(nn.Module):
    r"""Simple contraction $g(x) = ½x + b$ with trainable bias."""

    def __init__(self, bias: Tensor, /) -> None:
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return 0.5 * x + self.bias


class TestContractiveFlow(TestCase):
    VALUE_ATOL = 1e-3
    VALUE_RTOL = 1e-3
    BATCH_SIZE = 32
    PERF_SEED = 0
    PERF_INPUT_SIZE = 256
    PERF_ROUNDS = 10
    PERF_WARMUP_ROUNDS = 1

    @pytest.mark.parametrize(
        "flow_cls",
        [ContractiveTransform, ContractiveFP, ContractiveNew],
        ids=["loop", "fixpoint_solve", "while_loop"],
    )
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    @pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64], ids="input_size={}".format)
    def test_invertibility(
        self,
        flow_cls: type[TransformBase],
        device: str,
        seed: int,
        input_size: int,
    ) -> None:
        r"""Check forward/inverse round trips; does not test logabsdet (not implemented yet)."""
        torch.manual_seed(seed)
        layer = LinearContraction(input_size, input_size, bias=True).to(device)
        flow = flow_cls(layer)

        x = torch.randn(self.BATCH_SIZE, input_size, device=device)
        y = flow.encode(x)
        xhat = flow.decode(y)

        assert y.shape == x.shape
        assert xhat.shape == x.shape
        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

        y = torch.randn(self.BATCH_SIZE, input_size, device=device)
        x = flow.decode(y)
        yhat = flow.encode(x)

        assert x.shape == y.shape
        assert yhat.shape == y.shape
        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)

    def test_gradients_of_contraction_bias(self) -> None:
        r"""Check $∂‖x⁎‖²/∂b$ for $g(x)=½x+b$ against the analytic gradient."""
        y = torch.tensor([[0.7, -0.2], [0.1, 0.3]], dtype=torch.float64)
        contraction = ShiftedHalfContraction(
            torch.tensor([0.1, -0.2], dtype=torch.float64)
        )
        flow = ContractiveFP(contraction)

        x_star = flow.decode(y)
        loss = x_star.square().sum()
        loss.backward()

        grad_expected = (-4.0 / 3.0) * x_star.detach().sum(dim=0)

        self.assert_close(
            contraction.bias.grad,
            grad_expected,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gradients_of_contraction_linear(self) -> None:
        r"""Check $∂‖x⁎‖²/∂A$ for $g(x)=Ax$ against the analytic gradient."""
        y = torch.tensor([[0.7, -0.2], [0.1, 0.3]], dtype=torch.float64)
        layer = LinearContraction(2, 2, bias=False, dtype=torch.float64, c=0.97)
        original_parameter = layer.parametrizations.weight.original_parameter
        expected_weight = torch.tensor(
            [[0.2, 0.05], [-0.03, 0.1]],
            dtype=torch.float64,
        )

        with torch.no_grad():
            original_parameter.copy_(expected_weight)
        effective_weight = layer.weight.detach().clone()

        flow = ContractiveFP(layer)
        x_star = flow.decode(y)
        loss = x_star.square().sum()
        loss.backward()

        eye = torch.eye(layer.in_features, dtype=torch.float64)
        solve_term = torch.linalg.solve(eye + effective_weight, x_star.detach().T).T
        grad_expected = -2 * x_star.detach().T @ solve_term

        self.assert_close(
            original_parameter.grad,
            grad_expected,
            atol=1e-10,
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "flow_cls",
        [ContractiveTransform, ContractiveFP, ContractiveNew],
        ids=["loop", "fixpoint_solve", "while_loop"],
    )
    @pytest.mark.parametrize("device", DEVICES, ids=str)
    def test_decode_performance(
        self,
        benchmark: BenchmarkFixture,
        flow_cls: type[TransformBase],
        device: str,
    ) -> None:
        r"""Benchmark the compiled inverse pass on a representative large input."""
        benchmark.group = (
            f"contractive_decode/{device}/seed={self.PERF_SEED}/"
            f"input_size={self.PERF_INPUT_SIZE}"
        )
        torch.manual_seed(self.PERF_SEED)
        layer = LinearContraction(
            self.PERF_INPUT_SIZE,
            self.PERF_INPUT_SIZE,
            bias=True,
        ).to(device)
        flow = flow_cls(layer)
        compiled_decode = torch.compile(
            flow.decode,
            fullgraph=flow_cls is ContractiveNew,
        )

        # trigger compile
        y_demo = torch.randn(self.BATCH_SIZE, self.PERF_INPUT_SIZE, device=device)
        compiled_decode(y_demo)

        def setup() -> tuple[tuple, dict]:
            y = torch.randn(self.BATCH_SIZE, self.PERF_INPUT_SIZE, device=device)
            return (y,), {}

        benchmark.pedantic(
            compiled_decode,
            setup=setup,
            rounds=self.PERF_ROUNDS,
            warmup_rounds=self.PERF_WARMUP_ROUNDS,
        )
