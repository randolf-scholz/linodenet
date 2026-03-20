import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn

from linodenet.mappings import (
    ContractiveFP,
    ContractiveTransform,
    TransformBase,
)
from linodenet.mappings.linear import LinearContraction
from tests.testing import DEVICES, DTYPES, SEEDS_5, TestCase


class ShiftedHalfContraction(nn.Module):
    r"""Simple contraction $g(x) = ½x + b$ with trainable bias."""

    def __init__(self, bias: Tensor, /) -> None:
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return 0.5 * x + self.bias


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "flow_cls",
    [ContractiveTransform, ContractiveFP],
    ids=["loop", "fixpoint_solve"],
)
class TestCorrectness(TestCase):
    BATCH_SIZE = 32
    INPUT_SIZE = 8
    VALUE_TOL = {
        torch.float32: (1e-3, 1e-3),
        torch.float64: (2e-6, 1e-3),
    }
    GRAD_TOL = {
        torch.float32: (1e-3, 1e-3),
        torch.float64: (1e-5, 1e-5),
    }

    @pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
    @pytest.mark.parametrize("input_size", [4, 16, 64], ids="input_size={}".format)
    def test_invertibility(
        self,
        flow_cls: type[TransformBase],
        input_size: int,
        dtype: torch.dtype,
        device: str,
        seed: int,
    ) -> None:
        r"""Check forward/inverse round trips; does not test logabsdet (not implemented yet)."""
        atol, rtol = self.VALUE_TOL[dtype]
        torch.manual_seed(seed)
        layer = LinearContraction(
            input_size,
            input_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        flow = flow_cls(layer)

        x = torch.randn(self.BATCH_SIZE, input_size, device=device, dtype=dtype)
        y = flow.encode(x)
        xhat = flow.decode(y)

        assert y.shape == x.shape
        assert xhat.shape == x.shape
        self.assert_close(xhat, x, atol=atol, rtol=rtol)

        y = torch.randn(self.BATCH_SIZE, input_size, device=device, dtype=dtype)
        x = flow.decode(y)
        yhat = flow.encode(x)

        assert x.shape == y.shape
        assert yhat.shape == y.shape
        self.assert_close(yhat, y, atol=atol, rtol=rtol)

    def test_gradients_of_contraction_bias(
        self,
        flow_cls: type[TransformBase],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        r"""Check $∂‖x⁎‖²/∂b$ for $g(x)=½x+b$ against the analytic gradient."""
        atol, rtol = self.GRAD_TOL[dtype]
        y = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        bias = torch.randn(self.INPUT_SIZE, device=device, dtype=dtype)
        contraction = ShiftedHalfContraction(bias)
        flow = flow_cls(contraction)

        x_star = flow.decode(y)
        loss = x_star.square().sum()
        loss.backward()

        grad_expected = (-4.0 / 3.0) * x_star.detach().sum(dim=0)
        assert contraction.bias.grad is not None
        self.assert_close(
            contraction.bias.grad,
            grad_expected,
            atol=atol,
            rtol=rtol,
        )

    def test_gradients_of_contraction_linear(
        self,
        flow_cls: type[TransformBase],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        r"""Check $∂‖x⁎‖²/∂A$ for $g(x)=Ax$ against the analytic gradient."""
        atol, rtol = self.GRAD_TOL[dtype]
        y = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        layer = LinearContraction(
            self.INPUT_SIZE,
            self.INPUT_SIZE,
            bias=False,
            device=device,
            dtype=dtype,
            c=0.97,
        )
        flow = flow_cls(layer)
        x_star = flow.decode(y)

        with torch.no_grad():
            # Note: with `nn.Linear`, g(X) = XAᵀ, so
            #   X⁎ = Y - X⁎Aᵀ  ⟺  X⁎(I + Aᵀ) = Y.
            # Let M = I + Aᵀ. Then X⁎ = YM⁻¹ and
            #   ∆X⁎ = -YM⁻¹(∆A)ᵀM⁻¹ = -X⁎(∆A)ᵀM⁻¹.
            # For L = ‖X⁎‖² = tr(X⁎ᵀX⁎),
            #   ∆L = 2 tr(X⁎ᵀ∆X⁎)
            #      = -2 tr(M⁻¹X⁎ᵀX⁎(∆A)ᵀ)
            #      = -2 tr((M⁻¹X⁎ᵀX⁎)ᵀ∆A),
            # hence
            #   ∂L/∂A = -2M⁻¹X⁎ᵀX⁎ = -2(X⁎(I + A)⁻¹)ᵀX⁎.
            eye = torch.eye(layer.in_features, device=device, dtype=dtype)
            solve_term = torch.linalg.solve(eye + layer.weight, x_star, left=False)
            expected_grad = -2 * solve_term.T @ x_star

        loss = x_star.square().sum()
        loss.backward()
        assert layer.weight_parameter.grad is not None

        self.assert_close(
            layer.weight_parameter.grad,
            expected_grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "flow_cls",
    [ContractiveTransform, ContractiveFP],
    ids=["loop", "fixpoint_solve"],
)
class TestPerformance(TestCase):
    BATCH_SIZE = 32
    PERF_SEED = 0
    PERF_INPUT_SIZE = 256
    PERF_ROUNDS = 100
    PERF_WARMUP_ROUNDS = 1

    def test_decode_performance(
        self,
        benchmark: BenchmarkFixture,
        flow_cls: type[TransformBase],
        dtype: torch.dtype,
        device: str,
    ) -> None:
        r"""Benchmark the compiled inverse pass on a representative large input."""
        benchmark.group = (
            f"contractive_decode/{device}/{dtype}/seed={self.PERF_SEED}/"
            f"input_size={self.PERF_INPUT_SIZE}"
        )
        torch.manual_seed(self.PERF_SEED)
        layer = LinearContraction(
            self.PERF_INPUT_SIZE,
            self.PERF_INPUT_SIZE,
            bias=True,
            device=device,
            dtype=dtype,
        )
        flow = flow_cls(layer)

        # NOTE: required for fullgraph=True
        # REF: https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
        torch._dynamo.config.compiled_autograd = True  # noqa: SLF001
        compiled_decode = torch.compile(
            flow.decode,
            fullgraph=flow_cls is ContractiveFP,
        )

        # trigger compile
        y_demo = torch.randn(
            self.BATCH_SIZE,
            self.PERF_INPUT_SIZE,
            device=device,
            dtype=dtype,
        )
        compiled_decode(y_demo)

        def setup() -> tuple[tuple, dict]:
            y = torch.randn(
                self.BATCH_SIZE,
                self.PERF_INPUT_SIZE,
                device=device,
                dtype=dtype,
            )
            return (y,), {}

        benchmark.pedantic(
            compiled_decode,
            setup=setup,
            rounds=self.PERF_ROUNDS,
            warmup_rounds=self.PERF_WARMUP_ROUNDS,
        )
