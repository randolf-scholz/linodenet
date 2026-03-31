import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from linodenet.mappings import (
    LinearContraction,
    ResidualContraction,
    ResidualContractionFallback,
    TransformBase,
)
from linodenet.nn.parametrize import update_parametrizations
from tests.testing import DEVICES, DTYPES, SEEDS_5, TestSuite, pytest_xfail


class ShiftedHalfContraction(nn.Module):
    r"""Simple contraction $g(x) = ½x + b$ with trainable bias."""

    def __init__(self, bias: Tensor, /) -> None:
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return 0.5 * x + self.bias


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestReZero(TestSuite):
    BATCH_SIZE = 32
    INPUT_SIZE = 8
    TRAIN_STEPS = 5
    LEARNING_RATE = 0.5
    TARGET_SCALE = 0.35

    def make_model(self, /, *, device: str, dtype: torch.dtype) -> ResidualContraction:
        contraction = nn.Sequential(
            LinearContraction(self.INPUT_SIZE, 2 * self.INPUT_SIZE, bias=True),
            nn.ReLU(),
            LinearContraction(2 * self.INPUT_SIZE, self.INPUT_SIZE),
        )
        module = ResidualContraction(contraction, use_rezero=True)
        module = module.to(dtype=dtype, device=device)
        update_parametrizations(module)  # Important after .to()
        return module

    def make_test_case(self, device: str, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        target = x**2 - 1
        return x, target

    def test_initialization_is_identity(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        flow = self.make_model(device=device, dtype=dtype)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        y = flow.encode(x)
        self.assert_close(y, x, atol=0.0, rtol=0.0)

    def test_can_train(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(0)
        flow = self.make_model(device=device, dtype=dtype)
        x, target = self.make_test_case(device=device, dtype=dtype)

        with torch.no_grad():
            initial_loss = mse_loss(flow.encode(x), target)

        optimizer = torch.optim.SGD([flow.scalar], lr=self.LEARNING_RATE)

        for _ in range(self.TRAIN_STEPS):
            optimizer.zero_grad()
            loss = mse_loss(flow.encode(x), target)
            loss.backward()
            optimizer.step()
            update_parametrizations(flow)

        final_loss = mse_loss(flow.encode(x), target)

        assert final_loss < initial_loss

    def test_after_training_is_no_longer_identity(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(0)
        flow = self.make_model(device=device, dtype=dtype)
        x, target = self.make_test_case(device=device, dtype=dtype)

        optimizer = torch.optim.SGD([flow.scalar], lr=self.LEARNING_RATE)

        for _ in range(self.TRAIN_STEPS):
            optimizer.zero_grad()
            loss = mse_loss(flow.encode(x), target)
            loss.backward()
            optimizer.step()
            update_parametrizations(flow)

        y = flow.encode(x)

        self.assert_not_close(y, x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "flow_cls",
    [ResidualContractionFallback, ResidualContraction],
    ids=["loop", "fixpoint_solve"],
)
class TestCorrectness(TestSuite):
    BATCH_SIZE = 32
    INPUT_SIZE = 8
    FLOW_MAXITER = 128
    FLOW_ATOL = {
        torch.float32: 1e-6,
        torch.float64: 1e-8,
    }
    FLOW_RTOL = {
        torch.float32: 1e-6,
        torch.float64: 1e-8,
    }
    VALUE_TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-6, 1e-6),
    }
    GRAD_TOL = {
        torch.float32: (1e-4, 1e-4),
        torch.float64: (1e-6, 1e-6),
    }

    @pytest_xfail(
        condition=lambda *_, flow_cls, **__: flow_cls is ResidualContractionFallback,
        raises=AssertionError,
        reason="fallback is less accurate",
        strict=False,
    )
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
        torch.manual_seed(seed)
        atol, rtol = self.VALUE_TOL[dtype]
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

    @pytest_xfail(
        condition=lambda *_, flow_cls, **__: flow_cls is ResidualContractionFallback,
        raises=AssertionError,
        reason="fallback is less accurate",
        strict=False,
    )
    def test_gradients_of_contraction_bias(
        self,
        flow_cls: type[TransformBase],
        dtype: torch.dtype,
        device: str,
        seed: int,
    ) -> None:
        r"""Check $∂‖x⁎‖²/∂b$ for $g(x)=½x+b$ against the analytic gradient."""
        torch.manual_seed(seed)
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

    @pytest_xfail(
        condition=lambda *_, flow_cls, **__: flow_cls is ResidualContractionFallback,
        raises=AssertionError,
        reason="fallback is less accurate",
        strict=False,
    )
    def test_gradients_of_contraction_linear(
        self,
        flow_cls: type[TransformBase],
        dtype: torch.dtype,
        device: str,
        seed: int,
    ) -> None:
        r"""Check $∂‖x⁎‖²/∂A$ for the effective weight in $g(x)=Ax$ against the analytic gradient."""
        torch.manual_seed(seed)
        atol, rtol = self.GRAD_TOL[dtype]
        flow_atol = self.FLOW_ATOL[dtype]
        flow_rtol = self.FLOW_RTOL[dtype]
        y = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)
        layer = LinearContraction(
            self.INPUT_SIZE,
            self.INPUT_SIZE,
            bias=False,
            device=device,
            dtype=dtype,
            c=0.95,
        )
        flow = flow_cls(
            layer,
            maxiter=self.FLOW_MAXITER,
            atol=flow_atol,
            rtol=flow_rtol,
        )
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
        (actual_grad,) = torch.autograd.grad(loss, layer.weight)

        self.assert_close(
            actual_grad,
            expected_grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trace_estimator", ["hutch", "hutch++"])
class TestLogAbsDet(TestSuite):
    BATCH_SIZE = 8
    INPUT_SIZE = 4
    SCALE = 0.125
    SEED = 0
    VALUE_TOL = 0.0
    LOGABSDET_TOL = 2e-2
    NUM_TRACE_SAMPLES = 256
    NUM_SERIES_TERMS = 10

    class ScaledContraction(nn.Module):
        r"""Simple contraction $g(x) = αx$ with scalar $|α| < 1$."""

        scale: Tensor

        def __init__(self, scale: float, /) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(scale))

        def forward(self, x: Tensor, /) -> Tensor:
            return self.scale * x

    def test_scaled_contraction_matches_closed_form(
        self,
        dtype: torch.dtype,
        device: str,
        trace_estimator: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        flow = ResidualContraction(
            self.ScaledContraction(self.SCALE),
            trace_matvecs=self.NUM_TRACE_SAMPLES,
            num_series_terms=self.NUM_SERIES_TERMS,
            trace_estimator=trace_estimator,
        ).to(device=device, dtype=dtype)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)

        y, logabsdet = flow.encode_and_logabsdet(x)
        assert y.shape == (self.BATCH_SIZE, self.INPUT_SIZE)
        assert logabsdet.shape == (self.BATCH_SIZE,)

        expected_y = (1 + self.SCALE) * x
        expected_logabsdet = torch.full(
            (self.BATCH_SIZE,),
            self.INPUT_SIZE * torch.log1p(torch.tensor(self.SCALE, dtype=dtype)).item(),
            device=device,
            dtype=dtype,
        )
        self.assert_close(y, expected_y, atol=self.VALUE_TOL, rtol=0.0)
        self.assert_close(
            logabsdet,
            expected_logabsdet,
            atol=self.LOGABSDET_TOL,
            rtol=0.0,
        )


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestLogAbsDetExact(TestSuite):
    BATCH_SIZE = TestLogAbsDet.BATCH_SIZE
    INPUT_SIZE = TestLogAbsDet.INPUT_SIZE
    SCALE = TestLogAbsDet.SCALE
    SEED = TestLogAbsDet.SEED
    VALUE_TOL = TestLogAbsDet.VALUE_TOL
    NUM_TRACE_SAMPLES = TestLogAbsDet.NUM_TRACE_SAMPLES
    NUM_SERIES_TERMS = TestLogAbsDet.NUM_SERIES_TERMS
    ScaledContraction = TestLogAbsDet.ScaledContraction

    def test_scaled_contraction_matches_closed_form_exact(
        self,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        torch.manual_seed(self.SEED)
        flow = ResidualContraction(
            self.ScaledContraction(self.SCALE),
            trace_matvecs=self.NUM_TRACE_SAMPLES,
            num_series_terms=self.NUM_SERIES_TERMS,
            trace_estimator="exact",
        ).to(device=device, dtype=dtype)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device, dtype=dtype)

        y, logabsdet = flow.encode_and_logabsdet(x)
        assert y.shape == (self.BATCH_SIZE, self.INPUT_SIZE)
        assert logabsdet.shape == (self.BATCH_SIZE,)

        expected_y = (1 + self.SCALE) * x
        expected_logabsdet = torch.full(
            (self.BATCH_SIZE,),
            self.INPUT_SIZE * torch.log1p(torch.tensor(self.SCALE, dtype=dtype)).item(),
            device=device,
            dtype=dtype,
        )
        self.assert_close(y, expected_y, atol=self.VALUE_TOL, rtol=0.0)
        self.assert_close(logabsdet, expected_logabsdet, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "flow_cls",
    [ResidualContractionFallback, ResidualContraction],
    ids=["loop", "fixpoint_solve"],
)
class TestPerformance(TestSuite):
    BATCH_SIZE = 32
    PERF_SEED = 0
    PERF_INPUT_SIZE = 256
    PERF_ROUNDS = 20
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
        # pyrefly: ignore[bad-assignment]
        torch._dynamo.config.compiled_autograd = True  # noqa: SLF001
        compiled_decode = torch.compile(
            flow.decode,
            fullgraph=flow_cls is ResidualContraction,
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
