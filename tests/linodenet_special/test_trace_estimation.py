import itertools

import pytest
import torch
from torch import Tensor
from torch.linalg import matrix_norm

from linodenet_special.trace_estimation import (
    LogAbsDetEstimator,
    btrace_estimator,
    btrace_estimator_naive,
    btrace_estimator_new,
    hutchinson_estimator,
    xtrace_estimator,
    xtrace_estimator_corrected,
)
from tests.testing import DEVICES, TestCase


def ceil_power_of_ten(x: Tensor | float) -> float:
    r"""Return the least power of 10 strictly greater than x."""
    value = float(torch.as_tensor(x))
    if value <= 0.0:
        return 0.0

    exponent = int(torch.ceil(torch.log10(torch.tensor(value))).item())
    bound = 10.0**exponent
    if bound <= value:
        bound *= 10.0
    return bound


MATRIX_KINDS = ["randn", "symmetric", "skew-symmetric"]
MATRIX_SIZES = [32, 128, 512]
NUM_SAMPLES = {
    "same": 1,
    "half": 1 / 2,
    "small": 1 / 16,
}
ESTIMATORS = {
    "hutch": hutchinson_estimator,
    "xtrace": xtrace_estimator,
    "btrace": btrace_estimator,
    "xtrace-correction": xtrace_estimator_corrected,
    "btrace-correction": btrace_estimator_new,
}


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("samples", NUM_SAMPLES, ids="samples={}".format)
@pytest.mark.parametrize("size", MATRIX_SIZES, ids="size={}".format)
@pytest.mark.parametrize("method", ESTIMATORS)
class TestCorrectness(TestCase):
    NUM_SAMPLES = NUM_SAMPLES
    MATRIX_SIZES = MATRIX_SIZES
    MATRIX_KINDS = MATRIX_KINDS
    ESTIMATORS = ESTIMATORS

    BATCH_SIZE = 17
    SEED = 1000
    # indexed by kind, num_samples
    HUTCH_TOL = {
        ("randn", "same"):           1e-1,
        ("randn", "half"):           1e-1,
        ("randn", "small"):          1e0,
        ("skew-symmetric", "same"):  1e-8,
        ("skew-symmetric", "half"):  1e-8,
        ("skew-symmetric", "small"): 1e-7,
        ("symmetric", "same"):       1e-1,
        ("symmetric", "half"):       1e-1,
        ("symmetric", "small"):      1e0,
    }  # fmt: skip
    XTRACE_TOL = {
        ("randn", "same"):           1e-1,
        ("randn", "half"):           1e-1,
        ("randn", "small"):          1e0,
        ("skew-symmetric", "same"):  1e-7,
        ("skew-symmetric", "half"):  1e-7,
        ("skew-symmetric", "small"): 1e-7,
        ("symmetric", "same"):       1e-1,
        ("symmetric", "half"):       1e-1,
        ("symmetric", "small"):      1e0,
    }  # fmt: skip

    def make_problem(
        self,
        kind: str,
        size: int,
        /,
        *,
        samples: str | int,
        device: str,
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED)
        num_samples = (
            int(size * self.NUM_SAMPLES[samples])
            if isinstance(samples, str)
            else samples
        )

        match kind:
            case "symmetric":
                A = torch.randn(self.BATCH_SIZE, size, size, device=device)
                A = (A + A.mT) / (2 * size**0.5)
            case "skew-symmetric":
                A = torch.randn(self.BATCH_SIZE, size, size, device=device)
                A = (A - A.mT) / (2 * size**0.5)
            case "randn":
                A = torch.randn(self.BATCH_SIZE, size, size, device=device)
                A = A / size**0.5
            case _:
                raise ValueError(f"Unknown matrix_kind {kind!r}")

        X = torch.randn(self.BATCH_SIZE, num_samples, size, device=device)
        # X = torch.eye(size, device=device).expand(self.BATCH_SIZE, size, size) * (
        #     size**0.5
        # )
        # orthogonal correction.
        # Q, _ = torch.linalg.qr(X.mT, mode="reduced")  # (..., n, k) when k >= n
        # X = (num_samples**0.5) * Q.mT

        # assert X.shape == (self.BATCH_SIZE, num_samples, size)
        return A, X

    def compute_estimate(self, method: str, A: Tensor, x: Tensor) -> Tensor:
        match method:
            case "hutch":
                assert ESTIMATORS[method] is hutchinson_estimator
                return hutchinson_estimator(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A), x
                )
            case "xtrace":
                assert ESTIMATORS[method] is xtrace_estimator
                return xtrace_estimator(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A), x
                )
            case "xtrace-correction":
                assert ESTIMATORS[method] is xtrace_estimator_corrected
                return xtrace_estimator_corrected(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A), x
                )
            case "btrace":
                assert ESTIMATORS[method] is btrace_estimator
                return btrace_estimator(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A),
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A.mT),
                    x,
                    x,
                )
            case "btrace-correction":
                assert ESTIMATORS[method] is btrace_estimator_new
                return btrace_estimator(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A),
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A.mT),
                    x,
                    x,
                )
            case "btrace-naive":
                assert ESTIMATORS[method] is btrace_estimator_naive
                return btrace_estimator_naive(
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A),
                    lambda v: torch.einsum("...nd, ...md -> ...nm", v, A.mT),
                    x,
                    x,
                )
            case _:
                raise ValueError(f"Unknown estimation method {method!r}")

    def assert_trace_close(
        self,
        *,
        kind: str,
        method: str,
        size: int,
        samples: str,
        device: str,
        debug: bool = False,
    ) -> None:
        A, z = self.make_problem(kind, size, samples=samples, device=device)
        estimate = self.compute_estimate(method, A, z)
        truth = torch.einsum("...kk -> ...", A)  # batched trace
        errors = estimate - truth

        # scaling: see XTrace paper.
        # Var[tr] = E[|tr - tr(A)|²] ≤ η‖A‖⁎² ⇝ scaled_rmse ≤ η
        # reminder: ‖A‖⁎ = nuclear norm = sum of singular values.
        norms = matrix_norm(A, ord="nuc")
        scaled_rmse = (errors / norms).square().mean().sqrt()

        if debug:
            scaled_upper = ceil_power_of_ten(scaled_rmse)
            print(
                f"{method}, {kind=}, {size=:2d}, {samples=}, "
                f"{scaled_rmse=:.4f} (<{scaled_upper:.0e}), "
                f"‖A‖⁎≈{norms.mean():.2f}"
            )
            return

        tol = self.HUTCH_TOL if method == "hutchinson" else self.XTRACE_TOL
        tol_expected = tol[kind, samples]
        self.assert_upper_bounded(scaled_rmse, tol_expected)

    def test_symmetric(
        self, *, method: str, size: int, samples: str, device: str
    ) -> None:
        self.assert_trace_close(
            kind="symmetric", method=method, size=size, samples=samples, device=device
        )

    def test_skew(self, *, method: str, size: int, samples: str, device: str) -> None:
        self.assert_trace_close(
            kind="skew-symmetric",
            method=method,
            size=size,
            samples=samples,
            device=device,
        )

    def test_randn(self, *, method: str, size: int, samples: str, device: str) -> None:
        self.assert_trace_close(
            kind="randn",
            method=method,
            size=size,
            samples=samples,
            device=device,
        )


@pytest.mark.parametrize("method", TestCorrectness.ESTIMATORS)
@pytest.mark.parametrize("matrix_kind", TestCorrectness.MATRIX_KINDS)
def test_calibration(matrix_kind: str, method: str) -> None:
    suite = TestCorrectness()
    print()
    for matrix_size, samples in itertools.product(
        suite.MATRIX_SIZES, suite.NUM_SAMPLES
    ):
        suite.assert_trace_close(
            kind=matrix_kind,
            method=method,
            size=matrix_size,
            samples=samples,
            device="cuda" if torch.cuda.is_available() else "cpu",
            debug=True,
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_xtrace_estimator_corrected_single_and_batched(device: str) -> None:
    size = 4
    samples = torch.eye(size, device=device)

    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], device=device))
    trace = torch.trace(matrix)
    estimate = xtrace_estimator_corrected(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, matrix),
        samples,
    )
    torch.testing.assert_close(estimate, trace)

    batched_matrix = torch.stack(
        [
            matrix,
            torch.diag(torch.tensor([0.5, 1.5, 2.5, 3.5], device=device)),
        ]
    )
    batched_samples = samples.expand(len(batched_matrix), -1, -1)
    batched_trace = torch.einsum("...kk -> ...", batched_matrix)
    batched_estimate = xtrace_estimator_corrected(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, batched_matrix),
        batched_samples,
    )
    torch.testing.assert_close(batched_estimate, batched_trace)


def test_btrace_naive(device: str) -> None:
    size = 4
    samples = torch.eye(size)

    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    trace = torch.trace(matrix)
    estimate = btrace_estimator_naive(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, matrix),
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, matrix.mT),
        samples,
        samples,
    )
    torch.testing.assert_close(estimate, trace)

    batched_matrix = torch.stack(
        [
            matrix,
            torch.diag(torch.tensor([0.5, 1.5, 2.5, 3.5])),
        ]
    )
    batched_samples = samples.expand(len(batched_matrix), -1, -1)
    batched_trace = torch.einsum("...kk -> ...", batched_matrix)
    batched_estimate = btrace_estimator_naive(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, batched_matrix),
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, batched_matrix.mT),
        batched_samples,
        batched_samples,
    )
    torch.testing.assert_close(batched_estimate, batched_trace)


def test_btrace() -> None:
    size = 4
    samples = torch.eye(size)
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    matrix = torch.randn(4, 4)
    trace = torch.trace(matrix)
    estimate = btrace_estimator(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, matrix),
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, matrix.mT),
        samples,
        samples,
    )
    torch.testing.assert_close(estimate, trace)

    batched_matrix = torch.stack([matrix, matrix])
    batched_samples = samples.expand(len(batched_matrix), -1, -1)
    batched_trace = torch.einsum("...kk -> ...", batched_matrix)
    batched_estimate = btrace_estimator(
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, batched_matrix),
        lambda x: torch.einsum("...nd, ...md -> ...nm", x, batched_matrix.mT),
        batched_samples,
        batched_samples,
    )
    torch.testing.assert_close(batched_estimate, batched_trace)


class ScaledMap(torch.nn.Module):
    def __init__(self, scale: float, /) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor, /) -> Tensor:
        return self.scale * x


class TestLogAbsDetEstimator(TestCase):
    BATCH_SIZE = 16
    INPUT_SIZE = 4

    @pytest.mark.parametrize(
        ("method", "num_samples", "num_series_terms", "expected_method"),
        [
            ("exact", None, None, "compute_exact"),
            ("hutch", 8, 4, "compute_hutch"),
            ("xtrace", 8, 4, "compute_xtrace"),
        ],
    )
    def test_logabsdet_estimator_dispatch(
        self,
        method: str,
        num_samples: int | None,
        num_series_terms: int | None,
        expected_method: str,
    ) -> None:
        estimator = LogAbsDetEstimator(method, num_samples, num_series_terms)
        assert estimator.method.__name__ == expected_method

        fn = ScaledMap(0.125)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE)
        y, logabsdet = estimator(fn, x)

    @pytest.mark.parametrize("device", DEVICES, ids=str)
    def test_exact_estimator_matches_closed_form(self, device: str) -> None:
        estimator = LogAbsDetEstimator("exact", None, None).to(device=device)
        fn = ScaledMap(0.125).to(device=device)
        x = torch.randn(self.BATCH_SIZE, self.INPUT_SIZE, device=device)

        y, logabsdet = estimator(fn, x)

        expected_y = 0.125 * x
        expected_logabsdet = torch.full(
            (7,),
            4 * torch.log1p(torch.tensor(0.125, device=device)).item(),
            device=device,
        )
        assert torch.allclose(y, expected_y)
        assert torch.allclose(logabsdet, expected_logabsdet, atol=1e-6, rtol=0.0)
