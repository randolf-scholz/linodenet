import itertools

import pytest
import torch
from torch import Tensor
from torch.linalg import matrix_norm

from linodenet_special.trace_estimation import (
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


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("samples", NUM_SAMPLES, ids="samples={}".format)
@pytest.mark.parametrize("size", MATRIX_SIZES, ids="size={}".format)
@pytest.mark.parametrize("method", ["hutchinson", "xtrace"])
class TestCorrectness(TestCase):
    NUM_SAMPLES = NUM_SAMPLES
    MATRIX_SIZES = MATRIX_SIZES
    MATRIX_KINDS = MATRIX_KINDS
    BATCH_SIZE = 17
    SEED = 1000
    ESTIMATORS = {
        "hutchinson": hutchinson_estimator,
        "xtrace": xtrace_estimator,
        "correct": xtrace_estimator_corrected,
    }
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
        estimator = self.ESTIMATORS[method]
        estimate = estimator(
            lambda x: torch.einsum("...nd, ...md -> ...nm", x, A),
            z,
        )
        truth = torch.einsum("...kk -> ...", A)  # batched trace
        errors = estimate - truth

        # scaling: see XTrace paper.
        # Var[tr] = E[|tr - tr(A)|²] ≤ η‖A‖⁎² ⇝ scaled_rmse ≤ η
        # reminder: ‖A‖⁎ = nuclear norm = sum of singular values.
        rmse = errors.square().mean().sqrt()
        norms = matrix_norm(A, ord="nuc")
        scaled_rmse = (errors / norms).square().mean().sqrt()

        if debug:
            rmse_upper = ceil_power_of_ten(rmse)
            scaled_upper = ceil_power_of_ten(scaled_rmse)
            print(
                f"{method=}, {kind=}, {size=:2d}, {samples=}, "
                f"{rmse=:.4f} (<{rmse_upper:.0e}), "
                f"{scaled_rmse=:.4f} (<{scaled_upper:.0e}), ‖A‖⁎≈{norms.mean():.2f}, "
                f"{device=}"
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
