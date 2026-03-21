import itertools

import pytest
import torch
from torch import Tensor
from torch.linalg import matrix_norm

from linodenet_special.trace_estimation import hutchinson_estimator, xtrace_estimator
from tests.testing import DEVICES, DTYPES, TestCase


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


NUM_SAMPLES = [16, 32, 64]
MATRIX_SIZES = [64, 128, 256]
MATRIX_KINDS = ["randn", "symmetric", "skew-symmetric"]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("samples", NUM_SAMPLES, ids="samples={}".format)
@pytest.mark.parametrize("size", MATRIX_SIZES, ids="size={}".format)
@pytest.mark.parametrize("method", ["hutchinson", "xtrace"])
class TestCorrectness(TestCase):
    NUM_SAMPLES = NUM_SAMPLES
    MATRIX_SIZES = MATRIX_SIZES
    MATRIX_KINDS = MATRIX_KINDS
    BATCH_SIZE = 32
    SEED = 1000
    ESTIMATORS = {
        "hutchinson": hutchinson_estimator,
        "xtrace": xtrace_estimator,
    }
    HUTCH_TOL = {
        ("randn", 8):             1e0,
        ("randn", 32):            1e-1,
        ("randn", 128):           1e-1,
        ("skew-symmetric", 8):    1e-8,
        ("skew-symmetric", 32):   1e-8,
        ("skew-symmetric", 128):  1e-8,
        ("symmetric", 8):         1e0,
        ("symmetric", 32):        1e0,
        ("symmetric", 128):       1e-1,
    }  # fmt: skip
    XTRACE_TOL = {
        ("randn", 8):            1e1,
        ("randn", 32):           1e-1,
        ("randn", 128):          1e-4,
        ("skew-symmetric", 8):   1e-13,
        ("skew-symmetric", 32):  1e-13,
        ("skew-symmetric", 128): 1e-13,
        ("symmetric", 8):        1e1,
        ("symmetric", 32):       1e-1,
        ("symmetric", 128):      1e-4,
    }  # fmt: skip

    def make_problem(
        self,
        kind: str,
        size: int,
        /,
        *,
        samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED)

        match kind:
            case "symmetric":
                A = torch.randn(self.BATCH_SIZE, size, size, dtype=dtype, device=device)
                A = (A + A.mT) / (2 * size**0.5)
            case "skew-symmetric":
                A = torch.randn(self.BATCH_SIZE, size, size, dtype=dtype, device=device)
                A = (A - A.mT) / (2 * size**0.5)
            case "randn":
                A = torch.randn(self.BATCH_SIZE, size, size, dtype=dtype, device=device)
                A = A / size**0.5
            case _:
                raise ValueError(f"Unknown matrix_kind {kind!r}")

        x = torch.randn(
            self.BATCH_SIZE,
            samples,
            size,
            dtype=dtype,
            device=device,
        )
        return A, x

    def assert_trace_close(
        self,
        *,
        kind: str,
        method: str,
        size: int,
        samples: int,
        dtype: torch.dtype,
        device: str,
        debug: bool = False,
    ) -> None:
        A, z = self.make_problem(
            kind,
            size,
            samples=samples,
            dtype=dtype,
            device=device,
        )
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
                f"{method=}, {kind=}, {size=:2d}, {samples=:3d}, "
                f"{rmse=:.4f} (<{rmse_upper:.0e}), "
                f"{scaled_rmse=:.4f} (<{scaled_upper:.0e}), ‖A‖⁎≈{norms.mean():.2f}, "
                f"{dtype=}, {device=}"
            )
            return

        tol = self.HUTCH_TOL if method == "hutchinson" else self.XTRACE_TOL
        tol_expected = tol[kind, samples]
        self.assert_upper_bounded(scaled_rmse, tol_expected)

    def test_symmetric(
        self, *, method: str, size: int, samples: int, dtype: torch.dtype, device: str
    ) -> None:
        self.assert_trace_close(
            kind="symmetric",
            method=method,
            size=size,
            samples=samples,
            dtype=dtype,
            device=device,
        )

    def test_skew(
        self, *, method: str, size: int, samples: int, dtype: torch.dtype, device: str
    ) -> None:
        self.assert_trace_close(
            kind="skew-symmetric",
            method=method,
            size=size,
            samples=samples,
            dtype=dtype,
            device=device,
        )

    def test_randn(
        self, *, method: str, size: int, samples: int, dtype: torch.dtype, device: str
    ) -> None:
        self.assert_trace_close(
            kind="randn",
            method=method,
            size=size,
            samples=samples,
            dtype=dtype,
            device=device,
        )


@pytest.mark.parametrize("method", TestCorrectness.ESTIMATORS)
@pytest.mark.parametrize("matrix_kind", TestCorrectness.MATRIX_KINDS)
def test_calibration(matrix_kind: str, method: str) -> None:
    suite = TestCorrectness()
    print()
    for matrix_size, num_samples in itertools.product(
        suite.MATRIX_SIZES, suite.NUM_SAMPLES
    ):
        suite.assert_trace_close(
            kind=matrix_kind,
            method=method,
            size=matrix_size,
            samples=num_samples,
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            debug=True,
        )
