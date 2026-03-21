import pytest
import torch
from torch import Tensor

from linodenet_special.trace_estimation import hutchinson_estimator, xtrace_estimator
from tests.testing import DEVICES, DTYPES, TestCase


def make_matrix(
    matrix_kind: str,
    matrix_size: int,
    /,
    *,
    dtype: torch.dtype,
    device: str,
) -> Tensor:
    r"""Construct a test matrix with a controlled scale."""
    A = torch.randn(matrix_size, matrix_size, dtype=dtype, device=device)

    match matrix_kind:
        case "symmetric":
            return (A + A.mT) / (2 * matrix_size**0.5)
        case "skew_symmetric":
            return (A - A.mT) / (2 * matrix_size**0.5)
        case "randn":
            return A / matrix_size**0.5
        case _:
            raise ValueError(f"Unknown matrix_kind {matrix_kind!r}")


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("num_samples", [4, 8, 16, 32, 64], ids="samples={}".format)
@pytest.mark.parametrize("matrix_size", [4, 8, 16], ids="size={}".format)
@pytest.mark.parametrize("method", ["hutchinson", "xtrace"])
class TestCorrectness(TestCase):
    BATCH_SIZE = 32
    SEED = 1000
    ESTIMATORS = {
        "hutchinson": hutchinson_estimator,
        "xtrace": xtrace_estimator,
    }
    HUTCH_TOL = {
        (4, 4): (1e1, 1e0),
        (4, 8): (1e1, 1e0),
        (4, 16): (1e1, 1e0),
        (4, 32): (1e1, 1e0),
        (4, 64): (1e1, 1e0),
        (8, 4): (1e1, 1e0),
        (8, 8): (1e1, 1e0),
        (8, 16): (1e1, 1e0),
        (8, 32): (1e1, 1e0),
        (8, 64): (1e1, 1e0),
        (16, 4): (1e1, 1e0),
        (16, 8): (1e1, 1e0),
        (16, 16): (1e1, 1e0),
        (16, 32): (1e1, 1e0),
        (16, 64): (1e1, 1e0),
    }
    XTRACE_TOL = {
        (4, 4): (1e1, 1e0),
        (4, 8): (1e1, 1e0),
        (4, 16): (1e0, 1e0),
        (4, 32): (1e0, 1e0),
        (4, 64): (1e-1, 1e0),
        (8, 4): (1e1, 1e0),
        (8, 8): (1e1, 1e0),
        (8, 16): (1e1, 1e0),
        (8, 32): (1e0, 1e0),
        (8, 64): (1e-1, 1e0),
        (16, 4): (1e1, 1e0),
        (16, 8): (1e1, 1e0),
        (16, 16): (1e1, 1e0),
        (16, 32): (1e0, 1e0),
        (16, 64): (1e0, 1e0),
    }

    def make_problem(
        self,
        matrix_kind: str,
        matrix_size: int,
        /,
        *,
        num_samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> tuple[Tensor, Tensor]:
        torch.manual_seed(self.SEED)
        A = make_matrix(matrix_kind, matrix_size, dtype=dtype, device=device)
        samples = torch.randn(
            self.BATCH_SIZE,
            num_samples,
            matrix_size,
            dtype=dtype,
            device=device,
        )
        return A, samples

    def assert_trace_close(
        self,
        matrix_kind: str,
        method: str,
        matrix_size: int,
        num_samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        A, samples = self.make_problem(
            matrix_kind,
            matrix_size,
            num_samples=num_samples,
            dtype=dtype,
            device=device,
        )
        estimator = self.ESTIMATORS[method]
        estimate = estimator(
            lambda x: torch.einsum("...nd, md -> ...nm", x, A),
            samples,
        )
        truth = torch.trace(A)
        tol = self.HUTCH_TOL if method == "hutchinson" else self.XTRACE_TOL
        atol, rtol = tol[(matrix_size, num_samples)]

        self.assert_close(estimate, truth, atol=atol, rtol=rtol)

    def test_symmetric(
        self,
        method: str,
        matrix_size: int,
        num_samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.assert_trace_close(
            "symmetric", method, matrix_size, num_samples, dtype, device
        )

    def test_skew(
        self,
        method: str,
        matrix_size: int,
        num_samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.assert_trace_close(
            "skew_symmetric", method, matrix_size, num_samples, dtype, device
        )

    def test_randn(
        self,
        method: str,
        matrix_size: int,
        num_samples: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.assert_trace_close(
            "randn", method, matrix_size, num_samples, dtype, device
        )
