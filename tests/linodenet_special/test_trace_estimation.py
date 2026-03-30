from collections.abc import Callable

import torch
from torch import Tensor

from tests.testing import TestSuite


def linear_map(matrix: Tensor, /) -> Callable[[Tensor], Tensor]:
    def op(x: Tensor, /) -> Tensor:
        return torch.einsum("...ij, ...j -> ...i", matrix, x)

    return op


class TestTraceEstimator(TestSuite):
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    DTYPE = torch.float32

    def make_diagonal(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        diagonal = 0.5 + torch.rand(batch_size, input_size, device=device, dtype=dtype)
        matrix = torch.diag_embed(diagonal)
        trace = diagonal.sum(dim=-1)
        return matrix, trace

    def make_gaussian(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = (
            torch.randn(
                batch_size,
                input_size,
                input_size,
                device=device,
                dtype=dtype,
            )
            / input_size**0.5
        )
        trace = torch.einsum("...ii -> ...", matrix)
        return matrix, trace

    def make_symmetric(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        matrix = (matrix + matrix.mT) / (2 * input_size**0.5)
        trace = torch.einsum("...ii -> ...", matrix)
        return matrix, trace

    def make_skew_symmetric(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        matrix = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        matrix = (matrix - matrix.mT) / (2 * input_size**0.5)
        trace = torch.zeros(batch_size, device=device, dtype=dtype)
        return matrix, trace

    def make_linear_spectrum(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        spectrum = torch.linspace(0, 2, input_size, device=device, dtype=dtype)
        spectrum = spectrum.expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def make_exponential_spectrum(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        exponents = torch.arange(
            -(input_size // 2),
            (input_size + 1) // 2,
            device=device,
            dtype=dtype,
        )
        spectrum = (1.25**exponents).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def make_low_rank(
        self,
        /,
        *,
        batch_size: int | None = None,
        input_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch_size = self.BATCH_SIZE if batch_size is None else batch_size
        input_size = self.INPUT_SIZE if input_size is None else input_size
        dtype = self.DTYPE if dtype is None else dtype

        q = self._make_orthogonal_batch(
            batch_size=batch_size,
            input_size=input_size,
            dtype=dtype,
            device=device,
        )
        rank = input_size // 16
        spectrum = torch.cat(
            [
                torch.ones(rank, device=device, dtype=dtype),
                torch.zeros(input_size - rank, device=device, dtype=dtype),
            ]
        ).expand(batch_size, -1)
        matrix = torch.einsum("...ik, ...k, ...jk -> ...ij", q, spectrum, q)
        trace = spectrum.sum(dim=-1)
        return matrix, trace

    def _make_orthogonal_batch(
        self,
        /,
        *,
        batch_size: int,
        input_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> Tensor:
        gaussian = torch.randn(
            batch_size,
            input_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        q, _ = torch.linalg.qr(gaussian)
        return q


class TestTraceCorrectness(TestTraceEstimator):
    pass


class TestPowersCorrectness(TestTraceEstimator):
    pass


class TestLogAbsDetCorrectness(TestTraceEstimator):
    pass


class TestVisualizations(TestTraceEstimator):
    pass
