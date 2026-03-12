r"""Unit lower-triangular linear flow."""

__all__ = ["TriangularFlow"]

from typing import Final

import torch
from torch import Tensor, nn

from signatures import signature

from .base import TransformBase


class TriangularFlow(TransformBase):
    r"""An invertible linear layer with unit lower-triangular Jacobian.

    The transformation is parameterized as

    .. math:: y = (𝕀ₙ + L)x

    where $L$ is strictly lower triangular. This makes the weight matrix
    unit lower triangular, hence always invertible with determinant 1.
    """

    input_size: Final[int]
    r"""CONST: Input and output dimensionality."""

    lower: Tensor
    r"""PARAM: Unconstrained matrix whose strictly lower part defines the flow."""

    @property
    def config(self) -> dict[str, int]:
        return {"input_size": self.input_size}

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.lower = nn.Parameter(torch.zeros(input_size, input_size))

    @property
    def weight(self) -> Tensor:
        r"""Return the unit lower-triangular weight matrix."""
        return torch.eye(
            self.input_size,
            device=self.lower.device,
            dtype=self.lower.dtype,
        ) + self.lower.tril(diagonal=-1)

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor, /) -> Tensor:
        r"""Compute :math:`y = (\mathbb{I}_n + L)x`."""
        lower = self.lower.tril(diagonal=-1)
        update = torch.einsum("mn, ...n -> ...m", lower, x)
        return x + update

    @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor, /) -> Tensor:
        r"""Solve :math:`(\mathbb{I}_n + L)x = y` for :math:`x`."""
        x = torch.linalg.solve_triangular(
            self.weight,
            y[..., None],
            upper=False,
            unitriangular=True,
        )
        return x.squeeze(-1)

    @signature("(..., n) -> [(..., n), (...)]")
    def encode_and_logabsdet(self, x: Tensor, /) -> tuple[Tensor, Tensor]:
        y = self.encode(x)
        logabsdet = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        return y, logabsdet

    @signature("(..., n) -> [(..., n), (...)]")
    def decode_and_logabsdet(self, y: Tensor, /) -> tuple[Tensor, Tensor]:
        x = self.decode(y)
        logabsdet = torch.zeros(y.shape[:-1], device=y.device, dtype=y.dtype)
        return x, logabsdet
