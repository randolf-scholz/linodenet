r"""Unit lower-triangular linear flow."""

__all__ = ["TriangularTransform"]

from collections.abc import Sequence
from typing import Final

import torch
from torch import Tensor, nn

from signatures import signature

from ..base import TransformBase


class TriangularTransform(TransformBase):
    r"""An invertible linear layer with unit lower-triangular Jacobian.

    The transformation is parameterized as

    .. math:: y = P⁻¹(𝕀ₙ + L)Px

    where $L$ is strictly lower triangular. This makes the weight matrix
    unit lower triangular in the permuted basis, hence always invertible
    with determinant 1.
    """

    input_size: Final[int]
    r"""CONST: Input and output dimensionality."""

    lower: Tensor
    r"""PARAM: Unconstrained matrix whose strictly lower part defines the flow."""

    perm: Tensor
    r"""BUFFER: Permutation applied before the triangular map."""

    invperm: Tensor
    r"""BUFFER: Inverse permutation applied after the triangular map."""

    @property
    def config(self) -> dict[str, int | tuple[int, ...]]:
        return {
            "input_size": self.input_size,
            "permutation": tuple(int(x) for x in self.perm.tolist()),
        }

    def __init__(
        self,
        input_size: int,
        permutation: None | Sequence[int] | Tensor = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.lower = nn.Parameter(torch.zeros(input_size, input_size))
        perm = self._validate_permutation(input_size, permutation)
        invperm = torch.empty_like(perm)
        invperm[perm] = torch.arange(input_size, device=perm.device, dtype=perm.dtype)
        self.register_buffer("perm", perm, persistent=True)
        self.register_buffer("invperm", invperm, persistent=True)

    @staticmethod
    def _validate_permutation(
        input_size: int,
        permutation: None | Sequence[int] | Tensor,
        /,
    ) -> Tensor:
        if permutation is None:
            return torch.arange(input_size, dtype=torch.int64)

        perm = torch.as_tensor(permutation)
        assert perm.shape == (input_size,)
        assert set(perm.flatten().tolist()) == set(range(input_size))
        return perm

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
        r"""Compute :math:`y = P⁻¹(𝕀ₙ + L)Px`."""
        x = x[..., self.perm]
        lower = self.lower.tril(diagonal=-1)
        update = torch.einsum("mn, ...n -> ...m", lower, x)
        y = x + update
        return y[..., self.invperm]

    @signature("(..., n) -> (..., n)")
    def decode(self, y: Tensor, /) -> Tensor:
        r"""Solve :math:`P⁻¹(𝕀ₙ + L)Px = y` for :math:`x`."""
        y = y[..., self.perm]
        x = torch.linalg.solve_triangular(
            self.weight,
            y[..., None],
            upper=False,
            unitriangular=True,
        ).squeeze(-1)
        return x[..., self.invperm]

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
