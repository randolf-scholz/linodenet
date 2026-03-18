r"""Checks for testing certain vector properties (rank-1 tensors)."""

__all__ = [
    # ABCs & Protocols
    "VectorTest",
    "VectorTestWithArgs",
    # is_* checks
    "is_positive_vector",
    "is_stochastic_vector",
    "is_unit_vector",
]

from collections.abc import Callable
from typing import Concatenate, Protocol

import torch
from torch import Tensor

from linodenet.constants import ATOL, RTOL
from signatures import signature


class VectorTest(Protocol):
    r"""Protocol for testing certain vector properties."""

    @signature("(..., n) -> bool[(...)]")
    def __call__(
        self,
        x: Tensor,
        /,
        *,
        dim: int = -1,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> Tensor: ...


type VectorTestWithArgs = Callable[Concatenate[Tensor, ...], Tensor]


@signature("(..., n) -> bool[(...)]")
def is_unit_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has unit norm."""
    return torch.isclose(
        torch.linalg.vector_norm(x, dim=dim),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_positive_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is strictly positive."""
    return (x > 0).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_stochastic_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor lies on the probability simplex."""
    return is_positive_vector(x, dim=dim) & torch.isclose(
        x.sum(dim=dim),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )
