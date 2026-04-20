r"""Checks for testing certain vector properties (rank-1 tensors)."""

__all__ = [
    # ABCs & Protocols
    "VectorTest",
    "VectorTestWithArgs",
    # is_* checks
    "is_boolean_vector",
    "is_complex_vector",
    "is_discrete_vector",
    "is_one_hot_vector",
    "is_negative_vector",
    "is_nonnegative_vector",
    "is_nonpositive_vector",
    "is_nonzero_vector",
    "is_positive_vector",
    "is_one_vector",
    "is_real_vector",
    "is_sparse_vector",
    "is_stochastic_vector",
    "is_standardized_vector",
    "is_unit_ball_vector",
    "is_unit_cube_vector",
    "is_unit_l1_ball_vector",
    "is_unit_l1_sphere_vector",
    "is_unit_vector",
    "is_zero_vector",
    "is_zero_mean_vector",
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


def _vector_batch_shape(x: Tensor, dim: int, /) -> tuple[int, ...]:
    r"""Return the batch shape after removing the vector dimension."""
    dim %= x.ndim
    return tuple(size for axis, size in enumerate(x.shape) if axis != dim)


@signature("(..., n) -> bool[(...)]")
def is_real_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor has a real dtype."""
    return x.new_full(_vector_batch_shape(x, dim), not x.is_complex(), dtype=torch.bool)


@signature("(..., n) -> bool[(...)]")
def is_discrete_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor has integer or boolean dtype."""
    return x.new_full(
        _vector_batch_shape(x, dim),
        x.dtype
        in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        },
        dtype=torch.bool,
    )


@signature("(..., n) -> bool[(...)]")
def is_complex_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor can be interpreted as complex-valued."""
    return x.new_full(_vector_batch_shape(x, dim), True, dtype=torch.bool)


@signature("(..., n) -> bool[(...)]")
def is_boolean_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros and ones."""
    return ((x == 0) | (x == 1)).all(dim=dim)


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
def is_negative_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is strictly negative."""
    return (x < 0).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_nonnegative_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is entrywise nonnegative."""
    return (x >= 0).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_nonpositive_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor is entrywise nonpositive."""
    return (x <= 0).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_nonzero_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is not identically zero."""
    return ~torch.isclose(
        x,
        torch.zeros((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    ).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_zero_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros."""
    return (x == 0).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_one_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only ones."""
    return (x == 1).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_one_hot_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has exactly one 1 and zeros elsewhere."""
    is_one = torch.isclose(
        x,
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )
    is_zero = torch.isclose(
        x,
        torch.zeros((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )
    return (is_one.sum(dim=dim) == 1) & (is_zero | is_one).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_sparse_vector(
    x: Tensor,
    /,
    sparsity: float | None = None,
    *,
    dim: int = -1,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains sufficiently many exact zeros."""
    zero_fraction = (x == 0).to(dtype=torch.float32).mean(dim=dim)
    if sparsity is None:
        return zero_fraction > 0.0
    return zero_fraction >= sparsity


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
    return is_nonnegative_vector(x, dim=dim) & torch.isclose(
        x.sum(dim=dim),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_zero_mean_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has zero mean."""
    return torch.isclose(
        x.mean(dim=dim),
        torch.zeros((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_standardized_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has zero mean and unit variance."""
    return is_zero_mean_vector(x, dim=dim, rtol=rtol, atol=atol) & torch.isclose(
        x.var(dim=dim, correction=0),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_unit_ball_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has Euclidean norm at most one."""
    norm = torch.linalg.vector_norm(x, dim=dim)
    return (norm <= 1) | torch.isclose(
        norm,
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_unit_cube_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor lies in the ℓ∞ unit ball."""
    abs_x = x.abs()
    return (
        (abs_x <= 1)
        | torch.isclose(
            abs_x,
            torch.ones((), dtype=abs_x.dtype, device=abs_x.device),
            rtol=rtol,
            atol=atol,
        )
    ).all(dim=dim)


@signature("(..., n) -> bool[(...)]")
def is_unit_l1_ball_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has ℓ¹ norm at most one."""
    norm = torch.linalg.vector_norm(x, ord=1, dim=dim)
    return (norm <= 1) | torch.isclose(
        norm,
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )


@signature("(..., n) -> bool[(...)]")
def is_unit_l1_sphere_vector(
    x: Tensor,
    /,
    *,
    dim: int = -1,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor has ℓ¹ norm equal to one."""
    return torch.isclose(
        torch.linalg.vector_norm(x, ord=1, dim=dim),
        torch.ones((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )
