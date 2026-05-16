r"""Checks for testing certain tensor properties."""

__all__ = [
    # ABCs & Protocols
    "TensorTest",
    "TensorTestWithArgs",
    # is_* checks
    "is_boolean_tensor",
    "is_complex_tensor",
    "is_nonzero_tensor",
    "is_one_tensor",
    "is_real_tensor",
    "is_sparse_tensor",
    "is_zero_tensor",
    "is_tensor",
]

from collections.abc import Callable
from typing import Concatenate, Protocol

import torch
from torch import Tensor

from linodenet.constants import ATOL, RTOL
from signatures import signature


class TensorTest(Protocol):
    r"""Protocol for testing certain tensor properties."""

    @signature("(*shape) -> bool[batch]")
    def __call__(
        self,
        x: Tensor,
        /,
        *,
        shape: tuple[int, ...] | None = None,
        rtol: float = RTOL,
        atol: float = ATOL,
    ) -> Tensor: ...


type TensorTestWithArgs = Callable[Concatenate[Tensor, ...], Tensor]


def _tensor_batch_shape(x: Tensor, shape: tuple[int, ...] | None, /) -> tuple[int, ...]:
    r"""Return the batch shape after removing the trailing tensor axes."""
    if shape is None:
        return ()
    if len(shape) == 0:
        return x.shape
    if x.ndim < len(shape):
        return ()
    return x.shape[: -len(shape)]


def _matches_shape(x: Tensor, shape: tuple[int, ...] | None, /) -> bool:
    r"""Check whether the trailing dimensions match the requested tensor shape."""
    if shape is None:
        return True
    if len(shape) == 0:
        return True
    if x.ndim < len(shape):
        return False
    return x.shape[-len(shape) :] == shape


def _trailing_dims(x: Tensor, shape: tuple[int, ...], /) -> tuple[int, ...]:
    r"""Return the trailing dimensions corresponding to the requested tensor shape."""
    ndim = len(shape)
    return tuple(range(x.ndim - ndim, x.ndim))


@signature("(*shape) -> bool[batch]")
def is_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor matches the requested trailing shape."""
    batch_shape = _tensor_batch_shape(x, shape)
    return x.new_full(batch_shape, _matches_shape(x, shape), dtype=torch.bool)


@signature("(*shape) -> bool[batch]")
def is_real_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor has a real dtype."""
    return is_tensor(x, shape=shape) & x.new_full(
        _tensor_batch_shape(x, shape),
        not x.is_complex(),
        dtype=torch.bool,
    )


@signature("(*shape) -> bool[batch]")
def is_complex_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor can be interpreted as complex-valued."""
    return is_tensor(x, shape=shape)


@signature("(*shape) -> bool[batch]")
def is_boolean_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros and ones."""
    shape_ok = is_tensor(x, shape=shape)
    if shape is None:
        return shape_ok & ((x == 0) | (x == 1)).all()
    if not _matches_shape(x, shape):
        return shape_ok
    dims = _trailing_dims(x, shape)
    if not dims:
        return shape_ok & ((x == 0) | (x == 1))
    return shape_ok & ((x == 0) | (x == 1)).all(dim=dims)


@signature("(*shape) -> bool[batch]")
def is_zero_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only zeros."""
    shape_ok = is_tensor(x, shape=shape)
    if shape is None:
        return shape_ok & (x == 0).all()
    if not _matches_shape(x, shape):
        return shape_ok
    dims = _trailing_dims(x, shape)
    if not dims:
        return shape_ok & (x == 0)
    return shape_ok & (x == 0).all(dim=dims)


@signature("(*shape) -> bool[batch]")
def is_one_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains only ones."""
    shape_ok = is_tensor(x, shape=shape)
    if shape is None:
        return shape_ok & (x == 1).all()
    if not _matches_shape(x, shape):
        return shape_ok
    dims = _trailing_dims(x, shape)
    if not dims:
        return shape_ok & (x == 1)
    return shape_ok & (x == 1).all(dim=dims)


@signature("(*shape) -> bool[batch]")
def is_nonzero_tensor(
    x: Tensor,
    /,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> Tensor:
    r"""Check whether the given tensor is not identically zero."""
    shape_ok = is_tensor(x, shape=shape)
    nonzero = ~torch.isclose(
        x,
        torch.zeros((), dtype=x.dtype, device=x.device),
        rtol=rtol,
        atol=atol,
    )
    if shape is None:
        return shape_ok & nonzero.any()
    if not _matches_shape(x, shape):
        return shape_ok
    dims = _trailing_dims(x, shape)
    if not dims:
        return shape_ok & nonzero
    return shape_ok & nonzero.any(dim=dims)


@signature("(*shape) -> bool[batch]")
def is_sparse_tensor(
    x: Tensor,
    /,
    sparsity: float | None = None,
    *,
    shape: tuple[int, ...] | None = None,
    rtol: float = 0.0,  # noqa: ARG001
    atol: float = 0.0,  # noqa: ARG001
) -> Tensor:
    r"""Check whether the given tensor contains sufficiently many exact zeros."""
    shape_ok = is_tensor(x, shape=shape)
    zero_fraction = (x == 0).to(dtype=torch.float32)
    if shape is None:
        sparse_ok = zero_fraction.mean() > 0.0
    else:
        if not _matches_shape(x, shape):
            return shape_ok
        dims = _trailing_dims(x, shape)
        sparse_ok = zero_fraction if not dims else zero_fraction.mean(dim=dims)
        sparse_ok = sparse_ok > 0.0 if sparsity is None else sparse_ok >= sparsity
    return shape_ok & sparse_ok
