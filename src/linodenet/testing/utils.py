__all__ = [
    "flatten_nested_tensor",
    "get_device",
    "get_grads",
    "get_norm",
    "iter_tensors_requiring_grad",
    "get_shapes",
    "iter_parameters",
    "iter_tensors",
    "make_tensors_parameters",
    "to_device",
    "zero_grad",
]

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from itertools import chain
from typing import Any, overload

import torch
from torch import Tensor, nn
from torch.nn import Module

from linodenet.types import DeviceArg, Nested, Scalar

type Tree = Nested[Tensor | Scalar]
type Func = Callable[..., Nested[Tensor]]


def get_device(x: object, /) -> DeviceArg:
    r"""Return the device of the model / parameters."""
    match x:
        case Module() as model:
            return next(t.device for t in model.parameters())
        case Tensor() as tensor:
            return tensor.device
        case Mapping() as mapping:
            return get_device(next(iter(mapping.values())))
        case Iterable() as iterable:
            return get_device(next(iter(iterable)))
        case _:
            return None


@overload
def to_device[M: Module](x: M, /, *, device: DeviceArg = ...) -> M: ...
@overload
def to_device[T: Tensor](x: T, /, *, device: DeviceArg = ...) -> T: ...
@overload
def to_device[S: Scalar](x: S, /, *, device: DeviceArg = ...) -> S: ...
@overload
def to_device[T](
    map_: Mapping[str, T], /, *, device: DeviceArg = ...
) -> dict[str, T]: ...
@overload
def to_device[T](set_: AbstractSet[T], /, *, device: DeviceArg = ...) -> set[T]: ...
@overload
def to_device[*Ts](tup: tuple[*Ts], /, *, device: DeviceArg = ...) -> tuple[*Ts]: ...
@overload
def to_device[T](seq: list[T], /, *, device: DeviceArg = ...) -> list[T]: ...
@overload
def to_device[T](x: T, /, *, device: DeviceArg = ...) -> T: ...
def to_device(x: Any, /, *, device: DeviceArg = "cpu") -> Any:
    r"""Move a nested tensor to a device."""
    match x:
        case Tensor() as tensor:
            target_device = None if device is None else torch.device(device)
            return tensor.to(device=target_device)
        case Module() as module:
            target_device = None if device is None else torch.device(device)
            return module.to(device=target_device)
        case (None | bool() | int() | float() | str()) as scalar:  # Scalar
            # FIXME: https://github.com/python/cpython/issues/106246
            return scalar
        case tuple(tup):
            return tuple(to_device(item, device=device) for item in tup)
        case AbstractSet() as set_like:
            return {to_device(item, device=device) for item in set_like}
        case Sequence() as seq:
            return [to_device(item, device=device) for item in seq]
        case Mapping() as mapping:
            return {key: to_device(val, device=device) for key, val in mapping.items()}
        case _:
            return x


def iter_tensors(x: Module | Tree, /) -> Iterator[Tensor]:
    r"""Yields the tensors of a model."""
    match x:
        case Tensor() as tensor:
            yield tensor
        case Module() as module:
            yield from module.parameters()
        case None | bool() | int() | float() | str():  # Scalar
            # FIXME: https://github.com/python/cpython/issues/106246
            pass
        case Mapping() as mapping:
            yield from chain.from_iterable(iter_tensors(v) for v in mapping.values())
        case Iterable() as iterable:
            yield from chain.from_iterable(iter_tensors(item) for item in iterable)
        case _:
            raise TypeError(f"Unsupported input type {type(x)!r}")


def iter_tensors_requiring_grad(x: Module | Tree, /) -> Iterator[Tensor]:
    r"""Return the parameters of the model / parameters."""
    for w in iter_tensors(x):
        if w.requires_grad:
            yield w


def iter_parameters(x: Module | Tree, /) -> Iterator[nn.Parameter]:
    r"""Yields the parameters of the model."""
    for w in iter_tensors(x):
        if isinstance(w, nn.Parameter):
            yield w


def zero_grad(x: Module | Tree | Func, /) -> None:
    r"""Sets gradients of the model / parameters to None."""
    if isinstance(x, Module):
        x.zero_grad(set_to_none=True)
        return

    if callable(x):  # stateless function
        return

    for w in iter_tensors(x):
        if w.requires_grad:
            w.grad = None


def flatten_nested_tensor(x: Module | Tree, /) -> Tensor:
    r"""Flattens element of general Hilbert space, skips over scalars."""
    return torch.cat([x.flatten() for x in iter_tensors(x)])


def get_shapes(x: Module | Tree, /) -> list[tuple[int, ...]]:
    r"""Return the shapes of the tensors."""
    return [item.shape for item in iter_tensors(x)]


def get_grads(x: Module | Tree, /) -> list[Tensor]:
    r"""Return a cloned detached copy of the gradients."""
    return [
        w.grad.clone().detach()
        for w in iter_tensors(x)
        if w.requires_grad and w.grad is not None
    ]


def get_norm(x: Nested[Tensor], /, *, normalize: bool = True) -> Tensor:
    r"""Compute the (normalized) 2-norm of a tensor."""
    flattened = flatten_nested_tensor(x)
    if normalize:
        return torch.sqrt(torch.mean(flattened**2))
    return torch.sqrt(torch.sum(flattened**2))


@overload
def make_tensors_parameters(x: Tensor, /) -> nn.Parameter: ...
@overload
def make_tensors_parameters[S: Scalar](x: S, /) -> S: ...
@overload
def make_tensors_parameters[T](x: Mapping[str, T], /) -> dict[str, T]: ...
@overload
def make_tensors_parameters[T](x: AbstractSet[T], /) -> set[T]: ...
@overload
def make_tensors_parameters[*Ts](x: tuple[*Ts], /) -> tuple[*Ts]: ...
@overload
def make_tensors_parameters[T](x: list[T], /) -> list[T]: ...
def make_tensors_parameters(arg: Any, /) -> Any:
    r"""Make tensors parameters."""
    # FIXME: https://github.com/python/cpython/issues/106246. Use match-case when fixed.
    match arg:
        case Tensor() as x:
            return nn.Parameter(x) if not isinstance(x, nn.Parameter) else x
        case scalar if isinstance(scalar, Scalar.__value__):
            return scalar
        case Mapping() as mapping:
            return {key: make_tensors_parameters(val) for key, val in mapping.items()}
        case tuple(tup):
            return tuple(make_tensors_parameters(item) for item in tup)
        case AbstractSet() as set_like:
            return {make_tensors_parameters(item) for item in set_like}
        case Sequence() as seq:
            return [make_tensors_parameters(item) for item in seq]
        case other:
            raise TypeError(f"Unsupported input type {type(other)!r}")
