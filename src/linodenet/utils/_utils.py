r"""Utility functions."""

__all__ = [
    # Constants
    # Functions
    "autojit",
    "deep_dict_update",
    "deep_keyval_update",
    "flatten_nested_tensor",
    "initialize_from_config",
    "is_dunder",
    "is_private",
    "pad",
    # Classes
    "timer",
]

import gc
import logging
import sys
from collections.abc import Iterable, Mapping
from contextlib import ContextDecorator
from copy import deepcopy
from functools import wraps
from importlib import import_module
from time import perf_counter_ns
from types import ModuleType, TracebackType
from typing import Any, ClassVar, Literal, TypeVar

import torch
from torch import Tensor, jit, nn
from typing_extensions import Self

from linodenet.config import CONFIG

__logger__ = logging.getLogger(__name__)

R = TypeVar("R")
r"""Type hint return value."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""


@jit.script
def pad(
    x: Tensor,
    value: float,
    pad_width: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = pad_width
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)


def deep_dict_update(d: dict, new: Mapping, inplace: bool = False) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    References:
        - https://stackoverflow.com/a/30655448/9318372
    """
    if not inplace:
        d = deepcopy(d)

    for key, value in new.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            d[key] = new[key]
    return d


def deep_keyval_update(d: dict, **new_kv: Any) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    References:
        - https://stackoverflow.com/a/30655448/9318372
    """
    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_keyval_update(d.get(key, {}), **new_kv)
        elif key in new_kv:
            d[key] = new_kv[key]
    return d


def autojit(base_class: type[nnModuleType]) -> type[nnModuleType]:
    r"""Class decorator that enables automatic jitting of nn.Modules upon instantiation.

    Makes it so that

    .. code-block:: python

        class MyModule:
            ...


        model = jit.script(MyModule())

    and

    .. code-block:: python

        class MyModule:
            ...


        model = MyModule()

    are (roughly?) equivalent
    """
    assert issubclass(base_class, nn.Module)

    @wraps(base_class, updated=())
    class WrappedClass(base_class):  # type: ignore[misc, valid-type]
        r"""A simple Wrapper."""

        # noinspection PyArgumentList
        def __new__(cls, *args: Any, **kwargs: Any) -> nnModuleType:  # type: ignore[misc]
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: nnModuleType = base_class(*args, **kwargs)

            if CONFIG.autojit:
                scripted: nnModuleType = jit.script(instance)
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)
    return WrappedClass


def flatten_nested_tensor(inputs: Tensor | Iterable[Tensor]) -> Tensor:
    r"""Flattens element of general Hilbert space."""
    if isinstance(inputs, Tensor):
        return torch.flatten(inputs)
    if isinstance(inputs, Iterable):
        return torch.cat([flatten_nested_tensor(x) for x in inputs])
    raise ValueError(f"{inputs=} not understood")


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize a class from a dictionary."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module: ModuleType = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    obj = cls(**opts)
    assert isinstance(obj, nn.Module)
    return obj


def is_dunder(s: str, /) -> bool:
    r"""Check if name is a dunder method."""
    return s.isidentifier() and s.startswith("__") and s.endswith("__")


def is_private(s: str, /) -> bool:
    r"""Check if name is a private method."""
    return s.isidentifier() and s.startswith("_") and not s.endswith("__")


class timer(ContextDecorator):
    """Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__module__}/{__qualname__}")  # type: ignore[name-defined]

    start_time: int
    """Start time of the timer."""
    end_time: int
    """End time of the timer."""
    elapsed: float
    """Elapsed time of the timer in seconds."""

    def __enter__(self) -> Self:
        self.LOGGER.info("Flushing pending writes.")
        sys.stdout.flush()
        sys.stderr.flush()
        self.LOGGER.info("Disabling garbage collection.")
        gc.collect()
        gc.disable()
        self.LOGGER.info("Starting timer.")
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.end_time = perf_counter_ns()
        self.elapsed = (self.end_time - self.start_time) / 10**9
        self.LOGGER.info("Stopped timer.")
        gc.enable()
        self.LOGGER.info("Re-Enabled garbage collection.")
        return False
