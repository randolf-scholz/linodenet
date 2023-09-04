r"""Utility functions."""

__all__ = [
    # Constants
    # Functions
    "autojit",
    "assert_issubclass",
    "deep_dict_update",
    "deep_keyval_update",
    "initialize_from_config",
    "is_dunder",
    "is_private",
    "pad",
    "register_cache",
    # Classes
    "timer",
    "reset_caches",
]

import gc
import logging
import sys
import warnings
from collections.abc import Callable, Mapping
from contextlib import ContextDecorator
from copy import deepcopy
from functools import wraps
from importlib import import_module
from time import perf_counter_ns
from types import MethodType, ModuleType, TracebackType
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor, jit, nn
from typing_extensions import Self

from linodenet.config import CONFIG
from linodenet.types import module_var, type_var

__logger__ = logging.getLogger(__name__)


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


def deep_dict_update(d: dict, new: Mapping, /, *, inplace: bool = False) -> dict:
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


def deep_keyval_update(d: dict, /, **new_kv: Any) -> dict:
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


def autojit(base_class: module_var) -> module_var:
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
        def __new__(cls, *args: Any, **kwargs: Any) -> module_var:
            # Note: If __new__() does not return an instance of cls,
            # then the new instance's __init__() method will not be invoked.
            instance: module_var = base_class(*args, **kwargs)  # type: ignore[assignment]

            if CONFIG.autojit:
                scripted: module_var = jit.script(instance)  # pyright: ignore
                return scripted
            return instance

    assert issubclass(WrappedClass, base_class)  # pyright: ignore
    return WrappedClass  # type: ignore[return-value]


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


def register_cache(self: nn.Module, name: str, func: Callable[[], Tensor], /) -> None:
    """Register a cache to a module.

    - creates a buffer that stores the result of func().
    - creates a "recompute_{name}" that updates the buffer.
    - creates a "recompute_all" that recomputes all buffers in order.
    - creates a "cached_tensors" that returns a dictionary of all buffers.
    """
    if hasattr(self, name):
        raise AttributeError(f"{self} already has attribute {name}")

    if not hasattr(self, "cached_tensors"):
        cached_tensors: dict[str, Tensor] = {}
        self.cached_tensors = cached_tensors  # type: ignore[assignment]
        self.__annotations__["cached_tensors"] = dict[str, Tensor]

    cached_tensors = self.cached_tensors  # type: ignore[assignment]

    # register the buffer
    self.register_buffer("name", func())
    self.__annotations__[name] = Tensor
    cached_tensors[name] = getattr(self, name)

    # register the recompute function
    def recompute(obj: nn.Module) -> None:
        setattr(obj, name, func())

    setattr(self, f"recompute_{name}", MethodType(recompute, self))

    # register the recompute all function
    def recompute_all(obj: nn.Module) -> None:
        for key in obj.cached_tensors:  # type: ignore[union-attr]
            f = getattr(obj, f"recompute_{key}")
            f()

    self.recompute_all = MethodType(recompute_all, self)  # type: ignore[assignment]


class reset_caches(ContextDecorator):
    """Context manager for resetting caches."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__module__}/{__qualname__}")  # type: ignore[name-defined]

    def __init__(
        self,
        module: nn.Module,
        *,
        on_enter: bool = True,
        on_exit: bool = True,
    ) -> None:
        self.module = module
        self.on_enter = on_enter
        self.on_exit = on_exit

        num_caches = 0
        for layer in self.module.modules():
            if hasattr(layer, "cached_tensors"):
                num_caches += 1
        self.LOGGER.info("Found %d caches.", num_caches)
        if num_caches == 0:
            warnings.warn("No caches found.", RuntimeWarning, stacklevel=2)

    def recompute_all(self) -> None:
        """Recompute all cached tensors."""
        for layer in self.module.modules():
            f = getattr(layer, "recompute_all", lambda: None)
            f()

    def __enter__(self) -> Self:
        if self.on_enter:
            self.LOGGER.info("Resetting caches.")
            self.recompute_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if self.on_exit:
            self.LOGGER.info("Resetting caches.")
            self.module.recompute_all()  # type: ignore[operator]
        return False


def assert_issubclass(proto: type) -> Callable[[type_var], type_var]:
    """Assert that an object satisfies a protocol."""

    def decorator(cls: type_var) -> type_var:
        """Assert that a class satisfies a protocol."""
        assert issubclass(cls, proto), f"{cls} is not a {proto}"
        return cls  # type: ignore[return-value]

    return decorator
