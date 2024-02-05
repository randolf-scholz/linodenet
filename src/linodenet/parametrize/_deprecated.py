"""Deprecated parametrization utilities."""

__all__ = [
    "register_cache",
    "reset_caches",
]

import logging
import warnings
from collections.abc import Callable
from contextlib import ContextDecorator
from types import MethodType, TracebackType

from torch import Tensor, nn
from typing_extensions import ClassVar, Literal, Self


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

    cached_tensors = self.cached_tensors

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
        for key in obj.cached_tensors:
            f = getattr(obj, f"recompute_{key}")
            f()

    self.recompute_all = MethodType(recompute_all, self)  # type: ignore[assignment]


class reset_caches(ContextDecorator):
    """Context manager for resetting caches."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

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
            self.module.recompute_all()
        return False
