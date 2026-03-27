r"""Base classes and protocols for initializations."""

__all__ = ["InitializationFn", "Initialization"]

from typing import Any, Optional, Protocol

import torch
from torch import Tensor


class InitializationFn(Protocol):
    r"""Protocol for shape-bound initialization samplers."""

    def __call__(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *args: Any,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""Draw samples with batch shape `size`."""
        ...


class Initialization(InitializationFn, Protocol):
    r"""Protocol for shape-bound initialization samplers."""

    def __call__(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        r"""Draw samples with batch shape `size`."""
        ...
