r"""Base classes and protocols for initializations."""

__all__ = ["InitializationFn", "Initialization"]

from collections.abc import Callable
from typing import Concatenate, Optional, Protocol

import torch
from torch import Tensor


class Initialization(Protocol):
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


type InitializationFn = Callable[Concatenate[int | tuple[int, ...], ...], Tensor]
