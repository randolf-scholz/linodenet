r"""Base classes and protocols for initializations."""

__all__ = [
    "InitializationFn",
    "Initialization",
    "resolve_kernel_initialization",
]

from collections.abc import Callable
from typing import Concatenate, Optional, Protocol

import torch
from torch import Tensor, nn

from linodenet.utils import resolve_name

from . import modules


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


def resolve_kernel_initialization(
    input_size: int,
    kernel_initialization: str | Tensor | nn.Module,
    /,
) -> nn.Module:
    match kernel_initialization:
        case nn.Module() as initialization:
            return initialization

        case Tensor() as tensor:
            if tensor.shape != (input_size, input_size):
                raise ValueError(
                    f"Kernel has bad shape! {tensor.shape} but should be"
                    f" {(input_size, input_size)}"
                )
            return modules.Constant(tensor)

        case str(key):
            assert __package__ is not None
            pkg = __import__(__package__)
            initialization_cls = resolve_name(pkg.INITIALIZATIONS, key)

            try:
                initialization = initialization_cls(input_size)
            except Exception as exc:
                exc.add_note(
                    f"failed to initialize kernel_initialization {initialization_cls}"
                )
                raise

            assert isinstance(initialization, nn.Module)
            return initialization

        case _:
            raise TypeError(
                "kernel_initialization must be a string, tensor, or nn.Module, "
                f"got {type(kernel_initialization)!r}."
            )
