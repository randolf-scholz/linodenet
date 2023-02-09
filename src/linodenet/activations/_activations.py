r"""Activations of the LinODE-Net package."""

__all__ = [
    "Activation",
    "ActivationABC",
]

from abc import abstractmethod
from typing import Callable, Concatenate, ParamSpec, Protocol, runtime_checkable

from torch import Tensor, nn

P = ParamSpec("P")


@runtime_checkable
class Activation(Protocol[P]):
    """Protocol for Activation Components."""

    __call__: Callable[Concatenate[Tensor, P], Tensor]
    # def __call__(self, x: Tensor, *args: P.args, **kwargs: P.kwargs) -> Tensor:
    #     """Forward pass of the activation."""


class ActivationABC(nn.Module):
    """Abstract Base Class for Activation components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the activation.

        .. Signature: ``... -> ...``.

        Args:
            x: The input tensor to be activated.

        Returns:
            y: The activated tensor.
        """
