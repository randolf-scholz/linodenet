"""Invertible Transforms."""

__all__ = [
    "Transform",
    "InvertibleTransform",
]


from abc import abstractmethod

from torch import Tensor
from typing_extensions import Protocol

from linodenet.types import S, T


class Transform(Protocol[T, S]):
    """A protocol for transforms."""

    @abstractmethod
    def forward(self, x: T, /) -> S:
        """Forward pass of the transform."""
        ...

    @abstractmethod
    def log_abs_det_jacobian(self, x: T, /) -> Tensor:
        r"""Compute $\log|\det 𝐃f[x]|$."""
        ...


class InvertibleTransform(Protocol[T, S]):
    """A protocol for invertible transforms."""

    @abstractmethod
    def forward(self, x: T, /) -> S:
        """Forward pass of the transform."""
        ...

    @abstractmethod
    def inverse(self, y: S, /) -> T:
        """Inverse pass of the transform."""
        ...

    @abstractmethod
    def log_abs_det_jacobian(self, x: T, /) -> Tensor:
        r"""Compute $\log|\det 𝐃f[x]|$."""
        ...

    @abstractmethod
    def log_abs_det_jacobian_inverse(self, y: S, /) -> Tensor:
        r"""Compute $\log|\det 𝐃f⁻¹[y]|$.

        .. math:: 𝐃f⁻¹[y] = (𝐃f[f⁻¹[y]])⁻¹  (inverse function theorem)
        """
        ...
