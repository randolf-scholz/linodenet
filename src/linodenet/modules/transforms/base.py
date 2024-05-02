r"""Transforms.

Note:
    torch.distributions.Transform has some differences:
    - It is not a protocol.
    - `log_abs_det_jacobian` always requires 2 arguments, $x$ and $y$.
      The rationale is that for certain transforms, the Jacobian is
      much faster to compute if the output is known.
      For example: if $f(x) = xᵃ$, then $\log|\det 𝐃f[x]| = \log |a⋅y/x|$.
      is more efficient than $\log |a⋅xᵃ⁻¹|$.
      However, for many transforms, this is not true and knowing $y$ is not that helpful.
    - Instead, it makes more sense to have 2 methods: `log_abs_det_jacobian(x)`
      and `value_and_log_abs_det_jacobian(x) -> tuple[Tensor, Tensor]`,
      similar to jax's `value_and_grad`.
      Alternatively, one can store `y` in a buffer and reuse it if needed, i.e.
      methods that need `y` can call:

      >>> def log_abs_det_jacobian(self, x: Tensor, /, y: None | Tensor = None) -> Tensor:
      >>>     if y is None:
      >>>         if id(x) == id(self._last_x):
      >>>             y = self._last_y
      >>>         else:
      >>>             y = self.forward(x)
"""

__all__ = [
    # ABCs & Protocols
    "Transform",
    "InvertibleTransform",
]


from abc import abstractmethod

from torch import Tensor
from typing_extensions import Protocol

from linodenet.types import S, T


class Transform(Protocol[T, S]):
    r"""A protocol for transforms."""

    @abstractmethod
    def forward(self, x: T, /) -> S:
        r"""Forward pass of the transform."""
        ...

    @abstractmethod
    def log_abs_det_jacobian(self, x: T, /) -> Tensor:
        r"""Compute $\log|\det 𝐃f[x]|$."""
        ...


class InvertibleTransform(Protocol[T, S]):
    r"""A protocol for invertible transforms."""

    @abstractmethod
    def forward(self, x: T, /) -> S:
        r"""Forward pass of the transform."""
        ...

    @abstractmethod
    def inverse(self, y: S, /) -> T:
        r"""Inverse pass of the transform."""
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
