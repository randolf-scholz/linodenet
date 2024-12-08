r"""Implementation of invertible ResNets."""

__all__ = ["iResNetBlock"]


import warnings
from typing import Final, Optional, Self

import torch
from torch import Tensor, jit, nn


class iResNetBlock(nn.Module):
    r"""A single block of an iResNet.

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | http://proceedings.mlr.press/v97/behrmann19a.html
    """

    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""
    atol: Final[float]
    r"""CONST: Absolute tolerance for fixed point iteration"""
    rtol: Final[float]
    r"""CONST: Relative tolerance for fixed point iteration"""
    is_inverse: Final[bool]
    r"""CONST: Whether to use the inverse or the forward map"""
    converged: Tensor
    r"""BUFFER: Boolean tensor indicating convergence"""

    def __init__(
        self,
        layer: nn.Module,
        *,
        maxiter: int = 100,
        atol: float = 1e-5,
        rtol: float = 1e-4,
        inverse: Optional[Self] = None,
    ) -> None:
        super().__init__()
        self.block = layer
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.register_buffer("converged", torch.tensor(False))

        self.is_inverse = inverse is not None
        if inverse is None:
            cls = type(self)
            self.inverse = cls(
                layer,
                maxiter=maxiter,
                atol=atol,
                rtol=rtol,
                inverse=self,
            )

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n) -> (..., n)``."""
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r"""Compute the forward map with residual connection."""
        if self.is_inverse:
            return self._decode(x)
        return self._encode(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        if self.is_inverse:
            return self._encode(y)
        return self._decode(y)

    @jit.export
    def _encode(self, x: Tensor) -> Tensor:
        return x + self.block(x)

    @jit.export
    def _decode(self, y: Tensor) -> Tensor:
        x = y.clone().detach()
        # m = torch.isnan(y)
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.block(x)
            with torch.no_grad():
                residual = torch.sqrt(torch.nansum((x - x_prev).pow(2)))
                tol = self.atol + self.rtol * torch.sqrt(torch.nansum(x_prev.pow(2)))
                self.converged = residual < tol
                if self.converged:
                    break
        if not self.converged:
            warnings.warn(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual.item()} > {self.atol}.",
                stacklevel=2,
            )
        return x
