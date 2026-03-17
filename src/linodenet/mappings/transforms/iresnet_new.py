r"""Implementation of invertible ResNets."""

__all__ = ["iResNetBlock"]


import warnings
from typing import Final, Optional, Self

import torch
from torch import Tensor, nn


class iResNetBlock(nn.Module):
    r"""A single block of an iResNet.

    References:
        - | Invertible Residual Networks
          | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
          | International Conference on Machine Learning 2019
          | http://proceedings.mlr.press/v97/behrmann19a.html
        - https://github.com/jhjacobsen/invertible-resnet
    """

    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""
    atol: Final[float]
    r"""CONST: Absolute tolerance for fixed point iteration"""
    rtol: Final[float]
    r"""CONST: Relative tolerance for fixed point iteration"""
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

    def encode(self, x: Tensor) -> Tensor:
        r"""Computes $y = x + f(x)$."""
        return x + self.block(x)

    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        .. math:: x = y - f(x)

        Note that in this case the gradient can be computed through implicit differentiation:

        .. math::
            && z &= f(z, x, θ)
            \\ &⟹& ∂z/∂θ &= df/dθ = ∂f/∂z ∂z/∂θ + ∂f/∂x ∂x/∂θ + ∂f/∂θ
            \\ &⟹& (I-∂f/∂z) ∂z/∂θ &= ∂f/∂x ∂x/∂θ + ∂f/∂θ
            \\ &⟹& ∂z/∂θ &= (I-∂f/∂z)⁻¹ (∂f/∂x ∂x/∂θ + ∂f/∂θ)

        Moreover, the jacobian-vector product can be computed as:

        .. math:: vᵀ(∂z/∂θ) = ... = (∂f/∂x ∂x/∂θ + ∂f/∂θ)ᵀ(I-∂f/∂z)⁻ᵀv

        References:
            https://implicit-layers-tutorial.org/
        """
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
