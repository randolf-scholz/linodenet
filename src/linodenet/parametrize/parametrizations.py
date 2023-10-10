"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = [
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    "ReZero",
    # linodenet.projections
    "Diagonal",
    "SkewSymmetric",
    "SpectralNormalization",
    "Symmetric",
    "Traceless",
]

from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.lib import singular_triplet
from linodenet.parametrize.base import ParametrizationBase, ParametrizationDict
from linodenet.projections import diagonal, skew_symmetric, symmetric, traceless


class SpectralNormalization(ParametrizationDict):
    """Spectral normalization."""

    # constants
    GAMMA: Tensor
    ONE: Tensor
    maxiter: Final[Optional[int]]

    # cached
    u: Tensor
    v: Tensor
    sigma: Tensor
    weight: Tensor

    # Parameters
    original_weight: Tensor

    def __init__(
        self, weight: Tensor, /, gamma: float = 1.0, maxiter: Optional[int] = None
    ) -> None:
        super().__init__()
        assert isinstance(weight, nn.Parameter), "weight must be a parameter"

        self.maxiter = maxiter

        assert len(weight.shape) == 2
        m, n = weight.shape

        options: Any = {
            "dtype": weight.dtype,
            "device": weight.device,
            "layout": weight.layout,
        }

        # parametrized and cached
        self.register_parametrized_tensor("weight", weight)
        self.register_cached_tensor("u", torch.randn(m, **options))
        self.register_cached_tensor("v", torch.randn(n, **options))
        self.register_cached_tensor("sigma", torch.ones((), **options))

        # constants
        self.register_buffer("ONE", torch.ones((), **options))
        self.register_buffer("GAMMA", torch.full_like(self.ONE, gamma, **options))

    def parametrization(self) -> dict[str, Tensor]:
        """Perform spectral normalization w ↦ w/‖w‖₂."""
        # IMPORTANT: Use the original weight, not the cached weight!
        # For auxiliary tensors, use the cached tensors.
        sigma, u, v = singular_triplet(
            self.original_weight,
            u0=self.u,
            v0=self.v,
            maxiter=self.maxiter,
        )
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)
        weight = gamma * self.original_weight

        return {
            "weight": weight,
            "sigma": sigma,
            "u": u,
            "v": v,
        }


class ReZero(ParametrizationBase):
    """ReZero."""

    scalar: Tensor

    def __init__(
        self,
        tensor: Tensor,
        /,
        *,
        scalar: Optional[Tensor] = None,
        learnable: bool = True,
    ) -> None:
        super().__init__(tensor)
        self.learnable = learnable

        if scalar is None:
            initial_value = torch.tensor(0.0)
        else:
            initial_value = scalar

        if self.learnable:
            self.scalar = nn.Parameter(initial_value)
        else:
            self.scalar = initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x


class CayleyMap(ParametrizationBase):
    """Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    def __init__(self, tensor: Tensor) -> None:
        assert len(tensor.shape) == 2 and tensor.shape[0] == tensor.shape[1]
        n = tensor.shape[0]
        super().__init__(tensor)
        self.register_buffer("Id", torch.eye(n))

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id + x, self.Id - x).solution

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id - y, self.Id + y).solution


class MatrixExponential(ParametrizationBase):
    """Parametrize a matrix via matrix exponential."""

    def forward(self, X):
        return torch.matrix_exp(X)

    def right_inverse(self, Y):
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix logarithm, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class GramMatrix(ParametrizationBase):
    """Parametrize a matrix via gram matrix ($XᵀX$)."""

    def forward(self, X):
        return X.T @ X

    def right_inverse(self, Y):
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix square root, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


# region linodenet.projections ---------------------------------------------------------
class Symmetric(ParametrizationBase):
    """Parametrize a matrix to be symmetric."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            As opposed to the method in the documentation, we do not use the triu
            operator, so this is simply the identity, since `skew_symmetric` is
            already a projection.
        """
        return y


class SkewSymmetric(ParametrizationBase):
    """Parametrize a matrix to be skew-symmetric."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return skew_symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            As opposed to the method in the documentation, we do not use the triu
            operator, so this is simply the identity, since `skew_symmetric` is
            already a projection.
        """
        return y


class Diagonal(ParametrizationBase):
    """Parametrize a matrix to be diagonal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return diagonal(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            Since `diagonal` is a self-map projection, this is simply the identity.
        """
        return y


class Traceless(ParametrizationBase):
    """Parametrize a matrix to be traceless."""

    def forward(self, X):
        return traceless(X)

    def right_inverse(self, Y):
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            Since `traceless` is a self-map projection, this is simply the identity.
        """
        return Y


# endregion linodenet.projections ------------------------------------------------------
