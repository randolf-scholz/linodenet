"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = [
    # Parametrizations
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    "SpectralNormalization",
    # Learnable parametrizations
    "ReZero",
    # linodenet.projections
    "Hamiltonian",
    "Identity",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # linodenet.projections masked
    "Banded",
    "Diagonal",
    "LowerTriangular",
    "Masked",
    "UpperTriangular",
]

from typing import Final, Optional

import torch
from torch import BoolTensor, Tensor, jit, nn

from linodenet import projections
from linodenet.constants import ATOL, RTOL
from linodenet.lib import singular_triplet
from linodenet.parametrize.base import ParametrizationBase, ParametrizationMulticache


# region learnable parametrizations ----------------------------------------------------
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


# endregion learnable parametrizations -------------------------------------------------


# region static parametrizations -------------------------------------------------------
class SpectralNormalization(ParametrizationMulticache):
    """Spectral normalization $‖A‖₂≤γ$.

    Ensures that the spectral norm of the weight matrix is at most γ (default=1.0).

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.
    """

    original_parameter: nn.Parameter
    """PARAM: The original parameter, before parametrization."""
    cached_parameter: Tensor
    """BUFFER: The cached parameter, after parametrization."""
    sigma: Tensor
    """BUFFER: The cached singular value."""
    u: Tensor
    """BUFFER: The cached left singular vector."""
    v: Tensor
    """BUFFER: The cached right singular vector."""

    GAMMA: Tensor
    """CONST: The constant γ, the transformation ensures $‖A‖₂≤γ$."""
    ONE: Tensor
    """CONST: The constant 1."""
    maxiter: Final[Optional[int]]
    """CONST: The maximum number of iterations for the power method."""
    atol: Final[float]
    """CONST: The absolute tolerance for the power method."""
    rtol: Final[float]
    """CONST: The relative tolerance for the power method."""

    def __init__(
        self,
        weight: Tensor,
        /,
        *,
        gamma: float = 1.0,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__(weight)

        assert weight.ndim == 2, "weight must be a matrix"
        m, n = weight.shape
        options: dict = {  # FIXME: error with mypy without dict?
            "dtype": weight.dtype,
            "layout": weight.layout,
            "device": weight.device,
        }

        # constants
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter

        # register auxiliary cached tensors
        self.register_cached_tensor("sigma", torch.ones((), **options))
        self.register_cached_tensor("u", torch.randn(m, **options))
        self.register_cached_tensor("v", torch.randn(n, **options))

        # tensor constants
        self.register_buffer("ONE", torch.ones((), **options))
        self.register_buffer("GAMMA", torch.full_like(self.ONE, gamma, **options))

    def forward(self, weight: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Perform spectral normalization w ↦ w/‖w‖₂."""
        # We use the cached singular vectors as initial guess for the power method.
        sigma, u, v = singular_triplet(
            weight,
            u0=self.u,
            v0=self.v,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )
        # map A' ← A ⋅ min(1, γ/‖A₂‖), which is the largest value that ensures
        # ‖A'‖₂ ≤ min(γ, ‖A‖₂)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)

        # return the parametrized weight and the cached singular triplet
        return gamma * weight, {"sigma": sigma, "u": u, "v": v}


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


# endregion static parametrizations ----------------------------------------------------


# region linodenet.projections ---------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
class Identity(ParametrizationBase):
    """Parametrize a matriz as itself."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``... -> ...``."""
        return projections.identity(x)


class Orthogonal(ParametrizationBase):
    """Parametrize a matrix to be orthogonal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.orthogonal(x)


class Symmetric(ParametrizationBase):
    """Parametrize a matrix to be symmetric."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            As opposed to the method in the documentation, we do not use the `triu`
            operator in the forward pass. Our `symmetric` is simply the projection
            onto the matrix manifold and hence the right inverse is the identity.
        """
        return y


class SkewSymmetric(ParametrizationBase):
    """Parametrize a matrix to be skew-symmetric."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.skew_symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        Note:
            As opposed to the method in the documentation, we do not use the `triu`
            operator in the forward pass. Our `skew_symmetric` is simply the projection
            onto the matrix manifold and hence the right inverse is the identity.
        """
        return y


class Traceless(ParametrizationBase):
    """Parametrize a matrix to be traceless."""

    def forward(self, X):
        return projections.traceless(X)


class Normal(ParametrizationBase):
    """Parametrize a matrix to be normal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.normal(x)


class Symplectic(ParametrizationBase):
    """Parametrize a matrix to be symplectic."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.symplectic(x)


class Hamiltonian(ParametrizationBase):
    """Parametrize a matrix to be Hamiltonian."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.hamiltonian(x)


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
class Diagonal(ParametrizationBase):
    """Parametrize a matrix to be diagonal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.diagonal(x)


class UpperTriangular(ParametrizationBase):
    """Parametrize a matrix to be upper triangular."""

    upper: Final[int]
    """CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.upper_triangular(x, upper=self.upper)


class LowerTriangular(ParametrizationBase):
    """Parametrize a matrix to be lower triangular."""

    lower: Final[int]
    """CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, lower: int = 0) -> None:
        super().__init__(tensor)
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.lower_triangular(x, lower=self.lower)


class Masked(ParametrizationBase):
    """Parametrize a matrix to be masked."""

    mask: BoolTensor
    """CONST: The mask to consider"""

    def __init__(self, tensor: Tensor, /, *, mask: BoolTensor) -> None:
        super().__init__(tensor)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.masked(x, mask=self.mask)


class Banded(ParametrizationBase):
    """Parametrize a matrix to be banded."""

    upper: Final[int]
    """CONST: The upper diagonal to consider"""
    lower: Final[int]
    """CONST: The lower diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0, lower: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.banded(x, upper=self.upper, lower=self.lower)


# endregion masked ---------------------------------------------------------------------
# endregion linodenet.projections ------------------------------------------------------
