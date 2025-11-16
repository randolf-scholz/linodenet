r"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = [
    "MatrixSpace",
    # Parametrizations
    "Banded",
    "CayleyMap",
    "Diagonal",
    "GramMatrix",
    "Hamiltonian",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixExponential",
    "Normal",
    "OrthogonalProjection",
    "SkewSymmetric",
    "SpectralNormalization",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
]

from enum import StrEnum
from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet import projections
from linodenet.constants import ATOL, RTOL
from linodenet.lib import singular_triplet
from linodenet.parametrize.base import ParametrizationBase
from linodenet.testing import is_square


class MatrixSpace(StrEnum):
    r"""Enumeration of matrix spaces for parametrizations."""

    GENERAL = "general"
    LOW_RANK = "low_rank"

    SQUARE = "square"  # n x n matrices
    EVEN_SQUARE = "even_square"  # 2n x 2n matrices

    SYMMETRIC = "symmetric"  # 𝕊ₙ(R)
    SKEW_SYMMETRIC = "skew_symmetric"
    POSITIVE_DEFINITE = "positive_definite"  # 𝕊ₙ⁺(ℝ)
    NEGATIVE_DEFINITE = "negative_definite"  # 𝕊ₙ⁻(ℝ)
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"  # 𝕊ₙ⁺(ℝ) ∪ {0}
    NEGATIVE_SEMIDEFINITE = "negative_semidefinite"  # 𝕊ₙ⁻(ℝ) ∪ {0}

    # determinant-based
    SINGULAR = "singular"  # det=0
    INVERTIBLE = "invertible"  # GLₙ(R) (det≠0)
    POSITIVE_DETERMINANT = "positive_determinant"  # GLₙ⁺(R) (det>0)
    NEGATIVE_DETERMINANT = "negative_determinant"  # GLₙ⁻(R) (det<0)

    NORMAL = "normal"
    ORTHOGONAL = "orthogonal"  # Oₙ(R)
    SPECIAL_ORTHOGONAL = "special_orthogonal"  # SOₙ(R)
    PERMUTATION = "permutation"

    TRACELESS = "traceless"
    SYMPLECTIC = "symplectic"
    HAMILTONIAN = "hamiltonian"

    MASKED = "masked"
    DIAGONAL = "diagonal"
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    BANDED = "banded"

    STOCHASTIC = "stochastic"
    DOUBLY_STOCHASTIC = "doubly_stochastic"


# region learnable parametrizations ----------------------------------------------------


# endregion learnable parametrizations -------------------------------------------------


# region static parametrizations -------------------------------------------------------
class CayleyMap(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.SPECIAL_ORTHOGONAL

    Id: Tensor
    r"""BUFFER: The identity matrix."""

    def __init__(self, tensor: Tensor) -> None:
        if not (tensor.ndim == 2 and is_square(tensor)):
            raise ValueError(f"Expected square matrix, got {tensor.shape=}")
        n = tensor.shape[0]
        super().__init__(tensor)
        self.register_buffer("Id", torch.eye(n))

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id + x, self.Id - x).solution

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id - y, self.Id + y).solution


class MatrixExponential(ParametrizationBase):
    r"""Parametrize a matrix via matrix exponential.

    Note: The following restrictions hold:
        Mₙ(ℝ)  --exp-->  GLₙ(ℝ)
        𝕊ₙ(ℝ)  --exp-->  𝕊ₙ⁺(ℝ)
        𝔸ₙ(ℝ)  --exp-->  Oₙ(ℝ)
    """

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.INVERTIBLE

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.matrix_exp(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix logarithm, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class GramMatrix(ParametrizationBase):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.POSITIVE_SEMIDEFINITE

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return x.transpose(-2, -1) @ x

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix square root, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


# endregion static parametrizations ----------------------------------------------------


# region linodenet.projections ---------------------------------------------------------
# region matrix groups -----------------------------------------------------------------


class Symmetric(ParametrizationBase):
    r"""Parametrize a matrix to be symmetric."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.SYMMETRIC

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class SkewSymmetric(ParametrizationBase):
    r"""Parametrize a matrix to be skew-symmetric."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.SKEW_SYMMETRIC

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.skew_symmetric(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class OrthogonalProjection(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.ORTHOGONAL

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.orthogonal(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class Traceless(ParametrizationBase):
    r"""Parametrize a matrix to be traceless.

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They have the important property that $\det(\exp(X)) = 1$,
        which follows from the fact that $\det(\exp(X)) = \exp(\tr(X))$.
    """

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.TRACELESS

    def forward(self, x: Tensor) -> Tensor:
        return projections.traceless(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class Normal(ParametrizationBase):
    r"""Parametrize a matrix to be normal."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.NORMAL

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.normal(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class Symplectic(ParametrizationBase):
    r"""Parametrize a matrix to be symplectic."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.EVEN_SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.SYMPLECTIC

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return projections.symplectic(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return y


class Hamiltonian(ParametrizationBase):
    r"""Parametrize a matrix to be Hamiltonian."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.EVEN_SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.HAMILTONIAN

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return projections.hamiltonian(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``."""
        return y


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
class Diagonal(ParametrizationBase):
    r"""Parametrize a matrix to be diagonal."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.SQUARE
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.DIAGONAL

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.diagonal(x)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return y


class UpperTriangular(ParametrizationBase):
    r"""Parametrize a matrix to be upper triangular."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.UPPER_TRIANGULAR

    upper: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.upper_triangular(x, upper=self.upper)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


class LowerTriangular(ParametrizationBase):
    r"""Parametrize a matrix to be lower triangular."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.LOWER_TRIANGULAR

    lower: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, lower: int = 0) -> None:
        super().__init__(tensor)
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.lower_triangular(x, lower=self.lower)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


class Masked(ParametrizationBase):
    r"""Parametrize a matrix to be masked."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.MASKED

    mask: Tensor
    r"""CONST: Boolean mask to consider"""

    def __init__(self, tensor: Tensor, /, *, mask: Tensor) -> None:
        super().__init__(tensor)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.masked(x, mask=self.mask)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


class Banded(ParametrizationBase):
    r"""Parametrize a matrix to be banded."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.BANDED

    upper: Final[int]
    r"""CONST: The upper diagonal to consider"""
    lower: Final[int]
    r"""CONST: The lower diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0, lower: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.banded(x, upper=self.upper, lower=self.lower)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


# endregion masked ---------------------------------------------------------------------


class LowRank(ParametrizationBase):
    r"""Parametrize a matrix to be low-rank."""

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.LOW_RANK

    rank: Final[int]
    r"""CONST: The rank to consider"""

    def __init__(self, tensor: Tensor, /, *, rank: int = 1) -> None:
        super().__init__(tensor)
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.low_rank(x, rank=self.rank)

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


class SpectralNormalization(ParametrizationBase):
    r"""Spectral normalization $‖A‖₂≤γ$.

    Ensures that the spectral norm of the weight matrix is at most γ (default=1.0).

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.
    """

    DOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL
    CODOMAIN: Final[MatrixSpace] = MatrixSpace.GENERAL

    original_parameter: nn.Parameter
    r"""PARAM: The original parameter, before parametrization."""
    cached_parameter: Tensor
    r"""BUFFER: The cached parameter, after parametrization."""
    sigma: Tensor
    r"""BUFFER: The cached singular value."""
    u: Tensor
    r"""BUFFER: The cached left singular vector."""
    v: Tensor
    r"""BUFFER: The cached right singular vector."""

    GAMMA: Tensor
    r"""CONST: The constant γ, the transformation ensures $‖A‖₂≤γ$."""
    ONE: Tensor
    r"""CONST: The constant 1."""
    maxiter: Final[Optional[int]]
    r"""CONST: The maximum number of iterations for the power method."""
    atol: Final[float]
    r"""CONST: The absolute tolerance for the power method."""
    rtol: Final[float]
    r"""CONST: The relative tolerance for the power method."""

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
        if weight.ndim != 2:
            raise ValueError("weight must be a matrix")

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
        self.register_buffer("sigma", torch.ones((), **options))
        self.register_buffer("u", torch.randn(m, **options))
        self.register_buffer("v", torch.randn(n, **options))

        # tensor constants
        self.register_buffer("ONE", torch.ones((), **options))
        self.register_buffer("GAMMA", gamma * torch.ones((), **options))

    def forward(self, weight: Tensor) -> Tensor:
        r"""Perform spectral normalization w ↦ w/‖w‖₂.

        .. Signature:: ``(..., n, n) -> (..., n, n)``.
        """
        # We use the cached singular vectors as initial guess for the power method.
        sigma, u, v = singular_triplet(
            weight,
            u0=self.u,
            v0=self.v,
            atol=self.atol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )

        # store the buffers
        self.sigma = sigma
        self.u = u
        self.v = v

        # map A' ← A ⋅ min(1, γ/‖A₂‖), which is the largest value that ensures
        # ‖A'‖₂ ≤ min(γ, ‖A‖₂)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)

        # return the parametrized weight and the cached singular triplet
        return gamma * weight

    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


# endregion linodenet.projections ------------------------------------------------------
