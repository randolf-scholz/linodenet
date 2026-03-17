r"""Parametrizations for matrices (rank-2 tensors)."""

__all__ = [
    # Parametrizations
    "CayleyMap",
    "GramMatrix",
    "MatrixExponential",
    "SpectralNormalization",
    # inherited from linodenet.projections
    "Banded",
    "Diagonal",
    "Hamiltonian",
    "Identity",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "Normal",
    "OrthogonalProjection",
    "RankOne",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "Tridiagonal",
    "UpperTriangular",
]

from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet import projections
from linodenet.constants import ATOL, RTOL
from linodenet.domains import MatrixDomains
from linodenet.parametrize.base import ParametrizationBase
from linodenet.testing import is_square
from linodenet_special.fallbacks import singular_triplet
from signatures import signature

# reexport special projections
Banded = projections.Banded
Masked = projections.Masked
LowRank = projections.LowRank


class CayleyMap(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

    References:
        - https://pytorch.org/tutorials/intermediate/parametrizations.html
        - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL

    Id: Tensor
    r"""BUFFER: The identity matrix."""

    def __init__(self, tensor: Tensor) -> None:
        if not (tensor.ndim == 2 and is_square(tensor)):
            raise ValueError(f"Expected square matrix, got {tensor.shape=}")
        n = tensor.shape[0]
        super().__init__(tensor, unsafe=False)
        self.register_buffer("Id", torch.eye(n))

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.linalg.lstsq(self.Id + x, self.Id - x).solution

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        return torch.linalg.lstsq(self.Id - y, self.Id + y).solution


class MatrixExponential(ParametrizationBase):
    r"""Parametrize a matrix via matrix exponential.

    Note: The following restrictions hold:
        Mₙ(ℝ)  --exp-->  GLₙ(ℝ)
        𝕊ₙ(ℝ)  --exp-->  𝕊ₙ⁺(ℝ)
        𝔸ₙ(ℝ)  --exp-->  Oₙ(ℝ)
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.INVERTIBLE

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return torch.matrix_exp(x)

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""This requires the matrix logarithm, which is not implemented in PyTorch.

        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class GramMatrix(ParametrizationBase):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_SEMIDEFINITE

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(-2, -1) @ x

    @jit.export
    @signature("(..., n, n) -> (..., n, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        r"""This requires the matrix square root, which is not implemented in PyTorch.

        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class SpectralNormalization(nn.Module):
    r"""Spectral normalization $‖A‖₂≤γ$.

    Ensures that the spectral norm of the weight matrix is at most γ (default=1.0).

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL

    sigma: Tensor | None
    r"""BUFFER: The cached singular value."""
    u: Tensor | None
    r"""BUFFER: The cached left singular vector."""
    v: Tensor | None
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
        gamma: float = 1.0,
        atol: float = ATOL,
        rtol: float = RTOL,
        maxiter: Optional[int] = None,
    ) -> None:
        super().__init__()

        # constants
        self.atol = atol
        self.rtol = rtol
        self.maxiter = maxiter

        # shape-dependent buffers are initialized lazily on first use
        self.register_buffer("sigma", None, persistent=True)
        self.register_buffer("u", None, persistent=True)
        self.register_buffer("v", None, persistent=True)
        self.register_buffer("ONE", torch.tensor(1.0), persistent=True)
        self.register_buffer("GAMMA", torch.tensor(float(gamma)), persistent=True)

    @jit.export
    @signature("(..., m, n) -> (..., m, n)")
    def forward(self, weight: Tensor) -> Tensor:
        r"""Perform spectral normalization w ↦ w/‖w‖₂."""
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

    @jit.export
    @signature("(..., m, n) -> (..., m, n)")
    def right_inverse(self, y: Tensor) -> Tensor:
        return y


# Fixed projection modules are wrapped lazily by `register_parametrization`.
Diagonal = projections.Diagonal()
Hamiltonian = projections.Hamiltonian()
Identity = projections.Identity()
LowerTriangular = projections.LowerTriangular()
Normal = projections.Normal()
OrthogonalProjection = projections.Orthogonal()
RankOne = projections.RankOne()
SkewSymmetric = projections.SkewSymmetric()
Symmetric = projections.Symmetric()
Symplectic = projections.Symplectic()
Traceless = projections.Traceless()
Tridiagonal = projections.Tridiagonal()
UpperTriangular = projections.UpperTriangular()
