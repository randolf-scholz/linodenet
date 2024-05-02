r"""Parametrizations.

There are 2 types of parametrizations:

1. ad-hoc parametrizations
2. other parametrizations
"""

__all__ = [
    # Parametrizations
    "Banded",
    "CayleyMap",
    "Diagonal",
    "GramMatrix",
    "Hamiltonian",
    "Identity",
    "LowRank",
    "LowerTriangular",
    "Masked",
    "MatrixExponential",
    "Normal",
    "Orthogonal",
    "ReZero",
    "SkewSymmetric",
    "SpectralNormalization",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
]

import torch
from torch import BoolTensor, Tensor, jit, nn
from typing_extensions import Final, Optional

from linodenet import projections
from linodenet.constants import ATOL, RTOL
from linodenet.lib import singular_triplet
from linodenet.parametrize.base import ParametrizationBase, ParametrizationMulticache


# region learnable parametrizations ----------------------------------------------------
class ReZero(ParametrizationBase):
    r"""ReZero."""

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

        initial_value = torch.as_tensor(0.0 if scalar is None else scalar)
        self.scalar = nn.Parameter(initial_value) if self.learnable else initial_value

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(...,) -> (...,)``."""
        return self.scalar * x


# endregion learnable parametrizations -------------------------------------------------


# region static parametrizations -------------------------------------------------------
class SpectralNormalization(ParametrizationMulticache):
    r"""Spectral normalization $‖A‖₂≤γ$.

    Ensures that the spectral norm of the weight matrix is at most γ (default=1.0).

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.
    """

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
        # map A' ← A ⋅ min(1, γ/‖A₂‖), which is the largest value that ensures
        # ‖A'‖₂ ≤ min(γ, ‖A‖₂)
        gamma = torch.minimum(self.ONE, self.GAMMA / sigma)

        # return the parametrized weight and the cached singular triplet
        return gamma * weight, {"sigma": sigma, "u": u, "v": v}


class CayleyMap(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal via Cayley-Map.

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
    r"""Parametrize a matrix via matrix exponential."""

    def forward(self, X):
        return torch.matrix_exp(X)

    def right_inverse(self, y):
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix logarithm, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class GramMatrix(ParametrizationBase):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    def forward(self, X):
        return X.T @ X

    def right_inverse(self, y):
        """.. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix square root, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


# endregion static parametrizations ----------------------------------------------------


# region linodenet.projections ---------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
class Identity(ParametrizationBase):
    r"""Parametrize a matriz as itself."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``... -> ...``."""
        return projections.identity(x)


class Symmetric(ParametrizationBase):
    r"""Parametrize a matrix to be symmetric."""

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
    r"""Parametrize a matrix to be skew-symmetric."""

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


class LowRank(ParametrizationBase):
    r"""Parametrize a matrix to be low-rank."""

    rank: Final[int]
    r"""CONST: The rank to consider"""

    def __init__(self, tensor: Tensor, /, *, rank: int = 1) -> None:
        super().__init__(tensor)
        self.rank = rank

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.low_rank(x, rank=self.rank)


class Orthogonal(ParametrizationBase):
    r"""Parametrize a matrix to be orthogonal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.orthogonal(x)


class Traceless(ParametrizationBase):
    r"""Parametrize a matrix to be traceless."""

    def forward(self, X):
        return projections.traceless(X)


class Normal(ParametrizationBase):
    r"""Parametrize a matrix to be normal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.normal(x)


class Symplectic(ParametrizationBase):
    r"""Parametrize a matrix to be symplectic."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.symplectic(x)


class Hamiltonian(ParametrizationBase):
    r"""Parametrize a matrix to be Hamiltonian."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return projections.hamiltonian(x)


# endregion matrix groups --------------------------------------------------------------


# region masked ------------------------------------------------------------------------
class Diagonal(ParametrizationBase):
    r"""Parametrize a matrix to be diagonal."""

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.diagonal(x)


class UpperTriangular(ParametrizationBase):
    r"""Parametrize a matrix to be upper triangular."""

    upper: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.upper_triangular(x, upper=self.upper)


class LowerTriangular(ParametrizationBase):
    r"""Parametrize a matrix to be lower triangular."""

    lower: Final[int]
    r"""CONST: The diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, lower: int = 0) -> None:
        super().__init__(tensor)
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.lower_triangular(x, lower=self.lower)


class Masked(ParametrizationBase):
    r"""Parametrize a matrix to be masked."""

    mask: BoolTensor
    r"""CONST: The mask to consider"""

    def __init__(self, tensor: Tensor, /, *, mask: BoolTensor) -> None:
        super().__init__(tensor)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.masked(x, mask=self.mask)


class Banded(ParametrizationBase):
    r"""Parametrize a matrix to be banded."""

    upper: Final[int]
    r"""CONST: The upper diagonal to consider"""
    lower: Final[int]
    r"""CONST: The lower diagonal to consider"""

    def __init__(self, tensor: Tensor, /, *, upper: int = 0, lower: int = 0) -> None:
        super().__init__(tensor)
        self.upper = upper
        self.lower = lower

    def forward(self, x: Tensor) -> Tensor:
        """.. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return projections.banded(x, upper=self.upper, lower=self.lower)


# endregion masked ---------------------------------------------------------------------
# endregion linodenet.projections ------------------------------------------------------
