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
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    "UpperTriangular",
]

from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet import projections
from linodenet.constants import ATOL, RTOL
from linodenet.domains import MatrixDomains
from linodenet.lib import singular_triplet
from linodenet.parametrize.base import Parametrization, Parametrized
from linodenet.testing import is_square


class CayleyMap(Parametrization):
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
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id + x, self.Id - x).solution

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.linalg.lstsq(self.Id - y, self.Id + y).solution


class MatrixExponential(Parametrization):
    r"""Parametrize a matrix via matrix exponential.

    Note: The following restrictions hold:
        Mₙ(ℝ)  --exp-->  GLₙ(ℝ)
        𝕊ₙ(ℝ)  --exp-->  𝕊ₙ⁺(ℝ)
        𝔸ₙ(ℝ)  --exp-->  Oₙ(ℝ)
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SQUARE
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.INVERTIBLE

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return torch.matrix_exp(x)

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix logarithm, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class GramMatrix(Parametrization):
    r"""Parametrize a matrix via gram matrix ($XᵀX$)."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.POSITIVE_SEMIDEFINITE

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``."""
        return x.transpose(-2, -1) @ x

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n, n) -> (..., n, n)``.

        This requires the matrix square root, which is not implemented in PyTorch.
        See: https://github.com/pytorch/pytorch/issues/9983
        """
        raise NotImplementedError


class SpectralNormalization(Parametrization):
    r"""Spectral normalization $‖A‖₂≤γ$.

    Ensures that the spectral norm of the weight matrix is at most γ (default=1.0).

    Note:
        For $‖A‖₂<1$, it follows that $x↦Ax$ is a contraction mapping. In particular,
        the residual mapping $x↦x ± Ax$ is invertible in this case, and the inverse
        can be computed via fixpoint iteration.
    """

    DOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL
    CODOMAIN: Final[MatrixDomains] = MatrixDomains.GENERAL

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
        super().__init__(weight, unsafe=False)
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

    @jit.export
    def forward(self, weight: Tensor) -> Tensor:
        r"""Perform spectral normalization w ↦ w/‖w‖₂.

        .. Signature:: ``(..., m, n) -> (..., m, n)``.
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

    @jit.export
    def right_inverse(self, y: Tensor) -> Tensor:
        r""".. Signature:: ``(..., m, n) -> (..., m, n)``."""
        return y


class Banded(Parametrized):
    r"""Wrapper for ``linodenet.projections.Banded``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Banded(*args, **kwargs))


class Diagonal(Parametrized):
    r"""Wrapper for ``linodenet.projections.Diagonal``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Diagonal(*args, **kwargs))


class Hamiltonian(Parametrized):
    r"""Wrapper for ``linodenet.projections.Hamiltonian``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Hamiltonian(*args, **kwargs))


class Identity(Parametrized):
    r"""Wrapper for ``linodenet.projections.Identity``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Identity(*args, **kwargs))


class LowRank(Parametrized):
    r"""Wrapper for ``linodenet.projections.LowRank``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.LowRank(*args, **kwargs))


class LowerTriangular(Parametrized):
    r"""Wrapper for ``linodenet.projections.LowerTriangular``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.LowerTriangular(*args, **kwargs))


class Masked(Parametrized):
    r"""Wrapper for ``linodenet.projections.Masked``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Masked(*args, **kwargs))


class Normal(Parametrized):
    r"""Wrapper for ``linodenet.projections.Normal``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Normal(*args, **kwargs))


class OrthogonalProjection(Parametrized):
    r"""Wrapper for ``linodenet.projections.OrthogonalProjection``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Orthogonal(*args, **kwargs))


class SkewSymmetric(Parametrized):
    r"""Wrapper for ``linodenet.projections.SkewSymmetric``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.SkewSymmetric(*args, **kwargs))


class Symmetric(Parametrized):
    r"""Wrapper for ``linodenet.projections.Symmetric``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Symmetric(*args, **kwargs))


class Symplectic(Parametrized):
    r"""Wrapper for ``linodenet.projections.Symplectic``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Symplectic(*args, **kwargs))


class Traceless(Parametrized):
    r"""Wrapper for ``linodenet.projections.Traceless``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.Traceless(*args, **kwargs))


class UpperTriangular(Parametrized):
    r"""Wrapper for ``linodenet.projections.UpperTriangular``."""

    def __init__(self, tensor: Tensor, /, *args: Any, **kwargs: Any) -> None:
        super().__init__(tensor, projections.UpperTriangular(*args, **kwargs))
