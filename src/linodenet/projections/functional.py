r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "Projection",
    # Projections
    "banded",
    "contraction",
    "diagonal",
    "diagonally_dominant",
    "hamiltonian",
    "identity",
    "low_rank",
    "lower_triangular",
    "masked",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    "upper_triangular",
]

from typing import Protocol, runtime_checkable

import torch
from torch import BoolTensor, Tensor

from linodenet.constants import TRUE


@runtime_checkable
class Projection(Protocol):
    r"""Protocol for Projection Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the projection.

        .. Signature: ``(..., d) -> (..., f)``.
        """
        ...


# region projections -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    .. Signature:: ``... -> ...``

    .. math:: \min_Y ½∥X-Y∥_F^2
    """
    return x


def symmetric(x: Tensor) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """
    return (x + x.swapaxes(-1, -2)) / 2


def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """
    return (x - x.swapaxes(-1, -2)) / 2


def low_rank(x: Tensor, rank: int = 1) -> Tensor:
    r"""Return the closest low rank matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   rank(Y) ≤ k

    One can show analytically that Y = UₖΣₖVₖ^𝖳 is the unique minimizer,
    where X=UΣV^𝖳 is the SVD of X.
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return torch.einsum(
        "...ij, ...j, ...jk -> ...ik",
        U[..., :, :rank],
        S[..., :rank],
        Vh[..., :rank, :],
    )


def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References:
        https://math.stackexchange.com/q/2215359
    """
    U, _, Vh = torch.linalg.svd(x)
    return torch.einsum("...ij, ...jk -> ...ik", U, Vh)


def traceless(x: Tensor) -> Tensor:
    r"""Return the closest traceless matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   tr(Y) = 0

    One can show analytically that Y = X - (1/n)tr(X)𝕀ₙ is the unique minimizer.

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They have the important property that $\det(\exp(X)) = 1$,
        which follows from the fact that $\det(\exp(X)) = \exp(\r(X))$.
    """
    n = x.shape[-1]
    trace = x.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    return x - torch.einsum("..., mn -> ...mn", trace / n, eye)


def normal(x: Tensor) -> Tensor:
    r"""Return the closest normal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^⊤Y = YY^⊤

    **The Lagrangian:**

    .. math:: ℒ(Y, Λ) = ½∥X-Y∥_F^2 + ⟨Λ, [Y, Y^⊤]⟩

    **First order necessary KKT condition:**

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λ^⊤) - (Λ + Λ^⊤)Y
        \\⟺ Y &= X + [Y, Λ]

    **Second order sufficient KKT condition:**

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


def symplectic(x: Tensor) -> Tensor:
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


def hamiltonian(x: Tensor) -> Tensor:
    r"""Return the closest hamiltonian matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   (JY)^T = JA   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.

    Note:
        The Hamiltonian matrices are the skew-symmetric matrices
        with respect to the symplectic inner product.
        - The matrix exponential of a Hamiltonian matrix is symplectic.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
def masked(x: Tensor, mask: BoolTensor = TRUE) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕄⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕄⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    mask_ = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    return torch.where(mask_, x, zero)


def diagonal(x: Tensor) -> Tensor:
    r"""Return the closest diagonal matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕀⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕀⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    eye = torch.eye(x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.where(eye, x, zero)


def upper_triangular(x: Tensor, upper: int = 0) -> Tensor:
    r"""Return the closest upper triangular matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕌⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕌⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    return x.triu(diagonal=upper)


def lower_triangular(x: Tensor, lower: int = 0) -> Tensor:
    r"""Return the closest lower triangular matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝕃⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕃⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    return x.tril(diagonal=lower)


def banded(x: Tensor, upper: int = 0, lower: int = 0) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   𝔹⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝔹⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    x = x.triu(diagonal=upper)
    x = x.tril(diagonal=lower)
    return x


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
def contraction(x: Tensor, lipschitz_const: float = 1.0) -> Tensor:
    r"""Return the closest contraction matrix to X.

    .. Signature:: ``(..., m, n) -> (..., m, n)``

    .. math:: \min_Y ∥X-Y∥₂  s.t. ‖Y‖₂ ≤ θ

    One can show analytically that the unique smallest norm minimizer is
    $Y = \min(1, θ/σ) X$, where $σ = ‖X‖₂$ is the spectral norm of $X$.

    Proof:
        Apply SVD: $X = UΣV^𝖳$, then, the problem is equivalent to minimizing
        $∥UΣV^𝖳 - Y‖₂ = ∥Σ - U^𝖳 Y V‖₂ = ∥Σ - Z‖₂$ subject to $‖Z‖₂ ≤ θ$.
        Since $Σ$ is diagonal, one can show the problem is equivalent to minimizing
        $∥𝛔 - 𝐳‖₂$ subject to $‖𝐳‖_∞ ≤ θ$, where $𝐳 = \text{diag}(Z)$.
        Which is solved by $𝐳 = \min(1, θ/σ₁)⋅𝛔$.
    """
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=(-2, -1))
    factor = torch.minimum(lipschitz_const / sigma, torch.ones_like(sigma))
    return x * factor


def diagonally_dominant(x: Tensor) -> Tensor:
    r"""Return the closest diagonally dominant matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ∥X-Y∥_F  s.t. |Y_{ii}| ≥ ∑_{j≠i} |Y_{ij}| for all i = 1, …, n

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)
    """
    raise NotImplementedError


# endregion other projections ----------------------------------------------------------
# endregion projections ----------------------------------------------------------------
