r"""Projections for the Linear ODE Networks.

Notes:
    - See `linodenet.projections.functional` for functional implementations.
    - See `linodenet.projections.modules` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "FunctionalProjection",
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

from collections.abc import Callable

import torch
from torch import Tensor

from signatures import signature

type FunctionalProjection = Callable[[Tensor], Tensor]


# region projections -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
@signature("(...) -> (...)")
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    .. math:: \min_Y ½‖X-Y‖²
    """
    return x


@signature("(..., n, n) -> (..., n, n)")
def symmetric(x: Tensor) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ = Y

    One can show analytically that Y = ½(X + Xᵀ) is the unique minimizer.
    """
    return (x + x.swapaxes(-1, -2)) / 2


@signature("(..., n, n) -> (..., n, n)")
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ = -Y

    One can show analytically that Y = ½(X - Xᵀ) is the unique minimizer.
    """
    return (x - x.swapaxes(-1, -2)) / 2


@signature("(..., m, n) -> (..., m, n)")
def low_rank(x: Tensor, rank: int = 1) -> Tensor:
    r"""Return the closest low rank matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   rank(Y) ≤ k

    One can show analytically that Y = UₖΣₖVₖᵀ is the unique minimizer,
    where X=UΣVᵀ is the SVD of X.
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return torch.einsum(
        "...ij, ...j, ...jk -> ...ik",
        U[..., :, :rank],
        S[..., :rank],
        Vh[..., :rank, :],
    )


@signature("(..., n, n) -> (..., n, n)")
def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ Y = 𝕀 = YYᵀ

    One can show analytically that $Y = UVᵀ$ is the unique minimizer,
    where $X=UΣVᵀ$ is the SVD of $X$.

    References:
        https://math.stackexchange.com/q/2215359
    """
    U, _, Vh = torch.linalg.svd(x)
    return torch.einsum("...ij, ...jk -> ...ik", U, Vh)


@signature("(..., n, n) -> (..., n, n)")
def traceless(x: Tensor) -> Tensor:
    r"""Return the closest traceless matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   tr(Y) = 0

    One can show analytically that Y = X - (1/n)tr(X)𝕀ₙ is the unique minimizer.

    Note:
        Traceless matrices are also called *trace-free* or *trace-zero* matrices.
        They have the important property that $\det(\exp(X)) = 1$,
        which follows from the fact that $\det(\exp(X)) = \exp(\tr(X))$.
    """
    n = x.shape[-1]
    trace = x.diagonal(dim1=-1, dim2=-2).sum(dim=-1)
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    return x - torch.einsum("..., mn -> ...mn", trace / n, eye)


@signature("(..., n, n) -> (..., n, n)")
def normal(x: Tensor) -> Tensor:
    r"""Return the closest normal matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   YᵀY = YYᵀ

    **The Lagrangian:**

    .. math:: ℒ(Y, Λ) = ½‖X-Y‖² + ⟨Λ, [Y, Yᵀ]⟩

    **First order necessary KKT condition:**

    .. math::
            0 &= ∇ℒ(Y, Λ) = (Y-X) + Y(Λ + Λᵀ) - (Λ + Λᵀ)Y
        \\⟺ Y &= X + [Y, Λ]

    **Second order sufficient KKT condition:**

    .. math::
             ⟨∇h|S⟩=0     &⟹ ⟨S|∇²ℒ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|𝕀⊗𝕀 + Λ⊗𝕀 − 𝕀⊗Λ|S⟩ ≥ 0
         \\⟺ ⟨[Y, Λ]|S⟩=0 &⟹ ⟨S|S⟩ + ⟨[S, Λ]|S⟩ ≥ 0
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@signature("(..., 2n, 2n) -> (..., 2n, 2n)")
def symplectic(x: Tensor) -> Tensor:
    r"""Return the closest symplectic matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@signature("(..., 2n, 2n) -> (..., 2n, 2n)")
def hamiltonian(x: Tensor) -> Tensor:
    r"""Return the closest hamiltonian matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   (JY)ᵀ = JA   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½‖X-Y‖²   s.t.   Yᵀ J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.

    Note:
        The Hamiltonian matrices are the skew-symmetric matrices
        with respect to the symplectic inner product.
        - The matrix exponential of a Hamiltonian matrix is symplectic.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
@signature("[(..., m, n), (m, n)] -> (..., m, n)")
def masked(x: Tensor, mask: Tensor) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   M⊙Y = Y

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


@signature("(..., m, n) -> (..., m, n)")
def diagonal(x: Tensor) -> Tensor:
    r"""Return the closest diagonal matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   𝕀⊙Y = Y

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


@signature("(..., m, n) -> (..., m, n)")
def upper_triangular(x: Tensor, upper: int = 0) -> Tensor:
    r"""Return the closest upper triangular matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   U⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕌⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    return x.triu(diagonal=upper)


@signature("(..., m, n) -> (..., m, n)")
def lower_triangular(x: Tensor, lower: int = 0) -> Tensor:
    r"""Return the closest lower triangular matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   L⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝕃⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    return x.tril(diagonal=lower)


@signature("(..., m, n) -> (..., m, n)")
def banded(x: Tensor, lower: int, upper: int) -> Tensor:
    r"""Return the closest banded matrix to X.

    .. math:: \min_Y ½‖X-Y‖²   s.t.   B⊙Y = Y

    One can show analytically that the unique smallest norm minimizer is $Y = 𝔹⊙X$.

    See Also:
        - `projections.masked`
        - `projections.diagonal`
        - `projections.lower_triangular`
        - `projections.upper_triangular`
        - `projections.banded`
    """
    x = x.triu(diagonal=lower)
    x = x.tril(diagonal=upper)
    return x


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------
@signature("(..., m, n) -> (..., m, n)")
def contraction(x: Tensor, lipschitz_const: float = 1.0) -> Tensor:
    r"""Return the closest contraction matrix to X.

    .. math:: \min_Y ‖X-Y‖₂  s.t. ‖Y‖₂ ≤ θ

    One can show analytically that the unique smallest norm minimizer is
    $Y = \min(1, θ/σ) X$, where $σ = ‖X‖₂$ is the spectral norm of $X$.

    Proof:
        Apply SVD: $X = UΣVᵀ$, then, the problem is equivalent to minimizing
        $‖UΣVᵀ - Y‖₂ = ‖Σ - Uᵀ Y V‖₂ = ‖Σ - Z‖₂$ subject to $‖Z‖₂ ≤ θ$.
        Since $Σ$ is diagonal, one can show the problem is equivalent to minimizing
        $‖𝛔 - 𝐳‖₂$ subject to $‖𝐳‖_∞ ≤ θ$, where $𝐳 = \text{diag}(Z)$.
        Which is solved by $𝐳 = \min(1, θ/σ₁)⋅𝛔$.
    """
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=(-2, -1))
    factor = torch.minimum(lipschitz_const / sigma, torch.ones_like(sigma))
    return x * factor


@signature("(..., n, n) -> (..., n, n)")
def diagonally_dominant(x: Tensor) -> Tensor:
    r"""Return the closest diagonally dominant matrix to X.

    .. math:: \min_Y ‖X-Y‖_F  s.t. |Y_{ii}| ≥ ∑_{j≠i} |Y_{ij}| for all i = 1, …, n

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)
    """
    raise NotImplementedError


# endregion other projections ----------------------------------------------------------
# endregion projections ----------------------------------------------------------------
