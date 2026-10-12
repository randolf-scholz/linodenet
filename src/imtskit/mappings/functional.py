r"""Functional projections for vectors and matrices.

Notes:
    - See `imtskit.mappings.projections` for module-based implementations.
"""

__all__ = [
    # ABCs & Protocols
    "ProjectionFn",
    "ProjectionFnWithArgs",
    # matrix Projections
    "banded",
    "contraction",
    "lipschitz_bounded",
    "spectrally_normalized",
    "diagonal",
    "diagonally_dominant",
    "hamiltonian",
    "identity",
    "low_rank",
    "lower_triangular",
    "masked",
    "normal",
    "orthogonal",
    "rank_one",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    "tridiagonal",
    "upper_triangular",
    # vector projections
    "unit_vector",
]

from collections.abc import Callable
from typing import Concatenate

import torch
from torch import Tensor

from signatures import signature

type ProjectionFn = Callable[[Tensor], Tensor]
type ProjectionFnWithArgs = Callable[Concatenate[Tensor, ...], Tensor]


# region projections -------------------------------------------------------------------
def unit_vector(x: Tensor, /) -> Tensor:
    r"""Return the closest unit vector to X.

    The input must be nonzero; normalization is undefined for the zero vector.
    """
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


# region matrix groups -----------------------------------------------------------------
@signature("(...) -> (...)")
def identity(x: Tensor, /) -> Tensor:
    r"""Return x as-is.

    $$  \min_Y ½‖X-Y‖_F²  $$
    """
    return x


@signature("(..., n, n) -> (..., n, n)")
def symmetric(x: Tensor, /) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Yᵀ = Y  $$

    The unique minimizer is $Y=½(X+Xᵀ)$.
    """
    return (x + x.swapaxes(-1, -2)) / 2


@signature("(..., n, n) -> (..., n, n)")
def skew_symmetric(x: Tensor, /) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Yᵀ = -Y  $$

    The unique minimizer is $Y=½(X-Xᵀ)$.
    """
    return (x - x.swapaxes(-1, -2)) / 2


@signature("(..., m, n) -> (..., m, n)")
def rank_one(x: Tensor, /) -> Tensor:
    r"""Return the closest rank-1 matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \rank(Y) ≤ 1  $$

    This is the special case of `low_rank` with `rank=1`.
    """
    return low_rank(x, rank=1)


@signature("(..., n, n) -> (..., n, n)")
def orthogonal(x: Tensor, /) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} YᵀY = 𝕀 = YYᵀ  $$

    An SVD $X=UΣVᵀ$ gives a minimizer $Y=UVᵀ$. The minimizer is
    unique when $X$ is nonsingular.

    References:
        - [Nearest orthogonal matrix discussion](https://math.stackexchange.com/q/2215359)
    """
    U, _, Vh = torch.linalg.svd(x)
    return torch.einsum("...ij, ...jk -> ...ik", U, Vh)


@signature("(..., n, n) -> (..., n, n)")
def traceless(x: Tensor, /) -> Tensor:
    r"""Return the closest traceless matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \tr(Y) = 0  $$

    The unique minimizer is $Y=X-\tr(X)𝕀ₙ/n$.

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
def normal(x: Tensor, /) -> Tensor:
    r"""Describe the nearest-normal-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} YYᵀ=YᵀY  $$

    The Lagrangian is

    $$  ℒ(Y,Λ)=½‖X-Y‖_F²+⟨Λ,YYᵀ-YᵀY⟩  $$

    For a stationary point admitting a Lagrange multiplier, the first-order
    condition is

    $$  0=Y-X+(Λ+Λᵀ)Y-Y(Λ+Λᵀ)  $$

    For a perturbation $S$ satisfying the linearized constraint
    $SYᵀ+YSᵀ-SᵀY-YᵀS=0$, the second variation is

    $$  ∂²ℒ(Y,Λ)[S,S]=‖S‖_F²+2⟨Λ,SSᵀ-SᵀS⟩  $$

    Nonnegativity is a second-order necessary condition under the usual
    constraint regularity assumptions; it is not a sufficient condition.

    Raises:
        NotImplementedError: The nearest-normal-matrix solver is not implemented.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@signature("(..., 2n, 2n) -> (..., 2n, 2n)")
def symplectic(x: Tensor, /) -> Tensor:
    r"""Describe the nearest-symplectic-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} YᵀJY=J  $$

    Here $J$ is the canonical symplectic matrix

    $$  J=\bmat{ 𝟎 & 𝕀ₙ \\ -𝕀ₙ & 𝟎 }  $$

    Raises:
        NotImplementedError: The symplectic projection is not implemented.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@signature("(..., 2n, 2n) -> (..., 2n, 2n)")
def hamiltonian(x: Tensor, /) -> Tensor:
    r"""Describe the nearest-Hamiltonian-matrix problem in Frobenius norm.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} (JY)ᵀ=JY  $$

    Here $J$ is the canonical symplectic matrix

    $$  J=\bmat{ 𝟎 & 𝕀ₙ \\ -𝕀ₙ & 𝟎 }  $$

    Equivalently, $YᵀJ+JY=0$. The exponential of a Hamiltonian matrix is
    symplectic.

    Raises:
        NotImplementedError: The Hamiltonian projection is not implemented.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------


@signature("(..., m, n) -> (..., m, n)")
def diagonal(x: Tensor, /) -> Tensor:
    r"""Return the closest diagonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} 𝕀⊙Y = Y  $$

    The unique minimizer is $Y = 𝕀⊙X$.

    See Also:
        - `imtskit.mappings.functional.masked`
        - `imtskit.mappings.functional.diagonal`
        - `imtskit.mappings.functional.lower_triangular`
        - `imtskit.mappings.functional.upper_triangular`
        - `imtskit.mappings.functional.banded`
    """
    eye = torch.eye(x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device)
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    return torch.where(eye, x, zero)


@signature("(..., n, n) -> (..., n, n)")
def upper_triangular(x: Tensor, /, upper: int = 0) -> Tensor:
    r"""Return the closest upper triangular matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} U⊙Y = Y  $$

    The unique minimizer is $Y = U⊙X$.

    See Also:
        - `imtskit.mappings.functional.masked`
        - `imtskit.mappings.functional.diagonal`
        - `imtskit.mappings.functional.lower_triangular`
        - `imtskit.mappings.functional.upper_triangular`
        - `imtskit.mappings.functional.banded`
    """
    return x.triu(diagonal=upper)


@signature("(..., n, n) -> (..., n, n)")
def lower_triangular(x: Tensor, /, lower: int = 0) -> Tensor:
    r"""Return the closest lower triangular matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} L⊙Y = Y  $$

    The unique minimizer is $Y = L⊙X$.

    See Also:
        - `imtskit.mappings.functional.masked`
        - `imtskit.mappings.functional.diagonal`
        - `imtskit.mappings.functional.lower_triangular`
        - `imtskit.mappings.functional.upper_triangular`
        - `imtskit.mappings.functional.banded`
    """
    return x.tril(diagonal=lower)


@signature("(..., n, n) -> (..., n, n)")
def tridiagonal(x: Tensor, /) -> Tensor:
    r"""Return the closest tridiagonal matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} Y = T⊙Y  $$

    This is the special case of `banded` with `lower=-1` and `upper=1`.
    """
    return banded(x, lower=-1, upper=1)


# endregion masked projections ---------------------------------------------------------


# region other projections -------------------------------------------------------------


@signature("(..., n, n) -> (..., n, n)")
def diagonally_dominant(x: Tensor, /) -> Tensor:
    r"""Describe the nearest-diagonally-dominant-matrix problem.

    $$  \min_Y ‖X-Y‖_F  \qq{subject to} |Yᵢᵢ| ≥ ∑_{j≠i} |Yᵢⱼ| \qq{for all} i=1,…,n  $$

    References:
        Computing the nearest diagonally dominant matrix (Mendoza et al. 1998)

    Raises:
        NotImplementedError: The diagonally dominant projection is not implemented.
    """
    raise NotImplementedError


# endregion other projections ----------------------------------------------------------


# region special projections -----------------------------------------------------------
@signature("(..., m, n) -> (..., m, n)")
def lipschitz_bounded(x: Tensor, /, lipschitz_bound: float) -> Tensor:
    r"""Return the closest matrix to X with spectral norm at most $γ$.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ ≤ γ  $$

    A spectral-norm minimizer is $f(X) = \min(1, γ/‖X‖₂)⋅X$.

    Proof:
        For $γ≥0$ and $X≠0$, let $a=\min(1,γ/‖X‖₂)$. Then $Y=aX$ is feasible
        and attains the reverse-triangle-inequality lower bound
        $‖X-Y‖₂=\max(‖X‖₂-γ,0)$. This proves optimality, but not uniqueness.
    """
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=(-2, -1))
    factor = torch.minimum(lipschitz_bound / sigma, torch.ones_like(sigma))
    return x * factor


@signature("(..., m, n) -> (..., m, n)")
def spectrally_normalized(x: Tensor, /) -> Tensor:
    r"""Return the closest matrix to X with unit spectral norm.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ = 1  $$

    A spectral-norm minimizer is $f(X) = X/‖X‖₂$.

    The input must be nonzero; normalization is undefined for the zero matrix.
    """
    sigma = torch.linalg.matrix_norm(x, ord=2, dim=(-2, -1))
    return x / sigma


@signature("(..., m, n) -> (..., m, n)")
def contraction(x: Tensor, /, lipschitz_bound: float) -> Tensor:
    r"""Return the closest matrix to X with spectral norm at most $0<γ<1$.

    $$  \min_Y ‖X-Y‖₂ \qq{subject to} ‖Y‖₂ ≤ γ < 1  $$

    A spectral-norm minimizer is $f(X) = \min(1, γ/‖X‖₂)⋅X$.
    """
    if not 0 < lipschitz_bound < 1:
        raise ValueError("lipschitz_bound must be between 0 and 1")
    return lipschitz_bounded(x, lipschitz_bound=lipschitz_bound)


@signature("[(..., m, n), (m, n)] -> (..., m, n)")
def masked(x: Tensor, /, mask: Tensor) -> Tensor:
    r"""Return the closest matrix supported on the Boolean mask to X.

    $$
    \min_Y ½‖X-Y‖_F² \qq{subject to} M⊙Y = Y
    $$

    The unique minimizer is $Y = M⊙X$.

    See Also:
        - `imtskit.mappings.functional.masked`
        - `imtskit.mappings.functional.diagonal`
        - `imtskit.mappings.functional.lower_triangular`
        - `imtskit.mappings.functional.upper_triangular`
        - `imtskit.mappings.functional.banded`
    """
    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    mask_ = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    return torch.where(mask_, x, zero)


@signature("(..., m, n) -> (..., m, n)")
def banded(x: Tensor, /, lower: int, upper: int) -> Tensor:
    r"""Return the closest banded matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} B⊙Y = Y  $$

    The unique minimizer is $Y = B⊙X$.

    See Also:
        - `imtskit.mappings.functional.masked`
        - `imtskit.mappings.functional.diagonal`
        - `imtskit.mappings.functional.lower_triangular`
        - `imtskit.mappings.functional.upper_triangular`
        - `imtskit.mappings.functional.banded`
    """
    x = x.triu(diagonal=lower)
    x = x.tril(diagonal=upper)
    return x


@signature("(..., m, n) -> (..., m, n)")
def low_rank(x: Tensor, /, rank: int) -> Tensor:
    r"""Return the closest low rank matrix to X.

    $$  \min_Y ½‖X-Y‖_F² \qq{subject to} \rank(Y) ≤ k  $$

    An SVD $X=UΣVᵀ$ gives a minimizer $Y=UₖΣₖVₖᵀ$. If the truncation
    cuts through equal nonzero singular values, the minimizer need not be unique.
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    return torch.einsum(
        "...ij, ...j, ...jk -> ...ik",
        U[..., :, :rank],
        S[..., :rank],
        Vh[..., :rank, :],
    )


# endregion special projections --------------------------------------------------------


# endregion projections ----------------------------------------------------------------
