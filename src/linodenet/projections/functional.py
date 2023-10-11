r"""Projections for the Linear ODE Networks.

Notes
-----
Contains projections in functional form.
  - See `~linodenet.projections.modular` for modular implementations.
"""

__all__ = [
    # Protocol
    "Projection",
    # Projections
    "hamiltonian",
    "identity",
    "normal",
    "orthogonal",
    "skew_symmetric",
    "symmetric",
    "symplectic",
    "traceless",
    # masked
    "banded",
    "diagonal",
    "lower_triangular",
    "upper_triangular",
    "masked",
]

from typing import Protocol, runtime_checkable

import torch
from torch import BoolTensor, Tensor, jit

from linodenet.constants import TRUE


@runtime_checkable
class Projection(Protocol):
    """Protocol for Projection Components."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """Forward pass of the projection.

        .. Signature: ``(..., d) -> (..., f)``.
        """
        ...


# region projections -------------------------------------------------------------------
# region matrix groups -----------------------------------------------------------------
@jit.script
def identity(x: Tensor) -> Tensor:
    r"""Return x as-is.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2
    """
    return x


@jit.script
def symmetric(x: Tensor) -> Tensor:
    r"""Return the closest symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^⊤ = Y

    One can show analytically that Y = ½(X + X^⊤) is the unique minimizer.
    """
    return (x + x.swapaxes(-1, -2)) / 2


@jit.script
def skew_symmetric(x: Tensor) -> Tensor:
    r"""Return the closest skew-symmetric matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^⊤ = -Y

    One can show analytically that Y = ½(X - X^⊤) is the unique minimizer.
    """
    return (x - x.swapaxes(-1, -2)) / 2


@jit.script
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


@jit.script
def orthogonal(x: Tensor) -> Tensor:
    r"""Return the closest orthogonal matrix to X.

    .. Signature:: ``(..., n, n) -> (..., n, n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 Y = 𝕀 = YY^𝖳

    One can show analytically that $Y = UV^𝖳$ is the unique minimizer,
    where $X=UΣV^𝖳$ is the SVD of $X$.

    References
    ----------
    - `<https://math.stackexchange.com/q/2215359>`_
    """
    U, _, Vh = torch.linalg.svd(x)
    return torch.einsum("...ij, ...jk -> ...ik", U, Vh)


@jit.script
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


@jit.script
def symplectic(x: Tensor) -> Tensor:
    r"""Return the closest symplectic matrix to X.

    .. Signature:: ``(..., 2n, 2n) -> (..., 2n, 2n)``

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J=[𝟎, 𝕀; -𝕀, 𝟎]

    Alternatively, the above is equivalent to

    .. math:: \min_Y ½∥X-Y∥_F^2   s.t.   Y^𝖳 J Y = J   where   J= 𝔻₊₁-𝔻₋₁

    where $𝔻ₖ$ is the $2n×2n$ matrix with ones on the k-th diagonal.
    """
    raise NotImplementedError("TODO: implement Fixpoint / Gradient based algorithm.")


@jit.script
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
@jit.script
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


@jit.script
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


@jit.script
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


@jit.script
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


@jit.script
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
# endregion projections ----------------------------------------------------------------
