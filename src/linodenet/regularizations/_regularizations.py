r"""Regularizations for the Linear ODE Networks.

Notes:
    Contains regularizations in modular form.
    See `~linodenet.regularizations.functional` for functional implementations.
"""

__all__ = [
    # ABCs & Protocols
    "RegularizationABC",
    # Regularizations
    "LogDetExp",
    "MatrixNorm",
    # matrix groups
    "Hamiltonian",
    "Identity",
    "LowRank",
    "Normal",
    "Orthogonal",
    "SkewSymmetric",
    "Symmetric",
    "Symplectic",
    "Traceless",
    # masked
    "Banded",
    "Diagonal",
    "LowerTriangular",
    "Masked",
    "UpperTriangular",
]

from abc import abstractmethod
from typing import Final, Union

import torch
from torch import BoolTensor, Tensor, nn

from linodenet.regularizations.functional import (
    banded,
    diagonal,
    hamiltonian,
    identity,
    logdetexp,
    low_rank,
    lower_triangular,
    masked,
    matrix_norm,
    normal,
    orthogonal,
    skew_symmetric,
    symmetric,
    symplectic,
    traceless,
    upper_triangular,
)


class RegularizationABC(nn.Module):
    """Abstract Base Class for Regularization components."""

    @abstractmethod
    def forward(self, x: Tensor, /) -> Tensor:
        r"""Forward pass of the regularization.

        .. Signature: ``... -> 1``.

        Args:
            x: The input tensor to be regularized.

        Returns:
            r: The (scalar) regularization value .
        """


# region regularizations ---------------------------------------------------------------
class LogDetExp(nn.Module):
    r"""Bias $\det(e^A)$ towards 1.

    .. Signature:: ``(..., n, n) -> ...``

    By Jacobi's formula

    .. math:: \det(e^A) = e^{\tr(A)} ⟺ \log(\det(e^A)) = \tr(A)

    In particular, we can regularize the LinODE model by adding a regularization term of the form

    .. math:: |\tr(A)|^p
    """

    p: Final[float]
    size_normalize: Final[bool]

    def __init__(self, *, p: float = 1.0, size_normalize: bool = True) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias $\det(e^A)$ towards 1."""
        return logdetexp(x, p=self.p, size_normalize=self.size_normalize)


class MatrixNorm(nn.Module):
    r"""Return the matrix regularization term.

    .. Signature:: ``(..., n, n) -> ...``
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards zero matrix."""
        return matrix_norm(x, p=self.p, size_normalize=self.size_normalize)


# region matrix groups -----------------------------------------------------------------
class Identity(nn.Module):
    r"""Identity regularization.

    .. Signature:: ``(..., n, n) -> ...``
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards zero matrix."""
        return identity(x, p=self.p, size_normalize=self.size_normalize)


class Symmetric(nn.Module):
    r"""Bias the matrix towards being symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = +X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards symmetric matrix."""
        return symmetric(x, p=self.p, size_normalize=self.size_normalize)


class SkewSymmetric(nn.Module):
    r"""Bias the matrix towards being skew-symmetric.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤ = -X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards skew-symmetric matrix."""
        return skew_symmetric(x, p=self.p, size_normalize=self.size_normalize)


class LowRank(nn.Module):
    r"""Bias the matrix towards being low-rank.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. rank(X) ≤ k
    """

    rank: Final[int]
    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, rank: int = 1, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.rank = rank
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards low-rank matrix."""
        return low_rank(x, rank=self.rank, p=self.p, size_normalize=self.size_normalize)


class Orthogonal(nn.Module):
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards orthogonal matrix."""
        return orthogonal(x, p=self.p, size_normalize=self.size_normalize)


class Traceless(nn.Module):
    """Bias the matrix towards being traceless.

    .. Signature:: ``(..., n, n) -> ...``
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return traceless(x, p=self.p, size_normalize=self.size_normalize)


class Normal(nn.Module):
    r"""Bias the matrix towards being orthogonal.

    .. Signature:: ``(..., n, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. X^⊤X = 𝕀
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return normal(x, p=self.p, size_normalize=self.size_normalize)


class Symplectic(nn.Module):
    r"""Bias the matrix towards being symplectic.

    .. Signature:: ``(..., 2n, 2n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. J^TXJ = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return symplectic(x, p=self.p, size_normalize=self.size_normalize)


class Hamiltonian(nn.Module):
    r"""Bias the matrix towards being hamiltonian.

    .. Signature:: ``(..., 2n, 2n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. (JX)^T = JX
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards normal matrix."""
        return hamiltonian(x, p=self.p, size_normalize=self.size_normalize)


# endregion matrix groups --------------------------------------------------------------


# region masked projections ------------------------------------------------------------
class Diagonal(nn.Module):
    r"""Bias the matrix towards being diagonal.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕀⊙X = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]

    def __init__(
        self, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards diagonal matrix."""
        return diagonal(x, p=self.p, size_normalize=self.size_normalize)


class LowerTriangular(nn.Module):
    r"""Bias the matrix towards being lower triangular.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕃⊙X = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]
    lower: Final[int]

    def __init__(
        self, lower: int = 0, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.lower = lower
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards lower triangular matrix."""
        return lower_triangular(
            x, lower=self.lower, p=self.p, size_normalize=self.size_normalize
        )


class UpperTriangular(nn.Module):
    r"""Bias the matrix towards being upper triangular.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕌⊙X = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]
    upper: Final[int]

    def __init__(
        self, upper: int = 0, *, p: Union[str, int] = "fro", size_normalize: bool = True
    ) -> None:
        super().__init__()
        self.upper = upper
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards upper triangular matrix."""
        return upper_triangular(
            x, upper=self.upper, p=self.p, size_normalize=self.size_normalize
        )


class Banded(nn.Module):
    r"""Bias the matrix towards being banded.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝔹⊙X = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]
    upper: Final[int]
    lower: Final[int]

    def __init__(
        self,
        lower: int = 0,
        upper: int = 0,
        *,
        p: Union[str, int] = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards banded matrix."""
        return banded(
            x,
            upper=self.upper,
            lower=self.lower,
            p=self.p,
            size_normalize=self.size_normalize,
        )


class Masked(nn.Module):
    r"""Bias the matrix towards being masked.

    .. Signature:: ``(..., m, n) -> ...``

    .. math:: A ↦ ‖A-Π(A)‖_p
        where Π(A) = \argmin_X ½∥X-A∥_F^2 s.t. 𝕄⊙X = X
    """

    p: Final[Union[str, int]]
    size_normalize: Final[bool]
    mask: BoolTensor

    def __init__(
        self,
        mask: BoolTensor,
        *,
        p: Union[str, int] = "fro",
        size_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.mask = torch.as_tensor(mask, dtype=torch.bool)  # type: ignore[assignment]
        self.p = p
        self.size_normalize = size_normalize

    def forward(self, x: Tensor) -> Tensor:
        r"""Bias x towards masked matrix."""
        return masked(x, mask=self.mask, p=self.p, size_normalize=self.size_normalize)


# endregion masked projections ---------------------------------------------------------
# endregion regularizations ------------------------------------------------------------
