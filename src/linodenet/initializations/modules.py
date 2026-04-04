r"""Module wrappers for initializations.

Notes:
    - See `linodenet.initializations.functional` for functional implementations.
    - These wrappers bind structural shape arguments in `__init__` and accept only
      sample shape at call time.
"""

__all__ = [
    "DiagonallyDominant",
    "Gaussian",
    "LowRank",
    "Orthogonal",
    "SkewSymmetric",
    "SpecialOrthogonal",
    "Symmetric",
    "Symplectic",
    "Traceless",
]

from typing import Final, Optional

import torch
from torch import Tensor, nn

from linodenet.domains import MatrixDomains

from . import functional


class Gaussian(nn.Module):
    r"""Module wrapper for `gaussian`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.RECTANGULAR

    def __init__(
        self,
        dim: int | tuple[int, int],
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.loc = loc
        self.scale = scale

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.gaussian(
            size,
            dim=self.dim,
            loc=self.loc,
            scale=self.scale,
            dtype=dtype,
            device=device,
        )


class DiagonallyDominant(nn.Module):
    r"""Module wrapper for `diagonally_dominant`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.DIAGONALLY_DOMINANT

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.diagonally_dominant(
            size, dim=self.dim, dtype=dtype, device=device
        )


class Symmetric(nn.Module):
    r"""Module wrapper for `symmetric`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SYMMETRIC

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.symmetric(size, dim=self.dim, dtype=dtype, device=device)


class SkewSymmetric(nn.Module):
    r"""Module wrapper for `skew_symmetric`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SKEW_SYMMETRIC

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.skew_symmetric(size, dim=self.dim, dtype=dtype, device=device)


class Orthogonal(nn.Module):
    r"""Module wrapper for `orthogonal`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.COLUMN_ORTHOGONAL

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.orthogonal(size, dim=self.dim, dtype=dtype, device=device)


class SpecialOrthogonal(nn.Module):
    r"""Module wrapper for `special_orthogonal`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SPECIAL_ORTHOGONAL

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.special_orthogonal(
            size, dim=self.dim, dtype=dtype, device=device
        )


class LowRank(nn.Module):
    r"""Module wrapper for `low_rank`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.LOW_RANK

    def __init__(self, dim: int | tuple[int, int], *, rank: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.low_rank(
            size, dim=self.dim, rank=self.rank, dtype=dtype, device=device
        )


class Traceless(nn.Module):
    r"""Module wrapper for `traceless`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.TRACELESS

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.traceless(size, dim=self.dim, dtype=dtype, device=device)


class Symplectic(nn.Module):
    r"""Module wrapper for `symplectic`."""

    DOMAIN: Final[MatrixDomains] = MatrixDomains.SYMPLECTIC

    def __init__(self, dim: int | tuple[int, int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        size: int | tuple[int, ...] = (),
        /,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str | torch.device] = None,
    ) -> Tensor:
        return functional.symplectic(size, dim=self.dim, dtype=dtype, device=device)
