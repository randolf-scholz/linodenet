"""Implementations of various filters."""

__all__ = [
    # Classes
    "ResidualFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
    "KalmanFilter",
]

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, jit, nn
from typing_extensions import Any, Final, Optional, Self

from linodenet.constants import EMPTY_MAP
from linodenet.modules.filters.base import Filter
from linodenet.modules.filters.cells import KalmanCell, LinearKalmanCell, NonLinearCell
from linodenet.modules.layers import ReverseDense, ReZeroCell
from linodenet.utils import try_initialize_from_config


class ResidualFilter(nn.Module):
    """Wraps an existing Filter to return the residual $x' = x - F(y，x)$."""

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""

    # SUBMODULES
    cell: Filter
    """The wrapped Filter."""

    def __init__(self, cell: Filter, /):
        super().__init__()
        self.cell = cell
        self.input_size = cell.input_size
        self.hidden_size = cell.hidden_size

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        return x - self.cell(y, x)


class ResidualFilterBlock(nn.Module):
    r"""Filter cell combined with several output layers.

    .. math:: x' = x - Φ(F(y, x)) \\ Φ = Φₙ ◦ ... ◦ Φ₁
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
        "filter": KalmanCell,
        "layers": [ReverseDense.HP | {"bias": False}, ReZeroCell.HP],
    }
    r"""The HyperparameterDict of this class."""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any) -> Self:
        r"""Initialize from hyperparameters."""
        config = cls.HP | dict(cfg, **kwargs)

        # initialize cell:
        cell = try_initialize_from_config(
            config["filter"],
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
        )

        # initialize layers:
        layers: list[nn.Module] = []
        for layer in config["layers"]:
            module = try_initialize_from_config(layer, input_size=config["hidden_size"])
            layers.append(module)

        return cls(cell, *layers)

    def __init__(self, cell: nn.Module, /, *modules: nn.Module) -> None:
        super().__init__()

        self.cell = cell
        self.layers = nn.Sequential(*modules)

        try:
            self.input_size = int(cell.input_size)
            self.hidden_size = int(cell.hidden_size)
        except AttributeError as exc:
            raise ValueError(
                "First/Last modules must have `input_size` and `hidden_size`!"
            ) from exc

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        z = self.cell(y, x)
        return x - self.layers(z)


class SequentialFilter(nn.ModuleList):
    r"""Multiple Filters applied sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": False,
        "layers": [LinearKalmanCell, NonLinearCell, NonLinearCell],
    }
    r"""The HyperparameterDict of this class."""

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] = EMPTY_MAP, /, **kwargs: Any) -> Self:
        r"""Initialize from hyperparameters."""
        config = cls.HP | dict(cfg, **kwargs)

        # build the layers
        module_list: list[nn.Module] = []
        for layer in config["layers"]:
            module = try_initialize_from_config(layer, **config)
            module_list.append(module)

        return cls(module_list)

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        r"""Initialize from modules."""
        module_list = list(modules)

        if not module_list:
            raise ValueError("At least one module must be given!")

        try:
            self.input_size = int(module_list[0].input_size)
            self.hidden_size = int(module_list[-1].hidden_size)
        except AttributeError as exc:
            raise AttributeError(
                "First/Last modules must have `input_size` and `hidden_size`!"
            ) from exc

        super().__init__(module_list)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for module in self:
            x = module(y, x)
        return x


class KalmanFilter(nn.Module):
    r"""Classical Kalman Filter.

    .. math::
        x̂ₜ₊₁ &= x̂ₜ + Pₜ Hₜᵀ(Hₜ Pₜ   Hₜᵀ + Rₜ)⁻¹ (yₜ - Hₜ x̂ₜ) \\
        Pₜ₊₁ &= Pₜ - Pₜ Hₜᵀ(Hₜ Pₜ⁻¹ Hₜᵀ + Rₜ)⁻¹ Hₜ Pₜ⁻¹

    In the case of missing data:

    Substitute $yₜ← Sₜ⋅yₜ$, $Hₜ ← Sₜ⋅Hₜ$ and $Rₜ ← Sₜ⋅Rₜ⋅Sₜᵀ$ where $Sₜ$
    is the $mₜ×m$ projection matrix of the missing values. In this case:

    .. math::
        x̂' &= x̂ + P⋅Hᵀ⋅Sᵀ(SHPHᵀSᵀ + SRSᵀ)⁻¹ (Sy - SHx̂) \\
           &= x̂ + P⋅Hᵀ⋅Sᵀ(S (HPHᵀ + R) Sᵀ)⁻¹ S(y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅(S⁺S)ᵀ (HPHᵀ + R)⁻¹ (S⁺S) (y - Hx̂) \\
           &= x̂ + P⋅Hᵀ⋅∏ₘᵀ (HPHᵀ + R)⁻¹ ∏ₘ (y - Hx̂) \\
        P' &= P - P⋅Hᵀ⋅Sᵀ(S H P⁻¹ Hᵀ Sᵀ + SRSᵀ)⁻¹ SH P⁻¹ \\
           &= P - P⋅Hᵀ⋅(S⁺S)ᵀ (H P⁻¹ Hᵀ + R)⁻¹ (S⁺S) H P⁻¹ \\
           &= P - P⋅Hᵀ⋅∏ₘᵀ (H P⁻¹ Hᵀ + R)⁻¹ ∏ₘ H P⁻¹


    .. note::
        The Kalman filter is a linear filter. The non-linear version is also possible,
        the so called Extended Kalman-Filter. Here, the non-linearity is linearized at
        the time of update.

        ..math ::
            x̂' &= x̂ + P⋅Hᵀ(HPHᵀ + R)⁻¹ (y - h(x̂)) \\
            P' &= P -  P⋅Hᵀ(HPHᵀ + R)⁻¹ H P

        where $H = \frac{∂h}{∂x}|_{x̂}$. Note that the EKF is generally not an optimal
        filter.
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: The observation matrix."""
    R: Tensor
    r"""PARAM: The observation noise covariance matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    def __init__(self, /, input_size: int, hidden_size: int):
        super().__init__()

        # CONSTANTS
        self.input_size = input_size
        self.hidden_size = hidden_size

        # BUFFERS
        self.register_buffer("ZERO", torch.zeros(1))

        # PARAMETERS
        self.H = nn.Parameter(torch.empty(input_size, hidden_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.kaiming_normal_(self.H, nonlinearity="linear")
        nn.init.kaiming_normal_(self.R, nonlinearity="linear")

    @jit.export
    def forward(self, y: Tensor, x: Tensor, *, P: Optional[Tensor] = None) -> Tensor:
        r"""Return $x' = x + P⋅Hᵀ∏ₘᵀ(HPHᵀ + R)⁻¹ ∏ₘ (y - Hx)$."""
        P = torch.eye(x.shape[-1]) if P is None else P
        # create the mask
        mask = ~torch.isnan(y)
        H = self.H
        R = self.R
        r = torch.einsum("ij, ...j -> ...i", H, x) - y
        r = torch.where(mask, r, self.ZERO)
        z = torch.linalg.solve(H @ P @ H.t() + R, r)
        z = torch.where(mask, z, self.ZERO)
        return x - torch.einsum("ij, jk, ..k -> ...i", P, H.t(), z)
