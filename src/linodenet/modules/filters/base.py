r"""Different Filter models to be used in conjunction with LinODENet.

A Filter takes two positional inputs:
- An input tensor y: the current measurement of the system
- An input tensor x: the current estimation of the state of the system

Sometimes, we have a third input, so called covariates $u$.
There are two types of covariates:

- Control inputs: These are external variables that influence the system
- Exogenous inputs: Sometimes, there are two coupled systems, and we have access to
  measurements / predictions of the other system (example: weather forecast).
  In this case we can treat these variables as part of the state.


These are external variables that influence the system,
but are not part of the state.

Example:
    The linear state space system is given by the equations (without noise):

    .. math::
        ẋ(t) &= A(t)x(t) + B(t)u(t) \\
        y(t) &= C(t)x(t) + D(t)u(t)

    Here $u$ is the control input.
"""

__all__ = [
    # ABCs & Protocols
    "Cell",
    "Filter",
    "ProbabilisticFilter",
    "FilterABC",
    # Classes
    "KalmanFilter",
    "MissingValueFilter",
    "ResidualFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
]

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping

import torch
from torch import Tensor, jit, nn
from torch.distributions import Distribution
from typing_extensions import Any, Final, Optional, Protocol, Self, runtime_checkable

from linodenet.constants import EMPTY_MAP
from linodenet.modules.filters import KalmanCell, NonLinearCell
from linodenet.modules.layers import (
    ReverseDense,
    ReZeroCell,
)
from linodenet.utils import try_initialize_from_config


@runtime_checkable
class Cell(Protocol):
    """Protocol for cells."""

    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@runtime_checkable
class Filter(Protocol):
    """Protocol for filter.

    Additionally to the `Cell` protocol, a filter has knowledge of the observation model.
    This is a decoder that maps the hidden state to the observation space.
    """

    # CONSTANTS
    input_size: Final[int]  # type: ignore[misc]
    """The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    """The size of the hidden state $x$."""
    direct_observation: Final[bool]  # type: ignore[misc]
    """Whether the observation model is the identity or not."""

    # SUBMODULES
    decoder: nn.Module
    """The observation model."""

    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        """Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


@runtime_checkable
class ProbabilisticFilter(Protocol):
    """Protocol for probabilistic filters.

    The goal of a probabilistic filter is to update the distribution of the hidden state,
    given the current observation. This is done by updating the parameters of the distribution
    that represents the state.

    In practice, this is done by acting on the parameters that define the distribution model,
    rather than in function space itself.
    """

    decoder: Distribution
    """The model for the conditional distribution $p(y|x)$."""

    def __call__(self, y: Distribution, x: Distribution, /) -> Distribution:
        """Forward pass of the filter $p' = F(q, p)$."""
        ...


class FilterABC(nn.Module):
    r"""Base class for all filters.

    All filters should have a signature of the form:

    .. math::  x' = x + ϕ(y-h(x))

    Where $x$ is the current state of the system, $y$ is the current measurement, and
    $x'$ is the new state of the system. $ϕ$ is a function that maps the measurement
    to the state of the system. $h$ is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in Filters
    satisfying the idempotence property: if $y=h(x)$, then $x'=x$.
    """

    input_size: int
    """The size of the observable $y$."""
    hidden_size: int
    """The size of the hidden state $x$."""

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """
        ...


class MissingValueFilter(nn.Module):
    r"""Wraps an existing Filter to impute missing values via the observation model.

    Given a filter $F$ and an observation model $h$, this returns the filter defined by

    .. math:: x' = F( [m ? h(x) : y]，x)
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""
    direct_observation: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""

    # BUFFERS
    ZERO: Tensor
    """A buffer for the zero tensor."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "concat_mask": True,
        "input_size": None,
        "hidden_size": None,
        "autoregressive": True,
        "Cell": {
            "__name__": "GRUCell",
            "__module__": "torch.nn",
            "input_size": None,
            "hidden_size": None,
            "bias": True,
            "device": None,
            "dtype": None,
        },
    }
    r"""The HyperparameterDict of this class."""

    def __init__(
        self,
        cell: Cell = NotImplemented,
        /,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        *,
        decoder: str | Callable[[Tensor], Tensor] = "zero",
        concat_mask: bool = True,
        cell_type: type[Cell] = nn.GRUCell,
        cell_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.concat_mask = concat_mask
        self.register_buffer("ZERO", torch.tensor(0.0))
        self._decoder = self.zero if decoder is None else decoder

        if cell is NotImplemented:
            # create configuration
            options = dict(cell_kwargs)
            if input_size is not None:
                assert "input_size" not in options
                options["input_size"] = (1 + concat_mask) * input_size
            if hidden_size is not None:
                assert "hidden_size" not in options
                options["hidden_size"] = hidden_size

            # attempt to initialize cell from hyperparameters
            try:
                self.cell = cell_type(**options)
            except TypeError as exc:
                raise RuntimeError(
                    f"Failed to initialize cell of type {cell_type} with inputs {options}"
                ) from exc
        else:
            self.cell = cell
            self.input_size = input_size if input_size is not None else cell.input_size
            self.hidden_size = (
                hidden_size if hidden_size is not None else cell.hidden_size
            )

        match decoder:
            case "zero":
                self.decoder = Constant(self.zero)
            case "linear":
                self.decoder = nn.Linear(self.hidden_size, self.input_size)
            case Callable() as decoder_func:
                self.decoder = decoder_func
            case _:
                raise TypeError(f"Invalid decoder type: {decoder}")

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # impute missing value in observation with state estimate
        mask = torch.isnan(y)
        y = torch.where(mask, self.decoder(x), y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Flatten for RNN-Cell
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        # Apply filter
        result = self.cell(y, x)

        # Unflatten return value
        return result.view(mask.shape)


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
