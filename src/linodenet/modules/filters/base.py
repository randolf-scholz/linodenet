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
    "Filter",
    "FilterBase",
    "FilterList",
    # Classes
    "MissingValueFilter",
    "ReZeroFilter",
    "ResNetFilter",
    "ResidualFilter",
    "SequentialFilter",
    "KalmanFilter",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import (
    Any,
    Final,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

import torch
from torch import Tensor, jit, nn

from linodenet.constants import EMPTY_MAP
from linodenet.modules.filters.cells import Cell


@runtime_checkable
class Filter(Cell, Protocol):
    r"""Protocol for filter.

    Additionally to the `Cell` protocol, a filter has knowledge of the observation model.
    This is a decoder that maps the hidden state to the observation space.

    .. math:: y = h(x)

    Attributes:
        input_size: The size of the observable $y$.
        hidden_size: The size of the hidden state $x$.
    """

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter $x' = F(x, y)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


class FilterBase(nn.Module):
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

    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""
    decoder: Optional[nn.Module] = None
    r"""The observation model."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.decoder = decoder

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the state of the system.

        Returns:
            x̂: The updated state of the system.
        """
        ...


class FilterList(FilterBase, nn.ModuleList):
    r"""Container for multiple filters."""

    def __init__(self, filters: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        filter_list: list[Filter] = list(filters)

        if not filter_list:
            raise ValueError("At least one module must be given!")

        for module in filter_list:
            if not isinstance(module, Filter):
                raise TypeError("All modules must be Filters!")
            if not isinstance(module, nn.Module):
                raise TypeError("All filters must be nn.Modules!")

        FilterBase.__init__(
            self, filter_list[0].input_size, filter_list[-1].hidden_size
        )
        nn.ModuleList.__init__(self, cast(list[nn.Module], filter_list))

    @abstractmethod
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        ...


class MissingValueFilter(nn.Module):
    r"""Wraps an existing Filter so that it can handle missing values.

    .. math:: x' = F([m ? s : y]，x)

    where $s$ is the substitute value, for which there are several strategies:

    0. "default": uses "decoder", if available, and "zero" otherwise.
    1. "zero": Replace missing values with zeros.
    2. "last": Replace missing values with the last observed value. (initialized with zero)
    3. "decoder": Replace missing values with the output of the decoder: $s = h(x)$.
    4. Tensor: replaces missing values with a fixed tensor. (for example, the mean of the data)

    Optionally, the mask can be concatenated to the input.

    .. math:: x' = F([ỹ，m]，x)  \qq{where}  ỹ = [m ? s : y]
    """

    # CONSTANTS
    input_size: Final[int]
    r"""CONST: The size of the observable $y$."""
    hidden_size: Final[int]
    r"""CONST: The size of the hidden state $x$."""
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""
    imputation_strategy: Final[str]
    r"""CONST: The strategy to use for imputation."""

    # BUFFERS
    S: Tensor
    r"""A buffer for the substitute tensor."""
    HP: dict = {}

    # SUBMODULES
    # filter: Filter
    # r"""The wrapped Filter."""
    # decoder: Optional[nn.Module]
    # r"""The observation model."""

    class ImputationStrategy(StrEnum):
        r"""The strategy to use for imputation."""

        LAST = "last"
        r"""Impute with last observed value."""
        ZERO = "zero"
        r"""Impute with zeros."""
        CONSTANT = "constant"
        r"""Impute with a (possibly non-zero) constant value."""
        LEARNABLE = "learnable"
        r"""Impute with a (possibly non-zero) learnable value."""
        DECODER = "decoder"
        r"""Impute with decoder."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        filter_type: type[Filter],
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
        concat_mask: bool = True,
        imputation_strategy: str | Tensor = "default",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.concat_mask = bool(concat_mask)

        # initialize filter
        filter_input_size = self.input_size * (1 + self.concat_mask)
        filter_options = dict(filter_kwargs) | {
            "input_size": filter_input_size,
            "hidden_size": hidden_size,
        }
        self.filter = filter_type(**filter_options)
        self.decoder = getattr(self.filter, "decoder", None)

        if self.decoder is not None and not isinstance(self.decoder, nn.Module):
            raise TypeError("Decoder must be a nn.Module!")

        # initialize imputation strategy
        match imputation_strategy:
            case nn.Parameter() as param:
                strategy = (
                    self.ImputationStrategy.LEARNABLE
                    if param.requires_grad
                    else self.ImputationStrategy.CONSTANT
                )
                self.register_parameter("S", param)
            case Tensor() as tensor:
                strategy = self.ImputationStrategy.CONSTANT
                param = nn.Parameter(tensor, requires_grad=False)
                self.register_parameter("S", param)
            case str(name):
                strategy = self.ImputationStrategy(name)
                self.register_buffer("S", torch.zeros(self.input_size))
            case other:
                raise TypeError(f"Expected string or tensor, got {type(other)}")
        self.imputation_strategy = strategy

    @jit.export
    def impute(self, m: Tensor, y: Tensor, x: Tensor) -> Tensor:
        r"""Update the substitute value."""
        if self.imputation_strategy == "decoder" and self.decoder is not None:
            self.S = self.decoder(x)
        elif self.imputation_strategy == "last":
            self.S = torch.where(m, self.S, y)
        elif self.imputation_strategy in {"zero", "constant"}:
            pass
        else:
            raise RuntimeError(
                f"Invalid imputation strategy: {self.imputation_strategy}"
            )

        return self.S

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # impute missing values
        mask = torch.isnan(y)
        substitute = self.impute(mask, y, x)
        y = torch.where(mask, substitute, y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        return self.filter(y, x)


class ResidualFilter(FilterBase):
    r"""Wraps an existing Filter to return the residual $x' = x - η⋅F(y，x)$.

    Attributes:
        input_size: The size of the observable $y$.
        hidden_size: The size of the hidden state $x$.
        filter (Filter): The wrapped Filter.
        decoder (Optional[nn.Module]): The observation model.
    """

    # SUBMODULES
    filter: Filter
    r"""The wrapped Filter."""
    decoder: Optional[nn.Module]
    r"""The observation model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        filter_type: type[Filter],
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        options = dict(filter_kwargs) | {
            "input_size": input_size,
            "hidden_size": hidden_size,
        }
        self.filter = filter_type(**options)
        self.decoder = getattr(self.filter, "decoder", None)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        return x - self.filter(y, x)


class ReZeroFilter(nn.ModuleList):
    r"""Sequential Filter with ReZero connections.

    .. math:: xₖ₊₁ = xₖ + εₖ⋅Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    # Parameters
    weight: Tensor

    def __init__(self, layers: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        # TODO: Use intersection Type Filter & nn.Module
        module_list: list[Filter] = list(layers)

        if not module_list:
            raise ValueError("At least one module must be given!")

        self.input_size = int(module_list[0].input_size)
        self.hidden_size = int(module_list[-1].hidden_size)

        for module in module_list:
            if module.input_size != self.input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {self.input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != self.hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {self.hidden_size}, but {module=} has {module.hidden_size}"
                )
            assert isinstance(module, nn.Module)

        super().__init__(cast(list[nn.Module], module_list))
        # add the weight last.
        self.weight = nn.Parameter(torch.zeros(len(self)))

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for w, layer in zip(self.weight, self, strict=True):
            x = x + w * layer(y, x)
        return x


class ResNetFilter(nn.ModuleList):
    r"""Sequential Filter with residual connections.

    .. math:: xₖ₊₁ = xₖ + Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "layers": [],
    }
    r"""The HyperparameterDict of this class."""

    def __init__(self, layers: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        module_list: list[Filter] = list(layers)
        if not module_list:
            raise ValueError("At least one module must be given!")

        input_size = int(module_list[0].input_size)
        hidden_size = int(module_list[-1].hidden_size)

        for module in module_list:
            if not isinstance(module, Filter) or not isinstance(module, nn.Module):
                raise TypeError("All modules must be Filters!")
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {hidden_size}, but {module=} has {module.hidden_size}"
                )

        super().__init__(cast(list[nn.Module], module_list))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            x = x + layer(y, x)
        return x


class SequentialFilter(nn.ModuleList):
    r"""Multiple Filters passes applied sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    HP: dict = {
        "__name__": __qualname__,
        "__module__": __name__,
        "input_size": None,
        "hidden_size": None,
        "layers": [],
    }
    r"""The HyperparameterDict of this class."""

    def __init__(self, layers: Iterable[Filter], /) -> None:
        r"""Initialize from modules."""
        module_list: list[Filter] = list(layers)

        if not module_list:
            raise ValueError("At least one module must be given!")

        input_size = int(module_list[0].input_size)
        hidden_size = int(module_list[-1].hidden_size)

        for module in module_list:
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != hidden_size:
                raise ValueError(
                    "All modules must have the same hidden_size!"
                    f"Expected {hidden_size}, but {module=} has {module.hidden_size}"
                )
            assert isinstance(module, nn.Module)

        super().__init__(cast(list[nn.Module], module_list))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layers in self:
            x = layers(y, x)
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
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: The observation matrix."""
    R: Tensor
    r"""CONST: The observation noise covariance matrix."""
    L: Tensor
    r"""CONST: Lower triangular Cholesky factor of R."""

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
