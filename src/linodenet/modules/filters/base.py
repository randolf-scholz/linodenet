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
    "CellBase",
    "Filter",
    "FilterBase",
    # Classes
    "MissingValueFilter",
    "ReZeroFilter",
    "ResNetFilter",
    "ResidualFilter",
    "SequentialFilter",
    # Functions
    "filter_from_config",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Final, Optional, Protocol, overload, runtime_checkable

import torch
from torch import Tensor, jit, nn

from linodenet.constants import EMPTY_MAP
from linodenet.utils import try_initialize_from_config


@runtime_checkable
class Cell(Protocol):
    r"""Protocol for cells."""

    input_size: Final[int]  # type: ignore[misc]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    r"""The size of the hidden state $x$."""
    bias: Final[bool]
    r"""Whether to include a bias term or not."""

    @abstractmethod
    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None: ...

    @abstractmethod
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the filter $x' = F(y，x)$.

        .. Signature: ``[(..., n), (..., m)] -> (..., n)``.
        """
        ...


class CellBase(nn.Module):
    r"""Base class for filter-cells."""

    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""
    bias: Final[bool]
    r"""Whether to include a bias term or not."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)

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


@runtime_checkable
class Filter(Cell, Protocol):
    r"""Protocol for filter.

    Additionally to the `Cell` protocol, a filter has knowledge of the observation model.
    This is a decoder that maps the hidden state to the observation space.

    .. math:: y = h(x)
    """

    # CONSTANTS
    input_size: Final[int]  # type: ignore[misc]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]  # type: ignore[misc]
    r"""The size of the hidden state $x$."""
    bias: Final[bool]
    r"""Whether to include a bias term or not."""

    # SUBMODULES
    decoder: Optional[nn.Module] = None
    r"""The observation model."""

    @abstractmethod
    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        decoder: Optional[nn.Module] = None,
    ) -> None: ...

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
    bias: Final[bool]
    r"""Whether to include a bias term or not."""
    decoder: Optional[nn.Module] = None
    r"""The observation model."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.bias = bool(bias)
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


def _make_filter[F: Filter](filter_type: type[F], **config: Any) -> F:
    r"""Initialize a filter from a type."""
    try:
        return filter_type(**config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize filter {filter_type} with arguments {config}!"
        ) from exc


@overload
def filter_from_config[F: Filter](filter: F, /) -> F: ...
@overload
def filter_from_config[F: Filter](filter_kind: type[F], /, **config: Any) -> F: ...
@overload
def filter_from_config(filter_kind: str, /, **config: Any) -> Filter: ...
@overload
def filter_from_config(**config: Any) -> Filter: ...
def filter_from_config(filter_kind: object = None, /, **config: Any) -> Filter:
    r"""Initialize from a configuration."""
    match filter_kind:
        case str(name):
            filter_class = FILTERS[name]
            filter = _make_filter(filter_class, **config)
        case type() as filter_class if issubclass(filter_class, nn.Module):
            filter = _make_filter(filter_class, **config)
        case Filter() as filter if isinstance(filter, nn.Module):
            if config:
                raise ValueError("Cannot initialize instance with additional arguments")
        case None:
            filter = try_initialize_from_config(config)
        case _:
            raise TypeError(f"Invalid filter type: {filter_kind!r}")

    return filter


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
    HP = {}

    # SUBMODULES
    # filter: Filter
    # r"""The wrapped Filter."""
    # decoder: Optional[nn.Module]
    # r"""The observation model."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        *,
        concat_mask: bool = True,
        imputation_strategy: str | Tensor = "default",
        filter_type: str | type[Filter] | Filter = NotImplemented,
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.concat_mask = bool(concat_mask)

        # initialize filter
        options = dict(filter_kwargs)
        filter_input_size = self.input_size * (1 + self.concat_mask)
        options.update(input_size=filter_input_size, hidden_size=self.hidden_size)
        self.filter = filter_from_config(filter_type, **options)
        self.decoder = getattr(self.filter, "decoder", None)

        if self.decoder is not None and not isinstance(self.decoder, nn.Module):
            raise TypeError("Decoder must be a nn.Module!")

        # initialize imputation strategy
        self.register_buffer("S", torch.zeros(self.input_size))
        match imputation_strategy:
            case "default" if self.decoder is not None:
                self.imputation_strategy = "decoder"
            case "default" if self.decoder is None:
                self.imputation_strategy = "zero"
            case "zero":
                self.imputation_strategy = "zero"
            case "last":
                self.imputation_strategy = "last"
            case Tensor() as tensor:
                self.imputation_strategy = "constant"
                self.S = tensor
            case _:
                raise ValueError(f"Invalid imputation strategy: {imputation_strategy}")

    @jit.export
    def impute(self, m: Tensor, y: Tensor, x: Tensor) -> Tensor:
        r"""Update the substitute value."""
        if self.imputation_strategy == "decoder":
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
    r"""Wraps an existing Filter to return the residual $x' = x - F(y，x)$."""

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    # SUBMODULES
    filter: Filter
    r"""The wrapped Filter."""
    decoder: Optional[nn.Module] = None
    r"""The observation model."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size: Optional[int] = None,
        *,
        filter_type: str | type[Filter] | Filter = NotImplemented,
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        options = dict(filter_kwargs)
        options.update(input_size=self.input_size, hidden_size=self.hidden_size)
        self.filter = filter_from_config(filter_type, **options)
        self.decoder = self.filter.decoder

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        return x - self.filter(y, x)


class SequentialFilter(nn.ModuleList):
    r"""Multiple Filters passes applied sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    # CONSTANTS
    input_size: Final[int]
    r"""The size of the observable $y$."""
    hidden_size: Final[int]
    r"""The size of the hidden state $x$."""

    HP = {
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

        super().__init__(module_list)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layers in self:
            x = layers(y, x)
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

    HP = {
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

        super().__init__(module_list)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for layer in self:
            x = x + layer(y, x)
        return x


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
        super().__init__(module_list)
        weight = torch.zeros(len(self))
        self.register_parameter("weight", weight)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        for w, layer in zip(self.weight, self, strict=True):
            x = x + w * layer(y, x)
        return x
