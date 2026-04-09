r"""Different state-update models to be used in conjunction with LinODENet.

A state update is a map of the form $x' = F(y, x)$.
The square case `input_size == hidden_size` is common, but not universal.

A state updater takes two positional inputs:

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
    # Protocols
    "AbstractStateUpdate",
    "StateUpdater",
    # ABCs.
    "StateUpdaterBase",
    # classes
    "StateUpdaterList",
    "UpdateSequence",
    "ResidualUpdate",
    "MissingValueUpdate",
    # functions
    "is_state_updater",
]

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import (
    Any,
    Final,
    Protocol,
    TypeIs,
    cast,
    runtime_checkable,
)

import torch
from torch import Tensor, nn

from linodenet.constants import EMPTY_MAP
from linodenet.nn import ModuleSequence
from signatures import signature

from .imputation import (
    ConstantImputer,
    ImputationStrategy,
    ImputerProtocol,
    LastValueImputer,
    LearnableImputer,
    LinearImputer,
    ZeroImputer,
)


@runtime_checkable
class AbstractStateUpdate[X, Y](Protocol):
    r"""Abstract protocol for state-update callbacks.

    .. math::  x' = F(y, x)
    """

    def __call__(self, y: Y, x: X, /) -> X: ...


@runtime_checkable
class StateUpdater(AbstractStateUpdate[Tensor, Tensor], Protocol):
    r"""Protocol for vector-valued state updaters.

    .. math::  x' = F(y, x)
    """

    input_size: Final[int]  # type: ignore[misc]
    hidden_size: Final[int]  # type: ignore[misc]

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    @signature("[(..., d), (..., h)] -> (..., h)")
    def __call__(self, y: Tensor, x: Tensor, /) -> Tensor: ...


class StateUpdaterBase(nn.Module, StateUpdater):
    r"""Base class for state updaters.

    This base class is specialized to the case when X=Y=Tensor, and the arguments
    are vectors.

    .. math::  x' = F(y, x)
    """

    def __init__(self, /, input_size: int, hidden_size: int) -> None:
        # ⚠️ multiple inheritance ⚠️
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        nn.Module.__init__(self)
        StateUpdater.__init__(self, input_size, hidden_size)

    @abstractmethod
    @signature("[(..., d), (..., h)] -> (..., h)")
    def forward(self, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Forward pass of the state updater.

        Args:
            y: The current measurement of the system.
            x: The current estimation of the hidden state of the system.

        Returns:
            The updated state of the system.
        """
        ...


def is_state_updater(arg: object, /) -> TypeIs[StateUpdater]:
    r"""Check whether an object is a state updater."""
    input_size = getattr(arg, "input_size", None)
    hidden_size = getattr(arg, "hidden_size", None)
    return isinstance(input_size, int) and isinstance(hidden_size, int)


class StateUpdaterList[C: StateUpdaterBase](StateUpdaterBase, ModuleSequence[C]):
    r"""Base class for `nn.ModuleList` of state updaters that is itself a state updater.

    Note: This class takes care of tricky multiple inheritance issues with nn.Module.
    """

    def __init__(
        self, modules: Iterable[C] = (), /, *, input_size: int, hidden_size: int
    ) -> None:
        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        assert not hasattr(self, "_modules"), f"Module already initialized: {self}"
        ModuleSequence[C].__init__(self, modules)
        # Note: Need to call StateUpdater.__init__, not StateUpdaterBase.__init__
        #   Otherwise nn.Module.__init__ gets called twice!
        StateUpdater.__init__(self, input_size, hidden_size)

    @abstractmethod
    def forward(self, y_obs: Tensor, x_hat: Tensor, /) -> Tensor: ...


class UpdateSequence[C: StateUpdaterBase](StateUpdaterList[C]):
    r"""Apply multiple state updaters sequentially.

    .. math:: xₖ₊₁ = Fₖ(y, xₖ)
    """

    def __init__(self, modules: Iterable[C] = (), /) -> None:
        cells = list(modules)

        if not cells:
            raise ValueError("At least one module must be given!")

        input_size = cells[0].input_size
        hidden_size = cells[0].hidden_size
        for module in cells:
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

        super().__init__(modules, input_size=input_size, hidden_size=hidden_size)

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for cell in self:
            x = cell(y, x)
        return x


class ResidualUpdate[C: StateUpdaterBase](UpdateSequence[C]):
    r"""Sequential state updater with residual connections.

    .. math:: xₖ₊₁ = xₖ + αₖ⋅Fₖ(y, xₖ)

    Args:
        modules: An iterable of state updater modules to be applied sequentially.
        use_rezero: Whether to use rezero (default: True)

    A regular ResNet is obtained by setting all αₖ=1.0 and making them non-learnable.
    """

    alpha: Tensor
    r"""PARAM: The residual scaling factors αₖ."""
    use_rezero: Tensor
    r"""Whether to use rezero"""

    def __init__(
        self,
        modules: Iterable[C] = (),
        /,
        *,
        use_rezero: bool = True,
    ) -> None:
        super().__init__(modules)
        self.alpha = nn.Parameter(
            torch.zeros(len(self)) if use_rezero else torch.ones(len(self)),
            requires_grad=use_rezero,
        )

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        for alpha, cell in zip(self.alpha, self, strict=True):
            x = x.addcmul(alpha, cell(y, x))  # xₖ₊₁ <- xₖ + αₖfₖ(y, xₖ)
        return x


class MissingValueUpdate(StateUpdaterBase):
    r"""Wraps an existing state updater $F$ so that it can handle missing values.

    .. math:: x' &= F(u，x)   &   (u, m) = impute(y, x)

    where $u$ is an imputed value that is free of missing values.
    There are several available imputation strategies:

    0. "default": uses "decoder", if available, and "zero" otherwise.
    1. "zero": Replace missing values with zeros.
    2. "constant": Replace missing values with a constant value.
    3. "last": Replace missing values with the last observed value. (initialized with zero)
    4. "decoder": Replace missing values with the output of the decoder: $s = h(x)$.
    5. Tensor: replaces missing values with a fixed tensor. (for example, the mean of the data)

    Optionally, the mask can be concatenated to the input.

    .. math:: u = concat([impute(y, x)₀，impute(y, x)₁])
    """

    # CONSTANTS
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the input or not."""
    imputation_strategy: Final[str]
    r"""CONST: The strategy to use for imputation."""
    # BUFFERS
    mask: Tensor
    r"""BUFFER: The mask tensor (true if observed)."""
    imputed: Tensor
    r"""BUFFER: The most recent imputed value."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "filter_type": self.filter_type,
            "filter_kwargs": dict(self.filter_kwargs),
            "concat_mask": self.concat_mask,
            "imputation": self.imputation,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        filter_type: type[StateUpdater],
        filter_kwargs: Mapping[str, Any] = EMPTY_MAP,
        concat_mask: bool = True,
        imputation: str | float | Tensor | nn.Module = "zero",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.filter_type = filter_type
        self.filter_kwargs = dict(filter_kwargs)
        self.imputation = imputation
        self.concat_mask = bool(concat_mask)

        # initialize state updater
        filter_input_size = self.input_size * (1 + self.concat_mask)
        filter_options = dict(filter_kwargs) | {
            "input_size": filter_input_size,
            "hidden_size": hidden_size,
        }
        self.filter = filter_type(**filter_options)

        # initialize imputation strategy
        # imputation_strategy: ImputationStrategy
        imputer: ImputerProtocol
        match imputation:
            case "zero":
                imputation_strategy = ImputationStrategy.ZERO
                imputer = ZeroImputer()
            case "last":
                imputation_strategy = ImputationStrategy.LAST
                imputer = LastValueImputer()
            case "learnable":
                imputation_strategy = ImputationStrategy.LEARNABLE
                imputer = LearnableImputer(input_size)
            case "linear":
                imputation_strategy = ImputationStrategy.LINEAR
                imputer = LinearImputer(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                )
            case (Tensor() | float()) as value:
                imputation_strategy = ImputationStrategy.CONSTANT
                imputer = ConstantImputer(value)
            case nn.Module as module:
                imputation_strategy = "other"
                imputer = cast("ImputerProtocol", module)
            case _:
                raise ValueError(f"Unknown imputation strategy: {imputation}")

        # FIXME: https://github.com/python/mypy/issues/10736
        #   Need to unconditionally assign Final due to mypy bug
        self.imputation_strategy = imputation_strategy
        self.imputer = imputer

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        # impute missing values and store the inferred observation mask
        self.imputed, self.mask = self.imputer(y, x)
        u = self.imputed

        if self.concat_mask:
            u = torch.cat([u, self.mask], dim=-1)

        return self.filter(u, x)
