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
    # ABCs & Protocols
    "StateUpdater",
    "StateUpdaterBase",
    "AbstractStateUpdate",
    # functions
    "is_state_updater",
    "get_state_updater",
]

from abc import abstractmethod
from typing import Final, Protocol, TypeIs, overload, runtime_checkable

from torch import Tensor, nn

from blueprint import Makes, initialize
from signatures import signature


@runtime_checkable
class AbstractStateUpdate[X, Y](Protocol):
    r"""Abstract protocol for state-update callbacks.

    Currently unused and only included for documentation purposes.
    The state updates we consider in practice take Tensors as inputs and outputs.
    In principle, however, one could consider more general types.

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


@overload
def get_state_updater[T: StateUpdater](arg: Makes[T], /, **cfg: object) -> T: ...
@overload
def get_state_updater(arg: str | dict, /, **cfg: object) -> StateUpdater: ...
def get_state_updater(arg: object, /, **cfg: object) -> StateUpdater:
    r"""Initialize a state updater from a configuration.

    Args:
        arg: The configuration to initialize from. Can be one of the following:
            - A string name of a state updater in the `STATE_UPDATERS` dictionary.
            - A class that can be instantiated with the given configuration.
            - A dictionary (Blueprint) with instructions for initializing a state updater.
            - An instance of a state updater.
        **cfg: Additional keyword arguments to pass to the state updater when initializing.
    """
    match arg:
        # if a name, look up in the dictionary
        case str(name):
            from . import STATE_UPDATERS  # noqa: PLC0415

            try:
                obj = STATE_UPDATERS[name]
            except KeyError as exc:
                exc.add_note(
                    f"StateUpdater {name!r} not found in {list(STATE_UPDATERS)=}"
                )
                raise
            return get_state_updater(obj, **cfg)

        # if a class, try to instantiate it with the given configuration
        case type() as cls:
            try:
                return cls(**cfg)
            except TypeError as exc:
                exc.add_note(f"Failed to instantiate {cls} with arguments {cfg!r}")
                raise

        # if a config, extract the name and instantiate
        case dict(spec):
            result = initialize(spec)
            assert isinstance(result, StateUpdater)
            return result

        # if an instance, return as-is
        case StateUpdater() as instance:
            if cfg:
                raise ValueError(f"Cannot pass arguments to an instance: {instance!r}")
            return instance

        case _:
            raise TypeError(f"Invalid argument: {arg!r}")
