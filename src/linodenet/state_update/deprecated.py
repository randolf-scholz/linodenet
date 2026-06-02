r"""Deprecated Filters implementations."""

__all__ = [
    "UpdateList",
    "UpdateSequence",
    "UpdateResNet",
    "ReZeroUpdate",
    "PseudoKalmanUpdate",
    "AbstractSquareStateUpdate",
    "SquareStateUpdater",
    "is_square_state_updater",
    "SquareStateUpdaterBase",
]


from abc import abstractmethod
from collections.abc import Iterable
from typing import Final, Protocol, TypeIs, runtime_checkable

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature

from .base import AbstractStateUpdate, VectorStateUpdaterBase
from .kalman import _Alpha


@runtime_checkable
class AbstractSquareStateUpdate[Y](AbstractStateUpdate[Y, Y], Protocol):
    r"""Abstract protocol for square state-update callbacks.

    Currently unused and only included for documentation purposes.
    Square state updates are the special case where observation and state spaces coincide.
    In principle, however, one could consider more general types.

    .. math::  y' = F(y_obs, y_pred)
    """

    def __call__(self, y_obs: Y, y_pred: Y, /) -> Y: ...


@runtime_checkable
class SquareStateUpdater(AbstractSquareStateUpdate[Tensor], Protocol):
    r"""Protocol for vector-valued square state updaters.

    .. math::  y' = F(y_obs, y_pred)

    Note: Every SquareStateUpdater is also a StateUpdater with `hidden_size == input_size`.
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, /, input_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(input_size)

    @abstractmethod
    @signature("[(..., d), (..., d)] -> (..., d)")
    def __call__(self, y_obs: Tensor, y_pred: Tensor, /) -> Tensor: ...


class SquareStateUpdaterBase(VectorStateUpdaterBase):
    r"""Base class for square state updaters.

    This base class is specialized to the case when X=Y=Tensor, and the arguments
    are vectors.

    .. math::  y' = F(y_obs, y_pred)

    Where $x$ is the current state of the system, $y$ is the current measurement, and
    $x'$ is the new state of the system. $ϕ$ is a function that maps the measurement
    to the state of the system. $h$ is a function that maps the current state of the
    system to the measurement.

    Or multiple blocks of said form. In particular, we are interested in state updaters
    satisfying the idempotence property: if $y=h(x)$, then $x'=x$.
    """

    def __init__(self, /, input_size: int) -> None:
        super().__init__(input_size, input_size)

    @abstractmethod
    @signature("[(..., d), (..., d)] -> (..., d)")
    def forward(self, y_obs: Tensor, y_hat: Tensor, /) -> Tensor:
        r"""Forward pass of the state updater.

        Args:
            y_obs: The current measurement of the system.
            y_hat: The current estimation of the state of the system.

        Returns:
            The updated state of the system.
        """
        ...


def is_square_state_updater(arg: object, /) -> TypeIs[SquareStateUpdater]:
    r"""Check whether an object is a square state updater."""
    input_size = getattr(arg, "input_size", None)
    hidden_size = getattr(arg, "hidden_size", None)
    return (
        isinstance(input_size, int)
        and isinstance(hidden_size, int)
        and input_size == hidden_size
    )


class UpdateList[C: VectorStateUpdaterBase](SquareStateUpdaterBase, ModuleSequence[C]):
    r"""Base class for deprecated `nn.ModuleList` state updaters.

    Note: This class takes care of tricky multiple inheritance issues with nn.Module.
    """

    @property
    def config(self) -> dict:
        return {
            "modules": list(self),
            "input_size": self.input_size,
        }

    def __init__(self, modules: Iterable[C] = (), *, input_size: int) -> None:
        # ⚠️ multiple inheritance ⚠️
        # due to how nn.Module.__init__ works, it should only be ever called once
        # because it will overwrite internal state otherwise.
        # Therefore, we need to carefully manually reproduce the __init__ logic here.
        super(ModuleSequence, self).__init__(modules)
        self.input_size = int(input_size)  # type: ignore[misc]
        self.hidden_size = int(input_size)  # type: ignore[misc]

    @abstractmethod
    def forward(self, y_obs: Tensor, y_hat: Tensor, /) -> Tensor: ...


class UpdateSequence[C: VectorStateUpdaterBase](UpdateList[C]):
    r"""Apply multiple deprecated state updaters sequentially."""

    def __init__(self, modules: Iterable[C] = ()) -> None:
        filters = list(modules)

        if not filters:
            raise ValueError("At least one module must be given!")

        input_size = filters[0].input_size
        for module in filters:
            if module.input_size != input_size:
                raise ValueError(
                    "All modules must have the same input_size!"
                    f"Expected {input_size}, but {module=} has {module.input_size}"
                )
            if module.hidden_size != input_size:
                raise ValueError(
                    "All modules must have the same hidden_size as input_size!"
                    f"Expected {input_size}, but {module=} has {module.hidden_size}"
                )

        super().__init__(modules, input_size=input_size)

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        for cell in self:
            y = cell(y_obs, y)
        return y


class UpdateResNet[C: VectorStateUpdaterBase](UpdateSequence[C]):
    r"""Sequential state update with residual connections.

    .. math:: yₖ₊₁ = yₖ + Fₖ(y_obs, yₖ)
    """

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        for cell in self:
            y = y + cell(y_obs, y)
        return y


class ReZeroUpdate[C: VectorStateUpdaterBase](UpdateSequence[C]):
    r"""Sequential state update with ReZero connections.

    .. math:: xₖ₊₁ = xₖ + εₖ⋅Fₖ(y, xₖ)
    """

    @property
    def config(self) -> dict:
        return {"layers": list(self)}

    def __init__(self, layers: Iterable[C]) -> None:
        r"""Initialize from modules."""
        # TODO: Use intersection Type StateUpdater & nn.Module
        module_list: list[C] = list(layers)

        if not module_list:
            raise ValueError("At least one module must be given!")

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

        super().__init__(module_list)
        # add the weight last.
        self.weight = nn.Parameter(torch.zeros(len(self)))

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        for w, layer in zip(self.weight, self, strict=True):
            y = y + w * layer(y_obs, y)
        return y


class PseudoKalmanUpdate(VectorStateUpdaterBase):
    r"""A linear, autoregressive state update.

    .. math::  x̂' = x̂ - αP∏ₘᵀP⁻¹Πₘ(x̂ - x)

    - $α = 1$ is the "last-value" filter
    - $α = 0$ is the "first-value" filter
    - $α = ½$ is the standard Kalman filter, which takes the average between the
      state estimate and the observation.

    One idea: $P = 𝕀 + εA$, where $A$ is symmetric. In this case,
    $𝕀-εA$ is approximately equal to the inverse.

    We define the linearized filter as

    .. math::  x̂' = x̂ - α(𝕀 + εA)∏ₘᵀ(𝕀 - εA)Πₘ(x̂ - x)

    Where $ε$ is initialized as zero.
    """

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""
    kernel: Tensor
    r"""PARAM: The kernel matrix."""

    # BUFFERS
    ZERO: Tensor
    r"""BUFFER: A constant value of zero."""

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "alpha": float(self.alpha),
            "alpha_learnable": self.alpha.requires_grad,
        }

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        alpha: str | float = "last-value",
        alpha_learnable: bool = True,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        # PARAMETERS
        alpha_ = torch.tensor(_Alpha(alpha))
        self.alpha = nn.Parameter(alpha_, requires_grad=alpha_learnable)
        self.epsilon = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.weight = nn.Parameter(torch.empty(self.input_size, self.input_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

        # BUFFERS
        with torch.no_grad():
            kernel = self.epsilon * self.weight
            self.register_buffer("kernel", kernel)
            self.register_buffer("ZERO", torch.zeros(1))

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        # refresh buffer
        kernel = self.epsilon * self.weight

        # create the mask
        mask = ~torch.isnan(y)  # → [..., m]
        z = torch.where(mask, x - y, self.ZERO)  # → [..., m]
        z = z - torch.einsum("ij, ...j -> ...i", kernel, z)  # → [..., n]
        z = torch.where(mask, z, self.ZERO)
        z = z + torch.einsum("ij, ...j -> ...i", kernel, z)
        return x - self.alpha * z
