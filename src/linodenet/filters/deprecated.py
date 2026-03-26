r"""Deprecated Filters implementations."""

__all__ = [
    "FilterList",
    "FilterSequence",
    "FilterResNet",
    "ReZeroFilter",
    "PseudoKalmanCell",
]


from abc import abstractmethod
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from linodenet.nn import ModuleSequence
from signatures import signature

from .base import CellBase, FilterBase
from .kalman_cell import _Alpha


class FilterList[C: CellBase](FilterBase, ModuleSequence[C]):
    r"""Base class for nn.ModuleList of Filters that's a Filter itself.

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
        self.input_size = int(input_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
        self.hidden_size = int(input_size)  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]

    @abstractmethod
    def forward(self, y_obs: Tensor, y_hat: Tensor, /) -> Tensor: ...


class FilterSequence[C: CellBase](FilterList[C]):
    r"""Apply multiple Filters sequentially."""

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


class FilterResNet[C: CellBase](FilterSequence[C]):
    r"""Sequential Filter with Residual connections.

    .. math:: yₖ₊₁ = yₖ + Fₖ(y_obs, yₖ)
    """

    @signature("[(..., m), (..., n)] -> (..., n)")
    def forward(self, y_obs: Tensor, y: Tensor) -> Tensor:
        for cell in self:
            y = y + cell(y_obs, y)
        return y


class ReZeroFilter[C: CellBase](FilterSequence[C]):
    r"""Sequential Filter with ReZero connections.

    .. math:: xₖ₊₁ = xₖ + εₖ⋅Fₖ(y, xₖ)
    """

    @property
    def config(self) -> dict:
        return {"layers": list(self)}

    def __init__(self, layers: Iterable[C]) -> None:
        r"""Initialize from modules."""
        # TODO: Use intersection Type Filter & nn.Module
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


class PseudoKalmanCell(CellBase):
    r"""A Linear, Autoregressive Filter.

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
