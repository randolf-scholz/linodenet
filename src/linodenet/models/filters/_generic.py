"""Generic Filters that wrap other Filters."""

__all__ = [
    "MissingValueFilter",
    "ResidualFilterBlock",
    "SequentialFilter",
]

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, jit, nn
from typing_extensions import Any, Final, Self

from linodenet.constants import EMPTY_MAP
from linodenet.models.filters.base import (
    Filter,
    KalmanCell,
    LinearKalmanCell,
    NonLinearCell,
)
from linodenet.utils import (
    ReverseDense,
    ReZeroCell,
    deep_dict_update,
    deep_keyval_update,
    initialize_from_dict,
    try_initialize_from_config,
)


class MissingValueFilter(nn.Module):
    r"""Wraps an existing Filter to impute missing values with observation model.

    Given a filter $F$ and an observation model $h$, this returns:

    - $x' = F( [m ? h(x) : y]，x)$ if `concat_mask=False`
    - $x' = F( concat([m ? h(x) : y]，m)，x)$ if `concat_mask=True`
    """

    # CONSTANTS
    input_size: Final[int]
    """The size of the observable $y$."""
    hidden_size: Final[int]
    """The size of the hidden state $x$."""
    autoregressive: Final[bool]
    r"""CONST: Whether the filter is autoregressive or not."""
    concat_mask: Final[bool]
    r"""CONST: Whether to concatenate the mask to the inputs."""

    # PARAMETERS
    H: Tensor
    r"""PARAM: the observation matrix."""

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

    def __init__(self, /, input_size: int, hidden_size: int, **cfg: Any):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        # CONSTANTS
        self.concat_mask = config["concat_mask"]
        self.input_size = input_size * (1 + self.concat_mask)
        self.hidden_size = hidden_size
        self.autoregressive = config["autoregressive"]

        if self.autoregressive:
            assert (
                hidden_size == input_size
            ), "Autoregressive filter requires x_dim == y_dim"
            self.H = torch.eye(input_size)
        else:
            self.H = nn.Parameter(torch.empty(input_size, hidden_size))
            nn.init.kaiming_normal_(self.H, nonlinearity="linear")

        deep_keyval_update(
            config, input_size=self.input_size, hidden_size=self.hidden_size
        )

        # MODULES
        self.cell = initialize_from_dict(config["Cell"])

    @jit.export
    def h(self, x: Tensor) -> Tensor:
        r"""Apply the observation function."""
        if self.autoregressive:
            return x
        return torch.einsum("ij, ...j -> ...i", self.H, x)

    @jit.export
    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        mask = torch.isnan(y)

        # impute missing value in observation with state estimate
        y = torch.where(mask, self.h(x), y)

        if self.concat_mask:
            y = torch.cat([y, mask], dim=-1)

        # Flatten for RNN-Cell
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])

        # Apply filter
        result = self.cell(y, x)

        # De-Flatten return value
        return result.view(mask.shape)


class ResidualFilter(nn.Module):
    """Wraps an existing Filter to return the residual $x' = x - F(y，x)$."""

    cell: Filter
    """The wrapped Filter."""

    def __init__(self, cell: Filter, /):
        super().__init__()
        self.cell = cell

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

    .. math:: x' = Fₙ(y, zₙ₋₁) = Fₙ(y，Fₙ₋₁(y，...F₂(y，F₁(y, x))...))
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
