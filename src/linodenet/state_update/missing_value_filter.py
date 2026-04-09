r"""Wraps an existing state updater $F$ so that it can handle missing values."""

__all__ = ["MissingValueUpdate"]


from collections.abc import Mapping
from typing import Any, Final, cast

import torch
from torch import Tensor, nn

from linodenet.constants import EMPTY_MAP
from signatures import signature

from . import imputation as imp
from .base import StateUpdater, StateUpdaterBase
from .imputation import ImputationStrategy, ImputerProtocol


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
                imputer = imp.ZeroImputer()
            case "last":
                imputation_strategy = ImputationStrategy.LAST
                imputer = imp.LastValueImputer()
            case "learnable":
                imputation_strategy = ImputationStrategy.LEARNABLE
                imputer = imp.LearnableImputer(input_size)
            case "linear":
                imputation_strategy = ImputationStrategy.LINEAR
                imputer = imp.LinearImputer(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                )
            case (Tensor() | float()) as value:
                imputation_strategy = ImputationStrategy.CONSTANT
                imputer = imp.ConstantImputer(value)
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
