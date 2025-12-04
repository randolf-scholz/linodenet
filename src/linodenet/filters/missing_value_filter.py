r"""Wraps an existing Filter $F$ so that it can handle missing values."""

__all__ = ["MissingValueCell"]


from collections.abc import Mapping
from typing import Any, Final, cast

import torch
from torch import Tensor, jit, nn

import linodenet.imputation as imp
from linodenet.constants import EMPTY_MAP
from linodenet.filters.base import Cell, CellBase
from linodenet.imputation import ImputationStrategy, ImputerProtocol


class MissingValueCell(CellBase):
    r"""Wraps an existing Filter $F$ so that it can handle missing values.

    .. math:: x' &= F(u，x)   &   u = impute(m, y, x)

    where $u$ is an imputed value that is free of missing values.
    There are several available imputation strategies:

    0. "default": uses "decoder", if available, and "zero" otherwise.
    1. "zero": Replace missing values with zeros.
    2. "constant": Replace missing values with a constant value.
    2. "last": Replace missing values with the last observed value. (initialized with zero)
    3. "decoder": Replace missing values with the output of the decoder: $s = h(x)$.
    4. Tensor: replaces missing values with a fixed tensor. (for example, the mean of the data)

    Optionally, the mask can be concatenated to the input.

    .. math:: u = concat([impute(m, y, x)，m])
    """

    HP: dict[str, Any] = {}  # FIXME: Remove

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

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        cell_type: type[Cell],
        cell_kwargs: Mapping[str, Any] = EMPTY_MAP,
        concat_mask: bool = True,
        imputation: str | float | Tensor | nn.Module = "zero",
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size)
        self.concat_mask = bool(concat_mask)

        # initialize filter
        filter_input_size = self.input_size * (1 + self.concat_mask)
        filter_options = dict(cell_kwargs) | {
            "input_size": filter_input_size,
            "hidden_size": hidden_size,
        }
        self.cell = cell_type(**filter_options)

        # initialize imputation strategy
        # imputation_strategy: ImputationStrategy
        _imputer: ImputerProtocol
        match imputation:
            case "zero":
                imputation_strategy = ImputationStrategy.ZERO
                _imputer = imp.ZeroImputer()
            case "last":
                imputation_strategy = ImputationStrategy.LAST
                _imputer = imp.LastValueImputer()
            case "learnable":
                imputation_strategy = ImputationStrategy.LEARNABLE
                _imputer = imp.LearnableValueImputer((self.input_size,))
            case "linear":
                imputation_strategy = ImputationStrategy.LINEAR
                _imputer = imp.LinearImputer(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                )
            case (Tensor() | float()) as value:
                imputation_strategy = ImputationStrategy.CONSTANT
                _imputer = imp.ConstantValueImputer(value)
            case nn.Module as imputer:
                imputation_strategy = ImputationStrategy.OTHER
                _imputer = cast("ImputerProtocol", imputer)
            case _:
                raise ValueError(f"Unknown imputation strategy: {imputation}")

        # FIXME: https://github.com/python/mypy/issues/10736
        #   Need to unconditionally assign Final due to mypy bug
        self.imputation_strategy = imputation_strategy
        self._imputer = _imputer

    @jit.export
    def impute(self, mask: Tensor, y: Tensor, x: Tensor) -> Tensor:
        return self._imputer(mask, y, x)

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Signature: ``[(..., m), (..., n)] -> (..., n)``."""
        # compute and buffer mask
        self.mask = ~torch.isnan(y)
        # impute missing values
        self.imputed = self.impute(self.mask, y, x)
        u = self.imputed

        if self.concat_mask:
            u = torch.cat([u, self.mask], dim=-1)

        return self.cell(u, x)
