r"""Base imputer classes and functions."""

__all__ = [
    "ImputerProtocol",
    "ImputationStrategy",
    "zero_impute",
    "ZeroImputer",
    "ConstantImputer",
    "LastValueImputer",
    "LinearImputer",
]

from enum import StrEnum
from typing import Final, Protocol

import torch
from torch import Tensor, nn

from linodenet.signatures import signature


class ImputerProtocol(Protocol):
    r"""Protocol for imputer."""

    @signature("[(..., m), (..., m), (..., n)] -> (..., m)")
    def __call__(self, mask: Tensor, y: Tensor, x: Tensor, /) -> Tensor:
        r"""Impute missing values in a tensor.

        Args:
            mask: Mask tensor (true if observed, false if missing).
            y: Observed state.
            x: Estimated state.
        """
        ...


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
    LINEAR = "linear"
    r"""Impute with a linear function of the hidden state."""
    OTHER = "other"
    r"""Impute with decoder."""


class LinearImputer(nn.Module):
    r"""Impute missing values with a linear function of the hidden state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, mask: Tensor, y_obs: Tensor, y_hat: Tensor) -> Tensor:
        r"""Impute missing values with a linear function.

        .. math:: (m, y, x) ⟼ ⟦m ? y : Hx⟧

        Args:
            mask: Mask tensor (true if observed)
            y_obs: Observed state.
            y_hat: Estimated state.
        """
        return torch.where(mask, y_obs, self.linear(y_hat))


def zero_impute(mask: Tensor, y: Tensor, _: Tensor) -> Tensor:
    r"""Impute missing values with a constant.

    Args:
        mask (Tensor): Mask tensor (true if observed)
        y (Tensor): Observed state.
        _ (Tensor): Hidden state.
    """
    return torch.where(mask, y, torch.zeros_like(y))


class ZeroImputer(nn.Module):
    r"""Impute missing values with zero."""

    def forward(self, m: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values with zero.

        .. math:: (m, y, *) ⟼ ⟦m ? y : 0⟧

        Args:
            m (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        return torch.where(m, y, torch.zeros_like(y))


class ConstantImputer(nn.Module):
    r"""Impute missing values with a constant."""

    value: Tensor
    r"""Constant value to impute missing values."""
    learnable: Final[bool]
    r"""Whether the constant value is learnable or not."""

    def __init__(self, constant: float | Tensor, /, *, learnable: bool = False) -> None:
        super().__init__()
        tensor = torch.tensor(constant)
        self.learnable = learnable
        self.value = nn.Parameter(tensor, requires_grad=learnable)

    def forward(self, mask: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values with a constant.

        .. math:: (m, y, *) ⟼ ⟦m ? y : c⟧

        Args:
            mask (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        return torch.where(mask, y, self.value)


class LastValueImputer(nn.Module):
    r"""Impute missing values with the last observed value."""

    last_value: Tensor
    r"""Last observed value."""
    decay: Tensor
    r"""Decay factor for the last value."""

    def __init__(self, *, decay: float = 0.0, decay_learnable: bool = False) -> None:
        super().__init__()
        self.register_buffer("last_value", torch.zeros(()))
        self.decay = nn.Parameter(torch.tensor(decay), requires_grad=decay_learnable)

    def forward(self, mask: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values with the last observed value.

        .. math:: (m, yₖ, *) ⟼ ⟦m ? yₖ : \tilde{y}ₖ₋₁⟧

        Args:
            mask (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        # convex combination of last value and current value
        z = torch.where(mask, y, self.last_value)
        c = self.decay
        self.last_value = c * self.last_value + (1 - c) * z
        return self.last_value
