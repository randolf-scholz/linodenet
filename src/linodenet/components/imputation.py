r"""Imputers for missing data handling."""

__all__ = [
    # constants
    "IMPUTERS",
    # types
    "ImputerProtocol",
    "ImputationStrategy",
    # classes
    "ZeroImputer",
    "ConstantValueImputer",
    "LearnableValueImputer",
    "LastValueImputer",
    "LinearImputer",
    # functions
    "zero_impute",
]


from enum import StrEnum
from typing import Protocol

import torch
from torch import Tensor, nn


class ImputerProtocol(Protocol):
    r"""Protocol for imputer."""

    def __call__(self, m: Tensor, y: Tensor, x: Tensor) -> Tensor: ...


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


def zero_impute(m: Tensor, y: Tensor, _: Tensor) -> Tensor:
    r"""Impute missing values with a constant.

    Args:
        m (Tensor): Mask tensor (true if observed)
        y (Tensor): Observed state.
        _ (Tensor): Hidden state.
    """
    return torch.where(m, y, torch.zeros_like(y))


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


class ConstantValueImputer(nn.Module):
    r"""Impute missing values with a constant."""

    value: Tensor
    r"""Constant value to impute missing values."""

    def __init__(self, constant: float | Tensor, /) -> None:
        super().__init__()
        tensor = torch.tensor(constant)
        self.value = nn.Parameter(tensor, requires_grad=False)

    def forward(self, m: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values with a constant.

        .. math:: (m, y, *) ⟼ ⟦m ? y : c⟧

        Args:
            m (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        return torch.where(m, y, self.value)


class LearnableValueImputer(nn.Module):
    r"""Impute missing values in a tensor."""

    value: Tensor
    r"""Impute missing values in a tensor."""

    def __init__(self, shape: tuple[int, ...], /) -> None:
        super().__init__()
        tensor = torch.randn(shape)
        self.value = nn.Parameter(tensor, requires_grad=True)

    def forward(self, m: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values in a tensor.

        .. math:: (m, y, *) ⟼ ⟦m ? y : c⟧

        Args:
            m (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        return torch.where(m, y, self.value)


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

    def forward(self, m: Tensor, y: Tensor, _: Tensor) -> Tensor:
        r"""Impute missing values with the last observed value.

        .. math:: (m, yₖ, *) ⟼ ⟦m ? yₖ : \tilde{y}ₖ₋₁⟧

        Args:
            m (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        # convex combination of last value and current value
        z = torch.where(m, y, self.last_value)
        c = self.decay
        self.last_value = c * self.last_value + (1 - c) * z
        return self.last_value


class LinearImputer(nn.Module):
    r"""Impute missing values with a linear function of the hidden state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, m: Tensor, y: Tensor, x: Tensor) -> Tensor:
        r"""Impute missing values with a linear function.

        .. math:: (m, y, x) ⟼ ⟦m ? y : Hx⟧

        Args:
            m (Tensor): Mask tensor (true if observed)
            y (Tensor): Observed state.
            x (Tensor): Hidden state.
        """
        return torch.where(m, y, self.linear(x))


IMPUTERS: dict[str, type[ImputerProtocol]] = {
    "ZeroImputer"           : ZeroImputer,
    "ConstantValueImputer"  : ConstantValueImputer,
    "LearnableValueImputer" : LearnableValueImputer,
    "LastValueImputer"      : LastValueImputer,
    "LinearImputer"         : LinearImputer,
}  # fmt: skip
r"""Dictionary of available imputers."""
