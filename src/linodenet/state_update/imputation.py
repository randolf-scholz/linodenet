r"""Base imputer classes and functions."""

__all__ = [
    # constants
    "IMPUTERS",
    # types
    "ImputerProtocol",
    "ImputationStrategy",
    # classes
    "ZeroImputer",
    "ConstantImputer",
    "CorrelationImputer",
    "LearnableImputer",
    "LastValueImputer",
    "LinearImputer",
]

from enum import StrEnum
from typing import Final, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from signatures import signature


class ImputerProtocol(Protocol):
    r"""Protocol for imputer."""

    @signature("[(..., m), (..., n)] -> [(..., m), (..., m)]")
    def __call__(self, y_obs: Tensor, x: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Impute missing values in a tensor.

        Args:
            y_obs: Observed state.
            x: Estimated state.

        Returns:
            y: Imputed state, where missing values have been replaced with imputed values.
            m: Mask indicating which values were observed (True) vs imputed (False).
        """
        ...


class LinearImputer(nn.Module):
    r"""Impute missing values with a linear function of the hidden state."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, input_size, bias=use_bias)

    def forward(self, y: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with a linear function.

        .. math:: (y, x) ⟼ (u, m), \quad u = ⟦m ? y : Hx+b⟧

        Args:
            y: Observed state.
            x: Estimated state.
        """
        mask = ~y.isnan()
        return torch.where(mask, y, self.linear(x)), mask


class ZeroImputer(nn.Module):
    r"""Impute missing values with zero."""

    def forward(self, y: Tensor, _: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with zero.

        .. math:: (y, *) ⟼ (u, m), \quad u = ⟦m ? y : 0⟧

        Args:
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        mask = ~y.isnan()
        return torch.where(mask, y, torch.zeros_like(y)), mask


class ConstantImputer(nn.Module):
    r"""Impute missing values with a constant."""

    value: Tensor
    r"""Constant value to impute missing values."""

    def __init__(self, constant: float | Tensor, /) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(constant))

    def forward(self, y: Tensor, _: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with a constant.

        .. math:: (y, *) ⟼ (u, m), \quad u = ⟦m ? y : c⟧

        Args:
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        mask = ~y.isnan()
        return torch.where(mask, y, self.value), mask


class CorrelationImputer(nn.Module):
    r"""Fuse observations and decoder predictions via Gaussian conditioning."""

    decoder: nn.Module
    r"""Decoder used to predict the observation mean from the hidden state."""
    input_size: Final[int]
    r"""CONST: Number of observed features."""
    covariance_factor: Tensor
    r"""PARAM: Unconstrained lower-triangular covariance factor parameters."""
    eye: Tensor
    r"""BUFFER: Diagonal mask used to enforce a positive Cholesky diagonal."""
    identity: Tensor
    r"""BUFFER: Identity matrix used for masked linear solves."""

    def __init__(self, *, decoder: nn.Module) -> None:
        super().__init__()
        input_size = int(getattr(decoder, "output_size", -1))
        if input_size <= 0:
            raise ValueError(
                "CorrelationImputer requires decoder.output_size to be a positive integer."
            )

        self.decoder = decoder
        self.input_size = input_size
        self.covariance_factor = nn.Parameter(torch.eye(input_size))
        self.register_buffer("eye", torch.eye(input_size, dtype=torch.bool))
        self.register_buffer("identity", torch.eye(input_size))

    def get_cholesky(self) -> Tensor:
        r"""Return a valid Cholesky factor for the learned covariance."""
        lower = self.covariance_factor.tril()
        diag = lower.diagonal(dim1=-2, dim2=-1)
        positive_diag = F.softplus(diag) + 1e-6
        return torch.where(self.eye, positive_diag.unsqueeze(-1), lower)

    def get_covariance(self) -> Tensor:
        r"""Return the covariance matrix $Σ = LLᵀ$."""
        L = self.get_cholesky()
        return L @ L.mT

    def forward(self, y: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with the conditional Gaussian posterior mean.

        Let $ŷ = decoder(x)$ and assume the true observation follows
        $y ∼ 𝓝(ŷ, Σ)$. Conditioning on the observed coordinates selected by the
        mask $Π$ gives the full-vector update

        .. math:: y' = ŷ - Σ (ΠΣΠ + 𝕀 - Π)⁻¹ Π(ŷ - y)

        which is equivalent to

        .. math:: y'ₘ = ŷₘ + Σₘₒ Σₒₒ⁻¹ (yₒ - ŷₒ)

        Args:
            y: Observed state, possibly containing NaNs at missing coordinates.
            x: Hidden state used by the decoder.
        """
        mask = ~y.isnan()
        y_hat = self.decoder(x)
        if y_hat.shape[-1] != self.input_size:
            raise ValueError(
                f"Decoder output shape mismatch: expected last dimension "
                f"{self.input_size}, got {y_hat.shape[-1]}."
            )

        covariance = self.get_covariance()
        residual = torch.where(mask, y_hat - y, y_hat.new_zeros(()))

        observed = torch.einsum("...i, ...j -> ...ij", mask, mask)
        innovation_covariance = torch.where(observed, covariance, self.identity)

        L = torch.linalg.cholesky(innovation_covariance)
        z = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
        return y_hat - torch.einsum("ij, ...j -> ...i", covariance, z), mask


class LearnableImputer(nn.Module):
    r"""Impute missing values with a learnable value."""

    input_shape: Final[tuple[int, ...]]
    r"""CONST: The shape of the learnable imputation value."""
    value: Tensor
    r"""PARAM: Learnable value used to impute missing observations."""

    def __init__(self, input_shape: int | tuple[int, ...], /) -> None:
        super().__init__()
        self.input_shape = (
            (input_shape,) if isinstance(input_shape, int) else input_shape
        )
        self.value = nn.Parameter(torch.randn(self.input_shape))

    def forward(self, y: Tensor, _: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with a learnable value.

        .. math:: (y, *) ⟼ (u, m), \quad u = ⟦m ? y : c⟧

        Args:
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        mask = ~y.isnan()
        return torch.where(mask, y, self.value), mask


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

    def forward(self, y: Tensor, _: Tensor) -> tuple[Tensor, Tensor]:
        r"""Impute missing values with the last observed value.

        .. math:: (yₖ, *) ⟼ (u, m), \quad u = ⟦m ? yₖ : \tilde{y}ₖ₋₁⟧

        Args:
            y (Tensor): Observed state.
            _ (Tensor): Hidden state.
        """
        mask = ~y.isnan()
        # convex combination of last value and current value
        z = torch.where(mask, y, self.last_value)
        self.last_value = self.decay * self.last_value + (1 - self.decay) * z
        return self.last_value, mask


IMPUTERS: dict[str, type[ImputerProtocol]] = {
    "ZeroImputer"      : ZeroImputer,
    "ConstantImputer"  : ConstantImputer,
    "CorrelationImputer": CorrelationImputer,
    "LearnableImputer" : LearnableImputer,
    "LastValueImputer" : LastValueImputer,
    "LinearImputer"    : LinearImputer,
}  # fmt: skip


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
