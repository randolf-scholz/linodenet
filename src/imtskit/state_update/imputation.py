r"""Impute missing observations before a state update.

All imputers use true mask entries for coordinates to replace. If no mask is
provided, missing coordinates are inferred with `y.isnan()`. This convention
differs from the observed-coordinate masks used by state-update losses.
"""

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
    r"""Protocol for imputers using true mask entries for missing coordinates."""

    @signature("[(..., m), (..., n)] -> (..., m)")
    def __call__(
        self, y_obs: Tensor, x: Tensor, /, *, mask: Tensor | None = None
    ) -> Tensor:
        r"""Impute missing values in a tensor.

        Args:
            y_obs: Observed state.
            x: Estimated state.
            mask: Mask indicating which values to impute.
                If `None`, uses `y_obs.isnan()`.

        Returns:
            Imputed state, where the masked values have been replaced.
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

    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing values with a linear prediction from the hidden state.

        $$ u = \where(m, Hx+b, y) $$

        The bias $b$ is zero when `use_bias=False`.

        Args:
            y: Observation vector.
            x: Estimated hidden state.
            mask: Boolean mask with true entries at values to impute.
                If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        return torch.where(mask, self.linear(x), y)


class ZeroImputer(nn.Module):
    r"""Impute missing values with zero."""

    def forward(self, y: Tensor, _: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing values with zero.

        $$ u = \where(m, 0, y) $$

        Args:
            y: Observation vector.
            _: Hidden state, unused.
            mask: Boolean mask with true entries at values to impute.
                If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        return torch.where(mask, 0.0, y)


class ConstantImputer(nn.Module):
    r"""Impute missing values with a constant."""

    value: Tensor
    r"""Constant replacement value for missing observations."""

    def __init__(self, constant: float | Tensor, /) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(constant))

    def forward(self, y: Tensor, _: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing values with a fixed value $c$.

        $$ u = \where(m, c, y) $$

        The replacement value must broadcast to the observation shape.

        Args:
            y: Observation vector.
            _: Hidden state, unused.
            mask: Boolean mask with true entries at values to impute.
                If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        return torch.where(mask, self.value, y)


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

    def forward(self, y: Tensor, x: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing values with the conditional Gaussian posterior mean.

        Let $ŷ = decoder(x)$ and assume $y ∼ 𝓝(ŷ, Σ)$. For a missing-value
        mask $m$, the matrix $Π = 𝕀 - \diag(m)$ selects observed
        coordinates. Conditioning on those coordinates gives the full-vector mean

        $$ y' = ŷ - Σ (ΠΣΠ + 𝕀 - Π)⁻¹ Π(ŷ - y) $$

        whose missing-coordinate component is

        $$ y'ₘ = ŷₘ + Σₘₒ Σₒₒ⁻¹ (yₒ - ŷₒ) $$

        Here subscripts $m$ and $o$ denote missing and observed coordinates,
        respectively. Observed coordinates retain their supplied values; if
        all coordinates are missing, the result is the decoder prediction.
        Residuals at missing coordinates are set to zero before evaluation.

        Args:
            y: Observed state, possibly containing NaNs at missing coordinates.
            x: Hidden state used by the decoder.
            mask: Mask indicating which values to impute. If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        observed_mask = ~mask
        y_hat = self.decoder(x)
        if y_hat.shape[-1] != self.input_size:
            raise ValueError(
                f"Decoder output shape mismatch: expected last dimension "
                f"{self.input_size}, got {y_hat.shape[-1]}."
            )

        covariance = self.get_covariance()
        residual = torch.where(observed_mask, y_hat - y, y_hat.new_zeros(()))

        observed = torch.einsum("...i, ...j -> ...ij", observed_mask, observed_mask)
        innovation_covariance = torch.where(observed, covariance, self.identity)

        L = torch.linalg.cholesky(innovation_covariance)
        z = torch.cholesky_solve(residual.unsqueeze(-1), L).squeeze(-1)
        return y_hat - torch.einsum("ij, ...j -> ...i", covariance, z)


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

    def forward(self, y: Tensor, _: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing values with a learnable value $c$.

        $$ u = \where(m, c, y) $$

        The replacement value must broadcast to the observation shape.

        Args:
            y: Observation vector.
            _: Hidden state, unused.
            mask: Boolean mask with true entries at values to impute.
                If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        return torch.where(mask, self.value, y)


class LastValueImputer(nn.Module):
    r"""Impute missing observations using a stored value with optional smoothing.

    The stored value starts at zero. With the default `decay=0`, observed
    coordinates replace it and missing coordinates retain their previous value.
    A nonzero decay also smooths observed coordinates.
    """

    last_value: Tensor
    r"""Most recent stored value, including any decay smoothing."""
    decay: Tensor
    r"""Learnable or fixed smoothing coefficient; zero retains observations exactly."""

    def __init__(self, *, decay: float = 0.0, decay_learnable: bool = False) -> None:
        super().__init__()
        self.register_buffer("last_value", torch.zeros(()))
        self.decay = nn.Parameter(torch.tensor(decay), requires_grad=decay_learnable)

    def forward(self, y: Tensor, _: Tensor, /, *, mask: Tensor | None = None) -> Tensor:
        r"""Impute missing entries and update the stored value.

        $$\begin{aligned}
            zₖ &= \where(m, sₖ₋₁, yₖ)  \\
            sₖ &= δsₖ₋₁ + (1-δ)zₖ.
        \end{aligned}$$

        Here $δ$ is `decay`; the returned value is $sₖ$. For nonzero decay, this
        smooths observed coordinates as well as retaining missing ones.

        Args:
            y: Observation vector.
            _: Hidden state, unused.
            mask: Boolean mask with true entries at values to impute.
                If `None`, uses `y.isnan()`.
        """
        mask = y.isnan() if mask is None else mask
        # convex combination of last value and current value
        z = torch.where(mask, self.last_value, y)
        self.last_value = self.decay * self.last_value + (1 - self.decay) * z
        return self.last_value


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
