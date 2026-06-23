r"""Implementation of GRU-D model for time series forecasting."""

__all__ = [
    "GRU_D",
    "GRU_DCell",
    "DiagonalLinear",
]

from typing import Final

import torch
from torch import Tensor, nan, nn


class GRU_DCell(nn.Module):
    r"""Modified GRU cell for GRU-D."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reset_gate = nn.Linear(2 * input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(2 * input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(2 * input_size + hidden_size, hidden_size)

    def forward(self, x_hat: Tensor, h_hat: Tensor, m: Tensor, /) -> Tensor:
        # GRU update (eq 13-16)
        # rₖ = σ(Wᵣ x̂ₖ + Uᵣĥₖ₋₁ + Vᵣmₖ + bᵣ)              (13)
        # zₖ = σ(W_z x̂ₖ + U_z ĥₖ₋₁ + V_z mₖ + b_z)        (14)
        # h̃ₖ = tanh(Wx̂ₖ + U (rₖ ⊙ ĥₖ₋₁) + Vmₖ + b)        (15)
        # hₖ = (1 − zₖ) ⊙ ĥₖ₋₁ + zₖ ⊙ h̃ₖ                  (16)
        m = m.to(dtype=x_hat.dtype)  # convert bool to float
        u = torch.cat([x_hat, h_hat, m], dim=-1)
        r = torch.sigmoid(self.reset_gate(u))
        z = torch.sigmoid(self.update_gate(u))
        v = torch.cat([x_hat, r * h_hat, m], dim=-1)
        h_tilde = torch.tanh(self.output_gate(v))
        return (1 - z) * h_hat + z * h_tilde


class DiagonalLinear(nn.Module):
    r"""Diagonal linear transformation."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features))
        self.bias = nn.Parameter(torch.zeros(in_features))

    def forward(self, arg: Tensor, /) -> Tensor:
        # diag(W)x + b = w⊙x + b
        return arg * self.weight + self.bias


class GRU_D(nn.Module):
    r"""GRU-D model for time series forecasting with missing values.

    Processes a combined sequence of context and query time points sorted in
    non-decreasing order.  At each context step the state is updated with the
    GRU-D imputation rule; at each query step the decoder produces predictions.

    GRU-D equations (Che et al., 2018):
        delta_k per feature: reset if observed, accumulate otherwise
        gamma_x = exp(-max(0, W_gamma_x * delta + b)),  W_gamma_x diagonal
        gamma_h = exp(-max(0, W_gamma_h * delta + b))
        x_hat = m * x + (1-m) * (gamma_x * x' + (1-gamma_x) * x_tilde)
        h = GRU(x_hat, gamma_h * h, m)

    Reference:
        - | Recurrent Neural Networks for Multivariate Time Series with Missing Values
          | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag & Yan Liu
          | Nature Scientific Reports
          | https://www.nature.com/articles/s41598-018-24271-9
    """

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    batch_first: Final[bool]

    empirical_mean: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        empirical_mean: Tensor,
        output_size: int | None = None,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size if output_size is None else output_size
        self.batch_first = batch_first

        self.register_buffer("empirical_mean", empirical_mean)
        assert self.empirical_mean.shape == (self.input_size,)

        self.gamma_x_linear = DiagonalLinear(self.input_size)
        self.gamma_h_linear = nn.Linear(self.input_size, self.hidden_size)

        self.h0 = nn.Parameter(torch.zeros(self.hidden_size))
        self.gru_d_cell = GRU_DCell(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        times: Tensor,  # (..., $N + $K), possibly with trailing NaNs (padding)
        context_values: Tensor,  # (..., $N + $K, D), possibly with NaNs
        context_mask: Tensor,  # (..., $N + $K, D), bool
        query_mask: Tensor,  # (..., $N + $K, F), bool
        *,
        initial_state: Tensor | None = None,  # (..., H)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> Tensor:  # (..., $N + $K, F)
        r"""Filter and forecast over combined context/query time points."""
        # optional: sanitize values
        context_values = context_values.masked_fill(~context_mask, nan)

        # initialize mask over valid time steps
        valid_steps = times.isfinite() & (context_mask | query_mask).any(dim=-1)
        result_mask = query_mask

        if self.batch_first:
            times = times.moveaxis(-1, 0)
            context_values = context_values.moveaxis(-2, 0)
            context_mask = context_mask.moveaxis(-2, 0)
            query_mask = query_mask.moveaxis(-2, 0)
            valid_steps = valid_steps.moveaxis(-1, 0)

        num_steps, *batch_shape = times.shape
        assert context_values.shape == (num_steps, *batch_shape, self.input_size)
        assert context_mask.shape == (num_steps, *batch_shape, self.input_size)
        assert query_mask.shape == (num_steps, *batch_shape, self.input_size)
        assert valid_steps.shape == (num_steps, *batch_shape)

        # get initial state
        t = times[0] if initial_time is None else initial_time  # (...)
        h = (  # (..., H)
            self.h0.expand(*batch_shape, self.hidden_size)
            if initial_state is None
            else initial_state.expand(*batch_shape, self.hidden_size)
        )
        x = self.empirical_mean.expand(*batch_shape, self.input_size)  # (..., D)
        delta = times.new_zeros(*batch_shape, self.input_size)  # (..., D)

        predictions_list: list[Tensor] = []

        for t_obs, ctx_vals, ctx_mask, active in zip(
            times,
            context_values,
            context_mask,
            valid_steps,
            strict=True,
        ):
            inc = torch.where(active, t_obs - t, torch.zeros_like(t_obs)).unsqueeze(
                -1
            )  # (..., 1)
            t = torch.where(active, t_obs, t)  # (...)

            # per-feature delta: reset if observed, accumulate if not; unchanged if inactive
            delta = torch.where(  # (..., D)
                active[..., None],
                torch.where(ctx_mask, inc, delta + inc),
                delta,
            )

            # compute decay values γₜ, γₕ
            gamma_x = torch.exp(-torch.relu(self.gamma_x_linear(delta)))  # (..., D)
            gamma_h = torch.exp(-torch.relu(self.gamma_h_linear(delta)))  # (..., H)

            x = torch.where(active[..., None] & ctx_mask, ctx_vals, x)  # (..., D)
            x_hat = torch.where(  # (..., D)
                ctx_mask,
                ctx_vals,
                gamma_x * x + (1 - gamma_x) * self.empirical_mean,
            )
            h_hat = gamma_h * h  # (..., H)
            h_candidate = self.gru_d_cell(x_hat, h_hat, ctx_mask)  # (..., H)
            h = torch.where(active[..., None], h_candidate, h)  # (..., H)

            # make prediction
            y = self.decoder(h)

            predictions_list.append(y)  # (..., F)

        stack_dim = -2 if self.batch_first else 0
        result = torch.stack(predictions_list, dim=stack_dim)  # (..., N+K, F)
        return result.masked_fill(~result_mask, nan)
