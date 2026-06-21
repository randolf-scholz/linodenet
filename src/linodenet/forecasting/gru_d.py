r"""Implementation of GRU-D model for time series forecasting.

Reference:
    - | Recurrent Neural Networks for Multivariate Time Series with Missing Values
      | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag & Yan Liu
      | Nature Scientific Reports
      | https://www.nature.com/articles/s41598-018-24271-9
"""

__all__ = [
    "GRU_D",
    "GRU_DCell",
    "DiagonalLinear",
]

import torch
from torch import Tensor, nn


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
        # rₖ = σ(Wᵣ x̂ₖ + Uᵣĥₖ₋₁ + Vᵣmₖ + bᵣ)              (13)
        # zₖ = σ(W_z x̂ₖ + U_z ĥₖ₋₁ + V_z mₖ + b_z)        (14)
        # h̃ₖ = tanh(Wx̂ₖ + U (rₖ ⊙ ĥₖ₋₁) + Vmₖ + b)        (15)
        # hₖ = (1 − zₖ) ⊙ ĥₖ₋₁ + zₖ ⊙ h̃ₖ                  (16)
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
    r"""TODO: Implement GRU-D model for time series forecasting.

    ∆tₖ = tₖ-tₖ₋₁
    mₖ = ⟦xₖ = 𝙽𝙰 ? 0 : 1⟧
    δₖ = ⟦mₖ=1 ?  ∆tₖ : ∆tₖ + δₖ₋₁⟧
    ∆t₁ = 0
    δ₀ = 0

    rₖ = σ(Wᵣ xₖ + Uᵣhₖ₋₁ + bᵣ)  (3)
    zₖ = σ(W_z xₖ + U_z hₖ₋₁ + b_z )(4)tanh(Wx t + U (rt  h t −1) + b)
    h̃ₖ = tanh(Wxₖ + U (rₖ ⊙ h̃ₖ₋₁) + b)(5)
    hₖ = (1 − zₖ)⊙h̃ₖ₋₁ + zₖ⊙h̃ₖ

    GRU-D equations:

    γₜ = exp{ − max(0,Wᵧδₖ + bᵧ)}

    W_{γₓ} is chosen diagonal.

    xₖ' = ⟦mₖ ? xₖ : xₖ₋₁'⟧  (last observation)
    x̃ = emiprical mean over training data.

    x̂ₖ = mₖxₖ + (1 − mₖ) (γ_{xₖ}xₖ' + (1 − γ_{xₖ})x̃ₖ)
       = ⟦mₖ ? xₖ : γ_{xₖ}xₖ' + (1 − γ_{xₖ})x̃⟧  (imputation)

    ĥₖ₋₁ = γ_{hₖ} ⊙ hₖ₋₁

    rₖ = σ(Wᵣ x̂ₖ + Uᵣĥₖ₋₁ + Vᵣmₖ + bᵣ)          (13)
    zₖ = σ(W_z x̂ₖ + U_z ĥₖ₋₁ + V_z mₖ + b_z)        (14)
    h̃ₖ = tanh(Wx̂ₖ + U (rₖ ⊙ ĥₖ₋₁) + Vmₖ + b)(15)
    hₖ = (1 − zₖ) ⊙ ĥₖ₋₁ + zₖ ⊙ h̃ₖ  (16)

    GRU-mean:  xₖ ← ⟦mₖ ? xₖ : x̃⟧
    GRU-forward: xₖ ← ⟦mₖ ? xₖ : xₖ'⟧
    GRU-simple: xₖ ← [uₖ, mₖ, δₖ]  (uₖ = GRU-mean or GRU-forward)
    GRU-D: x̂ₖ ← ⟦mₖ ? xₖ : γ_{xₖ}xₖ' + (1 − γ_{xₖ})x̃⟧  (imputation)
    """

    empirical_mean: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        empirical_mean: Tensor,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size if output_size is None else output_size

        self.register_buffer("empirical_mean", empirical_mean)
        assert self.empirical_mean.shape == (self.input_size,)

        self.gamma_x_linear = DiagonalLinear(self.input_size)
        self.gamma_h_linear = nn.Linear(self.input_size, self.hidden_size)

        self.h0 = nn.Parameter(torch.zeros(self.hidden_size))
        self.gru_d_cell = GRU_DCell(self.input_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        times: Tensor,  # (..., $N)
        values: Tensor,  # (..., $N, D)
        query_times: Tensor,  # (..., $K)
        *,
        h0: Tensor | None = None,  # (..., H)
    ) -> Tensor:
        mask = values.isfinite()  # (..., $N, D), observed values
        # ∆tₖ = tₖ - tₖ₋₁; ∆t₁ = 0
        t0 = times[..., [0]]
        increments = times.diff(dim=-1, prepend=t0)

        # δ₀ = 0
        *batch_shape, _ = times.shape
        delta = times.new_zeros(*batch_shape, self.input_size)

        # initialize h0
        h = (
            self.h0.expand(*batch_shape, self.hidden_size)
            if h0 is None
            else h0.expand(*batch_shape, self.hidden_size)
        )
        # initialize x0 = x̃
        x = self.empirical_mean.expand(*batch_shape, self.input_size)

        # iterate over observations
        for m, x_obs, inc in zip(
            mask.unbind(-2),
            values.unbind(-2),
            increments.unbind(-1),
            strict=True,
        ):
            # δₖ = ⟦mₖ=1 ?  ∆tₖ : ∆tₖ + δₖ₋₁⟧;
            step = inc.unsqueeze(-1)
            delta = torch.where(m, step, delta + step)
            # last observation (x_{t'})
            x = torch.where(m, x_obs, x)

            # γₜ = exp{ − max(0,Wᵧδₖ + bᵧ)}  (eq 10)
            # for γₓ, W_{γₓ} is diagonal
            # for γₕ, W_{γₕ} is full matrix
            gamma_x = torch.exp(-torch.relu(self.gamma_x_linear(delta)))
            gamma_h = torch.exp(-torch.relu(self.gamma_h_linear(delta)))

            # x̂ₜ = ⟦mₖ ? xₖ : γ_{xₖ}xₖ' + (1 − γ_{xₖ})x̃⟧  (eq. 11)
            x_hat = torch.where(
                m,
                x_obs,
                gamma_x * x + (1 - gamma_x) * self.empirical_mean,
            )

            # ĥₖ₋₁ = γ_{hₖ} ⊙ hₖ₋₁  (eq 12)
            h_hat = gamma_h * h

            # GRU-D equations (13-16)
            h = self.gru_d_cell(x_hat, h_hat, m.to(x_hat))

        # Roll forward to query times with completely missing observations.
        query_increments = query_times.diff(dim=-1, prepend=times[..., [-1]])
        missing = torch.zeros(
            *batch_shape, self.input_size, dtype=torch.bool, device=values.device
        )
        h_queries: list[Tensor] = []

        for inc in query_increments.unbind(-1):
            step = inc.unsqueeze(-1)
            delta = delta + step

            gamma_x = torch.exp(-torch.relu(self.gamma_x_linear(delta)))
            gamma_h = torch.exp(-torch.relu(self.gamma_h_linear(delta)))
            x_hat = gamma_x * x + (1 - gamma_x) * self.empirical_mean
            h_hat = gamma_h * h

            h = self.gru_d_cell(x_hat, h_hat, missing.to(x_hat))
            h_queries.append(h)

        query_states = torch.stack(h_queries, dim=-2)  # (..., $K, H)
        return self.decoder(query_states)
