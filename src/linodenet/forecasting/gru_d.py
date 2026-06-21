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
        *batch_shape, _ = times.shape

        mask = values.isfinite()  # (..., $N, D), observed values
        valid_times = times.isfinite().unsqueeze(-1)  # (..., $N, 1)
        times = times.nan_to_num(0.0)
        t0 = times[..., [0]]
        # ∆tₖ = tₖ - tₖ₋₁; ∆t₁ = 0
        increments = times.diff(dim=-1, prepend=t0).unsqueeze(-1)  # (..., $N, 1)

        # δ₀ = 0
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
        for active, inc, m, x_obs in zip(
            valid_times.unbind(-2),
            increments.unbind(-2),
            mask.unbind(-2),
            values.unbind(-2),
            strict=True,
        ):
            # δₖ = ⟦mₖ=1 ?  ∆tₖ : ∆tₖ + δₖ₋₁⟧;
            delta = torch.where(active, torch.where(m, inc, delta + inc), delta)  # (..., D)  # fmt: skip
            # last observation (x_{t'})
            x = torch.where(active, torch.where(m, x_obs, x), x)  # (..., D)

            # γₜ = exp{ − max(0,Wᵧδₖ + bᵧ)}  (eq 10)
            # for γₓ, W_{γₓ} is diagonal
            # for γₕ, W_{γₕ} is full matrix
            gamma_x = torch.exp(-torch.relu(self.gamma_x_linear(delta)))  # (..., D)
            gamma_h = torch.exp(-torch.relu(self.gamma_h_linear(delta)))  # (..., H)

            # x̂ₜ = ⟦mₖ ? xₖ : γ_{xₖ}xₖ' + (1 − γ_{xₖ})x̃⟧  (eq. 11)
            x_hat = torch.where(  # (..., D)
                m,
                x_obs,
                gamma_x * x + (1 - gamma_x) * self.empirical_mean,
            )

            # ĥₖ₋₁ = γ_{hₖ} ⊙ hₖ₋₁  (eq 12)
            h_hat = gamma_h * h  # (..., H)

            # GRU-D equations (13-16)
            h_candidate = self.gru_d_cell(x_hat, h_hat, m)  # (..., H)
            h = torch.where(active, h_candidate, h)  # (..., H)

        # Roll forward to query times with completely missing observations.
        time_mask = valid_times.squeeze(-1)  # (..., $N)
        last_time_indices = time_mask.sum(dim=-1).clamp_min(1).unsqueeze(-1) - 1
        last_times = torch.take_along_dim(times, last_time_indices, dim=-1)

        query_valid = query_times.isfinite()
        query_times = query_times.nan_to_num(0.0)
        query_increments = query_times.diff(dim=-1, prepend=last_times).unsqueeze(-1)
        missing = mask.new_zeros(*batch_shape, self.input_size)

        h_queries_list: list[Tensor] = []

        for inc in query_increments.unbind(-2):
            delta = delta + inc  # (..., D)

            gamma_x = torch.exp(-torch.relu(self.gamma_x_linear(delta)))  # (..., D)
            gamma_h = torch.exp(-torch.relu(self.gamma_h_linear(delta)))  # (..., H)
            x_hat = gamma_x * x + (1 - gamma_x) * self.empirical_mean  # (..., D)
            h_hat = gamma_h * h  # (..., H)

            h = self.gru_d_cell(x_hat, h_hat, missing)  # (..., H)
            h_queries_list.append(h)

        h_query = torch.stack(h_queries_list, dim=-2)  # (..., $K, H)
        prediction = self.decoder(h_query)  # (..., $K, O)
        return prediction.masked_fill(~query_valid.unsqueeze(-1), torch.nan)
