r"""Implementation of GRU-D model for time series forecasting.

Reference:
    - | Recurrent Neural Networks for Multivariate Time Series with Missing Values
      | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag & Yan Liu
      | Nature Scientific Reports
      | https://www.nature.com/articles/s41598-018-24271-9
"""

__all__ = ["GRU_D"]

import torch
from torch import Tensor, nn


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

    def __init__(
        self, input_size, hidden_size, output_size, empirical_mean: Tensor
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.register_buffer("empirical_mean", None)
        self.empirical_mean = empirical_mean  # (D, )
        assert empirical_mean.shape == (self.input_size,)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        m = ~x.isnan()  # observed mask
        # ∆tₖ = tₖ - tₖ₋₁; ∆t₁ = 0
        increments = t.diff(prepend=torch.zeros_like(t[..., 0]))
        # δₖ = ⟦mₖ=1 ?  ∆tₖ : ∆tₖ + δₖ₋₁⟧; δ₀ = 0
        raise NotImplementedError
