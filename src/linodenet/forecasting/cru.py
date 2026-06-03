r"""Reimplementation of the Continuous Recurrent Unit (CRU)."""

__all__ = ["CRU"]

from typing import Final

import torch
from torch import Tensor, nn
from torch.distributions import Distribution


class ELU1P(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x < 0.0, x.exp(), x + 1.0)


def get_activation(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "elu":
            return nn.ELU()
        case "elup1":
            return ELU1P()
        case "exp":
            return ...
        case "square":
            return ...
        case "abs":
            return ...
        case _:
            raise NotImplementedError


class Encoder(nn.Module):
    r"""Returns $yₜ, σₜ^{obs} = f_θ(xₜ)$."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        num_hidden_layers: int = 2,
        activation_function: str = "relu",
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        if activation_function != "relu":
            raise NotImplementedError
        if variance_activation != "elup1":
            raise NotImplementedError

        self.hidden_layers = self._build_hidden_layers()
        self.mean_model = nn.Linear(hidden_size, output_size)
        self.variance_model = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            ELU1P(),
        )

    def _build_hidden_layers(self) -> nn.Module:
        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.hidden_layers(x)
        h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)
        return self.mean_model(h), self.variance_model(h)


class Decoder(nn.Module):
    r"""Returns $oₜ, σₜ^{out} = g_ϕ(μₜ⁺, Σₜ⁺)$."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        *,
        num_hidden_mean_model_layers: int = 2,
        num_hidden_variance_model_layers: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_mean_model_layers = num_hidden_mean_model_layers
        self.num_hidden_variance_model_layers = num_hidden_variance_model_layers

        self.mean_model = self._build_mean_model()
        self.variance_model = self._build_variance_model()

    def _build_mean_model(self) -> nn.Module:
        hidden_layers = []
        for _ in range(self.num_hidden_mean_model_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(2 * self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
        )

    def _build_variance_model(self) -> nn.Module:
        hidden_layers = []
        for _ in range(self.num_hidden_variance_model_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(3 * self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
            ELU1P(),
        )

    def forward(self, mean: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        return self.mean_model(mean), self.variance_model(covariance)


class CRU(nn.Module):
    r"""Continuous Recurrent Unit for probabilistic forecasting.

    The basic setup is a latent linear SDE with Gaussian observations.

    .. math:: dz = Azdt + Gdβ   \qquad   yₜ∼𝓝(Hzₜ, σₜ²𝕀)

    This is combined with an encoder decoder setup:

    .. math::
        μ_{t₀}⁺, Σ_{t₀}⁺ &= 0, σ₀⋅𝕀                \\
        yₜ, σₜ^{obs} &= f_θ(xₜ)                    \\
        μₜ⁻, Σₜ⁻ &= predict(μₛ⁺, Σₛ⁺, t - s)       \\
        μₜ⁺, Σₜ⁺ &= update(μₜ⁻, Σₜ⁻, yₜ, σₜ^{obs}) \\
        oₜ, σₜ^{out} &= g_ϕ(μₜ⁺, Σₜ⁺)              \\

    Where:
        - f_θ, the encoder, is a neural network
        - g_ϕ, the decoder, is a neural network
        - predict is performed via matrix exponential solution to the linear SDE
        - update is the regular kalman update

    Additionally, CRU makes the following simplifying assumptions:
        - latent space is twice the dimension of the latent observation space
        - trivial observation model: H = [𝕀_d, 𝟎_d]
        - block-wise diagonal covariance matrix: Σₜ = [[Σₜᵘ, Σₜˢ], [Σₜˢ, Σₜˡ]],
          where each block is a diagonal matrix Σₜᵘ=σₜᵘ𝕀, Σₜˢ=σₜˢ𝕀, Σₜˡ=σₜˡ𝕀
        - Hence, the kalman gain takes the form Kₜ=[Kₜᵘ; Kₜˡ], where
          - Kₜᵘ = diag(σₜᵘ / (σₜᵘ + σₜ^{obs}))
          - Kₜˡ = diag(σₜˢ / (σₜᵘ + σₜ^{obs}))
    """

    # Constants
    input_size: Final[int]
    r"""CONST: Dimensionality of observed context values."""
    output_size: Final[int]
    r"""CONST: Dimensionality of forecast targets."""
    latent_size: Final[int]
    r"""CONST: Dimensionality of the full latent Gaussian state."""
    latent_observation_size: Final[int]
    r"""CONST: Dimensionality of encoded latent observations."""
    batch_first: Final[bool]
    r"""CONST: Whether sequence tensors use shape ``batch × time × dim``."""
    validate_args: Final[bool]
    r"""CONST: Whether forward inputs should be validated before computation."""
    initial_variance: Final[float]
    r"""CONST: Initial diagonal covariance value for latent states."""
    variance_floor: Final[float]
    r"""CONST: Minimum variance used for numerical stability."""

    # Parameters
    initial_mean: Tensor
    r"""PARAM: Mean of the initial latent state."""
    initial_log_variance: Tensor
    r"""PARAM: Unconstrained diagonal variance of the initial latent state."""

    # Submodules
    encoder: nn.Module
    r"""MODULE: Maps observations to latent-observation Gaussian parameters."""
    transition: nn.Module
    r"""MODULE: Propagates latent Gaussian states through continuous time."""
    decoder: nn.Module
    r"""MODULE: Maps latent Gaussian states to predictive distributions."""

    # buffers
    block_banded_mask: Tensor
    r"""BUFFER: Block banded mask for transition matrix."""

    @property
    def config(self) -> dict[str, object]:
        r"""Return constructor-relevant configuration."""
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "latent_size": self.latent_size,
            "batch_first": self.batch_first,
            "validate_args": self.validate_args,
            "initial_variance": self.initial_variance,
            "variance_floor": self.variance_floor,
        }

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_units: int = 50,  # number of hidden units for the encoder and decoder
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        output_size: int | None = None,
        batch_first: bool = True,
        validate_args: bool = False,
        num_basis: int = 15,  # number of basis matrices for the transition model
        bandwidth: int = 3,  # bandwidth of the blocks of the transition matrix
        initial_variance: float = 10.0,
        variance_floor: float = 1e-6,
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        if latent_size % 2:
            raise ValueError("latent_size must be even.")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive.")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive.")
        if variance_activation != "elup1":
            raise NotImplementedError

        self.variance_activation = ELU1P()

        # The transition matrix A is parametrized as a linear combination ∑ₖαₖ(μₜ)Aₖ
        # where Aₖ = [[B₁₁, B₁₂], [B₂₁, B₂₂]] and each block is a d×d banded matrix of bandwidth b.
        # The number of parameters of a banded d×d matrix with bandwidth b is:
        #   d + 2*(T_{d-1} - T_{d-b-1}), where Tₙ is the n-th triangle number
        assert bandwidth >= 0, "bandwidth must be non-negative"
        assert bandwidth < latent_size, "bandwidth must be smaller than latent_size."
        T = lambda n: n * (n + 1) // 2
        num_params = latent_size + 2 * (
            T(latent_size - 1) - T(latent_size - bandwidth - 1)
        )
        self.transition_matrix_parameters = nn.Parameter(
            torch.zeros(num_basis, 4, num_params)
        )

        # create a mask for the transition matrix model
        band_mask = (
            torch.ones((latent_size, latent_size)).triu(-bandwidth).tril(bandwidth)
        )
        block_banded_mask = torch.cat([  # [[B, B], [B, B]]
            torch.cat([band_mask, band_mask], dim=-1),
            torch.cat([band_mask, band_mask], dim=-1),
        ], dim=-2)  # fmt: skip
        self.register_buffer("block_banded_mask", block_banded_mask)

        self.input_size = input_size
        self.output_size = input_size if output_size is None else output_size
        self.latent_size = latent_size
        self.latent_observation_size = latent_size // 2
        self.batch_first = batch_first
        self.validate_args = validate_args
        self.initial_variance = initial_variance
        self.variance_floor = variance_floor

        initial_log_variance = torch.log(torch.expm1(torch.tensor(initial_variance)))
        self.initial_mean = nn.Parameter(torch.zeros(latent_size))
        self.initial_log_variance = nn.Parameter(
            initial_log_variance.expand(latent_size).clone()
        )

    def forward(
        self,
        query_times: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        *,
        context_mask: Tensor | None = None,
    ) -> Distribution:
        r"""Return the predictive distribution at ``query_times``.

        Args:
            query_times: Times at which forecasts are requested.
            context_times: Times of the observed context sequence.
            context_values: Observed context values.
            context_mask: Boolean mask indicating observed entries in ``context_values``.

        Returns:
            Predictive distribution over target values at ``query_times``.
        """
        raise NotImplementedError

    def validate_inputs(
        self,
        query_times: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        *,
        context_mask: Tensor | None = None,
    ) -> None:
        r"""Validate public forecasting inputs."""
        raise NotImplementedError

    def encode_observations(
        self,
        observations: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Encode observations as latent-observation means and variances."""
        raise NotImplementedError

    def initial_state(
        self,
        batch_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Return initial latent mean and diagonal covariance."""
        raise NotImplementedError

    def filter_context(
        self,
        context_times: Tensor,
        context_values: Tensor,
        *,
        context_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Assimilate context observations into a latent posterior state."""
        raise NotImplementedError

    def update_state(
        self,
        prior_mean: Tensor,  # (..., 2d)
        prior_variance: tuple[Tensor, Tensor, Tensor],  # (..., d), (..., d), (..., d)
        observation_mean: Tensor,  # (..., d)
        observation_variance: Tensor,  # (..., d)
        *,
        observation_mask: Tensor,  # (...,)
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:  # (..., 2d), (σᵘ, σˡ, σˢ)
        r"""Apply the CRU/Kalman measurement update for one time step."""
        # assumptions:
        # H = [𝕀_d, 0_d]
        # Σₜ = [[Σₜᵘ, Σₜˢ], [Σₜˢ, Σₜˡ]], where
        #   Σₜᵘ=diag(σₜᵘ), Σₜˢ=diag(σₜˢ), Σₜˡ=diag(σₜˡ)
        # Note: the paper explicitly uses σ for the variance rather than stdv.
        # kalman gain: Kₜ = Σₜ⁻Hᵀ(HΣₜ⁻Hᵀ + Rₜ)⁻¹
        # compute the simplified kalman gain Kₜ=[Kₜᵘ; Kₜˡ], where
        #   - Kₜᵘ = diag(σₜᵘ / (σₜᵘ + σₜ^{obs}))
        #   - Kₜˡ = diag(σₜˢ / (σₜᵘ + σₜ^{obs}))
        d = observation_mean.shape[-1]
        mask = observation_mask.unsqueeze(-1)  # (..., 1)
        var_u, var_l, var_s = prior_variance
        denominator = var_u + observation_variance
        gain_u = var_u / denominator
        gain_l = var_s / denominator
        # μₜ⁺ = μₜ⁻ + Kₜ(yₜ - Hμₜ⁻), using H = [𝕀_d, 𝟎_d]
        residual = observation_mean - prior_mean[..., :d]
        post_mean = prior_mean + torch.where(
            mask,
            torch.cat([gain_u * residual, gain_l * residual], dim=-1),
            0.0,
        )
        # Σₜ⁺ = (I - KₜH)Σₜ⁻
        post_cov = (
            var_u - torch.where(mask, gain_u * var_u, 0.0),  # (1-Kᵘ)σᵘ
            var_l - torch.where(mask, gain_l * var_s, 0.0),  # σˡ - Kˡσˢ
            var_s - torch.where(mask, gain_u * var_s, 0.0),  # (1-Kᵘ)σˢ
        )

        # validation that resulting covariance is positive definite
        if __debug__:
            assert (var_u > 0).all()
            assert (var_l > 0).all()
            assert (var_s > 0).all()
            assert (var_u * var_l > var_s**2).all()

        return post_mean, post_cov

    def get_transition_model(self, mean):
        """Locally linear transition model.

        Q = diag(activation(q))
        Aₜ = ∑ₖαₖ(t)Aₖ, where αₖ(t) = w_ψ(μₜ⁺)
        Here Aₖ = [Aₖ¹¹, Aₖ¹²; Aₖ²¹, Aₖ²²],
        where each block is a band-matrix of bandwidth b.
        """
        Q = torch.diag_embed(self.variance_activation(self.q))
        *batch_shape, n = mean.shape
        alpha = self.transition_coefficient_model(mean)  # (..., k)
        weighted = torch.einsum(
            "...k, knb -> ...nb", alpha, self.transition_matrix_parameters
        )  # (..., 4, p(b))

        # block_banded_mask is (2d, 2d)
        A = self.block_banded_mask.to(dtype=weighted.dtype).expand(*batch_shape, n, n)
        A = A.masked_scatter(self.block_banded_mask, weighted)
        return A, Q

    def predict_state(
        self,
        posterior_mean: Tensor,  # (..., 2d)
        posterior_variance: tuple[Tensor, Tensor, Tensor],  # (...,d), (...,d), (...,d)
        delta_time: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        r"""Propagate a latent posterior through the continuous transition model."""
        # reconstruct Σ from σᵤ, σᵥ, σₛ
        var_u, var_l, var_s = posterior_variance
        cov_u = torch.diag_embed(var_u)
        cov_l = torch.diag_embed(var_l)
        cov_s = torch.diag_embed(var_s)
        cov = torch.cat([
            torch.cat([cov_u, cov_s], dim=-1),
            torch.cat([cov_s, cov_l], dim=-1),
        ], dim=-2)  # fmt: skip

        A, Q = self.get_transition_model()

        # compute van Loan matrix exponential
        n = posterior_mean.shape[-1]
        zero = torch.zeros_like(A)
        M = torch.cat([
            torch.cat([A, Q], dim=-1),
            torch.cat([zero, -A.mT], dim=-1),
        ], dim=-2)  # fmt: skip
        exp_Mt = torch.linalg.matrix_exp(M * delta_time)  # [[F, C], [0, -Fᵀ]]
        exp_At = exp_Mt[..., :n, :n]  # upper left block
        C = exp_Mt[..., :n, n:]  # upper right block

        # μₜ = eᴬᵗμ₀
        # Σₜ = eᴬᵗΣ₀eᴬᵀᵗ + Ceᴬᵀᵗ
        prior_mean = exp_At @ posterior_mean
        prior_cov = (exp_At @ cov + C) @ exp_At.mT  # [Σᵤ, Σₛ; Σₛ, Σˡ]

        # Note: If X is block-wise diagonal, then exp(X) is also block-wise diagonal.
        prior_var_u = prior_cov[..., :n, :n].diagonal(-2, -1)
        prior_var_s = prior_cov[..., :n, n:].diagonal(-2, -1)
        prior_var_l = prior_cov[..., n:, n:].diagonal(-2, -1)

        return prior_mean, (prior_var_u, prior_var_l, prior_var_s)

    def predict_query(
        self,
        query_times: Tensor,
        context_times: Tensor,
        posterior_mean: Tensor,
        posterior_variance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Predict latent Gaussian states at query times."""
        raise NotImplementedError

    def decode_state(
        self,
        latent_mean: Tensor,
        latent_variance: Tensor,
    ) -> Distribution:
        r"""Decode latent Gaussian states into a predictive distribution."""
        raise NotImplementedError
