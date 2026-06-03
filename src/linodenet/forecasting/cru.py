r"""Reimplementation of the Continuous Recurrent Unit (CRU)."""

__all__ = ["CRU"]

from typing import Final

import torch
from torch import Tensor, nn
from torch.distributions import Distribution


class ELU1P(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x < 0.0, x.exp(), x + 1.0)


class Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.exp()


class Square(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.square()


class Abs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


def new_activation(name: str) -> nn.Module:
    match name:
        case "relu":
            return nn.ReLU()
        case "elup1":
            return ELU1P()
        case "exp":
            return Exp()
        case "square":
            return Square()
        case "abs":
            return Abs()
        case "tanh":
            return nn.Tanh()
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


type CRU_covariance = tuple[Tensor, Tensor, Tensor]
r"""Parametrized representation (σᵤ, σₗ, σₛ), Σ = [[diag(σᵤ), diag(σₛ)], [diag(σₛ), diag(σₗ)]]"""


class CRU(nn.Module):
    r"""Continuous Recurrent Unit for probabilistic forecasting.

    The basic setup is a latent linear SDE with Gaussian observations.

    .. math:: dz = Azdt + Gdβ   \qquad   yₜ∼𝓝(Hzₜ, σₜ²𝕀)

    This is combined with an encoder decoder setup:

    .. code-block:: text

        μ_{t₀}⁺, Σ_{t₀}⁺ ← 0, σ₀⋅𝕀
        for t ∈ {t₁, …, tₙ}:
            yₜ, σₜ^{obs} ← f_θ(xₜ)
            μₜ⁻, Σₜ⁻ ← predict(μₛ⁺, Σₛ⁺, t - s)
            μₜ⁺, Σₜ⁺ ← update(μₜ⁻, Σₜ⁻, yₜ, σₜ^{obs})
            oₜ, σₜ^{out} ← g_ϕ(μₜ⁺, Σₜ⁺)
            s ← t

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
    variance_floor: Final[float]
    r"""CONST: Minimum variance used for numerical stability."""

    # Submodules
    encoder: nn.Module
    r"""MODULE: Maps observations to latent-observation Gaussian parameters."""
    decoder: nn.Module
    r"""MODULE: Maps latent Gaussian states to predictive distributions."""
    transition_coefficient_model: nn.Module
    r"""MODULE: Maps μₜ to the coefficient vector α(μₜ)."""

    # Buffers
    block_banded_mask: Tensor
    r"""BUFFER: Block banded mask for transition matrix."""
    initial_mean: Tensor
    r"""BUFFER: Initial mean."""
    initial_covariance: Tensor
    r"""BUFFER: Initial covariance."""

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
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        num_basis: int = 15,  # number of basis matrices for the transition model
        bandwidth: int = 3,  # bandwidth of the blocks of the transition matrix
        initial_variance: float = 10.0,
        variance_floor: float = 1e-6,
        variance_activation: str = "elup1",
        validate_args: bool = False,
    ) -> None:
        super().__init__()
        if latent_size % 2:
            raise ValueError("latent_size must be even.")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive.")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive.")

        self.input_size = input_size
        self.latent_size = latent_size
        self.latent_observation_size = latent_size // 2
        self.validate_args = validate_args
        self.initial_variance = initial_variance
        self.variance_floor = variance_floor

        self.variance_activation = new_activation(variance_activation)

        self.encoder = encoder
        self.decoder = decoder

        # The transition matrix A is parametrized as a linear combination ∑ₖαₖ(μₜ)Aₖ
        # where Aₖ = [[B₁₁, B₁₂], [B₂₁, B₂₂]] and each block is a d×d banded matrix of bandwidth b.
        # The number of parameters of a banded d×d matrix with bandwidth b is:
        #   d + 2*(T_{d-1} - T_{d-b-1}), where Tₙ is the n-th triangle number
        assert bandwidth >= 0, "bandwidth must be non-negative"
        assert bandwidth < latent_size, "bandwidth must be smaller than latent_size."
        T = lambda n: n * (n + 1) // 2
        num_params: int = latent_size + 2 * (
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

        # "For all experiments, we used a transition net with one linear layer and
        # softmax output."
        self.transition_coefficient_model = nn.Sequential(
            nn.Linear(self.latent_size, self.num_basis),
            nn.Softmax(dim=-1),
        )

        # NOTE: The reference implementation makes the initial variance trainable.
        # this however is not mentioned in the paper.
        # We use fixed buffers instead
        self.register_buffer("initial_mean", torch.zeros(self.latent_size))
        self.register_buffer(
            "initial_covariance", initial_variance * torch.eye(self.latent_size)
        )

    def forward(
        self,
        query_times: Tensor,  # [..., Q]
        context_times: Tensor,  # [..., T]
        context_values: Tensor,  # [..., T, D]
    ) -> Distribution:
        r"""Return the predictive distribution at ``query_times``.

        To create batches whose members have varying sequence length,
        use `torch.nn.rnn.utils.pad_sequence` with `padding_value=torch.nan`.

        Args:
            query_times: Times at which forecasts are requested.
            context_times: Times of the observed context sequence.
            context_values: Observed context values.

        Returns:
            Predictive distribution over target values at ``query_times``.
        """
        # ensure time stamps are sorted
        *batch_shape, seq_length, n = context_values.shape
        context_deltas = context_times.diff(prepend=context_times[..., [0]])
        query_deltas = query_times.diff(prepend=context_times[..., [-1]])
        assert (context_deltas > 0).all(), "context times not sorted"
        assert (query_deltas > 0).all(), "query times not sorted"

        # encode observations
        y_means, y_variances = self.encoder(context_values)

        # prepare initial state μ₀⁺, Σ₀⁺
        cov_u = self.initial_covariance[:n, :n].diagonal(-2, -1)
        cov_l = self.initial_covariance[n:, n:].diagonal(-2, -1)
        cov_s = self.initial_covariance[:n, n:].diagonal(-2, -1)
        post_mean = self.initial_mean
        post_cov = (cov_u, cov_l, cov_s)

        prior_means = []
        prior_covariances = []
        posterior_means = []
        posterior_covariances = []
        pred_means = []
        pred_variances = []

        # forward loop over context
        for dt, y, y_var in zip(context_deltas, y_means, y_variances, strict=True):
            prior_mean, prior_cov = self.propagate_state(dt, post_mean, post_cov)
            post_mean, post_cov = self.update_state(y, y_var, prior_mean, prior_cov)

            prior_means.append(prior_mean)
            prior_covariances.append(prior_cov)
            posterior_means.append(post_mean)
            posterior_covariances.append(post_cov)

        # forward loop over query
        # μₜ⁻, Σₜ⁻ ← predict(μₛ⁺, Σₛ⁺, t - s)
        # μₜ⁺, Σₜ⁺ ← update(μₜ⁻, Σₜ⁻, yₜ, σₜ^{obs})
        # oₜ, σₜ^{out} ← g_ϕ(μₜ⁺, Σₜ⁺)
        mean, cov = post_mean, post_cov
        for dt in context_deltas:
            mean, cov = self.propagate_state(dt, mean, cov)
            pred_mean, pred_var = self.decoder(mean, cov)
            pred_means.append(pred_mean)
            pred_variances.append(pred_var)

        return torch.stack(pred_means, dim=-2), torch.stack(pred_variances, dim=-2)

    def transition_matrix_model(self, mean: Tensor) -> Tensor:
        """Locally linear transition model.

        Aₜ = ∑ₖαₖ(t)Aₖ, where αₖ(t) = w_ψ(μₜ⁺)
        Here Aₖ = [Aₖ¹¹, Aₖ¹²; Aₖ²¹, Aₖ²²],
        where each block is a band-matrix of bandwidth b.
        """
        *batch_shape, n = mean.shape
        alpha = self.transition_coefficient_model(mean)  # (..., k)
        weighted = torch.einsum(
            "...k, knb -> ...nb", alpha, self.transition_matrix_parameters
        )  # (..., 4, p(b))

        # block_banded_mask is (2d, 2d)
        A = self.block_banded_mask.to(dtype=weighted.dtype).expand(*batch_shape, n, n)
        A = A.masked_scatter(self.block_banded_mask, weighted)
        return A

    def propagate_state(
        self,
        delta_time: Tensor,  # (...)
        posterior_mean: Tensor,  # (..., 2d)
        posterior_variance: CRU_covariance,  # (...,d), (...,d), (...,d)
    ) -> tuple[Tensor, CRU_covariance]:
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

        A = self.transition_matrix_model(posterior_mean)
        Q = torch.diag_embed(self.variance_activation(self.q))

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

    def update_state(
        self,
        observation_mean: Tensor,  # (..., d)
        observation_variance: Tensor,  # (..., d)
        prior_mean: Tensor,  # (..., 2d)
        prior_variance: CRU_covariance,  # (..., d), (..., d), (..., d)
        *,
        observation_mask: Tensor,  # (...,)
    ) -> tuple[Tensor, CRU_covariance]:  # (..., 2d), (σᵘ, σˡ, σˢ)
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
