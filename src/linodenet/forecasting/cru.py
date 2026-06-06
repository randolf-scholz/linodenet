r"""Reimplementation of the Continuous Recurrent Unit (CRU)."""

__all__ = [
    "CRU",
    "CRUConfig",
    "DecoderConfig",
    "EncoderConfig",
    "build_cru",
    "masked_apply",
]

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn.utils.rnn import pad_sequence


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
        output_size: int,
        hidden_size: int,
        *,
        num_hidden_layers: int = 2,
        activation_function: str = "relu",
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.feature_extractor = self._build_hidden_layers(
            num_hidden_layers,
            activation_function,
        )
        self.mean_model = nn.Linear(hidden_size, output_size)
        self.variance_model = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            new_activation(variance_activation),
        )

    def _build_hidden_layers(
        self, num_layers: int, activation_name: str, /
    ) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.feature_extractor(x)
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
        activation_function: str = "relu",
        variance_activation: str = "elup1",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mean_model = self._build_mean_model(
            num_hidden_mean_model_layers,
            activation_function,
        )
        self.variance_model = self._build_variance_model(
            num_hidden_variance_model_layers,
            activation_function,
            variance_activation,
        )

    def _build_mean_model(self, num_layers: int, activation_name: str, /) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(2 * self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
        )

    def _build_variance_model(
        self,
        num_layers: int,
        activation_name: str,
        variance_activation_name: str,
        /,
    ) -> nn.Module:
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.extend([
                nn.Linear(self.hidden_size, self.hidden_size),
                new_activation(activation_name),
                nn.LayerNorm(self.hidden_size),
            ])  # fmt: skip

        return nn.Sequential(
            nn.Linear(3 * self.input_size, self.hidden_size),
            new_activation(activation_name),
            nn.LayerNorm(self.hidden_size),
            *hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
            new_activation(variance_activation_name),
        )

    def forward(
        self,
        mean: Tensor,  # (..., 2d)
        covariance: Tensor,  # (..., d, 3)
    ) -> tuple[Tensor, Tensor]:  # (..., d),
        cov = torch.cat(covariance.unbind(dim=-1), dim=-1)
        return self.mean_model(mean), self.variance_model(cov)


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    r"""Configuration for the CRU observation encoder."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_layers: int = 2
    activation_function: str = "relu"
    variance_activation: str = "elup1"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    r"""Configuration for the CRU output decoder."""

    input_size: int
    output_size: int
    hidden_size: int
    num_hidden_mean_model_layers: int = 2
    num_hidden_variance_model_layers: int = 0
    activation_function: str = "relu"
    variance_activation: str = "elup1"


@dataclass(frozen=True, slots=True, kw_only=True)
class CRUConfig:
    r"""Hierarchical configuration for constructing a CRU."""

    input_size: int
    latent_size: int
    encoder: EncoderConfig
    decoder: DecoderConfig
    output_size: int | None = None
    num_basis: int = 15
    bandwidth: int = 3
    variance_activation: str = "elup1"
    initial_variance: float = 10.0
    validate_args: bool = False


def masked_apply[R: Tensor | tuple[Tensor, ...]](
    fn: Callable[..., R],  # [*(..., *dᵢ)] -> [*(..., *eᵢ)]
    args: tuple[Tensor, ...],
    mask: Tensor,
    *,
    fill_value: float = float("nan"),
) -> R:
    r"""Apply fn only to selected batch elements.

    Args:
        fn: Function to apply. Must accept tensors with shared batch shape.
        args: The arguments to fn. Must all have the same batch shape.
        mask: The boolean mask indicating which batch elements to apply fn to. Must have the same batch shape as args.
        fill_value: The value to fill masked out batch elements with.
    """
    batch_shape = mask.shape
    B = batch_shape.numel() if batch_shape else 1
    mask_flat = mask.reshape(B).bool()  # [B]

    xs_flat = []
    for x in args:
        event_shape = x.shape[len(batch_shape) :]
        assert x.shape == batch_shape + event_shape
        xs_flat.append(x.reshape(B, *event_shape))

    # apply fn over selected batch elements
    ys_valid = fn(*(x[mask_flat] for x in xs_flat))
    returns_tensor = isinstance(ys_valid, Tensor)
    ys_tuple: tuple[Tensor, ...] = (ys_valid,) if returns_tensor else ys_valid

    y_result = []
    for y in ys_tuple:
        y_flat = torch.full(
            (B, *y.shape[1:]),
            fill_value,
            dtype=y.dtype,
            device=y.device,
        )
        y_flat[mask_flat] = y
        y_result.append(y_flat.reshape(*batch_shape, *y.shape[1:]))
    return y_result[0] if returns_tensor else tuple(y_result)


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

    Note: Differences to the reference implementation.
        - We do not make the initial covariance trainable, as this is not
          mentioned in the paper.
        - ``latent_size`` corresponds to ``latent_observation_size``.
        - ``bandwidth`` must be in the range ``[0, latent_size - 1]``.
        - Batch-first layout is used by default.

    Note:
        CRU does not support input missingness. In their experiments, for instance
        on PhysioNet, missing features are simply imputed with zeros. The mask
        is not forwarded to the model, so it cannot distinguish between
        observed zeros and missing values.
    """

    # Constants
    input_size: Final[int]
    r"""CONST: Dimensionality of observed context values."""
    output_size: Final[int]
    r"""CONST: Dimensionality of forecast targets."""
    latent_size: Final[int]
    r"""CONST: Dimensionality of encoded latent observations."""
    validate_args: Final[bool]
    r"""CONST: Whether forward inputs should be validated before computation."""

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
    prior_means: Tensor
    r"""BUFFER: Prior mean trajectory from the last forward pass."""
    prior_variances: Tensor
    r"""BUFFER: Prior covariance trajectory from the last forward pass."""
    posterior_means: Tensor
    r"""BUFFER: Posterior mean trajectory from the last forward pass."""
    posterior_variances: Tensor
    r"""BUFFER: Posterior covariance trajectory from the last forward pass."""

    @property
    def config(self) -> dict[str, object]:
        r"""Return constructor-relevant configuration."""
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "latent_size": self.latent_size,
            "validate_args": self.validate_args,
            "initial_variance": self.initial_variance,
        }

    @staticmethod
    def nll(values: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        r"""Return NaN-aware diagonal Gaussian negative log-likelihood.

        The feature dimension is treated as the event dimension. The returned loss
        sums over features and averages over valid batch/time points.
        """
        assert values.shape == means.shape == variances.shape

        value_is_nan = values.isnan()
        value_is_observed = values.isfinite().all(dim=-1)
        value_is_missing = value_is_nan.all(dim=-1)
        assert (value_is_observed | value_is_missing).all()

        prediction_is_nan = means.isnan() & variances.isnan()
        assert (~prediction_is_nan | value_is_nan).all()

        assert value_is_observed.any()
        assert means[value_is_observed].isfinite().all()
        assert variances[value_is_observed].isfinite().all()
        assert (variances[value_is_observed] > 0).all()

        return (
            0.5
            * (
                (values[value_is_observed] - means[value_is_observed]).square()
                / variances[value_is_observed]
                + torch.log(2 * torch.pi * variances[value_is_observed])
            ).sum(dim=-1)
        ).mean()

    def __init__(
        self,
        input_size: int,
        latent_size: int,  # corresponds to latent_observation_size in CRU reference impl.
        *,
        output_size: int | None = None,
        encoder: nn.Module,
        decoder: nn.Module,
        num_basis: int = 15,  # number of basis matrices for the transition model
        bandwidth: int = 3,  # bandwidth of the blocks of the transition matrix
        initial_variance: float = 10.0,
        variance_activation: str = "elup1",
        validate_args: bool = False,
    ) -> None:
        super().__init__()
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive.")

        self.input_size = input_size
        if output_size is None:
            output_size = getattr(decoder, "output_size", None)
        if not isinstance(output_size, int):
            raise TypeError(
                "output_size must be provided if decoder has no output_size."
            )
        self.output_size = output_size
        self.latent_size = latent_size
        self.num_basis = num_basis
        self.validate_args = validate_args
        self.initial_variance = initial_variance

        self.variance_activation = new_activation(variance_activation)

        self.encoder = encoder
        self.decoder = decoder

        # The transition matrix A is parametrized as a linear combination ∑ₖαₖ(μₜ)Aₖ
        # where Aₖ = [[B₁₁, B₁₂], [B₂₁, B₂₂]] and each block is a d×d banded matrix of bandwidth b.
        # The number of parameters of a banded d×d matrix with bandwidth b is:
        #   d + 2*(T_{d-1} - T_{d-b-1}), where Tₙ is the n-th triangle number
        # note: reference allows bandwidth=latent_size, which is a bug.
        assert bandwidth >= 0, "bandwidth must be non-negative"
        assert bandwidth < latent_size, "bandwidth must be smaller than latent_size."
        T = lambda n: n * (n + 1) // 2
        num_params: int = latent_size + 2 * (
            T(latent_size - 1) - T(latent_size - bandwidth - 1)
        )
        self.transition_matrix_parameters = nn.Parameter(
            torch.zeros(num_basis, 2, 2, num_params)
        )

        # create a mask for the transition matrix model
        band_mask = (
            torch.ones((latent_size, latent_size), dtype=torch.bool)
            .triu(-bandwidth)
            .tril(bandwidth)
        )
        self.register_buffer(
            "block_banded_mask",
            band_mask.expand(2, 2, latent_size, latent_size),
        )

        # "For all experiments, we used a transition net with one linear layer and
        # softmax output."
        self.transition_coefficient_model = nn.Sequential(
            nn.Linear(2 * self.latent_size, self.num_basis),
            nn.Softmax(dim=-1),
        )

        # NOTE: The reference implementation makes the initial variance trainable.
        # this however is not mentioned in the paper. We use fixed buffers instead.
        self.register_buffer("initial_mean", torch.zeros(2 * self.latent_size))
        self.register_buffer(
            "initial_covariance", initial_variance * torch.eye(2 * self.latent_size)
        )
        self.register_buffer("q", torch.ones(2 * self.latent_size))
        self.register_buffer("prior_means", torch.empty(0), persistent=False)
        self.register_buffer("prior_variances", torch.empty(0), persistent=False)
        self.register_buffer("posterior_means", torch.empty(0), persistent=False)
        self.register_buffer("posterior_variances", torch.empty(0), persistent=False)

    def forecast_unbatched(
        self, args: Iterable[tuple[Tensor, Tensor, Tensor]], /
    ) -> Distribution:
        # convert list of tuples to tuples of tensors
        query_times: tuple[Tensor, ...]
        context_times: tuple[Tensor, ...]
        context_values: tuple[Tensor, ...]
        query_times, context_times, context_values = zip(*args, strict=True)

        # pad with NaN
        context_times_tensor = pad_sequence(
            context_times, batch_first=True, padding_value=torch.nan
        )
        context_values_tensor = pad_sequence(
            context_values, batch_first=True, padding_value=torch.nan
        )
        query_times_tensor = pad_sequence(
            query_times, batch_first=True, padding_value=torch.nan
        )
        return self(query_times_tensor, context_times_tensor, context_values_tensor)

    def forward(
        self,
        query_times: Tensor,  # [..., Q]
        context_times: Tensor,  # [..., T]
        context_values: Tensor,  # [..., T, N]
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
        batch_shape = context_times.shape[:-1]
        d = self.latent_size
        query_mask = query_times.isfinite()
        context_mask = context_times.isfinite()
        context_lengths = context_mask.sum(dim=-1)  # (...)
        last_context_time = torch.take_along_dim(
            context_times, (context_lengths - 1).unsqueeze(-1), dim=-1
        )
        context_deltas = context_times.diff(prepend=context_times[..., [0]])  # (..., T)
        query_deltas = query_times.diff(prepend=last_context_time)  # (..., Q)
        assert (context_deltas[context_mask] >= 0).all(), "context times not sorted"
        assert (query_deltas[query_mask] >= 0).all(), "query times not sorted"
        # we assume sequences were batched using NaN-padding.
        # since the model does not support missing values, we mark any observation vector
        # that contains any missing value as illegal.
        observation_valid = context_values.isfinite().all(dim=-1)  # (..., T)
        assert (context_mask == observation_valid).all()

        # encode observations
        y_means, y_variances = masked_apply(
            self.encoder, (context_values,), context_mask
        )  # (..., T, D), (..., T, D)

        # prepare initial state μ₀⁺, Σ₀⁺
        cov_u = self.initial_covariance[:d, :d].diagonal(dim1=-2, dim2=-1)
        cov_l = self.initial_covariance[d:, d:].diagonal(dim1=-2, dim2=-1)
        cov_s = self.initial_covariance[:d, d:].diagonal(dim1=-2, dim2=-1)
        post_mean = self.initial_mean.expand(*batch_shape, 2 * d)  # (..., 2d)
        post_cov = torch.stack(
            [cov_u, cov_l, cov_s],
            dim=-1,
        ).expand(*batch_shape, d, 3)  # (..., d, 3)

        prior_means_list = []
        prior_variances_list = []
        posterior_means_list = []
        posterior_variances_list = []
        pred_means_list = []
        pred_variances_list = []

        # forward loop over sequence length
        for dt, y, y_var, mask in zip(
            context_deltas.unbind(dim=-1),
            y_means.unbind(dim=-2),
            y_variances.unbind(dim=-2),
            context_mask.unbind(dim=-1),
            strict=True,
        ):
            prior_mean, prior_cov = masked_apply(
                self.propagate_state, (dt, post_mean, post_cov), mask
            )
            post_mean, post_cov = self.update_state(
                y, y_var, mask, prior_mean, prior_cov
            )

            prior_means_list.append(prior_mean)
            prior_variances_list.append(prior_cov)
            posterior_means_list.append(post_mean)
            posterior_variances_list.append(post_cov)

        # create buffers of the trajectory
        self.prior_means = torch.stack(prior_means_list, dim=-2)
        self.prior_variances = torch.stack(prior_variances_list, dim=-3)
        self.posterior_means = torch.stack(posterior_means_list, dim=-2)
        self.posterior_variances = torch.stack(posterior_variances_list, dim=-3)

        # select the last valid state for each batch element
        last_post_mean = torch.take_along_dim(
            self.posterior_means,  # (..., T, 2d)
            (context_lengths - 1).unsqueeze(-1).unsqueeze(-1),
            dim=-2,
        ).squeeze(-2)  # (..., 2d)
        last_post_cov = torch.take_along_dim(
            self.posterior_variances,  # (..., T, d, 3)
            (context_lengths - 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            dim=-3,
        ).squeeze(-3)  # (..., d, 3)

        # forward loop over query
        # μₜ⁻, Σₜ⁻ ← predict(μₛ⁺, Σₛ⁺, t - s)
        # μₜ⁺, Σₜ⁺ ← update(μₜ⁻, Σₜ⁻, yₜ, σₜ^{obs})
        # oₜ, σₜ^{out} ← g_ϕ(μₜ⁺, Σₜ⁺)
        mean, cov = last_post_mean, last_post_cov
        for dt, mask in zip(
            query_deltas.unbind(dim=-1),
            query_mask.unbind(dim=-1),
            strict=True,
        ):
            mean, cov = masked_apply(self.propagate_state, (dt, mean, cov), mask)
            pred_mean, pred_var = masked_apply(self.decoder, (mean, cov), mask)
            pred_means_list.append(pred_mean)
            pred_variances_list.append(pred_var)

        return (
            torch.stack(pred_means_list, dim=-2),
            torch.stack(pred_variances_list, dim=-2),
        )

    def transition_matrix_model(self, mean: Tensor) -> Tensor:
        """Locally linear transition model.

        Aₜ = ∑ₖαₖ(t)Aₖ, where αₖ(t) = w_ψ(μₜ⁺)
        Here Aₖ = [Aₖ¹¹, Aₖ¹²; Aₖ²¹, Aₖ²²],
        where each block is a band-matrix of bandwidth b.
        """
        *batch_shape, _ = mean.shape
        d = self.latent_size
        alpha = self.transition_coefficient_model(mean)  # (..., k)
        weighted = torch.einsum(
            "...k, kijp -> ...ijp", alpha, self.transition_matrix_parameters
        )  # (..., 2, 2, p(b))

        mask = self.block_banded_mask.expand(*batch_shape, 2, 2, d, d)
        blocks = weighted.new_zeros(*batch_shape, 2, 2, d, d)
        blocks = blocks.masked_scatter(mask, weighted)
        return torch.cat([
            torch.cat([blocks[..., 0, 0, :, :], blocks[..., 0, 1, :, :]], dim=-1),
            torch.cat([blocks[..., 1, 0, :, :], blocks[..., 1, 1, :, :]], dim=-1),
        ], dim=-2)  # fmt: skip

    def propagate_state(
        self,
        delta_time: Tensor,  # (...)
        posterior_mean: Tensor,  # (..., 2d)
        posterior_variance: Tensor,  # (..., d, 3)
    ) -> tuple[Tensor, Tensor]:
        r"""Propagate a latent posterior through the continuous transition model."""
        # reconstruct Σ from σᵤ, σᵥ, σₛ
        var_u, var_l, var_s = posterior_variance.unbind(dim=-1)
        cov_u = torch.diag_embed(var_u)
        cov_l = torch.diag_embed(var_l)
        cov_s = torch.diag_embed(var_s)
        cov = torch.cat([
            torch.cat([cov_u, cov_s], dim=-1),
            torch.cat([cov_s, cov_l], dim=-1),
        ], dim=-2)  # fmt: skip

        A = self.transition_matrix_model(posterior_mean)
        Q = torch.diag_embed(self.variance_activation(self.q)).expand_as(A)

        # compute van Loan matrix exponential
        n = posterior_mean.shape[-1]
        zero = torch.zeros_like(A)
        M = torch.cat([
            torch.cat([A, Q], dim=-1),
            torch.cat([zero, -A.mT], dim=-1),
        ], dim=-2)  # fmt: skip
        # eᴹᵗ = [[F, C], [0, -Fᵀ]]
        exp_Mt = torch.linalg.matrix_exp(M * delta_time[..., None, None])
        exp_At = exp_Mt[..., :n, :n]  # upper left block
        C = exp_Mt[..., :n, n:]  # upper right block

        # μₜ = eᴬᵗμ₀
        prior_mean = torch.einsum("...mn, ...n -> ...m", exp_At, posterior_mean)
        # Σₜ = eᴬᵗΣ₀eᴬᵀᵗ + Ceᴬᵀᵗ
        prior_cov = (exp_At @ cov + C) @ exp_At.mT  # [Σᵤ, Σₛ; Σₛ, Σˡ]

        # Note: If X is block-wise diagonal, then exp(X) is also block-wise diagonal.
        d = var_u.shape[-1]
        prior_var_u = prior_cov[..., :d, :d].diagonal(dim1=-2, dim2=-1)
        prior_var_s = prior_cov[..., :d, d:].diagonal(dim1=-2, dim2=-1)
        prior_var_l = prior_cov[..., d:, d:].diagonal(dim1=-2, dim2=-1)

        return prior_mean, torch.stack([prior_var_u, prior_var_l, prior_var_s], dim=-1)

    def update_state(
        self,
        observation_mean: Tensor,  # (..., d)
        observation_variance: Tensor,  # (..., d)
        observation_mask: Tensor,  # (...,)
        prior_mean: Tensor,  # (..., 2d)
        prior_variance: Tensor,  # (..., d, 3)
    ) -> tuple[Tensor, Tensor]:  # (..., 2d), (..., d, 3)
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
        var_u, var_l, var_s = prior_variance.unbind(dim=-1)
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
        post_cov = torch.stack([
            var_u - torch.where(mask, gain_u * var_u, 0.0),  # (1-Kᵘ)σᵘ
            var_l - torch.where(mask, gain_l * var_s, 0.0),  # σˡ - Kˡσˢ
            var_s - torch.where(mask, gain_u * var_s, 0.0),  # (1-Kᵘ)σˢ
        ], dim=-1)  # fmt: skip

        # validation that resulting covariance is positive definite
        # if __debug__:
        #     assert (var_u > 0).all()
        #     assert (var_l > 0).all()
        #     assert (var_s >= 0).all()
        #     assert (var_u * var_l > var_s**2).all()

        return post_mean, post_cov


def build_cru(config: CRUConfig | Mapping[str, object], /) -> CRU:
    r"""Construct a CRU from a hierarchical configuration object."""
    if isinstance(config, Mapping):
        config = dict(config)
        config["encoder"] = EncoderConfig(**config["encoder"])
        config["decoder"] = DecoderConfig(**config["decoder"])
        config = CRUConfig(**config)

    encoder = Encoder(
        config.encoder.input_size,
        config.encoder.output_size,
        config.encoder.hidden_size,
        num_hidden_layers=config.encoder.num_hidden_layers,
        activation_function=config.encoder.activation_function,
        variance_activation=config.encoder.variance_activation,
    )
    decoder = Decoder(
        config.decoder.input_size,
        config.decoder.output_size,
        config.decoder.hidden_size,
        num_hidden_mean_model_layers=config.decoder.num_hidden_mean_model_layers,
        num_hidden_variance_model_layers=(
            config.decoder.num_hidden_variance_model_layers
        ),
        activation_function=config.decoder.activation_function,
        variance_activation=config.decoder.variance_activation,
    )

    return CRU(
        config.input_size,
        config.latent_size,
        output_size=config.output_size,
        encoder=encoder,
        decoder=decoder,
        num_basis=config.num_basis,
        bandwidth=config.bandwidth,
        initial_variance=config.initial_variance,
        variance_activation=config.variance_activation,
        validate_args=config.validate_args,
    )
