r"""Clean state-space scaffold for GRU-ODE-Bayes.

References:
    - | GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series
      | Edward De Brouwer, Jaak Simm, Adam Arany, Yves Moreau
      | NeurIPS 2019
      | https://github.com/edebrouwer/gru_ode_bayes

This module intentionally starts from the model variant used by the reference
experiments: full autonomous GRU-ODE dynamics with log-variance Gaussian
Bayesian jumps.
"""

__all__ = [
    "Decoder",
    "GRUODEBayesConfig",
    "GRU_ODE",
    "GRU_Bayes",
    "GRU_ODE_Bayes",
    "ODE_Flow",
    "TorchODESolver",
    "gaussian_kl",
    "gaussian_kl_logvar",
    "apply_masked",
]

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch
import torchode as to
from torch import Tensor, nan, nn


def apply_masked[R: Tensor | tuple[Tensor, ...]](
    fn: Callable[..., R],  # [*(..., *dᵢ)] -> [*(..., *eᵢ)]
    args: tuple[Tensor, ...],
    mask: Tensor,  # (...)
    *,
    fill_value: float = float("nan"),
) -> R:  # *(..., *eᵢ)
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
        xs_flat.append(x.reshape(-1, *event_shape))

    # apply fn over selected batch elements
    ys_flat = fn(*(x[mask_flat] for x in xs_flat))

    if isinstance(ys_flat, Tensor):
        result = torch.full(
            (B, *ys_flat.shape[1:]),
            fill_value,
            dtype=ys_flat.dtype,
            device=ys_flat.device,
        )
        result[mask_flat] = ys_flat
        result = result.reshape(*batch_shape, *result.shape[1:])
        return result  # type: ignore[return-value]

    y_result: list[Tensor] = []
    for y in ys_flat:
        y_flat = torch.full(
            (B, *y.shape[1:]),
            fill_value,
            dtype=y.dtype,
            device=y.device,
        )
        y_flat[mask_flat] = y
        y_result.append(y_flat.reshape(*batch_shape, *y.shape[1:]))
    return tuple(y_result)  # type: ignore[return-value]


class TorchODESolver(nn.Module):
    r"""Torchode-backed ODE solver adapter with exact final-time landing."""

    method: Final[str]
    step_size: Final[float | None]

    @staticmethod
    def new(solver: str | nn.Module, /, *, step_size: float | None = None) -> nn.Module:
        r"""Return an ODE solver module from a name or custom module."""
        if isinstance(solver, nn.Module):
            return solver
        return TorchODESolver(solver, step_size=step_size)

    def __init__(
        self, method: str = "euler", *, step_size: float | None = None
    ) -> None:
        super().__init__()
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive when provided.")
        match method:
            case "euler" | "heun" | "dopri5" | "tsit5":
                self.method = method
            case "midpoint":
                raise NotImplementedError(
                    "torchode does not provide a midpoint solver."
                )
            case _:
                raise NotImplementedError(f"Unknown torchode solver {method!r}.")
        self.step_size = step_size

    def step_method(self, term: to.ODETerm, /) -> nn.Module:
        r"""Return the torchode step method for ``term``."""
        match self.method:
            case "euler":
                return to.Euler(term)
            case "heun":
                return to.Heun(term)
            case "dopri5":
                return to.Dopri5(term)
            case "tsit5":
                return to.Tsit5(term)
            case _:
                raise AssertionError("unreachable")

    def forward(
        self,
        vector_field: nn.Module,  # [(...), (..., H)] -> (..., H)
        delta_time: Tensor,  # (...)
        state: Tensor,  # (..., H)
    ) -> Tensor:  # (..., H)
        r"""Propagate ``state`` independently for exactly ``delta_time``."""
        assert (delta_time >= 0).all(), "delta_time must be non-negative."
        assert delta_time.shape == state.shape[:-1]
        positive = delta_time > 0
        if not positive.any():
            return state

        batch_shape = state.shape[:-1]
        hidden_size = state.shape[-1]
        y0 = state[positive].reshape(-1, hidden_size)
        t_end = delta_time[positive].reshape(-1)
        t_start = torch.zeros_like(t_end)
        t_eval = torch.stack([t_start, t_end], dim=-1)

        dt0 = (
            t_end if self.step_size is None else torch.full_like(t_end, self.step_size)
        )

        term = to.ODETerm(vector_field)
        solver = to.AutoDiffAdjoint(
            self.step_method(term),
            to.FixedStepController(),
        )
        solution = solver.solve(
            to.InitialValueProblem(y0=y0, t_eval=t_eval),
            term,
            dt0=dt0,
        )

        if (solution.status != to.Status.SUCCESS.value).any():
            raise RuntimeError(f"torchode solve failed with status {solution.status}.")

        result = state.clone()
        result[positive] = solution.ys[:, -1].reshape(-1, hidden_size)
        return result.reshape(*batch_shape, hidden_size)


class ODE_Flow(nn.Module):
    r"""Flow wrapper that evolves states with a vector field and ODE solver."""

    vector_field: nn.Module
    solver: nn.Module

    def __init__(self, vector_field: nn.Module, solver: nn.Module, /) -> None:
        super().__init__()
        self.vector_field = vector_field
        self.solver = solver

    def forward(
        self,
        delta_time: Tensor,  # (...)
        state: Tensor,  # (..., H)
    ) -> Tensor:  # (..., H)
        r"""Propagate ``state`` for ``delta_time``."""
        return self.solver(self.vector_field, delta_time, state)


class GRU_ODE(nn.Module):
    r"""GRU-ODE $d𝐡(t)/dt = (1-𝐳(t))⊙(𝐠(t)-𝐡(t)).

    .. note:: we assume $𝐱ₜ=0$ (autonomous case)

    .. math::
        𝐫(t) = σ(Wᵣ𝐱ₜ + Uᵣ𝐡ₜ₋₁ + bᵣ)
        𝐳(t) = σ(W₟𝐱ₜ + U₟𝐡ₜ₋₁ + b₟)
        𝐠(t) = tanh(Wₕ𝐱ₜ + Uₕ(𝐫(t)⊙𝐡ₜ₋₁) + bₕ)
    """

    def __init__(self, hidden_size: int, *, bias: bool = True) -> None:
        super().__init__()
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=bias)

    # (..., ), (..., H) -> (..., H)
    def forward(self, _time: Tensor, state: Tensor, /) -> Tensor:
        r"""Return the ODE derivative for ``state``."""
        reset = torch.sigmoid(self.lin_hr(state))
        update = torch.sigmoid(self.lin_hz(state))
        candidate = torch.tanh(self.lin_hh(reset * state))
        return (1 - update) * (candidate - state)


class Decoder(nn.Module):
    r"""Decode latent states into diagonal Gaussian observation parameters."""

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decoder_hidden_size: int,
        *,
        bias: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, decoder_hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(decoder_hidden_size, 2 * input_size, bias=bias),
        )

    # (..., H) -> (..., D), (..., D)
    def forward(self, state: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Return Gaussian observation parameters ``(mean, logvar)``."""
        mean, logvar = self.net(state).chunk(2, dim=-1)
        return mean, logvar


class GRU_Bayes(nn.Module):
    r"""Bayesian jump update network for partially observed Gaussian data.

    .. math::
        h₊ = GRU(h₋, f_{feat}(y, m, h₋))
        μ, log(σ²) = Decoder(h₋)
        s = (y - μ)/σ
        q = [y, μ, σ, s]
        f = m⊙relu(Wq)

    The decoder provides ``(mean, logvar)``. This module only computes the
    normalized residual features needed for the GRU update; it does not compute
    or return objective terms.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    feature_embedding_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feature_embedding_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_embedding_size = feature_embedding_size
        self.gru = nn.GRUCell(
            feature_embedding_size * input_size,
            hidden_size,
            bias=bias,
        )

        std = math.sqrt(2.0 / (4 + feature_embedding_size))
        self.weight = nn.Parameter(
            std * torch.randn(input_size, 4, feature_embedding_size)
        )
        self.bias_prep = nn.Parameter(
            0.1 + torch.zeros(input_size, feature_embedding_size)
        )

    def forward(
        self,
        state: Tensor,  # (..., H)
        observation: Tensor,  # (..., D), may contain NaNs
        mean: Tensor,  # (..., D)
        logvar: Tensor,  # (..., D)
    ) -> Tensor:  # (..., H)
        r"""Apply one discrete Bayesian jump update."""
        assert mean.shape == logvar.shape == observation.shape

        # compute u = f_feat(y, m, h₋)
        feature_mask = observation.isfinite()  # (..., D)
        has_obs = feature_mask.any(dim=-1)  # (...,)

        # this step is not properly explained in the paper,
        # but done in their experimental code; NaNs need to be removed at this stage.
        values = torch.where(feature_mask, observation, 0.0)
        sigma = torch.exp(0.5 * logvar)
        error = (values - mean) / sigma

        q = torch.stack([values, mean, logvar, error], dim=-1)  # (..., D, 4)
        # r_d ≔ ϕ(W_d q_d + b_d)
        r = torch.relu(  # (..., D, P)
            torch.einsum("...dn, dnp -> ...dp", q, self.weight) + self.bias_prep
        )
        # f_pred = flatten(m_d ⊙ r_d) (see Appendix D)
        u = torch.where(feature_mask.unsqueeze(-1), r, 0.0)
        prepared_features = u.flatten(start_dim=-2)  # (..., D*P)
        # compute new state
        new_state = self.gru(prepared_features, state)

        # keep old state if no observation at all.
        return torch.where(has_obs.unsqueeze(-1), new_state, state)


@dataclass(frozen=True, slots=True, kw_only=True)
class GRUODEBayesConfig:
    r"""Configuration for constructing a GRU-ODE-Bayes model."""

    input_size: int
    hidden_size: int
    decoder_hidden_size: int
    feature_embedding_size: int
    bias: bool = True
    dropout_rate: float = 0.0
    step_size: float | None = None
    solver: str | nn.Module = "euler"
    batch_first: bool = True


class GRU_ODE_Bayes(nn.Module):
    r"""GRU-ODE-Bayes as a state-space model.

    Assumptions:
        - Latent Gaussian-linear system with diagonal covariance.
        - autonomous ODE for μ(t) and \log σ²(t)
        - latent ODE: dh/dt = (1-zₜ)⊙(gₜ - hₜ)

    State:
        ``hₜ`` is the latent recurrent state.

    Propagation:
        ``hₜ⁻ = propagate_state(Δt, hₛ⁺)`` integrates the autonomous GRU-ODE.

    Update:
        ``hₜ⁺ = update_state(xₜ, mₜ, hₜ⁻)`` applies the Bayesian GRU jump from
        the Gaussian prediction ``p_θ(hₜ⁻) = (μₜ, log σ²ₜ)``.

    This is a scaffold for the cleaned reimplementation. It deliberately does
    not preserve the reference implementation's ``time_ptr``/``obs_idx`` forward
    API; that batching format should be handled by an adapter if needed.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    batch_first: Final[bool]

    prior_means: Tensor
    r"""BUFFER: Prior predictive means from the last forward pass."""
    prior_logvars: Tensor
    r"""BUFFER: Prior predictive log-variances from the last forward pass."""
    post_means: Tensor
    r"""BUFFER: Posterior predictive means from the last forward pass."""
    post_logvars: Tensor
    r"""BUFFER: Posterior predictive log-variances from the last forward pass."""

    @classmethod
    def from_config(
        cls,
        config: GRUODEBayesConfig | Mapping[str, Any],
        /,
    ) -> GRU_ODE_Bayes:
        r"""Construct a GRU-ODE-Bayes model from a configuration object."""
        if isinstance(config, Mapping):
            config = GRUODEBayesConfig(**config)

        decoder = Decoder(
            config.input_size,
            config.hidden_size,
            config.decoder_hidden_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
        )
        if config.step_size is not None and config.step_size <= 0:
            raise ValueError("step_size must be positive when provided.")
        flow = ODE_Flow(
            GRU_ODE(config.hidden_size, bias=config.bias),
            TorchODESolver.new(config.solver, step_size=config.step_size),
        )
        update_cell = GRU_Bayes(
            config.input_size,
            config.hidden_size,
            config.feature_embedding_size,
            bias=config.bias,
        )
        return GRU_ODE_Bayes(
            config.input_size,
            config.hidden_size,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
            batch_first=config.batch_first,
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        input_size: int,
        hidden_size: int,
        decoder_hidden_size: int,
        feature_embedding_size: int,
        bias: bool = True,
        dropout_rate: float = 0.0,
        step_size: float | None = None,
        solver: str | nn.Module = "euler",
        batch_first: bool = True,
    ) -> GRU_ODE_Bayes:
        r"""Construct a GRU-ODE-Bayes model from explicit hyperparameters."""
        return cls.from_config(
            GRUODEBayesConfig(
                input_size=input_size,
                hidden_size=hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                feature_embedding_size=feature_embedding_size,
                bias=bias,
                dropout_rate=dropout_rate,
                step_size=step_size,
                solver=solver,
                batch_first=batch_first,
            )
        )

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        decoder: nn.Module,
        flow: nn.Module,
        update_cell: nn.Module,
        batch_first: bool = True,
    ) -> None:
        r"""Initialize the experiment-aligned GRU-ODE-Bayes scaffold."""
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.decoder = decoder
        self.flow = flow
        self.update_cell = update_cell

        # The paper does not specify how h₀ is initialized. The reference code
        # supports h₀ = NN(static_covariates), but its experiments use fixed
        # static_covariates = 0, making this equivalent to a learned global h₀.
        self.initial_state = nn.Parameter(torch.zeros(hidden_size))

        self.register_buffer("prior_means", torch.empty(0), persistent=False)
        self.register_buffer("prior_logvars", torch.empty(0), persistent=False)
        self.register_buffer("posterior_means", torch.empty(0), persistent=False)
        self.register_buffer("posterior_logvars", torch.empty(0), persistent=False)

        # initialize weight (carried over from reference implementation)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if type(module) is nn.Linear:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.05)

    @staticmethod
    def nll_logvar(
        values: Tensor,
        mean: Tensor,
        logvar: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        r"""Return elementwise diagonal Gaussian NLL from log-variances."""
        if values.shape != mean.shape or values.shape != logvar.shape:
            raise ValueError("values, mean, and logvar must have equal shapes.")
        if mask is None:
            mask = values.isfinite()
        elif mask.shape != values.shape:
            raise ValueError("mask must match values shape.")

        valid = mask.bool()
        centered = torch.where(valid, values.nan_to_num(0.0) - mean, 0.0)
        nll = 0.5 * (
            centered.square() * torch.exp(-logvar) + logvar + math.log(2 * math.pi)
        )
        return torch.where(valid, nll, 0.0)

    @staticmethod
    def gaussian_bayes_kl_logvar(
        p_pre: tuple[Tensor, Tensor],  # μ_pre, log(σ²_pre) (..., d), (..., d)
        p_post: tuple[Tensor, Tensor],  # μ_post, log(σ²_post) (..., d), (..., d)
        p_obs: tuple[Tensor, Tensor],  # μ_obs, log(σ²_obs) (..., d), (..., d)
        /,
    ) -> Tensor:  # (...)
        r"""Return the KL divergence D_KL(p_bayes || p_post).

        Args:
            p_pre: the prior predictive distribution parameters (μ, log σ²)
            p_post: the posterior predictive distribution parameters (μ, log σ²)
            p_obs: the observational distribution. Allowed to contain NaNs.

        .. math::
            Dₖₗ(𝓝(μ₁, σ₁²), 𝓝(μ₂, σ₂²))
            = ½(log(σ₂²/σ₁²) + σ₁²/σ₂² + (μ₁ - μ₂)²/σ₂² - 1)
        """
        # Note: the reference code contains a deviation from the description in the paper:
        #   The publication specifies the KL term as d_KL(p_bayes || p_post), p_bayes ∝ p_pre ⋅ p_obs
        #   however, in the code they compute
        #   compute_KL_loss(p_obs = p[i_obs], X_obs = Xf_batch, M_obs = Mf_batch, obs_noise_std=self.obs_noise_std)
        #   but here, p_obs is the posterior predictive distribution.
        #   so actually the reference code ends up computing D_KL(p_post || p_obs)
        mu_pre, logvar_pre = p_pre
        mu_post, logvar_post = p_post
        mu_obs, logvar_obs = p_obs
        assert mu_pre.isfinite().all()
        assert logvar_pre.isfinite().all()
        assert mu_post.isfinite().all()
        assert logvar_post.isfinite().all()
        valid = mu_obs.isfinite() & logvar_obs.isfinite()

        # 1. compute μ_bayes = (σ²_obs / (σ²_pre + σ²_obs)) * μ_pre + (σ²_pre / (σ²_pre + σ²_obs)) * μ_obs
        #    log-space: compute normalized precisions wᵢ = τᵢ / (τ_pre + τ_obs), where τᵢ = 1 / σ²ᵢ.
        logprec_pre = -logvar_pre
        logprec_obs = -logvar_obs
        logprec_bayes = torch.logaddexp(logprec_pre, logprec_obs)
        weight_pre = torch.exp(logprec_pre - logprec_bayes)
        weight_obs = torch.exp(logprec_obs - logprec_bayes)
        mu_bayes = weight_pre * mu_pre + weight_obs * mu_obs

        # 2. compute σ²_bayes = (σ²_pre * σ²_obs) / (σ²_pre + σ²_obs)
        #    log-space: σ²_bayes = 1 / (τ_pre + τ_obs).
        logvar_bayes = -logprec_bayes

        # 3. compute marginal KL-divergence
        #  KL(N(μ₁, σ₁²), N(μ₂, σ₂²)) = ½(log(σ₂²/σ₁²) + σ₁²/σ₂² + (μ₁ - μ₂)²/σ₂² - 1)
        kl = 0.5 * (
            logvar_post
            - logvar_bayes
            + torch.exp(logvar_bayes - logvar_post)
            + (mu_bayes - mu_post).square() * torch.exp(-logvar_post)
            - 1
        )
        return torch.where(valid, kl, 0.0).sum(dim=-1)

    def propagate_state(
        self,
        delta_time: Tensor,  # (...)
        posterior_state: Tensor,  # (..., H)
    ) -> Tensor:  # (..., H)
        r"""Propagate a posterior state through the continuous GRU-ODE dynamics."""
        return self.flow(delta_time, posterior_state)

    def update_state(
        self,
        prior_state: Tensor,  # (..., H)
        observation: Tensor,  # (..., D)
    ) -> Tensor:  # (..., H)
        r"""Apply one Bayesian jump update to a prior latent state."""
        batch_shape = prior_state.shape[:-1]
        assert observation.shape == (*batch_shape, self.input_size)

        mean, logvar = self.decoder(prior_state)

        return self.update_cell(prior_state, observation, mean, logvar)

    def forward(
        self,
        times: Tensor,  # (..., $N + $K), possibly with trailing NaNs (padding)
        context_values: Tensor,  # (..., $N + $K, D), possibly with NaNs
        context_mask: Tensor,  # (..., $N + $K, D), bool
        query_mask: Tensor,  # (..., $N + $K, D), bool
        *,
        initial_state: Tensor | None = None,  # (..., H)
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> tuple[Tensor, Tensor]:  # (..., $N + $K, D), (..., $N + $K, D)
        r"""Filter and forecast over combined context/query time points.

        Context and query masks explicitly select valid feature-level entries.
        Context values outside ``context_mask`` are ignored.
        """
        # optional: sanitize values
        context_values = context_values.masked_fill(~context_mask, nan)

        # initialize mask over valid time steps:
        # those with finite time and at least one observation or query coordinate.
        valid_steps = times.isfinite() & (context_mask | query_mask).any(dim=-1)
        mean_mask = valid_steps.unsqueeze(dim=-1)

        if self.batch_first:
            # Move the time axis to the front.
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

        # initialize 𝐡₀
        t = times[0] if initial_time is None else initial_time
        post_state = self.initial_state if initial_state is None else initial_state
        post_state = post_state.expand(*batch_shape, self.hidden_size)

        prior_means_list = []
        prior_logvars_list = []
        post_means_list = []
        post_logvars_list = []

        for t_obs, observation, mask, active in zip(
            times,
            context_values,
            context_mask.any(dim=-1),
            valid_steps,
            strict=True,
        ):
            # propagate to the next time step
            delta = torch.where(active, t_obs - t, torch.zeros_like(t_obs - t))
            t = torch.where(active, t_obs, t)
            prior_state = apply_masked(
                self.propagate_state, (delta, post_state), active
            )

            # get the prior prediction
            prior_mean, prior_logvar = apply_masked(
                self.decoder, (prior_state,), active
            )
            # update the state
            updated_state = apply_masked(
                self.update_state, (prior_state, observation), mask
            )
            next_state = torch.where(mask[..., None], updated_state, prior_state)
            post_state = torch.where(active[..., None], next_state, post_state)

            # get the posterior prediciton
            posterior_mean, posterior_logvar = apply_masked(
                self.decoder, (post_state,), active
            )

            prior_means_list.append(prior_mean)
            prior_logvars_list.append(prior_logvar)
            post_means_list.append(posterior_mean)
            post_logvars_list.append(posterior_logvar)

        stack_dim = -2 if self.batch_first else 0
        self.prior_means = torch.stack(prior_means_list, dim=stack_dim)
        self.prior_logvars = torch.stack(prior_logvars_list, dim=stack_dim)
        self.post_means = torch.stack(post_means_list, dim=stack_dim)
        self.post_logvars = torch.stack(post_logvars_list, dim=stack_dim)

        self.prior_means = self.prior_means.masked_fill(~mean_mask, nan)
        self.prior_logvars = self.prior_logvars.masked_fill(~mean_mask, nan)
        self.post_means = self.post_means.masked_fill(~mean_mask, nan)
        self.post_logvars = self.post_logvars.masked_fill(~mean_mask, nan)

        return self.post_means, self.post_logvars


def gaussian_kl(
    left: tuple[Tensor, Tensor],  # (..., d), (..., d)
    right: tuple[Tensor, Tensor],  # (..., d), (..., d)
    /,
) -> Tensor:  # (...)
    r"""Return the KL divergence between two diagonal Gaussians.

    .. math::
        Dₖₗ(𝓝(μ₁, σ₁²), 𝓝(μ₂, σ₂²))
        = ½(log(σ₂²) - log(σ₁²) + (σ₁² + (μ₁ - μ₂)²) / σ₂² - 1)
    """
    mu_1, var_1 = left  # μ₁, σ₁²
    mu_2, var_2 = right  # μ₂, σ₂²
    return 0.5 * (
        torch.log(var_2)
        - torch.log(var_1)
        + (var_1 + torch.pow(mu_1 - mu_2, 2)) / var_2
        - 1
    )


def gaussian_kl_logvar(
    left: tuple[Tensor, Tensor],  # μ, log(σ²) (..., d), (..., d)
    right: tuple[Tensor, Tensor],  # μ, log(σ²) (..., d), (..., d)
    /,
) -> Tensor:  # (...)
    r"""Return the KL divergence between two diagonal Gaussians.

    .. math::
        Dₖₗ(𝓝(μ₁, σ₁²), 𝓝(μ₂, σ₂²))
        = ½(log(σ₂²/σ₁²) + σ₁²/σ₂² + (μ₁ - μ₂)²/σ₂² - 1)
    """
    mu_1, logvar_1 = left
    mu_2, logvar_2 = right
    return 0.5 * (
            logvar_2
            - logvar_1
            + torch.exp(logvar_1 - logvar_2)
            + torch.exp(-logvar_2) * (mu_1 - mu_2) ** 2
            - 1
    )  # fmt: skip
