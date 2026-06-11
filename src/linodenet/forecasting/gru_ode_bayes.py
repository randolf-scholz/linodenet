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
    "GRU_ODE",
    "GRU_Bayes",
    "GRU_ODE_Bayes",
    "TorchODESolver",
    "gaussian_kl",
    "gaussian_kl_logsigma",
]

import math
from collections.abc import Callable
from typing import Final

import torch
import torchode as to
from torch import Tensor, nn


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
    returns_tensor = isinstance(ys_flat, Tensor)
    ys_tuple: tuple[Tensor, ...] = (ys_flat,) if returns_tensor else ys_flat

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
        target_time = torch.as_tensor(
            delta_time, device=state.device, dtype=state.dtype
        )
        if bool((target_time < 0).any()):
            raise ValueError("delta_time must be non-negative.")
        if target_time.shape != state.shape[:-1]:
            raise ValueError("delta_time shape must match state batch shape.")
        positive = target_time > 0
        if not bool(positive.any()):
            return state

        batch_shape = state.shape[:-1]
        hidden_size = state.shape[-1]
        y0 = state[positive].reshape(-1, hidden_size)
        t_end = target_time[positive].reshape(-1)
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

        if not bool((solution.status == to.Status.SUCCESS.value).all()):
            raise RuntimeError(f"torchode solve failed with status {solution.status}.")

        result = state.clone()
        result[positive] = solution.ys[:, -1].reshape(-1, hidden_size)
        return result.reshape(*batch_shape, hidden_size)


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
        p_hidden: int,
        *,
        bias: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(hidden_size, p_hidden, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

    # (..., H) -> (..., D), (..., D)
    def forward(self, state: Tensor, /) -> tuple[Tensor, Tensor]:
        r"""Return Gaussian observation parameters ``(mean, logvar)``."""
        mean, logvar = self.net(state).chunk(2, dim=-1)
        return mean, logvar


class GRU_Bayes(nn.Module):
    r"""Bayesian jump update network for partially observed Gaussian data.

    .. math::
        h₊ = GRU(h₋, f_{prep}(y, m, h₋))
        μ, log(σ) = Decoder(h₋)
        s = (y - μ)/σ
        q = [y, μ, σ, s]
        f = m⊙relu(Wq)

    The decoder provides ``(mean, logvar)``. This module only computes the
    normalized residual features needed for the GRU update; it does not compute
    or return objective terms.
    """

    input_size: Final[int]
    hidden_size: Final[int]
    prep_hidden: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        prep_hidden: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prep_hidden = prep_hidden
        self.gru = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.weight = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

    def forward(
        self,
        state: Tensor,  # (..., H)
        observation: Tensor,  # (..., D), may contain NaNs
        mean: Tensor,  # (..., D)
        logvar: Tensor,  # (..., D)
    ) -> Tensor:  # (..., H)
        r"""Apply one discrete Bayesian jump update."""
        assert mean.shape == logvar.shape == observation.shape

        # compute u = f_{prep}(y, m, h₋)
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
        f_prep = u.reshape(*u.shape[:-2], -1)  # (..., D*P)
        # compute new state
        new_state = self.gru(f_prep, state)

        # keep old state if no observation at all.
        return torch.where(has_obs.unsqueeze(-1), new_state, state)


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
    step_size: Final[float | None]

    prior_states: Tensor
    r"""BUFFER: Prior hidden states from the last forward pass."""
    posterior_states: Tensor
    r"""BUFFER: Posterior hidden states from the last forward pass."""
    prior_means: Tensor
    r"""BUFFER: Prior predictive means from the last forward pass."""
    prior_logvars: Tensor
    r"""BUFFER: Prior predictive log-variances from the last forward pass."""
    posterior_means: Tensor
    r"""BUFFER: Posterior predictive means from the last forward pass."""
    posterior_logvars: Tensor
    r"""BUFFER: Posterior predictive log-variances from the last forward pass."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_hidden: int,
        *,
        prep_hidden: int,
        bias: bool = True,
        dropout_rate: float = 0.0,
        step_size: float | None = None,
        solver: str | nn.Module = "euler",
    ) -> None:
        r"""Initialize the experiment-aligned GRU-ODE-Bayes scaffold."""
        super().__init__()
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive when provided.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.step_size = step_size
        self.solver = TorchODESolver.new(solver, step_size=step_size)

        self.decoder = Decoder(
            input_size,
            hidden_size,
            p_hidden,
            bias=bias,
            dropout_rate=dropout_rate,
        )
        self.vector_field = GRU_ODE(hidden_size, bias=bias)
        self.gru_bayes = GRU_Bayes(input_size, hidden_size, prep_hidden, bias=bias)

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

    def propagate_state(
        self,
        delta_time: Tensor,  # (...)
        posterior_state: Tensor,  # (..., H)
    ) -> Tensor:  # (..., H)
        r"""Propagate a posterior state through the continuous GRU-ODE dynamics."""
        return self.solver(self.vector_field, delta_time, posterior_state)

    def update_state(
        self,
        prior_state: Tensor,  # (..., H)
        observation: Tensor,  # (..., N)
    ) -> Tensor:  # (..., H)
        r"""Apply one Bayesian jump update to a prior latent state."""
        batch_shape = prior_state.shape[:-1]
        assert observation.shape == (*batch_shape, self.input_size)

        mean, logvar = self.decoder(prior_state)

        return self.gru_bayes(prior_state, observation, mean, logvar)

    def forward(
        self,
        query_times: Tensor,  # (..., $K), possibly with trailing NaNs (padding)
        context_times: Tensor,  # (..., $N), possibly with trailing NaNs (padding)
        context_values: Tensor,  # (..., $N, D), possibly with NaNs
    ) -> tuple[Tensor, Tensor]:  # (..., $K, D), (..., $K, D)
        r"""Return predictive distribution parameters at ``query_times``.

        Padded time steps are represented by ``NaN`` times. Context values may
        contain feature-level ``NaN`` entries; finite entries are used in the
        Bayesian jump, missing entries are ignored.
        """
        *batch_shape, num_steps = context_times.shape
        assert context_values.shape == (*batch_shape, num_steps, self.input_size)

        query_mask = query_times.isfinite()
        context_mask = context_times.isfinite()

        context_lengths = context_mask.sum(dim=-1)  # (...)
        last_context_time = torch.take_along_dim(
            context_times, (context_lengths - 1).unsqueeze(-1), dim=-1
        )
        context_deltas = context_times.diff(prepend=context_times[..., [0]])
        query_deltas = query_times.diff(prepend=last_context_time)
        assert (context_deltas[context_mask] >= 0).all(), "context times not sorted"
        assert (query_deltas[query_mask] >= 0).all(), "query times not sorted"

        # initialize 𝐡₀
        post_state = self.initial_state.expand(*batch_shape, self.hidden_size)

        prior_means_list = []
        prior_logvars_list = []
        posterior_means_list = []
        posterior_logvars_list = []

        for dt, observation, mask in zip(
            context_deltas.unbind(dim=-1),
            context_values.unbind(dim=-2),
            context_mask.unbind(dim=-1),
            strict=True,
        ):
            prior_state = apply_masked(self.propagate_state, (dt, post_state), mask)
            prior_mean, prior_logvar = apply_masked(self.decoder, (prior_state,), mask)

            updated_state = apply_masked(
                self.update_state, (prior_state, observation), mask
            )
            post_state = torch.where(mask.unsqueeze(-1), updated_state, post_state)
            posterior_mean, posterior_logvar = apply_masked(
                self.decoder, (post_state,), mask
            )

            prior_means_list.append(prior_mean)
            prior_logvars_list.append(prior_logvar)
            posterior_means_list.append(posterior_mean)
            posterior_logvars_list.append(posterior_logvar)

        self.prior_means = torch.stack(prior_means_list, dim=-2)
        self.prior_logvars = torch.stack(prior_logvars_list, dim=-2)
        self.posterior_means = torch.stack(posterior_means_list, dim=-2)
        self.posterior_logvars = torch.stack(posterior_logvars_list, dim=-2)

        pred_means_list = []
        pred_logvars_list = []
        state = post_state
        for dt, mask in zip(
            query_deltas.unbind(dim=-1),
            query_mask.unbind(dim=-1),
            strict=True,
        ):
            next_state = apply_masked(self.propagate_state, (dt, state), mask)
            state = torch.where(mask.unsqueeze(-1), next_state, state)
            pred_mean, pred_logvar = apply_masked(self.decoder, (state,), mask)
            pred_means_list.append(pred_mean)
            pred_logvars_list.append(pred_logvar)

        if query_times.shape[-1] == 0:
            empty = self.initial_state.new_empty(*batch_shape, 0, self.input_size)
            return empty, empty

        return (
            torch.stack(pred_means_list, dim=-2),
            torch.stack(pred_logvars_list, dim=-2),
        )


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


def gaussian_kl_logsigma(
    left: tuple[Tensor, Tensor],  # μ, log(σ) (..., d), (..., d)
    right: tuple[Tensor, Tensor],  # μ, log(σ) (..., d), (..., d)
    /,
) -> Tensor:  # (...)
    r"""Return the KL divergence between two diagonal Gaussians.

    .. math::
        Dₖₗ(𝓝(μ₁, σ₁²), 𝓝(μ₂, σ₂²))
        = log(σ₂/σ₁) + ½(exp(-2(log(σ₂/σ₁))) + (μ₁ - μ₂)² exp(-2log(σ₂))) - ½
    """
    mu_1, logsigma_1 = left
    mu_2, logsigma_2 = right
    log_ratio = logsigma_2 - logsigma_1
    return 0.5 * (
        2 * log_ratio
        + torch.exp(-2 * log_ratio)
        + torch.exp(-2 * logsigma_2) * (mu_1 - mu_2) ** 2
        - 1
    )  # fmt: skip
