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
]

import math
from typing import Final

import torch
import torchode as to
from torch import Tensor, nn


class TorchODESolver(nn.Module):
    r"""Torchode-backed ODE solver adapter with exact final-time landing."""

    method: Final[str]
    step_size: Final[float | None]

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
        vector_field: nn.Module,
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
        if not bool((target_time > 0).any()):
            return state

        batch_shape = state.shape[:-1]
        hidden_size = state.shape[-1]
        y0 = state.reshape(-1, hidden_size)
        t_end = target_time.reshape(-1)
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
        return solution.ys[:, -1].reshape(*batch_shape, hidden_size)


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
    bias: Final[bool]

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
        self.bias = bias
        self.gru = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.weight = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

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
            torch.einsum("...dn, dnp -> ...dp", q, self.weight) + self.bias
        )
        # f_pred = flatten(m_d ⊙ r_d) (see Appendix D)
        u = torch.where(feature_mask, r, 0.0)
        f_prep = u.reshape(u.shape[:-2], -1)  # (..., D*P)
        # compute new state
        new_state = self.gru(f_prep, state)

        # keep old state if no observation at all.
        return torch.where(has_obs, new_state, state)


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

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p_hidden: int,
        *,
        prep_hidden: int,
        cov_size: int | None = None,
        cov_hidden: int | None = None,
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
        self.solver = self.new_solver(solver, step_size=step_size)

        self.decoder = Decoder(
            input_size,
            hidden_size,
            p_hidden,
            bias=bias,
            dropout_rate=dropout_rate,
        )
        self.vector_field = GRU_ODE(hidden_size, bias=bias)
        self.gru_bayes = GRU_Bayes(input_size, hidden_size, prep_hidden, bias=bias)

        if cov_size is None:
            self.covariates_map = None
            self.initial_state = nn.Parameter(torch.zeros(hidden_size))
        else:
            if cov_hidden is None:
                raise TypeError(
                    "cov_hidden must be provided when cov_size is provided."
                )
            self.covariates_map = nn.Sequential(
                nn.Linear(cov_size, cov_hidden, bias=bias),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(cov_hidden, hidden_size, bias=bias),
                nn.Tanh(),
            )
            self.register_parameter("initial_state", None)

        self.apply(self._init_weights)

    @staticmethod
    def new_solver(
        solver: str | nn.Module, /, *, step_size: float | None = None
    ) -> nn.Module:
        r"""Return an ODE solver module from a name or custom module."""
        if isinstance(solver, nn.Module):
            return solver
        return TorchODESolver(solver, step_size=step_size)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if type(module) is nn.Linear:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.05)

    def initialize_state(
        self,
        batch_shape: torch.Size | tuple[int, ...],
        *,
        covariates: Tensor | None = None,
    ) -> Tensor:
        r"""Return the initial latent state for a batch."""
        if self.covariates_map is None:
            if covariates is not None:
                raise ValueError("covariates were provided but cov_size is None.")
            return self.initial_state.expand(*batch_shape, self.hidden_size)

        if covariates is None:
            raise ValueError("covariates are required when cov_size is provided.")
        if covariates.shape[:-1] != tuple(batch_shape):
            raise ValueError("covariates batch shape does not match batch_shape.")
        return self.covariates_map(covariates)

    def predict_observation(self, state: Tensor) -> tuple[Tensor, Tensor]:
        r"""Return Gaussian observation parameters ``(mean, logvar)``."""
        return self.decoder(state)

    decode = predict_observation

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
        observation: Tensor,  # (..., N)
        observation_mask: Tensor,  # (..., N)
        prior_state: Tensor,  # (..., H)
    ) -> Tensor:  # (..., H)
        r"""Apply one Bayesian jump update to a prior latent state."""
        batch_shape = prior_state.shape[:-1]
        assert observation.shape == (*batch_shape, self.input_size)
        assert observation_mask.shape == observation.shape

        batch_size = math.prod(batch_shape) if batch_shape else 1
        state_flat = prior_state.reshape(batch_size, self.hidden_size)
        observation_flat = observation.reshape(batch_size, self.input_size)
        mask_flat = observation_mask.reshape(batch_size, self.input_size)
        mean_flat, logvar_flat = self.decoder(state_flat)

        state_flat = self.gru_bayes(
            state_flat, observation_flat, mask_flat, mean_flat, logvar_flat
        )
        return state_flat.reshape(*batch_shape, self.hidden_size)

    def forward(
        self,
        query_times: Tensor,
        context_times: Tensor,
        context_values: Tensor,
        *,
        covariates: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Return forecasts at ``query_times``.

        The full CRU-style padded-sequence loop is intentionally left for the
        next implementation step. Use ``initialize_state``, ``propagate_state``,
        ``update_state``, and ``predict_observation`` directly while wiring the
        training loop.
        """
        raise NotImplementedError
