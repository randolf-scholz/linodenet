r"""Minimal, unoptimized reimplementation of LinODEnet."""

__all__ = ["LinODEnet_v0"]

from collections.abc import Callable
from functools import partial
from typing import Final, Optional, cast

import torch
from torch import Tensor, nan, nn
from torch.linalg import solve, solve_triangular

from .utils import EventBatch


def lp_loss(
    x: Tensor,  # (..., d)
    y: Tensor,  # (..., d)
    /,
    *,
    p: float = 2.0,
    dim: int = -1,
    aggregation: str = "mean",
) -> Tensor:  # (...)
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""
    match aggregation:
        case "sum":
            return (x - y).abs().pow(p).sum(dim=dim)
        case "mean":
            return (x - y).abs().pow(p).mean(dim=dim)
        case _:
            raise ValueError(f"Unexpected aggregation: {aggregation!r}")


class LpLoss(nn.Module):
    r"""Compute a per-batch-element $Lᵖ$ reconstruction loss $‖x-y‖ₚᵖ$."""

    def __init__(
        self,
        p: float = 2.0,
        dim: int = -1,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if p <= 0:
            raise ValueError(f"Expected p > 0, got {p!r}.")
        if aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"Expected aggregation to be 'sum' or 'mean', got {aggregation!r}."
            )

        self.p = p
        self.dim = dim
        self.aggregation = aggregation

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(self, x: Tensor, y: Tensor, /) -> Tensor:
        return lp_loss(x, y, p=self.p, dim=self.dim, aggregation=self.aggregation)


class GradientStepUpdater(nn.Module):
    r"""Single gradient-step updater for latent distribution parameters.

    .. math:: ℒ(z) = ∇₟ℓ(f(z), y) + λ d(z, z₋)
                z' = z₋ - η∇₟ℒ(z₋)
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        loss: nn.Module | str = "l2",
        regularizer: nn.Module | str = "l2",
        regularization_strength: float = 1e-3,
        step_size: float = 1e-2,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.regularization_strength = nn.Parameter(
            torch.as_tensor(regularization_strength)
        )
        self.step_size = nn.Parameter(torch.as_tensor(step_size))

        match loss:
            case nn.Module():
                self.loss = loss
            case "l1":
                self.loss = LpLoss(p=1.0)
            case "l2":
                self.loss = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown loss: {loss!r}")

        match regularizer:
            case nn.Module():
                self.regularizer = regularizer
            case "l1":
                self.regularizer = LpLoss(p=1.0)
            case "l2":
                self.regularizer = LpLoss(p=2.0)
            case _:
                raise ValueError(f"Unknown regularizer: {regularizer!r}")

    @partial(torch.func.vmap, in_dims=(None, 0, 0, 0))
    @partial(torch.func.grad, argnums=1)
    def _grad_fn_flat_batch(self, z: Tensor, z_prev: Tensor, y: Tensor, /) -> Tensor:
        return (
            self.loss(self.decoder(z), y)  # ℓ(f(z), y)
            + self.regularization_strength * self.regularizer(z, z_prev)  # λ‖z-z₋‖²
        )

    def grad_fn(self, z: Tensor, z_prev: Tensor, y: Tensor, /) -> Tensor:
        r"""Return the gradient while preserving the input batch shape."""
        z_flat = z.reshape(-1, z.shape[-1])
        z_prev_flat = z_prev.reshape(-1, z_prev.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        grad = self._grad_fn_flat_batch(z_flat, z_prev_flat, y_flat)
        return grad.reshape_as(z)

    __call__: Callable[[Tensor, Tensor], Tensor]

    def forward(
        self,
        z: Tensor,  # (..., d)
        y: Tensor,  # (..., e)
        /,
    ) -> Tensor:  # (..., d)
        r"""Computes z_prev - η∇₟ℒ(z_prev), where ℒ(z) = ℓ(f(z), y) + λ d(z, z_prev)."""
        return z - self.step_size * self.grad_fn(z, z, y)


class ReZero[
    M: nn.Module = nn.Module,
    S: nn.Module = nn.Module,
](nn.Module):
    r"""ReZero module, learnable scalar with optional transformation.

    .. math:: x ⟼ φ(ε) ⋅ f(x)
    """

    scalar: Tensor
    r"""PARAM: The scalar to multiply the inputs by."""
    scalar_map: S
    r"""MODULE: Map applied to the scalar before scaling the input."""
    module: M
    r"""MODULE: Map applied to the inputs before scaling them."""

    @property
    def config(self) -> dict:
        return {
            "module": self.module,
            "scalar": self.scalar,
            "scalar_map": self.scalar_map,
        }

    def __init__[U: nn.Module = nn.Identity, V: nn.Module = nn.Identity](
        self: ReZero[U, V],
        module: U | None = None,
        *,
        scalar_map: V | None = None,
        initial_value: Tensor | float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(
            torch.as_tensor(initial_value), requires_grad=learnable
        )
        self.module = cast("U", nn.Identity() if module is None else module)
        self.scalar_map = cast("V", nn.Identity() if scalar_map is None else scalar_map)

    def forward(self, x: Tensor) -> Tensor:
        return self.scalar_map(self.scalar) * self.module(x)

    def right_inverse(self, y: Tensor) -> Tensor | None:
        if (right_inverse := getattr(self.module, "right_inverse", None)) is None:
            return None

        assert callable(right_inverse)
        return right_inverse(y / self.scalar_map(self.scalar))


def linear_flow(
    timedeltas: Tensor,  # (..., $n)
    x0: Tensor,  # (..., d)
    kernel: Tensor,  # (d, d)
    bias: Optional[Tensor] = None,
    /,
) -> Tensor:  # (..., $n, d)
    r"""Linear ODE.

    .. math:: dxₜ/dt = Axₜ + b

    Given x₀, then $xₜ = eᴬᵗx₀ + φ₁(At)bt$. Here, $φₖ(z) = ∑ₙ₌₀^∞ zⁿ/(n+k)!$ are the phi-functions,
    which can be computed from the matrix exponential of an augmented block matrix.
    In particular, φ₀(A) = eᴬ and φ₁(A) = (eᴬ - I)/A.
    """
    if bias is None:
        Adt = torch.einsum("..., kl -> ...kl", timedeltas, kernel)
        expAdt = torch.linalg.matrix_exp(Adt)  # (*bs, n)
        return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0)

    # use augmented block matrix [[A, b], [0, 0]]
    n = bias.shape[-1]
    M = torch.cat(
        [
            torch.cat([kernel, bias.unsqueeze(-1)], dim=-1),
            torch.zeros((1, n + 1), device=kernel.device, dtype=kernel.dtype),
        ],
        dim=0,
    )
    Mdt = torch.einsum("..., kl -> ...kl", timedeltas, M)
    expMdt = torch.linalg.matrix_exp(Mdt)
    expAdt = expMdt[..., :n, :n]
    phi1bt = expMdt[..., :n, -1]
    return torch.einsum("...nkl, ...l -> ...nk", expAdt, x0) + phi1bt


class LinearFlow(nn.Module):
    r"""Linear Flow, solves $ẋ = Ax$, i.e. $x_{t+∆t} = e^{A{∆t}}xₜ$.

    This is augmented by 2 techniques:

    1. parametrization of the kernel, e.g. restricting it to some subset of matrices,
       such as skew-symmetric matrices, which leads to stable dynamics.
    2. an optional ReZero gate applied to the kernel, which can be used to improve
       the learning dynamics.

    .. math:: e^{ρ(π(A))∆t}x
    """

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    use_rezero: Final[bool]
    r"""CONST: Whether the kernel is wrapped in ``ReZero``."""
    use_bias: Final[bool]
    r"""CONST: Whether the flow has a learnable affine bias."""

    # Parameters
    weight: Tensor
    r"""PARAM: The learnable weight-matrix of the linear ODE component."""
    bias: Tensor | None
    r"""PARAM: Optional learnable bias of the linear ODE component."""
    rezero: nn.Module
    r"""MODULE: Optional ReZero gate applied to the kernel."""
    # Buffers
    kernel: Tensor
    r"""BUFFER: The system matrix of the linear ODE component."""
    kernel_initialization: nn.Module
    r"""MODULE: Optional Initialization of the kernel."""
    kernel_parametrization: nn.Module | None
    r"""MODULE: Optional parametrization of the kernel."""

    def __init__(
        self,
        input_size: int,
        *,
        kernel_initialization: str | Tensor | nn.Module = "skew-symmetric",
        kernel_parametrization: Optional[str | nn.Module] = None,
        use_rezero: bool = True,
        use_bias: bool = False,
    ) -> None:
        r"""Initialize the Linear ODE Cell."""
        super().__init__()

        # initialize constants
        self.input_size = input_size
        self.output_size = input_size
        self.use_rezero = use_rezero
        self.use_bias = use_bias
        self.kernel_initialization = resolve_kernel_initialization(
            input_size, kernel_initialization
        )

        # initialize parameters
        self.weight = nn.Parameter(self.kernel_initialization())
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(input_size)) if self.use_bias else None,
        )

        # apply parametrization if given
        self.kernel_parametrization = resolve_matrix_parametrization(
            kernel_parametrization
        )
        register_parametrization(self, "weight", self.kernel_parametrization)

        # initialize buffers
        self.rezero = ReZero() if self.use_rezero else nn.Identity()
        self.register_buffer("kernel", self.kernel_parametrization(self.weight))

    def step(
        self,
        timedeltas: Tensor,  # (...)
        x0: Tensor,  # (..., d)
        /,
    ) -> Tensor:  # (..., d)
        r"""Propagate the linear ODE for a single time-delta.

        .. math:: step(∆t, x) = e^{ρ(π(A))∆t}x
        """
        return self.forward(timedeltas.unsqueeze(-1), x0).squeeze(-2)

    def forward(
        self,
        timedeltas: Tensor,  # (..., $n)
        x0: Tensor,  # (..., d)
        /,
    ) -> Tensor:  # (..., $n, d)
        r"""Propagate the linear ODE for a sequence of time-deltas.

        .. math:: step(∆tₙ, x) = e^{ρ(π(A))∆tₙ}x
        """
        # update buffer
        self.kernel = self.kernel_parametrization(self.weight)
        return linear_flow(timedeltas, x0, self.kernel, self.bias)

    def forecast(
        self,
        timestamps: Tensor,  # (..., $n),
        x0: Tensor,  # (..., d)
        *,
        t0: Tensor | float,  # (..., $n, d)
    ) -> Tensor:
        r"""Propagate the linear ODE for a sequence of timestamps.

        .. math::
            ∆tₙ &= tₙ - t₀ \\
            step(∆tₙ, x) &= e^{ρ(π(A))∆tₙ}x
        """
        return self(timestamps - t0, x0)


class LinearCell(nn.Module):
    r"""Linear innovation state update.

    .. math:: x' = x - ρ(K(x)⋅(h(x) - y))

    where $K(x)$ is a learnable innovation gain, $h$ is the observation map, and
    $ρ$ is a gate applied to the innovation correction. By default, $K$ is a
    learned constant matrix, but it can also be provided as a custom module that
    depends on the current hidden state $x$.

    The gain can be:

    - ``"constant"``: use a learned constant gain matrix. This is the default.
    - ``"attention"``: predict the gain matrix from $x$ with attention.
    - ``nn.Module``: use a custom user-provided state-dependent gain module.

    Standard gate options are:

    - ``"rezero"``: use a learnable ReZero scalar $ρ(z)=αz$ with $α$ initialized
      to zero, so that the cell starts as the identity map.
    - ``"identity"``: use $ρ(z)=z$ with no additional scaling.
    - ``None``: alias for ``"identity"``.
    - ``nn.Module``: use a custom user-provided gate.

    The observation map can be:

    - ``"linear"``: use a learned linear observation map.
    - ``"identity"``: use $h(x)=x$, which requires ``input_size == hidden_size``.
    - ``nn.Module``: use a custom user-provided observation map.
    """

    # PARAMETERS
    gain: nn.Module
    r"""MODULE: The innovation gain producing matrices $K(x)$."""
    observation_map: nn.Module
    r"""MODULE: The observation map used in the innovation term."""
    gate: nn.Module
    r"""MODULE: Optional gate for the innovation term."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        gain: str | nn.Module = "constant",
        gate: str | nn.Module | None = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate = resolve_gate(gate)

        match gain:
            case nn.Module():
                self.gain = gain
            case "constant":
                self.gain = Constant((hidden_size, input_size))
            case "attention":
                self.gain = AttentionGain(hidden_size, input_size)
            case str():
                raise ValueError(
                    "Unknown gain: "
                    f"{gain!r}. Expected 'constant', 'attention', or an nn.Module."
                )
            case _:
                raise TypeError(
                    f"gain must be a string or nn.Module, got {type(gain)!r}."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map
            case "linear":
                self.observation_map = nn.Linear(hidden_size, input_size, bias=False)
            case "identity":
                if input_size != hidden_size:
                    raise ValueError(
                        "observation_map='identity' requires input_size == hidden_size!"
                    )
                self.observation_map = nn.Identity()
            case str():
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )
            case _:
                raise TypeError(
                    "observation_map must be a string or nn.Module, "
                    f"got {type(observation_map)!r}."
                )

    def forward(self, y: Tensor, x: Tensor) -> Tensor:
        y_pred = self.observation_map(x)
        r = torch.where(y.isnan(), 0.0, y_pred - y)  # (..., input_size)
        K = self.gain(x)  # (hidden_size, input_size) or (..., hidden_size, input_size)
        correction = (r.unsqueeze(-2) @ K.mT).squeeze(-2)
        return x - self.gate(correction)


class KalmanCell(nn.Module):
    r"""Kalman-style hidden-state update with masked observations.

    .. math::
        x' = x - ρ\left(
            Σ(x)𝐃h(x)ᵀMᵀ (M(𝐃h(x)Σ(x)𝐃h(x)ᵀ + R)Mᵀ)⁻¹ (Mh(x) - y)
        \right)

    Here, $h(x)$ is the observation map, $𝐃h(x)$ is its local linearization at
    the current hidden state, and $Σ(x)$ is the hidden-state covariance. The
    masked observation model is $y_{\text{obs}} = My$, with local observation
    covariance $Σᵧᵧ(x) = 𝐃h(x)Σ(x)𝐃h(x)ᵀ + R$. In the implementation, $Σ(x)$ is
    represented through a covariance factor $L(x)$, typically a Cholesky
    factor, such that $Σ(x)=L(x)L(x)ᵀ$. The Jacobian action $𝐃h(x)L(x)$ is
    obtained by pushing the columns of $L(x)$ through the JVP of $h$ at $x$.
    $ρ$ is an optional gate applied to the Kalman correction. Standard gate
    options are the same as for `LinearInnovationCell`: ``"rezero"``,
    ``"identity"``, ``None``, or a custom `nn.Module`.

    Notes:
        LMMSE stands for linear minimum mean squared error: the best affine
        estimator under squared loss among estimators linear in the observations.
        BLUP stands for best linear unbiased predictor: the minimum-variance
        unbiased estimator within the same linear class.

    Remark:
        When $h$ is non-linear, this update uses the local Jacobian $𝐃h(x)$ at
        the current state, which is the same first-order linearization step used
        by the extended Kalman filter. In that sense, `KalmanCell` implements an
        EKF-style measurement update with a learned state covariance factor and
        optional gated correction.
    """

    observation_map: nn.Module
    r"""MODULE: Observation map $h$ from hidden to observation space."""
    covariance_factor: nn.Module
    r"""MODULE: Covariance factor $L(x)$ with $Σₓₓ(x)=L(x)L(x)ᵀ$."""
    noise_cholesky: Tensor
    r"""PARAM: Cholesky factor defining the observation noise covariance $R$."""
    gate: nn.Module
    r"""MODULE: Optional gate for the Kalman correction."""
    eye: Tensor
    r"""BUFFER: Identity matrix used to keep the covariance solve well-posed."""

    def __init__(
        self,
        /,
        input_size: int,
        hidden_size: int,
        *,
        noise: str = "scalar",
        covariance_factor: str | nn.Module = "constant",
        gate: str | nn.Module | None = "rezero",
        observation_map: str | nn.Module = "linear",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        m = self.hidden_size
        n = self.input_size
        self.gate = resolve_gate(gate)
        self.register_buffer("eye", torch.eye(n), persistent=False)

        match covariance_factor:
            case nn.Module():
                self.covariance_factor = covariance_factor

            case "constant":
                self.covariance_factor = Constant((m, m))
                register_parametrization(
                    self.covariance_factor,
                    "value",
                    surjections.CholeskyFactor(),
                )

            case "attention":
                self.covariance_factor = AttentionCovarianceFactor(m)

            case str():
                raise ValueError(
                    "Unknown covariance_factor: "
                    f"{covariance_factor!r}. Expected 'constant', 'attention', or an nn.Module."
                )

            case _:
                raise TypeError(
                    "covariance_factor must be a string or nn.Module, "
                    f"got {type(covariance_factor)!r}."
                )

        match observation_map:
            case nn.Module():
                self.observation_map = observation_map

            case "linear":
                self.observation_map = nn.Linear(hidden_size, input_size, bias=False)

            case "identity":
                if input_size != hidden_size:
                    raise ValueError(
                        "observation_map='identity' requires input_size == hidden_size!"
                    )
                self.observation_map = nn.Identity()

            case str():
                raise ValueError(
                    f"Unknown observation_map: {observation_map!r}. "
                    "Expected 'linear', 'identity', or an nn.Module."
                )

            case _:
                raise TypeError(
                    "observation_map must be a string or nn.Module, "
                    f"got {type(observation_map)!r}."
                )

        match noise:
            case "scalar":
                self.noise_cholesky = nn.Parameter(torch.zeros(()))
                register_parametrization(
                    self,
                    "noise_cholesky",
                    bijections.PositiveScalarMatrix(size=n),
                    unsafe=True,
                )

            case "diagonal":
                self.noise_cholesky = nn.Parameter(torch.normal(0, 1, size=(n,)))
                register_parametrization(
                    self,
                    "noise_cholesky",
                    bijections.PositiveDiagonal(),
                    unsafe=True,
                )

            case str():
                raise ValueError(
                    f"Unknown noise: {noise!r}. Expected 'scalar' or 'diagonal'."
                )

            case _:
                raise TypeError(f"noise must be a string, got {type(noise)!r}.")

    def forward(
        self,
        y: Tensor,  # (..., n)
        x: Tensor,  # (..., m)
    ) -> Tensor:  # (..., m)
        *batch_shape, _ = x.shape
        missing = y.isnan()

        y_pred, jvp_fn = torch.func.linearize(self.observation_map, x)

        # TODO: consider solving only over unmasked coordinates (requires flattening).
        L = self.covariance_factor(x).expand(
            *batch_shape, self.hidden_size, self.hidden_size
        )
        assert L.shape == (*batch_shape, self.hidden_size, self.hidden_size)

        # mask columns for unobserved values in cholesky factor
        J = torch.where(missing.unsqueeze(-2), self.eye, self.noise_cholesky)
        assert J.shape == (*batch_shape, self.input_size, self.input_size)

        # Restrict the residual r = Mh(x) - y_obs to the observed coordinates.
        r = torch.where(missing, torch.zeros_like(y_pred), y_pred - y)

        # Push the covariance-factor columns through 𝐃h(x) to obtain 𝐃h(x)L(x).
        batched_jvp_fn = torch.func.vmap(jvp_fn, -1, -1)
        MHL = ~missing[..., None] * batched_jvp_fn(L)  # shape: (..., n, m)
        assert MHL.shape == (*batch_shape, self.input_size, self.hidden_size)

        # u = (M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing)⁻¹r
        # note: M(𝐃h(x)LLᵀ𝐃h(x)ᵀ + JJᵀ)M + I_missing = J(𝕀 + BBᵀ)Jᵀ, B = J⁻¹M𝐃h(x)L
        # solve via: z = J⁻¹r, w = (𝕀 + BBᵀ)⁻¹z, u = J⁻ᵀw
        # middle part via woodbury: (𝕀 + BBᵀ)⁻¹ = 𝕀 - B(𝕀 + BᵀB)⁻¹Bᵀ (good if m>n)
        B = solve_triangular(J, MHL, upper=False)  # J⁻¹M𝐃h(x)L (..., n, m)
        z = solve_triangular(J, r.unsqueeze(-1), upper=False)  # J⁻¹r
        w = solve(self.eye + B @ B.mT, z)  # shape: (..., n, 1)
        u = solve_triangular(J.mT, w, upper=True).squeeze(-1)  # J⁻ᵀw (..., n)
        assert u.shape == (*batch_shape, self.input_size)

        # δ = Σₓ₞u = L(x)L(x)ᵀ𝐃h(x)ᵀu
        d = torch.einsum("...n, ...nm, ...km -> ...k", u, MHL, L)  # (..., m)

        return x - self.gate(d)


class LinODEnet_v0(nn.Module):
    r"""Encoder-Decoder Latent Linear ODE Network."""

    initial_state: Tensor
    batch_first: bool

    # submodules
    state_propagator: Callable[[Tensor, Tensor], Tensor]
    state_updater: Callable[[Tensor, Tensor], Tensor]
    encoder: Callable[[Tensor], Tensor]
    decoder: Callable[[Tensor], Tensor]

    # buffers
    prior_latent_states: Tensor  # (..., $N, L) or ($N, ..., L)
    posterior_latent_states: Tensor  # (..., $N, L) or ($N, ..., L)
    prior_predictions: Tensor  # (..., $N, D) or ($N, ..., D)
    posterior_predictions: Tensor  # (..., $N, D) or ($N, ..., D)

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        decoder: nn.Module,
        encoder: nn.Module,
        state_updater: nn.Module,
        state_propagator: nn.Module,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.encoder = encoder
        self.state_update = state_updater
        self.state_propagator = state_propagator

        self.batch_first = batch_first
        self.initial_state = nn.Parameter(torch.zeros(latent_size))
        self.register_buffer("prior_latent_states", None, persistent=False)
        self.register_buffer("posterior_latent_states", None, persistent=False)
        self.register_buffer("prior_predictions", None, persistent=False)
        self.register_buffer("posterior_predictions", None, persistent=False)

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $T, F], padded False
        context_values: Tensor,  # Float[..., $T, D], padded Nan, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        initial_state: Tensor | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> Tensor:  # (..., $T, F)
        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)
        Q = query_mask.movedim(seq_dim, 0)
        M = context_mask.movedim(seq_dim, 0)
        T0 = T[[0]] if initial_time is None else initial_time
        DT = T.diff(dim=0, prepend=T0)

        prior_states: list[Tensor] = []
        post_states: list[Tensor] = []
        prior_preds: list[Tensor] = []
        post_preds: list[Tensor] = []

        posterior_state: Tensor = (
            initial_state if initial_state is not None else self.initial_state
        )

        for delta_t, x_obs, m, q in zip(DT, X, M, Q, strict=True):
            # zₜ = flow(z(t-∆t), ∆t)
            prior_state = self.state_propagator(delta_t, posterior_state)

            # x̂ₜ = ϕ(zₜ)
            prior_prediction = self.decoder(prior_state)

            # x̂ₜ' = F(x̂ₜ, xₜ)
            posterior_prediction = self.filter(prior_prediction, x_obs)

            # zₜ' = ϕ⁻¹(x̂ₜ')
            post_state = self.encoder(posterior_prediction)

            prior_states.append(prior_state)
            post_states.append(post_state)
            prior_preds.append(prior_prediction)
            post_preds.append(posterior_prediction)

        # store buffers
        stack_dim = -2 if self.batch_first else 0
        self.prior_latent_states = torch.stack(prior_state, dim=stack_dim)
        self.posterior_latent_states = torch.stack(post_states, dim=stack_dim)
        self.prior_predictions = torch.stack(prior_preds, dim=stack_dim)
        self.posterior_predictions = torch.stack(post_preds, dim=stack_dim)

        return self.posterior_predictions

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        initial_state: Tensor | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> Tensor:  # Float[..., $K, F]
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        predictions = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )
        result = predictions[combined.query_indices]
        assert result.shape == query_mask.shape
        return result


class LinODEnet_v2(nn.Module):
    r"""Decoder-Only Latent Linear ODE Network."""

    initial_state: Tensor
    batch_first: bool

    # submodules
    state_propagator: Callable[[Tensor, Tensor], Tensor]
    state_updater: Callable[[Tensor, Tensor], Tensor]
    decoder: Callable[[Tensor], Tensor]

    # buffers
    prior_latent_states: Tensor  # (..., $N, L) or ($N, ..., L)
    posterior_latent_states: Tensor  # (..., $N, L) or ($N, ..., L)
    prior_predictions: Tensor  # (..., $N, D) or ($N, ..., D)
    posterior_predictions: Tensor  # (..., $N, D) or ($N, ..., D)

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        decoder: nn.Module,
        state_updater: nn.Module,
        state_propagator: nn.Module,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.state_update = state_updater
        self.state_propagator = state_propagator

        self.batch_first = batch_first
        self.initial_state = nn.Parameter(torch.zeros(latent_size))
        self.register_buffer("prior_latent_states", None, persistent=False)
        self.register_buffer("posterior_latent_states", None, persistent=False)
        self.register_buffer("prior_predictions", None, persistent=False)
        self.register_buffer("posterior_predictions", None, persistent=False)

    def forward(
        self,
        *,
        timestamps: Tensor,  # Float[..., $T], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $T, F], padded False
        context_values: Tensor,  # Float[..., $T, D], padded Nan, sparse
        context_mask: Tensor,  # Bool[..., $T, D], padded False
        initial_state: Tensor | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> Tensor:
        seq_dim = -2 if self.batch_first else -1
        T = timestamps[..., None].movedim(seq_dim, 0).squeeze(-1)  # ($N, ...)
        X = context_values.movedim(seq_dim, 0)  # ($N, ..., D)
        Q = query_mask.movedim(seq_dim, 0)
        M = context_mask.movedim(seq_dim, 0)
        T0 = T[[0]] if initial_time is None else initial_time
        DT = T.diff(dim=0, prepend=T0)

        prior_states: list[Tensor] = []
        post_states: list[Tensor] = []
        prior_preds: list[Tensor] = []
        post_preds: list[Tensor] = []

        posterior_state: Tensor = (
            initial_state if initial_state is not None else self.initial_state
        )

        for delta_t, x_obs, obs_mask, q in zip(DT, X, M, Q, strict=True):
            # zₜ = flow(z(t-∆t), ∆t)
            prior_state = self.state_propagator(delta_t, posterior_state)

            # zₜ' = F(zₜ, xₜ)
            posterior_state = self.state_updater(prior_state, x_obs)

            # x̂ₜ = ϕ(zₜ)
            prior_pred = self.decoder(prior_state)

            # x̂ₜ' = ϕ(zₜ')
            post_pred = self.decoder(posterior_state)

            prior_states.append(prior_state)
            post_states.append(posterior_state)
            prior_preds.append(prior_pred)
            post_preds.append(post_pred)

        stack_dim = -2 if self.batch_first else 0
        self.prior_latent_states = torch.stack(prior_states, dim=stack_dim)
        self.posterior_latent_states = torch.stack(post_states, dim=stack_dim)
        self.prior_predictions = torch.stack(prior_preds, dim=seq_dim)
        self.posterior_predictions = torch.stack(post_preds, dim=seq_dim)

        return self.posterior_predictions

    def predict(
        self,
        *,
        query_times: Tensor,  # Float[..., $K], padded NaN, non-decreasing
        query_mask: Tensor,  # Bool[..., $K, F]  padded False
        context_times: Tensor,  # Float[..., $N], padded NaN, non-decreasing
        context_mask: Tensor,  # Bool[..., $N, D], padded False
        context_values: Tensor,  # Float[..., $N, D], padded NaN, sparse
        initial_state: Tensor | None = None,  # Float[..., L]
        initial_time: Tensor | None = None,  # t₀, () or (...)
    ) -> Tensor:  # Float[..., $K, F]
        combined = EventBatch.from_request(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            batch_first=self.batch_first,
        )
        predictions = self.forward(
            timestamps=combined.timestamps,  # Float[..., $T], padded NaN, non-decreasing
            context_values=combined.context_values,  # Float[..., $T, D], padded NaN, sparse
            context_mask=combined.context_mask,  # Bool[..., $T, D], padded False
            query_mask=combined.query_mask,  # Bool[..., $T, F], padded False
            initial_state=initial_state,
            initial_time=initial_time,
        )
        result = predictions[combined.query_indices]
        assert result.shape == query_mask.shape
        return result
