r"""Minimal, unoptimized reimplementation of LinODEnet."""

__all__ = [
    "LinODEnet_v0",
    "LinODEnet",
    "LinearFlow",
    "ReZero",
    "make_linodenet",
    "linear_flow",
]

from collections.abc import Callable, Mapping
from typing import Any, Final, Optional, cast

import torch
from torch import Tensor, nn

from .state_update import GradientStepUpdater, InnovationCell
from .utils import EventBatch


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


class Symmetric(nn.Module):
    r"""Symmetric parametrization of the kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return (x + x.mT) / 2


class SkewSymmetric(nn.Module):
    r"""Skew-symmetric parametrization of the kernel."""

    def forward(self, x: Tensor) -> Tensor:
        return (x - x.mT) / 2


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

        # initialize parameters
        match kernel_initialization:
            case None:
                kernel = torch.randn(input_size, input_size)
            case "zero":
                kernel = torch.zeros(input_size, input_size)
            case "skew-symmetric":
                kernel = torch.randn(input_size, input_size)
                kernel = (kernel - kernel.mT) / 2
            case "symmetric":
                kernel = torch.randn(input_size, input_size)
                kernel = (kernel + kernel.mT) / 2
            case _:
                raise ValueError

        self.weight = nn.Parameter(kernel)
        self.register_parameter(
            "bias",
            nn.Parameter(torch.zeros(input_size)) if self.use_bias else None,
        )

        # apply parametrization if given
        match kernel_parametrization:
            case None | "identity":
                parametrization = nn.Identity()
            case "symmetric":
                parametrization = Symmetric()
            case "skew-symmetric":
                parametrization = SkewSymmetric()
            case _:
                raise NotImplementedError

        self.kernel_parametrization = nn.Sequential(
            parametrization,
            ReZero() if self.use_rezero else nn.Identity(),
        )

        # initialize buffers
        self.register_buffer("kernel", self.kernel_parametrization(self.weight))

    def forward(
        self,
        timedeltas: Tensor,  # (...)
        x0: Tensor,  # (..., d)
        /,
    ) -> Tensor:  # (..., d)
        r"""Propagate the linear ODE for a single time-delta.

        .. math:: step(∆t, x) = e^{ρ(π(A))∆t}x
        """
        return self.propagate(timedeltas.unsqueeze(-1), x0).squeeze(-2)

    def propagate(
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
        self.input_size = input_size
        self.latent_size = latent_size

        self.decoder = decoder
        self.encoder = encoder
        self.state_updater = state_updater
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
            posterior_prediction = self.state_updater(x_obs, prior_prediction)

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


class LinODEnet(nn.Module):
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
        self.input_size = input_size
        self.latent_size = latent_size

        self.decoder = decoder
        self.state_updater = state_updater
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
        valid_steps = (M | Q).any(dim=-1)
        _, *batch_shape = T.shape

        prior_states: list[Tensor] = []
        post_states: list[Tensor] = []
        prior_preds: list[Tensor] = []
        post_preds: list[Tensor] = []

        posterior_state: Tensor = (
            initial_state
            if initial_state is not None
            else self.initial_state.expand(*batch_shape, self.latent_size)
        )

        for delta_t, x_obs, obs_mask, active in zip(DT, X, M, valid_steps, strict=True):
            # zₜ = flow(z(t-∆t), ∆t)
            # prior_state = self.state_propagator(delta_t, posterior_state)
            prior_state = update_masked(
                posterior_state,
                self.state_propagator,
                args=(delta_t, posterior_state),
                batch_mask=active,
            )

            # zₜ' = F(zₜ, xₜ)
            posterior_state = self.state_updater(x_obs, prior_state, mask=obs_mask)

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


def make_linodenet(
    *,
    linodenet: Mapping[str, Any],
    state_updater: Mapping[str, Any],
    state_propagator: Mapping[str, Any],
    decoder: Mapping[str, Any],
) -> LinODEnet:
    r"""Instantiate :class:`LinODEnet` from constructor kwargs mappings."""
    decoder_module = nn.Linear(**dict(decoder))
    updater = GradientStepUpdater(decoder=decoder_module, **dict(state_updater))
    propagator = LinearFlow(**dict(state_propagator))
    return LinODEnet(
        decoder=decoder_module,
        state_updater=updater,
        state_propagator=propagator,
        **dict(linodenet),
    )


def update_masked(
    target: Tensor,  # (..., *e)
    fn: Callable[..., Tensor],  # [*(..., *dᵢ)] -> (..., *e)
    /,
    *,
    args: tuple[Tensor, ...],
    batch_mask: Tensor,  # Bool[...]
) -> Tensor:  # (..., *e)
    r"""Update ``target`` with ``fn`` applied to selected batch elements."""
    return target.masked_scatter(
        batch_mask.reshape(*batch_mask.shape, *(1,) * (target.ndim - batch_mask.ndim)),
        fn(*(x[batch_mask] for x in args)),
    )
