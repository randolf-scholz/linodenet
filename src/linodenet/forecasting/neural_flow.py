r"""Neural Flow state propagation operators and forecasting model.

This module reimplements the flow operators used in the Neural Flows
experiments without depending on :mod:`stribor`. The operator modules follow
the structural state-propagation protocol used by ``linodenet.state_propagation``:

``forward(timedeltas, state) -> trajectory``

where ``timedeltas`` has shape ``(..., T)``, ``state`` has shape ``(..., D)``,
and the returned trajectory has shape ``(..., T, D)``. The ``NeuralFlow``
forecasting model adapts these operators to the GRU-ODE-Bayes forecasting
interface.
"""

__all__ = [
    "CouplingFlow",
    "FlowModelName",
    "GRUFlow",
    "GRUFlowBlock",
    "NeuralFlow",
    "NeuralFlowConfig",
    "ResNetFlow",
    "TimeNetName",
]


from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Final, Literal, overload

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from .gru_ode_bayes import Decoder, GRU_Bayes, GRU_ODE_Bayes

FlowModelName = Literal["coupling", "resnet", "gru"]
TimeNetName = Literal[
    "TimeLinear",
    "TimeTanh",
    "TimeFourier",
    "TimeFourierBounded",
]


class ModuleSequence[M: nn.Module](nn.ModuleList, Sequence[M]):
    r"""Wrapper for ModuleList to make it a generic Sequence type."""

    if TYPE_CHECKING:
        _modules: Mapping[str, M]  # type: ignore[override]

        def __init__(self, modules: Iterable[M] = (), /) -> None: ...
        def __iter__(self) -> Iterator[M]: ...

    @overload
    def __getitem__(self, index: int, /) -> M: ...  # pyrefly: ignore[bad-override]
    @overload
    def __getitem__(self, index: slice, /) -> ModuleSequence[M]: ...
    def __getitem__(self, index: int | slice, /) -> M | ModuleSequence[M]:  # pyright: ignore[reportIncompatibleMethodOverride]
        if isinstance(index, slice):
            modules = list(self._modules.values())
            selection = modules[index]
            return ModuleSequence(selection)
        return self._modules[self._get_abs_string_index(index)]


def _make_mlp(
    input_size: int,
    hidden_dims: Sequence[int],
    output_size: int,
    *,
    activation: type[nn.Module] = nn.ReLU,
    final_activation: nn.Module | None = None,
    spectral: bool = False,
) -> nn.Sequential:
    r"""Return a feed-forward network used by the flow blocks."""
    layers: list[nn.Module] = []
    sizes = [input_size, *hidden_dims, output_size]
    for k, (in_features, out_features) in enumerate(pairwise(sizes)):
        layer = nn.Linear(in_features, out_features)
        layers.append(spectral_norm(layer, n_power_iterations=5) if spectral else layer)
        if k < len(sizes) - 2:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class TimeEmbedding(nn.Module):
    r"""Time embedding with value zero at ``t = 0``.

    Neural flows should start as the identity at zero elapsed time.  Each
    embedding therefore omits additive biases and uses features that vanish at
    the origin.
    """

    output_size: Final[int]
    num_frequencies: Final[int | None]
    kind: Final[TimeNetName]
    weight: Tensor
    frequencies: Tensor

    def __init__(
        self,
        output_size: int,
        *,
        kind: TimeNetName = "TimeLinear",
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.num_frequencies = num_frequencies
        self.kind = kind

        match kind:
            case "TimeLinear":
                self.weight = nn.Parameter(torch.ones(output_size))
                self.register_buffer("frequencies", torch.empty(0), persistent=False)
            case "TimeTanh":
                self.weight = nn.Parameter(torch.ones(output_size))
                self.register_buffer("frequencies", torch.empty(0), persistent=False)
            case "TimeFourier" | "TimeFourierBounded":
                if num_frequencies is None:
                    num_frequencies = output_size
                self.weight = nn.Parameter(
                    torch.empty(output_size, 2 * num_frequencies)
                )
                nn.init.xavier_uniform_(self.weight)
                frequencies = torch.arange(
                    1,
                    num_frequencies + 1,
                    dtype=torch.float32,
                )
                self.register_buffer("frequencies", frequencies, persistent=True)
            case _:
                raise ValueError(f"Unknown time network {kind!r}.")

    def forward(self, timedeltas: Tensor, /) -> Tensor:
        r"""Return a time-dependent multiplicative gate."""
        if timedeltas.shape[-1] != 1:
            raise ValueError("timedeltas must have a trailing singleton dimension.")

        match self.kind:
            case "TimeLinear":
                return timedeltas * self.weight
            case "TimeTanh":
                return torch.tanh(timedeltas * self.weight)
            case "TimeFourier" | "TimeFourierBounded":
                angles = timedeltas * self.frequencies.to(timedeltas)
                features = torch.cat([torch.sin(angles), torch.cos(angles) - 1], dim=-1)
                output = torch.einsum("...h, oh -> ...o", features, self.weight)
                if self.kind == "TimeFourierBounded":
                    return torch.tanh(output)
                return output
            case _:
                raise AssertionError("unreachable")


def _prepare_inputs(
    timedeltas: Tensor, state: Tensor, input_size: int
) -> tuple[Tensor, Tensor]:
    r"""Broadcast ``timedeltas`` and ``state`` to trajectory tensors."""
    if state.shape[-1] != input_size:
        raise ValueError(
            f"state has incompatible last dimension: expected {input_size}, "
            f"got {state.shape[-1]}."
        )
    if timedeltas.ndim < 1:
        raise ValueError("timedeltas must have at least one dimension.")

    batch_shape = torch.broadcast_shapes(timedeltas.shape[:-1], state.shape[:-1])
    num_steps = timedeltas.shape[-1]
    timedeltas = timedeltas.expand(*batch_shape, num_steps).unsqueeze(-1)
    state = state.expand(*batch_shape, input_size).unsqueeze(-2)
    state = state.expand(*batch_shape, num_steps, input_size)
    return timedeltas, state


def _ordered_mask(input_size: int, layer_index: int, /) -> Tensor:
    r"""Return the alternating ordered mask used by coupling layers."""
    if input_size == 1:
        return torch.zeros(input_size, dtype=torch.bool)
    split = (input_size + 1) // 2
    mask = torch.zeros(input_size, dtype=torch.bool)
    if layer_index % 2 == 0:
        mask[:split] = True
    else:
        mask[split:] = True
    return mask


class CouplingFlowBlock(nn.Module):
    r"""Continuous affine coupling block.

    The block uses an affine update whose shift and log-scale vanish at
    ``t = 0`` through a time gate:

    ``y_B = exp(s(x_A, t)) * x_B + b(x_A, t)``.
    """

    input_size: Final[int]
    mask: Tensor
    net: nn.Sequential
    time_net: TimeEmbedding

    def __init__(
        self,
        input_size: int,
        hidden_dims: Sequence[int],
        *,
        layer_index: int,
        time_net: TimeNetName,
        time_hidden_size: int | None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.net = _make_mlp(input_size + 1, hidden_dims, 2 * input_size)
        self.time_net = TimeEmbedding(
            2 * input_size,
            kind=time_net,
            num_frequencies=time_hidden_size,
        )
        self.register_buffer(
            "mask",
            _ordered_mask(input_size, layer_index),
            persistent=True,
        )

    def _affine_parameters(
        self, x: Tensor, timedeltas: Tensor
    ) -> tuple[Tensor, Tensor]:
        mask = self.mask.to(device=x.device)
        conditioner = torch.where(mask, x, torch.zeros_like(x))
        params = self.net(torch.cat([conditioner, timedeltas], dim=-1))
        params = params * self.time_net(timedeltas)
        shift, log_scale = params.chunk(2, dim=-1)
        log_scale = 0.8 * torch.tanh(log_scale)
        active = ~mask
        shift = torch.where(active, shift, torch.zeros_like(shift))
        log_scale = torch.where(active, log_scale, torch.zeros_like(log_scale))
        return shift, log_scale

    def forward(self, x: Tensor, timedeltas: Tensor, /) -> Tensor:
        shift, log_scale = self._affine_parameters(x, timedeltas)
        return torch.exp(log_scale) * x + shift

    def inverse(self, y: Tensor, timedeltas: Tensor, /) -> Tensor:
        shift, log_scale = self._affine_parameters(y, timedeltas)
        return (y - shift) * torch.exp(-log_scale)


class CouplingFlow(ModuleSequence[CouplingFlowBlock]):
    r"""Affine coupling neural flow.

    Args:
        input_size: State dimension.
        num_layers: Number of coupling blocks.
        hidden_dims: Hidden dimensions of each coupling network.
        time_net: Time embedding family.
        time_hidden_size: Number of Fourier frequencies for Fourier time nets.
    """

    input_shape: Final[tuple[int, ...]]
    input_size: Final[int]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        *,
        num_layers: int,
        hidden_dims: Sequence[int] = (),
        time_net: TimeNetName = "TimeLinear",
        time_hidden_size: int | None = None,
    ) -> None:
        blocks = [
            CouplingFlowBlock(
                input_size,
                hidden_dims,
                layer_index=k,
                time_net=time_net,
                time_hidden_size=time_hidden_size,
            )
            for k in range(num_layers)
        ]
        super().__init__(blocks)
        self.input_shape = (input_size,)
        self.input_size = input_size
        self.num_layers = num_layers

    def step(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        return self.forward(timedeltas.unsqueeze(-1), state).squeeze(-2)

    def forward(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in self:
            x = block(x, timedeltas)
        return x

    def inverse(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Apply the inverse flow for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in reversed(self):
            x = block.inverse(x, timedeltas)
        return x


class ResNetFlowBlock(nn.Module):
    r"""Time-gated residual flow block."""

    input_size: Final[int]
    residual: nn.Sequential
    time_net: TimeEmbedding

    def __init__(
        self,
        input_size: int,
        *,
        hidden_dims: Sequence[int],
        time_net: TimeNetName,
        time_hidden_size: int | None,
        invertible: bool,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.residual = _make_mlp(
            input_size + 1,
            hidden_dims,
            input_size,
            spectral=invertible,
        )
        self.time_net = TimeEmbedding(
            input_size,
            kind=time_net,
            num_frequencies=time_hidden_size,
        )
        self.residual_scale = 0.8 if invertible else 1.0

    def forward(self, x: Tensor, timedeltas: Tensor, /) -> Tensor:
        residual = self.residual(torch.cat([x, timedeltas], dim=-1))
        residual = self.residual_scale * torch.tanh(residual)
        return x + self.time_net(timedeltas) * residual

    def inverse(
        self, y: Tensor, timedeltas: Tensor, /, *, iterations: int = 100
    ) -> Tensor:
        x = y
        for _ in range(iterations):
            residual = self.residual(torch.cat([x, timedeltas], dim=-1))
            residual = self.residual_scale * torch.tanh(residual)
            x = y - self.time_net(timedeltas) * residual
        return x


class ResNetFlow(ModuleSequence[ResNetFlowBlock]):
    r"""Residual neural flow."""

    input_shape: Final[tuple[int, ...]]
    input_size: Final[int]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        *,
        num_layers: int,
        hidden_dims: Sequence[int] = (),
        time_net: TimeNetName = "TimeLinear",
        time_hidden_size: int | None = None,
        invertible: bool = True,
    ) -> None:
        blocks = [
            ResNetFlowBlock(
                input_size,
                hidden_dims=hidden_dims,
                time_net=time_net,
                time_hidden_size=time_hidden_size,
                invertible=invertible,
            )
            for _ in range(num_layers)
        ]
        super().__init__(blocks)
        self.input_shape = (input_size,)
        self.input_size = input_size
        self.num_layers = num_layers

    def step(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        return self.forward(timedeltas.unsqueeze(-1), state).squeeze(-2)

    def forward(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in self:
            x = block(x, timedeltas)
        return x

    def inverse(
        self,
        timedeltas: Tensor,
        state: Tensor,
        /,
        *,
        iterations: int = 100,
    ) -> Tensor:
        r"""Apply the fixed-point inverse for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in reversed(self):
            x = block.inverse(x, timedeltas, iterations=iterations)
        return x


class GRUFlowBlock(nn.Module):
    r"""Single invertible GRU-flow block."""

    input_size: Final[int]
    alpha: Final[float]
    beta: Final[float]
    lin_hh: nn.Module
    lin_hz: nn.Module
    lin_hr: nn.Module
    time_net: TimeEmbedding

    def __init__(
        self,
        input_size: int,
        *,
        time_net: TimeNetName,
        time_hidden_size: int | None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.lin_hh = spectral_norm(
            nn.Linear(input_size + 1, input_size),
            n_power_iterations=5,
        )
        self.lin_hz = spectral_norm(
            nn.Linear(input_size + 1, input_size),
            n_power_iterations=5,
        )
        self.lin_hr = spectral_norm(
            nn.Linear(input_size + 1, input_size),
            n_power_iterations=5,
        )
        self.time_net = TimeEmbedding(
            input_size,
            kind=time_net,
            num_frequencies=time_hidden_size,
        )
        self.alpha = 2 / 5
        self.beta = 4 / 5

    def residual(self, state: Tensor, timedeltas: Tensor, /) -> Tensor:
        r"""Return the GRU-style residual update."""
        inp = torch.cat([state, timedeltas], dim=-1)
        reset = self.beta * torch.sigmoid(self.lin_hr(inp))
        update = self.alpha * torch.sigmoid(self.lin_hz(inp))
        candidate = torch.tanh(
            self.lin_hh(torch.cat([reset * state, timedeltas], dim=-1))
        )
        return update * (candidate - state)

    def forward(self, state: Tensor, timedeltas: Tensor, /) -> Tensor:
        r"""Apply one GRU-flow block."""
        return state + self.time_net(timedeltas) * self.residual(state, timedeltas)

    def inverse(
        self,
        state: Tensor,
        timedeltas: Tensor,
        /,
        *,
        iterations: int = 100,
    ) -> Tensor:
        r"""Invert the block by fixed-point iteration."""
        x = state
        for _ in range(iterations):
            x = state - self.time_net(timedeltas) * self.residual(x, timedeltas)
        return x


class GRUFlow(ModuleSequence[GRUFlowBlock]):
    r"""GRU neural flow."""

    input_shape: Final[tuple[int, ...]]
    input_size: Final[int]
    num_layers: Final[int]

    def __init__(
        self,
        input_size: int,
        *,
        num_layers: int,
        time_net: TimeNetName = "TimeLinear",
        time_hidden_size: int | None = None,
    ) -> None:
        blocks = [
            GRUFlowBlock(
                input_size,
                time_net=time_net,
                time_hidden_size=time_hidden_size,
            )
            for _ in range(num_layers)
        ]
        super().__init__(blocks)
        self.input_shape = (input_size,)
        self.input_size = input_size
        self.num_layers = num_layers

    def step(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        return self.forward(timedeltas.unsqueeze(-1), state).squeeze(-2)

    def forward(self, timedeltas: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in self:
            x = block(x, timedeltas)
        return x

    def inverse(
        self,
        timedeltas: Tensor,
        state: Tensor,
        /,
        *,
        iterations: int = 100,
    ) -> Tensor:
        r"""Apply the fixed-point inverse for each requested time delta."""
        timedeltas, x = _prepare_inputs(timedeltas, state, self.input_size)
        for block in reversed(self):
            x = block.inverse(x, timedeltas, iterations=iterations)
        return x


@dataclass(frozen=True, slots=True, kw_only=True)
class NeuralFlowConfig:
    r"""Configuration for constructing a neural-flow forecasting model."""

    input_size: int
    hidden_size: int
    decoder_hidden_size: int
    feature_embedding_size: int
    flow_model: FlowModelName = "gru"
    flow_layers: int = 1
    flow_hidden_layers: int = 1
    time_net: TimeNetName = "TimeLinear"
    time_hidden_size: int | None = 1
    invertible: bool = True
    bias: bool = True
    dropout_rate: float = 0.0


class NeuralFlow(GRU_ODE_Bayes):
    r"""GRU-ODE-Bayes forecasting model with neural-flow state propagation.

    This matches the forecasting setup in the Neural Flows experiments: the
    decoder and Bayesian GRU jump are unchanged from GRU-ODE-Bayes, while the
    continuous ODE solver is replaced by a learned flow on the latent state.
    """

    @classmethod
    def from_config(
        cls,
        config: NeuralFlowConfig | Mapping[str, Any],
        /,
    ) -> NeuralFlow:
        r"""Construct a neural-flow forecasting model from a configuration."""
        if isinstance(config, Mapping):
            config = NeuralFlowConfig(**config)

        if config.flow_layers < 1:
            raise ValueError("flow_layers must be positive.")
        if config.flow_hidden_layers < 0:
            raise ValueError("flow_hidden_layers must be non-negative.")

        decoder = Decoder(
            config.input_size,
            config.hidden_size,
            config.decoder_hidden_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
        )
        update_cell = GRU_Bayes(
            config.input_size,
            config.hidden_size,
            config.feature_embedding_size,
            bias=config.bias,
        )
        hidden_dims = [config.hidden_size] * config.flow_hidden_layers

        match config.flow_model:
            case "coupling":
                flow = CouplingFlow(
                    config.hidden_size,
                    num_layers=config.flow_layers,
                    hidden_dims=hidden_dims,
                    time_net=config.time_net,
                    time_hidden_size=config.time_hidden_size,
                )
            case "resnet":
                flow = ResNetFlow(
                    config.hidden_size,
                    num_layers=config.flow_layers,
                    hidden_dims=hidden_dims,
                    time_net=config.time_net,
                    time_hidden_size=config.time_hidden_size,
                    invertible=config.invertible,
                )
            case "gru":
                flow = GRUFlow(
                    config.hidden_size,
                    num_layers=config.flow_layers,
                    time_net=config.time_net,
                    time_hidden_size=config.time_hidden_size,
                )
            case _:
                raise ValueError(f"Unknown neural flow model {config.flow_model!r}.")

        return NeuralFlow(
            config.input_size,
            config.hidden_size,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        input_size: int,
        hidden_size: int,
        decoder_hidden_size: int,
        feature_embedding_size: int,
        flow_model: FlowModelName = "gru",
        flow_layers: int = 1,
        flow_hidden_layers: int = 1,
        time_net: TimeNetName = "TimeLinear",
        time_hidden_size: int | None = 1,
        invertible: bool = True,
        bias: bool = True,
        dropout_rate: float = 0.0,
    ) -> NeuralFlow:
        r"""Construct a neural-flow forecasting model from hyperparameters."""
        return NeuralFlow.from_config(
            NeuralFlowConfig(
                input_size=input_size,
                hidden_size=hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                feature_embedding_size=feature_embedding_size,
                flow_model=flow_model,
                flow_layers=flow_layers,
                flow_hidden_layers=flow_hidden_layers,
                time_net=time_net,
                time_hidden_size=time_hidden_size,
                invertible=invertible,
                bias=bias,
                dropout_rate=dropout_rate,
            )
        )

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        decoder: nn.Module,
        flow: CouplingFlow | ResNetFlow | GRUFlow,
        update_cell: nn.Module,
    ) -> None:
        r"""Initialize the neural-flow forecasting model."""
        super().__init__(
            input_size,
            hidden_size,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

    def propagate_state(
        self,
        delta_time: Tensor,
        posterior_state: Tensor,
    ) -> Tensor:
        r"""Propagate a posterior state through the neural flow."""
        trajectory = self.flow(delta_time.unsqueeze(-1), posterior_state)
        return trajectory.squeeze(-2)
