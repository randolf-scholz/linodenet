r"""Neural Flow state propagation operators and forecasting model.

This module reimplements the flow operators used in the Neural Flows
experiments without depending on :mod:`stribor`. The operator modules follow
the single-step propagation interface used by ``GRU_ODE_Bayes``:

``forward(delta_time, state) -> state``

where ``delta_time`` has shape ``(...)`` and ``state`` has shape ``(..., D)``.
The ``NeuralFlow`` forecasting model adapts these operators to the
GRU-ODE-Bayes forecasting interface.
"""

__all__ = [
    "CouplingFlow",
    "CouplingFlowBlock",
    "FlowModelName",
    "GRUFlow",
    "GRUFlowBlock",
    "ModuleSequence",
    "NeuralFlow",
    "NeuralFlowConfig",
    "ResNetFlow",
    "ResNetFlowBlock",
    "TimeEmbedding",
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

        # noinspection PyMissingConstructor
        def __init__(self, _: Iterable[M] = (), /) -> None: ...
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
    num_frequencies: Final[int]
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
        if num_frequencies is None:
            num_frequencies = output_size

        self.output_size = output_size
        self.num_frequencies = num_frequencies
        self.kind = kind

        n = self.output_size
        k = self.num_frequencies

        match kind:
            case "TimeLinear":
                self.weight = nn.Parameter(torch.ones(n))
                self.register_buffer("frequencies", None)
            case "TimeTanh":
                self.weight = nn.Parameter(torch.ones(n))
                self.register_buffer("frequencies", None)
            case "TimeFourier" | "TimeFourierBounded":
                self.weight = nn.Parameter(torch.empty(n, 2 * k))
                nn.init.xavier_uniform_(self.weight)
                frequencies = torch.arange(1, k + 1, dtype=torch.get_default_dtype())
                self.register_buffer("frequencies", frequencies)
            case _:
                raise ValueError(f"Unknown time network {kind!r}.")

    def forward(self, deltas: Tensor, /) -> Tensor:
        r"""Return a time-dependent multiplicative gate."""
        # deltas: (...), output: (..., E)
        match self.kind:
            case "TimeLinear":
                return deltas[..., None] * self.weight
            case "TimeTanh":
                return torch.tanh(deltas[..., None] * self.weight)
            case "TimeFourier":
                angles = deltas[..., None] * self.frequencies  # (..., F)
                features = torch.cat([torch.sin(angles), torch.cos(angles) - 1], dim=-1)
                output = torch.einsum("...h, oh -> ...o", features, self.weight)
                return output
            case "TimeFourierBounded":
                angles = deltas[..., None] * self.frequencies  # (..., F)
                features = torch.cat([torch.sin(angles), torch.cos(angles) - 1], dim=-1)
                output = torch.einsum("...h, oh -> ...o", features, self.weight)
                return torch.tanh(output)
            case _:
                raise AssertionError("unreachable")


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
        self.register_buffer("mask", _ordered_mask(input_size, layer_index))

    def _affine_parameters(self, delta: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        # delta: (...), x: (..., H)
        conditioner = torch.where(self.mask, x, torch.zeros_like(x))  # (..., H)
        inputs = torch.cat([conditioner, delta[..., None]], dim=-1)  # (..., H+1)
        params = self.net(inputs) * self.time_net(delta)  # (..., 2H)
        shift, log_scale = params.chunk(2, dim=-1)  # (..., H), (..., H)
        log_scale = 0.8 * torch.tanh(log_scale)
        shift = torch.where(~self.mask, shift, torch.zeros_like(shift))
        log_scale = torch.where(~self.mask, log_scale, torch.zeros_like(log_scale))
        return shift, log_scale

    def forward(self, delta: Tensor, x: Tensor, /) -> Tensor:
        # [delta=(...), state=(..., H)] -> (..., H)
        shift, log_scale = self._affine_parameters(delta, x)
        return torch.exp(log_scale) * x + shift

    def inverse(self, delta: Tensor, y: Tensor, /) -> Tensor:
        # [delta=(...), state=(..., H)] -> (..., H)
        shift, log_scale = self._affine_parameters(delta, y)
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

    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in self:
            state = block(delta, state)
        return state

    def inverse(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Apply the inverse flow for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in reversed(self):
            state = block.inverse(delta, state)
        return state


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

    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        # [delta=(...), state=(..., H)] -> (..., H)
        inputs = torch.cat([state, delta[..., None]], dim=-1)  # (..., H+1)
        residual = self.residual(inputs)
        residual = self.residual_scale * torch.tanh(residual)
        return state + self.time_net(delta) * residual

    def inverse(self, delta: Tensor, y: Tensor, /, *, iterations: int = 100) -> Tensor:
        # [delta=(...), state=(..., H)] -> (..., H)
        for _ in range(iterations):
            inputs = torch.cat([y, delta[..., None]], dim=-1)  # (..., H+1)
            residual = self.residual(inputs)
            residual = self.residual_scale * torch.tanh(residual)
            y = y - self.time_net(delta) * residual
        return y


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

    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in self:
            state = block(delta, state)
        return state

    def inverse(
        self, delta: Tensor, state: Tensor, /, *, iterations: int = 100
    ) -> Tensor:
        r"""Apply the fixed-point inverse for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in reversed(self):
            state = block.inverse(delta, state, iterations=iterations)
        return state


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

    def residual(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Return the GRU-style residual update."""
        # delta: (..., 1), state: (..., H)
        inp = torch.cat([state, delta], dim=-1)  # (..., H+1)
        reset = self.beta * torch.sigmoid(self.lin_hr(inp))
        update = self.alpha * torch.sigmoid(self.lin_hz(inp))
        candidate_inp = torch.cat([reset * state, delta], dim=-1)  # (..., H+1)
        candidate = torch.tanh(self.lin_hh(candidate_inp))
        return update * (candidate - state)

    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Apply one GRU-flow block."""
        # [delta=(...), state=(..., H)] -> (..., H)
        return state + self.time_net(delta) * self.residual(delta[..., None], state)

    def inverse(
        self, delta: Tensor, state: Tensor, /, *, iterations: int = 100
    ) -> Tensor:
        r"""Invert the block by fixed-point iteration."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for _ in range(iterations):
            state = state - self.time_net(delta) * self.residual(
                delta[..., None], state
            )
        return state


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

    def forward(self, delta: Tensor, state: Tensor, /) -> Tensor:
        r"""Propagate ``state`` for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in self:
            state = block(delta, state)
        return state

    def inverse(
        self, delta: Tensor, state: Tensor, /, *, iterations: int = 100
    ) -> Tensor:
        r"""Apply the fixed-point inverse for a single time delta."""
        # [delta=(...), state=(..., H)] -> (..., H)
        for block in reversed(self):
            state = block.inverse(delta, state, iterations=iterations)
        return state


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
