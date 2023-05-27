r"""Contains implementations of ODE models."""

__all__ = [
    # Classes
    "LinODE",
    "LinODEnet",
]

import logging
import warnings
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import Initialization
from linodenet.models.embeddings import ConcatEmbedding, ConcatProjection
from linodenet.models.encoders import ResNet
from linodenet.models.filters import Filter, RecurrentCellFilter
from linodenet.models.system import LinODECell
from linodenet.projections import Projection
from linodenet.utils import deep_dict_update, initialize_from_config, pad

__logger__ = logging.getLogger(__name__)


class LinODE(nn.Module):
    r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "cell": LinODECell.HP,
        "kernel_initialization": None,
        "kernel_projection": None,
    }
    r"""Dictionary of hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # Buffers
    xhat: Tensor
    r"""BUFFER: The forward prediction."""

    # Parameters
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""

    # Functions
    kernel_initialization: Initialization
    r"""FUNC: Parameter-less function that draws a initial system matrix."""
    kernel_projection: Projection
    r"""FUNC: Regularization function for the kernel."""

    def __init__(
        self,
        input_size: int,
        **cfg: Any,
    ):
        super().__init__()
        config = deep_dict_update(self.HP, cfg)

        config["cell"]["input_size"] = input_size

        self.input_size = input_size
        self.output_size = input_size
        self.cell: nn.Module = initialize_from_config(config["cell"])

        # Buffers
        self.register_buffer("xhat", torch.tensor(()), persistent=False)
        assert isinstance(self.cell.kernel, Tensor)
        self.register_buffer("kernel", self.cell.kernel, persistent=False)

    @jit.export
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., N), (..., d)] -> (..., N, d)``.

        Returns the estimated true state of the system at the times $t∈T$.
        """
        DT = torch.moveaxis(torch.diff(T), -1, 0)
        X: list[Tensor] = [x0]

        # iterate over LEN, this works even when no BATCH dim present.
        for dt in DT:
            X.append(self.cell(dt, X[-1]))

        # shape: [LEN, ..., DIM]
        Xhat = torch.stack(X, dim=0)
        # shape: [..., LEN, DIM]
        self.xhat = torch.moveaxis(Xhat, 0, -2)

        return self.xhat


class LatentLinODECell(nn.Module):
    """Latent Linear ODE Cell."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "latent_size": None,
        "output_size": None,
        "System": LinODECell.HP,
        "Embedding": ConcatEmbedding.HP,
        "Projection": ConcatProjection.HP,
        "Filter": RecurrentCellFilter.HP | {"autoregressive": True},
        "Encoder": ResNet.HP,
        "Decoder": ResNet.HP,
    }
    r"""Dictionary of Hyperparameters."""

    # CONSTANTS
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    latent_size: Final[int]
    r"""CONST: The dimensionality of the linear ODE."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the padding."""
    padding_size: Final[int]
    r"""CONST: The dimensionality of the padded state."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    validate_inputs: Final[bool]
    r"""CONST: Whether to validate the inputs."""

    # BUFFERS
    x_pre: Tensor
    r"""BUFFER: Stores pre-jump values."""
    x_post: Tensor
    r"""BUFFER: Stores post-jump values."""
    z_pre: Tensor
    r"""BUFFER: Stores pre-jump latent values."""
    z_post: Tensor
    r"""BUFFER: Stores post-jump latent values."""
    dt: Tensor
    r"""BUFFER: Stores the timedelta values."""

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: Optional[int] = None,
        **cfg: Any,
    ) -> None:
        # LOGGER = __logger__.getChild(self.__class__.__name__)
        super().__init__()

        config = deep_dict_update(self.HP, cfg)
        self.validate_inputs = config.get("validate_inputs", False)

        hidden_size = hidden_size if hidden_size is not None else input_size
        if hidden_size < input_size:
            warnings.warn(
                "hidden_size < input_size. Setting hidden_size=input_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            hidden_size = input_size

        # CONSTANTS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = input_size
        self.padding_size = self.hidden_size - self.input_size

        # BUFFERS
        self.register_buffer("x_pre", torch.tensor(()), persistent=False)
        self.register_buffer("x_post", torch.tensor(()), persistent=False)
        self.register_buffer("z_pre", torch.tensor(()), persistent=False)
        self.register_buffer("z_post", torch.tensor(()), persistent=False)
        self.register_buffer("dt", torch.tensor(()), persistent=False)

        # Submodules
        self.embedding: nn.Module = initialize_from_config(config["Embedding"])
        self.encoder: nn.Module = initialize_from_config(config["Encoder"])
        self.system: nn.Module = initialize_from_config(config["System"])
        self.decoder: nn.Module = initialize_from_config(config["Decoder"])
        self.projection: nn.Module = initialize_from_config(config["Projection"])
        self.filter: Filter = initialize_from_config(config["Filter"])

        # Parameters
        self.kernel = self.system.kernel
        self.z0 = nn.Parameter(torch.randn(self.latent_size))

    def forward(self, x_obs: Tensor, z: Tensor, dt: Tensor) -> Tensor:
        """Propagate the latent state forward in time.

        .. Signature:: ``[(..., N), (..., d), (...,)] -> (..., N, d)``.

        Args:
            x_obs: The observation at the current time step. May contain NaNs.
            z: The latent state at the current time step. Must not contain NaNs.
            dt: The time delta between the time of z and the time of x_obs.

        Note:
            Contrary to a standard RNNCell, the LatentLinODECell requires an optional time step `dt` to be passed.
        """
        # Store the time delta.
        self.dt = dt

        # Decode the latent state at the observation time.
        # (..., LAT) -> (..., DIM)
        self.x_pre = self.projection(self.decoder(z))

        # Update the state estimate by filtering the observation.
        # (..., DIM), (..., DIM) → (..., DIM)
        self.x_post = self.filter(x_obs, self.x_pre)

        # Encode the latent state at the observation time.
        # (..., DIM) → (..., LAT)
        self.z_post = self.encoder(self.embedding(self.x_post))

        # Propagate the latent state forward in time.
        # (...,), (..., LAT) -> (..., LAT)
        self.z_pre = self.system(self.dt, self.z_post)

        return self.z_post


class LinODEnet(nn.Module):
    r"""Linear ODE Network.

    +-------------------------------------------------+-------------------+
    | Component                                       | Formula           |
    +=================================================+===================+
    | Decoder π (default: :class:`~iResNet`)          | xᵢ  =  π(zᵢ)      |
    +-------------------------------------------------+-------------------+
    | Filter  F (default: :class:`~torch.nn.GRUCell`) | xᵢ' = F(xᵢ, oᵢ)   |
    +-------------------------------------------------+-------------------+
    | Encoder Φ (default: :class:`~iResNet`)          | zᵢ' = Φ(xᵢ')      |
    +-------------------------------------------------+-------------------+
    | System  S (default: :class:`~LinODECell`)       | zᵢ₊₁ = S(zᵢ, ∆tᵢ) |
    +-------------------------------------------------+-------------------+
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": None,
        "hidden_size": None,
        "latent_size": None,
        "output_size": None,
        "System": LinODECell.HP,
        "Embedding": ConcatEmbedding.HP,
        "Projection": ConcatProjection.HP,
        "Filter": RecurrentCellFilter.HP | {"autoregressive": True},
        "Encoder": ResNet.HP,
        "Decoder": ResNet.HP,
    }
    r"""Dictionary of Hyperparameters."""

    # Constants
    name: Final[str] = __name__
    r"""str: The name of the model."""
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    latent_size: Final[int]
    r"""CONST: The dimensionality of the linear ODE."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the padding."""
    padding_size: Final[int]
    r"""CONST: The dimensionality of the padded state."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    validate_inputs: Final[bool]
    r"""CONST: Whether to validate the inputs."""

    # Buffers
    ZERO: Tensor
    r"""BUFFER: A tensor of value float(0.0)"""
    NAN: Tensor
    r"""BUFFER: A tensor of value float('nan')"""
    x_pre: Tensor
    r"""BUFFER: Stores pre-jump values."""
    x_post: Tensor
    r"""BUFFER: Stores post-jump values."""
    z_pre: Tensor
    r"""BUFFER: Stores pre-jump latent values."""
    z_post: Tensor
    r"""BUFFER: Stores post-jump latent values."""
    timedeltas: Tensor
    r"""BUFFER: Stores the timedelta values."""
    predictions: Tensor
    r"""BUFFER: Stores the predictions."""

    # Parameters:
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""
    z0: Tensor
    r"""PARAM: Learnable initial latent state."""

    # Sub-Modules
    # encoder: Any
    # r"""MODULE: Responsible for embedding `x̂→ẑ`."""
    # embedding: nn.Module
    # r"""MODULE: Responsible for embedding `x̂→ẑ`."""
    # system: nn.Module
    # r"""MODULE: Responsible for propagating `ẑ_t→ẑ_{t+∆t}`."""
    # decoder: nn.Module
    # r"""MODULE: Responsible for projecting `ẑ→x̂`."""
    # projection: nn.Module
    # r"""MODULE: Responsible for projecting `ẑ→x̂`."""
    # filter: nn.Module
    # r"""MODULE: Responsible for updating `(x̂, x_obs) →x̂'`."""

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: Optional[int] = None,
        **cfg: Any,
    ) -> None:
        super().__init__()

        # Fix the hidden size
        hidden_size = hidden_size if hidden_size is not None else input_size
        if hidden_size < input_size:
            warnings.warn(
                "hidden_size < input_size. Setting hidden_size=input_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            hidden_size = input_size

        # Config
        config = deep_dict_update(self.HP, cfg)
        config["Encoder"]["input_size"] = latent_size
        config["Decoder"]["input_size"] = latent_size
        config["System"]["input_size"] = latent_size
        config["Filter"]["input_size"] = hidden_size
        config["Filter"]["hidden_size"] = hidden_size
        config["Embedding"]["input_size"] = hidden_size
        config["Embedding"]["output_size"] = latent_size
        config["Projection"]["input_size"] = latent_size
        config["Projection"]["output_size"] = hidden_size

        # Constants
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = input_size
        self.padding_size = self.hidden_size - self.input_size
        self.validate_inputs = config.get("validate_inputs", False)

        # Buffers
        self.register_buffer("ZERO", torch.tensor(0.0), persistent=True)
        self.register_buffer("NAN", torch.tensor(float("nan")), persistent=True)
        self.register_buffer("timedeltas", torch.tensor(()), persistent=False)
        self.register_buffer("x_pre", torch.tensor(()), persistent=False)
        self.register_buffer("x_post", torch.tensor(()), persistent=False)
        self.register_buffer("z_pre", torch.tensor(()), persistent=False)
        self.register_buffer("z_post", torch.tensor(()), persistent=False)
        self.register_buffer("predictions", torch.tensor(()), persistent=False)

        # Submodules
        self.embedding: nn.Module = initialize_from_config(config["Embedding"])
        self.encoder: nn.Module = initialize_from_config(config["Encoder"])
        self.system: nn.Module = initialize_from_config(config["System"])
        self.decoder: nn.Module = initialize_from_config(config["Decoder"])
        self.projection: nn.Module = initialize_from_config(config["Projection"])
        self.filter: Filter = initialize_from_config(config["Filter"])

        # Parameters
        assert isinstance(self.system.kernel, Tensor)
        self.kernel = self.system.kernel
        self.z0 = nn.Parameter(torch.randn(self.latent_size))

    @jit.export
    def forward(
        self,
        T: Tensor,
        X: Tensor,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        r""".. Signature:: ``[(..., n), (..., n, d)] -> (..., n, d)``.

        **Model Sketch**::

            ⟶ [ODE] ⟶ (ẑᵢ)                (ẑᵢ') ⟶ [ODE] ⟶
                       ↓                   ↑
                      [Ψ]                 [Φ]
                       ↓                   ↑
                      (x̂ᵢ) → [ filter ] → (x̂ᵢ')
                                 ↑
                              (tᵢ, xᵢ)

        Args:
            T: Tensor, shape=(...,LEN) or PackedSequence
                The timestamps of the observations.
            X: Tensor, shape=(..., LEN, DIM) or PackedSequence
                The observed, noisy values at times $t∈T$. Use ``NaN`` to indicate missing values.
            t0: Tensor, shape=(..., 1), optional
                The timestamps of the initial condition. Defaults to ``T[...,0]``.
            z0: Tensor, shape=(..., DIM), optional
                The initial condition. Defaults to ``z0 = self.z0``.

        Returns:
            X̂_post: Tensor, shape=(..., LEN, DIM)
                The estimated true state of the system at the times $t⁺∈T$ (post-update).

        References:
            - https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
        """
        # Pad the input
        if self.padding_size:
            # TODO: write bug report for bogus behaviour
            # dim = -1
            # shape = list(X.shape)
            # shape[dim] = self.padding_size
            # z = torch.full(shape, float("nan"), dtype=X.dtype, device=X.device)
            # X = torch.cat([X, z], dim=dim)
            X = pad(X, float("nan"), self.padding_size)

        # prepend a single zero for the first iteration.
        # T = pad(T, 0.0, 1, prepend=True)
        # DT = torch.diff(T)  # (..., LEN) → (..., LEN)
        t0 = t0 if t0 is not None else T[..., 0].unsqueeze(-1)
        z0 = z0 if z0 is not None else self.z0

        # Move sequence to the front
        DT = torch.diff(T, prepend=t0)  # (..., LEN) → (..., LEN)
        DT = DT.moveaxis(-1, 0)  # (..., LEN) → (LEN, ...)
        X = torch.moveaxis(X, -2, 0)  # (...,LEN,DIM) → (LEN,...,DIM)

        # Initialize buffers
        z_pre_list: list[Tensor] = []
        x_pre_list: list[Tensor] = []
        x_post_list: list[Tensor] = []
        z_post_list: list[Tensor] = []

        z_post = z0

        for dt, x_obs in zip(DT, X):
            # Propagate the latent state forward in time.
            z_pre = self.system(dt, z_post)  # (...,), (..., LAT) -> (..., LAT)

            # Decode the latent state at the observation time.
            x_pre = self.projection(self.decoder(z_pre))  # (..., LAT) -> (..., DIM)

            # Update the state estimate by filtering the observation.
            x_post = self.filter(x_obs, x_pre)  # (..., DIM), (..., DIM) → (..., DIM)

            # Encode the latent state at the observation time.
            z_post = self.encoder(self.embedding(x_post))  # (..., DIM) → (..., LAT)

            # Save all tensors for later.
            z_pre_list.append(z_pre)
            x_pre_list.append(x_pre)
            x_post_list.append(x_post)
            z_post_list.append(z_post)

        self.x_pre = torch.stack(x_pre_list, dim=-2)
        self.x_post = torch.stack(x_post_list, dim=-2)
        self.z_pre = torch.stack(z_pre_list, dim=-2)
        self.z_post = torch.stack(z_post_list, dim=-2)
        self.timedeltas = DT.moveaxis(0, -1)

        yhat = self.x_post[..., : self.output_size]
        return yhat

    @jit.export
    def predict(
        self,
        q: Tensor,
        t: Tensor,
        x: Tensor,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict the future of the system.

        .. Signature:: ``[(..., m), (..., n), (..., n, d)] -> (..., m, d)``.
        """
        t0 = t0 if t0 is not None else t[..., 0].unsqueeze(-1)
        z0 = z0 if z0 is not None else self.z0

        # check compatible shapes
        self._validate_inputs(q, t, x, t0, z0)

        # mix the time and the query points
        time = torch.cat([t, q], dim=-1)
        sorted_index = torch.argsort(time, dim=-1)
        time = time.gather(-1, sorted_index)

        # mix the observations and dummy observations
        x_padding = torch.full(q.shape + x.shape[-1:], fill_value=torch.nan)
        values = torch.cat([x, x_padding], dim=-2)
        values = values.gather(-2, sorted_index.unsqueeze(-1).expand_as(values))

        # create a mask for the query points
        query_mask = torch.cat(
            [
                torch.zeros_like(t, dtype=torch.bool),
                torch.ones_like(q, dtype=torch.bool),
            ],
            dim=-1,
        ).gather(-1, sorted_index)

        # Move sequence to the front
        DT = torch.diff(time, prepend=t0)  # (..., LEN) → (..., LEN)
        DT = DT.moveaxis(-1, 0)  # (..., LEN) → (LEN, ...)
        X = torch.moveaxis(values, -2, 0)  # (..., LEN, DIM) → (LEN, ..., DIM)

        # Initialize buffers
        zhat_pre_list: list[Tensor] = []
        xhat_pre_list: list[Tensor] = []
        xhat_post_list: list[Tensor] = []
        zhat_post_list: list[Tensor] = []
        predictions: list[Tensor] = []

        z_post = z0

        for is_query, dt, x_obs in zip(query_mask, DT, X):
            if is_query:
                z_pre = self.system(dt, z_post)  # (...,), (..., LAT) -> (..., LAT)
                x_pre = self.projection(self.decoder(z_pre))  # (..., LAT) -> (..., DIM)
                z_post = self.encoder(self.embedding(x_pre))  # (..., DIM) → (..., LAT)
                predictions.append(x_pre)
                continue

            # Propagate the latent state forward in time.
            z_pre = self.system(dt, z_post)  # (...,), (..., LAT) -> (..., LAT)

            # Decode the latent state at the observation time.
            x_pre = self.projection(self.decoder(z_pre))  # (..., LAT) -> (..., DIM)

            # Update the state estimate by filtering the observation.
            x_post = self.filter(x_obs, x_pre)  # (..., DIM), (..., DIM) → (..., DIM)

            # Encode the latent state at the observation time.
            z_post = self.encoder(self.embedding(x_post))  # (..., DIM) → (..., LAT)

            # Save all tensors for later.
            zhat_pre_list.append(z_pre)
            xhat_pre_list.append(x_pre)
            xhat_post_list.append(x_post)
            zhat_post_list.append(z_post)

        self.timedeltas = DT.moveaxis(0, -1)
        self.x_pre = torch.stack(xhat_pre_list, dim=-2)
        self.x_post = torch.stack(xhat_post_list, dim=-2)
        self.z_pre = torch.stack(zhat_pre_list, dim=-2)
        self.z_post = torch.stack(zhat_post_list, dim=-2)
        self.predictions = torch.stack(predictions, dim=-2)

        return self.predictions

    @staticmethod
    def _validate_inputs(
        q: Tensor, t: Tensor, x: Tensor, t0: Tensor, z0: Tensor
    ) -> None:
        """Validate the inputs to the model."""
        assert t.shape == x.shape[:-1]
        assert q.shape[:-1] == t.shape[:-1]
        assert t0.shape == t.shape[:-1]
        assert z0.shape[:-1] == x.shape[-1:]
        assert all(t0 < t)
        assert all(t < q)

    def _validate_model(self) -> None:
        """Validate the model."""
        assert self.system is not None
        assert self.encoder is not None
        assert self.projection is not None
        assert self.filter is not None
        assert self.embedding is not None
        assert self.decoder is not None


# from typing import NamedTuple
#
#
# class Context(NamedTuple):
#     observations: tuple[Tensor, Tensor]
#     covariates: tuple[Tensor, Tensor]
#     metadata: Tensor
