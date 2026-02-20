r"""Contains implementations of ODE models."""

__all__ = [
    # Classes
    "LinODEnet",
    "LatentLinODECell",
]

import warnings
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn

from blueprint import Blueprint, ObjectBlueprint, initialize
from linodenet.embeddings import ConcatEmbedding
from linodenet.encoders import ResNet
from linodenet.lib import pad
from linodenet.projections.surjections import ConcatProjection
from linodenet.signatures import signature
from linodenet.system import LinODECell


def _module_blueprint[T](cls: type[T]) -> ObjectBlueprint[T]:
    return {
        "__module_name__": cls.__module__,
        "__class_name__": cls.__qualname__,
        "__args__": [],
        "__kwargs__": {},
    }


_DEFAULT_EMBEDDING_BLUEPRINT = _module_blueprint(ConcatEmbedding)
_DEFAULT_ENCODER_BLUEPRINT = _module_blueprint(ResNet)
_DEFAULT_SYSTEM_BLUEPRINT = _module_blueprint(LinODECell)
_DEFAULT_DECODER_BLUEPRINT = _module_blueprint(ResNet)
_DEFAULT_PROJECTION_BLUEPRINT = _module_blueprint(ConcatProjection)
_DEFAULT_FILTER_BLUEPRINT = _module_blueprint(nn.GRUCell)


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

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "latent_size": self.latent_size,
            "hidden_size": self.hidden_size,
            "system": self.system,
            "embedding": self.embedding,
            "projection": self.projection,
            "filter": self.filter,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "validate_inputs": self.validate_inputs,
        }

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        hidden_size: Optional[int] = None,
        embedding: nn.Module | Blueprint[nn.Module] = _DEFAULT_EMBEDDING_BLUEPRINT,
        encoder: nn.Module | Blueprint[nn.Module] = _DEFAULT_ENCODER_BLUEPRINT,
        system: nn.Module | Blueprint[nn.Module] = _DEFAULT_SYSTEM_BLUEPRINT,
        decoder: nn.Module | Blueprint[nn.Module] = _DEFAULT_DECODER_BLUEPRINT,
        projection: nn.Module | Blueprint[nn.Module] = _DEFAULT_PROJECTION_BLUEPRINT,
        filter: nn.Module | Blueprint[nn.Module] = _DEFAULT_FILTER_BLUEPRINT,  # noqa: A002
        validate_inputs: bool = False,
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

        # Constants
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = input_size
        self.padding_size = self.hidden_size - self.input_size
        self.validate_inputs = validate_inputs

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
        self.embedding = initialize(embedding)
        self.encoder = initialize(encoder)
        self.system = initialize(system)
        self.decoder = initialize(decoder)
        self.projection = initialize(projection)
        self.filter = initialize(filter)
        # TODO: check sizes are compatible

        # Parameters
        kernel = getattr(self.system, "kernel", None)
        if not isinstance(kernel, Tensor):
            raise TypeError("The system must have a kernel attribute!")
        self.kernel = kernel
        self.z0 = nn.Parameter(torch.randn(self.latent_size))

    @jit.export
    @signature("[(..., $n), (..., $n, d)] -> (..., $n, d)")
    def forward(
        self,
        T: Tensor,
        X: Tensor,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Forward pass of the LinODEnet model.

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

        for dt, x_obs in zip(DT, X, strict=True):
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
    @signature("[(..., $m), (..., $n), (..., $n, d)] -> (..., $m, d)")
    def predict(
        self,
        q: Tensor,
        t: Tensor,
        x: Tensor,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Predict the future of the system."""
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

        for is_query, dt, x_obs in zip(query_mask, DT, X, strict=True):
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

    @jit.export
    def _validate_inputs(
        self, q: Tensor, t: Tensor, x: Tensor, t0: Tensor, z0: Tensor
    ) -> None:
        r"""Validate the inputs to the model."""
        if t.shape != x.shape[:-1]:
            raise ValueError(f"Expected shape {x.shape[:-1]}, got {t.shape}")
        if q.shape[:-1] != t.shape[:-1]:
            raise ValueError(f"Expected shape {t.shape[:-1]}, got {q.shape[:-1]}")
        if t0.shape != t.shape[:-1]:
            raise ValueError(f"Expected shape {t.shape[:-1]}, got {t0.shape}")
        if z0.shape[:-1] != x.shape[-1:]:
            raise ValueError(f"Expected shape {x.shape[-1:]}, got {z0.shape[:-1]}")
        if not all(t0 < t):
            raise ValueError(f"Expected {t0} < {t}")
        if not all(t < q):
            raise ValueError(f"Expected {t} < {q}")

    @jit.export
    def _validate_model(self) -> None:
        r"""Validate the model."""
        for key in [
            "embedding",
            "encoder",
            "system",
            "decoder",
            "projection",
            "filter",
        ]:
            if getattr(self, key, None) is None:
                raise ValueError(f"{key} is not set!")


# class Context(NamedTuple):
#     observations: tuple[Tensor, Tensor]
#     covariates: tuple[Tensor, Tensor]
#     metadata: Tensor
class LatentLinODECell(nn.Module):
    r"""Latent Linear ODE Cell."""

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

    @property
    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "latent_size": self.latent_size,
            "hidden_size": self.hidden_size,
            "System": self.system,
            "Embedding": self.embedding,
            "Projection": self.projection,
            "Filter": self.filter,
            "Encoder": self.encoder,
            "Decoder": self.decoder,
            "validate_inputs": self.validate_inputs,
        }

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        *,
        hidden_size: Optional[int] = None,
        embedding: nn.Module | Blueprint[nn.Module] = _DEFAULT_EMBEDDING_BLUEPRINT,
        encoder: nn.Module | Blueprint[nn.Module] = _DEFAULT_ENCODER_BLUEPRINT,
        system: nn.Module | Blueprint[nn.Module] = _DEFAULT_SYSTEM_BLUEPRINT,
        decoder: nn.Module | Blueprint[nn.Module] = _DEFAULT_DECODER_BLUEPRINT,
        projection: nn.Module | Blueprint[nn.Module] = _DEFAULT_PROJECTION_BLUEPRINT,
        filter: nn.Module | Blueprint[nn.Module] = _DEFAULT_FILTER_BLUEPRINT,  # noqa: A002
        validate_inputs: bool = False,
    ) -> None:
        super().__init__()
        self.validate_inputs = validate_inputs

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
        self.embedding = initialize(embedding)
        self.encoder = initialize(encoder)
        self.system = initialize(system)
        self.decoder = initialize(decoder)
        self.projection = initialize(projection)
        self.filter = initialize(filter)

        # Parameters
        self.kernel = self.system.kernel
        self.z0 = nn.Parameter(torch.randn(self.latent_size))

    @signature("[(..., d), (..., l), (...,)] -> (..., l)")
    def forward(self, x_obs: Tensor, z: Tensor, dt: Tensor) -> Tensor:
        r"""Propagate the latent state forward in time.

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
