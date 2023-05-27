r"""Latent State Space Model."""

__all__ = [
    # Classes
    "LatentStateSpaceModel",
]

import logging
import warnings
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from typing_extensions import Self

from linodenet.models.embeddings import ConcatEmbedding, ConcatProjection
from linodenet.models.encoders import ResNet
from linodenet.models.filters import Filter, RecurrentCellFilter
from linodenet.models.system import LinODECell
from linodenet.utils import deep_dict_update, initialize_from_config, pad

__logger__ = logging.getLogger(__name__)


class LatentStateSpaceModel(nn.Module):
    r"""Linear ODE Network is a FESD model.

    +---------------------------------------------------+--------------------------------------+
    | Component                                         | Formula                              |
    +===================================================+======================================+
    | Filter  `F` (default: :class:`~torch.nn.GRUCell`) | `\hat x_i' = F(\hat x_i, x_i)`       |
    +---------------------------------------------------+--------------------------------------+
    | Encoder `ϕ` (default: :class:`~iResNet`)          | `\hat z_i' = ϕ(\hat x_i')`           |
    +---------------------------------------------------+--------------------------------------+
    | System  `S` (default: :class:`~LinODECell`)       | `\hat z_{i+1} = S(\hat z_i', Δ t_i)` |
    +---------------------------------------------------+--------------------------------------+
    | Decoder `π` (default: :class:`~iResNet`)          | `\hat x_{i+1}  =  π(\hat z_{i+1})`   |
    +---------------------------------------------------+--------------------------------------+

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    hidden_size: int
        The dimensionality of the latent space.
    output_size: int
        The dimensionality of the output space.
    ZERO: Tensor
        BUFFER: A constant tensor of value float(0.0)
    xhat_pre: Tensor
        BUFFER: Stores pre-jump values.
    xhat_post: Tensor
        BUFFER: Stores post-jump values.
    zhat_pre: Tensor
        BUFFER: Stores pre-jump latent values.
    zhat_post: Tensor
        BUFFER: Stores post-jump latent values.
    kernel: Tensor
        PARAM: The system matrix of the linear ODE component.
    encoder: nn.Module
        MODULE: Responsible for embedding $x̂→ẑ$.
    embedding: nn.Module
        MODULE: Responsible for embedding $x̂→ẑ$.
    system: nn.Module
        MODULE: Responsible for propagating $ẑ_t→ẑ_{t+{∆t}}$.
    decoder: nn.Module
        MODULE: Responsible for projecting $ẑ→x̂$.
    projection: nn.Module
        MODULE: Responsible for projecting $ẑ→x̂$.
    filter: nn.Module
        MODULE: Responsible for updating $(x̂, x_{obs}) →x̂'$.
    """
    LOGGER = __logger__.getChild(f"{__package__}/{__qualname__}")  # type: ignore[name-defined]

    name: Final[str] = __name__
    r"""str: The name of the model."""

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

    # Buffers
    ZERO: Tensor
    r"""BUFFER: A tensor of value float(0.0)"""
    NAN: Tensor
    r"""BUFFER: A tensor of value float(0.0)"""
    xhat_pre: Tensor
    r"""BUFFER: Stores pre-jump values."""
    xhat_post: Tensor
    r"""BUFFER: Stores post-jump values."""
    zhat_pre: Tensor
    r"""BUFFER: Stores pre-jump latent values."""
    zhat_post: Tensor
    r"""BUFFER: Stores post-jump latent values."""
    timedeltas: Tensor
    r"""BUFFER: Stores the timedelta values."""

    # Parameters:
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""
    z0: Tensor
    r"""PARAM: The initial latent state."""

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

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        r"""Constructs a new model from a configuration dictionary."""
        config = deep_dict_update(cls.HP, cfg)
        input_size = config["input_size"]
        latent_size = config["latent_size"]
        hidden_size = config.get("hidden_size", input_size)
        # padding_size = hidden_size - input_size
        # output_size = config.get("output_size", input_size)

        if hidden_size < input_size:
            warnings.warn(
                "hidden_size < input_size. Falling back to using no hidden units.",
                RuntimeWarning,
                stacklevel=2,
            )

            hidden_size = input_size
        assert hidden_size >= input_size

        config["Encoder"] |= {"input_size": latent_size}
        config["Decoder"] |= {"input_size": latent_size}
        config["System"] |= {"input_size": latent_size}
        config["Filter"] |= {"input_size": hidden_size}
        config["Filter"] |= {"hidden_size": hidden_size}

        cls.LOGGER.debug("Initializing Encoder %s", config["Encoder"])
        encoder: nn.Module = initialize_from_config(config["Encoder"])
        cls.LOGGER.debug("Initializing System %s", config["Encoder"])
        system: nn.Module = initialize_from_config(config["System"])
        cls.LOGGER.debug("Initializing Decoder %s", config["Encoder"])
        decoder: nn.Module = initialize_from_config(config["Decoder"])
        cls.LOGGER.debug("Initializing Filter %s", config["Encoder"])
        filter: nn.Module = initialize_from_config(config["Filter"])

        return cls(
            encoder=encoder,
            system=system,
            decoder=decoder,
            filter=filter,
            padding_size=hidden_size - input_size,
        )

    def __init__(
        self,
        *,
        encoder: nn.Module,
        system: nn.Module,
        decoder: nn.Module,
        filter: nn.Module,
        padding_size: int = 0,
        **cfg: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.system = system
        self.decoder = decoder
        self.filter: Filter = filter

        self.input_size = self.filter.input_size  # type: ignore[assignment]
        self.output_size = self.filter.output_size  # type: ignore[assignment]
        self.latent_size = self.system.input_size  # type: ignore[assignment]
        self.hidden_size = -1
        self.padding_size = padding_size

        # self.input_size =  filter.input_size  # type: ignore[assignment]
        # self.output_size = filter.output_size  # type: ignore[assignment]
        # self.latent_size = system.input_size  # type: ignore[assignment]
        # self.hidden_size = filter.hidden_size  # type: ignore[assignment]
        # self.padding_size = padding_size

        assert isinstance(self.system.kernel, Tensor)
        self.kernel = self.system.kernel
        self.z0 = nn.Parameter(torch.randn(self.latent_size))

        # Buffers
        self.register_buffer("ZERO", torch.tensor(0.0), persistent=False)
        self.register_buffer("NAN", torch.tensor(float("nan")), persistent=False)
        self.register_buffer("timedeltas", torch.tensor(()), persistent=False)
        self.register_buffer("xhat_pre", torch.tensor(()), persistent=False)
        self.register_buffer("xhat_post", torch.tensor(()), persistent=False)
        self.register_buffer("zhat_pre", torch.tensor(()), persistent=False)
        self.register_buffer("zhat_post", torch.tensor(()), persistent=False)

    @jit.export
    def forward(
        self,
        T: Tensor,
        X: Tensor,
        t0: Optional[Tensor] = None,
        z0: Optional[Tensor] = None,
    ) -> Tensor:
        r""".. Signature:: ``[(..., n), (...,n,d) -> (..., N, d)``.

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

        # # prepend a single zero for the first iteration.
        # # T = pad(T, 0.0, 1, prepend=True)
        # # DT = torch.diff(T)  # (..., LEN) → (..., LEN)
        # DT = torch.diff(T, prepend=T[..., 0].unsqueeze(-1))  # (..., LEN) → (..., LEN)
        #
        # # Move sequence to the front
        # DT = DT.moveaxis(-1, 0)  # (..., LEN) → (LEN, ...)
        # X = torch.moveaxis(X, -2, 0)  # (...,LEN,DIM) → (LEN,...,DIM)

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
        Zhat_pre: list[Tensor] = []
        Xhat_pre: list[Tensor] = []
        Xhat_post: list[Tensor] = []
        Zhat_post: list[Tensor] = []

        z_post = z0

        for dt, x_obs in zip(DT, X):
            # Propagate the latent state forward in time.
            z_pre = self.system(dt, z_post)  # (...,), (..., LAT) -> (..., LAT)

            # Decode the latent state at the observation time.
            x_pre = self.decoder(z_pre)  # (..., LAT) -> (..., DIM)

            # Update the state estimate by filtering the observation.
            x_post = self.filter(x_obs, x_pre)  # (..., DIM), (..., DIM) → (..., DIM)

            # Encode the latent state at the observation time.
            z_post = self.encoder(x_post)  # (..., DIM) → (..., LAT)

            # Save all tensors for later.
            Zhat_pre.append(z_pre)
            Xhat_pre.append(x_pre)
            Xhat_post.append(x_post)
            Zhat_post.append(z_post)

        self.xhat_pre = torch.stack(Xhat_pre, dim=-2)
        self.xhat_post = torch.stack(Xhat_post, dim=-2)
        self.zhat_pre = torch.stack(Zhat_pre, dim=-2)
        self.zhat_post = torch.stack(Zhat_post, dim=-2)
        self.timedeltas = DT.moveaxis(0, -1)

        yhat = self.xhat_pre[..., : self.output_size]
        return yhat
