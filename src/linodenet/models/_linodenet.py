r"""Contains implementations of ODE models."""

__all__ = [
    # Classes
    "LinODE",
    "LinODEnet",
]

import logging
from typing import Any, Final

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import FunctionalInitialization
from linodenet.models.embeddings import ConcatEmbedding, ConcatProjection
from linodenet.models.encoders import iResNet
from linodenet.models.filters import Filter, RecurrentCellFilter
from linodenet.models.system import LinODECell
from linodenet.projections import Projection
from linodenet.util import deep_dict_update, initialize_from_config

# TODO: Use Unicode variable names once https://github.com/pytorch/pytorch/issues/65653 is fixed.

__logger__ = logging.getLogger(__name__)


class LinODE(nn.Module):
    r"""Linear ODE module, to be used analogously to `scipy.integrate.odeint`.

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    kernel: Tensor
        The system matrix
    kernel_initialization: Callable[None, Tensor]
        Parameter-less function that draws a initial system matrix
    """

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
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

    # Parameters
    kernel: Tensor
    r"""PARAM: The system matrix of the linear ODE component."""

    # Buffers
    xhat: Tensor
    r"""BUFFER: The forward prediction."""

    # Functions
    kernel_initialization: FunctionalInitialization
    r"""FUNC: Parameter-less function that draws a initial system matrix."""
    kernel_projection: Projection
    r"""FUNC: Regularization function for the kernel."""

    def __init__(
        self,
        input_size: int,
        **HP: Any,
    ):
        super().__init__()
        self.CFG = HP = deep_dict_update(self.HP, HP)

        HP["cell"]["input_size"] = input_size

        self.input_size = input_size
        self.output_size = input_size
        self.cell: nn.Module = initialize_from_config(HP["cell"])

        # Buffers
        self.register_buffer("xhat", torch.tensor(()), persistent=False)
        assert isinstance(self.cell.kernel, Tensor)
        self.register_buffer("kernel", self.cell.kernel, persistent=False)

    @jit.export
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., N), (..., d)] -> (..., N, d)``.

        Parameters
        ----------
        T: Tensor, shape=(...,LEN)
        x0: Tensor, shape=(...,DIM)

        Returns
        -------
        Xhat: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times $t∈T$.
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


class LinODEnet(nn.Module):
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

    name: Final[str] = __name__
    r"""str: The name of the model."""

    HP = {
        "__name__": __qualname__,  # type: ignore[name-defined]
        "__doc__": __doc__,
        "__module__": __module__,  # type: ignore[name-defined]
        "input_size": int,
        "hidden_size": int,
        "output_size": int,
        "System": LinODECell.HP,
        "Embedding": ConcatEmbedding.HP,
        "Projection": ConcatProjection.HP,
        "Filter": RecurrentCellFilter.HP | {"autoregressive": True},
        "Encoder": iResNet.HP,
        "Decoder": iResNet.HP,
    }
    r"""Dictionary of Hyperparameters."""

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the linear ODE."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    # Buffers
    zero: Tensor
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

    def __init__(self, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()

        LOGGER = __logger__.getChild(self.__class__.__name__)

        self.CFG = HP = deep_dict_update(self.HP, HP)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        HP["Encoder"]["input_size"] = hidden_size
        HP["Decoder"]["input_size"] = hidden_size
        HP["System"]["input_size"] = hidden_size
        HP["Filter"]["hidden_size"] = input_size
        HP["Filter"]["input_size"] = input_size
        HP["Embedding"]["input_size"] = input_size
        HP["Embedding"]["hidden_size"] = hidden_size
        HP["Projection"]["input_size"] = input_size
        HP["Projection"]["hidden_size"] = hidden_size

        LOGGER.debug("%s Initializing Embedding %s", self.name, HP["Embedding"])
        self.embedding: nn.Module = initialize_from_config(HP["Embedding"])
        LOGGER.debug("%s Initializing Embedding %s", self.name, HP["Embedding"])
        self.projection: nn.Module = initialize_from_config(HP["Projection"])
        LOGGER.debug("%s Initializing Encoder %s", self.name, HP["Encoder"])
        self.encoder: nn.Module = initialize_from_config(HP["Encoder"])
        LOGGER.debug("%s Initializing System %s", self.name, HP["Encoder"])
        self.system: nn.Module = initialize_from_config(HP["System"])
        LOGGER.debug("%s Initializing Decoder %s", self.name, HP["Encoder"])
        self.decoder: nn.Module = initialize_from_config(HP["Decoder"])
        LOGGER.debug("%s Initializing Filter %s", self.name, HP["Encoder"])
        self.filter: Filter = initialize_from_config(HP["Filter"])

        assert isinstance(self.system.kernel, Tensor)
        self.kernel = self.system.kernel
        self.z0 = nn.Parameter(torch.randn(self.hidden_size))

        # Buffers
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        self.register_buffer("timedeltas", torch.tensor(()), persistent=False)
        self.register_buffer("xhat_pre", torch.tensor(()), persistent=False)
        self.register_buffer("xhat_post", torch.tensor(()), persistent=False)
        self.register_buffer("zhat_pre", torch.tensor(()), persistent=False)
        self.register_buffer("zhat_post", torch.tensor(()), persistent=False)

    @jit.export
    def forward(self, T: Tensor, X: Tensor) -> Tensor:
        r""".. Signature:: ``[(..., n), (...,n,d) -> (..., N, d)``.

        **Model Sketch**::

            ⟶ [ODE] ⟶ (ẑᵢ)                (ẑᵢ') ⟶ [ODE] ⟶
                       ↓                   ↑
                      [Ψ]                 [Φ]
                       ↓                   ↑
                      (x̂ᵢ) → [ filter ] → (x̂ᵢ')
                                 ↑
                              (tᵢ, xᵢ)

        Parameters
        ----------
        T: Tensor, shape=(...,LEN) or PackedSequence
            The timestamps of the observations.
        X: Tensor, shape=(...,LEN,DIM) or PackedSequence
            The observed, noisy values at times $t∈T$. Use ``NaN`` to indicate missing values.

        Returns
        -------
        X̂_pre: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times $t⁻∈T$ (pre-update).
        X̂_post: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times $t⁺∈T$ (post-update).

        References
        ----------
        - https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
        """
        BATCH_SIZE = X.shape[:-2]
        # prepend a single zero for the first iteration.
        pad_dim = list(BATCH_SIZE) + [1]
        pad = torch.zeros(pad_dim, device=T.device, dtype=T.dtype)
        DT = torch.diff(T, prepend=pad, dim=-1)  # (..., LEN) → (..., LEN)
        DT = DT.moveaxis(-1, 0)  # (..., LEN) → (LEN, ...)
        X = torch.moveaxis(X, -2, 0)  # (...,LEN,DIM) → (LEN,...,DIM)

        Zhat_pre: list[Tensor] = []
        Xhat_pre: list[Tensor] = []
        Xhat_post: list[Tensor] = []
        Zhat_post: list[Tensor] = []

        ẑ_post = self.z0

        for dt, x_obs in zip(DT, X):
            # Propagate the latent state forward in time.
            ẑ_pre = self.system(dt, ẑ_post)  # (...,), (...,LAT) -> (...,LAT)

            # Decode the latent state at the observation time.
            x̂_pre = self.projection(self.decoder(ẑ_pre))  # (...,LAT) -> (...,DIM)

            # Update the state estimate by filtering the observation.
            x̂_post = self.filter(x_obs, x̂_pre)  # (...,DIM), (..., DIM) → (...,DIM)

            # Encode the latent state at the observation time.
            ẑ_post = self.encoder(self.embedding(x̂_post))  # (...,DIM) → (...,LAT)

            # Save all tensors for later.
            Zhat_pre.append(ẑ_pre)
            Xhat_pre.append(x̂_pre)
            Xhat_post.append(x̂_post)
            Zhat_post.append(ẑ_post)

        self.xhat_pre = torch.stack(Xhat_pre, dim=-2)
        self.xhat_post = torch.stack(Xhat_post, dim=-2)
        self.zhat_pre = torch.stack(Zhat_pre, dim=-2)
        self.zhat_post = torch.stack(Zhat_post, dim=-2)
        self.timedeltas = DT.moveaxis(0, -1)

        return self.xhat_post
