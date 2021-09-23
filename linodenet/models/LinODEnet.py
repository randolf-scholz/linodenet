r"""Contains implementations of ODE models."""

import logging
from typing import Any, Final, Optional, Union

import torch
from torch import Tensor, jit, nn

from linodenet.initializations import INITIALIZATIONS, Initialization, gaussian
from linodenet.models.iResNet import iResNet
from linodenet.projections import PROJECTIONS, Projection
from linodenet.util import deep_dict_update

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "ConcatEmbedding",
    "ConcatProjection",
    "LinODE",
    "LinODECell",
    "LinODEnet",
]


class LinODECell(jit.ScriptModule):
    r"""Linear System module, solves `ẋ = Ax`, i.e. `x̂ = e^{A\Delta t}x`.

    Parameters
    ----------
    input_size: int
    kernel_initialization: Union[Tensor, Callable[int, Tensor]]

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    kernel: Tensor
        The system matrix
    kernel_initialization: Callable[[], Tensor]
        Parameter-less function that draws a initial system matrix
    kernel_projection: Callable[[Tensor], Tensor]
        Regularization function for the kernel
    """

    input_size: Final[int]
    output_size: Final[int]

    kernel: Tensor
    # kernel_initialization: Callable[[], Tensor]
    kernel_projection: Projection

    def __init__(
        self,
        input_size: int,
        kernel_initialization: Optional[Union[str, Tensor, Initialization]] = None,
        kernel_projection: Optional[Union[str, Projection]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size

        def kernel_initialization_dispatch():
            if kernel_initialization is None:
                return lambda: gaussian(input_size)
            if kernel_initialization in INITIALIZATIONS:
                _init = INITIALIZATIONS[kernel_initialization]
                return lambda: _init(input_size)
            if callable(kernel_initialization):
                assert Tensor(kernel_initialization(input_size)).shape == (
                    input_size,
                    input_size,
                )
                return lambda: Tensor(kernel_initialization(input_size))
            if isinstance(kernel_initialization, Tensor):
                assert kernel_initialization.shape == (input_size, input_size)
                return lambda: kernel_initialization
            assert Tensor(kernel_initialization).shape == (input_size, input_size)
            return lambda: Tensor(kernel_initialization)

        # this looks funny, but it needs to be written that way to be compatible with torchscript
        def kernel_regularization_dispatch():
            if kernel_projection is None:
                _kernel_regularization = PROJECTIONS["identity"]
            elif kernel_projection in PROJECTIONS:
                _kernel_regularization = PROJECTIONS[kernel_projection]
            elif callable(kernel_projection):
                _kernel_regularization = kernel_projection
            else:
                raise NotImplementedError(f"{kernel_projection=} unknown")
            return _kernel_regularization

        self._kernel_initialization = kernel_initialization_dispatch()
        self._kernel_regularization = kernel_regularization_dispatch()
        self.kernel = nn.Parameter(self._kernel_initialization())

    def kernel_initialization(self) -> Tensor:
        r"""Draw an initial kernel matrix (random or static)."""
        return self._kernel_initialization()

    @jit.script_method
    def kernel_regularization(self, w: Tensor) -> Tensor:
        r"""Regularize the Kernel, e.g. by projecting onto skew-symmetric matrices."""
        return self._kernel_regularization(w)

    @jit.script_method
    def forward(self, Δt: Tensor, x0: Tensor) -> Tensor:
        # TODO: optimize if clauses away by changing definition in constructor.
        r"""Signature: `[...,]×[...,d] ⟶ [...,d]`.

        Parameters
        ----------
        Δt: Tensor, shape=(...,)
            The time difference `t_1 - t_0` between `x_0` and `x̂`.
        x0:  Tensor, shape=(...,DIM)
            Time observed value at `t_0`

        Returns
        -------
        xhat:  Tensor, shape=(...,DIM)
            The predicted value at `t_1`
        """
        A = self.kernel_regularization(self.kernel)
        AΔt = torch.einsum("kl, ... -> ...kl", A, Δt)
        expAΔt = torch.matrix_exp(AΔt)
        xhat = torch.einsum("...kl, ...l -> ...k", expAΔt, x0)
        return xhat


class LinODE(jit.ScriptModule):
    r"""Linear ODE module, to be used analogously to :func:`scipy.integrate.odeint`.

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

    input_size: Final[int]
    output_size: Final[int]

    kernel: Tensor
    # kernel_initialization: Callable[[], Tensor]
    kernel_projection: Projection

    def __init__(
        self,
        input_size: int,
        kernel_initialization: Optional[Union[str, Tensor, Initialization]] = None,
        kernel_projection: Optional[Union[str, Projection]] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.cell = LinODECell(input_size, kernel_initialization, kernel_projection)
        self.kernel = self.cell.kernel

    @jit.script_method
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r"""Signature: `[...,N]×[...,d] ⟶ [...,N,d]`.

        Parameters
        ----------
        T: Tensor, shape=(...,LEN)
        x0: Tensor, shape=(...,DIM)

        Returns
        -------
        Xhat: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times `t∈T`
        """
        ΔT = torch.moveaxis(torch.diff(T), -1, 0)
        X = torch.jit.annotate(list[Tensor], [])
        X += [x0]

        # iterate over LEN, this works even when no BATCH dim present.
        for Δt in ΔT:
            X += [self.cell(Δt, X[-1])]

        # shape: [LEN, ..., DIM]
        Xhat = torch.stack(X, dim=0)

        return torch.moveaxis(Xhat, 0, -2)


class LinODEnet(jit.ScriptModule):
    r"""Linear ODE Network is a FESD model.

    +---------------------------------------------------+--------------------------------------+
    | Component                                         | Formula                              |
    +===================================================+======================================+
    | Filter  `F` (default: :class:`~torch.nn.GRUCell`) | `\hat x_i' = F(\hat x_i, x_i)`       |
    +---------------------------------------------------+--------------------------------------+
    | Encoder `ϕ` (default: :class:`~.iResNet`)         | `\hat z_i' = ϕ(\hat x_i')`           |
    +---------------------------------------------------+--------------------------------------+
    | System  `S` (default: :class:`~.LinODECell`)      | `\hat z_{i+1} = S(\hat z_i', Δ t_i)` |
    +---------------------------------------------------+--------------------------------------+
    | Decoder `π` (default: :class:`~.iResNet`)         | `\hat x_{i+1}  =  π(\hat z_{i+1})`   |
    +---------------------------------------------------+--------------------------------------+

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    hidden_size: int
        The dimensionality of the latent space.
    output_size: int
        The dimensionality of the output space.
    kernel: Tensor
        The system matrix
    kernel_initialization: Callable[None, Tensor]
        Parameter-less function that draws a initial system matrix
    padding: Tensor
        The learned padding parameters
    """

    HP: dict[str, Any] = {
        "input_size": int,
        "hidden_size": int,
        "output_size": int,
        "embedding_type": "linear",
        "concat_mask": True,
        "System": LinODECell,
        "System_cfg": {"input_size": int, "kernel_initialization": "skew-symmetric"},
        "Filter": nn.GRUCell,
        "Filter_cfg": {"input_size": int, "hidden_size": int, "bias": True},
        "Encoder": iResNet,
        "Encoder_cfg": {"input_size": int, "nblocks": 5},
        "Decoder": iResNet,
        "Decoder_cfg": {"input_size": int, "nblocks": 5},
    }

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    concat_mask: Final[bool]
    kernel: Tensor
    padding: Tensor

    ZERO: Tensor

    def __init__(self, input_size: int, hidden_size: int, **HP: Any):
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size = input_size
        self.hidden_size = input_size
        self.output_size = input_size
        self.concat_mask = HP["concat_mask"]

        HP["Encoder_cfg"]["input_size"] = hidden_size
        HP["Decoder_cfg"]["input_size"] = hidden_size
        HP["System_cfg"]["input_size"] = hidden_size
        HP["Filter_cfg"]["hidden_size"] = input_size
        HP["Filter_cfg"]["input_size"] = (1 + self.concat_mask) * input_size

        self.encoder = HP["Encoder"](**HP["Encoder_cfg"])
        self.decoder = HP["Decoder"](**HP["Decoder_cfg"])
        self.filter = HP["Filter"](**HP["Filter_cfg"])
        self.system = HP["System"](**HP["System_cfg"])

        embedding_type = HP["embedding_type"]

        if embedding_type == "linear":
            self.embedding = nn.Linear(input_size, hidden_size)
            self.projection = nn.Linear(hidden_size, input_size)
        elif embedding_type == "concat":
            assert input_size <= hidden_size, f"{embedding_type=} not possible"
            self.embedding = ConcatEmbedding(input_size, hidden_size)
            self.projection = ConcatProjection(input_size, hidden_size)
        else:
            raise NotImplementedError(
                f"{embedding_type=}" + "not in {'linear', 'concat'}"
            )

        self.kernel = self.system.kernel
        self.register_buffer("ZERO", torch.tensor(0.0))
        # self.ZERO = torch.tensor(0.0)

    @jit.script_method
    def forward(self, T: Tensor, X: Tensor) -> tuple[Tensor, Tensor]:
        r"""Signature: `[...,N]×[...,N,d] ⟶ [...,N,d]`.

        **Model Sketch:**

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
            The observed, noisy values at times `t∈T`. Use ``NaN`` to indicate missing values.

        Returns
        -------
        X̂_pre: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times `t⁻∈T` (pre-update).
        X̂_post: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times `t⁺∈T` (post-update).

        References
        ----------
        - https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
        """
        ΔT = torch.moveaxis(torch.diff(T), -1, 0)
        X = torch.moveaxis(X, -2, 0)

        X̂_pre = torch.jit.annotate(list[Tensor], [])
        X̂_post = torch.jit.annotate(list[Tensor], [])

        # Initialization
        # TODO: do something smarter than zero initialization!
        X0 = torch.where(torch.isnan(X[0]), self.ZERO, X[0])

        X̂_pre += [X0]  # here one would expect zero or some other initialization
        X̂_post += [X0]

        # IDEA: The problem is the initial state of RNNCell is not defined and typically put equal
        # to zero. Staying with the idea that the Cell acts as a filter, that is updates the state
        # estimation given an observation, we could "trust" the original observation in the sense
        # that we solve the fixed point equation h0 = g(x0, h0) and put the solution as the initial
        # state.
        # issue: if x0 is really sparse this is useless.
        # better idea: we probably should go back and forth.
        # other idea: use a set-based model and put h = g(T,X), including the whole TS.
        # This set model can use triplet notation.
        # bias weighting towards close time points

        for Δt, x in zip(ΔT, X):
            # Encode
            ẑ_post = self.encoder(self.embedding(X̂_post[-1]))

            # Propagate
            ẑ_pre = self.system(Δt, ẑ_post)

            # Decode
            x̂_pre = self.projection(self.decoder(ẑ_pre))

            # Compute update
            mask = torch.isnan(x)
            x̃ = torch.where(mask, x̂_pre, x)

            if self.concat_mask:
                x̃ = torch.cat([x̃, mask], dim=-1)

            # Flatten for GRU-Cell
            x̂_pre = x̂_pre.view(-1, x̂_pre.shape[-1])
            x̃ = x̃.view(-1, x̃.shape[-1])

            # Apply filter
            x̂_post = self.filter(x̃, x̂_pre)

            # xhat = self.control(xhat, u)
            # u: possible controls:
            #  1. set to value
            #  2. add to value
            # do these via indicator variable
            # u = (time, value, mode-indicator, col-indicator)
            # => apply control to specific column.

            X̂_pre += [x̂_pre.view(x.shape)]
            X̂_post += [x̂_post.view(x.shape)]

        X̂_pre = torch.stack(X̂_pre, dim=-2)
        X̂_post = torch.stack(X̂_post, dim=-2)
        return X̂_pre, X̂_post


class ConcatEmbedding(jit.ScriptModule):
    r"""Maps `x ⟼ [x,w]`.

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    pad_size:    int
    """

    input_size: Final[int]
    hidden_size: Final[int]
    pad_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        assert input_size <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size
        self.padding = nn.Parameter(torch.randn(self.pad_size))

    @jit.script_method
    def forward(self, X: Tensor) -> Tensor:
        r"""Signature: `[..., d] ⟶ [..., d+e]`.

        Parameters
        ----------
        X: Tensor, shape=(...,DIM)

        Returns
        -------
        Tensor, shape=(...,LAT)
        """
        shape = list(X.shape[:-1]) + [self.pad_size]
        return torch.cat([X, self.padding.expand(shape)], dim=-1)


class ConcatProjection(jit.ScriptModule):
    r"""Maps `z = [x,w] ⟼ x`.

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        assert input_size <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size

    @jit.script_method
    def forward(self, Z: Tensor) -> Tensor:
        r"""Signature: `[..., d+e] ⟶ [..., d]`.

        Parameters
        ----------
        Z: Tensor, shape=(...,LEN,LAT)

        Returns
        -------
        Tensor, shape=(...,LEN,DIM)
        """
        return Z[..., : self.input_size]

    # TODO: Add variant with filter in latent space
    # TODO: Add Controls
