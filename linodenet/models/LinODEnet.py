r"""
LinODEnet
=========

Contains implementations of

- class:`~.LinODECell`
- class:`~.LinODE`
- class:`~.LinODEnet`
"""

from typing import Union, Callable, Final, Any

import torch
from numpy.typing import ArrayLike
from torch import nn, Tensor, jit

from tsdm.util import deep_dict_update

from linodenet.init import gaussian
from .iResNet import iResNet

Initialization = Union[ArrayLike, Tensor, Callable[[int], Tensor]]


class LinODECell(jit.ScriptModule):
    r"""Linear System module, solves $\dot X = Ax$

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
    kernel_regularization: Callable[[Tensor], Tensor]
        Regularization function for the kernel
    """

    input_size: Final[int]
    output_size: Final[int]

    kernel: Tensor
    # kernel_initialization: Callable[[], Tensor]
    # kernel_regularization: Callable[[Tensor], Tensor]

    def __init__(self, input_size: int,
                 kernel_initialization: Initialization = None,
                 kernel_regularization: Callable[[Tensor], Tensor] = None):
        r"""
        Parameters
        ----------
        input_size: int
        kernel_initialization: Union[Tensor, Callable[int, Tensor]]
        """

        super().__init__()
        self.input_size  = input_size
        self.output_size = input_size

        if kernel_initialization is None:
            def _kernel_initialization():
                return gaussian(input_size)
        elif callable(kernel_initialization):
            def _kernel_initialization():
                return Tensor(kernel_initialization(input_size))
        elif isinstance(kernel_initialization, Tensor):
            def _kernel_initialization():
                return kernel_initialization
        else:
            def _kernel_initialization():
                return Tensor(kernel_initialization)

        def __kernel_initialization() -> Tensor:
            return _kernel_initialization()

        self.kernel_initialization = __kernel_initialization

        self.kernel = nn.Parameter(self.kernel_initialization())

        if kernel_regularization is None:
            def _kernel_regularization(w: Tensor) -> Tensor:
                return w
        elif kernel_regularization == "symmetric":
            def _kernel_regularization(w: Tensor) -> Tensor:
                return (w+w.T)/2
        elif kernel_regularization == "skew-symmetric":
            def _kernel_regularization(w: Tensor) -> Tensor:
                return (w-w.T)/2
        else:
            raise NotImplementedError(F"{kernel_regularization=} unknown")
        self._kernel_regularization = _kernel_regularization

    @jit.script_method
    def kernel_regularization(self, w: Tensor) -> Tensor:
        return self._kernel_regularization(w)

    @jit.script_method
    def forward(self, Δt: Tensor, x0: Tensor) -> Tensor:
        r"""Forward using the matrix exponential $\hat X = e^{A\Delta t}X$

        TODO: optimize if clauses away by changing definition in constructor.

        Parameters
        ----------
        Δt: Tensor, shape=(...,)
            The time difference $t_1 - t_0$ between $X$ and $\hat X$.
        x0:  Tensor, shape=(...,DIM)
            Time observed value at $t_0$

        Returns
        -------
        xhat:  Tensor, shape=(...,DIM)
            The predicted value at $t_1$
        """
        A = self.kernel_regularization(self.kernel)
        AΔt = torch.einsum('kl, ... -> ...kl', A, Δt)
        expAΔt = torch.matrix_exp(AΔt)
        xhat = torch.einsum('...kl, ...l -> ...k', expAΔt, x0)
        return xhat


class LinODE(jit.ScriptModule):
    r"""Linear ODE module, to be used analogously to :func:`~scipy.integrate.odeint`

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
    kernel_initialization: Callable[[], Tensor]
    kernel_regularization: Callable[[Tensor], Tensor]

    def __init__(self, input_size: int,
                 kernel_initialization: Initialization = None,
                 kernel_regularization: Callable[[Tensor], Tensor] = None):
        r"""

        Parameters
        ----------
        input_size: int
        kernel_initialization: Callable[int, Tensor]] or Tensor
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.cell = LinODECell(input_size, kernel_initialization, kernel_regularization)
        self.kernel = self.cell.kernel

    @jit.script_method
    def forward(self, T: Tensor, x0: Tensor) -> Tensor:
        r"""
        Propagate x0

        Parameters
        ----------
        T: Tensor, shape=(...,LEN)
        x0: Tensor, shape=(...,DIM)

        Returns
        -------
        Xhat: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times $t\in T$
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
    r"""Linear ODE Network, consisting of 4 components:

    1. Encoder $\phi$ (default: :class:`~.iResNet`)
    2. Filter  $F$    (default: :class:`~torch.nn.GRUCell`)
    3. System  $\Psi$ (default: :class:`~.LinODECell`)
    4. Decoder $\pi$  (default: :class:`~.iResNet`)

    .. math::
        \hat x_i  &=  \pi(\hat z_i) \\
        \hat x_i' &= F(\hat x_i, x_i) \\
        \hat z_i' &= \phi(\hat x_i') \\
        \hat z_{i+1} &= \Psi(\hat z_i', \Delta t_i)


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

    HP: dict = {
        'input_size': int,
        'hidden_size': int,
        'output_size': int,
        'embedding_type' : 'linear',
        'concat_mask' : True,
        'System': LinODECell,
        'System_cfg': {'input_size': int, 'kernel_initialization': None},
        'Filter': nn.GRUCell,
        'Filter_cfg': {'input_size': int, 'hidden_size': int, 'bias': True},
        'Encoder': iResNet,
        'Encoder_cfg': {'input_size': int, 'nblocks': 5},
        'Decoder': iResNet,
        'Decoder_cfg': {'input_size': int, 'nblocks': 5},
    }

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    concat_mask: Final[bool]
    kernel: Tensor
    padding: Tensor

    def __init__(self, input_size: int, hidden_size: int, **HP: Any):
        r"""

        Parameters
        ----------
        input_size:  int
            The dimensionality of the input space.
        hidden_size: int
            The dimensionality of the latent space. Must be greater than ``input_size``.
        HP: dict
            Hyperparameter configuration
        """
        super().__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size = input_size
        self.hidden_size = input_size
        self.output_size = input_size
        self.concat_mask = HP['concat_mask']

        HP['Encoder_cfg']['input_size'] = hidden_size
        HP['Decoder_cfg']['input_size'] = hidden_size
        HP['System_cfg']['input_size'] = hidden_size
        HP['Filter_cfg']['hidden_size'] = input_size
        HP['Filter_cfg']['input_size'] = (1 + self.concat_mask) * input_size

        self.encoder = HP['Encoder'](**HP['Encoder_cfg'])
        self.decoder = HP['Decoder'](**HP['Decoder_cfg'])
        self.filter = HP['Filter'](**HP['Filter_cfg'])
        self.system = HP['System'](**HP['System_cfg'])

        embedding_type = HP['embedding_type']

        if embedding_type == 'linear':
            self.embedding = nn.Linear(input_size, hidden_size)
            self.projection = nn.Linear(hidden_size, input_size)
        elif embedding_type == 'concat':
            assert input_size <= hidden_size, F"{embedding_type=} not possible"
            self.embedding = ConcatEmbedding(input_size, hidden_size)
            self.projection = ConcatProjection(input_size, hidden_size)
        else:
            raise NotImplementedError(F"{embedding_type=}" + "not in {'linear', 'concat'}")

        self.kernel = self.system.kernel

    # @jit.script_method
    # def embedding(self, x: Tensor) -> Tensor:
    #     r"""Maps $X\mapsto \big(\begin{smallmatrix}X\\w\end{smallmatrix}\big)$
    #     if ``embedding_type='concat'``
    #
    #     Else linear function $X\mapsto XA$ if ``embedding_type='linear'``
    #
    #     Parameters
    #     ----------
    #     x: Tensor, shape=(...,DIM)
    #
    #     Returns
    #     -------
    #     Tensor, shape=(...,LAT)
    #     """
    #     return torch.cat([x, self.padding], dim=-1)
    #
    # @jit.script_method
    # def projection(self, z: Tensor) -> Tensor:
    #     r"""Maps $Z = \big(\begin{smallmatrix}X\\w\end{smallmatrix}\big) \mapsto X$
    #     if ``embedding_type='concat'``
    #
    #     Else linear function $Z\mapsto ZA$ if ``embedding_type='linear'``
    #
    #     Parameters
    #     ----------
    #     z: Tensor, shape=(...,LAT)
    #
    #     Returns
    #     -------
    #     Tensor, shape=(...,DIM)
    #     """
    #     return z[..., :self.input_size]

    @jit.script_method
    def forward(self, T: Tensor, X: Tensor) -> Tensor:
        r"""Implementation of equations (1-4)

        Optimization notes: https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Parameters
        ----------
        T: Tensor, shape=(...,LEN) or PackedSequence
            The timestamps of the observations.
        X: Tensor, shape=(...,LEN,DIM) or PackedSequence
            The observed, noisy values at times $t\in T$. Use ``NaN`` to indicate missing values.

        Returns
        -------
        Xhat: Tensor, shape=(...,LEN,DIM)
            The estimated true state of the system at the times $t\in T$.
        """

        ΔT = torch.moveaxis(torch.diff(T), -1, 0)
        X = torch.moveaxis(X, -2, 0)

        Xhat = torch.jit.annotate(list[Tensor], [])
        # initial imputation: put zeros
        # TODO: do something smarter!
        zero = torch.tensor(0, device=X.device, dtype=X.dtype)
        X0 = torch.where(torch.isnan(X[0]), zero, X[0])
        Xhat += [X0]

        # IDEA: The problem is the initial state of RNNCell is not defined and typically put equal
        # to zero. Staying with the idea that the Cell acts as a filter, that is updates the state
        # estimation given an observation, we could "trust" the original observation in the sense
        # that we solve the fixed point equation h0 = g(x0, h0) and put the solution as the initial
        # state.
        # issue: if x0 is really sparse this is useless.
        # better idea: we probably should go back and forth.
        # other idea: use a set-based model and put h = g(T,X), including the whole TS.
        # This set model can use triplet notation.
        # bias wheigting towards close time points

        for Δt, x in zip(ΔT, X):
            # Encode
            zhat = self.encoder(self.embedding(Xhat[-1]))

            # Propagate
            zhat = self.system(Δt, zhat)

            # Decode
            xhat = self.projection(self.decoder(zhat))

            # Compute update
            mask = torch.isnan(x)
            xtilde = torch.where(mask, xhat, x)

            if self.concat_mask:
                xtilde = torch.cat([xtilde, mask], dim=-1)

            # Flatten for GRU-Cell
            xhat = xhat.view(-1, xhat.shape[-1])
            xtilde = xtilde.view(-1, xtilde.shape[-1])

            # Apply filter
            xhat = self.filter(xtilde, xhat)

            # xhat = self.control(xhat, u)
            # u: possible controls:
            #  1. set to value
            #  2. add to value
            # do these via indicator variable
            # u = (time, value, mode-indicator, col-indicator)
            # => apply control to specific column.

            Xhat += [xhat.view(x.shape)]

        return torch.stack(Xhat, dim=-2)


class ConcatEmbedding(jit.ScriptModule):
    r"""Maps $X\mapsto \big(\begin{smallmatrix}X\\w\end{smallmatrix}\big)$

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    pad_size:    int
    """
    input_size  : Final[int]
    hidden_size : Final[int]
    pad_size    : Final[int]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        assert input_size <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pad_size = hidden_size - input_size
        self.padding = nn.Parameter(torch.randn(self.pad_size))

    @jit.script_method
    def forward(self, X: Tensor) -> Tensor:
        """
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
    r"""Maps $Z = \big(\begin{smallmatrix}X\\w\end{smallmatrix}\big) \mapsto X$

    Attributes
    ----------
    input_size:  int
    hidden_size: int
    """

    input_size  : Final[int]
    hidden_size : Final[int]

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        assert input_size <= hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size

    @jit.script_method
    def forward(self, Z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        Z: Tensor, shape=(...,LEN,LAT)

        Returns
        -------
        Tensor, shape=(...,LEN,DIM)
        """
        return Z[..., :self.input_size]
