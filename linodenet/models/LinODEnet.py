import torch
from torch import nn, Tensor, jit
from typing import Union, List, Callable, Final
import numpy as np

from tsdm.util import deep_dict_update
from .iResNet import iResNet


class LinODECell(jit.ScriptModule):
    r"""Linear System module, solves $\dot x = Ax$

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
    kernel_initialization: Callable[None, Tensor]

    def __init__(self, input_size: int,
                 kernel_initialization: Union[Tensor, Callable[int, Tensor]] = None,
                 ):
        r"""

        Parameters
        ----------
        input_size: int
        kernel_initialization: Union[Tensor, Callable[int, Tensor]]
        """

        super(LinODECell, self).__init__()
        self.input_size = input_size
        self.output_size = input_size

        if kernel_initialization is None:
            self.kernel_initialization = lambda: torch.randn(input_size, input_size) / np.sqrt(input_size)
        elif callable(kernel_initialization):
            self.kernel_initialization = lambda: Tensor(kernel_initialization(input_size))
        elif type(kernel_initialization) == torch.Tensor:
            self._kernel_initialization = kernel_initialization.clone().detach()
            self.kernel_initialization = lambda: self._kernel_initialization.clone()
        else:
            self.kernel_initialization = lambda: Tensor(kernel_initialization)

        self.kernel = nn.Parameter(self.kernel_initialization())

    @jit.script_method
    def forward(self, Δt: Tensor, x: Tensor) -> Tensor:
        return self.__forward__(Δt, x)

    def __forward__(self, Δt: Tensor, x: Tensor) -> Tensor:
        r"""Forward using the matrix exponential $\hat x = e^{A\Delta t}x$

        TODO: optimize if clauses away by changing definition in constructor.

        Parameters
        ----------
        Δt: Tensor
            The time difference $t_1 - t_0$ between $x$ and $\hat x$.
        x:  Tensor
            Time observed value at $t_0$

        Returns
        -------
        xhat:  Tensor
            The predicted value at $t_1$
        """

        AΔt = torch.einsum('kl, ... -> ...kl', self.kernel, Δt)
        expAΔt = torch.matrix_exp(AΔt)
        xhat = torch.einsum('...kl, ...l -> ...k', expAΔt, x)

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
    kernel_initialization: Callable[None, Tensor]

    def __init__(self, input_size: int, kernel_initialization: Union[Tensor, Callable[int, Tensor]] = None):
        r"""

        Parameters
        ----------
        input_size: int
        kernel_initialization: Union[Tensor, Callable[int, Tensor]]
        """
        super(LinODE, self).__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.cell = LinODECell(input_size, kernel_initialization)
        self.kernel = self.cell.kernel

    def __forward__(self, x0: Tensor, T: Tensor) -> Tensor:
        r"""
        Propagate x0

        Parameters
        ----------
        x0: Tensor
        T: Tensor

        Returns
        -------
        Xhat: Tensor
            The estimated true state of the system at the times $t\in T$
        """
        ΔT = torch.diff(T)
        x = torch.jit.annotate(List[Tensor], [])
        x += [x0]

        for i, Δt in enumerate(ΔT):
            x += [self.cell(Δt, x[-1])]

        return torch.stack(x)

    @jit.script_method
    def forward(self, x0: Tensor, T: Tensor) -> Tensor:
        return self.__forward__(x0, T)


class LinODEnet(jit.ScriptModule):
    r"""Linear ODE Network, consisting of 4 components:

    1. Encoder $\phi$ (default: :class:`~.iResNet`)
    2. Filter  $F$    (default: :class:`~torch.nn.GRUCell`)
    3. Process $\Psi$ (default: :class:`~.LinODECell`)
    4. Decoder $\pi$  (default: :class:`~.iResNet`)

    .. math::
        \begin{aligned}
            \hat x_i  &=  \pi(\hat z_i) \\
            \hat x_i' &= F(\hat x_i, x_i) \\
            \hat z_i' &= \phi(\hat x_i') \\
            \hat z_{i+1} &= \Psi(\hat z_i', \Delta t_i)
        \end{aligned}


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
    """

    HP = {
        'input_size': None,
        'hidden_size': None,
        'Process': LinODECell,
        'Process_cfg': {'input_size': None, 'kernel_initialization': None},
        'Updater': nn.GRUCell,
        'Updater_cfg': {'input_size': None, 'hidden_size': None, 'bias': True},
        'Encoder': iResNet,
        'Encoder_cfg': {'input_size': None, 'nblocks': 5},
        'Decoder': iResNet,
        'Decoder_cfg': {'input_size': None, 'nblocks': 5},
    }

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    kernel: Tensor

    def __init__(self, input_size: int, hidden_size: int, **HP):
        r"""

        Parameters
        ----------
        input_size:  int
            The dimensionality of the input space.
        hidden_size: int
            The dimensionality of the latent space. Must be greater than the dimensionality of the input space.
        HP: dict
            Hyperparameter configuration
        """
        assert hidden_size > input_size
        super(LinODEnet, self).__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size = input_size
        self.hidden_size = input_size
        self.output_size = input_size
        HP['Process_cfg']['input_size'] = hidden_size
        HP['Encoder_cfg']['input_size'] = hidden_size
        HP['Decoder_cfg']['input_size'] = hidden_size
        HP['Updater_cfg']['hidden_size'] = input_size
        HP['Updater_cfg']['input_size'] = 2 * input_size

        self.padding = nn.Parameter(torch.randn(hidden_size - input_size))
        self.Encoder = HP['Encoder'](**HP['Encoder_cfg'])
        self.Decoder = HP['Decoder'](**HP['Decoder_cfg'])
        self.Updater = HP['Updater'](**HP['Updater_cfg'])
        self.Process = HP['Process'](**HP['Process_cfg'])
        self.kernel = self.Process.kernel

    @jit.script_method
    def embed(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.padding], dim=-1)

    @jit.script_method
    def project(self, z: Tensor) -> Tensor:
        return z[..., :self.input_size]

    def __forward__(self, T: Tensor, X: Tensor) -> Tensor:
        r"""Implementation inspired by https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Parameters
        ----------
        T: Tensor
            The timestamps of the observations.
        X: Tensor
            The observed, potentially noisy values at times $t\in T$. Use ``NaN`` to indicate missing values.

        Returns
        -------
        Xhat: Tensor
            The estimated true state of the system at the times $t\in T$.
        """
        ΔT = torch.diff(T)
        Xhat = torch.jit.annotate(List[Tensor], [])
        # initialize with zero, todo: do something smarter!
        Xhat += [torch.where(torch.isnan(X[0]), torch.zeros(1), X[0])]

        for Δt, x in zip(ΔT, X):
            # Encode
            zhat = self.Encoder(self.embed(Xhat[-1]))

            # Propagate
            zhat = self.Process(Δt, zhat)

            # Decode
            xhat = self.project(self.Decoder(zhat))

            # Compute update
            mask = torch.isnan(x)
            xtilde = torch.where(mask, xhat, x)
            chat = torch.unsqueeze(torch.cat([xtilde, mask], dim=-1), dim=0)
            Xhat += [self.Updater(chat, torch.unsqueeze(xhat, dim=0)).squeeze()]

        return torch.stack(Xhat)

    @jit.script_method
    def forward(self, T: Tensor, X: Tensor) -> Tensor:
        return self.__forward__(T, X)


class LinODEnet_v2(jit.ScriptModule):
    r"""Use Linear embedding instead of padding

    """

    HP = {
        'input_size': int,
        'hidden_size': int,
        'Process': LinODECell,
        'Process_cfg': {'input_size': int, 'kernel_initialization': None},
        'Updater': nn.GRUCell,
        'Updater_cfg': {'input_size': int, 'hidden_size': int, 'bias': True},
        'Encoder': iResNet,
        'Encoder_cfg': {'input_size': int, 'nblocks': 5},
        'Decoder': iResNet,
        'Decoder_cfg': {'input_size': int, 'nblocks': 5},
    }

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]

    def __init__(self, input_size: int, hidden_size: int, **HP):
        assert hidden_size > input_size
        super(LinODEnet_v2, self).__init__()

        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size = input_size
        self.hidden_size = input_size
        self.output_size = input_size
        HP['Process_cfg']['input_size'] = hidden_size
        HP['Encoder_cfg']['input_size'] = hidden_size
        HP['Decoder_cfg']['input_size'] = hidden_size
        HP['Updater_cfg']['hidden_size'] = input_size
        HP['Updater_cfg']['input_size'] = 2 * input_size

        self.Updater = HP['Updater'](**HP['Updater_cfg'])
        self.Process = HP['Process'](**HP['Process_cfg'])
        self.Encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            HP['Encoder'](**HP['Encoder_cfg']),
        )
        self.Decoder = nn.Sequential(
            HP['Decoder'](**HP['Decoder_cfg']),
            nn.Linear(hidden_size, input_size),
        )

    @jit.script_method
    def forward(self, T: Tensor, X: Tensor) -> Tensor:
        r"""
        Input:  times t, measurements x. NaN for missing value.

        Output: prediction y(t_i) for all t
        """

        ΔT = torch.diff(T)
        Xhat = torch.empty((len(T), self.output_size))

        # initialize with zero, todo: do something smarter!
        xhat = torch.where(torch.isnan(X[0]), torch.zeros(1), X[0])
        Xhat[0] = xhat

        for i, (Δt, x) in enumerate(zip(ΔT, X)):
            # Encode
            zhat = self.Encoder(xhat)

            # Propagate
            zhat = self.Process(Δt, zhat)

            # Decode
            xhat = self.Decoder(zhat)

            # Compute update
            mask = torch.isnan(x)
            xtilde = torch.where(mask, xhat, x)
            chat = torch.unsqueeze(torch.cat([xtilde, mask], dim=-1), dim=0)
            xhat = self.Updater(chat, torch.unsqueeze(xhat, dim=0)).squeeze()
            Xhat[i + 1] = xhat

        return Xhat
