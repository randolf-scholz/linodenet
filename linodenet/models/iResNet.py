r"""
iResNet
=======

Contains implementations of

- class:`~.LinearContraction`
- class:`~.iResNetBlock`
- class:`~.iResNet`
"""
from math import sqrt
from typing import Union, Final

import torch
from torch import jit, nn, Tensor
from torch.nn import functional
from tsdm.util import ACTIVATIONS, deep_dict_update


class DummyModel(jit.ScriptModule):
    """My dummy model"""

    def __init__(self, const: float):
        super().__init__()
        self.const = Tensor(const)

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        """This method shows as a :undoc-member: in the documentation"""
        return self.my_forward(x)

    def my_forward(self, x: Tensor) -> Tensor:
        """This method shows as a :member: in the documentation"""
        return x + self.const


class LinearContraction(jit.ScriptModule):
    r"""A linear layer $f(X) = A\cdot X$ satisfying the contraction property
    $\|f(X)-f(y)\|_2 \le \|X-y\|_2$

    This is achieved by normalizing the weight matrix by
    $\tilde{A} = A \cdot \min\big(\tfrac{c}{\|A\|_2}, 1\big)$, where $c<1$ is a hyperparameter.

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    c: Tensor
        The regularization hyperparameter
    weight: Tensor
        The weight matrix
    bias: Tensor or None
        The bias Tensor if present, else None.
    """
    input_size: Final[int]
    output_size: Final[int]

    c: Tensor
    weight: Tensor
    bias: Union[Tensor, None]

    def __init__(self, input_size: int, output_size: int, c: float = 0.97, bias: bool = True):
        r"""
        Parameters
        ----------
        input_size: int
            The dimensionality of the input space.
        output_size: int
            The dimensionality of the output space.
        c: float
            The regularization hyperparameter.
        bias: bool
            Whether to include bias term.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.c = torch.tensor(float(c))

        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset both weight matrix and bias vector"""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return 'input_size={}, output_size={}, bias={}'.format(
            self.input_size, self.output_size, self.bias is not None
        )

    def __forward__(self, x: Tensor) -> Tensor:
        r"""

        $X\mapsto \tilde A\cdot X$ where $\tilde{A} = A \cdot \min\big(\tfrac{c}{\|A\|_2}, 1\big)$

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        σ_max = torch.linalg.norm(self.weight, ord=2)
        one = torch.tensor(1, dtype=x.dtype, device=x.device)
        fac = torch.minimum(self.c / σ_max, one)
        return functional.linear(x, fac * self.weight, self.bias)

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        """This method shows as a :undoc-member: in the documentation"""
        return self.__forward__(x)


class iResNetBlock(jit.ScriptModule):
    r"""Invertible ResNet-Block of the form $g(X)=\phi(W_1\cdot W_2\cdot X)$,
    where $W_1 \cdot W_2$ is a low rank factorization.

    Alternative: $g(X) = W_3\phi(W_2\phi(W_1\cdot X))$

    All linear layers must be :class:`LinearContraction` layers.
    The activation function must have Lipschitz constant $\le 1$ such as :class:`~torch.nn.ReLU`,
    :class:`~torch.nn.ELU` or :class:`~torch.nn.Tanh`)

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    hidden_size: int, default = :math:`\lfloor\sqrt{n}\rfloor`
        The dimensionality of the latent space.
    output_size: int
        The dimensionality of the output space.
    maxiter: int
        Maximum number of iteration in `inverse` pass
    bottleneck:  nn.Sequential
        The bottleneck layers
    bias: bool, default=True
        Whether to use bias
    HP: dict
        Nested dictionary containing the hyperparameters.
    """
    input_size:  Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    maxiter:     Final[int] = 10
    atol:        Final[float] = 1e-08
    rtol:        Final[float] = 1e-05

    HP: dict = {
        'atol' : 1e-08,
        'rtol' : 1e-05,
        'maxiter': 10,
        'activation': 'ReLU',
        'activation_config': {'inplace': False},
        'bias': True,
        'output_size' : None,
        'hidden_size' : None,
        'input_size'  : None,
    }

    def __init__(self, input_size: int, **HP):
        r"""
        Parameters
        ----------
        input_size: int
            The dimensionality of the input space.
        HP: dict
            Nested dictionary containing the hyperparameters.
        """
        super().__init__()

        self.HP['input_size'] = input_size
        deep_dict_update(self.HP, HP)
        HP = self.HP

        self.input_size  = HP['input_size'] = input_size
        self.output_size = HP['input_size'] = input_size
        self.hidden_size = HP['hidden_size'] = HP['hidden_size'] or int(sqrt(input_size))

        self.atol = HP['atol']
        self.rtol = HP['rtol']
        self.maxiter = HP['maxiter']
        self.bias = HP['bias']

        self.bottleneck = nn.Sequential(
            LinearContraction(self.input_size, self.hidden_size, self.bias),
            LinearContraction(self.hidden_size, self.input_size, self.bias),
            ACTIVATIONS[HP['activation']](**HP['activation_config']),
        )

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        return x + self.bottleneck(x)

    @jit.script_method
    def inverse(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration. Terminates once ``maxiter``
        or tolerance threshold :math:`|X'-X| \le \text{atol} + \text{rtol}\cdot |X|` is reached.


        Parameters
        ----------
        y: Tensor

        Returns
        -------
        Tensor
        """
        xhat_dash = y.clone()
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            xhat = xhat_dash
            xhat_dash = y - self.bottleneck(xhat)
            residual = torch.abs(xhat_dash - xhat) - self.rtol * torch.absolute(xhat)

            if torch.all(residual <= self.atol):
                break
        # if k == self.maxiter:
        #     warnings.warn(F"No convergence in {self.maxiter} iterations. "
        #                   F"Max residual:{torch.max(residual)} > {self.atol}.")
        return xhat_dash


class iResNet(jit.ScriptModule):
    r"""Invertible ResNet consists of a stack of :class:`iResNetBlock` modules.

    Attributes
    ----------
    input_size: int
        The dimensionality of the input space.
    output_size: int
        The dimensionality of the output space.
    blocks:  nn.Sequential
        Sequential model consisting of the iResNetBlocks
    reversed_blocks: nn.Sequential
        The same blocks in reversed order
    HP: dict
        Nested dictionary containing the hyperparameters.
    """

    input_size:  Final[int]
    output_size: Final[int]

    HP: dict = {
        'maxiter': 10,
        'input_size': None,
        'dropout': None,
        'bias': True,
        'nblocks': 5,
        'iResNetBlock': {
            'input_size': None,
            'activation': 'ReLU',
            'activation_config': {'inplace': False},
            'bias': True,
            'hidden_size': None,
            'maxiter': 100,
        },
    }

    def __init__(self, input_size, **HP):
        r"""
        Parameters
        ----------
        input_size: int
        HP: dict
        """
        super().__init__()

        self.HP['input_size'] = input_size
        deep_dict_update(self.HP, HP)

        self.input_size = input_size
        self.output_size = input_size
        self.HP['iResNetBlock']['input_size'] = self.input_size

        self.nblocks = self.HP['nblocks']
        self.maxiter = self.HP['maxiter']
        self.bias = self.HP['bias']

        blocks = []

        for _ in range(self.nblocks):
            blocks += [iResNetBlock(**self.HP['iResNetBlock'])]
            # TODO: add regularization

        self.blocks = nn.Sequential(*blocks)
        self.reversed_blocks = nn.Sequential(*reversed(blocks))

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        x: Tensor

        Returns
        -------
        xhat: Tensor
        """

        return self.blocks(x)

    @jit.script_method
    def inverse(self, y: Tensor) -> Tensor:
        r"""Computes the inverse through fix point iteration in each block in reversed order.

        Parameters
        ----------
        y: Tensor

        Returns
        -------
        yhat: Tensor
        """
        for block in self.reversed_blocks:
            # `reversed` does not work in torchscript v1.8.1
            y = block.inverse(y)

        return y

    # TODO: delete this?
    # @jit.script_method
    # def alt_inverse(self, y: Tensor,
    #                 maxiter: int = 1000, rtol: float = 1e-05, atol: float = 1e-08) -> Tensor:
    #     r"""
    #     Parameters
    #     ----------
    #     y: Tensor
    #     maxiter: int
    #     rtol: float
    #     atol: float
    #
    #     Returns
    #     -------
    #     yhat: Tensor
    #     """
    #     xhat = y.clone()
    #     xhat_dash = y.clone()
    #     residual = torch.zeros_like(y)
    #
    #     for k in range(self.maxiter):
    #         xhat_dash = y - self(xhat)
    #         residual = torch.abs(xhat_dash - xhat) - rtol * torch.absolute(xhat)
    #
    #         if torch.all(residual <= atol):
    #             return xhat_dash
    #         else:
    #             xhat = xhat_dash
    #
    # warnings.warn(F"No convergence in {maxiter} iterations. "
    #               F"Max residual:{torch.max(residual)} > {atol}.")
    #     return xhat_dash
