import math
import warnings
from typing import Union

import torch
from torch import jit, nn, Tensor
from torch.nn import functional
from tsdm.util import ACTIVATIONS, deep_dict_update


class LinearContraction(jit.ScriptModule):
    r"""
    LinearContraction implementation
    """
    __constants__ = ['input_size', 'output_size']
    input_size: int
    output_size: int
    weight: Tensor
    bias: Union[Tensor, None]

    def __init__(self, input_size: int, output_size: int, bias: bool = True) -> None:
        r"""
        Initialize Linear Contraction

        Parameters
        ----------
        input_size: int
        output_size: int
        bias: bool
        """
        super(LinearContraction, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'input_size={}, output_size={}, bias={}'.format(
            self.input_size, self.output_size, self.bias is not None
        )

    @jit.script_method
    def alt_forward(self, x: Tensor, c: float = 0.97) -> Tensor:
        σ_max = torch.linalg.norm(self.weight, ord=2)
        fac = torch.minimum(c / σ_max, torch.ones(1))
        return functional.linear(x, fac * self.weight, self.bias)


class iResNetBlock(jit.ScriptModule):
    r"""
    invertible ResNet-Block implementation
    """

    __constants__ = ['input_size', 'output_size', 'maxiter']
    input_size: int
    hidden_size: int
    output_size: int
    maxiter: int
    bias: bool

    HP = {
        'activation': 'ReLU',
        'activation_config': {'inplace': False},
        'bias': True,
        'hidden_size': None,
        'input_size': None,
        'maxiter': 100,
    }

    def __init__(self, input_size: int, **HP):
        r"""
        Initialize iResNetBlock

        Parameters
        ----------
        input_size: int
        HP: dict
        """
        super(iResNetBlock, self).__init__()

        self.HP['input_size'] = input_size
        deep_dict_update(self.HP, HP)

        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = self.HP['hidden_size'] or math.ceil(math.sqrt(input_size))

        self.maxiter = self.HP['maxiter']
        self.bias = self.HP['bias']

        activation = ACTIVATIONS[self.HP['activation']]

        self.bottleneck = nn.Sequential(
            LinearContraction(self.input_size, self.hidden_size, self.bias),
            LinearContraction(self.hidden_size, self.input_size, self.bias),
            activation(**self.HP['activation_config']),
        )

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass

        Parameters
        ----------
        x: :class:`torch.Tensor`

        Returns
        -------
        xhat: :class:`torch.Tensor`
        """
        xhat = x + self.bottleneck(x)

        return xhat

    @jit.script_method
    def inverse(self, y: Tensor,
                maxiter: int = 1000, rtol: float = 1e-05, atol: float = 1e-08) -> Tensor:
        r"""
        Inverse pass

        Parameters
        ----------
        y: :class:`torch.Tensor`
        maxiter: int
        rtol: float
        atol: float

        Returns
        -------
        xhat: :class:`torch.Tensor`
        """
        xhat = y.clone()
        xhat_dash = y.clone()
        residual = torch.zeros_like(y)

        for k in range(self.maxiter):
            xhat_dash = y - self.bottleneck(xhat)
            residual = torch.abs(xhat_dash - xhat) - rtol * torch.absolute(xhat)

            if torch.all(residual <= atol):
                return xhat_dash
            else:
                xhat = xhat_dash

        warnings.warn(F"No convergence in {maxiter} iterations. Max residual:{torch.max(residual)} > {atol}.")
        return xhat_dash


class iResNet(jit.ScriptModule):
    r"""
    invertible ResNet implementation

    """
    HP = {
        'maxiter': 10,
        'input_size': None,
        'dropout': None,
        'bias': True,
        'nBlocks': 5,
        'iResNetBlock': {
            'input_size': None,
            'activation': 'ReLU',
            'activation_config': {'inplace': False},
            'bias': True,
            'hidden_size': None,
            'maxiter': 100,
        },
    }

    input_size: int
    output_size: int
    nblocks: int

    def __init__(self, input_size, **HP) -> None:
        r"""
        Initialize iResNet

        Parameters
        ----------
        input_size: int
        HP: dict
        """
        super(iResNet, self).__init__()

        self.HP['input_size'] = input_size
        deep_dict_update(self.HP, HP)

        self.input_size = input_size
        self.output_size = input_size
        self.HP['iResNetBlock']['input_size'] = self.input_size

        self.nblocks = self.HP['nBlocks']
        self.maxiter = self.HP['maxiter']
        self.bias = self.HP['bias']

        self.blocks = nn.Sequential(*[
            iResNetBlock(**self.HP['iResNetBlock']) for _ in range(self.nblocks)
        ])

        self.reversed_blocks = nn.Sequential(*reversed(list(self.blocks)))

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass

        Parameters
        ----------
        x: :class:`torch.Tensor`

        Returns
        -------
        xhar: :class:`torch.Tensor`
        """

        for block in self.blocks:
            x = block(x)

        return x

    @jit.script_method
    def inverse(self, y: Tensor) -> Tensor:
        r"""
        Inverse pass

        Parameters
        ----------
        y: :class:`torch.Tensor`

        Returns
        -------
        yhat: :class:`torch.Tensor`
        """

        with torch.no_grad():
            for block in self.reversed_blocks:
                # `reversed` does not work in torchscript v1.8.1
                y = block.inverse(y)

        return y

    @jit.script_method
    def alt_inverse(self, y: Tensor,
                    maxiter: int = 1000, rtol: float = 1e-05, atol: float = 1e-08) -> Tensor:
        r"""
        Alternative Inverse Pass

        Parameters
        ----------
        y: :class:`torch.Tensor`
        maxiter: int
        rtol: float
        atol: float

        Returns
        -------
        yhat: :class:`torch.Tensor`
        """

        xhat = y.clone()
        xhat_dash = y.clone()
        residual = torch.zeros_like(y)

        for k in range(self.maxiter):
            xhat_dash = y - self(xhat)
            residual = torch.abs(xhat_dash - xhat) - rtol * torch.absolute(xhat)

            if torch.all(residual <= atol):
                return xhat_dash
            else:
                xhat = xhat_dash

        warnings.warn(F"No convergence in {maxiter} iterations. Max residual:{torch.max(residual)} > {atol}.")
        return xhat_dash
