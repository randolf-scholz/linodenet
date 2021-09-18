r"""Implementation of invertible ResNets."""

import logging
from math import sqrt
from typing import Any, Final, Union

import torch
from torch import Tensor, jit, nn
from torch.nn import functional

from linodenet.util import ACTIVATIONS, deep_dict_update

# from torch.nn.utils.parametrizations import spectral_norm


logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "iResNet",
    "iResNetBlock",
    "LinearContraction",
]


@jit.script
def spectral_norm(
    A: Tensor, atol: float = 1e-4, rtol: float = 1e-3, maxiter: int = 10
) -> Tensor:
    r"""Compute the spectral norm :math:`‖A‖_2`.

    Simple power iteration, using the fact that :math:`‖A‖_2=λ_{𝗆𝖺𝗑}(A^𝖳A)`.
    """
    m, n = A.shape

    with torch.no_grad():
        x = torch.randn(n, device=A.device, dtype=A.dtype)
        x = x / torch.linalg.norm(x)

        z = A.T @ (A @ x)
        c, d = torch.linalg.norm(z, dim=0), torch.linalg.norm(x, dim=0)
        λ = c / d
        r = z - λ * x

        for _ in range(maxiter):
            x = z / c
            z = A.T @ (A @ x)
            c, d = torch.linalg.norm(z, dim=0), torch.linalg.norm(x, dim=0)
            λ = c / d
            r = z - λ * x
            if torch.linalg.norm(r) <= rtol * torch.linalg.norm(λ * x) + atol:
                break
        # raise RuntimeWarning("No convergence in {maxiter} iterations.")
        return torch.sqrt(λ)


class SpectralNorm(jit.ScriptModule):
    r"""`‖A‖_2=λ_{𝗆𝖺𝗑}(A^𝖳A)`.

    The spectral norm `∥A∥_2 ≔ 𝗌𝗎𝗉_x ∥Ax∥_2 / ∥x∥_2` can be shown to be equal to
    `σ_\max(A) = √{λ_{𝗆𝖺𝗑} (AᵀA)}`, the largest singular value of `A`.

    It can be computed efficiently via Power iteration.

    One can show that the derivative is equal to:

    .. math::
        \frac{∂½∥A∥_2}/{∂A} = uvᵀ

    where `u,v` are the left/right-singular vector corresponding to `σ_\max`
    """


class LinearContraction(jit.ScriptModule):
    r"""A linear layer `f(x) = A⋅x` satisfying the contraction property `‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2`.

    This is achieved by normalizing the weight matrix by
    `A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)`, where `c<1` is a hyperparameter.

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

    C: Tensor
    ONE: Tensor

    weight: Tensor
    bias: Union[Tensor, None]

    def __init__(
        self, input_size: int, output_size: int, c: float = 0.97, bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.C = torch.tensor(float(c), requires_grad=False)
        self.ONE = torch.tensor(1.0, requires_grad=False)
        self.weight = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self) -> str:
    #     return "input_size={}, output_size={}, bias={}".format(
    #         self.input_size, self.output_size, self.bias is not None
    #     )

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""Signature: `[...,n] ⟶ [...,n]`.

        Parameters
        ----------
        x: Tensor

        Returns
        -------
        Tensor
        """
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        σ_max = spectral_norm(self.weight)
        fac = torch.minimum(self.C / σ_max, self.ONE)
        return functional.linear(x, fac * self.weight, self.bias)


#
#
#
# class LinearContraction(jit.ScriptModule):
#     r"""A linear layer `f(x) = A⋅x` satisfying the contraction property `‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2`.
#
#     This is achieved by normalizing the weight matrix by
#     `A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)`, where `c<1` is a hyperparameter.
#     Equivalently, we can scale the input by the appropriate factor.
#
#     Attributes
#     ----------
#     input_size:  int
#         The dimensionality of the input space.
#     output_size: int
#         The dimensionality of the output space.
#     c: Tensor
#         The regularization hyperparameter
#     weight: Tensor
#         The weight matrix
#     bias: Tensor or None
#         The bias Tensor if present, else None.
#     """
#
#     input_size: Final[int]
#     output_size: Final[int]
#
#     C: Tensor
#     ONE: Tensor
#
#     weight: Tensor
#     bias: Tensor
#     # linear: nn.Module
#
#     def __init__(
#         self, input_size: int, output_size: int, c: float = 0.97, bias: bool = True
#     ):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.C = torch.tensor(float(c))
#         self.ONE = torch.tensor(1.0)
#         self.linear = spectral_norm(nn.Linear(input_size, output_size, bias=bias))
#         self.weight = self.linear.weight
#         self.bias = self.linear.bias
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         r"""Reset both weight matrix and bias vector."""
#         nn.initializations.kaiming_uniform_(self.weight, a=sqrt(5))
#         if self.bias is not None:
#             bound = 1 / sqrt(self.input_size)
#             nn.initializations.uniform_(self.bias, -bound, bound)
#
#     # def extra_repr(self) -> str:
#     #     return "input_size={}, output_size={}, bias={}".format(
#     #         self.input_size, self.output_size, self.bias is not None
#     #     )
#
#     @jit.script_method
#     def forward(self, x: Tensor) -> Tensor:
#         r"""Signature: `[...,n] ⟶ [...,n]`.
#
#         Parameters
#         ----------
#         x: Tensor
#
#         Returns
#         -------
#         Tensor
#         """
#         # σ_max = torch.linalg.norm(self.weight, ord=2)
#         # fac = torch.minimum(self.C / σ_max, self.ONE)
#         return self.linear(self.C*x)
#         # return functional.linear(x, fac * self.weight, self.bias)


class iResNetBlock(jit.ScriptModule):
    r"""Invertible ResNet-Block of the form `g(x)=ϕ(W_1⋅W_2⋅x)`.

    By default, `W_1⋅W_2` is a low rank factorization.

    Alternative: `g(x) = W_3ϕ(W_2ϕ(W_1⋅x))`

    All linear layers must be :class:`LinearContraction` layers.
    The activation function must have Lipschitz constant `≤1` such as :class:`~torch.nn.ReLU`,
    :class:`~torch.nn.ELU` or :class:`~torch.nn.Tanh`)

    Attributes
    ----------
    input_size:  int
        The dimensionality of the input space.
    hidden_size: int, default=⌊√n⌋
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

    input_size: Final[int]
    hidden_size: Final[int]
    output_size: Final[int]
    maxiter: Final[int]
    atol: Final[float]
    rtol: Final[float]

    HP: dict = {
        "atol": 1e-08,
        "rtol": 1e-05,
        "maxiter": 10,
        "activation": "ReLU",
        "activation_config": {"inplace": False},
        "bias": True,
        "output_size": None,
        "hidden_size": None,
        "input_size": None,
    }

    def __init__(self, input_size: int, **HP: Any):
        super().__init__()

        self.HP["input_size"] = input_size
        deep_dict_update(self.HP, HP)
        HP = self.HP

        HP["input_size"] = input_size
        HP["input_size"] = input_size
        HP["hidden_size"] = HP["hidden_size"] or int(sqrt(input_size))

        self.input_size = HP["input_size"]
        self.output_size = HP["input_size"]
        self.hidden_size = HP["hidden_size"]

        self.atol = HP["atol"]
        self.rtol = HP["rtol"]
        self.maxiter = HP["maxiter"]
        self.bias = HP["bias"]

        self.bottleneck = nn.Sequential(
            LinearContraction(self.input_size, self.hidden_size, self.bias),
            LinearContraction(self.hidden_size, self.input_size, self.bias),
            ACTIVATIONS[HP["activation"]](**HP["activation_config"]),  # type: ignore
        )

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""Signature: `[...,n] ⟶ [...,n]`.

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
        r"""Compute the inverse through fixed point iteration.

        Terminates once `maxiter` or tolerance threshold
        `|x'-x|≤\text{atol} + \text{rtol}⋅|x|` is reached.

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
                return xhat_dash

        print(
            f"No convergence in {self.maxiter} iterations. "
            f"Max residual:{torch.max(residual)} > {self.atol}."
        )
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

    input_size: Final[int]
    output_size: Final[int]

    HP: dict = {
        "maxiter": 10,
        "input_size": None,
        "dropout": None,
        "bias": True,
        "nblocks": 5,
        "iResNetBlock": {
            "input_size": None,
            "activation": "ReLU",
            "activation_config": {"inplace": False},
            "bias": True,
            "hidden_size": None,
            "maxiter": 100,
        },
    }

    def __init__(self, input_size: int, **HP: Any):
        super().__init__()

        self.HP["input_size"] = input_size
        deep_dict_update(self.HP, HP)

        self.input_size = input_size
        self.output_size = input_size
        self.HP["iResNetBlock"]["input_size"] = self.input_size

        self.nblocks = self.HP["nblocks"]
        self.maxiter = self.HP["maxiter"]
        self.bias = self.HP["bias"]

        blocks = []

        for _ in range(self.nblocks):
            blocks += [iResNetBlock(**self.HP["iResNetBlock"])]
            # TODO: add regularization

        self.blocks = nn.Sequential(*blocks)
        self.reversed_blocks = nn.Sequential(*reversed(blocks))

    @jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        r"""Signature: `[...,n] ⟶ [...,n]`.

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
        r"""Compute the inverse through fix point iteration in each block in reversed order.

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
