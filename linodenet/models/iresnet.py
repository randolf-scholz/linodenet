r"""Implementation of invertible ResNets."""

__all__ = [
    # Classes
    "iResNet",
    "iResNetBlock",
    "LinearContraction",
    "spectral_norm",
    "SpectralNorm",
]

import logging
from math import sqrt
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.linalg import matrix_norm, vector_norm
from torch.nn import functional

from linodenet.util import ACTIVATIONS, Activation, autojit, deep_dict_update

LOGGER = logging.getLogger(__name__)


@jit.script
def spectral_norm(
    A: Tensor, atol: float = 1e-4, rtol: float = 1e-3, maxiter: int = 10
) -> Tensor:
    r"""Compute the spectral norm `‖A‖_2` by power iteration.

    Stopping criterion:
    - maxiter reached
    - `‖ (A^TA -λI)x ‖_2 ≤ 𝗋𝗍𝗈𝗅⋅‖ λx ‖_2 + 𝖺𝗍𝗈𝗅`

    Parameters
    ----------
    A: tensor
    atol: float = 1e-4
    rtol: float =  1e-3,
    maxiter: int = 10

    Returns
    -------
    Tensor
    """
    m, n = A.shape

    with torch.no_grad():
        x = torch.randn(n, device=A.device, dtype=A.dtype)
        x = x / vector_norm(x)

        z = A.T @ (A @ x)
        c, d = vector_norm(z, dim=0), vector_norm(x, dim=0)
        λ = c / d
        r = z - λ * x

        for _ in range(maxiter):
            if vector_norm(r) <= rtol * vector_norm(λ * x) + atol:
                break
            x = z / c
            z = A.T @ (A @ x)
            c, d = vector_norm(z, dim=0), vector_norm(x, dim=0)
            λ = c / d
            r = z - λ * x

        σ_max = torch.sqrt(λ)
        return σ_max


class SpectralNorm(torch.autograd.Function):
    r"""`‖A‖_2=λ_{𝗆𝖺𝗑}(A^𝖳A)`.

    The spectral norm `∥A∥_2 ≔ 𝗌𝗎𝗉_x ∥Ax∥_2 / ∥x∥_2` can be shown to be equal to
    `σ_\max(A) = √{λ_{𝗆𝖺𝗑} (AᵀA)}`, the largest singular value of `A`.

    It can be computed efficiently via Power iteration.

    One can show that the derivative is equal to:

    .. math::
        \frac{∂½∥A∥_2}/{∂A} = uvᵀ

    where `u,v` are the left/right-singular vector corresponding to `σ_\max`
    """

    @staticmethod
    def forward(ctx: Any, *tensors: Tensor, **kwargs: Any) -> Tensor:
        r"""Forward pass.

        Parameters
        ----------
        ctx
        tensors
        kwargs

        Returns
        -------
        Tensor
        """
        A = tensors[0]
        atol: float = kwargs["atol"] if "atol" in kwargs else 1e-6
        rtol: float = kwargs["rtol"] if "rtol" in kwargs else 1e-6
        maxiter: int = kwargs["maxiter"] if "maxiter" in kwargs else 1000
        m, n = A.shape
        x = torch.randn(n, device=A.device, dtype=A.dtype)
        x = x / vector_norm(x)

        z = A.T @ (A @ x)
        c, d = vector_norm(z, dim=0), vector_norm(x, dim=0)
        λ = c / d
        r = z - λ * x

        for _ in range(maxiter):
            if vector_norm(r) <= rtol * vector_norm(λ * x) + atol:
                break
            x = z / c
            z = A.T @ (A @ x)
            c, d = vector_norm(z, dim=0), vector_norm(x, dim=0)
            λ = c / d
            r = z - λ * x

        σ_max = torch.sqrt(λ)
        v = x / vector_norm(x)
        u = A @ v / σ_max
        u /= vector_norm(u)
        ctx.save_for_backward(u, v)
        return σ_max

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor:
        r"""Backward pass.

        Parameters
        ----------
        ctx
        grad_outputs
        """
        u, v = ctx.saved_tensors
        return grad_outputs[0] * torch.outer(u, v)


@autojit
class LinearContraction(nn.Module):
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
    spectral_norm: Tensor

    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self, input_size: int, output_size: int, c: float = 0.97, bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # self.spectral_norm = matrix_norm(self.weight, ord=2)
        self.register_buffer("ONE", torch.tensor(1.0))
        self.register_buffer("C", torch.tensor(float(c)))
        self.register_buffer("spectral_norm", matrix_norm(self.weight, ord=2))

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

    @jit.export
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
        # σ_max = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.weight, ord=2)
        fac = torch.minimum(self.C / self.spectral_norm, self.ONE)
        return functional.linear(x, fac * self.weight, self.bias)


@autojit
class AltLinearContraction(nn.Module):
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
    kernel: Tensor
        The weight matrix
    bias: Tensor or None
        The bias Tensor if present, else None.
    """

    # Constants
    input_size: Final[int]
    r"""CONST:  Number of inputs"""
    output_size: Final[int]
    r"""CONST: Number of outputs"""
    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""

    # Buffers
    C: Tensor
    r"""BUFFER: The regularization strength."""
    ONE: Tensor
    r"""BUFFER: Constant value of float(1.0)."""
    spectral_norm: Tensor
    r"""BUFFER: The largest singular value."""
    u: Tensor
    r"""BUFFER: The left singular vector."""
    v: Tensor
    r"""BUFFER: The right singular vector."""

    # Parameters
    kernel: Tensor
    r"""PARAM: the weight matrix"""
    bias: Optional[Tensor]
    r"""PARAM: The bias term"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        c: float = 0.97,
        bias: bool = True,
        maxiter: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.maxiter = maxiter

        self.kernel = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # self.spectral_norm = matrix_norm(self.weight, ord=2)
        self.register_buffer("ONE", torch.tensor(1.0))
        self.register_buffer("C", torch.tensor(float(c)))
        self.register_buffer("spectral_norm", matrix_norm(self.kernel, ord=2))
        # self.register_buffer(
        #     "u",
        # )
        # self.register_buffer(
        #     "v",
        # )

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.kernel, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    # def extra_repr(self) -> str:
    #     return "input_size={}, output_size={}, bias={}".format(
    #         self.input_size, self.output_size, self.bias is not None
    #     )

    @jit.export
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
        # σ_max = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.kernel, ord=2)
        fac = torch.minimum(self.C / self.spectral_norm, self.ONE)
        return functional.linear(x, fac * self.kernel, self.bias)


@autojit
class iResNetBlock(nn.Module):
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

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    hidden_size: Final[int]
    r"""CONST: The dimensionality of the latents."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""
    maxiter: Final[int]
    r"""CONST: The maximum number of steps in inverse pass."""
    atol: Final[float]
    r"""CONST: The absolute tolerance threshold value."""
    rtol: Final[float]
    r"""CONST: The relative tolerance threshold value."""

    # Buffers
    residual: Tensor

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
    r"""The hyperparameter dictionary"""

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
        self._Activation: type[Activation] = ACTIVATIONS[HP["activation"]]
        self.activation = self._Activation(**HP["activation_config"])  # type: ignore[call-arg]

        self.bottleneck = nn.Sequential(
            LinearContraction(self.input_size, self.hidden_size, self.bias),
            LinearContraction(self.hidden_size, self.input_size, self.bias),
            self.activation,
        )

    @jit.export
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

    @jit.export
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
        x = y.clone()
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x, x_prev = y - self.bottleneck(x), x
            residual = torch.abs(x - x_prev) - self.rtol * torch.absolute(x_prev)

            if torch.all(residual <= self.atol):
                return x

        print(
            f"No convergence in {self.maxiter} iterations. "
            f"Max residual:{torch.max(residual)} > {self.atol}."
        )
        return x


@autojit
class iResNet(nn.Module):
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

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

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
    r"""The hyperparameter dictionary"""

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

        self.blocks = nn.Sequential(*blocks)

    @jit.export
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

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fix point iteration in each block in reversed order.

        Parameters
        ----------
        y: Tensor

        Returns
        -------
        yhat: Tensor
        """
        for block in self.blocks[::-1]:  # traverse in reverse
            y = block.inverse(y)

        return y

    # TODO: delete this?
    # @jit.export
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
