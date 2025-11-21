r"""Implementation of invertible ResNets."""

__all__ = [
    # Classes
    "LinearContraction",
    "SpectralNorm",
    "iResNet",
    "iResNetBlock",
    "iResNetLayer",
    "AltLinearContraction",
]

import warnings
from math import sqrt
from typing import Any, Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.linalg import matrix_norm, vector_norm
from torch.nn import functional

from linodenet.activations import MODULAR_ACTIVATIONS, Activation
from linodenet.layers import ReZero
from linodenet.torch_generics import ModuleSequence
from linodenet.utils import deep_dict_update


class SpectralNorm(torch.autograd.Function):
    r"""$‖A‖_2=λ_\max(A^⊤A)$.

    The spectral norm $∥A∥_2 ≔ \sup_x ∥Ax∥_2 / ∥x∥_2$ can be shown to be equal to
    $σ_{\max}(A) = \sqrt{λ_{\max} (A^⊤A)}$, the largest singular value of $A$.

    It can be computed efficiently via Power iteration.

    One can show that the derivative is equal to:

    .. math::  \pdv{½∥A∥_2}{A} = uv^⊤

    where $u,v$ are the left/right-singular vector corresponding to $σ_\max$

    References:
        `Spectral Normalization for Generative Adversarial Networks <https://openreview.net/forum?id=B1QRgziT->`_
        Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
        `International Conference on Learning Representations 2018 <https://iclr.cc/Conferences/2018>`_
    """

    @staticmethod
    def forward(ctx: Any, *tensors: Tensor, **kwargs: Any) -> Tensor:
        r""".. Signature:: ``(m, n) -> 1``."""
        A = tensors[0]
        if A.ndim != 2:
            raise ValueError(f"Expected 2d input, got {A.shape}.")

        atol: float = kwargs.get("atol", 1e-6)
        rtol: float = kwargs.get("rtol", 1e-6)
        maxiter: int = kwargs.get("maxiter", 1000)
        # initialize u and v, median should be useful guess.
        u = u_next = A.median(dim=1).values
        v = v_next = A.median(dim=0).values
        sigma: Tensor = torch.einsum("ij, i, j ->", A, u, v)

        for _ in range(maxiter):
            u = u_next / torch.norm(u_next)
            v = v_next / torch.norm(v_next)
            # choose optimal σ given u and v: σ = argmin ‖A - σuvᵀ‖²
            sigma = torch.einsum("ij, i, j ->", A, u, v)  # u.T @ A @ v
            # Residual: if Av = σu and Aᵀu = σv
            u_next = A @ v
            v_next = A.T @ u
            sigma_u = sigma * u
            sigma_v = sigma * v
            ru = u_next - sigma * u
            rv = v_next - sigma * v
            if (
                vector_norm(ru) <= rtol * vector_norm(sigma_u) + atol
                and vector_norm(rv) <= rtol * vector_norm(sigma_v) + atol
            ):
                break

        ctx.save_for_backward(u, v)
        return sigma

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor:
        u, v = ctx.saved_tensors
        return torch.einsum("..., i, j -> ...ij", grad_outputs[0], u, v)

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        r"""Jacobian-vector product forward mode."""
        u, v = ctx.saved_tensors
        return torch.einsum("...ij, i, j -> ...", grad_inputs[0], u, v)


class LinearContraction(nn.Module):
    r"""A linear layer $f(x) = A⋅x$ satisfying the contraction property $‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2$.

    This is achieved by normalizing the weight matrix by
    $A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)$, where $c<1$ is a hyperparameter.

    Attributes:
        input_size:  int
            The dimensionality of the input space.
        output_size: int
            The dimensionality of the output space.
        c: Tensor
            The regularization hyperparameter.
        spectral_norm: Tensor
            BUFFER: The value of `‖W‖_2`
        weight: Tensor
            The weight matrix.
        bias: Tensor or None
            The bias Tensor if present, else None.
    """

    input_size: Final[int]
    output_size: Final[int]

    # Constants
    c: Tensor
    r"""CONST: The regularization hyperparameter."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    # Buffers
    spectral_norm: Tensor
    r"""BUFFER: The value of $‖W‖_2$"""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Optional[Tensor]
    r"""PARAM: The bias term."""

    def __init__(
        self, input_size: int, output_size: int, *, c: float = 0.97, bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(Tensor(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.register_buffer("one", torch.tensor(1.0), persistent=True)
        self.register_buffer("c", torch.tensor(float(c)), persistent=True)
        self.register_buffer(
            "spectral_norm", matrix_norm(self.weight, ord=2), persistent=False
        )

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
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        # self.spectral_norm = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.weight, ord=2)
        gamma = torch.minimum(self.c / self.spectral_norm, self.one)
        return functional.linear(x, gamma * self.weight, self.bias)


class AltLinearContraction(nn.Module):
    r"""A linear layer `f(x) = A⋅x` satisfying the contraction property `‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2`.

    This is achieved by normalizing the weight matrix by
    `A' = A⋅\min(\tfrac{c}{‖A‖_2}, 1)`, where `c<1` is a hyperparameter.

    Attributes:
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
    c: Tensor
    r"""BUFFER: The regularization strength."""
    one: Tensor
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
        *,
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
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("c", torch.tensor(float(c)))
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
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        # σ_max = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]
        self.spectral_norm = matrix_norm(self.kernel, ord=2)
        fac = torch.minimum(self.c / self.spectral_norm, self.one)
        return functional.linear(x, fac * self.kernel, self.bias)


class iResNetLayer(nn.Module):
    r"""A single layer of an iResNet."""

    maxiter: Final[int]
    r"""CONST: Maximum number of steps in power-iteration"""
    atol: Final[float]
    r"""CONST: Absolute tolerance for fixed point iteration"""
    rtol: Final[float]
    r"""CONST: Relative tolerance for fixed point iteration"""
    converged: Tensor
    r"""BUFFER: Boolean tensor indicating convergence"""

    def __init__(
        self,
        layer: nn.Module,
        *,
        maxiter: int = 1000,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.register_buffer("converged", torch.tensor(False))

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        return x + self.layer(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $‖x'-x‖≤\text{rtol}⋅‖x‖+\text{atol}$ is reached.
        """
        x = y.clone()
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x_prev = x
            x = y - self.layer(x)
            residual = (x - x_prev).norm()
            self.converged = residual < self.atol + self.rtol * x_prev.norm()
            if self.converged:
                break
        if not self.converged:
            warnings.warn(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual} > {self.atol}.",
                stacklevel=2,
            )
        return x


class iResNetBlock(nn.Module):
    r"""Invertible ResNet-Block of the form $g(x)=ϕ(W_1⋅W_2⋅x)$.

    By default, $W_1⋅W_2$ is a low rank factorization.

    Alternative: $g(x) = W_3ϕ(W_2ϕ(W_1⋅x))$.

    All linear layers must be `LinearContraction` layers.
    The activation function must have Lipschitz constant $≤1$ such as `~torch.nn.ReLU`,
    `~torch.nn.ELU` or `~torch.nn.Tanh`)

    Attributes:
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
        residual: Tensor
            BUFFER: The termination error during backward propagation.
        bottleneck: nn.Sequential
            The bottleneck layer.
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
    use_rezero: Final[bool]
    r"""CONST: Whether to apply ReZero technique."""

    # Buffers
    residual: Tensor
    r"""BUFFER: The termination error during backward propagation."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "atol": 1e-08,
        "rtol": 1e-05,
        "maxiter": 10,
        "activation": "ReLU",
        "activation_config": {"inplace": False},
        "bias": True,
        "rezero": False,
        "output_size": None,
        "hidden_size": None,
        "input_size": None,
    }
    r"""The hyperparameter dictionary"""

    def __init__(self, input_size: int, **HP: Any):
        super().__init__()
        self.HP = HP = deep_dict_update(self.HP, HP)

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
        self._Activation: type[Activation] = MODULAR_ACTIVATIONS[HP["activation"]]
        self.activation = self._Activation(**HP["activation_config"])

        layers: list[nn.Module] = [
            LinearContraction(self.input_size, self.hidden_size, bias=self.bias),
            LinearContraction(self.hidden_size, self.input_size, bias=self.bias),
        ]

        self.use_rezero = HP["rezero"]
        self.rezero = ReZero() if self.use_rezero else None
        if self.use_rezero:
            assert self.rezero is not None
            layers.append(self.rezero)

        self.bottleneck = nn.Sequential(*layers)
        self.register_buffer("residual", torch.tensor(()), persistent=False)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        return x + self.bottleneck(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        x = y.clone()
        residual = torch.zeros_like(y)

        for _ in range(self.maxiter):
            x, x_prev = y - self.bottleneck(x), x
            self.residual = torch.abs(x - x_prev) - self.rtol * torch.absolute(x_prev)

            if torch.all(self.residual <= self.atol):
                return x

        warnings.warn(
            f"No convergence in {self.maxiter} iterations. "
            f"Max residual:{torch.max(residual)} > {self.atol}.",
            stacklevel=2,
        )
        return x


class iResNet(nn.Module):
    r"""Invertible ResNet consists of a stack of `iResNetBlock` modules.

    Attributes:
        input_size: int
            The dimensionality of the input space.
        output_size: int
            The dimensionality of the output space.
        blocks:  nn.Sequential
            Sequential model consisting of the iResNetBlocks
        HP: dict
            Nested dictionary containing the hyperparameters.

    References:
        Invertible Residual Networks
        Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
        International Conference on Machine Learning 2019
        http://proceedings.mlr.press/v97/behrmann19a.html
    """

    # Constants
    input_size: Final[int]
    r"""CONST: The dimensionality of the inputs."""
    output_size: Final[int]
    r"""CONST: The dimensionality of the outputs."""

    HP = {
        "__name__": __qualname__,
        "__module__": __name__,
        "maxiter": 10,
        "input_size": None,
        "dropout": None,
        "bias": True,
        "nblocks": 5,
        "rezero": False,
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
        self.HP = HP = deep_dict_update(self.HP, HP)

        HP["input_size"] = input_size

        self.input_size = input_size
        self.output_size = input_size
        HP["iResNetBlock"]["input_size"] = self.input_size
        HP["iResNetBlock"]["rezero"] = HP["rezero"]

        self.nblocks = HP["nblocks"]
        self.maxiter = HP["maxiter"]
        self.bias = HP["bias"]

        blocks = []

        for _ in range(self.nblocks):
            blocks += [iResNetBlock(**HP["iResNetBlock"])]

        self.blocks = ModuleSequence(blocks)

    @jit.export
    def encode(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        return self.blocks(x)

    @jit.export
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fix point iteration in each block in reversed order."""
        for block in self.blocks[::-1]:  # traverse in reverse
            y = block.decode(y)
        return y
