r"""Implementation of invertible ResNets."""

__all__ = [
    # Classes
    "iResNet",
    "iResNetBlock",
    "iResNetLayer",
]

import warnings
from math import sqrt
from typing import Any, Final

import torch
from torch import Tensor, nn

from linodenet.mappings.linear import LinearContraction
from linodenet.nn import ModuleSequence, ReZero
from linodenet.nn.activations import get_activation
from linodenet.utils import deep_dict_update
from signatures import signature


class iResNetLayer(nn.Module):
    r"""A single layer of an iResNet.

    References:
    - | Invertible Residual Networks
      | Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen
      | International Conference on Machine Learning 2019
      | http://proceedings.mlr.press/v97/behrmann19a.html
    - https://github.com/jhjacobsen/invertible-resnet
    """

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

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor) -> Tensor:
        return x + self.layer(x)

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

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int | None = None,
        output_size: int | None = None,
        atol: float = 1e-08,
        rtol: float = 1e-05,
        maxiter: int = 10,
        activation: str | nn.Module = "ReLU",
        bias: bool = True,
        rezero: bool = False,
    ) -> None:
        super().__init__()
        cfg = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "atol": atol,
            "rtol": rtol,
            "maxiter": maxiter,
            "activation": activation,
            "activation_config": {"inplace": False},
            "bias": bias,
            "rezero": rezero,
        }
        self.HP = HP = deep_dict_update(self.HP, cfg)

        HP["input_size"] = input_size
        HP["hidden_size"] = HP["hidden_size"] or int(sqrt(input_size))

        self.input_size = input_size
        self.output_size = HP["input_size"]
        self.hidden_size = HP["hidden_size"]

        self.atol = HP["atol"]
        self.rtol = HP["rtol"]
        self.maxiter = HP["maxiter"]
        self.bias = HP["bias"]
        self.activation = get_activation(HP["activation"], **HP["activation_config"])

        layers: list[nn.Module] = [
            LinearContraction(self.input_size, self.hidden_size, bias=self.bias),
            LinearContraction(self.hidden_size, self.input_size, bias=self.bias),
        ]

        self.use_rezero = HP["rezero"]
        self.rezero = ReZero(nn.Identity()) if self.use_rezero else None
        if self.use_rezero:
            assert self.rezero is not None
            layers.append(self.rezero)

        self.bottleneck = nn.Sequential(*layers)
        self.register_buffer("residual", torch.tensor(()), persistent=False)

    @signature("(..., n) -> (..., n)")
    def encode(self, x: Tensor) -> Tensor:
        return x + self.bottleneck(x)

    @signature("(..., n) -> (..., n)")
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
        blocks: nn.Sequential
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

    @signature("(..., *xs) -> (..., *xs)")
    def encode(self, x: Tensor) -> Tensor:
        return self.blocks(x)

    @signature("(..., *xs) -> (..., *xs)")
    def decode(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fix point iteration in each block in reversed order."""
        for block in self.blocks[::-1]:  # traverse in reverse
            y = block.decode(y)
        return y
