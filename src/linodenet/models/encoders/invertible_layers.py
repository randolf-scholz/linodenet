"""Implementation of invertible layers.

Layers:
- Affine: $y = Ax+b$ and $x = A⁻¹(y-b)$
    - A diagonal
    - A triangular
    - A tridiagonal
- Element-wise:
    - Monotonic
- Shears (coupling flows): $y_A = f(x_A, x_B)$ and $y_B = x_B$
    - Example: $y_A = x_A + e^{x_B}$ and $y_B = x_B$
- Residual: $y = x + F(x)$
    - Contractive: $y = F(x)$ with $‖F‖<1$
    - Low Rank Perturbation: $y = x + ABx$
- Continuous Time Flows: $ẋ=f(t, x)$
"""

from math import sqrt
from typing import Final, Optional

import torch
from torch import Tensor, jit, nn
from torch.linalg import matrix_norm
from torch.nn import functional


class LinearContraction(nn.Module):
    r"""A linear layer $f(x) = A⋅x$ satisfying the contraction property $‖f(x)-f(y)‖_2 ≤ ‖x-y‖_2$.

    This is achieved by normalizing the weight matrix by $A' = A/(σ+c)$, where
    $σ=‖A‖₂$ and $c>0$ is a hyperparameter.

    Attributes
    ----------
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

    # Constants
    input_size: Final[int]
    output_size: Final[int]
    c: Final[float]
    r"""CONST: The regularization hyperparameter."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    # Buffers
    sigma_cached: Tensor
    r"""BUFFER: Cached value of $‖W‖_2$"""
    recompute_sigma: Tensor
    r"""BUFFER: A boolean tensor indicating whether to recompute $‖W‖_2$"""
    u: Tensor
    r"""BUFFER: Cached left singular vector of $W$."""
    v: Tensor
    r"""BUFFER: Cached right singular vector of $W$."""

    # Parameters
    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Optional[Tensor]
    r"""PARAM: The bias term."""

    def __init__(
        self, input_size: int, output_size: int, *, c: float = 0.05, bias: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(Tensor(output_size, input_size))
        self.c = c

        if bias:
            self.bias = nn.Parameter(Tensor(output_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.register_buffer("one", torch.tensor(1.0), persistent=True)
        # self.register_buffer("c", torch.tensor(float(c)), persistent=True)
        self.register_buffer(
            "spectral_norm", matrix_norm(self.weight, ord=2), persistent=False
        )
        self.register_full_backward_pre_hook(self.recompute_sigma)

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def recompute_sigma(self, grad_output: list[Tensor]) -> None:
        self.recompute_sigma = torch.tensor(True)

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        # σ_max, _ = torch.lobpcg(self.weight.T @ self.weight, largest=True)
        # σ_max = torch.linalg.norm(self.weight, ord=2)
        # self.spectral_norm = spectral_norm(self.weight)
        # σ_max = torch.linalg.svdvals(self.weight)[0]

        if self.recompute_sigma:
            self.sigma_cached = matrix_norm(self.weight, ord=2)
            self.recompute_sigma = torch.tensor(False)

        fac = 1.0 / (self.c + self.sigma_cached)
        return functional.linear(x, fac * self.weight, self.bias)
