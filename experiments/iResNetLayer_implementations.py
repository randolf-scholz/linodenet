#!/usr/bin/env python

from math import sqrt
from typing import Final, Optional

import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch import Tensor, jit, nn
from torch.linalg import matrix_norm
from torch.nn import functional

from linodenet.utils import timer


class iResNetLayer(nn.Module):
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
    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer(x)

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        x = y.clone()
        residual = torch.zeros_like(y)

        for it in range(self.maxiter):
            x_prev = x
            x = y - self.layer(x)
            residual = (x - x_prev).norm()
            self.converged = residual < self.atol + self.rtol * x_prev.norm()
            if self.converged:
                # print(f"Converged in {it} iterations.")
                break
        if not self.converged:
            print(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual} > {self.atol}."
            )
        return x


class OptimizediResNetLayer(nn.Module):
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
        m: int,
        *,
        maxiter: int = 1000,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer = nn.Linear(m, m, bias=bias)
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.register_buffer("converged", torch.tensor(False))

    @jit.export
    def forward(self, x: Tensor) -> Tensor:
        sigma = matrix_norm(self.layer.weight, ord=2)
        cached_weight = self.layer.weight / sigma
        return x + functional.linear(x, cached_weight, self.layer.bias)

    @jit.export
    def inverse(self, y: Tensor) -> Tensor:
        r"""Compute the inverse through fixed point iteration.

        Terminates once ``maxiter`` or tolerance threshold
        $|x'-x|≤\text{atol} + \text{rtol}⋅|x|$ is reached.
        """
        x = y.clone()
        residual = torch.zeros_like(y)
        sigma = matrix_norm(self.layer.weight, ord=2)
        cached_weight = self.layer.weight / sigma

        for it in range(self.maxiter):
            x_prev = x
            x = y - functional.linear(x, cached_weight, self.layer.bias)
            residual = (x - x_prev).norm()
            self.converged = residual < self.atol + self.rtol * x_prev.norm()
            if self.converged:
                # print(f"Converged in {it} iterations.")
                break
        if not self.converged:
            print(
                f"No convergence in {self.maxiter} iterations. "
                f"Max residual:{residual} > {self.atol}."
            )
        return x


class LinearContraction(nn.Module):
    # Constants
    input_size: Final[int]
    output_size: Final[int]
    c: Final[float]
    r"""CONST: The maximal Lipschitz constant."""
    one: Tensor
    r"""CONST: A tensor with value 1.0"""

    # Buffers
    cached_sigma: Tensor
    r"""BUFFER: Cached value of $‖W‖_2$"""
    cached_weight: Tensor
    r"""BUFFER: Cached value of $W$/‖W‖₂."""
    refresh_cache: Tensor
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
        self, input_size: int, output_size: int, *, c: float = 1.0, bias: bool = False
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
        self.register_buffer("cached_sigma", torch.tensor(1.0))
        self.register_buffer("cached_weight", self.weight.clone())
        self.register_buffer("refresh_cache", torch.tensor(True))

        self.register_forward_pre_hook(self.__renormalize_weight)
        self.register_full_backward_pre_hook(self.raise_flag)

    def reset_parameters(self) -> None:
        r"""Reset both weight matrix and bias vector."""
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            bound = 1 / sqrt(self.input_size)
            nn.init.uniform_(self.bias, -bound, bound)

    # @jit.export
    def forward(self, x: Tensor) -> Tensor:
        r""".. Signature:: ``(..., n) -> (..., n)``."""
        return functional.linear(x, self.cached_weight, self.bias)

    # @jit.export
    def renormalize_weight(self, inputs: tuple[Tensor] = (torch.tensor([]),)) -> None:
        """Renormalizes weight so that ‖W‖₂ ≤ 1-δ."""
        if self.refresh_cache:
            self.refresh_cache = torch.tensor(False)
            self.cached_sigma = matrix_norm(self.weight, ord=2)
            gamma = 1 / self.cached_sigma
            self.cached_weight = gamma * self.weight
            # self.weight[:] = gamma * self.weight

    @staticmethod
    def __renormalize_weight(self, inputs: tuple[Tensor]) -> None:
        return self.renormalize_weight()

    @staticmethod
    def raise_flag(self, grad_output: list[Tensor]) -> None:
        # pass  # WTF! just pass will throw an error?!
        # print("here!")
        self.refresh_cache = torch.tensor(True)

    # with torch.no_grad():
    #      self.refresh_cache = torch.tensor(True)

    #     @staticmethod
    #     def __raise_refresh_cache_flag(self, grad_output: list[Tensor] = ()) -> None:
    #         self.raise_refresh_cache_flag()

    #     # @jit.export
    #     def raise_refresh_cache_flag(self) -> None:
    #         # print(grad_output)
    #         # self.refresh_cache = torch.tensor(True)
    #         # self.cached_weight = torch.tensor([])
    #         # self.cached_sigma = torch.tensor([])
    #         pass

    # fac = 1.0 / (self.c + self.sigma_cached)
    # return functional.linear(x, fac * self.weight, self.bias)


class NaiveContraction(nn.Module):
    # Parameters
    weight: Tensor
    r"""PARAM: The weight matrix."""
    bias: Optional[Tensor]
    r"""PARAM: The bias term."""

    def __init__(
        self, input_size: int, output_size: int, *, c: float = 1.0, bias: bool = False
    ):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size, bias=bias)
        self.weight = self.layer.weight
        self.bias = self.layer.bias

    def forward(self, x):
        sigma = matrix_norm(self.weight, ord=2)
        return functional.linear(x, self.weight / sigma, self.bias)
        # return self.layer(x / sigma)


def test_implementation(model: nn.Module, x: Tensor, xi: Tensor) -> Tensor:
    # test naive contraction
    with timer() as t:
        model.zero_grad(set_to_none=True)
        r = torch.tensor(0.0)
        for y in Y:
            x = model.inverse(y)
            r += (x * xi).sum()
        r.backward()
    print(f"Finished in {t.elapsed:.3f} s")
    return model.layer.weight.grad.clone().detach().flatten()


if __name__ == "__main__":
    # main program
    T, N, m, n = 64, 128, 256, 128
    A0 = torch.randn(m, m)
    X = torch.randn(T, N, m)
    Y = torch.randn(T, N, m)
    xi = torch.randn(N, m)

    models = {
        "naive": jit.script(iResNetLayer(NaiveContraction(m, m))),
        "cached": jit.script(iResNetLayer(LinearContraction(m, m))),
        "optimized": jit.script(OptimizediResNetLayer(m)),
    }

    grads = {}
    for name, model in models.items():
        print(f"Testing {name}")
        model.layer.weight.data = A0.clone()
        grads[name] = test_implementation(model, X, xi).numpy()

    G = np.array(list(grads.values()))
    print(distance_matrix(G, G))
