import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet_special.fallbacks.fixpoint_iteration import (
    fixpoint_solve,
    fixpoint_solve_functional,
)


def linear_fixed_point(x: Tensor, A: Tensor, b: Tensor, /) -> Tensor:
    r"""Linear contraction $f(x, A, b) = xAᵀ + b$."""
    return x @ A.mT + b


class LinearLayer(nn.Module):
    r"""Linear contraction $f(x, b) = xAᵀ + b$ with trainable matrix $A$."""

    def __init__(self, weight: Tensor, /) -> None:
        super().__init__()
        self.A = nn.Parameter(weight)

    def forward(self, x: Tensor, b: Tensor, /) -> Tensor:
        return linear_fixed_point(x, self.A, b)


class LinearFixpointModel(nn.Module):
    r"""Model using `fixpoint_solve` with internal weight and external bias."""

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        /,
        *,
        maxiter: int,
        atol: float,
        rtol: float,
    ) -> None:
        super().__init__()
        self.input_size = weight.shape[-1]
        self.layer = LinearLayer(weight)
        self.bias = nn.Parameter(bias)
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol

    def forward(self, y: Tensor, /) -> Tensor:
        x0 = y.clone()
        return fixpoint_solve(
            lambda z: self.layer(z),
            x0,
            args=(self.bias,),
            maxiter=self.maxiter,
            atol=self.atol,
            rtol=self.rtol,
        )


def main():
    r"""Test fixpoint solve."""
    y = torch.randn(32, 2, requires_grad=True)
    W0 = torch.tensor([[0.2, -0.1], [0.05, 0.15]])
    x0 = torch.zeros_like(y)
    b0 = torch.tensor([0.3, -0.2])

    model = LinearFixpointModel(
        W0,
        b0,
        maxiter=10,
        atol=1e-6,
        rtol=1e-6,
    )

    y_star = fixpoint_solve(
        lambda z: z @ W0.mH,
        y,
    )
    loss = y_star.square().sum()
    loss.backward()

    print("EAGER OK")
    print(f"{y.grad=}")

    shift = b0.clone().requires_grad_()

    def solve_compiled(c: Tensor) -> None:
        x_star = fixpoint_solve_functional(
            linear_fixed_point,
            x0,
            args=(W0, c),
            maxiter=10,
            atol=1e-6,
            rtol=1e-6,
        )
        loss = x_star.square().sum()
        loss.backward()

    compiled_solve = torch.compile(solve_compiled)
    compiled_solve(shift)

    print("COMPILED OK")
    print(f"{shift.grad=}")
    print(
        "NOTE: compiling `fixpoint_solve` itself, or enabling compiled autograd, "
        "still fails here with `NotImplementedError: Cannot access storage of TensorWrapper`."
    )


if __name__ == "__main__":
    main()
