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
            self.layer,
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

    y.grad = None

    torch._dynamo.config.compiled_autograd = True

    @torch.compile
    def train():
        y_star = fixpoint_solve(
            lambda z: z @ W0.mH,
            y,
        )
        loss = y_star.square().sum()
        loss.backward()

    try:
        train()
    except Exception as e:
        print("COMPILED AUTOGRAD FAILED")
        print(e)


if __name__ == "__main__":
    main()
