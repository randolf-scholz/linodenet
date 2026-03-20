import math

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet_special.fallbacks.fixpoint_iteration import (
    fixpoint_solve,
    fixpoint_solve_functional,
)
from tests.testing import DEVICES, TestCase


class ShiftedHalfMap(nn.Module):
    r"""Simple affine contraction $f(x) = ½x + b$ with learnable bias."""

    def __init__(self, bias: Tensor, /) -> None:
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return 0.5 * x + self.bias


class TestFixPointIteration(TestCase):
    VALUE_ATOL = 1e-6
    VALUE_RTOL = 1e-6

    def test_cosine_forward(self) -> None:
        r"""Solve the classical fixed point equation $x = \cos(x)$."""

        def fn(z: Tensor) -> Tensor:
            return torch.cos(z)

        x0 = torch.zeros((), dtype=torch.float64)
        x = fixpoint_solve_functional(fn, x0, maxiter=128, atol=1e-10, rtol=0.0)

        self.assert_close(
            x,
            torch.tensor(0.7390851332151607),
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
        )

    def test_cosine_backward(self) -> None:
        r"""Differentiate the parameterized fixed point near $x = \cos(x)$."""

        def fn(z: Tensor, c: Tensor) -> Tensor:
            return torch.cos(z) + c

        shift = torch.zeros((), requires_grad=True)
        x0 = torch.zeros((), dtype=torch.float64)
        x = fixpoint_solve_functional(fn, x0, shift, maxiter=128, atol=1e-10, rtol=0.0)
        loss = x.square()
        loss.backward()

        expected_x = torch.tensor(0.7390851332151607)
        expected_grad = 2 * expected_x.item() / (1 + math.sin(expected_x.item()))

        self.assert_close(x, expected_x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        assert shift.grad is not None
        self.assert_close(
            shift.grad,
            torch.tensor(expected_grad),
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
        )

    def test_fixpoint_solve_compile(self) -> None:
        r"""Check that `fixpoint_solve` works under `torch.compile`."""

        def fn(z: Tensor, c: Tensor) -> Tensor:
            return 0.25 * torch.cos(z) + c

        x0 = torch.zeros((), dtype=torch.float64)

        def solve(shift: Tensor) -> Tensor:
            return fixpoint_solve_functional(
                fn, x0, shift, maxiter=128, atol=1e-10, rtol=0.0
            )

        eager_shift = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        eager_x = solve(eager_shift)
        eager_x.square().backward()

        compiled_shift = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        compiled_solve = torch.compile(solve)
        compiled_x = compiled_solve(compiled_shift)
        compiled_x.square().backward()

        assert eager_shift.grad is not None
        assert compiled_shift.grad is not None
        self.assert_close(
            compiled_x,
            eager_x,
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
        )
        self.assert_close(
            compiled_shift.grad,
            eager_shift.grad,
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
        )

    def test_fixpoint_solve_module_parameter_gradient(self) -> None:
        r"""Check that module parameters used via `fn` receive gradients."""
        module = ShiftedHalfMap(torch.tensor([0.1, -0.2], dtype=torch.float64))
        x0 = torch.zeros(2, dtype=torch.float64)

        x_star = fixpoint_solve(module, x0, maxiter=128, atol=1e-12, rtol=0.0)
        loss = x_star.square().sum()
        loss.backward()

        grad_expected = 8.0 * module.bias.detach()

        assert module.bias.grad is not None
        self.assert_close(
            module.bias.grad,
            grad_expected,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_fixpoint_solve_functional_module_parameter_gradient_regression(
        self,
    ) -> None:
        r"""Check that `fixpoint_solve_functional` still drops module parameter grads."""
        module = ShiftedHalfMap(torch.tensor([0.1, -0.2], dtype=torch.float64))
        x0 = torch.zeros(2, dtype=torch.float64)

        x_star = fixpoint_solve_functional(
            module,
            x0,
            maxiter=128,
            atol=1e-12,
            rtol=0.0,
        )
        loss = x_star.square().sum()

        with pytest.raises(
            RuntimeError,
            match="does not require grad and does not have a grad_fn",
        ):
            loss.backward()

        assert module.bias.grad is None

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("eager", [True, False], ids=["eager", "compiled"])
    def test_fixpoint_solve_linear_contraction(self, device: str, eager) -> None:
        r"""Check a float32 linear contraction in eager and compiled modes."""
        torch.manual_seed(0)
        dim = 8
        weight = torch.randn(dim, dim, device=device, dtype=torch.float32)
        weight = 0.95 * weight / torch.linalg.matrix_norm(weight, ord=2)
        bias = torch.randn(dim, device=device, dtype=torch.float32)
        x0 = torch.zeros(dim, device=device, dtype=torch.float32)

        def fn(x: Tensor, A: Tensor, b: Tensor) -> Tensor:
            return F.linear(x, A) + b

        def solve(A: Tensor, b: Tensor) -> Tensor:
            return fixpoint_solve_functional(
                fn, x0, A, b, maxiter=256, atol=1e-5, rtol=1e-5
            )

        impl = solve if eager else torch.compile(solve)
        test_weight = weight.detach().clone().requires_grad_()
        test_bias = bias.detach().clone().requires_grad_()
        x = impl(test_weight, test_bias)
        loss = x.square().mean()
        loss.backward()

        expected_x = torch.linalg.solve(
            torch.eye(dim, device=device, dtype=torch.float32) - weight,
            bias,
        )

        assert test_weight.grad is not None
        assert test_bias.grad is not None
        self.assert_close(x, expected_x, atol=1e-4, rtol=1e-4)
        assert torch.isfinite(test_weight.grad).all()
        assert torch.isfinite(test_bias.grad).all()
