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


class LinearLayer(nn.Module):
    r"""Linear contraction $f(x, b) = xAᵀ + b$ with trainable matrix $A$."""

    def __init__(self, weight: Tensor, /) -> None:
        super().__init__()
        self.A = nn.Parameter(weight)

    def forward(self, x: Tensor, b: Tensor, /) -> Tensor:
        return x @ self.A.mT + b


class LinearTensorLayer(nn.Module):
    r"""Linear contraction with tensor-valued internal weight for gradcheck."""

    def __init__(self, weight: Tensor, /) -> None:
        super().__init__()
        self.A = weight

    def forward(self, x: Tensor, b: Tensor, /) -> Tensor:
        return x @ self.A.mT + b


class LinearFixpointModel(nn.Module):
    r"""Model using `fixpoint_solve` with internal weight and external bias."""

    def __init__(self, weight: Tensor, bias: Tensor, /) -> None:
        super().__init__()
        self.input_size = weight.shape[-1]
        self.layer = LinearLayer(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, y: Tensor, /) -> Tensor:
        x0 = y.clone()
        return fixpoint_solve(
            self.layer, x0, self.bias, maxiter=256, atol=1e-12, rtol=0.0
        )


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


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("eager", [True, False], ids=["eager", "compiled"])
class TestCorrectness(TestCase):
    W0 = [[0.2, -0.1], [0.05, 0.15]]
    b0 = [0.3, -0.2]

    def test_linear_module_and_input_gradients_match_closed_form(
        self, eager: bool, device: str
    ) -> None:
        r"""Check gradients for internal $A$ and input $b$ against the exact solve."""
        weight = torch.tensor(self.W0, dtype=torch.float64, device=device)
        bias = torch.tensor(self.b0, dtype=torch.float64, device=device)
        y = torch.tensor([0.7, -0.4], dtype=torch.float64, device=device)

        model = LinearFixpointModel(weight, bias)
        impl = model if eager else torch.compile(model)
        x = impl(y)
        loss = x.square().sum()
        loss.backward()

        assert model.layer.A.grad is not None
        assert model.bias.grad is not None

        reference_weight = weight.clone().requires_grad_()
        reference_bias = bias.clone().requires_grad_()
        eye = torch.eye(model.input_size, dtype=torch.float64, device=device)
        reference_x = torch.linalg.solve(eye - reference_weight, reference_bias)
        reference_loss = reference_x.square().sum()
        reference_loss.backward()

        assert reference_weight.grad is not None
        assert reference_bias.grad is not None
        self.assert_close(x, reference_x.detach(), atol=1e-10, rtol=1e-10)
        self.assert_close(
            model.layer.A.grad,
            reference_weight.grad,
            atol=1e-10,
            rtol=1e-10,
        )
        self.assert_close(
            model.bias.grad,
            reference_bias.grad,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gradcheck_linear_module_and_input_gradients(
        self, eager: bool, device: str
    ) -> None:
        r"""Check `fixpoint_solve` gradients for internal $A$ and input $b$."""
        weight = torch.tensor(
            self.W0,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        bias = torch.tensor(
            self.b0,
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(A: Tensor, b: Tensor) -> Tensor:
            layer = LinearTensorLayer(A)
            x0 = torch.zeros_like(b)
            return fixpoint_solve(layer, x0, b, maxiter=256, atol=1e-12, rtol=0.0)

        impl = func if eager else torch.compile(func)
        assert torch.autograd.gradcheck(
            impl,
            (weight, bias),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )
