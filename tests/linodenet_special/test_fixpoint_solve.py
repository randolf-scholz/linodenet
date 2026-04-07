import math
from collections.abc import Callable
from enum import StrEnum
from typing import Concatenate, NamedTuple

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from linodenet_special.fallbacks.fixpoint_iteration import (
    fixpoint_solve,
    fixpoint_solve_functional,
)
from tests.testing import DEVICES, TestSuite, pytest_xfail

# TODO: neither of these seem to do good things here...
# torch._dynamo.config.trace_autograd_ops = True
# torch._dynamo.config.compiled_autograd = True


def compile_fresh(fn, /):
    r"""Compile `fn` after clearing Dynamo state from earlier test cases."""
    torch._dynamo.reset()  # noqa: SLF001
    return torch.compile(fn)


class ShiftedHalfMap(nn.Module):
    r"""Simple affine contraction $f(x) = ½x + b$ with learnable bias."""

    def __init__(self, bias: Tensor, /) -> None:
        super().__init__()
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return 0.5 * x + self.bias


class LinearModule(nn.Module):
    r"""Linear contraction $f(x) = xAᵀ + b$ with trainable weight and bias."""

    def __init__(self, weight: Tensor, bias: Tensor, /) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class TestCase(NamedTuple):
    fn: Callable[Concatenate[Tensor, ...], Tensor]
    x: Tensor
    args: tuple[Tensor, ...]


class Mode(StrEnum):
    EAGER = "eager"
    COMPILE_FORWARD = "compile-forward"
    COMPILE_BACKWARD = "compile-backward"


class TestFixpoint(TestSuite):
    VALUE_ATOL = 1e-6
    VALUE_RTOL = 1e-6

    def select_check(self, mode: Mode, /):
        match mode:
            case Mode.EAGER:
                return self.check_eager
            case Mode.COMPILE_FORWARD:
                return self.check_compiled_forward
            case Mode.COMPILE_BACKWARD:
                return self.check_compiled_backward

    def assert_test_case_grads(self, case: TestCase, /) -> None:
        if case.args:
            assert case.args[0].grad is not None
            assert case.args[1].grad is not None
            return

        assert isinstance(case.fn, nn.Module)
        for parameter in case.fn.parameters():
            assert parameter.grad is not None

    def make_linear_functional(
        self,
        batch_size: int,
        input_size: int,
        /,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TestCase:
        y = torch.randn(batch_size, input_size, device=device, dtype=dtype)
        weight = torch.randn(input_size, input_size, device=device, dtype=dtype)
        weight = 0.95 * weight / torch.linalg.matrix_norm(weight, ord=2)
        weight = nn.Parameter(weight)
        bias = nn.Parameter(torch.randn(input_size, device=device, dtype=dtype))
        return TestCase(F.linear, y, (weight, bias))

    def make_linear_module(
        self,
        batch_size: int,
        input_size: int,
        /,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> TestCase:
        case = self.make_linear_functional(
            batch_size,
            input_size,
            device=device,
            dtype=dtype,
        )
        weight, bias = case.args
        module = LinearModule(weight.detach().clone(), bias.detach().clone())
        return TestCase(module, case.x, ())

    def check_eager(self, solver, case: TestCase, /) -> None:
        y_star = solver(case.fn, case.x, *case.args)
        loss = y_star.square().sum()
        loss.backward()
        self.assert_test_case_grads(case)

    def check_compiled_forward(self, solver, case: TestCase, /) -> None:
        torch._dynamo.reset()  # noqa: SLF001

        @torch.compile
        def forward(y0: Tensor) -> Tensor:
            y_star = solver(case.fn, y0, *case.args)
            return y_star.square().sum()

        loss = forward(case.x)
        loss.backward()
        self.assert_test_case_grads(case)

    def check_compiled_backward(self, solver, case: TestCase, /) -> None:
        torch._dynamo.reset()  # noqa: SLF001

        @torch.compile
        def backward(y0: Tensor) -> None:
            y_star = solver(case.fn, y0, *case.args)
            loss = y_star.square().sum()
            loss.backward()

        backward(case.x)
        self.assert_test_case_grads(case)


class TestFixpointSolveFunctional(TestFixpoint):
    SEED = 0

    @pytest.mark.parametrize("mode", list(Mode), ids=str)
    @pytest.mark.parametrize("module", [False, True], ids=["functional", "module"])
    def test_linear(self, mode: Mode, module: bool) -> None:
        case = (
            self.make_linear_module(5, 3)
            if module
            else self.make_linear_functional(5, 3)
        )

        def solver(fn, x, /, *args):
            return fixpoint_solve_functional(
                fn,
                x,
                args=args,
                maxiter=128,
                atol=1e-6,
                rtol=1e-6,
            )

        check = self.select_check(mode)
        with pytest_xfail(condition=module):
            check(solver, case)

    @pytest.mark.parametrize("eager", [True, False], ids=["eager", "compiled"])
    def test_gradcheck(self, eager: bool) -> None:
        r"""Check `fixpoint_solve_functional` gradients for a linear map."""
        torch.manual_seed(self.SEED)
        case = self.make_linear_functional(5, 3, dtype=torch.float64)
        x = case.x.detach().requires_grad_()
        weight, bias = (arg.detach().requires_grad_() for arg in case.args)

        def func(y: Tensor, A: Tensor, b: Tensor) -> Tensor:
            return fixpoint_solve_functional(
                case.fn,
                y,
                args=(A, b),
                maxiter=128,
                atol=1e-10,
                rtol=1e-10,
            )

        impl = func if eager else compile_fresh(func)

        assert torch.autograd.gradcheck(
            impl,
            (x, weight, bias),
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
        )

    def test_cosine_forward_and_backward(self) -> None:
        r"""Check value and gradient for the fixed point near $x = \cos(x)$."""

        def fn(z: Tensor, c: Tensor) -> Tensor:
            return torch.cos(z) + c

        shift = torch.zeros((), requires_grad=True)
        x0 = torch.zeros((), dtype=torch.float64)
        x = fixpoint_solve_functional(
            fn, x0, args=(shift,), maxiter=128, atol=1e-10, rtol=0.0
        )
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
                fn, x0, args=(shift,), maxiter=128, atol=1e-10, rtol=0.0
            )

        eager_shift = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        eager_x = solve(eager_shift)
        eager_x.square().backward()

        compiled_shift = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        compiled_solve = compile_fresh(solve)
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
    def test_linear_closed_form(self, device: str, eager: bool) -> None:
        r"""Check `fixpoint_solve_functional` against the exact linear fixed point."""
        torch.manual_seed(self.SEED)
        case = self.make_linear_functional(5, 3, device=device, dtype=torch.float64)
        x = case.x.detach()
        weight, bias = (arg.detach().requires_grad_() for arg in case.args)

        def solve(y: Tensor, A: Tensor, b: Tensor) -> Tensor:
            return fixpoint_solve_functional(
                case.fn,
                y,
                args=(A, b),
                maxiter=128,
                atol=1e-10,
                rtol=1e-10,
            )

        impl = solve if eager else compile_fresh(solve)
        x_star = impl(x, weight, bias)
        loss = x_star.square().sum()
        loss.backward()

        assert weight.grad is not None
        assert bias.grad is not None

        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        eye = torch.eye(reference_weight.shape[-1], dtype=x.dtype, device=x.device)
        reference_x = torch.linalg.solve(eye - reference_weight, reference_bias)
        reference_x = reference_x.expand_as(x_star)
        reference_loss = reference_x.square().sum()
        reference_loss.backward()

        assert reference_weight.grad is not None
        assert reference_bias.grad is not None
        self.assert_close(x_star, reference_x.detach(), atol=1e-10, rtol=1e-10)
        self.assert_close(weight.grad, reference_weight.grad, atol=1e-9, rtol=1e-9)
        self.assert_close(bias.grad, reference_bias.grad, atol=1e-9, rtol=1e-9)


class TestFixpointSolve(TestFixpoint):
    SEED = 0

    @pytest.mark.parametrize("mode", list(Mode), ids=str)
    @pytest.mark.parametrize("module", [False, True], ids=["functional", "module"])
    def test_linear(self, mode: Mode, module: bool) -> None:
        case = (
            self.make_linear_module(5, 3)
            if module
            else self.make_linear_functional(5, 3)
        )

        def solver(fn, x, /, *args):
            return fixpoint_solve(
                fn,
                x,
                args=args,
                maxiter=128,
                atol=1e-6,
                rtol=1e-6,
            )

        check = self.select_check(mode)
        check(solver, case)

    @pytest.mark.parametrize("eager", [True, False], ids=["eager", "compiled"])
    @pytest.mark.parametrize("module", [False, True], ids=["functional", "module"])
    def test_gradcheck(self, eager: bool, module: bool) -> None:
        r"""Check `fixpoint_solve` gradients for a linear map."""
        torch.manual_seed(self.SEED)
        case = (
            self.make_linear_module(5, 3, dtype=torch.float64)
            if module
            else self.make_linear_functional(5, 3, dtype=torch.float64)
        )
        x = case.x.detach().requires_grad_()
        args = tuple(arg.detach().requires_grad_() for arg in case.args)

        def func(y: Tensor, /, *parameters: Tensor) -> Tensor:
            return fixpoint_solve(
                case.fn,
                y,
                args=parameters,
                maxiter=128,
                atol=1e-10,
                rtol=1e-10,
            )

        impl = func if eager else compile_fresh(func)

        assert torch.autograd.gradcheck(
            impl,
            (x, *args),
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
            fast_mode=True,
        )

    @pytest.mark.parametrize("module", [False, True], ids=["functional", "module"])
    def test_linear_closed_form(self, module: bool) -> None:
        r"""Check `fixpoint_solve` against the exact linear fixed point."""
        torch.manual_seed(self.SEED)
        case = (
            self.make_linear_module(5, 3, dtype=torch.float64)
            if module
            else self.make_linear_functional(5, 3, dtype=torch.float64)
        )
        x = case.x.detach().requires_grad_()
        x_star = fixpoint_solve(
            case.fn,
            x,
            args=case.args,
            maxiter=128,
            atol=1e-10,
            rtol=1e-10,
        )
        loss = x_star.square().sum()
        loss.backward()

        if module:
            assert isinstance(case.fn, LinearModule)
            weight = case.fn.weight
            bias = case.fn.bias
        else:
            weight, bias = case.args

        assert weight.grad is not None
        assert bias.grad is not None
        assert x.grad is None

        reference_weight = weight.detach().clone().requires_grad_()
        reference_bias = bias.detach().clone().requires_grad_()
        eye = torch.eye(reference_weight.shape[-1], dtype=x.dtype, device=x.device)
        reference_x = torch.linalg.solve(eye - reference_weight, reference_bias)
        reference_x = reference_x.expand_as(x_star)
        reference_loss = reference_x.square().sum()
        reference_loss.backward()

        assert reference_weight.grad is not None
        assert reference_bias.grad is not None
        self.assert_close(x_star, reference_x.detach(), atol=1e-10, rtol=1e-10)
        self.assert_close(weight.grad, reference_weight.grad, atol=1e-9, rtol=1e-9)
        self.assert_close(bias.grad, reference_bias.grad, atol=1e-9, rtol=1e-9)

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

    def test_fixpoint_solve_constant_map_without_grad_dependencies(self) -> None:
        r"""Check a constant map fixed point with no differentiable dependencies."""

        def fn(x: Tensor, /) -> Tensor:
            return torch.full_like(x, 0.25)

        x0 = torch.zeros(3, dtype=torch.float64)
        x_star = fixpoint_solve(fn, x0, maxiter=32, atol=1e-12, rtol=0.0)

        self.assert_close(
            x_star,
            torch.full_like(x0, 0.25),
            atol=1e-12,
            rtol=0.0,
        )
        assert not x_star.requires_grad
