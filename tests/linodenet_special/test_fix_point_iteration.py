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
from tests.testing import DEVICES, DTYPES, TestSuite, pytest_xfail


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


def linear_fixed_point(x: Tensor, A: Tensor, b: Tensor, /) -> Tensor:
    r"""Linear contraction $f(x, A, b) = xAᵀ + b$."""
    return x @ A.mT + b


class LinearLayer(nn.Module):
    r"""Linear contraction $f(x, b) = xAᵀ + b$ with trainable matrix $A$."""

    def __init__(self, weight: Tensor, /) -> None:
        super().__init__()
        self.A = nn.Parameter(weight)

    def forward(self, x: Tensor, b: Tensor, /) -> Tensor:
        return x @ self.A.mT + b


class LinearModule(nn.Module):
    r"""Linear contraction $f(x) = xAᵀ + b$ with trainable weight and bias."""

    def __init__(self, weight: Tensor, bias: Tensor, /) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: Tensor, /) -> Tensor:
        return F.linear(x, self.weight, self.bias)


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
            maxiter=10,
            atol=1e-6,
            rtol=1e-6,
        )


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
                fn, x0, args=(A, b), maxiter=256, atol=1e-5, rtol=1e-5
            )

        impl = solve if eager else compile_fresh(solve)
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


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("device", DEVICES)
class TestCorrectness(TestSuite):
    BATCH_SIZE = 5
    INPUT_SIZE = 2
    MAXITER = 100
    SOLVER_TOL = {
        torch.float32: (1e-6, 1e-6),
        torch.float64: (1e-8, 1e-8),
    }
    GRADCHECK_TOL = {
        torch.float32: (1e-3, 1e-3, 1e-4),
        torch.float64: (1e-6, 1e-6, 1e-8),
    }

    W0 = [[0.2, -0.1], [0.05, 0.15]]
    b0 = [0.3, -0.2]

    def test_linear_module_and_input_gradients_match_closed_form(
        self, device: str, dtype: torch.dtype
    ) -> None:
        r"""Check gradients for internal $A$ and input $b$ against the exact solve."""
        atol, rtol = self.SOLVER_TOL[dtype]
        weight = torch.tensor(self.W0, dtype=dtype, device=device)
        bias = torch.tensor(self.b0, dtype=dtype, device=device)
        y = torch.randn(
            self.BATCH_SIZE,
            self.INPUT_SIZE,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        model = LinearFixpointModel(
            weight,
            bias,
            maxiter=self.MAXITER,
            atol=atol,
            rtol=rtol,
        )

        x_star = model(y)  # X⁎ = X⁎Aᵀ + 𝟏bᵀ ⟺  (𝕀-A)⁻¹X⁎ = 𝟏bᵀ
        loss = x_star.square().sum()
        loss.backward()

        assert model.layer.A.grad is not None
        assert model.bias.grad is not None
        assert y.grad is None

        # SEC: compute reference gradient, using X⁎ = X⁎Aᵀ + 𝟏bᵀ  ⟺  (𝕀-A)X⁎ = 𝟏bᵀ
        reference_weight = weight.clone().requires_grad_()
        reference_bias = bias.clone().requires_grad_()
        eye = torch.eye(model.input_size, dtype=dtype, device=device)
        reference_x = torch.linalg.solve(eye - reference_weight, reference_bias)
        reference_x = reference_x.expand(self.BATCH_SIZE, -1)
        reference_loss = reference_x.square().sum()
        reference_loss.backward()

        assert reference_weight.grad is not None
        assert reference_bias.grad is not None
        self.assert_close(x_star, reference_x.detach(), atol=atol, rtol=rtol)
        self.assert_close(
            model.layer.A.grad,
            reference_weight.grad,
            atol=atol,
            rtol=rtol,
        )
        self.assert_close(
            model.bias.grad,
            reference_bias.grad,
            atol=atol,
            rtol=rtol,
        )
