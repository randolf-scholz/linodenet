import math

import torch

from linodenet_special.fallbacks.fixpoint_iteration import fixpoint_solve
from tests.testing import TestCase


class TestFixPointIteration(TestCase):
    VALUE_ATOL = 1e-6
    VALUE_RTOL = 1e-6

    def test_cosine_forward(self) -> None:
        r"""Solve the classical fixed point equation $x = \cos(x)$."""

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.cos(x)

        x0 = torch.zeros(())
        x = fixpoint_solve(fn, x0, maxiter=128, atol=1e-10, rtol=0.0)

        self.assert_close(
            x,
            torch.tensor(0.7390851332151607),
            atol=self.VALUE_ATOL,
            rtol=self.VALUE_RTOL,
        )

    def test_cosine_backward(self) -> None:
        r"""Differentiate the parameterized fixed point near $x = \cos(x)$."""

        def fn(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return torch.cos(x) + shift

        shift = torch.zeros((), requires_grad=True)
        x0 = torch.zeros(())
        x = fixpoint_solve(fn, x0, shift, maxiter=128, atol=1e-10, rtol=0.0)
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
