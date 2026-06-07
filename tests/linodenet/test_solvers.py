r"""Tests for experimental ODE solvers."""

from typing import NamedTuple

import pytest
import torch
from torch import Tensor

from linodenet.solvers import ODESolverMethod, solve_ivp


class UnivariateLinearSystem(NamedTuple):
    r"""Univariate ODE ``dx/dt = λx``."""

    x0: Tensor
    rate: Tensor
    t0: Tensor
    t1: Tensor

    @property
    def args(self) -> tuple[Tensor]:
        r"""Return explicit tensor arguments for the custom autograd operator."""
        return (self.rate,)

    def vector_field(
        self,
        _time: Tensor,
        state: Tensor,
        rate: Tensor,
        /,
    ) -> Tensor:
        r"""Return ``dx/dt = λx``."""
        return rate * state

    def analytic_solution(self) -> Tensor:
        r"""Return the exact terminal state ``x₀ exp(λ(t₁-t₀))``."""
        return self.x0 * torch.exp(self.rate * (self.t1 - self.t0))


class MultivariateLinearSystem(NamedTuple):
    r"""Multivariate ODE ``dx/dt = Ax``."""

    x0: Tensor
    matrix: Tensor
    t0: Tensor
    t1: Tensor

    @property
    def args(self) -> tuple[Tensor]:
        r"""Return explicit tensor arguments for the custom autograd operator."""
        return (self.matrix,)

    def vector_field(
        self,
        _time: Tensor,
        state: Tensor,
        matrix: Tensor,
        /,
    ) -> Tensor:
        r"""Return ``dx/dt = Ax``."""
        return matrix @ state

    def analytic_solution(self) -> Tensor:
        r"""Return the exact terminal state ``exp(A(t₁-t₀))x₀``."""
        transition = torch.linalg.matrix_exp((self.t1 - self.t0) * self.matrix)
        return transition @ self.x0


class HarmonicOscillator(NamedTuple):
    r"""Harmonic oscillator ODE in position/momentum coordinates."""

    x0: Tensor
    frequency: Tensor
    t0: Tensor
    t1: Tensor

    @property
    def args(self) -> tuple[Tensor]:
        r"""Return explicit tensor arguments for the custom autograd operator."""
        return (self.frequency,)

    def matrix(self, frequency: Tensor) -> Tensor:
        r"""Return the oscillator matrix ``[[0, 1], [-ω², 0]]``."""
        zero = torch.zeros_like(frequency)
        one = torch.ones_like(frequency)
        return torch.stack(
            [
                torch.stack([zero, one]),
                torch.stack([-frequency.square(), zero]),
            ]
        )

    def vector_field(
        self,
        _time: Tensor,
        state: Tensor,
        frequency: Tensor,
        /,
    ) -> Tensor:
        r"""Return ``dq/dt=p`` and ``dp/dt=-ω²q``."""
        return self.matrix(frequency) @ state

    def analytic_solution(self) -> Tensor:
        r"""Return the exact terminal state via matrix exponential."""
        transition = torch.linalg.matrix_exp(
            (self.t1 - self.t0) * self.matrix(self.frequency)
        )
        return transition @ self.x0


class TestSolver:
    r"""Tests for fixed-step ODE solvers."""

    @staticmethod
    def make_univariate_linear_system() -> UnivariateLinearSystem:
        r"""Return a stable univariate linear ODE test case."""
        return UnivariateLinearSystem(
            x0=torch.tensor(1.25, dtype=torch.float64, requires_grad=True),
            rate=torch.tensor(-0.4, dtype=torch.float64, requires_grad=True),
            t0=torch.tensor(0.0, dtype=torch.float64),
            t1=torch.tensor(0.5, dtype=torch.float64),
        )

    @staticmethod
    def make_multivariate_linear_system() -> MultivariateLinearSystem:
        r"""Return a stable multivariate linear ODE test case."""
        matrix = torch.tensor(
            [[-0.30, 0.10], [-0.20, -0.40]],
            dtype=torch.float64,
            requires_grad=True,
        )
        return MultivariateLinearSystem(
            x0=torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True),
            matrix=matrix,
            t0=torch.tensor(0.0, dtype=torch.float64),
            t1=torch.tensor(0.5, dtype=torch.float64),
        )

    @staticmethod
    def make_harmonic_oscillator() -> HarmonicOscillator:
        r"""Return a harmonic oscillator test case."""
        return HarmonicOscillator(
            x0=torch.tensor([1.0, 0.25], dtype=torch.float64, requires_grad=True),
            frequency=torch.tensor(1.7, dtype=torch.float64, requires_grad=True),
            t0=torch.tensor(0.0, dtype=torch.float64),
            t1=torch.tensor(0.5, dtype=torch.float64),
        )

    @pytest.mark.parametrize("method", ["euler", "midpoint", "heun"])
    def test_forward_and_backward_runs(self, method: ODESolverMethod) -> None:
        r"""Check that forward and backward execute and produce finite gradients."""
        case = self.make_univariate_linear_system()
        actual = solve_ivp(
            case.vector_field,
            case.x0,
            case.t0,
            case.t1,
            step_size=0.05,
            method=method,
            args=case.args,
        )

        loss = actual.square()
        loss.backward()

        assert actual.isfinite().all()
        assert case.x0.grad is not None
        assert case.rate.grad is not None
        assert case.x0.grad.isfinite().all()
        assert case.rate.grad.isfinite().all()

    @pytest.mark.parametrize(
        ("method", "atol", "rtol"),
        [
            ("euler", 1e-3, 1e-3),
            ("midpoint", 1e-5, 1e-5),
            ("heun", 1e-5, 1e-5),
        ],
    )
    def test_univariate_solution_matches_analytic(
        self, method: ODESolverMethod, atol: float, rtol: float
    ) -> None:
        r"""Compare ``dx/dt = λx`` solution with ``x₀ exp(λt)``."""
        case = self.make_univariate_linear_system()
        actual = solve_ivp(
            case.vector_field,
            case.x0,
            case.t0,
            case.t1,
            step_size=0.005,
            method=method,
            args=case.args,
        )
        expected = case.analytic_solution()

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("method", "atol", "rtol"),
        [
            ("euler", 1e-3, 1e-3),
            ("midpoint", 1e-5, 1e-5),
            ("heun", 1e-5, 1e-5),
        ],
    )
    def test_multivariate_solution_matches_analytic(
        self, method: ODESolverMethod, atol: float, rtol: float
    ) -> None:
        r"""Compare ``dx/dt = Ax`` solution with ``exp(At)x₀``."""
        case = self.make_multivariate_linear_system()
        actual = solve_ivp(
            case.vector_field,
            case.x0,
            case.t0,
            case.t1,
            step_size=0.005,
            method=method,
            args=case.args,
        )
        expected = case.analytic_solution()

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        ("method", "atol", "rtol"),
        [
            ("euler", 5e-3, 5e-3),
            ("midpoint", 1e-4, 1e-4),
            ("heun", 1e-4, 1e-4),
        ],
    )
    def test_harmonic_oscillator_solution_matches_analytic(
        self, method: ODESolverMethod, atol: float, rtol: float
    ) -> None:
        r"""Compare harmonic oscillator solution with matrix exponential."""
        case = self.make_harmonic_oscillator()
        actual = solve_ivp(
            case.vector_field,
            case.x0,
            case.t0,
            case.t1,
            step_size=0.005,
            method=method,
            args=case.args,
        )
        expected = case.analytic_solution()

        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
