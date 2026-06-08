r"""Compare third-party ODE solvers on learnable linear systems."""

from collections.abc import Callable
from typing import ClassVar, Literal

import pytest
import torch
import torchode as to
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

SolverName = Literal["torchdiffeq", "torchode"]
SolverMethod = Literal["euler", "heun"]


class LinearSystem(nn.Module):
    r"""Learnable linear ODE system ``dx/dt = Ax``."""

    def __init__(self, num_dim: int, /) -> None:
        super().__init__()
        self.linear = nn.Linear(num_dim, num_dim, bias=False, dtype=torch.float64)
        matrix = torch.tensor(
            [
                [-0.30, 0.10, 0.00],
                [-0.20, -0.40, 0.15],
                [0.05, -0.10, -0.25],
            ],
            dtype=torch.float64,
        )
        with torch.no_grad():
            self.linear.weight.copy_(matrix)

    def forward(self, _time: Tensor, state: Tensor, /) -> Tensor:
        r"""Evaluate the vector field at ``state``."""
        return self.linear(state)

    def solve_analytic(self, y0: Tensor, t_eval: Tensor, /) -> Tensor:
        r"""Return exact states ``exp(At)x₀``."""
        matrix = self.linear.weight.detach()
        if y0.ndim == 1:
            initial = y0.detach()
            return torch.stack(
                [torch.linalg.matrix_exp(time * matrix) @ initial for time in t_eval]
            )

        rows = []
        for initial, times in zip(y0.detach(), t_eval, strict=True):
            states = [
                torch.linalg.matrix_exp(time * matrix) @ initial for time in times
            ]
            rows.append(torch.stack(states))
        return torch.stack(rows)


def solve_torchdiffeq(
    system: LinearSystem,
    y0: Tensor,
    t_eval: Tensor,
    lengths: Tensor | None = None,
    /,
    *,
    method: SolverMethod,
    step_size: float,
) -> Tensor:
    r"""Solve via ``torchdiffeq`` and gather heterogeneous batch grids."""
    method_name = {"euler": "euler", "heun": "heun2"}[method]

    if y0.ndim == 1:
        return odeint(
            system,
            y0,
            t_eval,
            method=method_name,
            options={"step_size": step_size},
        )

    if lengths is None:
        raise ValueError("lengths must be provided for batched torchdiffeq solves.")

    requested_times = []
    for i, length in enumerate(lengths.tolist()):
        requested_times.append(t_eval[i, :length])
    common_times = torch.unique(torch.cat(requested_times), sorted=True)
    common_solution = odeint(
        system,
        y0,
        common_times,
        method=method_name,
        options={"step_size": step_size},
    )

    rows = []
    for i, sample_times in enumerate(requested_times):
        indices = torch.searchsorted(common_times, sample_times)
        rows.append(common_solution.index_select(0, indices)[:, i])
    return pad_sequence(rows, batch_first=True, padding_value=torch.nan)


def solve_torchode(
    system: LinearSystem,
    y0: Tensor,
    t_eval: Tensor,
    lengths: Tensor | None = None,
    /,
    *,
    method: SolverMethod,
    step_size: float,
) -> Tensor:
    r"""Solve via ``torchode`` with a fixed-step controller."""
    del lengths
    squeeze = y0.ndim == 1
    y0_batch = y0.unsqueeze(0) if squeeze else y0
    t_eval_batch = t_eval.unsqueeze(0) if t_eval.ndim == 1 else t_eval

    term = to.ODETerm(system)
    step_method = {"euler": to.Euler, "heun": to.Heun}[method](term)
    solver = to.AutoDiffAdjoint(
        step_method,
        to.FixedStepController(),
        max_steps=TestSolvers.MAX_STEPS + 1,
    )
    solution = solver.solve(
        to.InitialValueProblem(y0=y0_batch, t_eval=t_eval_batch),
        term,
        dt0=torch.full(
            (y0_batch.shape[0],),
            step_size,
            device=y0_batch.device,
            dtype=t_eval_batch.dtype,
        ),
    )

    assert (solution.status == to.Status.SUCCESS.value).all()
    return solution.ys.squeeze(0) if squeeze else solution.ys


def solve(
    solver: SolverName,
    system: LinearSystem,
    y0: Tensor,
    t_eval: Tensor,
    lengths: Tensor | None = None,
    /,
    *,
    step_size: float,
    method: SolverMethod,
) -> Tensor:
    r"""Dispatch to the requested third-party solver wrapper."""
    wrapper: Callable[..., Tensor]
    match solver:
        case "torchdiffeq":
            wrapper = solve_torchdiffeq
        case "torchode":
            wrapper = solve_torchode
        case _:
            raise ValueError

    return wrapper(system, y0, t_eval, lengths, method=method, step_size=step_size)


class TestSolvers:
    r"""Tests for third-party ODE solvers with autograd support."""

    BATCH_SIZE: ClassVar[int] = 4
    STEP_SIZE: ClassVar[float] = 0.005
    NUM_DIM: ClassVar[int] = 3
    MAX_STEPS: ClassVar[int] = 8
    MIN_STEPS: ClassVar[int] = 4

    CASES: ClassVar[tuple[tuple[SolverName, SolverMethod], ...]] = (
        ("torchdiffeq", "euler"),
        ("torchdiffeq", "heun"),
        ("torchode", "euler"),
        ("torchode", "heun"),
    )

    @classmethod
    def make_times(cls, *, batch_size: int | None = None) -> tuple[Tensor, Tensor]:
        r"""Return padded random time grids and their unpadded lengths."""
        generator = torch.Generator().manual_seed(0)
        size = 1 if batch_size is None else batch_size
        lengths = torch.randint(
            cls.MIN_STEPS,
            cls.MAX_STEPS + 1,
            (size,),
            generator=generator,
        )
        times = []
        for k in lengths.tolist():
            interval = k * cls.STEP_SIZE
            random_times = interval * torch.rand(
                k - 1,
                generator=generator,
                dtype=torch.float64,
            )
            times.append(
                torch.cat([torch.zeros(1, dtype=torch.float64), random_times])
                .sort()
                .values
            )

        padded = pad_sequence(times, batch_first=True, padding_value=torch.nan)
        final_times = torch.tensor([time[-1] for time in times], dtype=torch.float64)
        padded = torch.where(padded.isnan(), final_times[:, None], padded)
        return padded, lengths

    @staticmethod
    def assert_finite_gradients(
        system: LinearSystem, y0: Tensor, actual: Tensor, /
    ) -> None:
        r"""Assert that the solver output supports finite backpropagation."""
        loss = actual.nan_to_num().square().sum()
        loss.backward()

        assert y0.grad is not None
        assert system.linear.weight.grad is not None
        assert y0.grad.isfinite().all()
        assert system.linear.weight.grad.isfinite().all()

    @pytest.mark.parametrize(("solver", "method"), CASES)
    def test_unbatched(self, solver: SolverName, method: SolverMethod) -> None:
        r"""Check unbatched forward accuracy and gradients."""
        system = LinearSystem(self.NUM_DIM)
        y0 = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64, requires_grad=True)
        t_eval = self.make_times()[0].squeeze(0)

        actual = solve(
            solver, system, y0, t_eval, method=method, step_size=self.STEP_SIZE
        )
        expected = system.solve_analytic(y0, t_eval)

        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-4)
        self.assert_finite_gradients(system, y0, actual)

    @pytest.mark.parametrize(("solver", "method"), CASES)
    def test_batched(self, solver: SolverName, method: SolverMethod) -> None:
        r"""Check batched forward accuracy and gradients on heterogeneous time grids."""
        system = LinearSystem(self.NUM_DIM)
        y0 = torch.tensor(
            [
                [1.0, -0.5, 0.25],
                [0.5, 0.75, -1.0],
                [-0.25, 0.5, 1.25],
                [1.5, -1.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        t_eval, lengths = self.make_times(batch_size=self.BATCH_SIZE)
        assert t_eval.shape == (self.BATCH_SIZE, lengths.max().item())
        assert lengths.shape == (self.BATCH_SIZE,)
        assert y0.shape == (self.BATCH_SIZE, self.NUM_DIM)

        actual = solve(
            solver, system, y0, t_eval, lengths, method=method, step_size=self.STEP_SIZE
        )
        expected = system.solve_analytic(y0, t_eval)
        mask = torch.arange(t_eval.shape[1]) < lengths[:, None]

        torch.testing.assert_close(actual[mask], expected[mask], atol=5e-4, rtol=5e-4)
        self.assert_finite_gradients(system, y0, actual)
