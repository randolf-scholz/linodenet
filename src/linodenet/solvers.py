r"""Experimental ODE solvers for PyTorch.

The implementation in this module is intentionally small and conservative. The
forward pass runs under ``torch.no_grad`` and the backward pass implements a
discrete adjoint by recomputing one-step vector-Jacobian products from the
stored no-grad trajectory.
"""

__all__ = [
    "ODESolver",
    "euler_step",
    "heun_step",
    "midpoint_step",
    "odeint_forward",
    "solve_ivp",
]

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

import torch
from torch import Tensor


class ODESolver(StrEnum):
    r"""Explicit one-step ODE solver methods."""

    EULER = "euler"
    MIDPOINT = "midpoint"
    HEUN = "heun"

    @property
    def step_fn(self) -> Callable[..., Tensor]:
        r"""Return the one-step update function."""
        match self:
            case ODESolver.EULER:
                return euler_step
            case ODESolver.MIDPOINT:
                return midpoint_step
            case ODESolver.HEUN:
                return heun_step


ODESolverMethod = ODESolver | Literal["euler", "midpoint", "heun"]


def euler_step(
    vector_field: Callable[..., Tensor],  # [(...), (..., D), *args] -> (..., D)
    time: Tensor,  # (...)
    state: Tensor,  # (..., D)
    step_size: Tensor,  # (...)
    args: tuple[Tensor, ...] = (),  # arbitrary tensor parameters
) -> Tensor:  # (..., D)
    r"""Return one explicit Euler step."""
    return state + step_size * vector_field(time, state, *args)


def midpoint_step(
    vector_field: Callable[..., Tensor],  # [(...), (..., D), *args] -> (..., D)
    time: Tensor,  # (...)
    state: Tensor,  # (..., D)
    step_size: Tensor,  # (...)
    args: tuple[Tensor, ...] = (),  # arbitrary tensor parameters
) -> Tensor:  # (..., D)
    r"""Return one explicit midpoint step."""
    half_step = 0.5 * step_size
    midpoint = state + half_step * vector_field(time, state, *args)
    return state + step_size * vector_field(time + half_step, midpoint, *args)


def heun_step(
    vector_field: Callable[..., Tensor],  # [(...), (..., D), *args] -> (..., D)
    time: Tensor,  # (...)
    state: Tensor,  # (..., D)
    step_size: Tensor,  # (...)
    args: tuple[Tensor, ...] = (),  # arbitrary tensor parameters
) -> Tensor:  # (..., D)
    r"""Return one explicit trapezoidal/Heun step."""
    slope_start = vector_field(time, state, *args)
    slope_end = vector_field(
        time + step_size,
        state + step_size * slope_start,
        *args,
    )
    return state + 0.5 * step_size * (slope_start + slope_end)


def _num_steps(start_time: Tensor, target_time: Tensor, step_size: float) -> int:
    interval = float((target_time - start_time).detach().cpu())
    if interval < 0:
        raise ValueError("t_eval must be sorted and greater than or equal to t0.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    return int(torch.ceil(torch.tensor(interval / step_size)).item())


@torch.no_grad()
def odeint_forward(
    vector_field: Callable[..., Tensor],
    y0: Tensor,
    t0: Tensor,
    t_eval: Tensor,
    *,
    step_size: float,
    method: ODESolverMethod = "euler",
    args: tuple[Tensor, ...] = (),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Integrate an ODE and return the full history and requested states.

    Args:
        vector_field: Callable ``f(t, y, *args)`` returning ``dy/dt``.
        y0: Initial state.
        t0: Scalar initial time.
        t_eval: Sorted evaluation times greater than or equal to ``t0``.
        step_size: Maximum fixed step size. A final remainder step lands exactly
            on each requested evaluation time.
        method: One of ``"euler"``, ``"midpoint"``, or ``"heun"``.
        args: Extra tensor arguments passed to ``vector_field``.

    Returns:
        Tuple ``(solution, history, times, output_indices)``. ``solution`` has
        shape ``(len(t_eval), *y0.shape)`` and contains the requested states.
    """
    t0 = torch.as_tensor(t0, device=y0.device, dtype=y0.dtype)
    t_eval = torch.as_tensor(t_eval, device=y0.device, dtype=y0.dtype).reshape(-1)
    step_size_t = torch.as_tensor(step_size, device=y0.device, dtype=y0.dtype)

    step_fn = ODESolver(method).step_fn
    time = t0.clone()
    state = y0.clone()
    history = [state]
    times = [time]
    output_indices: list[int] = []

    for target_time in t_eval:
        num_steps = _num_steps(time, target_time, step_size)
        for _ in range(num_steps):
            remaining = target_time - time
            dt = torch.minimum(step_size_t, remaining)
            state = step_fn(vector_field, time, state, dt, args=args)
            time = time + dt
            history.append(state)
            times.append(time)
        output_indices.append(len(history) - 1)

    history_t = torch.stack(history)
    times_t = torch.stack(times)
    output_indices_t = torch.tensor(
        output_indices,
        device=y0.device,
        dtype=torch.int64,
    )
    return (
        history_t.index_select(0, output_indices_t),
        history_t,
        times_t,
        output_indices_t,
    )


class _DiscreteAdjoint(torch.autograd.Function):
    r"""ODE integration with custom discrete-adjoint backward."""

    @staticmethod
    def forward(
        ctx: Any,
        vector_field: Callable[..., Tensor],
        y0: Tensor,
        t0: Tensor,
        t_eval: Tensor,
        step_size: float,
        method: ODESolverMethod,
        *args: Tensor,
    ) -> Tensor:
        solution, history, times, output_indices = odeint_forward(
            vector_field,
            y0,
            t0,
            t_eval,
            step_size=step_size,
            method=method,
            args=args,
        )
        ctx.vector_field = vector_field
        ctx.step_size = step_size
        ctx.step_fn = ODESolver(method).step_fn
        ctx.save_for_backward(history, times, output_indices, *args)
        return solution

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Tensor | None,
    ) -> tuple[None, Tensor, None, None, None, None, *tuple[Tensor | None, ...]]:
        (grad_solution,) = grad_outputs
        history, times, output_indices, *args = ctx.saved_tensors
        grad_by_history = torch.zeros_like(history)
        if grad_solution is not None:
            grad_by_history = grad_by_history.index_add(
                0,
                output_indices,
                grad_solution,
            )

        num_steps = history.shape[0] - 1
        grad_y = torch.zeros_like(history[-1])
        grad_args: list[Tensor | None] = [
            torch.zeros_like(arg) if arg.requires_grad else None for arg in args
        ]

        for k in range(num_steps - 1, -1, -1):
            previous_time = times[k]
            time = times[k + 1]
            dt = time - previous_time
            grad_y = grad_y + grad_by_history[k + 1]
            with torch.enable_grad():
                state = history[k].detach().requires_grad_(True)
                active_args = tuple(
                    arg.detach().requires_grad_(arg.requires_grad) for arg in args
                )
                next_state = ctx.step_fn(
                    ctx.vector_field,
                    previous_time,
                    state,
                    dt,
                    args=active_args,
                )
                grads = torch.autograd.grad(
                    next_state,
                    (state, *active_args),
                    grad_y,
                    allow_unused=True,
                )

            grad_y = torch.zeros_like(state) if grads[0] is None else grads[0]
            for i, grad_arg in enumerate(grads[1:]):
                grad_arg_accumulator = grad_args[i]
                if grad_arg is not None and grad_arg_accumulator is not None:
                    grad_args[i] = grad_arg_accumulator + grad_arg

        return None, grad_y + grad_by_history[0], None, None, None, None, *grad_args


def solve_ivp(
    vector_field: Callable[..., Tensor],
    y0: Tensor,
    t0: float | Tensor,
    t_eval: float | Tensor,
    *,
    step_size: float,
    method: ODESolverMethod = "euler",
    args: tuple[Tensor, ...] = (),
) -> Tensor:
    r"""Integrate ``dy/dt = f(t, y, *args)`` from ``t0`` to ``t_eval``.

    Evaluation times are currently assumed to be sorted and greater than or
    equal to ``t0``. This wrapper returns the state at each evaluation time and
    uses a custom discrete adjoint for gradients with respect to ``y0`` and
    tensor ``args``.
    """
    t0_t = torch.as_tensor(t0, device=y0.device, dtype=y0.dtype)
    t_eval_t = torch.as_tensor(t_eval, device=y0.device, dtype=y0.dtype)
    return _DiscreteAdjoint.apply(
        vector_field,
        y0,
        t0_t,
        t_eval_t,
        step_size,
        method,
        *args,
    )
