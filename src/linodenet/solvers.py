r"""Experimental ODE solvers for PyTorch.

The implementation in this module is intentionally small and conservative. The
forward pass uses ``torch.while_loop`` under ``torch.no_grad`` and the backward
pass implements a discrete adjoint by recomputing one-step vector-Jacobian
products from the stored no-grad trajectory.
"""

__all__ = [
    "ODESolver",
    "ODESolverState",
    "euler_step",
    "heun_step",
    "midpoint_step",
    "odeint_forward",
    "solve_ivp",
]

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Final, Literal, NamedTuple

import torch
from torch import Tensor

_MIN_STEPS: Final[int] = 1


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


class ODESolverState(NamedTuple):
    r"""Loop state for fixed-step ODE integration."""

    step_index: Tensor
    time: Tensor
    state: Tensor
    history: Tensor


def euler_step(
    vector_field: Callable[..., Tensor],  # ((), (..., D), *args) -> (..., D)
    time: Tensor,  # ()
    state: Tensor,  # (..., D)
    step_size: Tensor,  # ()
    *args: Tensor,  # arbitrary tensor parameters
) -> Tensor:
    r"""Return one explicit Euler step."""
    return state + step_size * vector_field(time, state, *args)


def midpoint_step(
    vector_field: Callable[..., Tensor],  # ((), (..., D), *args) -> (..., D)
    time: Tensor,  # ()
    state: Tensor,  # (..., D)
    step_size: Tensor,  # ()
    *args: Tensor,  # arbitrary tensor parameters
) -> Tensor:
    r"""Return one explicit midpoint step."""
    half_step = 0.5 * step_size
    midpoint = state + half_step * vector_field(time, state, *args)
    return state + step_size * vector_field(time + half_step, midpoint, *args)


def heun_step(
    vector_field: Callable[..., Tensor],  # ((), (..., D), *args) -> (..., D)
    time: Tensor,  # ()
    state: Tensor,  # (..., D)
    step_size: Tensor,  # ()
    *args: Tensor,  # arbitrary tensor parameters
) -> Tensor:
    r"""Return one explicit trapezoidal/Heun step."""
    slope_start = vector_field(time, state, *args)
    slope_end = vector_field(
        time + step_size,
        state + step_size * slope_start,
        *args,
    )
    return state + 0.5 * step_size * (slope_start + slope_end)


def _num_steps(t0: Tensor, t1: Tensor, step_size: float) -> int:
    interval = float((t1 - t0).detach().cpu())
    if interval < 0:
        raise ValueError("t1 must be greater than or equal to t0.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    return max(_MIN_STEPS, int(torch.ceil(torch.tensor(interval / step_size)).item()))


@torch.no_grad()
def odeint_forward(
    vector_field: Callable[..., Tensor],
    y0: Tensor,
    t0: Tensor,
    t1: Tensor,
    *,
    step_size: float,
    method: ODESolverMethod = "euler",
    args: tuple[Tensor, ...] = (),
) -> Tensor:
    r"""Integrate an ODE with ``torch.while_loop`` and return the full history.

    Args:
        vector_field: Callable ``f(t, y, *args)`` returning ``dy/dt``.
        y0: Initial state.
        t0: Scalar initial time.
        t1: Scalar final time.
        step_size: Maximum fixed step size. A final remainder step lands exactly
            on ``t1``.
        method: One of ``"euler"``, ``"midpoint"``, or ``"heun"``.
        args: Extra tensor arguments passed to ``vector_field``.

    Returns:
        Tensor of shape ``(num_steps + 1, *y0.shape)`` containing the trajectory.
    """
    t0 = torch.as_tensor(t0, device=y0.device, dtype=y0.dtype)
    t1 = torch.as_tensor(t1, device=y0.device, dtype=y0.dtype)
    step_size_t = torch.as_tensor(step_size, device=y0.device, dtype=y0.dtype)
    num_steps = _num_steps(t0, t1, step_size)

    history = torch.empty((num_steps + 1, *y0.shape), device=y0.device, dtype=y0.dtype)
    history[0] = y0

    step_fn = ODESolver(method).step_fn

    def cond_fn(loop_state: ODESolverState, /) -> Tensor:
        step_index, _, _, _ = loop_state
        return step_index < num_steps

    def body_fn(loop_state: ODESolverState, /) -> ODESolverState:
        step_index, time, state, history = loop_state
        remaining = t1 - time
        dt = torch.minimum(step_size_t, remaining)
        next_state = step_fn(vector_field, time, state, dt, *args)
        next_index = step_index + 1
        next_time = time + dt
        next_history = history.clone().index_copy(
            0,
            next_index.reshape(1),
            next_state.unsqueeze(0),
        )
        return ODESolverState(next_index, next_time, next_state, next_history)

    step_index = torch.zeros((), device=y0.device, dtype=torch.int64)
    initial_state = ODESolverState(step_index, t0.clone(), y0.clone(), history)
    final_state = torch.while_loop(cond_fn, body_fn, (initial_state,))
    return final_state.history


class _DiscreteAdjoint(torch.autograd.Function):
    r"""ODE integration with custom discrete-adjoint backward."""

    @staticmethod
    def forward(
        ctx: Any,
        vector_field: Callable[..., Tensor],
        y0: Tensor,
        t0: Tensor,
        t1: Tensor,
        step_size: float,
        method: ODESolverMethod,
        *args: Tensor,
    ) -> Tensor:
        history = odeint_forward(
            vector_field,
            y0,
            t0,
            t1,
            step_size=step_size,
            method=method,
            args=args,
        )
        ctx.vector_field = vector_field
        ctx.step_size = step_size
        ctx.step_fn = ODESolver(method).step_fn
        ctx.save_for_backward(history, t0, t1, *args)
        return history[-1]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Tensor | None,
    ) -> tuple[None, Tensor, None, None, None, None, *tuple[Tensor | None, ...]]:
        (grad_y,) = grad_outputs
        history, t0, t1, *args = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(history[-1])

        num_steps = history.shape[0] - 1
        step_size = torch.as_tensor(
            ctx.step_size,
            device=history.device,
            dtype=history.dtype,
        )
        time = torch.as_tensor(t1, device=history.device, dtype=history.dtype)
        grad_args: list[Tensor | None] = [
            torch.zeros_like(arg) if arg.requires_grad else None for arg in args
        ]

        for k in range(num_steps - 1, -1, -1):
            previous_time = torch.maximum(time - step_size, t0)
            dt = time - previous_time
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
                    *active_args,
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
            time = previous_time

        return None, grad_y, None, None, None, None, *grad_args


def solve_ivp(
    vector_field: Callable[..., Tensor],
    y0: Tensor,
    t0: float | Tensor,
    t1: float | Tensor,
    *,
    step_size: float,
    method: ODESolverMethod = "euler",
    args: tuple[Tensor, ...] = (),
) -> Tensor:
    r"""Integrate ``dy/dt = f(t, y, *args)`` from ``t0`` to ``t1``.

    This wrapper returns only the terminal state and uses a custom discrete
    adjoint for gradients with respect to ``y0`` and tensor ``args``.
    """
    t0_t = torch.as_tensor(t0, device=y0.device, dtype=y0.dtype)
    t1_t = torch.as_tensor(t1, device=y0.device, dtype=y0.dtype)
    return _DiscreteAdjoint.apply(
        vector_field,
        y0,
        t0_t,
        t1_t,
        step_size,
        method,
        *args,
    )
