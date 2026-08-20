from collections.abc import Callable
from typing import cast

import torch
import torchode as to
from torch import Tensor, nn


class LinearSystem(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(
        self,
        _: Tensor,
        y: Tensor,
        /,
    ) -> Tensor:
        return self.linear(y)


def apply_batched[R: Tensor | tuple[Tensor, ...]](
    fn: Callable[..., R], batch_shape: tuple[int, ...], args: tuple[Tensor, ...], /
) -> R:
    args_flat = []
    for x in args:
        event_shape = x.shape[len(batch_shape) :]
        assert x.shape == batch_shape + event_shape
        args_flat.append(x.reshape(-1, *event_shape))

    ys_flat = fn(*args_flat)
    if isinstance(ys_flat, Tensor):
        return cast("R", ys_flat.reshape(*batch_shape, *ys_flat.shape[1:]))
    return cast("R", tuple(y.reshape(*batch_shape, *y.shape[1:]) for y in ys_flat))


def test_batched_single() -> None:
    batch_shape = (16,)
    dim = 8
    steps = 10
    system = LinearSystem(dim)
    y0 = torch.randn(*batch_shape, dim)
    t_eval = torch.rand(*batch_shape, steps)
    system = to.ODETerm(system)
    solver = to.AutoDiffAdjoint(
        step_method=to.Dopri5(term=system),
        step_size_controller=to.IntegralController(
            atol=1e-6,
            rtol=1e-3,
            term=system,
        ),
    )
    sol = solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
    print(sol.stats)


def test_batched_multiple() -> None:
    batch_shape = (1, 2, 3)
    dim = 8
    steps = 10
    system = LinearSystem(dim)
    y0 = torch.randn(*batch_shape, dim)
    t_eval = torch.rand(*batch_shape, steps)
    system = to.ODETerm(system)
    solver = to.AutoDiffAdjoint(
        step_method=to.Dopri5(term=system),
        step_size_controller=to.IntegralController(
            atol=1e-6,
            rtol=1e-3,
            term=system,
        ),
    )

    def solve(y0_: Tensor, t_eval_: Tensor) -> Tensor:
        sol = solver.solve(to.InitialValueProblem(y0=y0_, t_eval=t_eval_))
        torch.testing.assert_close(sol.ts, t_eval_)
        return sol.ys

    result = apply_batched(solve, batch_shape, (y0, t_eval))
    assert result.shape == (*batch_shape, steps, dim)
