import torch
from torch import Tensor, nn


@torch.no_grad()
def _fixpoint_iteration(fn, x: Tensor) -> Tensor:
    for _ in range(5):
        x = fn(x)
    return x


def fixpoint_solve_old(fn, x0: Tensor, *extra_args: Tensor) -> Tensor:
    with torch.no_grad():
        x_star = _fixpoint_iteration(lambda z: fn(z, *extra_args), x0)

    x_star1 = fn(x_star, *extra_args)
    x_star = x_star1.clone().detach().requires_grad_()
    x_star2 = fn(x_star, *extra_args)

    def backward_solve(g: Tensor) -> Tensor:
        return _fixpoint_iteration(
            lambda u: g + torch.autograd.grad(x_star2, x_star, u, retain_graph=True)[0],
            g,
        )

    x_star1.register_hook(backward_solve)
    return x_star1


def fixpoint_solve_new(fn, x0: Tensor, *extra_args: Tensor) -> Tensor:
    with torch.no_grad():
        x_star = _fixpoint_iteration(lambda z: fn(z, *extra_args), x0)

    x_star, vjp_fn, *_ = torch.func.vjp(
        lambda z: fn(z, *extra_args),
        x_star,
    )

    def backward_solve(g: Tensor) -> Tensor:
        return _fixpoint_iteration(lambda u: g + vjp_fn(u)[0], g)

    x_star.register_hook(backward_solve)
    return x_star


def check_eager(solver):
    y = torch.randn(5, 3)
    W = nn.Parameter(torch.randn(3, 3))
    b = nn.Parameter(torch.randn(3))
    y_star = solver(lambda z: z @ W.mH + b, y)
    loss = y_star.square().sum()
    loss.backward()
    assert W.grad is not None


def check_compiled_forward(solver):
    torch._dynamo.reset()
    y = torch.randn(5, 3)
    W = nn.Parameter(torch.randn(3, 3))
    b = nn.Parameter(torch.randn(3))

    @torch.compile
    def forward(y0) -> Tensor:
        y_star = solver(lambda z: z @ W.mH + b, y0)
        return y_star.square().sum()

    loss = forward(y)
    loss.backward()
    assert W.grad is not None


def check_compiled_backward(solver):
    torch._dynamo.reset()
    y = torch.randn(5, 3)
    W = nn.Parameter(torch.randn(3, 3))
    b = nn.Parameter(torch.randn(3))

    @torch.compile
    def backward(y0) -> None:
        y_star = solver(lambda z: z @ W.mH + b, y0)
        loss = y_star.square().sum()
        loss.backward()

    backward(y)
    assert W.grad is not None


def check(solver):
    checks = {
        "eager": check_eager,
        "compiled_forward": check_compiled_forward,
        "compiled_backward": check_compiled_backward,
    }
    excs: dict[str, Exception | None] = {}

    for name, check in checks.items():
        try:
            check(solver)
        except Exception as exc:
            excs[name] = exc
        else:
            excs[name] = None

    print(
        f"{solver.__name__}:\n"
        + "\n".join(
            f"  {name}: {'OK' if exc is None else 'FAIL'}" for name, exc in excs.items()
        )
    )
    for name, exc in excs.items():
        if exc is not None:
            print(f"{name}: {exc}")


def test() -> None:
    check(fixpoint_solve_old)
    check(fixpoint_solve_new)


if __name__ == "__main__":
    # torch._dynamo.config.compiled_autograd = True
    # test(fixpoint_solve_old)
    # test(fixpoint_solve_new)
    check_compiled_forward(fixpoint_solve_new)
    # check_compiled_backward(fixpoint_solve_new)
