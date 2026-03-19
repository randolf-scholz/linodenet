import torch
from torch import Tensor


def pad(
    x: Tensor,
    value: float,
    padding_size: int,
    dim: int = -1,
    prepend: bool = False,
) -> Tensor:
    r"""Pad a tensor with a constant value along a given dimension."""
    shape = list(x.shape)
    shape[dim] = padding_size
    z = torch.full(shape, value, dtype=x.dtype, device=x.device)

    if prepend:
        return torch.cat((z, x), dim=dim)
    return torch.cat((x, z), dim=dim)


ARGS = [
    # vary padding_size
    (torch.randn(2, 3), 0.0, 1, -1),
    (torch.randn(2, 3), 0.0, 2, -1),
    (torch.randn(2, 3), 0.0, 3, -1),
    # vary padding_size
    (torch.randn(2, 3), 0.0, 1, -1),
    (torch.randn(2, 3), 0.0, 2, -1),
    (torch.randn(2, 3), 0.0, 3, -1),
    # vary shape
    (torch.randn(5), 0.0, 2, -1),
    (torch.randn(2, 6), 0.0, 2, -1),
    (torch.randn(2, 3, 4), 0.0, 2, -1),
    # vary dim
    (torch.randn(4, 3, 2), 0.0, 2, -1),
    (torch.randn(4, 3, 2), 0.0, 2, -2),
    (torch.randn(4, 3, 2), 0.0, 2, -3),
]


def test_script() -> None:
    compiled = torch.jit.script(pad)
    for args in ARGS:
        compiled(*args)


def test_compile() -> None:
    compiled = torch.compile(pad, dynamic=True)
    for args in ARGS:
        compiled(*args)
