r"""Test the `torch.jit` module with a protocol subclass."""

import pytest
import torch
from torch import Tensor, jit


def test_jit_list() -> None:
    def norm_list(x: Tensor, dim: list[int]) -> Tensor:
        return x.abs().sum(dim=dim)

    jit.script(norm_list)


def test_jit_list_call_with_tuple() -> None:
    def norm_list(x: Tensor, dim: list[int]) -> Tensor:
        return x.abs().sum(dim=dim)

    fn = jit.script(norm_list)
    arg = torch.randn(5, 3, 3)
    fn(arg, dim=(0, 1))


def test_jit_list_call_with_size() -> None:
    def norm_list(x: Tensor, dim: list[int]) -> Tensor:
        return x.abs().sum(dim=dim)

    fn = jit.script(norm_list)
    arg = torch.randn(5, 3, 3)
    fn(arg, dim=torch.Size((0, 1)))


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/pytorch/pytorch/issues/64700",
    raises=RuntimeError,
)
def test_jit_tuple() -> None:
    def norm_tuple(x: Tensor, dim: tuple[int, ...]) -> Tensor:
        return x.abs().sum(dim=dim)

    jit.script(norm_tuple)


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/pytorch/pytorch/issues/64700",
    raises=RuntimeError,
)
def test_jit_torch_size() -> None:
    def norm_size(x: Tensor, dim: torch.Size) -> Tensor:
        return x.abs().sum(dim=dim)

    jit.script(norm_size)
