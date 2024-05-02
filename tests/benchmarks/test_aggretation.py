r"""Compare speed of different aggregation algorithms."""

import torch
from pytest import mark
from torch import Tensor, jit


def square_norm(x: Tensor) -> Tensor:
    r"""Manual Square norm."""
    return (x**2).sum()


aggregations = {
    "norm2-mul-sum": lambda x: x.mul(x).sum(),
    "norm2-pow2-sum": lambda x: x.pow(2).sum(),
    "norm2-jit": jit.script(square_norm),
    "norm2-einsum": lambda x: torch.einsum("..., ... -> ", x, x),
}


@mark.parametrize("name", aggregations)
@mark.benchmark
class TestAggregation:
    r"""Compare speed of different aggregation algorithms."""

    small = torch.randn(100)
    medium = torch.randn(1000)
    large = torch.randn(10_000)
    xlarge = torch.randn(100_000)
    xxlarge = torch.randn(1_000_000)
    xxxlarge = torch.randn(10_000_000)

    def test_small(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.small)

    def test_medium(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.medium)

    def test_large(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.large)

    def test_xlarge(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.xlarge)

    def test_xxlarge(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.xxlarge)

    def test_xxxlarge(self, benchmark, name):
        func = aggregations[name]
        benchmark(func, self.xxxlarge)


#
# @mark.parametrize("size", [10, 100, 1000, 10000])
# @mark.parametrize("name", aggregations)
# @mark.benchmark(group_by="name")
# def test_speed(benchmark, name: str, size: int) -> None:
#     r"""Compare speed of different aggregation algorithms."""
#     func = aggregations[name]
#     x = torch.randn(size)
#     benchmark(func, x)
