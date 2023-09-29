#!/usr/bin/env python
"""Compare speed of different aggregation algorithms."""

import pytest
from torch import Tensor, jit


@pytest.mark.skip(reason="Not implemented")
def test_speed():
    @jit.script
    def square_norm(x: Tensor) -> Tensor:
        return (x**2).sum()

    # %timeit
    # x.abs().sum()
    # %timeit
    # x.norm()
    # %timeit
    # square_norm(x)
    # %timeit(x ** 2).sum()
    # %timeit(x * x).sum()
    # %timeit
    # torch.einsum("..., ... -> ", x, x)


if __name__ == "__main__":
    pass
