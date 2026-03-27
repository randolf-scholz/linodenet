import pytest
import torch

from linodenet.initializations.thomson_initialization import (
    OptimizerStatus,
    thomson_initialization,
)


@pytest.mark.parametrize("num", [2, 3, 4, 8, 16, 32, 128], ids="num={}".format)
@pytest.mark.parametrize("dim", [2, 3, 4, 8, 16, 32, 128], ids="dim={}".format)
def test_thomson_initialization(num: int, dim: int) -> None:
    sol = thomson_initialization(num, dim, seed=0)
    assert sol.status is OptimizerStatus.SUCCESS


def test_thomson_initialization_respects_dtype() -> None:
    sol = thomson_initialization(4, 3, dtype=torch.float64, seed=0)

    assert sol.status is OptimizerStatus.SUCCESS
    assert sol.x.dtype is torch.float64
    assert sol.jac.dtype is torch.float64
