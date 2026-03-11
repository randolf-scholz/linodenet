import pytest
from mnf.thomson_init import OptimizerStatus, thomson_initialization


@pytest.mark.parametrize("num", [2, 3, 4, 8, 16, 32, 128], ids="num={}".format)
@pytest.mark.parametrize("dim", [2, 3, 4, 8, 16, 32, 128], ids="dim={}".format)
def test_thomson_initialization(num: int, dim: int) -> None:
    sol = thomson_initialization(num, dim, seed=0)
    assert sol.status is OptimizerStatus.SUCCESS
