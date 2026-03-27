import pytest
import torch
from torch.linalg import vector_norm

from linodenet.initializations.thomson_initialization import wide_angle_sphere_init
from tests.testing import DEVICES


@pytest.mark.parametrize("num", [1, 2, 3, 4, 8, 16, 32, 128], ids="num={}".format)
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 8, 16, 32, 128], ids="dim={}".format)
@pytest.mark.parametrize("device", DEVICES)
def test_wide_angle_sphere_init(num: int, dim: int, device: str) -> None:
    points = wide_angle_sphere_init(num, dim, seed=0, device=device)
    norms = vector_norm(points, dim=1)
    assert points.shape == (num, dim)
    assert points.isfinite().all()
    assert torch.allclose(norms, torch.ones_like(norms))


def test_wide_angle_sphere_init_respects_dtype() -> None:
    points = wide_angle_sphere_init(4, 3, dtype=torch.float64, seed=0)

    assert points.dtype is torch.float64
