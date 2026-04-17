import torch
from torch import Tensor

from linodenet_special import matrix_log, matrix_sqrt
from tests.testing import SEEDS_10


def _skew_symmetric(x: Tensor, /) -> Tensor:
    return 0.5 * (x - x.mT)


@torch.no_grad()
def test_matrix_log_roundtrip() -> None:
    for seed in SEEDS_10:
        torch.manual_seed(seed)
        x = torch.randn(8, 5, 5)
        y = torch.matrix_exp(_skew_symmetric(x))
        z = matrix_log(y)

        torch.testing.assert_close(
            torch.matrix_exp(z),
            y.to(dtype=z.dtype),
            atol=1e-5,
            rtol=1e-5,
        )


@torch.no_grad()
def test_matrix_sqrt_roundtrip() -> None:
    for seed in SEEDS_10:
        torch.manual_seed(seed)
        x = torch.randn(8, 5, 5)
        y = x.mT @ x
        z = matrix_sqrt(y)

        assert torch.allclose(z @ z, y.to(dtype=z.dtype), atol=1e-5, rtol=1e-5)
        assert torch.allclose(z, z.mT, atol=1e-6, rtol=1e-6)
