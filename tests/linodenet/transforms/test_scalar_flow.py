import pytest
import torch
from torch.func import grad, vmap

from linodenet.mappings.transforms.scalar import (
    CELU,
    ELU,
    EntLU,
    Sigmoid,
    SmoothSoftsign,
    Softplus,
    Softsign,
    Tanh,
    Tanhshrink,
)

from .test_transform import TestTransform

SCALAR_TRANSFORMS = {
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "softsign": Softsign(),
    "smooth-softsign": SmoothSoftsign(),
    "softplus": Softplus(),
    "elu": ELU(),
    "celu": CELU(),
    "entlu": EntLU(),
    "tanhshrink": Tanhshrink(),
}


class TestScalarFlow(TestTransform):
    BATCH_SIZE = 64

    @pytest.mark.parametrize(
        ("transform", "x", "y"),
        [
            pytest.param(
                Sigmoid(),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(0.8).add_(0.1),
                id="sigmoid",
            ),
            pytest.param(
                Tanh(),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(1.8).sub_(0.9),
                id="tanh",
            ),
            pytest.param(
                Softsign(),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(1.8).sub_(0.9),
                id="softsign",
            ),
            pytest.param(
                SmoothSoftsign(),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(1.8).sub_(0.9),
                id="smooth-softsign",
            ),
            pytest.param(
                Softplus(),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(5.0).add_(0.1),
                id="softplus",
            ),
            pytest.param(
                ELU(alpha=0.7),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(5.0).sub_(0.65),
                id="elu",
            ),
            pytest.param(
                CELU(alpha=0.7),
                torch.randn(BATCH_SIZE),
                torch.rand(BATCH_SIZE).mul_(5.0).sub_(0.65),
                id="celu",
            ),
        ],
    )
    def test_invertibility(self, transform, x: torch.Tensor, y: torch.Tensor) -> None:
        self.assert_invertible(
            transform,
            x,
            y,
            atol=1e-5,
            rtol=1e-5,
            logdet_atol=1e-5,
            logdet_rtol=1e-5,
        )

    @pytest.mark.parametrize("case", SCALAR_TRANSFORMS)
    def test_logabsdet_matches_finite_difference(self, case: str) -> None:
        transform = SCALAR_TRANSFORMS[case]
        x = torch.linspace(-5, 5, 100)
        _, actual = transform.encode_and_logabsdet(x)
        grad_fn = vmap(grad(transform.encode))
        derivatives = grad_fn(x)
        expected = derivatives.abs().log()
        assert actual.isfinite().all()
        assert expected.isfinite().all()
        if case in {"tanh", "sigmoid"}:
            # the derivatives go to zero fast. derivatives become numerically zero
            # actual logabsdet is more robust.
            atol, rtol = 1e-3, 1e-3
        else:
            atol, rtol = 1e-6, 1e-6
        self.assert_close(actual, expected, atol=atol, rtol=rtol)
