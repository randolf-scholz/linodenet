import pytest
import torch
from torch.autograd import gradcheck
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

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(Sigmoid(), id="sigmoid"),
            pytest.param(Tanh(), id="tanh"),
            pytest.param(Softsign(), id="softsign"),
            pytest.param(SmoothSoftsign(), id="smooth-softsign"),
            pytest.param(Softplus(), id="softplus"),
            pytest.param(ELU(alpha=0.7), id="elu"),
            pytest.param(CELU(alpha=0.7), id="celu"),
            pytest.param(EntLU(), id="entlu"),
            pytest.param(Tanhshrink(), id="tanhshrink"),
        ],
    )
    def test_logabsdet_matches_finite_difference(self, transform) -> None:
        x = torch.linspace(-10, 10, 1000)
        # if isinstance(transform, EntLU):
        #     x = x.clamp(min=-0.75, max=0.75)
        #     signs = x.sign().where(x != 0, torch.ones_like(x))
        #     x = torch.where(x.abs() < 0.25, 0.25 * signs, x)
        # elif isinstance(transform, Tanhshrink):
        #     x = x + 0.25 * x.sign().where(x != 0, torch.ones_like(x))
        _, logabsdet = transform.encode_and_logabsdet(x)
        assert logabsdet.isfinite().all()
        derivative = vmap(grad(transform.encode))(x)
        self.assert_close(logabsdet, derivative.abs().log(), atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(
        ("transform", "x", "y"),
        [
            pytest.param(
                Sigmoid(),
                torch.randn(8, dtype=torch.float64),
                torch.rand(8, dtype=torch.float64).mul_(0.8).add_(0.1),
                id="sigmoid",
            ),
            pytest.param(
                Tanh(),
                torch.randn(8, dtype=torch.float64),
                torch.rand(8, dtype=torch.float64).mul_(1.8).sub_(0.9),
                id="tanh",
            ),
            pytest.param(
                Softsign(),
                torch.randn(8, dtype=torch.float64).add_(0.25),
                torch.rand(8, dtype=torch.float64).mul_(1.6).sub_(0.8).add_(0.1),
                id="softsign",
            ),
            pytest.param(
                SmoothSoftsign(),
                torch.randn(8, dtype=torch.float64),
                torch.rand(8, dtype=torch.float64).mul_(1.8).sub_(0.9),
                id="smooth-softsign",
            ),
            pytest.param(
                Softplus(),
                torch.randn(8, dtype=torch.float64),
                torch.rand(8, dtype=torch.float64).mul_(5.0).add_(0.1),
                id="softplus",
            ),
            pytest.param(
                ELU(alpha=0.7),
                torch.randn(8, dtype=torch.float64).add_(0.25),
                torch.rand(8, dtype=torch.float64).mul_(5.0).sub_(0.65),
                id="elu",
            ),
            pytest.param(
                CELU(alpha=0.7),
                torch.randn(8, dtype=torch.float64).add_(0.25),
                torch.rand(8, dtype=torch.float64).mul_(5.0).sub_(0.65),
                id="celu",
            ),
            pytest.param(
                EntLU(),
                torch.linspace(-0.8, 0.8, 8, dtype=torch.float64),
                None,
                id="entlu",
            ),
        ],
    )
    def test_gradcheck(
        self,
        transform,
        x: torch.Tensor,
        y: torch.Tensor | None,
    ) -> None:
        x = x.requires_grad_()
        assert gradcheck(transform.encode, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
        if y is not None:
            y = y.requires_grad_()
            assert gradcheck(transform.decode, (y,), eps=1e-6, atol=1e-4, rtol=1e-3)
