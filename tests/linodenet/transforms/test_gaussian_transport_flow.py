import pytest
import torch

from linodenet.mappings.transforms import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
)
from linodenet_special import (
    bimodal_to_gaussian_value_and_grad,
    gaussian_to_bimodal_value_and_grad,
    gaussian_to_mixture_value_and_grad,
    mixture_to_gaussian_value_and_grad,
)
from tests.testing import SEEDS_10, TestSuite


class TestGaussianTransportFlow(TestSuite):
    VALUE_ATOL = 1e-5
    VALUE_RTOL = 1e-5
    LOGABSDET_ATOL = 1e-5
    LOGABSDET_RTOL = 1e-5
    BATCH_SIZE = 128

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    def test_bimodal_to_gaussian_logabsdet(self, seed: int) -> None:
        torch.manual_seed(seed)
        flow = BimodalToGaussian()
        with torch.no_grad():
            flow.mean.copy_(torch.tensor(1.75))
            flow.log_std.copy_(torch.tensor(-0.2))

        x = torch.linspace(-6.0, 6.0, self.BATCH_SIZE)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)
        expected_y, expected_grad = bimodal_to_gaussian_value_and_grad(
            x, flow.mean, flow.stddev
        )

        assert y.shape == x.shape
        assert forward_logabsdet.shape == x.shape
        assert xhat.shape == x.shape
        assert inverse_logabsdet.shape == x.shape

        self.assert_close(y, expected_y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet,
            expected_grad.log(),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )

    @pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
    def test_gaussian_to_bimodal_logabsdet(self, seed: int) -> None:
        torch.manual_seed(seed)
        flow = GaussianToBimodal()
        with torch.no_grad():
            flow.mean.copy_(torch.tensor(1.5))
            flow.log_std.copy_(torch.tensor(0.1))

        y = torch.linspace(-4.0, 4.0, self.BATCH_SIZE)
        x, forward_logabsdet = flow.encode_and_logabsdet(y)
        yhat, inverse_logabsdet = flow.decode_and_logabsdet(x)
        expected_x, expected_grad = gaussian_to_bimodal_value_and_grad(
            y, flow.mean, flow.stddev
        )

        self.assert_close(x, expected_x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet,
            expected_grad.log(),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )

    @pytest.mark.parametrize("seed", SEEDS_10[:3], ids="seed={}".format)
    def test_mixture_to_gaussian_logabsdet(self, seed: int) -> None:
        torch.manual_seed(seed)
        flow = MixtureToGaussian(3)
        with torch.no_grad():
            flow.weights.copy_(torch.tensor([-0.2, 0.1, 0.5]))
            flow.means.copy_(torch.tensor([-2.0, 0.0, 1.5]))
            flow.log_std.copy_(torch.tensor([-0.4, 0.0, 0.3]))

        x = torch.linspace(-7.0, 7.0, self.BATCH_SIZE)
        y, forward_logabsdet = flow.encode_and_logabsdet(x)
        xhat, inverse_logabsdet = flow.decode_and_logabsdet(y)
        weights = flow.weights.softmax(dim=-1)
        expected_y, expected_grad = mixture_to_gaussian_value_and_grad(
            x, weights, flow.means, flow.stddev
        )

        assert forward_logabsdet.shape == x.shape
        self.assert_close(y, expected_y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet,
            expected_grad.log(),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
        self.assert_close(xhat, x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )

    @pytest.mark.parametrize("seed", SEEDS_10[:3], ids="seed={}".format)
    def test_gaussian_to_mixture_logabsdet(self, seed: int) -> None:
        torch.manual_seed(seed)
        flow = GaussianToMixture(3)
        with torch.no_grad():
            flow.weights.copy_(torch.tensor([0.2, -0.1, 0.7]))
            flow.means.copy_(torch.tensor([-1.0, 0.5, 2.0]))
            flow.log_std.copy_(torch.tensor([0.0, -0.3, 0.2]))

        y = torch.linspace(-4.0, 4.0, self.BATCH_SIZE)
        x, forward_logabsdet = flow.encode_and_logabsdet(y)
        yhat, inverse_logabsdet = flow.decode_and_logabsdet(x)
        weights = flow.weights.softmax(dim=-1)
        expected_x, expected_grad = gaussian_to_mixture_value_and_grad(
            y, weights, flow.means, flow.stddev
        )

        assert forward_logabsdet.shape == y.shape
        self.assert_close(x, expected_x, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet,
            expected_grad.log(),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
        self.assert_close(yhat, y, atol=self.VALUE_ATOL, rtol=self.VALUE_RTOL)
        self.assert_close(
            forward_logabsdet + inverse_logabsdet,
            torch.zeros_like(forward_logabsdet),
            atol=self.LOGABSDET_ATOL,
            rtol=self.LOGABSDET_RTOL,
        )
