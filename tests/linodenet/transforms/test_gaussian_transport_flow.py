import pytest
import torch

from linodenet.mappings.transforms import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
)
from tests.testing import SEEDS_10

from .test_transform import TestTransform


@pytest.mark.parametrize("seed", SEEDS_10, ids="seed={}".format)
class TestGaussianTransportFlow(TestTransform):
    VALUE_TOL = (1e-5, 1e-5)
    BATCH_SIZE = 128

    def test_bimodal_to_gaussian(self, seed: int) -> None:
        torch.manual_seed(seed)
        value_atol, value_rtol = self.VALUE_TOL
        flow = BimodalToGaussian()
        with torch.no_grad():
            flow.mean.copy_(torch.tensor(1.75))
            flow.log_std.copy_(torch.tensor(-0.2))

        x = torch.linspace(-6.0, 6.0, self.BATCH_SIZE)
        y = flow.encode(x)
        self.assert_invertible(flow, x, y, atol=value_atol, rtol=value_rtol)
        inverse = GaussianToBimodal()
        with torch.no_grad():
            inverse.mean.copy_(flow.mean)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, x, y, atol=value_atol, rtol=value_rtol)

    def test_gaussian_to_bimodal(self, seed: int) -> None:
        torch.manual_seed(seed)
        value_atol, value_rtol = self.VALUE_TOL
        flow = GaussianToBimodal()
        with torch.no_grad():
            flow.mean.copy_(torch.tensor(1.5))
            flow.log_std.copy_(torch.tensor(0.1))

        y = torch.linspace(-4.0, 4.0, self.BATCH_SIZE)
        x = flow.encode(y)
        self.assert_invertible(flow, y, x, atol=value_atol, rtol=value_rtol)
        inverse = BimodalToGaussian()
        with torch.no_grad():
            inverse.mean.copy_(flow.mean)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, y, x, atol=value_atol, rtol=value_rtol)

    def test_mixture_to_gaussian(self, seed: int) -> None:
        torch.manual_seed(seed)
        value_atol, value_rtol = self.VALUE_TOL
        flow = MixtureToGaussian(3)
        with torch.no_grad():
            flow.weights.copy_(torch.tensor([-0.2, 0.1, 0.5]))
            flow.means.copy_(torch.tensor([-2.0, 0.0, 1.5]))
            flow.log_std.copy_(torch.tensor([-0.4, 0.0, 0.3]))

        x = torch.linspace(-7.0, 7.0, self.BATCH_SIZE)
        y = flow.encode(x)
        self.assert_invertible(flow, x, y, atol=value_atol, rtol=value_rtol)
        inverse = GaussianToMixture(3)
        with torch.no_grad():
            inverse.weights.copy_(flow.weights)
            inverse.means.copy_(flow.means)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, x, y, atol=value_atol, rtol=value_rtol)

    def test_gaussian_to_mixture(self, seed: int) -> None:
        torch.manual_seed(seed)
        value_atol, value_rtol = self.VALUE_TOL
        flow = GaussianToMixture(3)
        with torch.no_grad():
            flow.weights.copy_(torch.tensor([0.2, -0.1, 0.7]))
            flow.means.copy_(torch.tensor([-1.0, 0.5, 2.0]))
            flow.log_std.copy_(torch.tensor([0.0, -0.3, 0.2]))

        y = torch.linspace(-4.0, 4.0, self.BATCH_SIZE)
        x = flow.encode(y)
        self.assert_invertible(flow, y, x, atol=value_atol, rtol=value_rtol)
        inverse = MixtureToGaussian(3)
        with torch.no_grad():
            inverse.weights.copy_(flow.weights)
            inverse.means.copy_(flow.means)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, y, x, atol=value_atol, rtol=value_rtol)
