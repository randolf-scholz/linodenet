import pytest
import torch
from torch import Tensor

from linodenet.mappings.transforms import (
    BimodalToGaussian,
    GaussianToBimodal,
    GaussianToMixture,
    MixtureToGaussian,
)
from tests.testing import SEEDS_5, as_torch_generator

from .test_transform import TestTransform


@pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
class TestGaussianTransportFlow(TestTransform):
    VALUE_TOL = (1e-5, 1e-5)
    NUM_STEPS = 128
    NUM_COMPONENTS = 3
    TEST_RANGE = (-5.0, 5.0)

    def make_bimodal_test_case(
        self, *, rng: int | torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor]:
        generator = as_torch_generator(rng)
        start, end = self.TEST_RANGE
        values = torch.linspace(start, end, self.NUM_STEPS)
        mean = 3 * torch.rand((), generator=generator) - 1.5
        log_std = torch.rand((), generator=generator) - 0.5
        return values, mean, log_std

    def make_mixture_test_case(
        self, *, rng: int | torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        generator = as_torch_generator(rng)
        start, end = self.TEST_RANGE
        values = torch.linspace(start, end, self.NUM_STEPS)
        weights = torch.randn(self.NUM_COMPONENTS, generator=generator)
        means = 4 * torch.rand(self.NUM_COMPONENTS, generator=generator) - 2
        log_std = torch.rand(self.NUM_COMPONENTS, generator=generator) - 0.5
        return values, weights, means, log_std

    def test_bimodal_to_gaussian(self, *, seed: int) -> None:
        value_atol, value_rtol = self.VALUE_TOL
        x, mean, log_std = self.make_bimodal_test_case(rng=seed)
        flow = BimodalToGaussian()
        with torch.no_grad():
            flow.mean.copy_(mean)
            flow.log_std.copy_(log_std)

        y = flow.encode(x)
        self.assert_invertible(flow, x, y, atol=value_atol, rtol=value_rtol)
        inverse = GaussianToBimodal()
        with torch.no_grad():
            inverse.mean.copy_(flow.mean)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, x, y, atol=value_atol, rtol=value_rtol)

    def test_gaussian_to_bimodal(self, *, seed: int) -> None:
        value_atol, value_rtol = self.VALUE_TOL
        y, mean, log_std = self.make_bimodal_test_case(rng=seed)
        flow = GaussianToBimodal()
        with torch.no_grad():
            flow.mean.copy_(mean)
            flow.log_std.copy_(log_std)

        x = flow.encode(y)
        self.assert_invertible(flow, y, x, atol=value_atol, rtol=value_rtol)
        inverse = BimodalToGaussian()
        with torch.no_grad():
            inverse.mean.copy_(flow.mean)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, y, x, atol=value_atol, rtol=value_rtol)

    def test_mixture_to_gaussian(self, seed: int) -> None:
        value_atol, value_rtol = self.VALUE_TOL
        x, weights, means, log_std = self.make_mixture_test_case(rng=seed)
        flow = MixtureToGaussian(self.NUM_COMPONENTS)
        with torch.no_grad():
            flow.weights.copy_(weights)
            flow.means.copy_(means)
            flow.log_std.copy_(log_std)

        y = flow.encode(x)
        self.assert_invertible(flow, x, y, atol=value_atol, rtol=value_rtol)
        inverse = GaussianToMixture(self.NUM_COMPONENTS)
        with torch.no_grad():
            inverse.weights.copy_(flow.weights)
            inverse.means.copy_(flow.means)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, x, y, atol=value_atol, rtol=value_rtol)

    def test_gaussian_to_mixture(self, *, seed: int) -> None:
        value_atol, value_rtol = self.VALUE_TOL
        y, weights, means, log_std = self.make_mixture_test_case(rng=seed)
        flow = GaussianToMixture(self.NUM_COMPONENTS)
        with torch.no_grad():
            flow.weights.copy_(weights)
            flow.means.copy_(means)
            flow.log_std.copy_(log_std)

        x = flow.encode(y)
        self.assert_invertible(flow, y, x, atol=value_atol, rtol=value_rtol)
        inverse = MixtureToGaussian(self.NUM_COMPONENTS)
        with torch.no_grad():
            inverse.weights.copy_(flow.weights)
            inverse.means.copy_(flow.means)
            inverse.log_std.copy_(flow.log_std)
        self.assert_dual(flow, inverse, y, x, atol=value_atol, rtol=value_rtol)
