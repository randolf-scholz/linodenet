r"""Tests for the Uniform distribution."""

import pytest
import torch
from torch.distributions import Uniform as TorchUniform

from linodenet.distributions import Uniform
from linodenet.distributions.uniform import (
    uniform_cdf,
    uniform_entropy,
    uniform_icdf,
    uniform_log_prob,
    uniform_mean,
    uniform_median,
    uniform_mode,
    uniform_sample,
    uniform_stddev,
    uniform_variance,
)


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
def test_uniform_shapes(batch_shape: tuple[int, ...]) -> None:
    r"""Uniform samples and log-probabilities follow the expected shapes."""
    low = torch.zeros(batch_shape)
    high = torch.ones(batch_shape)
    dist = Uniform(low, high)

    assert dist.event_shape == ()
    assert dist.batch_shape == batch_shape
    params_low, params_high = dist.params
    assert torch.allclose(params_low, low)
    assert torch.allclose(params_high, high)

    samples = dist.sample(3)
    assert samples.shape == (3, *batch_shape)
    assert torch.all(low <= samples)
    assert torch.all(samples < high)

    log_prob = dist.log_prob(samples)
    assert log_prob.shape == (3, *batch_shape)
    assert torch.allclose(log_prob, torch.zeros_like(log_prob))


def test_uniform_matches_torch() -> None:
    r"""The local implementation matches torch's Uniform formulas."""
    low = torch.tensor([-1.0, 0.5, 2.0])
    high = torch.tensor([0.0, 2.5, 4.0])
    value = torch.tensor([-0.5, 1.25, 4.0])
    quantile = torch.tensor([0.25, 0.5, 0.75])

    dist = Uniform(low, high)
    reference = TorchUniform(low, high)

    assert torch.allclose(dist.mean, reference.mean)
    assert torch.allclose(dist.variance, reference.variance)
    assert torch.allclose(dist.stddev, reference.stddev)
    assert torch.allclose(dist.cdf(value), reference.cdf(value))
    assert torch.allclose(dist.icdf(quantile), reference.icdf(quantile))
    assert torch.equal(dist.log_prob(value), reference.log_prob(value))
    assert torch.allclose(dist.entropy(), reference.entropy())


def test_uniform_rejects_invalid_bounds() -> None:
    r"""The interval bounds must satisfy $low < high$ elementwise."""
    with pytest.raises(ValueError, match="low < high"):
        Uniform(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))


def test_uniform_functional_api() -> None:
    r"""The functional helpers match torch's Uniform formulas."""
    low = torch.tensor([-1.0, 0.5, 2.0])
    high = torch.tensor([0.0, 2.5, 4.0])
    value = torch.tensor([-0.5, 1.25, 4.0])
    quantile = torch.tensor([0.25, 0.5, 0.75])
    reference = TorchUniform(low, high)
    params = (low, high)

    samples = uniform_sample(params, 5)
    assert samples.shape == (5, 3)
    assert torch.all(low <= samples)
    assert torch.all(samples < high)

    assert torch.allclose(uniform_mean(params), reference.mean)
    assert torch.allclose(uniform_median(params), reference.mean)
    with pytest.raises(NotImplementedError):
        uniform_mode(params)
    assert torch.allclose(uniform_variance(params), reference.variance)
    assert torch.allclose(uniform_stddev(params), reference.stddev)
    assert torch.equal(uniform_log_prob(params, value), reference.log_prob(value))
    assert torch.allclose(uniform_cdf(params, value), reference.cdf(value))
    assert torch.allclose(uniform_icdf(params, quantile), reference.icdf(quantile))
    assert torch.allclose(uniform_entropy(params), reference.entropy())
