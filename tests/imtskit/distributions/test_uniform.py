r"""Tests for the Uniform distribution."""

import pytest
import torch
from torch.distributions import Uniform as TorchUniform

from imtskit.distributions import Uniform
from imtskit.distributions.uniform import UniformParams


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
def test_uniform_shapes(batch_shape: tuple[int, ...]) -> None:
    r"""Uniform samples and log-probabilities follow the expected shapes."""
    low = torch.zeros(batch_shape)
    high = torch.ones(batch_shape)
    dist = Uniform(low, high)

    assert dist.event_shape == ()
    assert dist.batch_shape == batch_shape
    assert isinstance(dist.params, UniformParams)
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


def test_uniform_params_match_torch() -> None:
    r"""The parameter tuple operations match torch's Uniform formulas."""
    low = torch.tensor([-1.0, 0.5, 2.0])
    high = torch.tensor([0.0, 2.5, 4.0])
    value = torch.tensor([-0.5, 1.25, 4.0])
    quantile = torch.tensor([0.25, 0.5, 0.75])
    reference = TorchUniform(low, high)
    params = UniformParams(low, high)

    samples = params.sample(5)
    assert samples.shape == (5, 3)
    assert torch.all(low <= samples)
    assert torch.all(samples < high)

    assert torch.allclose(params.mean, reference.mean)
    assert torch.allclose(params.median, reference.mean)
    with pytest.raises(NotImplementedError):
        _ = params.mode
    assert torch.allclose(params.variance, reference.variance)
    assert torch.allclose(params.stddev, reference.stddev)
    assert torch.equal(params.log_prob(value), reference.log_prob(value))
    assert torch.allclose(params.cdf(value), reference.cdf(value))
    assert torch.allclose(params.icdf(quantile), reference.icdf(quantile))
    assert torch.allclose(params.entropy(), reference.entropy())


def test_uniform_params_follow_buffers() -> None:
    r"""Parameters reference the current buffers after a dtype conversion."""
    dist = Uniform(torch.tensor(0.0, dtype=torch.float32), 1.0)
    original = dist.params
    dist.to(dtype=torch.float64)

    assert original.low.dtype == torch.float32
    assert dist.params.low is dist.low
    assert dist.params.high is dist.high
    assert dist.params.low.dtype == torch.float64
    assert dist.params.high.dtype == torch.float64
    assert dist.params.sample().dtype == torch.float64
