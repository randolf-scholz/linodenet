r"""Tests for the univariate Normal distribution."""

import pytest
import torch
from torch.distributions import Normal as TorchNormal

from imtskit.distributions import Normal


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
def test_normal_shapes(batch_shape: tuple[int, ...]) -> None:
    r"""Normal samples and log-probabilities follow the expected shapes."""
    loc = torch.zeros(batch_shape)
    scale = torch.ones(batch_shape)
    dist = Normal(loc, scale)

    assert dist.event_shape == ()
    assert dist.batch_shape == batch_shape
    assert isinstance(dist.params, Normal.Params)
    params_loc, params_scale = dist.params
    assert torch.allclose(params_loc, loc)
    assert torch.allclose(params_scale, scale)

    samples = dist.sample(3)
    assert samples.shape == (3, *batch_shape)

    log_prob = dist.log_prob(samples)
    assert log_prob.shape == (3, *batch_shape)


def test_normal_matches_torch() -> None:
    r"""The local implementation matches torch's Normal formulas."""
    loc = torch.tensor([-1.0, 0.5, 2.0])
    scale = torch.tensor([0.5, 1.25, 2.0])
    value = torch.tensor([-0.5, 1.25, 4.0])
    quantile = torch.tensor([0.25, 0.5, 0.75])

    dist = Normal(loc, scale)
    reference = TorchNormal(loc, scale)

    assert torch.allclose(dist.mean, reference.mean)
    assert torch.allclose(dist.median, reference.mean)
    assert torch.allclose(dist.mode, reference.mode)
    assert torch.allclose(dist.variance, reference.variance)
    assert torch.allclose(dist.stddev, reference.stddev)
    assert torch.allclose(dist.cdf(value), reference.cdf(value))
    assert torch.allclose(dist.icdf(quantile), reference.icdf(quantile))
    assert torch.allclose(dist.log_prob(value), reference.log_prob(value))
    assert torch.allclose(dist.entropy(), reference.entropy())


def test_normal_rejects_invalid_scale() -> None:
    r"""The scale parameter must satisfy $scale > 0$ elementwise."""
    with pytest.raises(ValueError, match="scale > 0"):
        Normal(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))


def test_normal_params_match_torch() -> None:
    r"""The parameter tuple operations match torch's Normal formulas."""
    loc = torch.tensor([-1.0, 0.5, 2.0])
    scale = torch.tensor([0.5, 1.25, 2.0])
    value = torch.tensor([-0.5, 1.25, 4.0])
    quantile = torch.tensor([0.25, 0.5, 0.75])
    reference = TorchNormal(loc, scale)
    params = Normal.Params(loc, scale)

    samples = params.sample(5)
    assert samples.shape == (5, 3)

    assert torch.allclose(params.mean, reference.mean)
    assert torch.allclose(params.median, reference.mean)
    assert torch.allclose(params.mode, reference.mode)
    assert torch.allclose(params.variance, reference.variance)
    assert torch.allclose(params.stddev, reference.stddev)
    assert torch.allclose(params.log_prob(value), reference.log_prob(value))
    assert torch.allclose(params.cdf(value), reference.cdf(value))
    assert torch.allclose(params.icdf(quantile), reference.icdf(quantile))
    assert torch.allclose(params.entropy(), reference.entropy())


def test_normal_params_follow_buffers() -> None:
    r"""Parameters reference the current buffers after a dtype conversion."""
    dist = Normal(torch.tensor(0.0, dtype=torch.float32), 1.0)
    original = dist.params
    dist.to(dtype=torch.float64)

    assert original.loc.dtype == torch.float32
    assert dist.params.loc is dist.loc
    assert dist.params.scale is dist.scale
    assert dist.params.loc.dtype == torch.float64
    assert dist.params.scale.dtype == torch.float64
    assert dist.params.sample().dtype == torch.float64
