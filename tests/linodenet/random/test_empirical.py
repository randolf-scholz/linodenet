r"""Test Dirac and Empirical distributions."""

import pytest
import torch

from linodenet.random.distributions.empirical import Dirac, Empirical


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2,), (1, 1), (2, 2)])
def test_dirac(batch_shape: tuple[int, ...], event_shape: tuple[int, ...]) -> None:
    r"""Test the Dirac distribution."""
    value = torch.randn(event_shape)
    samples = torch.randn((*batch_shape, *event_shape))
    dist = Dirac(value)

    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == batch_shape
    assert torch.all(sample_ll == -torch.inf)

    value_ll = dist.log_prob(value)
    assert value_ll.item() == torch.inf


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2,), (1, 1), (2, 2)])
@pytest.mark.parametrize("num_samples", [1, 2])
def test_empirical(
    batch_shape: tuple[int, ...], event_shape: tuple[int, ...], num_samples: int
) -> None:
    r"""Test the empirical distribution."""
    value = torch.randn(num_samples, *event_shape)
    samples = torch.randn((*batch_shape, *event_shape))
    dist = Empirical(value)

    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == batch_shape
    assert torch.all(sample_ll == -torch.inf)

    value_ll = dist.log_prob(value)
    assert value_ll.shape == (num_samples,)
    assert (value_ll == torch.inf).all()
