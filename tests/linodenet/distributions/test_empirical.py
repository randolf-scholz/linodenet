r"""Test Dirac and Empirical distributions."""

import pytest
import torch

from linodenet.distributions import Dirac, Empirical


@pytest.mark.parametrize("batch_shape", [(), (1,), (2,), (1, 1), (2, 2)])
@pytest.mark.parametrize("event_shape", [(), (1,), (2,), (1, 1), (2, 2)])
def test_dirac(batch_shape: tuple[int, ...], event_shape: tuple[int, ...]) -> None:
    r"""Test the Dirac distribution."""
    dataset = torch.randn(event_shape)
    samples = torch.randn((*batch_shape, *event_shape))
    dist = Dirac(dataset)

    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == batch_shape
    assert torch.all(sample_ll == -torch.inf)

    value_ll = dist.log_prob(dataset)
    assert value_ll.item() == torch.inf


@pytest.mark.parametrize(
    "event_shape", [(), (1,), (2,), (1, 1), (2, 2)], ids=lambda es: f"event_shape={es}"
)
def test_empirical(
    event_shape: tuple[int, ...],
) -> None:
    r"""Test the empirical distribution."""
    # test without batch shape
    dataset_size: int = 5
    num_samples: int = 3

    dataset = torch.randn(dataset_size, *event_shape)
    dist = Empirical(dataset)
    assert dist.event_shape == event_shape
    assert dist.batch_shape == ()

    # samples from the distribution itself
    samples = dist.sample(num_samples)
    assert samples.shape == (num_samples, *event_shape)
    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == (num_samples,)
    assert (sample_ll == torch.inf).all()

    # sample from outside the distribution
    samples = torch.randn((num_samples, *event_shape))
    assert samples.shape == (num_samples, *event_shape)
    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == (num_samples,)
    assert torch.all(sample_ll == -torch.inf)


@pytest.mark.parametrize(
    "batch_shape", [(), (1,), (2,), (1, 1), (2, 2)], ids=lambda bs: f"batch_shape={bs}"
)
@pytest.mark.parametrize(
    "event_shape", [(), (1,), (2,), (1, 1), (2, 2)], ids=lambda es: f"event_shape={es}"
)
def test_empirical_with_batch_shape(
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
) -> None:
    r"""Test the empirical distribution."""
    # test without batch shape
    dataset_size: int = 5
    num_samples: int = 3

    dataset = torch.randn(*batch_shape, dataset_size, *event_shape)
    dist = Empirical(dataset, ndim=len(event_shape))
    assert dist.event_shape == event_shape
    assert dist.batch_shape == batch_shape

    # samples from the distribution itself
    samples = dist.sample(num_samples)
    assert samples.shape == (num_samples, *batch_shape, *event_shape)
    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == (num_samples, *batch_shape)
    assert (sample_ll == torch.inf).all()

    # sample from outside the distribution
    samples = torch.randn((num_samples, *batch_shape, *event_shape))
    assert samples.shape == (num_samples, *batch_shape, *event_shape)
    sample_ll = dist.log_prob(samples)
    assert sample_ll.shape == (num_samples, *batch_shape)
    assert torch.all(sample_ll == -torch.inf)
