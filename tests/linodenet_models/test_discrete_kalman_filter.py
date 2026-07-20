r"""Tests for discrete Kalman filtering."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet_models import DiscreteTimeKalmanFilter
from linodenet_models.kalman_filter import (
    marginal_gaussian_log_prob,
)
from linodenet_models.utils import SplitTimeData

from .base import TestDiscreteTimeModel, assert_probabilistic_self_consistent


def make_model() -> DiscreteTimeKalmanFilter:
    r"""Instantiate a small stable discrete Kalman filter."""
    torch.manual_seed(0)
    return DiscreteTimeKalmanFilter(
        3,
        5,
        system_matrix=0.1 * torch.randn(5, 5),
        observation_matrix=torch.randn(3, 5),
        process_covariance=0.2,
        measurement_covariance=0.5,
        initial_mean=torch.randn(5),
        initial_covariance=2.0,
    )


def test_forward_returns_posterior_latent_states() -> None:
    r"""Check forward filters the event stream and yields latent posteriors."""
    model = make_model()
    steps = torch.tensor([0, 1, 3])
    context_values = torch.randn(3, 3)
    context_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
            [True, False, True],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_mask = torch.zeros_like(context_mask)

    means, covs = model.forward(
        steps=steps,
        context_values=context_values,
        context_mask=context_mask,
        query_mask=query_mask,
    )

    assert means.shape == (3, 5)
    assert covs.shape == (3, 5, 5)
    assert means[context_mask.any(dim=-1)].isfinite().all()
    assert covs[context_mask.any(dim=-1)].isfinite().all()
    assert means[~context_mask.any(dim=-1)].isnan().all()
    assert covs[~context_mask.any(dim=-1)].isnan().all()
    assert_close(means, model.posterior_latent_means, equal_nan=True)
    assert_close(covs, model.posterior_latent_covariances, equal_nan=True)


def test_predict_returns_query_marginals() -> None:
    r"""Check predict decodes posterior query states in observation space."""
    model = make_model()
    context_steps = torch.tensor([0, 1, 2])
    context_values = torch.randn(3, 3)
    context_mask = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, False, False],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_steps = torch.tensor([1, 3])
    query_mask = torch.tensor([[False, True, False], [True, False, True]])

    mean, cov = model.predict(
        query_times=query_steps,
        query_mask=query_mask,
        context_times=context_steps,
        context_values=context_values,
        context_mask=context_mask,
    )

    assert mean.shape == (2, 3)
    assert cov.shape == (2, 3, 3)
    assert mean.isfinite().all()
    assert cov.isfinite().all()
    assert_close(mean, model.pred_means)
    assert_close(cov, model.pred_covs)


def test_sample_and_log_prob_consistent() -> None:
    r"""Check joint sampling/scoring matches standalone masked scoring."""
    model = make_model()
    context_steps = torch.tensor([[0, 1, 2], [0, 2, 0]])
    context_values = torch.randn(2, 3, 3)
    context_mask = torch.tensor(
        [
            [[True, False, True], [False, False, False], [True, True, True]],
            [[True, True, False], [True, False, True], [False, False, False]],
        ]
    )
    context_values = context_values.masked_fill(~context_mask, torch.nan)
    query_steps = torch.tensor([[1, 3], [2, 4]])
    query_mask = torch.tensor(
        [
            [[False, True, False], [True, False, True]],
            [[False, True, False], [True, True, False]],
        ]
    )

    samples, log_prob = model.sample_and_log_prob(
        (2, 3),
        query_times=query_steps,
        query_mask=query_mask,
        context_times=context_steps,
        context_values=context_values,
        context_mask=context_mask,
    )
    expected = marginal_gaussian_log_prob(
        samples,
        mean=model.pred_means.expand(2, 3, *model.pred_means.shape),
        cov=model.pred_covs.expand(2, 3, *model.pred_covs.shape),
        mask=query_mask.expand(2, 3, *query_mask.shape),
    )

    assert samples.shape == (2, 3, 2, 2, 3)
    assert log_prob.shape == (2, 3, 2, 2)
    assert samples[..., query_mask].isfinite().all()
    assert samples[..., ~query_mask].isnan().all()
    assert_close(log_prob, expected)


def test_predict_rejects_float_times() -> None:
    r"""Check the discrete API rejects continuous-time inputs."""
    model = make_model()
    context_values = torch.randn(2, 3)
    context_mask = torch.ones_like(context_values, dtype=torch.bool)
    query_mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(AssertionError):
        model.predict(
            query_times=torch.tensor([2.0]),
            query_mask=query_mask,
            context_times=torch.tensor([0.0, 1.0]),
            context_values=context_values,
            context_mask=context_mask,
        )


class KalmanFilterTestConfig(NamedTuple):
    r"""Configuration used by shared discrete Kalman filter tests."""

    input_size: int
    hidden_size: int


class TestDiscreteKalmanFilter(TestDiscreteTimeModel[DiscreteTimeKalmanFilter]):
    r"""Shared forecasting-model tests for discrete Kalman filters."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[KalmanFilterTestConfig] = KalmanFilterTestConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=5,
    )

    @pytest.fixture
    def model_config(self) -> KalmanFilterTestConfig:
        r"""Configuration used to instantiate the Kalman filter under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> DiscreteTimeKalmanFilter:
        r"""Instantiate a discrete Kalman filter from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, KalmanFilterTestConfig):
            raise TypeError("model_config must be a KalmanFilterTestConfig.")

        input_size = model_config.input_size
        hidden_size = model_config.hidden_size
        return DiscreteTimeKalmanFilter(
            input_size,
            hidden_size,
            system_matrix=0.1 * torch.randn(hidden_size, hidden_size),
            observation_matrix=torch.randn(input_size, hidden_size),
            process_covariance=0.2,
            measurement_covariance=0.5,
            initial_mean=torch.randn(hidden_size),
            initial_covariance=2.0,
            learnable=True,
        )

    def forecast(
        self,
        model: DiscreteTimeKalmanFilter,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return Kalman filter predictions for sequential forecasting inputs."""
        assert inputs.target_values is not None
        pred_mean, pred_cov = model.predict(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
        )

        *batch_shape, query_size, query_dim = inputs.target_values.shape
        assert pred_mean.shape == (*batch_shape, query_size, query_dim)
        assert pred_cov.shape == (*batch_shape, query_size, query_dim, query_dim)
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_cov[inputs.query_mask].isfinite().all()
        assert pred_mean[~inputs.query_mask].isnan().all()
        assert pred_cov[~inputs.query_mask].isnan().all()

        return pred_mean, pred_cov

    def loss(
        self,
        model: DiscreteTimeKalmanFilter,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return a supervised loss for Kalman filter predictions."""
        del model
        pred_mean, pred_variance = predictions
        mask = targets.isfinite().all(dim=-1)
        mean_loss = F.mse_loss(pred_mean[mask], targets[mask])
        variance_loss = F.mse_loss(
            pred_variance[mask],
            torch.ones_like(pred_variance[mask]),
        )
        return 100.0 * (mean_loss + 1e-3 * variance_loss)

    def test_query_only_steps_do_not_update_latent_state(self) -> None:
        r"""Check all-missing observations leave the propagated state unchanged."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        steps = torch.tensor([0, 1, 3])
        values = torch.full((3, self.STANDARD_CONFIG.input_size), torch.nan)
        context_mask = torch.zeros_like(values, dtype=torch.bool)
        query_mask = torch.ones_like(values, dtype=torch.bool)

        ctx_steps = context_mask.any(dim=-1)  # [F, F, F]
        q_steps = query_mask.any(dim=-1)  # [T, T, T]
        model.predict(
            query_times=steps[q_steps],
            query_mask=query_mask[q_steps],
            context_times=steps[ctx_steps],
            context_values=values[ctx_steps],
            context_mask=context_mask[ctx_steps],
        )

        assert_close(model.posterior_latent_means, model.prior_latent_means)
        assert_close(model.posterior_latent_covariances, model.prior_latent_covariances)

    def test_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check Kalman log-probabilities are returned per step."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        steps = torch.tensor([0, 1, 2, 4])
        context_values = torch.randn(4, dim)
        context_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [False, False, False],
                [True, True, True],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [False, False, False],
                [False, True, False],
                [True, False, True],
                [False, False, False],
            ]
        )
        values = torch.randn(4, dim).masked_fill(~query_mask, torch.nan)

        ctx_steps = context_mask.any(dim=-1)  # [T, T, F, T]
        q_steps = query_mask.any(dim=-1)  # [F, T, T, F]
        context_steps = steps[ctx_steps]
        query_steps = steps[q_steps]
        ctx_values = context_values[ctx_steps]
        ctx_mask = context_mask[ctx_steps]
        qry_mask = query_mask[q_steps]
        qry_values = values[q_steps]

        log_prob = model.log_prob(
            qry_values,
            query_times=query_steps,
            query_mask=qry_mask,
            context_times=context_steps,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        mean, cov = model.predict(
            query_times=query_steps,
            query_mask=qry_mask,
            context_times=context_steps,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        expected = marginal_gaussian_log_prob(
            qry_values, mean=mean, cov=cov, mask=qry_mask
        )

        assert log_prob.shape == query_steps.shape
        assert_close(log_prob, expected)

    def test_sample_returns_time_marginal_samples(self) -> None:
        r"""Check Kalman samples have the requested sample and query shape."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        steps = torch.tensor([0, 1, 3])
        context_values = torch.randn(3, dim)
        context_mask = torch.tensor(
            [
                [True, False, True],
                [False, False, False],
                [True, True, False],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
                [False, False, False],
            ]
        )

        ctx_steps = context_mask.any(dim=-1)  # [T, F, T]
        q_steps = query_mask.any(dim=-1)  # [T, T, F]
        context_steps = steps[ctx_steps]
        query_steps = steps[q_steps]
        ctx_values = context_values[ctx_steps]
        ctx_mask = context_mask[ctx_steps]
        qry_mask = query_mask[q_steps]

        samples = model.sample(
            5,
            query_times=query_steps,
            query_mask=qry_mask,
            context_times=context_steps,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )

        assert samples.shape == (5, *qry_mask.shape)
        assert samples[:, qry_mask].isfinite().all()
        assert samples[:, ~qry_mask].isnan().all()

    def test_probabilistic_self_consistency_at_init(self) -> None:
        r"""Check self-consistency at initialization."""
        generator = torch.Generator()
        torch.manual_seed(0)

        model = self.make_model(self.STANDARD_CONFIG)
        data = self.make_request(
            rng=generator,
            batch_shape=(),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
        )

        assert_probabilistic_self_consistent(
            model,
            data,
            rng=generator,
            num_futures=8192,
            num_probes=8,
            atol=3e-2,
            rtol=1e-2,
        )

    def test_probabilistic_self_consistency_trained(self) -> None:
        r"""Check self-consistency after a few training steps."""
        generator = torch.Generator()
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)

        train_data = self.make_request(
            rng=generator,
            batch_shape=(4,),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
        )
        eval_data = self.make_request(
            rng=generator,
            batch_shape=(),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
        )

        assert train_data.target_values is not None
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(self.NUM_STEPS):
            optimizer.zero_grad()
            predictions = self.forecast(model, train_data)
            loss = self.loss(model, predictions, train_data.target_values)
            loss.backward()
            optimizer.step()

        assert_probabilistic_self_consistent(
            model,
            eval_data,
            rng=generator,
            num_futures=8192,
            num_probes=8,
            atol=3e-2,
            rtol=1e-2,
        )
