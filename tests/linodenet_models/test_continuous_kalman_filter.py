r"""Tests for continuous Kalman filtering."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch.testing import assert_close

from linodenet_models.kalman_filter import (
    ContinuousTimeKalmanFilter,
    marginal_gaussian_log_prob,
    marginal_gaussian_sample,
    marginal_gaussian_sample_and_log_prob,
)
from linodenet_models.utils import SplitTimeData

from .base import (
    TestContinuousTimeModel,
    assert_probabilistic_self_consistent,
    make_continuous_time_request,
)


def test_marginal_gaussian_log_prob_matches_explicit_subvectors() -> None:
    r"""Check masked Gaussian scoring equals explicit marginal scoring."""
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    values = torch.tensor([[1.5, 10.0, -0.25], [99.0, 2.5, 10.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    actual = marginal_gaussian_log_prob(values, mean=mean, cov=cov, mask=mask)
    expected = torch.stack(
        [
            MultivariateNormal(
                mean[0, [0, 2]],
                covariance_matrix=cov[0][[0, 2]][:, [0, 2]],
            ).log_prob(values[0, [0, 2]]),
            MultivariateNormal(
                mean[1, [1]],
                covariance_matrix=cov[1][[1]][:, [1]],
            ).log_prob(values[1, [1]]),
        ]
    )

    assert_close(actual, expected)


def test_marginal_gaussian_log_prob_zero_for_empty_mask() -> None:
    r"""Check empty marginals have unit likelihood."""
    values = torch.full((2, 3), torch.nan)
    mean = torch.full((2, 3), torch.nan)
    cov = torch.full((2, 3, 3), torch.nan)
    mask = torch.zeros(2, 3, dtype=torch.bool)

    actual = marginal_gaussian_log_prob(values, mean=mean, cov=cov, mask=mask)

    assert_close(actual, torch.zeros(2))


def test_marginal_gaussian_sample_uses_masked_marginals() -> None:
    r"""Check masked Gaussian samples have sample shape and NaN padding."""
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    samples = marginal_gaussian_sample((2, 3), mean=mean, cov=cov, mask=mask)
    log_prob = marginal_gaussian_log_prob(
        samples,
        mean=mean.expand(2, 3, *mean.shape),
        cov=cov.expand(2, 3, *cov.shape),
        mask=mask.expand(2, 3, *mask.shape),
    )

    assert samples.shape == (2, 3, *mean.shape)
    assert samples[..., mask].isfinite().all()
    assert samples[..., ~mask].isnan().all()
    assert log_prob.isfinite().all()


def test_marginal_gaussian_sample_and_log_prob_reuses_samples() -> None:
    r"""Check joint sampling/scoring matches standalone masked scoring."""
    torch.manual_seed(0)
    mean = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    cov = torch.tensor(
        [
            [
                [2.0, 0.3, 0.4],
                [0.3, 1.0, 0.2],
                [0.4, 0.2, 1.5],
            ],
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.5],
                [0.2, 0.5, 3.0],
            ],
        ]
    )
    mask = torch.tensor([[True, False, True], [False, True, False]])

    samples, log_prob = marginal_gaussian_sample_and_log_prob(
        (2, 3),
        mean=mean,
        cov=cov,
        mask=mask,
    )
    expected = marginal_gaussian_log_prob(
        samples,
        mean=mean.expand(2, 3, *mean.shape),
        cov=cov.expand(2, 3, *cov.shape),
        mask=mask.expand(2, 3, *mask.shape),
    )

    assert samples.shape == (2, 3, *mean.shape)
    assert log_prob.shape == (2, 3, *mean.shape[:-1])
    assert_close(log_prob, expected)


class KalmanFilterTestConfig(NamedTuple):
    r"""Configuration used by shared continuous Kalman filter tests."""

    input_size: int
    hidden_size: int


class TestKalmanFilter(TestContinuousTimeModel[ContinuousTimeKalmanFilter]):
    r"""Shared forecasting-model tests for continuous Kalman filters."""

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

    def make_model(self, model_config: object, /) -> ContinuousTimeKalmanFilter:
        r"""Instantiate a continuous Kalman filter from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, KalmanFilterTestConfig):
            raise TypeError("model_config must be a KalmanFilterTestConfig.")

        input_size = model_config.input_size
        hidden_size = model_config.hidden_size
        return ContinuousTimeKalmanFilter(
            input_size,
            hidden_size,
            system_matrix=0.05 * torch.randn(hidden_size, hidden_size),
            observation_matrix=torch.randn(input_size, hidden_size),
            process_noise=0.2,
            measurement_noise=0.5,
            initial_mean=torch.randn(hidden_size),
            initial_covariance=2.0 * torch.eye(hidden_size),
            initial_state_learnable=True,
            process_noise_learnable=True,
            observation_noise_learnable=True,
        )

    def forecast(
        self,
        model: ContinuousTimeKalmanFilter,
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
        # assert posterior_covariance[inputs.query_mask.any(dim=-1)].isfinite().all()
        assert pred_mean[~inputs.query_mask].isnan().all()
        assert pred_cov[~inputs.query_mask].isnan().all()

        return pred_mean, pred_cov

    def loss(
        self,
        model: ContinuousTimeKalmanFilter,
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
        return mean_loss + 1e-3 * variance_loss

    def test_query_only_steps_do_not_update_latent_state(self) -> None:
        r"""Check all-missing observations leave the propagated state unchanged."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        times = torch.tensor([0.0, 0.5, 1.0])
        values = torch.full((3, self.STANDARD_CONFIG.input_size), torch.nan)
        context_mask = torch.zeros_like(values, dtype=torch.bool)
        query_mask = torch.ones_like(values, dtype=torch.bool)

        ctx_steps = context_mask.any(dim=-1)  # [F, F, F]
        q_steps = query_mask.any(dim=-1)  # [T, T, T]
        model.predict(
            query_times=times[q_steps],
            query_mask=query_mask[q_steps],
            context_times=times[ctx_steps],
            context_values=values[ctx_steps],
            context_mask=context_mask[ctx_steps],
        )

        assert_close(model.post_latent_means, model.prior_latent_means)
        assert_close(model.post_latent_covs, model.prior_latent_covs)

    def test_default_system_matrix_is_skew_symmetric(self) -> None:
        r"""Check default continuous-time dynamics are norm-preserving."""
        torch.manual_seed(0)
        model = ContinuousTimeKalmanFilter(
            self.STANDARD_CONFIG.input_size,
            self.STANDARD_CONFIG.hidden_size,
        )

        skew_residual = model.system_matrix + model.system_matrix.mT
        assert_close(skew_residual, torch.zeros_like(skew_residual))

    def test_initial_time_defaults_to_first_time_step(self) -> None:
        r"""Check default initial time matches the first time step."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        times = torch.tensor([0.0, 0.5, 1.0])
        values = torch.randn(3, self.STANDARD_CONFIG.input_size)
        context_mask = torch.ones_like(values, dtype=torch.bool)
        query_mask = torch.ones_like(values, dtype=torch.bool)

        # All steps are both context and query.
        default_mean, default_cov = model.predict(
            query_times=times,
            query_mask=query_mask,
            context_times=times,
            context_values=values,
            context_mask=context_mask,
        )
        explicit_mean, explicit_cov = model.predict(
            query_times=times,
            query_mask=query_mask,
            context_times=times,
            context_values=values,
            context_mask=context_mask,
            initial_time=times[0],
        )

        assert_close(explicit_mean, default_mean)
        assert_close(explicit_cov, default_cov)

    def test_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check Kalman log-probabilities are returned per time step."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
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
        context_times = times[ctx_steps]
        query_times = times[q_steps]
        ctx_values = context_values[ctx_steps]
        ctx_mask = context_mask[ctx_steps]
        qry_mask = query_mask[q_steps]
        qry_values = values[q_steps]

        log_prob = model.log_prob(
            qry_values,
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        mean, cov = model.predict(
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        expected = marginal_gaussian_log_prob(
            qry_values, mean=mean, cov=cov, mask=qry_mask
        )

        assert log_prob.shape == query_times.shape
        assert_close(log_prob, expected)

    def test_sample_returns_time_marginal_samples(self) -> None:
        r"""Check Kalman samples have the requested sample and query shape."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0])
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
        context_times = times[ctx_steps]
        query_times = times[q_steps]
        ctx_values = context_values[ctx_steps]
        ctx_mask = context_mask[ctx_steps]
        qry_mask = query_mask[q_steps]

        samples = model.sample(
            5,
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )

        assert samples.shape == (5, *qry_mask.shape)
        assert samples[:, qry_mask].isfinite().all()
        assert samples[:, ~qry_mask].isnan().all()

    def test_sample_and_log_prob_returns_time_marginal_likelihoods(self) -> None:
        r"""Check Kalman samples and sample log-probabilities use query masks."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        dim = self.STANDARD_CONFIG.input_size
        times = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])
        context_values = torch.randn(2, 3, dim)
        context_mask = torch.tensor(
            [
                [[True, False, True], [False, False, False], [True, True, True]],
                [[True, True, False], [True, False, True], [False, False, False]],
            ]
        )
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.tensor(
            [
                [[False, True, False], [True, False, True], [False, False, False]],
                [[False, False, False], [False, True, False], [True, True, False]],
            ]
        )

        # Both batches have exactly 2 context steps and 2 query steps.
        has_context = context_mask.any(dim=-1)  # (2, 3): [[T,F,T],[T,T,F]]
        has_query = query_mask.any(dim=-1)  # (2, 3): [[T,T,F],[F,T,T]]
        context_times = times[has_context].reshape(2, 2)
        query_times = times[has_query].reshape(2, 2)
        ctx_values = context_values[has_context].reshape(2, 2, dim)
        ctx_mask = context_mask[has_context].reshape(2, 2, dim)
        qry_mask = query_mask[has_query].reshape(2, 2, dim)

        samples, log_prob = model.sample_and_log_prob(
            (2, 3),
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        expected = marginal_gaussian_log_prob(
            samples,
            mean=model.pred_means.expand(2, 3, *model.pred_means.shape),
            cov=model.pred_covs.expand(2, 3, *model.pred_covs.shape),
            mask=qry_mask.expand(2, 3, *qry_mask.shape),
        )

        assert samples.shape == (2, 3, 2, 2, dim)
        assert log_prob.shape == (2, 3, 2, 2)
        assert samples[..., qry_mask].isfinite().all()
        assert samples[..., ~qry_mask].isnan().all()
        assert_close(log_prob, expected)

    @pytest.mark.parametrize("size", [(), (5,), (1, 2, 3)])
    def test_sample_and_log_prob_consistent(self, size: tuple[int, ...]) -> None:
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        D = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_values = torch.randn(4, D)
        context_mask = torch.tensor([[True] * D, [True] * D, [False] * D, [True] * D])
        context_values = context_values.masked_fill(~context_mask, torch.nan)
        query_mask = torch.zeros(4, D, dtype=torch.bool)
        query_mask[2] = True
        query_mask[3] = True

        ctx_steps = context_mask.any(dim=-1)  # [T, T, F, T]
        q_steps = query_mask.any(dim=-1)  # [F, F, T, T]
        context_times = times[ctx_steps]
        query_times = times[q_steps]
        ctx_values = context_values[ctx_steps]
        ctx_mask = context_mask[ctx_steps]
        qry_mask = query_mask[q_steps]

        samples, log_prob_direct = model.sample_and_log_prob(
            size,
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )
        log_prob_via_sample = model.log_prob(
            samples,
            query_times=query_times,
            query_mask=qry_mask,
            context_times=context_times,
            context_values=ctx_values,
            context_mask=ctx_mask,
        )

        assert_close(log_prob_direct, log_prob_via_sample)

    def test_probabilistic_self_consistency_at_init(self) -> None:
        r"""Check self-consistency at initialization."""
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        generator = torch.Generator()
        data = make_continuous_time_request(
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

        train_data = make_continuous_time_request(
            rng=generator,
            batch_shape=(4,),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
        )
        eval_data = make_continuous_time_request(
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
