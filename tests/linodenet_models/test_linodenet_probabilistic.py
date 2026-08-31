from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import Tensor, nan

from linodenet.state_update import GaussianForwardUpdater, GaussianReverseUpdater
from linodenet_models.decoders import TransformSequence
from linodenet_models.linodenet_probabilistic import (
    KoopmanFilter,
    LinearGaussianFlow,
    LinodenetProbabilistic,
    make_koopman_filter,
    make_linodenet_prob,
)
from linodenet_models.profiti import Shiesh
from linodenet_models.utils import SplitTimeData

from .base import (
    TestContinuousTimeModel,
    assert_probabilistic_self_consistent,
    make_continuous_time_request,
)


class LinodenetProbabilisticTestConfig(NamedTuple):
    r"""Configuration used by shared probabilistic Linodenet tests."""

    input_size: int
    state_update: str = "forward"
    retention: float = 0.6
    retention_learnable: bool = True


class TestLinodenetProbabilistic(TestContinuousTimeModel[LinodenetProbabilistic]):
    r"""Shared forecasting-model tests for probabilistic Linodenet."""

    GRADIENT_WARMUP_STEPS = 1
    NUM_STEPS = 4

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[LinodenetProbabilisticTestConfig] = (
        LinodenetProbabilisticTestConfig(input_size=CONTEXT_SHAPE[0])
    )

    @pytest.fixture
    def model_config(self) -> LinodenetProbabilisticTestConfig:
        r"""Configuration used to instantiate the model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> LinodenetProbabilistic:
        r"""Instantiate a probabilistic Linodenet model."""
        if not isinstance(model_config, LinodenetProbabilisticTestConfig):
            raise TypeError("model_config must be a LinodenetProbabilisticTestConfig.")
        return make_linodenet_prob(
            input_size=model_config.input_size,
            state_update=model_config.state_update,
            retention=model_config.retention,
            retention_learnable=model_config.retention_learnable,
            decoder="shiesh",
        )

    def forecast(
        self, model: LinodenetProbabilistic, args: SplitTimeData
    ) -> tuple[Tensor, ...]:
        r"""Return marginal log-likelihoods for target values."""
        assert args.target_values is not None
        query_valid = args.query_mask.any(dim=-1)
        log_prob = model.log_prob(
            args.target_values,
            context_times=args.context_times,
            context_values=args.context_values,
            context_mask=args.context_mask,
            query_times=args.query_times,
            query_mask=args.query_mask,
        ).masked_fill(~query_valid, nan)

        assert log_prob.shape == args.query_times.shape
        assert log_prob[query_valid].isfinite().all()
        assert log_prob[~query_valid].isnan().all()
        return (log_prob,)

    def loss(
        self,
        model: LinodenetProbabilistic,
        predictions: tuple[Tensor, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""Return negative mean marginal log-likelihood."""
        del model
        (log_prob,) = predictions
        return -log_prob[targets.isfinite().any(dim=-1)].mean()

    def test_probabilistic_self_consistency_at_init(self) -> None:
        r"""Check self-consistency at initialization.

        p(yₜ∣H) = E_{y⁎∼p(yₜ⁎∣H)}[p(yₜ∣H⊕(t⁎, y⁎))]
        """
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
        )

        assert_probabilistic_self_consistent(
            model,
            data,
            rng=generator,
            num_futures=8192,
            num_probes=8,
            atol=3e-1,
            rtol=5e-2,
        )

    def test_probabilistic_self_consistency_trained(self) -> None:
        r"""Check self-consistency after a few training steps.

        p(yₜ∣H) = E_{y⁎∼p(yₜ⁎∣H)}[p(yₜ∣H⊕(t⁎, y⁎))]
        """
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
        )
        eval_data = make_continuous_time_request(
            rng=generator,
            batch_shape=(),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
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
            atol=3e-1,
            rtol=5e-2,
        )


def test_make_linodenet_prob_instantiates_expected_components() -> None:
    r"""The helper should wire decoder, updater, and propagator components."""
    model = make_linodenet_prob(
        input_size=3,
        state_update="reverse",
        retention=0.7,
        retention_learnable=False,
        decoder="shiesh-lowrank-shiesh",
        low_rank=2,
        batch_first=False,
        state_propagator={"use_bias": True},
    )

    assert isinstance(model, LinodenetProbabilistic)
    assert isinstance(model.decoder, TransformSequence)
    assert isinstance(model.decoder[0], Shiesh)
    assert isinstance(model.state_updater, GaussianReverseUpdater)
    assert isinstance(model.state_propagator, LinearGaussianFlow)
    assert model.batch_first is False
    assert model.state_propagator.bias is not None
    assert all(
        not parameter.requires_grad
        for parameter in model.state_updater.retention_mu.parameters()
    )
    torch.testing.assert_close(
        model.state_updater.retention_mu(None),
        torch.tensor(0.7),
    )


def test_make_linodenet_prob_selects_forward_updater() -> None:
    r"""The helper should allow selecting the forward Gaussian updater."""
    model = make_linodenet_prob(input_size=3, state_update="forward")

    assert isinstance(model.state_updater, GaussianForwardUpdater)


def test_probabilistic_sample_api_shapes() -> None:
    r"""Sampling APIs should return padded samples and per-time log-probs."""
    data = make_continuous_time_request(
        rng=0,
        batch_shape=(2,),
        min_steps=2,
        max_steps=4,
        context_shape=(3,),
    )
    model = make_linodenet_prob(input_size=3)

    samples, log_prob = model.sample_and_log_prob(
        5,
        context_times=data.context_times,
        context_values=data.context_values,
        context_mask=data.context_mask,
        query_times=data.query_times,
        query_mask=data.query_mask,
    )

    assert samples.shape == (5, *data.query_mask.shape)
    assert log_prob.shape == (5, *data.query_times.shape)
    assert samples[data.query_mask.expand_as(samples)].isfinite().all()
    assert samples[~data.query_mask.expand_as(samples)].isnan().all()
    assert log_prob[:, data.query_mask.any(dim=-1)].isfinite().all()


class TestKoopmanFilter(TestContinuousTimeModel[KoopmanFilter]):
    r"""Shared forecasting-model tests for the noisy flow-observation filter."""

    GRADIENT_WARMUP_STEPS = 1
    NUM_STEPS = 4

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[LinodenetProbabilisticTestConfig] = (
        LinodenetProbabilisticTestConfig(input_size=CONTEXT_SHAPE[0])
    )

    @pytest.fixture
    def model_config(self) -> LinodenetProbabilisticTestConfig:
        r"""Configuration used to instantiate the model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["dense", "sparse"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Exercise the masked iEKF update for sparse observations."""
        return request.param

    def make_model(self, model_config: object, /) -> KoopmanFilter:
        r"""Instantiate a high-dimensional Koopman filter."""
        if not isinstance(model_config, LinodenetProbabilisticTestConfig):
            raise TypeError("model_config must be a LinodenetProbabilisticTestConfig.")
        return make_koopman_filter(
            input_size=model_config.input_size,
            latent_size=model_config.input_size + 2,
            decoder="lowrank",
            low_rank=2,
            n_iter=1,
        )

    def forecast(self, model: KoopmanFilter, args: SplitTimeData) -> tuple[Tensor, ...]:
        r"""Return estimated per-timestamp ELBOs for target values."""
        assert args.target_values is not None
        query_valid = args.query_mask.any(dim=-1)
        bound = model.log_prob(
            args.target_values,
            context_times=args.context_times,
            context_values=args.context_values,
            context_mask=args.context_mask,
            query_times=args.query_times,
            query_mask=args.query_mask,
            num_samples=8,
        ).masked_fill(~query_valid, nan)

        assert bound.shape == args.query_times.shape
        assert bound[query_valid].isfinite().all()
        assert bound[~query_valid].isnan().all()
        return (bound,)

    def loss(
        self, model: KoopmanFilter, predictions: tuple[Tensor, ...], targets: Tensor
    ) -> Tensor:
        r"""Return the negative mean estimated ELBO."""
        del model
        (bound,) = predictions
        return -bound[targets.isfinite().any(dim=-1)].mean()


def test_koopman_filter_iekf_ignores_fully_missing_observations() -> None:
    r"""A fully missing timestamp must leave the Gaussian state unchanged."""
    model = make_koopman_filter(input_size=3, latent_size=5, decoder="lowrank")
    mean = torch.randn(2, 5)
    covariance = torch.eye(5).expand(2, 5, 5)
    values = torch.full((2, 3), nan)
    mask = torch.zeros_like(values, dtype=torch.bool)

    updated_mean, updated_covariance = model.update_iekf(
        values, (mean, covariance), mask=mask
    )

    torch.testing.assert_close(updated_mean, mean)
    torch.testing.assert_close(updated_covariance, covariance)
