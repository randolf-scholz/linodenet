import math
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nan

from linodenet.state_update import GaussianForwardUpdater, GaussianReverseUpdater
from linodenet_models.decoders import TransformSequence
from linodenet_models.linodenet_probabilistic import (
    LinearGaussianFlow,
    LinodenetProbabilistic,
    make_linodenet_prob,
)
from linodenet_models.profiti import Shiesh
from linodenet_models.utils import SplitTimeData

from .base import TestContinuousTimeModel, make_continuous_time_request


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
        self,
        model: LinodenetProbabilistic,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return marginal log-likelihoods for target values."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)
        log_prob = model.log_prob(
            inputs.target_values,
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
        ).masked_fill(~query_valid, nan)

        assert log_prob.shape == inputs.query_times.shape
        assert log_prob[query_valid].isfinite().all()
        assert log_prob[~query_valid].isnan().all()
        return (log_prob,)

    def loss(
        self,
        model: LinodenetProbabilistic,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return negative mean marginal log-likelihood."""
        del model
        (log_prob,) = predictions
        return -log_prob[targets.isfinite().any(dim=-1)].mean()

    def assert_probabilistic_self_consistent(
        self,
        model: LinodenetProbabilistic,
        /,
        *,
        seed: int,
        num_futures: int = 8192,
        num_probe: int = 8,
    ) -> None:
        r"""Check self-consistency by Monte Carlo marginalization over y⁎."""
        data = make_continuous_time_request(
            seed=seed,
            batch_shape=(),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
        )

        with torch.no_grad():
            # S=num_futures, P=num_probe

            # 1. Fix t⁎ and t, then sample y⁎∼p(yₜ⁎∣H) and probes yₜ.
            query_times = data.query_times[:2]
            query_mask = data.query_mask[:2]
            torch.manual_seed(1)
            futures = model.sample(  # (S, D)
                num_futures,
                query_times=query_times[:1],
                query_mask=query_mask[:1],
                context_times=data.context_times,
                context_values=data.context_values,
                context_mask=data.context_mask,
            )[:, 0]
            y_probe = model.sample(  # (P, 1, D)
                num_probe,
                query_times=query_times[1:2],
                query_mask=query_mask[1:2],
                context_times=data.context_times,
                context_values=data.context_values,
                context_mask=data.context_mask,
            )

            # 2. Score probes under the original predictive law p(yₜ∣H).
            base_log_prob = model.log_prob(  # (P,)
                y_probe,
                query_times=query_times[1:2],
                query_mask=query_mask[1:2],
                context_times=data.context_times,
                context_values=data.context_values,
                context_mask=data.context_mask,
            )[:, 0]

            # 3. Append each sampled y⁎ to H as a hypothetical observation.
            updated_context_times = torch.cat(  # (S, N+1)
                [
                    data.context_times.expand(num_futures, -1),
                    query_times[:1].expand(num_futures, 1),
                ],
                dim=-1,
            )
            updated_context_values = torch.cat(  # (S, N+1, D)
                [
                    data.context_values.expand(num_futures, -1, -1),
                    futures[:, None, :],
                ],
                dim=-2,
            )
            updated_context_mask = torch.cat(  # (S, N+1, D)
                [
                    data.context_mask.expand(num_futures, -1, -1),
                    query_mask[:1].expand(num_futures, 1, -1),
                ],
                dim=-2,
            )

            # 4. Score p(yₜ∣H⊕(t⁎, y⁎)) for every probe and sampled y⁎.
            conditional_log_prob = model.log_prob(  # (P, S)
                y_probe[:, None].expand(num_probe, num_futures, 1, -1),
                query_times=query_times[1:].expand(num_futures, 1),
                query_mask=query_mask[1:].expand(num_futures, 1, -1),
                context_times=updated_context_times,
                context_values=updated_context_values,
                context_mask=updated_context_mask,
            )[:, :, 0]

            # 5. Average densities in log-space over y⁎ samples.
            mixture_log_prob = (  # (P,)
                torch.logsumexp(conditional_log_prob, dim=-1) - math.log(num_futures)
            )

        single_future_error = (conditional_log_prob - base_log_prob[:, None]).abs()
        assert single_future_error.max() > 1.0
        torch.testing.assert_close(
            mixture_log_prob,
            base_log_prob,
            atol=3e-2,
            rtol=1e-2,
        )

    def test_probabilistic_self_consistency_at_init(self) -> None:
        r"""Check self-consistency at initialization.

        p(yₜ∣H) = E_{y⁎∼p(yₜ⁎∣H)}[p(yₜ∣H⊕(t⁎, y⁎))]
        """
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)

        self.assert_probabilistic_self_consistent(model, seed=5)

    def test_probabilistic_self_consistency_trained(self) -> None:
        r"""Check self-consistency after a few training steps.

        p(yₜ∣H) = E_{y⁎∼p(yₜ⁎∣H)}[p(yₜ∣H⊕(t⁎, y⁎))]
        """
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        train_data = make_continuous_time_request(
            seed=2,
            batch_shape=(4,),
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

        self.assert_probabilistic_self_consistent(model, seed=3)


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
        seed=0,
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
