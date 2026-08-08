r"""Tests for neural continuous-discrete state-space models."""

from typing import ClassVar

import pytest
import torch

from linodenet_models.ncdssm import NCDSSM, NCDSSMConfig
from linodenet_models.utils import SplitTimeData

from .base import TestProbabilisticModel


class TestNCDSSM(TestProbabilisticModel[NCDSSM]):
    r"""Run the shared probabilistic-forecasting contract against NCDSSM."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    STANDARD_CONFIG: ClassVar[NCDSSMConfig] = NCDSSMConfig(
        input_size=CONTEXT_SHAPE[0],
        output_size=OUTPUT_SHAPE[0],
        latent_size=4,
        auxiliary_size=5,
        encoder_hidden_size=7,
        decoder_hidden_size=11,
        initial_variance=1.0,
        min_variance=1e-4,
        validate_args=True,
    )

    @pytest.fixture
    def model_config(self) -> NCDSSMConfig:
        r"""Configuration used to instantiate the NCDSSM under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether roughly half of the context features are unavailable."""
        return request.param

    def make_model(self, model_config: object, /) -> NCDSSM:
        r"""Instantiate an NCDSSM from the shared standard configuration."""
        if not isinstance(model_config, NCDSSMConfig):
            raise TypeError("model_config must be an NCDSSMConfig.")
        return NCDSSM.from_config(model_config)

    def forecast(
        self, model: NCDSSM, inputs: SplitTimeData, /
    ) -> tuple[torch.Tensor, ...]:
        r"""Return NCDSSM output moments and time-marginal likelihoods."""
        assert inputs.target_values is not None
        log_prob = model.log_prob(
            inputs.target_values,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
        )
        return (
            model.pred_means,
            model.pred_variances,
            log_prob.unsqueeze(-1).expand_as(model.pred_means),
        )

    def loss(
        self,
        model: NCDSSM,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return negative predictive log-likelihood over observed targets."""
        del model, targets
        _, _, log_prob = predictions
        return -log_prob[..., 0].mean()

    def test_instantiation_from_config(self) -> None:
        model = NCDSSM.from_config(self.STANDARD_CONFIG)

        assert model.input_size == self.STANDARD_CONFIG.input_size
        assert model.output_size == self.STANDARD_CONFIG.output_size
        assert model.latent_size == self.STANDARD_CONFIG.latent_size
        assert model.auxiliary_size == self.STANDARD_CONFIG.auxiliary_size
        encoder_layer = model.encoder.network[0]
        emission_layer = model.emission.network[-1]
        assert isinstance(encoder_layer, torch.nn.Linear)
        assert isinstance(emission_layer, torch.nn.Linear)
        assert encoder_layer.in_features == 2 * model.input_size
        assert emission_layer.out_features == 2 * model.output_size

    def test_instantiation_from_parameters(self) -> None:
        config = self.STANDARD_CONFIG
        model = NCDSSM.from_parameters(
            input_size=config.input_size,
            output_size=config.output_size,
            latent_size=config.latent_size,
            auxiliary_size=config.auxiliary_size,
            encoder_hidden_size=config.encoder_hidden_size,
            decoder_hidden_size=config.decoder_hidden_size,
        )

        assert model.input_size == config.input_size
        assert model.output_size == config.output_size
        assert model.latent_size == config.latent_size
        assert model.auxiliary_size == config.auxiliary_size
