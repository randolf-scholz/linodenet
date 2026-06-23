r"""Tests for neural-flow forecasting models."""

from typing import ClassVar

import pytest
import torch
from torch import nan

from linodenet.forecasting.gru_ode_bayes import GRU_Bayes, GRU_ODE_Bayes
from linodenet.forecasting.neural_flow import (
    CouplingFlow,
    FlowModelName,
    GRUFlow,
    NeuralFlow,
    NeuralFlowConfig,
    ResNetFlow,
)
from linodenet.forecasting.utils import BatchedDenseArgs

from .base import SequentialData, TestForecastingModel


class TestNeuralFlow(TestForecastingModel[NeuralFlow]):
    r"""Shared forecasting-model tests for NeuralFlow."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[NeuralFlowConfig] = NeuralFlowConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=6,
        decoder_hidden_size=8,
        feature_embedding_size=3,
        flow_model="gru",
        flow_layers=2,
        flow_hidden_layers=1,
        time_net="TimeTanh",
        time_hidden_size=4,
        invertible=True,
        bias=True,
        dropout_rate=0.0,
    )

    @pytest.fixture
    def model_config(self) -> NeuralFlowConfig:
        r"""Configuration used to instantiate the NeuralFlow model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> NeuralFlow:
        r"""Instantiate a NeuralFlow model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, NeuralFlowConfig):
            raise TypeError("model_config must be a NeuralFlowConfig.")
        return NeuralFlow.from_config(model_config)

    def forecast(
        self, model: NeuralFlow, inputs: SequentialData, /
    ) -> tuple[torch.Tensor, ...]:
        r"""Return NeuralFlow predictions for sequential forecasting inputs."""
        if not isinstance(model, NeuralFlow):
            raise TypeError("model must be a NeuralFlow.")
        dense = BatchedDenseArgs(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_values.isfinite(),
            query_times=inputs.query_times,
            query_mask=inputs.query_mask.unsqueeze(-1).expand_as(inputs.query_values),
            query_values=inputs.query_values,
        )
        combined = dense.to_combined()
        assert combined.query_values is not None
        log_prob = model.log_prob(
            combined.query_values,
            times=combined.times,
            context_values=combined.context_values,
            context_mask=combined.context_mask,
            query_mask=combined.query_mask,
        )
        posterior_mean = model.post_means
        posterior_logvar = model.post_logvars
        query_steps = combined.query_mask.any(dim=-1)
        query_mean = posterior_mean[query_steps]
        query_logvar = posterior_logvar[query_steps]
        query_log_prob = log_prob[query_steps]
        query_mean[~combined.query_mask[query_steps]] = nan
        query_logvar[~combined.query_mask[query_steps]] = nan
        pred_mean = torch.full_like(inputs.query_values, nan)
        pred_logvar = torch.full_like(inputs.query_values, nan)
        pred_log_prob = inputs.query_values.new_full(inputs.query_times.shape, nan)
        pred_mean[inputs.query_mask] = query_mean
        pred_logvar[inputs.query_mask] = query_logvar
        pred_log_prob[inputs.query_mask] = query_log_prob

        assert pred_mean.shape == inputs.query_values.shape
        assert pred_logvar.shape == inputs.query_values.shape
        assert pred_log_prob.shape == inputs.query_times.shape
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_logvar[inputs.query_mask].isfinite().all()
        assert pred_log_prob[inputs.query_mask].isfinite().all()
        return pred_mean, pred_logvar, pred_log_prob.unsqueeze(-1).expand_as(pred_mean)

    def loss(
        self,
        model: NeuralFlow,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return NeuralFlow negative log-likelihood for predictions."""
        del model
        _, _, pred_log_prob = predictions
        mask = targets.isfinite()
        return -pred_log_prob[..., 0][mask.any(dim=-1)].sum() / mask.sum()

    def test_instantiation_from_config(self) -> None:
        config = self.STANDARD_CONFIG

        model = NeuralFlow.from_config(config)

        assert isinstance(model, NeuralFlow)
        assert isinstance(model, GRU_ODE_Bayes)
        assert isinstance(model.flow, GRUFlow)
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert model.flow.input_size == config.hidden_size
        assert model.flow.num_layers == config.flow_layers
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size

    def test_instantiation_from_parameters(self) -> None:
        config = self.STANDARD_CONFIG

        model = NeuralFlow.from_parameters(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            decoder_hidden_size=config.decoder_hidden_size,
            feature_embedding_size=config.feature_embedding_size,
            flow_model=config.flow_model,
            flow_layers=config.flow_layers,
            flow_hidden_layers=config.flow_hidden_layers,
            time_net=config.time_net,
            time_hidden_size=config.time_hidden_size,
            invertible=config.invertible,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
        )

        assert isinstance(model, NeuralFlow)
        assert isinstance(model.flow, GRUFlow)
        assert model.flow.num_layers == config.flow_layers

    @pytest.mark.parametrize("size", [(), (5,), (1, 2, 3)])
    def test_sample_and_log_prob_consistent(self, size: tuple[int, ...]) -> None:
        torch.manual_seed(0)
        model = self.make_model(self.STANDARD_CONFIG)
        D = self.STANDARD_CONFIG.input_size
        times = torch.tensor([0.0, 0.5, 1.0, 1.5])
        context_values = torch.randn(4, D)
        context_mask = torch.tensor([[True] * D, [True] * D, [False] * D, [True] * D])
        context_values = context_values.masked_fill(~context_mask, nan)
        query_mask = torch.zeros(4, D, dtype=torch.bool)
        query_mask[2] = True
        query_mask[3] = True

        samples, log_prob_direct = model.sample_and_log_prob(
            size,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )
        log_prob_via_sample = model.log_prob(
            samples,
            times=times,
            context_values=context_values,
            context_mask=context_mask,
            query_mask=query_mask,
        )

        torch.testing.assert_close(log_prob_direct, log_prob_via_sample)


@pytest.mark.parametrize(
    ("flow_model", "flow_type"),
    [
        ("gru", GRUFlow),
        ("resnet", ResNetFlow),
        ("coupling", CouplingFlow),
    ],
)
def test_flow_layers(
    flow_model: FlowModelName,
    flow_type: type[GRUFlow | ResNetFlow | CouplingFlow],
) -> None:
    r"""Check that each reference neural-flow variant can forecast."""
    model = NeuralFlow.from_parameters(
        input_size=3,
        hidden_size=5,
        decoder_hidden_size=7,
        feature_embedding_size=2,
        flow_model=flow_model,
        flow_layers=2,
        flow_hidden_layers=1,
        time_net="TimeTanh",
        time_hidden_size=4,
    )
    context_times = torch.tensor([[0.0, 0.3, 0.7], [0.0, 0.5, nan]])
    context_values = torch.randn(2, 3, 3)
    context_values[1, 2] = nan
    query_times = torch.tensor([[1.0, 1.4], [0.8, nan]])
    query_mask = query_times.isfinite().unsqueeze(-1).expand(2, 2, 3)

    combined = BatchedDenseArgs(
        context_times=context_times,
        context_values=context_values,
        context_mask=context_values.isfinite(),
        query_times=query_times,
        query_mask=query_mask,
    ).to_combined()
    posterior_mean, posterior_logvar = model(
        combined.times,
        combined.context_values,
        combined.context_mask,
        combined.query_mask,
    )
    query_steps = combined.query_mask.any(dim=-1)
    query_mean = posterior_mean[query_steps]
    query_logvar = posterior_logvar[query_steps]
    query_mean[~combined.query_mask[query_steps]] = nan
    query_logvar[~combined.query_mask[query_steps]] = nan
    pred_mean = context_values.new_full((2, 2, 3), nan)
    pred_logvar = context_values.new_full((2, 2, 3), nan)
    pred_mean[query_mask.any(dim=-1)] = query_mean
    pred_logvar[query_mask.any(dim=-1)] = query_logvar

    assert isinstance(model.flow, flow_type)
    assert pred_mean.shape == (2, 2, 3)
    assert pred_logvar.shape == (2, 2, 3)
    assert pred_mean[query_mask].isfinite().all()
    assert pred_logvar[query_mask].isfinite().all()
    assert pred_mean[~query_mask].isnan().all()
    assert pred_logvar[~query_mask].isnan().all()
