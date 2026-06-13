r"""Tests for neural-flow forecasting models."""

from typing import ClassVar

import pytest
import torch

from linodenet.forecasting.gru_ode_bayes import GRU_Bayes, GRU_ODE_Bayes
from linodenet.forecasting.neural_flow import (
    CouplingFlow,
    FlowModelName,
    GRUFlow,
    NeuralFlow,
    NeuralFlowConfig,
    ResNetFlow,
)

from .base import TestForecastingModel


class TestModel(TestForecastingModel[NeuralFlow]):
    r"""Shared forecasting-model tests for NeuralFlow."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (4,)
    BATCH_SHAPE: ClassVar[tuple[int, ...]] = (4,)
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

    def loss(
        self,
        model: NeuralFlow,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return NeuralFlow negative log-likelihood for predictions."""
        pred_mean, pred_logvar = predictions
        mask = targets.isfinite()
        return (
            model.nll_logvar(targets, pred_mean, pred_logvar, mask).sum() / mask.sum()
        )

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


@pytest.mark.parametrize(
    ("flow_model", "flow_type"),
    [
        ("gru", GRUFlow),
        ("resnet", ResNetFlow),
        ("coupling", CouplingFlow),
    ],
)
def test_neural_flow_variants(
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
    context_times = torch.tensor([[0.0, 0.3, 0.7], [0.0, 0.5, torch.nan]])
    context_values = torch.randn(2, 3, 3)
    context_values[1, 2] = torch.nan
    query_times = torch.tensor([[1.0, 1.4], [0.8, torch.nan]])

    pred_mean, pred_logvar = model(query_times, context_times, context_values)
    query_mask = query_times.isfinite()

    assert isinstance(model.flow, flow_type)
    assert pred_mean.shape == (2, 2, 3)
    assert pred_logvar.shape == (2, 2, 3)
    assert pred_mean[query_mask].isfinite().all()
    assert pred_logvar[query_mask].isfinite().all()
    assert pred_mean[~query_mask].isnan().all()
    assert pred_logvar[~query_mask].isnan().all()
