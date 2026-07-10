r"""Tests for LinODEnet model construction helpers."""

from types import MappingProxyType
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nan, nn
from torch.nn import functional as F

from linodenet_models.linodenet import LinearFlow, LinODEnet, make_linodenet
from linodenet_models.state_update import GradientStepUpdater, LpLoss
from linodenet_models.utils import SplitTimeData

from .base import TestForecastingModel, make_forecasting_request


class LinODEnetTestConfig(NamedTuple):
    r"""Configuration used by shared LinODEnet forecasting-model tests."""

    input_size: int
    latent_size: int


class TestLinODEnet(TestForecastingModel[LinODEnet]):
    r"""Shared forecasting-model tests for LinODEnet."""

    GRADIENT_WARMUP_STEPS = 1

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[LinODEnetTestConfig] = LinODEnetTestConfig(
        input_size=CONTEXT_SHAPE[0],
        latent_size=4,
    )

    @pytest.fixture
    def model_config(self) -> LinODEnetTestConfig:
        r"""Configuration used to instantiate the LinODEnet model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> LinODEnet:
        r"""Instantiate a LinODEnet model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, LinODEnetTestConfig):
            raise TypeError("model_config must be a LinODEnetTestConfig.")
        return make_linodenet(
            linodenet={
                "input_size": model_config.input_size,
                "latent_size": model_config.latent_size,
            },
            decoder={
                "in_features": model_config.latent_size,
                "out_features": model_config.input_size,
            },
            state_propagator={
                "input_size": model_config.latent_size,
                "kernel_initialization": "zero",
                "kernel_parametrization": "identity",
                "use_rezero": False,
            },
            state_updater={},
        )

    def forecast(
        self,
        model: LinODEnet,
        inputs: SplitTimeData,
        /,
    ) -> tuple[torch.Tensor, ...]:
        r"""Return LinODEnet predictions for sequential forecasting inputs."""
        assert inputs.target_values is not None
        query_valid = inputs.query_mask.any(dim=-1)
        pred = model.predict(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=inputs.context_mask,
            query_times=inputs.query_times,
            query_mask=inputs.query_mask,
        ).masked_fill(~inputs.query_mask, nan)

        assert pred.shape == inputs.target_values.shape
        assert pred[query_valid].isfinite().all()
        assert pred[~query_valid].isnan().all()
        return (pred,)

    def loss(
        self,
        model: LinODEnet,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return mean squared error for LinODEnet predictions."""
        del model
        (forecast,) = predictions
        mask = targets.isfinite()
        return F.mse_loss(forecast[mask], targets[mask])


def test_make_linodenet_instantiates_expected_components() -> None:
    r"""The helper should assemble the default LinearFlow/Linear/update stack."""
    model = make_linodenet(
        linodenet=MappingProxyType(
            {"input_size": 3, "latent_size": 5, "batch_first": False}
        ),
        decoder=MappingProxyType({"in_features": 5, "out_features": 3, "bias": False}),
        state_propagator=MappingProxyType(
            {
                "input_size": 5,
                "kernel_initialization": "zero",
                "kernel_parametrization": "identity",
                "use_rezero": False,
                "use_bias": True,
            }
        ),
        state_updater=MappingProxyType(
            {
                "loss": "l1",
                "regularizer": "l2",
                "regularization_strength": 0.25,
                "step_size": 0.5,
            }
        ),
    )

    assert isinstance(model, LinODEnet)
    assert isinstance(model.decoder, nn.Linear)
    assert isinstance(model.state_updater, GradientStepUpdater)
    assert isinstance(model.state_propagator, LinearFlow)
    assert model.decoder is model.state_updater.decoder
    assert model.batch_first is False
    assert model.decoder.in_features == 5
    assert model.decoder.out_features == 3
    assert model.decoder.bias is None
    assert model.state_propagator.input_size == 5
    assert model.state_propagator.bias is not None
    assert model.state_propagator.use_rezero is False
    assert isinstance(model.state_updater.loss, LpLoss)
    assert isinstance(model.state_updater.regularizer, LpLoss)
    assert model.state_updater.loss.p == 1.0
    assert model.state_updater.regularizer.p == 2.0
    torch.testing.assert_close(
        model.state_updater.regularization_strength.detach(),
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        model.state_updater.step_size.detach(), torch.tensor(0.5)
    )


def test_make_linodenet_uses_decoder_kwargs_for_prediction_shape() -> None:
    r"""The helper should wire a decoder compatible with the model dimensions."""
    model = make_linodenet(
        linodenet={"input_size": 2, "latent_size": 4},
        decoder={"in_features": 4, "out_features": 2},
        state_propagator={"input_size": 4},
        state_updater={},
    )
    latent_state = torch.randn(7, 4)

    prediction = model.decoder(latent_state)

    assert prediction.shape == (7, 2)


def test_linodenet_forward_succeeds_on_dense_context_sequence() -> None:
    r"""A direct forward pass should run on an unpadded dense context timeline."""
    data = make_forecasting_request(
        seed=0,
        batch_shape=(4,),
        min_steps=2,
        max_steps=6,
        context_shape=(3,),
        input_missingness=True,
    )
    model = make_linodenet(
        linodenet={"input_size": 3, "latent_size": 4},
        decoder={"in_features": 4, "out_features": 3},
        state_propagator={
            "input_size": 4,
            "kernel_initialization": "zero",
            "kernel_parametrization": "identity",
            "use_rezero": False,
        },
        state_updater={},
    )

    prediction = model(
        timestamps=data.context_times,
        query_mask=data.context_mask,
        context_values=data.context_values,
        context_mask=data.context_mask,
    )

    assert prediction.shape == data.context_values.shape
    # assert prediction.isfinite().all()
