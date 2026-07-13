r"""Tests for LinODEnet model construction helpers."""

from types import MappingProxyType
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import nan, nn
from torch.nn import functional as F

from linodenet_models.linodenet import LinearFlow, LinODEnet, make_linodenet
from linodenet_models.state_update import (
    GradientStepUpdater,
    InnovationCell,
    KalmanCell,
    LpLoss,
)
from linodenet_models.utils import SplitTimeData

from .base import TestForecastingModel, make_forecasting_request


class LinODEnetTestConfig(NamedTuple):
    r"""Configuration used by shared LinODEnet forecasting-model tests."""

    input_size: int
    latent_size: int
    updater: str = "gradient"
    updater_config: str | None = None


class TestLinODEnet(TestForecastingModel[LinODEnet]):
    r"""Shared forecasting-model tests for LinODEnet."""

    GRADIENT_WARMUP_STEPS = 1

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[LinODEnetTestConfig] = LinODEnetTestConfig(
        input_size=CONTEXT_SHAPE[0],
        latent_size=8,
    )
    MODEL_CONFIGS: ClassVar[dict[str, LinODEnetTestConfig]] = {
        "gradient": STANDARD_CONFIG,
        "innovation_constant": STANDARD_CONFIG._replace(
            updater="innovation",
            updater_config="constant",
        ),
        "innovation_attention": STANDARD_CONFIG._replace(
            updater="innovation",
            updater_config="attention",
        ),
        "kalman_constant": STANDARD_CONFIG._replace(
            updater="kalman",
            updater_config="constant",
        ),
        "kalman_attention": STANDARD_CONFIG._replace(
            updater="kalman",
            updater_config="attention",
        ),
    }

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
        decoder = nn.Sequential(
            nn.Linear(model_config.latent_size, 2 * model_config.latent_size),
            nn.GELU(),
            nn.Linear(2 * model_config.latent_size, model_config.input_size),
        )
        propagator = LinearFlow(
            model_config.latent_size,
            kernel_initialization="zero",
            kernel_parametrization="identity",
            use_rezero=False,
        )
        match model_config.updater:
            case "gradient":
                updater = GradientStepUpdater(decoder=decoder)
            case "innovation":
                updater = InnovationCell(
                    model_config.input_size,
                    model_config.latent_size,
                    gain=model_config.updater_config or "constant",
                    observation_map=decoder,
                )
            case "kalman":
                updater = KalmanCell(
                    model_config.input_size,
                    model_config.latent_size,
                    covariance_factor=model_config.updater_config or "constant",
                    observation_map=decoder,
                )
            case _:
                raise ValueError(f"Unknown updater: {model_config.updater!r}.")

        return LinODEnet(
            input_size=model_config.input_size,
            latent_size=model_config.latent_size,
            decoder=decoder,
            state_propagator=propagator,
            state_updater=updater,
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

    def assert_self_consistent(self, model: LinODEnet, /, *, seed: int) -> None:
        r"""Check that treating a prediction as an observation is a no-op."""
        data = make_forecasting_request(
            seed=seed,
            batch_shape=(4,),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
            input_missingness=True,
        )
        prediction = model.predict(
            context_times=data.context_times,
            context_values=data.context_values,
            context_mask=data.context_mask,
            query_times=data.query_times,
            query_mask=data.query_mask,
        )

        context_axis = -1 - len(self.CONTEXT_SHAPE)
        updated_context_times = torch.cat(
            [data.context_times, data.query_times[..., :1]],
            dim=-1,
        )
        updated_context_values = torch.cat(
            [data.context_values, prediction[..., :1, :].detach()],
            dim=context_axis,
        )
        updated_context_mask = torch.cat(
            [data.context_mask, data.query_mask[..., :1, :]],
            dim=context_axis,
        )

        updated_prediction = model.predict(
            context_times=updated_context_times,
            context_values=updated_context_values,
            context_mask=updated_context_mask,
            query_times=data.query_times[..., 1:],
            query_mask=data.query_mask[..., 1:, :],
        )
        torch.testing.assert_close(
            updated_prediction,
            prediction[..., 1:, :],
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("config_key", MODEL_CONFIGS)
    def test_self_consistency_at_init(self, config_key: str) -> None:
        r"""Check that linodenet is self-consistent at initialization.

        .. math:: ŷ(t∣H) = ŷ(t ∣ H⊕(τ, ŷ(τ∣H)))
        """
        # 1. initialize model
        torch.manual_seed(0)
        model_config = self.MODEL_CONFIGS[config_key]
        model = self.make_model(model_config)

        # 2. check self-consistency on random data
        # 2.a make predictions on random data
        # 2.b append first prediction to context
        # 2.c predict using updated context, compare to 2.a
        self.assert_self_consistent(model, seed=1)

    @pytest.mark.parametrize("config_key", MODEL_CONFIGS)
    def test_self_consistency_trained(self, config_key: str) -> None:
        r"""Check that linodenet is self-consistent after training.

        .. math:: ŷ(t∣H) = ŷ(t ∣ H⊕(τ, ŷ(τ∣H)))
        """
        # 1. initialize model
        torch.manual_seed(0)
        model_config = self.MODEL_CONFIGS[config_key]
        model = self.make_model(model_config)

        # 2. train model on random data for 3 iterations
        train_data = make_forecasting_request(
            seed=2,
            batch_shape=(4,),
            min_steps=4,
            max_steps=4,
            context_shape=self.CONTEXT_SHAPE,
            output_shape=self.OUTPUT_SHAPE,
        )
        assert train_data.target_values is not None
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            predictions = self.forecast(model, train_data)
            loss = self.loss(model, predictions, train_data.target_values)
            loss.backward()
            optimizer.step()

        # 3. check self-consistency on random data
        self.assert_self_consistent(model, seed=3)


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
    assert model.state_updater.loss.p == 1.0
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
