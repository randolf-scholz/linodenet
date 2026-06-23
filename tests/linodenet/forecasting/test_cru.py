r"""Tests for CRU model construction."""

from pathlib import Path
from typing import ClassVar

import pytest
import torch
import yaml

from linodenet.forecasting.cru import (
    CRU,
    CRUConfig,
    Decoder,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    apply_masked,
    build_cru,
)
from linodenet.forecasting.utils import BatchedDenseArgs

from .base import SequentialData, TestForecastingModel

# language=yaml
CRU_CONFIG_YAML = r"""
input_size: 6
output_size: 2
latent_size: 5
encoder:
  input_size: 6
  output_size: 5
  hidden_size: 8
  num_hidden_layers: 1
  activation_function: tanh
  variance_activation: exp
decoder:
  input_size: 5
  output_size: 2
  hidden_size: 12
  num_hidden_mean_model_layers: 1
  num_hidden_variance_model_layers: 1
  activation_function: tanh
  variance_activation: square
num_basis: 3
bandwidth: 1
variance_activation: abs
initial_variance: 4.0
validate_args: true
"""


class TestMaskedApply:
    r"""Tests for applying functions only to valid batch elements."""

    def test_applies_unary_torch_function(self) -> None:
        x = torch.tensor(
            [
                [1.0, -2.0],
                [torch.nan, torch.nan],
                [3.0, -4.0],
            ]
        )
        mask = x.isfinite().all(dim=-1)

        result = apply_masked(torch.square, (x,), mask, fill_value=-1.0)

        expected = torch.tensor(
            [
                [1.0, 4.0],
                [-1.0, -1.0],
                [9.0, 16.0],
            ]
        )
        assert torch.equal(result, expected)

    def test_applies_binary_torch_function(self) -> None:
        x = torch.tensor(
            [
                [1.0, 2.0],
                [torch.nan, torch.nan],
                [3.0, 4.0],
            ]
        )
        y = torch.tensor(
            [
                [10.0, 20.0],
                [torch.nan, torch.nan],
                [30.0, 40.0],
            ]
        )
        mask = x.isfinite().all(dim=-1) & y.isfinite().all(dim=-1)

        result = apply_masked(torch.add, (x, y), mask, fill_value=0.0)

        expected = torch.tensor(
            [
                [11.0, 22.0],
                [0.0, 0.0],
                [33.0, 44.0],
            ]
        )
        assert torch.equal(result, expected)

    def test_preserves_multi_dimensional_batch_shape(self) -> None:
        x = torch.arange(24.0).reshape(2, 3, 4)
        x[0, 1] = torch.nan
        x[1, 2] = torch.nan
        mask = x.isfinite().all(dim=-1)

        result = apply_masked(torch.sin, (x,), mask)

        assert result.shape == x.shape
        assert torch.allclose(result[mask], torch.sin(x[mask]))
        assert torch.isnan(result[~mask]).all()

    def test_torch_compile_fullgraph(self) -> None:
        x = torch.arange(12.0).reshape(3, 4)
        x[1] = torch.nan

        compiled = torch.compile(
            lambda values: apply_masked(
                torch.sin,
                (values,),
                values.isfinite().all(dim=-1),
            ),
            fullgraph=True,
        )

        result = compiled(x)
        expected = apply_masked(torch.sin, (x,), x.isfinite().all(dim=-1))
        mask = x.isfinite().all(dim=-1)
        assert result.shape == x.shape
        assert torch.allclose(result[mask], expected[mask])
        assert torch.isnan(result[~mask]).all()

    def test_torch_export_with_linear_module(self) -> None:
        class MaskedLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 3)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return apply_masked(
                    self.linear,
                    (values,),
                    values.isfinite().all(dim=-1),
                    fill_value=0.0,
                )

        torch.manual_seed(0)
        model = MaskedLinear()
        x = torch.randn(5, 4)
        x[1] = torch.nan
        x[3] = torch.nan

        exported = torch.export.export(model, (x,))
        result = exported.module()(x)
        expected = model(x)

        assert result.shape == (5, 3)
        assert result.isfinite().all()
        assert torch.allclose(result, expected)

    def test_linear_forward_backward_with_nan_only_batch_elements(self) -> None:
        torch.manual_seed(0)
        linear = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        x[1] = torch.nan
        x[3] = torch.nan
        x.requires_grad_()
        mask = x.isfinite().all(dim=-1)

        result = apply_masked(linear, (x,), mask, fill_value=0.0)
        loss = result.sum()
        loss.backward()

        assert result.shape == (5, 3)
        assert result.isfinite().all()
        assert x.grad is not None
        assert x.grad.isfinite().all()
        assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
        for parameter in linear.parameters():
            assert parameter.grad is not None
            assert parameter.grad.isfinite().all()


class TestModel(TestForecastingModel):
    r"""Tests for direct CRU model construction."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (5,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    STANDARD_CONFIG: ClassVar[CRUConfig] = CRUConfig(
        input_size=CONTEXT_SHAPE[0],
        output_size=OUTPUT_SHAPE[0],
        latent_size=4,
        encoder=EncoderConfig(
            input_size=CONTEXT_SHAPE[0],
            output_size=4,
            hidden_size=7,
            num_hidden_layers=1,
            activation_function="tanh",
            variance_activation="exp",
        ),
        decoder=DecoderConfig(
            input_size=4,
            output_size=OUTPUT_SHAPE[0],
            hidden_size=11,
            num_hidden_mean_model_layers=1,
            num_hidden_variance_model_layers=1,
            activation_function="tanh",
            variance_activation="square",
        ),
        num_basis=4,
        bandwidth=1,
        variance_activation="abs",
        initial_variance=2.0,
        validate_args=True,
    )

    @pytest.fixture
    def model_config(self) -> CRUConfig:
        r"""Configuration used to instantiate the CRU under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> CRU:
        r"""Instantiate a CRU from :attr:`STANDARD_CONFIG` without ``build_cru``."""
        if not isinstance(model_config, CRUConfig):
            raise TypeError("model_config must be a CRUConfig.")
        config = model_config
        encoder = Encoder(
            config.encoder.input_size,
            config.encoder.output_size,
            config.encoder.hidden_size,
            num_hidden_layers=config.encoder.num_hidden_layers,
            activation_function=config.encoder.activation_function,
            variance_activation=config.encoder.variance_activation,
        )
        decoder = Decoder(
            config.decoder.input_size,
            config.decoder.output_size,
            config.decoder.hidden_size,
            num_hidden_mean_model_layers=(config.decoder.num_hidden_mean_model_layers),
            num_hidden_variance_model_layers=(
                config.decoder.num_hidden_variance_model_layers
            ),
            activation_function=config.decoder.activation_function,
            variance_activation=config.decoder.variance_activation,
        )
        model = CRU(
            config.input_size,
            config.latent_size,
            output_size=config.output_size,
            encoder=encoder,
            decoder=decoder,
            num_basis=config.num_basis,
            bandwidth=config.bandwidth,
            initial_variance=config.initial_variance,
            variance_activation=config.variance_activation,
            validate_args=config.validate_args,
        )
        with torch.no_grad():
            model.transition_matrix_parameters.copy_(
                torch.randn_like(model.transition_matrix_parameters)
            )
        return model

    def make_cru(self) -> CRU:
        r"""Instantiate a CRU from :attr:`STANDARD_CONFIG` without ``build_cru``."""
        return self.make_model(self.STANDARD_CONFIG)

    def forecast(
        self, model: CRU, inputs: SequentialData, /
    ) -> tuple[torch.Tensor, ...]:
        r"""Return CRU predictions for sequential forecasting inputs."""
        context_mask = inputs.context_mask.unsqueeze(-1).expand_as(
            inputs.context_values
        )
        query_mask = inputs.query_mask.unsqueeze(-1).expand_as(inputs.query_values)
        combined = BatchedDenseArgs(
            context_times=inputs.context_times,
            context_values=inputs.context_values,
            context_mask=context_mask,
            query_times=inputs.query_times,
            query_mask=query_mask,
            query_values=inputs.query_values,
        ).to_combined()

        all_pred_means, all_pred_vars = model(
            combined.times,
            combined.context_values,
            combined.context_mask,
            combined.query_mask,
        )

        # Extract predictions at query steps and scatter into (*, Q, F) output.
        has_query = combined.query_mask.any(dim=-1)
        pred_mean = torch.full_like(inputs.query_values, torch.nan)
        pred_var = torch.full_like(inputs.query_values, torch.nan)
        pred_mean[inputs.query_mask] = all_pred_means[has_query]
        pred_var[inputs.query_mask] = all_pred_vars[has_query]

        assert pred_mean.shape == inputs.query_values.shape
        assert pred_var.shape == inputs.query_values.shape
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_var[inputs.query_mask].isfinite().all()
        return pred_mean, pred_var

    def loss(
        self,
        model: CRU,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return CRU negative log-likelihood for predictions."""
        pred_mean, pred_var = predictions
        return model.nll(targets, pred_mean, pred_var)

    def test_make_cru_instantiates_standard_model(self) -> None:
        config = self.STANDARD_CONFIG
        model = self.make_cru()

        assert isinstance(model, CRU)
        assert model.input_size == config.input_size
        assert model.output_size == config.output_size
        assert model.latent_size == config.latent_size
        assert model.num_basis == config.num_basis
        assert model.initial_variance == config.initial_variance
        assert model.validate_args is config.validate_args
        assert model.initial_covariance.shape == (
            2 * config.latent_size,
            2 * config.latent_size,
        )
        assert torch.equal(
            model.initial_covariance,
            config.initial_variance * torch.eye(2 * config.latent_size),
        )

        assert isinstance(model.encoder, Encoder)
        assert model.encoder.input_size == config.encoder.input_size
        assert model.encoder.output_size == config.encoder.output_size
        assert model.encoder.hidden_size == config.encoder.hidden_size
        assert isinstance(model.decoder, Decoder)
        assert model.decoder.input_size == config.decoder.input_size
        assert model.decoder.output_size == config.decoder.output_size
        assert model.decoder.hidden_size == config.decoder.hidden_size

    def test_rejects_finite_context_time_with_missing_value(self) -> None:
        # CRU does not support feature-level missingness: passing a context_mask
        # that is partially True for a single time step should raise.
        model = self.make_cru()
        D = self.STANDARD_CONFIG.input_size
        F = self.STANDARD_CONFIG.output_size or D
        times = torch.tensor([0.0, 0.5, 1.0, 2.0])
        context_values = torch.randn(4, D)
        # step 1 has feature 0 missing → partial context_mask row
        context_mask = torch.ones(4, D, dtype=torch.bool)
        context_mask[1, 0] = False
        query_mask = torch.zeros(4, F, dtype=torch.bool)
        query_mask[3] = True

        with pytest.raises((AssertionError, ValueError)):
            model(times, context_values, context_mask, query_mask)

    def test_rejects_padded_context_time_with_finite_value(self) -> None:
        # A time step with times=NaN but context_mask=True is contradictory.
        model = self.make_cru()
        D = self.STANDARD_CONFIG.input_size
        F = self.STANDARD_CONFIG.output_size or D
        times = torch.tensor([0.0, 1.0, torch.nan, 2.0])
        context_values = torch.randn(4, D)
        # step 2 has NaN time but context_mask=True → invalid
        context_mask = torch.tensor([[True] * D, [True] * D, [True] * D, [False] * D])
        query_mask = torch.zeros(4, F, dtype=torch.bool)
        query_mask[3] = True

        with pytest.raises((AssertionError, ValueError)):
            model(times, context_values, context_mask, query_mask)


def test_build_cru_instantiates_from_dataclass_config() -> None:
    config = TestModel.STANDARD_CONFIG

    model = build_cru(config)

    assert isinstance(model, CRU)
    assert model.input_size == config.input_size
    assert model.output_size == config.output_size
    assert model.latent_size == config.latent_size
    assert model.num_basis == config.num_basis
    assert model.initial_variance == config.initial_variance
    assert model.validate_args is config.validate_args
    assert model.initial_covariance.shape == (
        2 * config.latent_size,
        2 * config.latent_size,
    )
    assert torch.equal(
        model.initial_covariance,
        config.initial_variance * torch.eye(2 * config.latent_size),
    )

    assert isinstance(model.encoder, Encoder)
    assert model.encoder.input_size == config.encoder.input_size
    assert model.encoder.output_size == config.encoder.output_size
    assert model.encoder.hidden_size == config.encoder.hidden_size
    assert model.decoder.input_size == config.decoder.input_size
    assert model.decoder.output_size == config.decoder.output_size
    assert model.decoder.hidden_size == config.decoder.hidden_size


def test_build_cru_instantiates_from_mapping_config() -> None:
    model = build_cru(
        {
            "input_size": 2,
            "output_size": 1,
            "latent_size": 2,
            "encoder": {"input_size": 2, "output_size": 2, "hidden_size": 3},
            "decoder": {"input_size": 2, "output_size": 1, "hidden_size": 5},
            "num_basis": 2,
            "bandwidth": 1,
        }
    )

    assert isinstance(model, CRU)
    assert model.encoder.output_size == 2
    assert model.decoder.input_size == 2
    assert model.transition_matrix_parameters.shape[0] == 2


def test_transition_matrix_model_packs_banded_blocks_in_quadrants() -> None:
    model = build_cru(
        {
            "input_size": 2,
            "output_size": 1,
            "latent_size": 3,
            "encoder": {"input_size": 2, "output_size": 3, "hidden_size": 5},
            "decoder": {"input_size": 3, "output_size": 1, "hidden_size": 7},
            "num_basis": 1,
            "bandwidth": 0,
        }
    )

    parameters = 1 + torch.arange(
        model.transition_matrix_parameters.numel(),
        dtype=model.transition_matrix_parameters.dtype,
    ).reshape_as(model.transition_matrix_parameters)
    with torch.no_grad():
        model.transition_matrix_parameters.copy_(parameters)

    actual = model.transition_matrix_model(torch.zeros(2 * model.latent_size))

    expected = torch.tensor([
        [ 1,  0,  0,  4,  0,  0],
        [ 0,  2,  0,  0,  5,  0],
        [ 0,  0,  3,  0,  0,  6],
        [ 7,  0,  0, 10,  0,  0],
        [ 0,  8,  0,  0, 11,  0],
        [ 0,  0,  9,  0,  0, 12],
    ], dtype=actual.dtype)  # fmt: skip

    assert torch.equal(actual, expected)


def test_build_cru_instantiates_from_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "cru.yaml"
    config_path.write_text(CRU_CONFIG_YAML, encoding="utf-8")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_cru(config)

    assert isinstance(model, CRU)
    assert model.input_size == 6
    assert model.output_size == 2
    assert model.latent_size == 5
    assert model.encoder.input_size == 6
    assert model.encoder.output_size == 5
    assert model.decoder.input_size == 5
    assert model.decoder.output_size == 2
    assert model.num_basis == 3
    assert model.validate_args is True
