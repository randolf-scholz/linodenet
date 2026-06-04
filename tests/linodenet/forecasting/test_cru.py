r"""Tests for CRU model construction."""

from pathlib import Path

import torch
import yaml

from linodenet.forecasting.cru import (
    CRU,
    CRUConfig,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    build_cru,
)

CRU_CONFIG_YAML = r"""
input_size: 6
output_size: 2
latent_size: 10
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
bandwidth: 2
variance_activation: abs
batch_first: true
initial_variance: 4.0
variance_floor: 0.0001
validate_args: true
"""


def test_build_cru_instantiates_from_dataclass_config() -> None:
    config = CRUConfig(
        input_size=5,
        output_size=3,
        latent_size=8,
        encoder=EncoderConfig(
            input_size=5,
            output_size=4,
            hidden_size=7,
            num_hidden_layers=1,
            activation_function="tanh",
            variance_activation="exp",
        ),
        decoder=DecoderConfig(
            input_size=4,
            output_size=3,
            hidden_size=11,
            num_hidden_mean_model_layers=1,
            num_hidden_variance_model_layers=1,
            activation_function="tanh",
            variance_activation="square",
        ),
        num_basis=4,
        bandwidth=2,
        variance_activation="abs",
        initial_variance=2.0,
        variance_floor=1e-5,
        batch_first=True,
        validate_args=True,
    )

    model = build_cru(config)

    assert isinstance(model, CRU)
    assert model.input_size == config.input_size
    assert model.output_size == config.output_size
    assert model.latent_size == config.latent_size
    assert model.latent_observation_size == 4
    assert model.num_basis == config.num_basis
    assert model.initial_variance == config.initial_variance
    assert model.variance_floor == config.variance_floor
    assert model.batch_first is config.batch_first
    assert model.validate_args is config.validate_args
    assert model.initial_covariance.shape == (config.latent_size, config.latent_size)
    assert torch.equal(
        model.initial_covariance,
        config.initial_variance * torch.eye(config.latent_size),
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
            "latent_size": 4,
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


def test_build_cru_instantiates_from_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "cru.yaml"
    config_path.write_text(CRU_CONFIG_YAML, encoding="utf-8")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_cru(config)

    assert isinstance(model, CRU)
    assert model.input_size == 6
    assert model.output_size == 2
    assert model.latent_size == 10
    assert model.latent_observation_size == 5
    assert model.encoder.input_size == 6
    assert model.encoder.output_size == 5
    assert model.decoder.input_size == 5
    assert model.decoder.output_size == 2
    assert model.num_basis == 3
    assert model.batch_first is True
    assert model.validate_args is True
