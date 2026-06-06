r"""Tests for CRU model construction."""

from pathlib import Path
from typing import ClassVar, NamedTuple

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from linodenet.forecasting.cru import (
    CRU,
    CRUConfig,
    Decoder,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    build_cru,
    masked_apply,
)

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
batch_first: true
initial_variance: 4.0
variance_floor: 0.0001
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

        result = masked_apply(torch.square, (x,), mask, fill_value=-1.0)

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

        result = masked_apply(torch.add, (x, y), mask, fill_value=0.0)

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

        result = masked_apply(torch.sin, (x,), mask)

        assert result.shape == x.shape
        assert torch.allclose(result[mask], torch.sin(x[mask]))
        assert torch.isnan(result[~mask]).all()

    def test_torch_compile_fullgraph(self) -> None:
        x = torch.arange(12.0).reshape(3, 4)
        x[1] = torch.nan

        compiled = torch.compile(
            lambda values: masked_apply(
                torch.sin,
                (values,),
                values.isfinite().all(dim=-1),
            ),
            fullgraph=True,
        )

        result = compiled(x)
        expected = masked_apply(torch.sin, (x,), x.isfinite().all(dim=-1))
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
                return masked_apply(
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

        result = masked_apply(linear, (x,), mask, fill_value=0.0)
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


class CRUData(NamedTuple):
    r"""Random CRU input/output data with original sequence lengths."""

    context_times: torch.Tensor
    context_values: torch.Tensor
    context_lengths: torch.Tensor
    query_times: torch.Tensor
    query_values: torch.Tensor
    query_lengths: torch.Tensor

    @property
    def context_mask(self) -> torch.Tensor:
        r"""Boolean mask for valid context sequence entries."""
        return (
            torch.arange(
                self.context_times.shape[-1], device=self.context_lengths.device
            )
            < self.context_lengths[..., None]
        )

    @property
    def query_mask(self) -> torch.Tensor:
        r"""Boolean mask for valid query sequence entries."""
        return (
            torch.arange(self.query_times.shape[-1], device=self.query_lengths.device)
            < self.query_lengths[..., None]
        )


class TestModel:
    r"""Tests for direct CRU model construction."""

    STANDARD_CONFIG: ClassVar[CRUConfig] = CRUConfig(
        input_size=5,
        output_size=3,
        latent_size=4,
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
        bandwidth=1,
        variance_activation="abs",
        initial_variance=2.0,
        variance_floor=1e-5,
        batch_first=True,
        validate_args=True,
    )

    @classmethod
    def make_cru(cls) -> CRU:
        r"""Instantiate a CRU from :attr:`STANDARD_CONFIG` without ``build_cru``."""
        config = cls.STANDARD_CONFIG
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
            variance_floor=config.variance_floor,
            variance_activation=config.variance_activation,
            batch_first=config.batch_first,
            validate_args=config.validate_args,
        )
        with torch.no_grad():
            model.transition_matrix_parameters.copy_(
                torch.randn_like(model.transition_matrix_parameters)
            )
        return model

    @classmethod
    def make_data(
        cls,
        *,
        seed: int,
        batch_shape: int | tuple[int, ...],
        min_steps: int,
        max_steps: int,
    ) -> CRUData:
        r"""Sample random CRU context and query data."""
        if min_steps < 1:
            raise ValueError("min_steps must be positive.")
        if max_steps < min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps.")

        config = cls.STANDARD_CONFIG
        generator = torch.Generator().manual_seed(seed)
        batch_shape = torch.Size(
            (batch_shape,) if isinstance(batch_shape, int) else batch_shape
        )
        if any(size < 1 for size in batch_shape):
            raise ValueError("batch_shape entries must be positive.")

        num_batches = max(batch_shape.numel(), 1)
        context_steps = torch.randint(
            min_steps, max_steps + 1, (num_batches,), generator=generator
        )
        query_steps = torch.randint(
            min_steps, max_steps + 1, (num_batches,), generator=generator
        )

        times = []
        values = []
        for steps, dim in (
            (context_steps, config.input_size),
            (query_steps, config.output_size),
        ):
            time_sequences = [
                torch.sort(torch.rand(int(num_steps), generator=generator)).values
                for num_steps in steps
            ]
            value_sequences = [
                torch.randn(int(num_steps), dim, generator=generator)
                for num_steps in steps
            ]

            if batch_shape == ():
                times.append(time_sequences[0])
                values.append(value_sequences[0])
            else:
                times.append(
                    pad_sequence(
                        time_sequences,
                        batch_first=True,
                        padding_value=torch.nan,
                    ).reshape(*batch_shape, -1)
                )
                values.append(
                    pad_sequence(
                        value_sequences,
                        batch_first=True,
                        padding_value=torch.nan,
                    ).reshape(*batch_shape, -1, dim)
                )

        context_times, query_times = times
        context_values, query_values = values
        context_lengths = context_steps.reshape(batch_shape)
        query_lengths = query_steps.reshape(batch_shape)
        context_length = int(context_steps.max())
        query_length = int(query_steps.max())
        context_end_times = torch.take_along_dim(
            context_times,
            (context_lengths - 1).unsqueeze(-1),
            dim=-1,
        ).squeeze(-1)
        query_times = query_times + context_end_times[..., None]

        assert context_lengths.shape == batch_shape
        assert query_lengths.shape == batch_shape
        assert context_times.shape == (*batch_shape, context_length)
        assert context_values.shape == (*batch_shape, context_length, config.input_size)
        assert query_times.shape == (*batch_shape, query_length)
        assert query_values.shape == (*batch_shape, query_length, config.output_size)
        return CRUData(
            context_times=context_times,
            context_values=context_values,
            context_lengths=context_lengths,
            query_times=query_times,
            query_values=query_values,
            query_lengths=query_lengths,
        )

    def test_make_cru_instantiates_standard_model(self) -> None:
        config = self.STANDARD_CONFIG
        model = self.make_cru()

        assert isinstance(model, CRU)
        assert model.input_size == config.input_size
        assert model.output_size == config.output_size
        assert model.latent_size == config.latent_size
        assert model.num_basis == config.num_basis
        assert model.initial_variance == config.initial_variance
        assert model.variance_floor == config.variance_floor
        assert model.batch_first is config.batch_first
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

    def test_make_data_samples_unbatched_sequences(self) -> None:
        min_steps, max_steps = 2, 5
        data = self.make_data(
            seed=0,
            batch_shape=(),
            min_steps=min_steps,
            max_steps=max_steps,
        )

        assert data.context_times.ndim == 1
        assert data.query_times.ndim == 1
        assert data.context_lengths.shape == ()
        assert data.query_lengths.shape == ()
        assert min_steps <= int(data.context_lengths) <= max_steps
        assert min_steps <= int(data.query_lengths) <= max_steps
        assert data.context_values.shape == (
            data.context_times.shape[-1],
            self.STANDARD_CONFIG.input_size,
        )
        assert data.query_values.shape == (
            data.query_times.shape[-1],
            self.STANDARD_CONFIG.output_size,
        )
        assert data.context_times.shape == (int(data.context_lengths),)
        assert data.query_times.shape == (int(data.query_lengths),)
        assert data.context_mask.shape == data.context_times.shape
        assert data.query_mask.shape == data.query_times.shape
        assert data.context_mask.all()
        assert data.query_mask.all()
        assert torch.diff(data.context_times).ge(0).all()
        assert torch.diff(data.query_times).ge(0).all()

    def test_make_data_samples_padded_batched_sequences(self) -> None:
        batch_shape = (2, 3)
        data = self.make_data(
            seed=0,
            batch_shape=batch_shape,
            min_steps=2,
            max_steps=5,
        )
        context_length = int(data.context_lengths.max())
        query_length = int(data.query_lengths.max())
        input_size = self.STANDARD_CONFIG.input_size
        output_size = self.STANDARD_CONFIG.output_size

        assert data.context_lengths.shape == batch_shape
        assert data.query_lengths.shape == batch_shape
        assert data.context_times.shape == (*batch_shape, context_length)
        assert data.query_times.shape == (*batch_shape, query_length)
        assert data.context_values.shape == (*batch_shape, context_length, input_size)
        assert data.query_values.shape == (*batch_shape, query_length, output_size)
        assert data.context_mask.shape == data.context_times.shape
        assert data.query_mask.shape == data.query_times.shape
        assert torch.equal(data.context_mask.sum(dim=-1), data.context_lengths)
        assert torch.equal(data.query_mask.sum(dim=-1), data.query_lengths)
        assert torch.isnan(data.context_times).any()
        assert torch.isnan(data.query_times).any()
        assert data.context_values[data.context_mask].isfinite().all()
        assert data.query_values[data.query_mask].isfinite().all()

        for times in data.context_times.reshape(-1, context_length):
            finite_times = times[times.isfinite()]
            assert torch.diff(finite_times).ge(0).all()
        for times in data.query_times.reshape(-1, query_length):
            finite_times = times[times.isfinite()]
            assert torch.diff(finite_times).ge(0).all()

    def test_unbatched_forward(self) -> None:
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(), min_steps=2, max_steps=5)

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        assert pred_mean.shape == data.query_values.shape
        assert pred_var.shape == data.query_values.shape
        assert pred_mean.isfinite().all()
        assert pred_var.isfinite().all()
        assert pred_var.ge(0).all()

    def test_batched_forward(self) -> None:
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(4,), min_steps=2, max_steps=5)

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        assert pred_mean.shape == data.query_values.shape
        assert pred_var.shape == data.query_values.shape
        assert pred_mean[data.query_mask].isfinite().all()
        assert pred_var[data.query_mask].isfinite().all()
        assert pred_var[data.query_mask].ge(0).all()

    def test_training_unbatched(self) -> None:
        torch.manual_seed(0)
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(), min_steps=5, max_steps=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        initial_loss = CRU.nll(data.query_values, pred_mean, pred_var)

        for _ in range(3):
            optimizer.zero_grad()
            pred_mean, pred_var = model(
                data.query_times,
                data.context_times,
                data.context_values,
            )
            loss = CRU.nll(data.query_values, pred_mean, pred_var)
            loss.backward()

            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        final_loss = CRU.nll(data.query_values, pred_mean, pred_var)

        for name, parameter in model.named_parameters():
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss

    def test_training_batched(self) -> None:
        torch.manual_seed(0)
        model = self.make_cru()
        data = self.make_data(seed=0, batch_shape=(8,), min_steps=2, max_steps=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )

        initial_loss = CRU.nll(data.query_values, pred_mean, pred_var)

        for _ in range(3):
            optimizer.zero_grad()
            pred_mean, pred_var = model(
                data.query_times,
                data.context_times,
                data.context_values,
            )
            loss = CRU.nll(data.query_values, pred_mean, pred_var)
            loss.backward()

            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        pred_mean, pred_var = model(
            data.query_times,
            data.context_times,
            data.context_values,
        )
        final_loss = CRU.nll(data.query_values, pred_mean, pred_var)

        for name, parameter in model.named_parameters():
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss


def test_build_cru_instantiates_from_dataclass_config() -> None:
    config = TestModel.STANDARD_CONFIG

    model = build_cru(config)

    assert isinstance(model, CRU)
    assert model.input_size == config.input_size
    assert model.output_size == config.output_size
    assert model.latent_size == config.latent_size
    assert model.num_basis == config.num_basis
    assert model.initial_variance == config.initial_variance
    assert model.variance_floor == config.variance_floor
    assert model.batch_first is config.batch_first
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
    assert model.batch_first is True
    assert model.validate_args is True
