r"""Base test classes for forecasting models."""

from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.testing import assert_close


class SequentialData(NamedTuple):
    r"""Random forecasting input/output data with original sequence lengths."""

    context_times: Tensor
    context_values: Tensor
    context_lengths: Tensor
    query_times: Tensor
    query_values: Tensor
    query_lengths: Tensor

    @property
    def context_mask(self) -> Tensor:
        r"""Boolean mask for valid context sequence entries."""
        return (
            torch.arange(
                self.context_times.shape[-1], device=self.context_lengths.device
            )
            < self.context_lengths[..., None]
        )

    @property
    def query_mask(self) -> Tensor:
        r"""Boolean mask for valid query sequence entries."""
        return (
            torch.arange(self.query_times.shape[-1], device=self.query_lengths.device)
            < self.query_lengths[..., None]
        )

    @classmethod
    def new_sample(
        cls,
        *,
        seed: int,
        batch_shape: int | tuple[int, ...],
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...] | None = None,
        input_missingness: bool = False,
    ) -> SequentialData:
        r"""Sample random context and query data for a forecasting model."""
        if min_steps < 1:
            raise ValueError("min_steps must be positive.")
        if max_steps < min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps.")
        if output_shape is None:
            output_shape = context_shape

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
        for steps, value_shape in (
            (context_steps, context_shape),
            (query_steps, output_shape),
        ):
            time_sequences = [
                torch.sort(torch.rand(int(num_steps), generator=generator)).values
                for num_steps in steps
            ]
            value_sequences = [
                torch.randn(int(num_steps), *value_shape, generator=generator)
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
                    ).reshape(*batch_shape, -1, *value_shape)
                )

        context_times, query_times = times
        context_values, query_values = values

        if input_missingness:
            miss_mask = torch.rand(context_values.shape, generator=generator) < 0.5
            context_values = context_values.masked_fill(miss_mask, torch.nan)

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
        assert context_values.shape == (*batch_shape, context_length, *context_shape)
        assert query_times.shape == (*batch_shape, query_length)
        assert query_values.shape == (*batch_shape, query_length, *output_shape)
        return SequentialData(
            context_times=context_times,
            context_values=context_values,
            context_lengths=context_lengths,
            query_times=query_times,
            query_values=query_values,
            query_lengths=query_lengths,
        )


class TestForecastingModel[M: nn.Module](ABC):
    r"""Shared behavioral tests for forecasting models."""

    SEED: ClassVar[int] = 0
    MIN_STEPS: ClassVar[int] = 2
    MAX_STEPS: ClassVar[int] = 5
    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE

    @abstractmethod
    def make_model(self, model_config: object, /) -> M:
        r"""Instantiate the forecasting model under test."""
        raise NotImplementedError

    @abstractmethod
    def forecast(self, model: M, inputs: SequentialData, /) -> tuple[Tensor, ...]:
        r"""Return model predictions for sequential forecasting inputs."""
        raise NotImplementedError

    @abstractmethod
    def loss(
        self, model: M, predictions: tuple[Tensor, ...], targets: Tensor
    ) -> Tensor:
        r"""Return a scalar training loss for model predictions."""
        raise NotImplementedError

    @pytest.fixture
    def seed(self) -> int:
        r"""Random seed used for synthetic data and model initialization."""
        return self.SEED

    @pytest.fixture
    def min_steps(self) -> int:
        r"""Minimum number of context and query time steps."""
        return self.MIN_STEPS

    @pytest.fixture
    def max_steps(self) -> int:
        r"""Maximum number of context and query time steps."""
        return self.MAX_STEPS

    @pytest.fixture
    def context_shape(self) -> tuple[int, ...]:
        r"""Context value event shape."""
        return self.CONTEXT_SHAPE

    @pytest.fixture
    def output_shape(self, context_shape: tuple[int, ...]) -> tuple[int, ...]:
        r"""Query value event shape."""
        return context_shape if self.OUTPUT_SHAPE is None else self.OUTPUT_SHAPE

    @pytest.fixture(
        params=[(), (8,), (1, 2, 3)],
        ids=["batch_shape=()", "batch_shape=(8,)", "batch_shape=(1,2,3)"],
    )
    def batch_shape(self, request: pytest.FixtureRequest) -> tuple[int, ...]:
        r"""Batch shape used for batched tests."""
        return request.param

    @pytest.fixture
    def model_config(self) -> object:
        r"""Configuration object passed to :meth:`make_model`."""
        return None

    @pytest.fixture
    def input_missingness(self) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return False

    def test_forward_unbatched(
        self,
        model_config: object,
        seed: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        data = SequentialData.new_sample(
            seed=seed,
            batch_shape=(),
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        torch.manual_seed(seed)
        model = self.make_model(model_config)
        self.forecast(model, data)

    def test_forward_batched(
        self,
        model_config: object,
        seed: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        data = SequentialData.new_sample(
            seed=seed,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        torch.manual_seed(seed)
        model = self.make_model(model_config)
        self.forecast(model, data)

    def test_forward_batched_matches_forward_unbatched(
        self,
        model_config: object,
        seed: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check batched predictions do not depend on sequence padding."""
        generator = torch.Generator().manual_seed(seed)
        context_lengths = torch.tensor([2, 17])
        query_lengths = torch.tensor([17, 2])
        context_size = int(context_lengths.max().item())
        query_size = int(query_lengths.max().item())

        context_times = torch.full((2, context_size), torch.nan)
        query_times = torch.full((2, query_size), torch.nan)
        context_values = torch.full((2, context_size, *context_shape), torch.nan)
        query_values = torch.full((2, query_size, *output_shape), torch.nan)

        for k, (context_length_tensor, query_length_tensor) in enumerate(
            zip(context_lengths, query_lengths, strict=True)
        ):
            context_length = int(context_length_tensor.item())
            query_length = int(query_length_tensor.item())
            context_times[k, :context_length] = torch.linspace(0.0, 1.0, context_length)
            query_times[k, :query_length] = torch.linspace(1.1, 2.1, query_length)
            context_values[k, :context_length] = torch.randn(
                context_length,
                *context_shape,
                generator=generator,
            )
            query_values[k, :query_length] = torch.randn(
                query_length,
                *output_shape,
                generator=generator,
            )

        if input_missingness:
            miss_mask = torch.rand(context_values.shape, generator=generator) < 0.5
            context_values = context_values.masked_fill(miss_mask, torch.nan)

        data = SequentialData(
            context_times=context_times,
            context_values=context_values,
            context_lengths=context_lengths,
            query_times=query_times,
            query_values=query_values,
            query_lengths=query_lengths,
        )

        torch.manual_seed(seed)
        model = self.make_model(model_config)
        batched_predictions = self.forecast(model, data)
        expected_predictions = [
            prediction.new_full(prediction.shape, torch.nan)
            for prediction in batched_predictions
        ]

        for k, (context_length_tensor, query_length_tensor) in enumerate(
            zip(context_lengths, query_lengths, strict=True)
        ):
            context_length = int(context_length_tensor.item())
            query_length = int(query_length_tensor.item())
            single_data = SequentialData(
                context_times=context_times[k : k + 1, :context_length],
                context_values=context_values[k : k + 1, :context_length],
                context_lengths=context_lengths[k : k + 1],
                query_times=query_times[k : k + 1, :query_length],
                query_values=query_values[k : k + 1, :query_length],
                query_lengths=query_lengths[k : k + 1],
            )

            for prediction, single_prediction in zip(
                expected_predictions,
                self.forecast(model, single_data),
                strict=True,
            ):
                prediction[k : k + 1, :query_length] = single_prediction

        query_mask = data.query_mask
        for prediction, expected in zip(
            batched_predictions,
            expected_predictions,
            strict=True,
        ):
            mask = query_mask
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, torch.nan),
                expected,
                equal_nan=True,
                rtol=0.0,
                atol=1e-4,
            )

    def test_padding_invariance(
        self,
        model_config: object,
        seed: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check predictions are unchanged by extra NaN tail padding."""
        data = SequentialData.new_sample(
            seed=seed,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        padding = 32
        batch_dims = data.context_times.shape[:-1]
        query_size = data.query_times.shape[-1]

        padded_data = SequentialData(
            context_times=torch.cat(
                [
                    data.context_times,
                    data.context_times.new_full((*batch_dims, padding), torch.nan),
                ],
                dim=-1,
            ),
            context_values=torch.cat(
                [
                    data.context_values,
                    data.context_values.new_full(
                        (*batch_dims, padding, *context_shape),
                        torch.nan,
                    ),
                ],
                dim=-1 - len(context_shape),
            ),
            context_lengths=data.context_lengths,
            query_times=torch.cat(
                [
                    data.query_times,
                    data.query_times.new_full((*batch_dims, padding), torch.nan),
                ],
                dim=-1,
            ),
            query_values=torch.cat(
                [
                    data.query_values,
                    data.query_values.new_full(
                        (*batch_dims, padding, *output_shape),
                        torch.nan,
                    ),
                ],
                dim=-1 - len(output_shape),
            ),
            query_lengths=data.query_lengths,
        )

        torch.manual_seed(seed)
        model = self.make_model(model_config)
        predictions = self.forecast(model, data)
        padded_predictions = self.forecast(model, padded_data)

        query_mask = data.query_mask
        for prediction, padded_prediction in zip(
            predictions,
            padded_predictions,
            strict=True,
        ):
            query_axis = prediction.ndim - len(output_shape) - 1
            index = [slice(None)] * padded_prediction.ndim
            index[query_axis] = slice(None, query_size)
            padded_window = padded_prediction[tuple(index)]
            mask = query_mask
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, torch.nan),
                padded_window.masked_fill(~mask, torch.nan),
                equal_nan=True,
                rtol=0.0,
                atol=1e-4,
            )

    def test_training_unbatched(
        self,
        model_config: object,
        seed: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        torch.manual_seed(seed)
        data = SequentialData.new_sample(
            seed=seed,
            batch_shape=(),
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        model = self.make_model(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        predictions = self.forecast(model, data)
        initial_loss = self.loss(model, predictions, data.query_values)

        for _ in range(3):
            optimizer.zero_grad()
            predictions = self.forecast(model, data)
            loss = self.loss(model, predictions, data.query_values)
            loss.backward()

            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        predictions = self.forecast(model, data)
        final_loss = self.loss(model, predictions, data.query_values)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss

    def test_training_batched(
        self,
        model_config: object,
        seed: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        torch.manual_seed(seed)
        data = SequentialData.new_sample(
            seed=seed,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        model = self.make_model(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        predictions = self.forecast(model, data)
        initial_loss = self.loss(model, predictions, data.query_values)

        for _ in range(3):
            optimizer.zero_grad()
            predictions = self.forecast(model, data)
            loss = self.loss(model, predictions, data.query_values)
            loss.backward()

            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                assert parameter.grad.abs().sum() > 0, name

            optimizer.step()

        predictions = self.forecast(model, data)
        final_loss = self.loss(model, predictions, data.query_values)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss
