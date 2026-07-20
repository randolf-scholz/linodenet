r"""Base test classes for forecasting models."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import pytest
import torch
from torch import Tensor, nan, nn
from torch.testing import assert_close

from linodenet_models import ProbabilisticForecastingModel
from linodenet_models.utils import SplitTimeData


def _as_generator(seed: int | torch.Generator, /) -> torch.Generator:
    r"""Return a torch generator from an integer seed or existing generator."""
    return torch.Generator().manual_seed(seed) if isinstance(seed, int) else seed


def assert_probabilistic_self_consistent(
    model: ProbabilisticForecastingModel,
    data: SplitTimeData,
    *,
    rng: int | torch.Generator,
    num_futures: int,
    num_probes: int,
    atol: float,
    rtol: float,
) -> None:
    r"""Check self-consistency by Monte Carlo marginalization over y⁎.

    The identity is over predictive densities:
    $p(yₜ∣H) = E_{y⁎∼p(yₜ⁎∣H)}[p(yₜ∣H⊕(t⁎, y⁎))]$.
    In log-space this is estimated with ``logmeanexp`` over conditional
    log-densities, not with an arithmetic mean of log-densities.
    """
    query_times = data.query_times[:2]
    query_mask = data.query_mask[:2]

    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        if isinstance(rng, torch.Generator):
            torch.random.set_rng_state(rng.get_state())
        else:
            torch.manual_seed(rng)

        futures = model.sample(
            num_futures,
            query_times=query_times[:1],
            query_mask=query_mask[:1],
            context_times=data.context_times,
            context_values=data.context_values,
            context_mask=data.context_mask,
        )[:, 0]
        y_probe = model.sample(
            num_probes,
            query_times=query_times[1:2],
            query_mask=query_mask[1:2],
            context_times=data.context_times,
            context_values=data.context_values,
            context_mask=data.context_mask,
        )

        base_log_prob = model.log_prob(
            y_probe,
            query_times=query_times[1:2],
            query_mask=query_mask[1:2],
            context_times=data.context_times,
            context_values=data.context_values,
            context_mask=data.context_mask,
        )

        updated_context_times = torch.cat(
            [
                data.context_times.expand(num_futures, -1),
                query_times[:1].expand(num_futures, 1),
            ],
            dim=-1,
        )
        updated_context_values = torch.cat(
            [
                data.context_values.expand(num_futures, -1, -1),
                futures[:, None, :],
            ],
            dim=-2,
        )
        updated_context_mask = torch.cat(
            [
                data.context_mask.expand(num_futures, -1, -1),
                query_mask[:1].expand(num_futures, 1, -1),
            ],
            dim=-2,
        )
        conditional_log_prob = model.log_prob(
            y_probe[:, None].expand(num_probes, num_futures, 1, -1),
            query_times=query_times[1:].expand(num_futures, 1),
            query_mask=query_mask[1:].expand(num_futures, 1, -1),
            context_times=updated_context_times,
            context_values=updated_context_values,
            context_mask=updated_context_mask,
        )[:, :, 0]
        mixture_log_prob = torch.logsumexp(conditional_log_prob, dim=-1) - math.log(
            num_futures
        )
        if isinstance(rng, torch.Generator):
            rng.set_state(torch.random.get_rng_state())

    base_log_prob = base_log_prob[:, 0]
    single_future_error = (conditional_log_prob - base_log_prob[:, None]).abs()
    assert single_future_error.max() > 0.0
    assert_close(mixture_log_prob, base_log_prob, atol=atol, rtol=rtol)


def make_continuous_time_request(
    *,
    rng: int | torch.Generator,
    batch_shape: int | tuple[int, ...],
    min_steps: int,
    max_steps: int,
    context_shape: tuple[int, ...],
    output_shape: tuple[int, ...] | None = None,
    input_missingness: bool = False,
    target_missingness: bool = False,
    batch_first: bool = True,
) -> SplitTimeData:
    r"""Sample random dense forecasting inputs for a forecasting model."""
    rng = _as_generator(rng)
    batch_shape = (batch_shape,) if isinstance(batch_shape, int) else batch_shape
    output_shape = output_shape if output_shape is not None else context_shape

    ctx_lengths = torch.randint(  # (...)
        min_steps, max_steps + 1, size=batch_shape, generator=rng
    )
    qry_lengths = torch.randint(  # (...)
        min_steps, max_steps + 1, size=batch_shape, generator=rng
    )
    ctx_size = max_steps  # int(ctx_lengths.max())
    qry_size = max_steps  # int(qry_lengths.max())

    # sample values
    ctx_values = torch.randn(*batch_shape, ctx_size, *context_shape, generator=rng)
    tgt_values = torch.randn(*batch_shape, qry_size, *output_shape, generator=rng)

    # sample time points
    ctx_seq_shape = (*batch_shape, ctx_size, 1)  # padded with one
    qry_seq_shape = (*batch_shape, qry_size, 1)
    ctx_times = torch.sort(torch.rand(ctx_seq_shape, generator=rng), dim=-2).values
    qry_times = torch.sort(torch.rand(qry_seq_shape, generator=rng), dim=-2).values
    qry_times = qry_times + ctx_times[..., [-1], :]  # add last time point

    # create valid mask by broadcasting over sequence length.
    ctx_valid = torch.arange(ctx_size) < ctx_lengths[..., None]  # (..., $N)
    qry_valid = torch.arange(qry_size) < qry_lengths[..., None]  # (..., $K)
    ctx_valid = ctx_valid.unsqueeze(-1)  # (..., $N, 1)
    qry_valid = qry_valid.unsqueeze(-1)  # (..., $K, 1)
    assert ctx_valid.shape == (*batch_shape, ctx_size, 1)
    assert qry_valid.shape == (*batch_shape, qry_size, 1)

    # mask time stamps and values.
    ctx_times = ctx_times.masked_fill(~ctx_valid, nan)
    qry_times = qry_times.masked_fill(~qry_valid, nan)
    ctx_values = ctx_values.masked_fill(~ctx_valid, nan)
    tgt_values = tgt_values.masked_fill(~qry_valid, nan)

    # mask by feature missingness
    # sample one value per time stamp that is always observed
    ctx_safe = torch.randint(
        0, math.prod(context_shape), size=ctx_seq_shape, generator=rng
    )
    qry_safe = torch.randint(
        0, math.prod(output_shape), size=qry_seq_shape, generator=rng
    )
    ctx_mask = ctx_valid & (
        torch.ones_like(ctx_values, dtype=torch.bool)
        if not input_missingness
        else torch.rand_like(ctx_values, generator=rng) > 0.5
    ).scatter(-1, ctx_safe, True)
    qry_mask = qry_valid & (
        torch.ones_like(tgt_values, dtype=torch.bool)
        if not target_missingness
        else torch.rand_like(tgt_values, generator=rng) > 0.5
    ).scatter(-1, qry_safe, True)
    ctx_values = ctx_values.masked_fill(~ctx_mask, nan)
    tgt_values = tgt_values.masked_fill(~qry_mask, nan)

    # normalize to batch_first, equip floats with grads and return
    seq_dim = -2 if batch_first else 0
    return SplitTimeData(
        context_times=ctx_times.movedim(-2, seq_dim).squeeze(-1).requires_grad_(),
        context_mask=ctx_mask.movedim(-2, seq_dim),
        context_values=ctx_values.movedim(-2, seq_dim).requires_grad_(),
        query_times=qry_times.movedim(-2, seq_dim).squeeze(-1).requires_grad_(),
        query_mask=qry_mask.movedim(-2, seq_dim),
        target_values=tgt_values.movedim(-2, seq_dim).requires_grad_(),
        batch_first=batch_first,
    )


make_forecasting_request = make_continuous_time_request


def make_discrete_time_request(
    *,
    rng: int | torch.Generator,
    batch_shape: int | tuple[int, ...],
    min_steps: int,
    max_steps: int,
    context_shape: tuple[int, ...],
    output_shape: tuple[int, ...] | None = None,
    input_missingness: bool = False,
    target_missingness: bool = False,
    batch_first: bool = True,
) -> SplitTimeData:
    r"""Sample random dense integer-step forecasting inputs."""
    rng = _as_generator(rng)
    batch_shape = (batch_shape,) if isinstance(batch_shape, int) else batch_shape
    output_shape = output_shape if output_shape is not None else context_shape

    ctx_lengths = torch.randint(  # (...)
        min_steps, max_steps + 1, size=batch_shape, generator=rng
    )
    qry_lengths = torch.randint(  # (...)
        min_steps, max_steps + 1, size=batch_shape, generator=rng
    )
    ctx_size = max_steps
    qry_size = max_steps

    # sample values
    ctx_values = torch.randn(*batch_shape, ctx_size, *context_shape, generator=rng)
    tgt_values = torch.randn(*batch_shape, qry_size, *output_shape, generator=rng)

    # sample discrete step indices
    ctx_seq_shape = (*batch_shape, ctx_size, 1)  # padded with one
    qry_seq_shape = (*batch_shape, qry_size, 1)
    ctx_steps = torch.sort(
        torch.randint(0, 2 * max_steps, size=ctx_seq_shape, generator=rng),
        dim=-2,
    ).values
    qry_offsets = torch.sort(
        torch.randint(1, 2 * max_steps + 1, size=qry_seq_shape, generator=rng),
        dim=-2,
    ).values
    last_indices = (ctx_lengths - 1).unsqueeze(-1).unsqueeze(-1)
    last_ctx_steps = ctx_steps.take_along_dim(last_indices, dim=-2)
    qry_steps = last_ctx_steps + qry_offsets

    # create valid mask by broadcasting over sequence length.
    ctx_valid = torch.arange(ctx_size) < ctx_lengths[..., None]  # (..., $N)
    qry_valid = torch.arange(qry_size) < qry_lengths[..., None]  # (..., $K)
    ctx_valid = ctx_valid.unsqueeze(-1)  # (..., $N, 1)
    qry_valid = qry_valid.unsqueeze(-1)  # (..., $K, 1)
    assert ctx_valid.shape == (*batch_shape, ctx_size, 1)
    assert qry_valid.shape == (*batch_shape, qry_size, 1)

    # mask step indices and values.
    ctx_steps = ctx_steps.masked_fill(~ctx_valid, 0)
    qry_steps = qry_steps.masked_fill(~qry_valid, 0)
    ctx_values = ctx_values.masked_fill(~ctx_valid, nan)
    tgt_values = tgt_values.masked_fill(~qry_valid, nan)

    # mask by feature missingness
    # sample one value per time stamp that is always observed
    ctx_safe = torch.randint(
        0, math.prod(context_shape), size=ctx_seq_shape, generator=rng
    )
    qry_safe = torch.randint(
        0, math.prod(output_shape), size=qry_seq_shape, generator=rng
    )
    ctx_mask = ctx_valid & (
        torch.ones_like(ctx_values, dtype=torch.bool)
        if not input_missingness
        else torch.rand_like(ctx_values, generator=rng) > 0.5
    ).scatter(-1, ctx_safe, True)
    qry_mask = qry_valid & (
        torch.ones_like(tgt_values, dtype=torch.bool)
        if not target_missingness
        else torch.rand_like(tgt_values, generator=rng) > 0.5
    ).scatter(-1, qry_safe, True)
    ctx_values = ctx_values.masked_fill(~ctx_mask, nan)
    tgt_values = tgt_values.masked_fill(~qry_mask, nan)

    # normalize to batch_first, equip floats with grads and return
    seq_dim = -2 if batch_first else 0
    return SplitTimeData(
        context_times=ctx_steps.movedim(-2, seq_dim).squeeze(-1),
        context_mask=ctx_mask.movedim(-2, seq_dim),
        context_values=ctx_values.movedim(-2, seq_dim).requires_grad_(),
        query_times=qry_steps.movedim(-2, seq_dim).squeeze(-1),
        query_mask=qry_mask.movedim(-2, seq_dim),
        target_values=tgt_values.movedim(-2, seq_dim).requires_grad_(),
        batch_first=batch_first,
        validate_args=False,
    )


class TestContinuousTimeModel[M: nn.Module](ABC):
    r"""Shared behavioral tests for continuous-time models."""

    SEED: ClassVar[int] = 0
    MIN_STEPS: ClassVar[int] = 2
    MAX_STEPS: ClassVar[int] = 5
    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    GRADIENT_WARMUP_STEPS: ClassVar[int] = 0
    NUM_STEPS: ClassVar[int] = 3
    DIFFERENTIABLE_TIMES: ClassVar[bool] = True

    @abstractmethod
    def make_model(self, model_config: object, /) -> M:
        r"""Instantiate the forecasting model under test."""
        raise NotImplementedError

    @abstractmethod
    def forecast(self, model: M, inputs: SplitTimeData, /) -> tuple[Tensor, ...]:
        r"""Return model predictions for sequential forecasting inputs."""
        raise NotImplementedError

    @abstractmethod
    def loss(
        self, model: M, predictions: tuple[Tensor, ...], targets: Tensor
    ) -> Tensor:
        r"""Return a scalar training loss for model predictions."""
        raise NotImplementedError

    def make_request(
        self,
        *,
        rng: int | torch.Generator,
        batch_shape: int | tuple[int, ...],
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...] | None = None,
        input_missingness: bool = False,
        target_missingness: bool = False,
        batch_first: bool = True,
    ) -> SplitTimeData:
        r"""Sample synthetic forecasting inputs for this model family."""
        return make_continuous_time_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
            target_missingness=target_missingness,
            batch_first=batch_first,
        )

    @pytest.fixture
    def rng(self) -> int:
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
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        data = self.make_request(
            rng=rng,
            batch_shape=(),
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        torch.manual_seed(rng)
        model = self.make_model(model_config)
        self.forecast(model, data)

    def test_forward_batched(
        self,
        model_config: object,
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        data = self.make_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        torch.manual_seed(rng)
        model = self.make_model(model_config)
        self.forecast(model, data)

    def test_forward_batched_matches_forward_unbatched(
        self,
        model_config: object,
        rng: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check batched predictions do not depend on sequence padding."""
        generator = torch.Generator().manual_seed(rng)
        context_lengths = torch.tensor([2, 17])
        query_lengths = torch.tensor([17, 2])
        context_size = int(context_lengths.max().item())
        query_size = int(query_lengths.max().item())

        context_times = torch.full((2, context_size), nan)
        query_times = torch.full((2, query_size), nan)
        context_values = torch.full((2, context_size, *context_shape), nan)
        target_values = torch.full((2, query_size, *output_shape), nan)

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
            target_values[k, :query_length] = torch.randn(
                query_length,
                *output_shape,
                generator=generator,
            )

        if input_missingness:
            flat = context_values.reshape(
                *context_values.shape[: context_values.ndim - len(context_shape)], -1
            )
            C = flat.shape[-1]
            random_observed = torch.rand(flat.shape, generator=generator) >= 0.5
            fallback_idx = torch.randint(
                0, C, flat.shape[:-1], generator=generator
            ).unsqueeze(-1)
            fallback_observed = torch.zeros_like(flat, dtype=torch.bool).scatter_(
                -1, fallback_idx, True
            )
            miss_mask = ~(random_observed | fallback_observed).reshape(
                context_values.shape
            )
            context_values = context_values.masked_fill(miss_mask, nan)

        context_mask = context_values.isfinite()
        query_mask = target_values.isfinite()
        data = SplitTimeData(
            context_times=context_times,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_times,
            query_mask=query_mask,
            target_values=target_values,
        )

        torch.manual_seed(rng)
        model = self.make_model(model_config)
        batched_predictions = self.forecast(model, data)
        expected_predictions = [
            prediction.new_full(prediction.shape, nan)
            for prediction in batched_predictions
        ]

        for k, (context_length_tensor, query_length_tensor) in enumerate(
            zip(context_lengths, query_lengths, strict=True)
        ):
            context_length = int(context_length_tensor.item())
            query_length = int(query_length_tensor.item())
            single_data = SplitTimeData(
                context_times=context_times[k : k + 1, :context_length],
                context_values=context_values[k : k + 1, :context_length],
                context_mask=context_mask[k : k + 1, :context_length],
                query_times=query_times[k : k + 1, :query_length],
                query_mask=query_mask[k : k + 1, :query_length],
                target_values=target_values[k : k + 1, :query_length],
            )

            for prediction, single_prediction in zip(
                expected_predictions,
                self.forecast(model, single_data),
                strict=True,
            ):
                prediction[k : k + 1, :query_length] = single_prediction

        query_valid = data.query_mask.any(dim=-1)
        for prediction, expected in zip(
            batched_predictions,
            expected_predictions,
            strict=True,
        ):
            mask = query_valid
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, nan),
                expected,
                equal_nan=True,
                rtol=1e-6,
                atol=1e-4,
            )

    def test_padding_invariance(
        self,
        model_config: object,
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check predictions are unchanged by extra NaN tail padding."""
        data = self.make_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        padding = 32
        assert data.target_values is not None
        batch_dims = data.context_times.shape[:-1]
        query_size = data.query_times.shape[-1]

        padded_data = SplitTimeData(
            context_times=torch.cat(
                [
                    data.context_times,
                    data.context_times.new_full((*batch_dims, padding), nan),
                ],
                dim=-1,
            ),
            context_values=torch.cat(
                [
                    data.context_values,
                    data.context_values.new_full(
                        (*batch_dims, padding, *context_shape),
                        nan,
                    ),
                ],
                dim=-1 - len(context_shape),
            ),
            context_mask=torch.cat(
                [
                    data.context_mask,
                    data.context_mask.new_zeros((*batch_dims, padding, *context_shape)),
                ],
                dim=-1 - len(context_shape),
            ),
            query_times=torch.cat(
                [
                    data.query_times,
                    data.query_times.new_full((*batch_dims, padding), nan),
                ],
                dim=-1,
            ),
            target_values=torch.cat(
                [
                    data.target_values,
                    data.target_values.new_full(
                        (*batch_dims, padding, *output_shape),
                        nan,
                    ),
                ],
                dim=-1 - len(output_shape),
            ),
            query_mask=torch.cat(
                [
                    data.query_mask,
                    data.query_mask.new_zeros((*batch_dims, padding, *output_shape)),
                ],
                dim=-1 - len(output_shape),
            ),
        )

        torch.manual_seed(rng)
        model = self.make_model(model_config)
        predictions = self.forecast(model, data)
        padded_predictions = self.forecast(model, padded_data)

        query_valid = data.query_mask.any(dim=-1)
        for prediction, padded_prediction in zip(
            predictions,
            padded_predictions,
            strict=True,
        ):
            query_axis = query_valid.ndim - 1
            mask = query_valid
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            padded_window = padded_prediction.narrow(query_axis, 0, query_size)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, nan),
                padded_window.masked_fill(~mask, nan),
                equal_nan=True,
                rtol=0.0,
                atol=1e-4,
            )

    def test_training_unbatched(
        self,
        model_config: object,
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        assert self.GRADIENT_WARMUP_STEPS < self.NUM_STEPS
        torch.manual_seed(rng)
        data = self.make_request(
            rng=rng,
            batch_shape=(),
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        context_values = data.context_values
        assert data.target_values is not None
        model = self.make_model(model_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        predictions = self.forecast(model, data)
        initial_loss = self.loss(model, predictions, data.target_values)

        for step in range(self.NUM_STEPS):
            optimizer.zero_grad()
            predictions = self.forecast(model, data)
            loss = self.loss(model, predictions, data.target_values)
            loss.backward()

            assert context_values.grad is not None
            valid_context = data.context_mask
            assert context_values.grad[valid_context].isfinite().all()
            assert context_values.grad[valid_context].abs().sum() > 0

            for name, parameter in model.named_parameters():
                min_grad = torch.finfo(parameter.dtype).eps
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                if step >= self.GRADIENT_WARMUP_STEPS:
                    assert parameter.grad.abs().max() > min_grad, name

            optimizer.step()

        predictions = self.forecast(model, data)
        final_loss = self.loss(model, predictions, data.target_values)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss

    def test_training_batched(
        self,
        model_config: object,
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        assert self.GRADIENT_WARMUP_STEPS < self.NUM_STEPS
        torch.manual_seed(rng)
        data = self.make_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        if self.DIFFERENTIABLE_TIMES:
            assert data.context_times.requires_grad
        assert data.context_values.requires_grad
        assert data.target_values is not None

        context_values = data.context_values

        model = self.make_model(model_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        predictions = self.forecast(model, data)
        initial_loss = self.loss(model, predictions, data.target_values)

        for step in range(self.NUM_STEPS):
            optimizer.zero_grad()
            predictions = self.forecast(model, data)
            loss = self.loss(model, predictions, data.target_values)
            loss.backward()

            assert context_values.grad is not None
            valid_context = data.context_mask
            assert context_values.grad[valid_context].isfinite().all()
            assert context_values.grad[valid_context].abs().sum() > 0

            for name, parameter in model.named_parameters():
                min_grad = torch.finfo(parameter.dtype).eps
                if not parameter.requires_grad:
                    continue
                assert parameter.grad is not None, name
                assert parameter.grad.isfinite().all(), name
                if step >= self.GRADIENT_WARMUP_STEPS:
                    assert parameter.grad.abs().max() > min_grad, name

            optimizer.step()

        predictions = self.forecast(model, data)
        final_loss = self.loss(model, predictions, data.target_values)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            assert not torch.equal(parameter, initial_parameters[name]), name
        assert final_loss < initial_loss


class TestDiscreteTimeModel[M: nn.Module](TestContinuousTimeModel[M], ABC):
    r"""Shared behavioral tests for discrete-time models."""

    DIFFERENTIABLE_TIMES: ClassVar[bool] = False

    def make_request(
        self,
        *,
        rng: int | torch.Generator,
        batch_shape: int | tuple[int, ...],
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...] | None = None,
        input_missingness: bool = False,
        target_missingness: bool = False,
        batch_first: bool = True,
    ) -> SplitTimeData:
        r"""Sample synthetic integer-step forecasting inputs."""
        return make_discrete_time_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
            target_missingness=target_missingness,
            batch_first=batch_first,
        )

    def test_forward_batched_matches_forward_unbatched(
        self,
        model_config: object,
        rng: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check batched predictions do not depend on sequence padding."""
        generator = torch.Generator().manual_seed(rng)
        context_lengths = torch.tensor([2, 17])
        query_lengths = torch.tensor([17, 2])
        context_size = int(context_lengths.max().item())
        query_size = int(query_lengths.max().item())

        context_steps = torch.zeros(2, context_size, dtype=torch.long)
        query_steps = torch.zeros(2, query_size, dtype=torch.long)
        context_values = torch.full((2, context_size, *context_shape), nan)
        target_values = torch.full((2, query_size, *output_shape), nan)

        for k, (context_length_tensor, query_length_tensor) in enumerate(
            zip(context_lengths, query_lengths, strict=True)
        ):
            context_length = int(context_length_tensor.item())
            query_length = int(query_length_tensor.item())
            context_steps[k, :context_length] = torch.arange(context_length)
            query_steps[k, :query_length] = context_length + torch.arange(query_length)
            context_values[k, :context_length] = torch.randn(
                context_length,
                *context_shape,
                generator=generator,
            )
            target_values[k, :query_length] = torch.randn(
                query_length,
                *output_shape,
                generator=generator,
            )

        if input_missingness:
            flat = context_values.reshape(
                *context_values.shape[: context_values.ndim - len(context_shape)], -1
            )
            C = flat.shape[-1]
            random_observed = torch.rand(flat.shape, generator=generator) >= 0.5
            fallback_idx = torch.randint(
                0, C, flat.shape[:-1], generator=generator
            ).unsqueeze(-1)
            fallback_observed = torch.zeros_like(flat, dtype=torch.bool).scatter_(
                -1, fallback_idx, True
            )
            miss_mask = ~(random_observed | fallback_observed).reshape(
                context_values.shape
            )
            context_values = context_values.masked_fill(miss_mask, nan)

        context_mask = context_values.isfinite()
        query_mask = target_values.isfinite()
        data = SplitTimeData(
            context_times=context_steps,
            context_values=context_values,
            context_mask=context_mask,
            query_times=query_steps,
            query_mask=query_mask,
            target_values=target_values,
            validate_args=False,
        )

        torch.manual_seed(rng)
        model = self.make_model(model_config)
        batched_predictions = self.forecast(model, data)
        expected_predictions = [
            prediction.new_full(prediction.shape, nan)
            for prediction in batched_predictions
        ]

        for k, (context_length_tensor, query_length_tensor) in enumerate(
            zip(context_lengths, query_lengths, strict=True)
        ):
            context_length = int(context_length_tensor.item())
            query_length = int(query_length_tensor.item())
            single_data = SplitTimeData(
                context_times=context_steps[k : k + 1, :context_length],
                context_values=context_values[k : k + 1, :context_length],
                context_mask=context_mask[k : k + 1, :context_length],
                query_times=query_steps[k : k + 1, :query_length],
                query_mask=query_mask[k : k + 1, :query_length],
                target_values=target_values[k : k + 1, :query_length],
                validate_args=False,
            )

            for prediction, single_prediction in zip(
                expected_predictions,
                self.forecast(model, single_data),
                strict=True,
            ):
                prediction[k : k + 1, :query_length] = single_prediction

        query_valid = data.query_mask.any(dim=-1)
        for prediction, expected in zip(
            batched_predictions,
            expected_predictions,
            strict=True,
        ):
            mask = query_valid
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, nan),
                expected,
                equal_nan=True,
                rtol=1e-6,
                atol=1e-4,
            )

    def test_padding_invariance(
        self,
        model_config: object,
        rng: int,
        min_steps: int,
        max_steps: int,
        context_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        batch_shape: tuple[int, ...],
        input_missingness: bool,
    ) -> None:
        r"""Check predictions are unchanged by extra tail padding."""
        data = self.make_request(
            rng=rng,
            batch_shape=batch_shape,
            min_steps=min_steps,
            max_steps=max_steps,
            context_shape=context_shape,
            output_shape=output_shape,
            input_missingness=input_missingness,
        )
        padding = 32
        assert data.target_values is not None
        batch_dims = data.context_times.shape[:-1]
        query_size = data.query_times.shape[-1]

        padded_data = SplitTimeData(
            context_times=torch.cat(
                [
                    data.context_times,
                    data.context_times.new_zeros((*batch_dims, padding)),
                ],
                dim=-1,
            ),
            context_values=torch.cat(
                [
                    data.context_values,
                    data.context_values.new_full(
                        (*batch_dims, padding, *context_shape),
                        nan,
                    ),
                ],
                dim=-1 - len(context_shape),
            ),
            context_mask=torch.cat(
                [
                    data.context_mask,
                    data.context_mask.new_zeros((*batch_dims, padding, *context_shape)),
                ],
                dim=-1 - len(context_shape),
            ),
            query_times=torch.cat(
                [
                    data.query_times,
                    data.query_times.new_zeros((*batch_dims, padding)),
                ],
                dim=-1,
            ),
            target_values=torch.cat(
                [
                    data.target_values,
                    data.target_values.new_full(
                        (*batch_dims, padding, *output_shape),
                        nan,
                    ),
                ],
                dim=-1 - len(output_shape),
            ),
            query_mask=torch.cat(
                [
                    data.query_mask,
                    data.query_mask.new_zeros((*batch_dims, padding, *output_shape)),
                ],
                dim=-1 - len(output_shape),
            ),
            validate_args=False,
        )

        torch.manual_seed(rng)
        model = self.make_model(model_config)
        predictions = self.forecast(model, data)
        padded_predictions = self.forecast(model, padded_data)

        query_valid = data.query_mask.any(dim=-1)
        for prediction, padded_prediction in zip(
            predictions,
            padded_predictions,
            strict=True,
        ):
            query_axis = query_valid.ndim - 1
            mask = query_valid
            while mask.ndim < prediction.ndim:
                mask = mask.unsqueeze(dim=-1)
            padded_window = padded_prediction.narrow(query_axis, 0, query_size)
            mask = mask.expand_as(prediction)
            assert_close(
                prediction.masked_fill(~mask, nan),
                padded_window.masked_fill(~mask, nan),
                equal_nan=True,
                rtol=0.0,
                atol=1e-4,
            )
