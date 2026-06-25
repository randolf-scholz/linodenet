r"""Tests for Marginalizable Normalizing Flow forecasting."""

from typing import ClassVar, NamedTuple

import pytest
import torch
from torch import Tensor

from linodenet.forecasting.mnf import MarginalizableNormalizingFlow, MixtureWeightsModel

from .base import SequentialData, TestForecastingModel


class MNFConfig(NamedTuple):
    r"""Configuration for a MarginalizableNormalizingFlow."""

    n_feats: int
    n_heads: int
    num_flow_layers: int
    num_bins: int = 8
    bounds: tuple[float, float] = (-5.0, 5.0)


class TestMNF(TestForecastingModel[MarginalizableNormalizingFlow]):
    r"""Tests for the MarginalizableNormalizingFlow density model."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (3,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[MNFConfig] = MNFConfig(
        n_feats=3,
        n_heads=4,
        num_flow_layers=2,
        num_bins=8,
    )

    @pytest.fixture
    def model_config(self) -> MNFConfig:
        r"""Configuration used to instantiate the MNF model under test."""
        return self.STANDARD_CONFIG

    @pytest.fixture(params=[False, True], ids=["no_missingness", "input_missingness"])
    def input_missingness(self, request: pytest.FixtureRequest) -> bool:
        r"""Whether to randomly mask half of the context values with NaN."""
        return request.param

    def make_model(self, model_config: object, /) -> MarginalizableNormalizingFlow:
        r"""Instantiate an MNF from the given config."""
        if not isinstance(model_config, MNFConfig):
            raise TypeError(
                f"model_config must be an MNFConfig, got {type(model_config)}"
            )
        return MarginalizableNormalizingFlow(
            n_feats=model_config.n_feats,
            n_heads=model_config.n_heads,
            num_flow_layers=model_config.num_flow_layers,
            num_bins=model_config.num_bins,
            bounds=model_config.bounds,
        )

    def forecast(
        self, model: MarginalizableNormalizingFlow, inputs: SequentialData, /
    ) -> tuple[Tensor, ...]:
        r"""Return per-query-step log-probabilities under the MNF density.

        MNF is a static density model: it evaluates each query time step
        independently, ignoring temporal context.
        """
        target_values = inputs.target_values  # (*batch_shape, T, D)
        query_mask = inputs.query_valid  # (*batch_shape, T)
        batch_shape = target_values.shape[:-2]
        T = target_values.shape[-2]
        D = target_values.shape[-1]

        flat_values = target_values.reshape(-1, D)  # (B*T, D)
        flat_mask = query_mask.reshape(-1)  # (B*T,)

        log_prob_flat = flat_values.new_full((flat_values.shape[0],), torch.nan)
        if flat_mask.any():
            log_prob_flat[flat_mask] = model.log_prob(flat_values[flat_mask])

        log_prob = log_prob_flat.reshape(*batch_shape, T)  # (*batch_shape, T)
        # Expand to (*batch_shape, T, D) to match the output_shape convention used by
        # the base test class (query_axis = ndim - len(output_shape) - 1).
        return (log_prob.unsqueeze(-1).expand_as(target_values),)

    def loss(
        self,
        model: MarginalizableNormalizingFlow,
        predictions: tuple[Tensor, ...],
        targets: Tensor,
    ) -> Tensor:
        r"""Return mean negative log-likelihood over valid query positions."""
        del model, targets
        (log_prob_expanded,) = (
            predictions  # (*batch_shape, T, D) — all D slices identical
        )
        log_prob = log_prob_expanded[..., 0]  # (*batch_shape, T)
        valid = log_prob.isfinite()
        return -log_prob[valid].sum() / valid.sum()


class TestMixtureWeightsModel:
    def test_mixture_weights_model_returns_one_weight_vector_per_batch_element(
        self,
    ) -> None:
        r"""Each learned mixture query should produce a normalized weight vector."""
        torch.manual_seed(0)
        model = MixtureWeightsModel(
            num_components=3,
            num_heads=2,
            dim_input=5,
            dim_hidden=8,
        )
        embeddings = torch.randn(2, 4, 5)
        valid_mask = torch.tensor(
            [[True, True, False, True], [True, False, False, False]]
        )
        embeddings = embeddings.masked_fill(~valid_mask.unsqueeze(-1), torch.nan)

        weights = model(embeddings, valid_mask=valid_mask)

        assert weights.shape == (2, 3)
        assert weights.isfinite().all()
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(2, dtype=weights.dtype, device=weights.device),
        )

    def test_mixture_weights_model_returns_zero_for_fully_masked_sequences(
        self,
    ) -> None:
        r"""A fully padded sequence should still yield a finite normalized vector."""
        model = MixtureWeightsModel(
            num_components=2,
            num_heads=2,
            dim_input=4,
            dim_hidden=6,
        )
        embeddings = torch.full((1, 3, 4), torch.nan)
        valid_mask = embeddings.isfinite().all(dim=-1)

        weights = model(embeddings, valid_mask=valid_mask)

        assert weights.shape == (1, 2)
        assert weights.isfinite().all()
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(1, dtype=weights.dtype, device=weights.device),
        )
