r"""Tests for GRU-ODE-Bayes forecasting."""

from typing import ClassVar

import pytest
import torch

from linodenet.forecasting.gru_ode_bayes import (
    GRU_ODE,
    Decoder,
    GRU_Bayes,
    GRU_ODE_Bayes,
    GRUODEBayesConfig,
    ODE_Flow,
    TorchODESolver,
    apply_masked,
    gaussian_kl,
    gaussian_kl_logvar,
)

from .base import SequentialData, TestForecastingModel


class TestGRU_ODE_Bayes(TestForecastingModel[GRU_ODE_Bayes]):
    r"""Tests for direct GRU-ODE-Bayes model construction."""

    CONTEXT_SHAPE: ClassVar[tuple[int, ...]] = (5,)
    OUTPUT_SHAPE: ClassVar[tuple[int, ...]] = CONTEXT_SHAPE
    STANDARD_CONFIG: ClassVar[GRUODEBayesConfig] = GRUODEBayesConfig(
        input_size=CONTEXT_SHAPE[0],
        hidden_size=7,
        decoder_hidden_size=11,
        feature_embedding_size=3,
        bias=True,
        dropout_rate=0.0,
        step_size=0.1,
        solver="euler",
    )

    @pytest.fixture
    def model_config(self) -> GRUODEBayesConfig:
        r"""Configuration used to instantiate the GRU-ODE-Bayes model under test."""
        return self.STANDARD_CONFIG

    def make_model(self, model_config: object, /) -> GRU_ODE_Bayes:
        r"""Instantiate a GRU-ODE-Bayes model from :attr:`STANDARD_CONFIG`."""
        if not isinstance(model_config, GRUODEBayesConfig):
            raise TypeError("model_config must be a GRUODEBayesConfig.")
        config = model_config
        decoder = Decoder(
            config.input_size,
            config.hidden_size,
            config.decoder_hidden_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
        )
        flow = ODE_Flow(
            GRU_ODE(config.hidden_size, bias=config.bias),
            TorchODESolver.new(config.solver, step_size=config.step_size),
        )
        update_cell = GRU_Bayes(
            config.input_size,
            config.hidden_size,
            config.feature_embedding_size,
            bias=config.bias,
        )
        return GRU_ODE_Bayes(
            config.input_size,
            config.hidden_size,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

    def make_cru(self) -> GRU_ODE_Bayes:
        r"""Instantiate a GRU-ODE-Bayes model from :attr:`STANDARD_CONFIG`."""
        return self.make_model(self.STANDARD_CONFIG)

    def forecast(
        self, model: GRU_ODE_Bayes, inputs: SequentialData, /
    ) -> tuple[torch.Tensor, ...]:
        r"""Return GRU-ODE-Bayes predictions for sequential forecasting inputs."""
        pred_mean, pred_logvar = model(
            inputs.query_times,
            inputs.context_times,
            inputs.context_values,
        )

        assert pred_mean.shape == inputs.query_values.shape
        assert pred_logvar.shape == inputs.query_values.shape
        assert pred_mean[inputs.query_mask].isfinite().all()
        assert pred_logvar[inputs.query_mask].isfinite().all()
        return pred_mean, pred_logvar

    def loss(
        self,
        model: GRU_ODE_Bayes,
        predictions: tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return GRU-ODE-Bayes negative log-likelihood for predictions."""
        pred_mean, pred_logvar = predictions
        mask = targets.isfinite()
        return (
            model.nll_logvar(targets, pred_mean, pred_logvar, mask).sum() / mask.sum()
        )

    def test_instantiation(self) -> None:
        config = self.STANDARD_CONFIG
        model = self.make_cru()

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.vector_field, GRU_ODE)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert model.flow.vector_field.lin_hh.in_features == config.hidden_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.input_size == config.input_size
        assert model.update_cell.hidden_size == config.hidden_size
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size
        assert model.initial_state.shape == (config.hidden_size,)

    def test_instantiation_from_config(self) -> None:
        config = self.STANDARD_CONFIG

        model = GRU_ODE_Bayes.from_config(config)

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size

    def test_instantiation_from_parameters(self) -> None:
        config = self.STANDARD_CONFIG

        model = GRU_ODE_Bayes.from_parameters(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            decoder_hidden_size=config.decoder_hidden_size,
            feature_embedding_size=config.feature_embedding_size,
            bias=config.bias,
            dropout_rate=config.dropout_rate,
            step_size=config.step_size,
            solver=config.solver,
        )

        assert isinstance(model, GRU_ODE_Bayes)
        assert model.input_size == config.input_size
        assert model.hidden_size == config.hidden_size
        assert model.decoder.input_size == config.input_size
        assert model.decoder.hidden_size == config.hidden_size
        assert isinstance(model.flow, ODE_Flow)
        assert isinstance(model.flow.solver, TorchODESolver)
        assert model.flow.solver.step_size == config.step_size
        assert isinstance(model.update_cell, GRU_Bayes)
        assert model.update_cell.feature_embedding_size == config.feature_embedding_size


class TestGRUODEBayes:
    r"""Tests for the cleaned GRU-ODE-Bayes implementation."""

    def test_forward_with_padded_partial_observations(self) -> None:
        torch.manual_seed(0)
        decoder = Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7)
        flow = ODE_Flow(
            GRU_ODE(5),
            TorchODESolver.new("euler", step_size=0.1),
        )
        update_cell = GRU_Bayes(
            input_size=3,
            hidden_size=5,
            feature_embedding_size=2,
        )
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=decoder,
            flow=flow,
            update_cell=update_cell,
        )

        context_times = torch.tensor(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.25, torch.nan],
            ]
        )
        context_values = torch.randn(2, 3, 3)
        context_values[0, 1, 2] = torch.nan
        context_values[1, 2] = torch.nan
        query_times = torch.tensor(
            [
                [1.2, 1.5],
                [0.5, torch.nan],
            ]
        )

        pred_mean, pred_logvar = model(query_times, context_times, context_values)
        query_mask = query_times.isfinite()
        context_mask = context_times.isfinite()

        assert pred_mean.shape == (2, 2, 3)
        assert pred_logvar.shape == (2, 2, 3)
        assert pred_mean[query_mask].isfinite().all()
        assert pred_logvar[query_mask].isfinite().all()
        assert pred_mean[~query_mask].isnan().all()
        assert pred_logvar[~query_mask].isnan().all()

        assert model.prior_means.shape == (2, 3, 3)
        assert model.prior_logvars.shape == (2, 3, 3)
        assert model.posterior_means.shape == (2, 3, 3)
        assert model.posterior_logvars.shape == (2, 3, 3)
        assert model.prior_means[context_mask].isfinite().all()
        assert model.posterior_logvars[context_mask].isfinite().all()
        assert model.prior_means[~context_mask].isnan().all()
        assert model.posterior_logvars[~context_mask].isnan().all()

    def test_apply_masked_update_state_with_empty_selection(self) -> None:
        r"""Check all-padding masks do not require forward-loop special cases."""
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            decoder=Decoder(input_size=3, hidden_size=5, decoder_hidden_size=7),
            flow=ODE_Flow(
                GRU_ODE(5),
                TorchODESolver.new("euler", step_size=0.1),
            ),
            update_cell=GRU_Bayes(
                input_size=3,
                hidden_size=5,
                feature_embedding_size=2,
            ),
        )
        state = torch.randn(2, 5)
        observation = torch.full((2, 3), torch.nan)
        mask = torch.zeros(2, dtype=torch.bool)

        result = apply_masked(model.update_state, (state, observation), mask)

        assert result.shape == state.shape
        assert result.isnan().all()

    def test_gaussian_bayes_kl_logvar_matches_variance_formula(self) -> None:
        r"""Test the paper-side KL term against the direct variance formula."""
        mu_pre = torch.tensor([[0.0, 1.0, -1.0], [2.0, 0.0, 1.0]])
        var_pre = torch.tensor([[1.5, 0.4, 2.0], [0.8, 1.2, 0.7]])
        mu_post = torch.tensor([[0.2, 0.7, -0.8], [1.5, -0.1, 0.4]])
        var_post = torch.tensor([[1.1, 0.6, 1.4], [0.9, 1.5, 0.5]])
        mu_obs = torch.tensor([[0.4, torch.nan, -1.2], [2.3, 0.2, 0.8]])
        var_obs = torch.tensor([[0.3, torch.nan, 0.5], [0.2, 0.9, 0.4]])

        valid = mu_obs.isfinite() & var_obs.isfinite()
        mu_obs_clean = mu_obs.nan_to_num(0.0)
        var_obs_clean = var_obs.nan_to_num(1.0)

        var_bayes = var_pre * var_obs_clean / (var_pre + var_obs_clean)
        mu_bayes = (
            var_obs_clean / (var_pre + var_obs_clean) * mu_pre
            + var_pre / (var_pre + var_obs_clean) * mu_obs_clean
        )
        expected = torch.where(
            valid,
            gaussian_kl((mu_bayes, var_bayes), (mu_post, var_post)),
            0.0,
        ).sum(dim=-1)

        actual = GRU_ODE_Bayes.gaussian_bayes_kl_logvar(
            (mu_pre, var_pre.log()),
            (mu_post, var_post.log()),
            (mu_obs, var_obs.log()),
        )

        torch.testing.assert_close(actual, expected)

    def test_gaussian_kl_logvar_matches_variance_formula(self) -> None:
        r"""Test log-variance Gaussian KL against the variance-space helper."""
        mu_1 = torch.tensor([[0.0, 1.0], [2.0, -1.0]])
        var_1 = torch.tensor([[0.5, 2.0], [1.2, 0.7]])
        mu_2 = torch.tensor([[0.4, 0.5], [1.5, -0.5]])
        var_2 = torch.tensor([[0.8, 1.5], [0.9, 2.5]])

        expected = gaussian_kl((mu_1, var_1), (mu_2, var_2))
        actual = gaussian_kl_logvar((mu_1, var_1.log()), (mu_2, var_2.log()))

        torch.testing.assert_close(actual, expected)
