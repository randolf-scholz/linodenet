r"""Tests for GRU-ODE-Bayes forecasting."""

import torch

from linodenet.forecasting.gru_ode_bayes import GRU_ODE_Bayes


class TestGRUODEBayes:
    r"""Tests for the cleaned GRU-ODE-Bayes implementation."""

    def test_forward_with_padded_partial_observations(self) -> None:
        torch.manual_seed(0)
        model = GRU_ODE_Bayes(
            input_size=3,
            hidden_size=5,
            p_hidden=7,
            prep_hidden=2,
            step_size=0.1,
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
