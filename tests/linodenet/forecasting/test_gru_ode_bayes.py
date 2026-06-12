r"""Tests for GRU-ODE-Bayes forecasting."""

import torch

from linodenet.forecasting.gru_ode_bayes import (
    GRU_ODE_Bayes,
    gaussian_kl,
    gaussian_kl_logvar,
)


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
