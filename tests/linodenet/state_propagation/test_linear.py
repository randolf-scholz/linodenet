r"""Tests for linear state propagation flows."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from linodenet.state_propagation.flows.linear import (
    linear_flow,
    linear_gaussian_flow,
)
from tests.testing import SEEDS_5


def _make_linear_system(
    seed: int,
    *,
    with_bias: bool,
    dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    A = torch.tensor(
        rng.normal(size=(dim, dim)) / np.sqrt(dim),
        dtype=torch.float64,
    )
    b = (
        torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
        if with_bias
        else None
    )
    x0 = torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
    timedeltas = torch.tensor(
        np.unique(np.concatenate(([0.0], rng.uniform(0.0, 1.5, size=6), [2.0]))),
        dtype=torch.float64,
    )
    return A, b, x0, timedeltas


def _make_linear_gaussian_system(
    seed: int,
    *,
    with_bias: bool,
    dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    A = torch.tensor(
        rng.normal(size=(dim, dim)) / np.sqrt(dim),
        dtype=torch.float64,
    )
    b = (
        torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)
        if with_bias
        else None
    )
    mu0 = torch.tensor(rng.normal(size=(dim,)), dtype=torch.float64)

    sigma_factor = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    sigma0 = torch.tensor(
        sigma_factor @ sigma_factor.T + 0.1 * np.eye(dim),
        dtype=torch.float64,
    )

    q_factor = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    Q = torch.tensor(
        q_factor @ q_factor.T + 0.1 * np.eye(dim),
        dtype=torch.float64,
    )

    timedeltas = torch.tensor(
        np.unique(np.concatenate(([0.0], rng.uniform(0.0, 1.0, size=5), [1.5]))),
        dtype=torch.float64,
    )
    return A, b, Q, mu0, sigma0, timedeltas


def _solve_linear_ivp(
    A: torch.Tensor,
    b: torch.Tensor | None,
    x0: torch.Tensor,
    timedeltas: torch.Tensor,
) -> torch.Tensor:
    A_np = A.numpy()
    b_np = None if b is None else b.numpy()
    x0_np = x0.numpy()
    t_eval = timedeltas.numpy()

    def func(_: float, x: np.ndarray) -> np.ndarray:
        return A_np @ x if b_np is None else A_np @ x + b_np

    sol = solve_ivp(
        func,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=x0_np,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert sol.success, sol.message
    return torch.tensor(sol.y.T, dtype=torch.float64)


def _solve_linear_gaussian_moment_ivp(
    A: torch.Tensor,
    b: torch.Tensor | None,
    Q: torch.Tensor,
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    timedeltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A_np = A.numpy()
    b_np = None if b is None else b.numpy()
    Q_np = Q.numpy()
    mu0_np = mu0.numpy()
    sigma0_np = sigma0.numpy()
    dim = A.shape[-1]
    t_eval = timedeltas.numpy()

    def func(_: float, state: np.ndarray) -> np.ndarray:
        mu = state[:dim]
        sigma = state[dim:].reshape(dim, dim)
        dmu = A_np @ mu if b_np is None else A_np @ mu + b_np
        dsigma = A_np @ sigma + sigma @ A_np.T + Q_np
        return np.concatenate((dmu, dsigma.reshape(-1)))

    initial_state = np.concatenate((mu0_np, sigma0_np.reshape(-1)))
    sol = solve_ivp(
        func,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=initial_state,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert sol.success, sol.message

    expected_mu = torch.tensor(sol.y[:dim].T, dtype=torch.float64)
    expected_sigma = torch.tensor(
        sol.y[dim:].T.reshape(len(t_eval), dim, dim),
        dtype=torch.float64,
    )
    return expected_mu, expected_sigma


@pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_linear_flow_matches_solve_ivp_fp64(seed: int, with_bias: bool) -> None:
    A, b, x0, timedeltas = _make_linear_system(seed, with_bias=with_bias)

    actual = linear_flow(timedeltas, x0, A, b)
    expected = _solve_linear_ivp(A, b, x0, timedeltas)

    torch.testing.assert_close(actual, expected, atol=5e-10, rtol=5e-9)


@pytest.mark.parametrize("seed", SEEDS_5, ids="seed={}".format)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "with_bias"])
def test_linear_gaussian_flow_matches_solve_ivp_fp64(
    seed: int,
    with_bias: bool,
) -> None:
    A, b, Q, mu0, sigma0, timedeltas = _make_linear_gaussian_system(
        seed, with_bias=with_bias
    )

    actual_mu_tensor, actual_sigma_tensor = linear_gaussian_flow(
        A, b, Q, timedeltas, (mu0, sigma0)
    )

    expected_mu, expected_sigma = _solve_linear_gaussian_moment_ivp(
        A, b, Q, mu0, sigma0, timedeltas
    )

    expected_sigma = 0.5 * (expected_sigma + expected_sigma.transpose(-1, -2))
    torch.testing.assert_close(
        actual_sigma_tensor,
        actual_sigma_tensor.transpose(-1, -2),
        atol=1e-12,
        rtol=1e-12,
    )

    eigenvalues = torch.linalg.eigvalsh(actual_sigma_tensor)
    assert torch.all(eigenvalues > 1e-10), (
        "Expected propagated covariance to be positive definite, "
        f"got min eigenvalue {eigenvalues.min().item():.3e}."
    )

    torch.testing.assert_close(actual_mu_tensor, expected_mu, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(
        actual_sigma_tensor,
        expected_sigma,
        atol=2e-9,
        rtol=2e-8,
    )
